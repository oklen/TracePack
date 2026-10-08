"""Save, at every compaction, what is needed to branch three conditions from that one state.

INVENTORY §4 closed the S/T/R experiment on the existing archives with one sentence: **the mid-run
workspace does not exist**.  The archives keep the forgotten events and the packets, but not the
tree the agent had edited, so "continue from here with and without the evidence" could not be set
up -- the three conditions would have differed in their workspace as well as in the evidence.

Two facts make the fix cheap.

* The `.oh` event log is **append-only, one immutable JSON per event**, so the log at the END of a
  run already contains every prefix.  Nothing has to be copied mid-run: a snapshot only has to
  record *which* Condensation the prefix ends at, and a branch truncates the log there.
* A workspace is fully described, for this protocol, by its diff against `base_commit` -- that is
  already how `swe_pilot.export_patch` states the answer, so the snapshot and the graded patch are
  in the same language.

**Why a conversation callback and not a condenser subclass.**  The first version built a subclass
of whichever condenser the arm asked for.  It never ran: the SDK serialises the agent (condenser
included) through a discriminated union keyed on `kind`, and a class created at runtime is not in
that registry -- persistence died with `PydanticSerializationError: ... _serialize_by_kind:
RecursionError` four seconds into the run, before any compaction.  A callback is also the better
fit for what this is: instrumentation that must not be part of the object under test.

The one thing to be careful about is **not perturbing the run being measured**.  `git add -A`
writes an index, and the agent can and does run `git status`; a snapshot that staged the tree would
show up inside the experiment.  So every git call here runs with `GIT_INDEX_FILE` pointed at a
throwaway file, and the repository's own index is never touched.

Failures are logged and swallowed.  A snapshot is instrumentation; it must never end a two-hour
cell.  A missing snapshot is visible afterwards (the `.json` is simply not there) and a cell with
missing snapshots is dropped, which is the right direction for the error to point.
"""
from __future__ import annotations

import glob
import json
import logging
import os
import subprocess
import tempfile
import time

logger = logging.getLogger(__name__)

GIT = "git -c core.fileMode=false "


def workspace_diff(workspace: str, base_commit: str, timeout: int = 600) -> str:
    """The tree as a patch against base_commit, computed without touching the real index."""
    fd, idx = tempfile.mkstemp(prefix="snapidx_")
    os.close(fd)
    os.unlink(idx)                      # git wants to create it itself
    env = dict(os.environ)
    env["GIT_INDEX_FILE"] = idx
    try:
        subprocess.run(GIT + "add -A", shell=True, cwd=workspace, env=env,
                       capture_output=True, text=True, timeout=timeout)
        p = subprocess.run(GIT + "diff --cached --no-color %s" % base_commit, shell=True,
                           cwd=workspace, env=env, capture_output=True, text=True,
                           errors="replace", timeout=timeout)
        return p.stdout or ""
    finally:
        try:
            os.unlink(idx)
        except OSError:
            pass


def write_snapshot(outdir: str, workspace: str, base_commit: str, summary: str,
                   forgotten, event_id: str = "") -> dict:
    """One snapshot: which compaction, the summary as the agent will see it, and the tree."""
    os.makedirs(outdir, exist_ok=True)
    i = len(glob.glob(os.path.join(outdir, "snap-*.json")))
    t0 = time.time()
    diff = workspace_diff(workspace, base_commit)
    row = {
        "i": i,
        "ts": int(time.time() * 1000),
        # the branch truncates the event log at the (i+1)-th Condensation; the id is recorded so a
        # branch can assert it cut at the snapshot it thinks it did
        "event_id": str(event_id or ""),
        "forgotten": [str(x) for x in (forgotten or ())],
        # the summary AFTER the arm has merged its packet in -- this is the string a branch rewrites
        "summary": summary or "",
        "summary_chars": len(summary or ""),
        "diff_chars": len(diff),
        "diff_files": sum(1 for l in diff.splitlines() if l.startswith("diff --git a/")),
        "ms": int(1000 * (time.time() - t0)),
    }
    with open(os.path.join(outdir, "snap-%03d.diff" % i), "w", encoding="utf-8") as fh:
        fh.write(diff)
    with open(os.path.join(outdir, "snap-%03d.json" % i), "w", encoding="utf-8") as fh:
        json.dump(row, fh, ensure_ascii=False)
    return row


def make_callback(workspace: str, base_commit: str, outdir: str):
    """A Conversation callback that snapshots on every Condensation event.

    Signature is deliberately loose: the SDK has passed callbacks a single event, and taking
    *args and picking the Condensation out of them survives that being wrapped in a state object
    later -- an instrumentation callback that raises would take the cell with it.
    """

    def _cb(*args, **kwargs):
        try:
            ev = None
            for a in list(args) + list(kwargs.values()):
                if type(a).__name__ == "Condensation":
                    ev = a
                    break
            if ev is None:
                return
            write_snapshot(outdir, workspace, base_commit,
                           getattr(ev, "summary", "") or "",
                           getattr(ev, "forgotten_event_ids", None),
                           getattr(ev, "id", ""))
        except Exception as exc:                                    # pragma: no cover - reported
            logger.warning("snapshot failed: %s: %s", type(exc).__name__, exc)

    return _cb

def make_round_callback(workspace: str, base_commit: str, outdir: str):
    """A Conversation callback that snapshots the tree after every executor round.

    §9.1 fixes the observation points at rounds 1 / 2 / 4 / 8 BEFORE any data exists, and §9.1 also
    forbids the measurement from touching the executor: "不向执行器反馈结果，也不在其工作区留下文件、
    缓存或其他状态变化".  Grading in place would do both.  So the branch saves a diff per round and
    the grading happens afterwards, in a fresh cell, off these diffs -- which is exactly the escape
    the plan offers: "若逐轮复制过于昂贵，改为保存各轮状态后离线验收".

    One round = one ActionEvent.  The count is kept here rather than read off the event log so a
    resumed branch numbers its own rounds from 1, not from the prefix's length.
    """
    state = {"n": 0}

    def _cb(*args, **kwargs):
        try:
            ev = None
            for a in list(args) + list(kwargs.values()):
                if type(a).__name__ == "ActionEvent":
                    ev = a
                    break
            if ev is None:
                return
            state["n"] += 1
            n = state["n"]
            os.makedirs(outdir, exist_ok=True)
            diff = workspace_diff(workspace, base_commit)
            with open(os.path.join(outdir, "round-%03d.diff" % n), "w", encoding="utf-8") as fh:
                fh.write(diff)
            with open(os.path.join(outdir, "rounds.jsonl"), "a", encoding="utf-8") as fh:
                fh.write(json.dumps({"round": n, "ts": int(time.time() * 1000),
                                     "diff_chars": len(diff),
                                     "tool": getattr(ev, "tool_name", None)}, ensure_ascii=False) + "\n")
        except Exception as exc:                                    # pragma: no cover - reported
            logger.warning("round snapshot failed: %s: %s", type(exc).__name__, exc)

    return _cb

#: build junk that must never reach a grader, same list swe_pilot.export_patch uses
JUNK = ("__pycache__/", ".pyc", ".egg-info/", ".pytest_cache/", ".mypy_cache/", ".coverage",
        ".hypothesis/")


def clean_patch(patch: str) -> tuple:
    """Drop the sections a grader cannot use, exactly as swe_pilot.export_patch does.

    This is not tidiness, it is correctness, and it cost a whole block to learn.  The first S/T/R
    block graded every round of all three conditions as unresolved -- on an instance whose own
    prefix run had resolved.  The round diffs carried nine **mode-only** sections
    (`old mode 100755 / new mode 100644`, from the official image's `chmod -R 777 /testbed`)
    alongside the one real change; `git apply` refuses the whole patch, so nothing was applied and
    every round graded false.  Filtering the same file down to its one real section graded
    **resolved=True**.

    A binary section does the same thing for the same reason -- "a binary section without --binary
    cannot be applied and takes the whole patch down with it" -- and build junk merely inflates it.

    Returns (patch, counts).  Snapshots stay RAW on disk: the diff is also how a branch restores the
    workspace, and a restore should reproduce the tree, not a tidied version of it.  The filtering
    belongs at the grading boundary.
    """
    import re
    parts = re.split(r"(?m)^(?=diff --git )", patch or "")
    kept, dropped = [], {"binary": 0, "junk": 0, "mode_only": 0}
    for part in parts:
        if not part.strip():
            continue
        head = part.split("\n", 1)[0]
        path = head[len("diff --git a/"):].split(" b/")[0] if head.startswith("diff --git a/") else head
        if "\nBinary files " in part or "GIT binary patch" in part:
            dropped["binary"] += 1
            continue
        if any(j in path for j in JUNK):
            dropped["junk"] += 1
            continue
        if "old mode" in part and "@@" not in part and "new file" not in part and "deleted file" not in part:
            dropped["mode_only"] += 1
            continue
        kept.append(part)
    return "".join(kept), {"sections": len(kept), "dropped": dropped}
