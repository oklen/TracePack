"""Put back the file views a compaction dropped: the selection rule of the SWE-bench "keep" arm (K).

When a coding agent's context is compacted, the summary replaces the file views it had read.  The
files are still on disk, so nothing is lost for good, but the agent comes back to read them again:
within 10 steps of a compaction it re-reads 46% of the files it had read before, against 9% without
one (docs/RESEARCH.md).  It comes back to look, not to edit (only 4% of those re-reads edit the file
within 3 steps), and telling it "you read this, unchanged since" does not stop it (`PREREG_keep_probe.md`
§1, §8: re-reads of the put-back files -56% with the real view, +4% with that note).  This module
decides which of the agent's own forgotten views to put back after the summary, verbatim, so it does
not have to re-read them.  :mod:`tracepack.condenser.oh_keep` is the OpenHands binding.

The rule (`select`) is a pure function of the persisted event prefix, identical to the one the
measured runs used (research ``pilot/swe/swe_keep.py``, md5 d2ddabd4...):

1. candidates = `file_editor` views whose events some compaction has forgotten; the latest per file;
2. drop one if the file was edited after that read (by the editor, or by a terminal command that
   writes files), if a later read of the same file is still in the context, if it is a directory
   listing, if its output is over ``MAX_READ_CHARS``, or if the file's name is not in the summary;
3. keep the newest first, at most ``MAX_FILES`` files and ``MAX_TOTAL_CHARS`` characters;
4. put back each kept view's whole batch: every action the agent issued in the same model response
   (same ``llm_response_id``) and every observation of those actions.

Measured on SWE-bench Verified (OpenHands SDK, executor gpt-5.6-terra, 32k compaction line;
`PREREG_keep.md` §8): 242 checkpoints at real compaction points, 4 runs per checkpoint and arm, against
the stock summarizer from the same checkpoint.  Hitting the 64-step limit 19.2% -> 8.4%
(-10.8 [-18.6, -3.0]), re-compactions per run 2.86 -> 1.63, steps -18.1% post hoc (-10.5% on the 115
checkpoints not used to design the rule), solved 46.7% -> 50.0% (+3.3 [-1.4, +10.3], tied), tokens
+12.8%.  The pre-registered cost-per-solve gate did not pass.

Stdlib only; runs on python 3.9.  ``PROVENANCE`` cites the reading behind every constant, and the
selfcheck refuses a constant without one (the same rule as :mod:`tracepack.condenser.recipe`).
"""
from __future__ import annotations

import collections
import os
import re

from ..core.schema import SchemaError

__all__ = ["select", "latest_condensation", "forgotten_ids", "MAX_FILES", "MAX_READ_CHARS",
           "MAX_TOTAL_CHARS", "PROVENANCE"]

#: At most this many files are put back per compaction.
MAX_FILES = 4
#: A single view over this many characters is never put back.
MAX_READ_CHARS = 16000
#: All the put-back views together stay under this many characters (about 8k tokens).
MAX_TOTAL_CHARS = 32000
#: Length of the agent's own note recorded with each kept view (logged only; never sent to the model).
NOTE_CHARS = 150
#: Where SWE-bench images mount the repository.  Only shortens paths (keys and logs); paths outside it
#: stay absolute, which is what OpenHands' file editor uses anyway.
WORKSPACE_ROOT = "/testbed"
#: `file_editor` commands that change a file.
EDIT_COMMANDS = ("str_replace", "insert", "create", "undo_edit")

#: constant -> the reading that fixes its value.  Cited by file and section so it can be checked.
PROVENANCE = {
    "MAX_FILES": "PREREG_keep_probe.md §2 -- frozen before any run, sized against the offline re-read "
                 "coverage in §1 (keep_offline.py over 684 runs): keeping the latest forgotten view of "
                 "every file covers 67% of the in-window re-reads at ~11.6k tokens per compaction; only "
                 "the files the summary names, 44% at ~4.5k.  At the probe's checkpoints the rule chose "
                 "1/2/3/4 files 41/39/39/47 times (166 checkpoints), so the cap of 4 binds often.",
    "MAX_READ_CHARS": "PREREG_keep_probe.md §2 -- one view over 16,000 characters would take half of "
                      "MAX_TOTAL_CHARS by itself.  Frozen with the rest of the rule; not measured "
                      "against other values.",
    "MAX_TOTAL_CHARS": "PREREG_keep_probe.md §2 / PREREG_keep.md §2 -- about 8k tokens, a quarter of the "
                       "32,768-token compaction line.  The put-back views are not counted toward that "
                       "line (oh_keep inserts them into the view, not the state), because counting them "
                       "made the probe compact earlier (0.19 -> 0.55 re-compactions in 12 steps, §8).  "
                       "Measured as frozen: PREREG_keep.md §8 (tokens +12.8%, dollars per run 0.994).",
    "NOTE_CHARS": "Logging only.  The probe's NT arm sent this note INSTEAD of the view and it did not "
                  "reduce re-reads (+4%, PREREG_keep_probe.md §8); K sends the real view.",
    "WORKSPACE_ROOT": "Every measured run was on SWE-bench images, which mount the repository at /testbed.",
    "EDIT_COMMANDS": "The OpenHands file_editor commands that write a file; a view taken before one of "
                     "them is stale (rule 2).",
}

BIG = 10 ** 9


# --------------------------------------------------------------------------- event helpers
# The persisted OpenHands event shape: dicts with "kind" (ActionEvent, ObservationEvent, Condensation,
# ...), "id", and for actions "tool_name", "action", "llm_response_id", "tool_call_id", "thought"; for
# observations "action_id", "tool_call_id", "observation"; for compactions "forgotten_event_ids" and
# "summary".


def _act(o: dict) -> dict:
    return o.get("action") if isinstance(o.get("action"), dict) else {}


def _obs_text(ob) -> str:
    """Every string under the observation record, joined (the length the caps measure)."""
    if ob is None:
        return ""
    out = []

    def walk(v):
        if isinstance(v, str):
            out.append(v)
        elif isinstance(v, dict):
            for x in v.values():
                walk(x)
        elif isinstance(v, list):
            for x in v:
                walk(x)
    walk(ob.get("observation"))
    return "\n".join(out)


def _thought(o: dict) -> str:
    t = o.get("thought")
    if isinstance(t, list):
        return " ".join((x.get("text") or "") for x in t if isinstance(x, dict)).strip()
    return t.strip() if isinstance(t, str) else ""


def _note(o: dict) -> str:
    """The agent's own words on an action: its thought, or what it wrote with the think tool."""
    t = _thought(o)
    if not t and o.get("tool_name") == "think":
        t = str(_act(o).get("thought") or "").strip()
    return t


def _one(s: str, n: int) -> str:
    s = " ".join((s or "").split())
    return s if len(s) <= n else s[:n - 1] + "…"


def _norm_path(p) -> str:
    """Relative to WORKSPACE_ROOT when under it, else as given."""
    p = str(p or "").strip().strip("'\"")
    p = re.sub(r"^%s(/|$)" % re.escape(WORKSPACE_ROOT), "", p)
    return re.sub(r"^\./", "", p)


def _is_dir_view(ob) -> bool:
    return "Here's the files and directories" in _obs_text(ob)[:400]


def forgotten_ids(objs: list, upto: int) -> set:
    """Every event id that the compactions up to and including ``objs[upto]`` have forgotten."""
    fs = set()
    for o in objs[:upto + 1]:
        if o.get("kind") == "Condensation":
            fs.update(str(x) for x in (o.get("forgotten_event_ids") or ()))
    return fs


def latest_condensation(objs: list):
    """Index of the last Condensation in ``objs``, or None."""
    return max((i for i, o in enumerate(objs) if o.get("kind") == "Condensation"), default=None)


def _view(o: dict):
    """(path, first line, last line) of a file_editor view of a file, else None."""
    if o.get("kind") != "ActionEvent" or o.get("tool_name") != "file_editor":
        return None
    a = _act(o)
    if a.get("command") != "view":
        return None
    p = _norm_path(a.get("path"))
    if not p or "." not in os.path.basename(p):
        return None
    vr = a.get("view_range")
    if isinstance(vr, list) and len(vr) == 2:
        try:
            lo, hi = int(vr[0]), int(vr[1])
        except (TypeError, ValueError):
            return None
        return p, lo, (BIG if hi == -1 else hi)
    return p, 1, BIG


def _edited_path(o: dict):
    if o.get("kind") != "ActionEvent":
        return None
    a = _act(o)
    if o.get("tool_name") == "file_editor" and a.get("command") in EDIT_COMMANDS:
        return _norm_path(a.get("path"))
    return None


_TERM_WRITE = re.compile(r"sed\s+-i|>\s*[\w./-]+\.\w+|>>|\btee\b|\bpatch\b|git\s+(checkout|apply|stash|restore)")


def _term_edits(o: dict) -> bool:
    """A terminal command that may write files (sed -i, redirection, patch, git checkout ...)."""
    if o.get("kind") != "ActionEvent" or o.get("tool_name") != "terminal":
        return False
    return bool(_TERM_WRITE.search(str(_act(o).get("command") or "")))


# --------------------------------------------------------------------------- the rule


def select(objs: list, upto: int) -> dict:
    """Which forgotten views to put back at the Condensation ``objs[upto]``.

    ``objs`` is the conversation's events in order (the persisted JSON shape); only ``objs[:upto + 1]``
    is read and nothing is modified.  Returns ``items`` (one per kept file, newest first), ``unforget``
    (the event ids to put back, sorted), ``chars`` (the kept views' size), ``batch_siblings`` (actions
    that came back only because they shared a model response with a kept view), ``dropped`` (how many
    latest views each rule turned away) and ``n_candidates_latest``.
    """
    cond = objs[upto]
    if cond.get("kind") != "Condensation":
        raise ValueError("objs[upto] is not a Condensation")
    fs = forgotten_ids(objs, upto)
    obs = {str(o.get("action_id")): o for o in objs[:upto] if o.get("kind") == "ObservationEvent"}
    acts = [(i, o) for i, o in enumerate(objs[:upto]) if o.get("kind") == "ActionEvent"]
    step_of = {str(o.get("id")): n + 1 for n, (_i, o) in enumerate(acts)}
    summary = cond.get("summary") or ""

    latest, edits, term_edit_at = {}, collections.defaultdict(list), []
    for i, o in acts:
        v = _view(o)
        if v:
            latest[v[0]] = (i, o, v)
        e = _edited_path(o)
        if e:
            edits[e].append(i)
        if _term_edits(o):
            term_edit_at.append((i, str(_act(o).get("command") or "")))

    dropped = collections.Counter()
    cands = []
    for p, (i, o, v) in latest.items():
        oid = str(o.get("id"))
        if oid not in fs:
            dropped["latest_in_context"] += 1
            continue
        if any(j > i for j in edits.get(p, ())) or any(j > i and os.path.basename(p) in c for j, c in term_edit_at):
            dropped["edited_since"] += 1
            continue
        ob = obs.get(oid)
        if ob is None:
            dropped["no_observation"] += 1
            continue
        if _is_dir_view(ob):
            dropped["dir_view"] += 1
            continue
        n = len(_obs_text(ob))
        if n > MAX_READ_CHARS:
            dropped["over_read_cap"] += 1
            continue
        if os.path.basename(p) not in summary:
            dropped["not_in_summary"] += 1
            continue
        cands.append((i, o, v, n))
    cands.sort(key=lambda c: -c[0])
    chosen, total = [], 0
    for c in cands:
        if len(chosen) >= MAX_FILES:
            dropped["over_file_cap"] += 1
            continue
        if total + c[3] > MAX_TOTAL_CHARS:
            dropped["over_total_cap"] += 1
            continue
        chosen.append(c)
        total += c[3]

    # The whole batch comes back: the SDK drops a tool call whose model response (same llm_response_id)
    # is partly forgotten, so every action of that response and its observations are put back together.
    by_resp = collections.defaultdict(list)
    for _i, o in acts:
        by_resp[o.get("llm_response_id")].append(o)
    unforget, n_sib = set(), 0
    for _i, o, _v, _n in chosen:
        batch = by_resp.get(o.get("llm_response_id")) or [o]
        tcids = {str(x.get("tool_call_id")) for x in batch}
        aids = {str(x.get("id")) for x in batch}
        n_sib += len(batch) - 1
        for x in objs[:upto]:
            if str(x.get("id")) in aids:
                unforget.add(str(x.get("id")))
            elif x.get("kind") != "ActionEvent" and (str(x.get("action_id")) in aids or
                                                     (x.get("tool_call_id") and str(x.get("tool_call_id")) in tcids)):
                unforget.add(str(x.get("id")))
    unforget &= fs

    # For the log: the agent's own words on its next action after each kept read.
    obs_pos = {str(o.get("action_id")): i for i, o in enumerate(objs[:upto]) if o.get("kind") == "ObservationEvent"}
    items = []
    for i, o, v, n in chosen:
        oid = str(o.get("id"))
        note = ""
        p = obs_pos.get(oid)
        if p is not None:
            nxt = next((x for j, x in acts if j > p), None)
            if nxt is not None:
                note = _one(_note(nxt), NOTE_CHARS)
        items.append({"action_id": oid, "obs_id": str(obs[oid].get("id")), "path": v[0],
                      "range": [v[1], (-1 if v[2] == BIG else v[2])], "step": step_of[oid],
                      "chars": n, "note": note})
    return {"items": items, "unforget": sorted(unforget), "chars": total, "batch_siblings": n_sib,
            "dropped": dict(dropped), "n_candidates_latest": len(latest)}


def _selfcheck() -> None:
    """Every constant cites its reading (a missing citation is a SchemaError, as in recipe.py)."""
    missing = [k for k in ("MAX_FILES", "MAX_READ_CHARS", "MAX_TOTAL_CHARS", "NOTE_CHARS", "WORKSPACE_ROOT",
                           "EDIT_COMMANDS") if k not in PROVENANCE]
    if missing:
        raise SchemaError("keep: constants without a PROVENANCE entry: %s" % ", ".join(missing))


_selfcheck()
