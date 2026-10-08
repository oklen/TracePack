"""tracepack.hooks -- Claude Code hook entry points (each reads the hook's JSON on stdin).

    python3 -m tracepack.hooks pre-compact      # PreCompact: remember the /compact instructions
    python3 -m tracepack.hooks session-start    # SessionStart: remember the session; after a compaction,
                                                # add back the exact records the summary is likely to drop

Claude Code runs PreCompact, then SessionStart(source="compact"), then PostCompact, and writes the
compaction summary into the transcript only after all three. So at SessionStart(compact) everything
in the transcript is out of the model's context, and the best hints about what comes next are the
user's `/compact` instructions (seen only by PreCompact) and the latest requests.

Settings (environment): TRACEPACK_INJECT_BUDGET (tokens, default 1500; 0 turns injection off).

A hook must never break a session: every failure is logged to the data folder and the hook exits 0.
"""
from __future__ import annotations

import json
import os
import re
import sys
import time
import traceback

from tracepack import session as S


def _int_env(*names, default: int) -> int:
    for n in names:
        v = os.environ.get(n)
        if v not in (None, ""):
            try:
                return int(float(v))
            except ValueError:
                pass
    return default


def _state_path(kind: str, session_id: str) -> str:
    d = os.path.join(S.data_dir(), kind)
    os.makedirs(d, exist_ok=True)
    return os.path.join(d, "%s.json" % re.sub(r"[^0-9A-Za-z_-]", "_", session_id or "unknown"))


def pre_compact(event: dict) -> str:
    path = event.get("transcript_path") or ""
    rec = {"at": time.time(), "session_id": event.get("session_id"), "transcript_path": path,
           "trigger": event.get("trigger"), "custom_instructions": event.get("custom_instructions") or "",
           "size": os.path.getsize(path) if path and os.path.isfile(path) else 0}
    with open(_state_path("precompact", event.get("session_id") or ""), "w", encoding="utf-8") as fh:
        json.dump(rec, fh)
    return ""


def session_start(event: dict, record: bool = True) -> str:
    """Return the hook's stdout (a JSON string) or "" for no output."""
    path = event.get("transcript_path") or ""
    if record:
        S.record_session(os.environ.get("CLAUDE_PID", ""), event.get("session_id") or "", path, event.get("cwd") or "")
    if event.get("source") != "compact":
        return ""
    budget = _int_env("TRACEPACK_INJECT_BUDGET", "CLAUDE_PLUGIN_OPTION_INJECT_BUDGET", default=S.DEFAULT_INJECT_BUDGET)
    if budget <= 0:
        return ""
    if not path or not os.path.isfile(path):
        path = S.resolve_transcript(event.get("session_id") or "", event.get("cwd") or "")
    pre = {}
    try:
        with open(_state_path("precompact", event.get("session_id") or ""), encoding="utf-8") as fh:
            pre = json.load(fh)
        if time.time() - float(pre.get("at", 0)) > 3600:
            pre = {}                                   # stale: from an earlier compaction
    except (OSError, ValueError):
        pre = {}
    t0 = time.time()
    res = S.post_compact_packet(path, budget=budget, custom_instructions=pre.get("custom_instructions", ""),
                                upto_bytes=int(pre.get("size") or 0),
                                max_mb=_int_env("TRACEPACK_MAX_MB", "CLAUDE_PLUGIN_OPTION_MAX_MB",
                                                default=S.DEFAULT_MAX_MB))
    n = len(res.get("entries") or [])
    if record:
        meta = {"at": time.strftime("%Y-%m-%d %H:%M:%S"), "session_id": event.get("session_id"), "transcript": path,
                "found": bool(res.get("found")), "records": n, "tokens": res.get("tokens"), "budget": budget,
                "chars": len(res.get("text") or ""), "redacted": res.get("redacted", 0),
                "trigger": pre.get("trigger"), "custom_instructions": pre.get("custom_instructions", ""),
                "elapsed_ms": int((time.time() - t0) * 1000)}
        try:
            with open(os.path.join(S.data_dir(), "last_inject.json"), "w", encoding="utf-8") as fh:
                json.dump(meta, fh, indent=1)
            with open(os.path.join(S.data_dir(), "last_inject.txt"), "w", encoding="utf-8") as fh:
                fh.write(res.get("text") or "")
        except OSError:
            pass
    if not res.get("found"):
        return ""
    return json.dumps({
        "systemMessage": "TracePack restored %d exact record%s (%s tokens) from before the compaction · "
                         "/tracepack:status" % (n, "" if n == 1 else "s", "{:,}".format(res.get("tokens") or 0)),
        "hookSpecificOutput": {"hookEventName": "SessionStart", "additionalContext": res["text"]},
    }, ensure_ascii=False)


HANDLERS = {"session-start": session_start, "pre-compact": pre_compact}


def main(argv=None) -> int:
    argv = sys.argv[1:] if argv is None else argv
    name = argv[0] if argv else ""
    try:
        raw = sys.stdin.buffer.read().decode("utf-8", "replace")
        event = json.loads(raw) if raw.strip() else {}
        fn = HANDLERS.get(name)
        out = fn(event) if fn else ""
        if out:
            sys.stdout.write(out)
            sys.stdout.flush()
    except Exception:
        try:
            with open(os.path.join(S.data_dir(), "hook_errors.log"), "a", encoding="utf-8") as fh:
                fh.write("%s %s\n%s\n" % (time.strftime("%Y-%m-%d %H:%M:%S"), name, traceback.format_exc()))
        except OSError:
            pass
    return 0


if __name__ == "__main__":
    sys.exit(main())
