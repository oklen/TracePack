"""tracepack -- command line.

    tracepack recall "exact error from the last pytest run"     # the current project's newest session
    tracepack recall "port the user gave" --session <id|path> --budget 1500
    tracepack expand 417                                         # one record in full
    tracepack status                                             # what TracePack sees for this session
    tracepack last                                               # what was added after the last compaction
    tracepack sessions                                           # this project's recent sessions
    tracepack demo                                               # try it on a built-in example session
    tracepack doctor                                             # check the install end to end
    tracepack mcp                                                # run the MCP server on stdio
"""
from __future__ import annotations

import argparse
import json
import os
import sys
import tempfile
import time

from tracepack import __version__
from tracepack import session as S


def _mb(n: int) -> str:
    return "%.1f MB" % (n / (1024.0 * 1024.0))


def _path(a) -> str:
    return S.resolve_transcript(getattr(a, "session", ""), getattr(a, "cwd", ""))


def cmd_recall(a) -> int:
    path = _path(a)
    res = S.recall(path, a.query, budget=a.budget, include_recent=a.all, max_mb=a.max_mb)
    if a.json:
        print(json.dumps(res, ensure_ascii=False, indent=1))
    else:
        print(res["text"])
        print("\n-- %s tokens of %s · searched %s records · session %s" % (
            res.get("tokens"), res.get("budget"), res.get("searched", 0), os.path.basename(path)[:-6]), file=sys.stderr)
    return 0


def cmd_expand(a) -> int:
    res = S.expand(_path(a), a.line, start_line=a.start_line, max_tokens=a.max_tokens, max_mb=a.max_mb)
    print(json.dumps(res, ensure_ascii=False, indent=1) if a.json else res["text"])
    return 0 if res.get("found") else 1


def cmd_status(a) -> int:
    path = _path(a)
    st = S.status(path, max_mb=a.max_mb)
    last = S.last_injection()
    if a.json:
        last.pop("text", None)
        print(json.dumps(dict(st, last_inject=last), indent=1))
        return 0
    print("TracePack %s" % __version__)
    print("  session      %s" % os.path.basename(path)[:-6])
    print("  transcript   %s (%s%s)" % (path, _mb(st["bytes"]),
                                         ", searching the last %s" % _mb(st["bytes_read"]) if st["truncated"] else ""))
    print("  records      %d (%d tool calls/outputs), parsed in %d ms" % (st["events"], st["tool_records"], st["load_ms"]))
    print("  compactions  %d" % st["compactions"])
    print("  recallable   %d records no longer in the model's context" % st["recallable"])
    if last:
        print("  last added   %s · %s records · %s tokens · %s ms (tracepack last)" % (
            last.get("at"), last.get("records"), last.get("tokens"), last.get("elapsed_ms")))
    else:
        print("  last added   nothing yet (TracePack adds records right after a compaction)")
    return 0


def cmd_last(a) -> int:
    last = S.last_injection()
    if not last:
        print("Nothing has been added yet. TracePack adds records right after a compaction (/compact or auto).")
        return 1
    print("%s · session %s · %s records · %s tokens · %s chars · %s ms" % (
        last.get("at"), last.get("session_id"), last.get("records"), last.get("tokens"), last.get("chars"),
        last.get("elapsed_ms")))
    if last.get("custom_instructions"):
        print("/compact instructions: %s" % last["custom_instructions"])
    print()
    print(last.get("text") or "(no records matched)")
    return 0


def cmd_sessions(a) -> int:
    cwd = a.cwd or os.environ.get("CLAUDE_PROJECT_DIR") or os.getcwd()
    paths = S.project_transcripts(cwd)[: a.n]
    if not paths:
        print("no sessions for %s" % cwd)
        return 1
    for p in paths:
        st = os.stat(p)
        print("%s  %8s  %s" % (time.strftime("%Y-%m-%d %H:%M", time.localtime(st.st_mtime)), _mb(st.st_size),
                                os.path.basename(p)[:-6]))
    return 0


def cmd_demo(a) -> int:
    from tracepack.selftest import write_demo_session
    with tempfile.TemporaryDirectory() as d:
        demo = write_demo_session(os.path.join(d, "demo.jsonl"))
        print("A made-up session: a benchmark is profiled, fixed and re-run, the user names a migration that\n"
              "must not be touched, then the conversation is compacted. The summary keeps the plan, not the numbers.\n")
        print("$ tracepack recall \"p95 latency before and after the fix\"\n")
        print(S.recall(demo, "p95 latency before and after the fix", budget=700)["text"])
    return 0


def cmd_doctor(a) -> int:
    ok = True
    print("TracePack %s" % __version__)
    py = sys.version_info
    good = py >= (3, 9)
    ok &= good
    print("  [%s] python %d.%d.%d (needs 3.9+)" % ("ok" if good else "!!", py[0], py[1], py[2]))
    pdir = S.projects_dir()
    has = os.path.isdir(pdir)
    print("  [%s] transcripts folder %s" % ("ok" if has else "--", pdir))
    from tracepack.selftest import DEMO_FACTS, write_demo_session
    from tracepack import hooks, mcp_server
    with tempfile.TemporaryDirectory() as d:
        demo = write_demo_session(os.path.join(d, "demo.jsonl"))
        t0 = time.time()
        res = S.recall(demo, "p95 latency before and after the fix", budget=600)
        good = bool(res.get("found")) and DEMO_FACTS["before_p95"] in res["text"] and DEMO_FACTS["after_p95"] in res["text"]
        ok &= good
        print("  [%s] recall finds exact values in a demo session (%d ms)" % ("ok" if good else "!!", (time.time() - t0) * 1000))
        pre = os.path.join(d, "pre.jsonl")
        with open(demo, encoding="utf-8") as src, open(pre, "w", encoding="utf-8") as dst:
            dst.writelines(l for l in src if "compact_boundary" not in l and "isCompactSummary" not in l)
        out = hooks.session_start({"source": "compact", "transcript_path": pre, "session_id": "doctor"}, record=False)
        good = bool(out) and DEMO_FACTS["after_p95"] in out
        ok &= good
        print("  [%s] after-compaction hook adds the latest output (%d chars)" % ("ok" if good else "!!", len(out)))
        r = mcp_server.handle({"jsonrpc": "2.0", "id": 1, "method": "tools/list"})
        good = bool(r) and [t["name"] for t in r["result"]["tools"]] == ["recall", "expand", "status"]
        ok &= good
        print("  [%s] MCP server lists recall / expand / status" % ("ok" if good else "!!"))
    try:
        with open(os.path.join(S.data_dir(), "hook_errors.log"), encoding="utf-8") as fh:
            errs = fh.read().strip().count("Traceback")
    except OSError:
        errs = 0
    print("  [%s] hook errors logged: %d (%s)" % ("ok" if not errs else "!!", errs, os.path.join(S.data_dir(), "hook_errors.log")))
    if has:
        cwd = a.cwd or os.environ.get("CLAUDE_PROJECT_DIR") or os.getcwd()
        n = len(S.project_transcripts(cwd))
        print("  [%s] %d session(s) for this project (%s)" % ("ok" if n else "--", n, cwd))
    print("all good" if ok else "something is wrong; see the lines marked !!")
    return 0 if ok else 1


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(prog="tracepack", description="Exact recall for coding-agent sessions after compaction.")
    p.add_argument("--version", action="version", version="tracepack " + __version__)
    sub = p.add_subparsers(dest="cmd")

    def common(sp, js=True):
        sp.add_argument("--session", default="", help="session id or transcript path (default: newest for --cwd)")
        sp.add_argument("--cwd", default="", help="project folder (default: $CLAUDE_PROJECT_DIR or the current folder)")
        sp.add_argument("--max-mb", type=float, default=S.DEFAULT_MAX_MB, help="read at most this much of a huge transcript")
        if js:
            sp.add_argument("--json", action="store_true", help="machine-readable output")

    r = sub.add_parser("recall", help="exact records from earlier in a session")
    r.add_argument("query")
    r.add_argument("--budget", type=int, default=S.DEFAULT_RECALL_BUDGET, help="token cap (default %(default)s)")
    r.add_argument("--all", action="store_true", help="also search what is still in the model's context")
    common(r)
    r.set_defaults(fn=cmd_recall)

    e = sub.add_parser("expand", help="one record in full, by its transcript line")
    e.add_argument("line")
    e.add_argument("--start-line", type=int, default=1)
    e.add_argument("--max-tokens", type=int, default=2000)
    common(e)
    e.set_defaults(fn=cmd_expand)

    s = sub.add_parser("status", help="what TracePack sees for a session")
    common(s)
    s.set_defaults(fn=cmd_status)

    sub.add_parser("last", help="what was added after the last compaction").set_defaults(fn=cmd_last)

    ss = sub.add_parser("sessions", help="this project's recent sessions")
    ss.add_argument("--cwd", default="")
    ss.add_argument("-n", type=int, default=10)
    ss.set_defaults(fn=cmd_sessions)

    sub.add_parser("demo", help="try recall on a built-in example session").set_defaults(fn=cmd_demo)

    d = sub.add_parser("doctor", help="check the install end to end")
    d.add_argument("--cwd", default="")
    d.set_defaults(fn=cmd_doctor)

    m = sub.add_parser("mcp", help="run the MCP server on stdio")
    m.set_defaults(fn=lambda a: (__import__("tracepack.mcp_server", fromlist=["main"]).main(), 0)[1])

    h = sub.add_parser("hook", help="Claude Code hook entry point (reads JSON on stdin)")
    h.add_argument("event", choices=["session-start", "pre-compact"])
    h.set_defaults(fn=lambda a: __import__("tracepack.hooks", fromlist=["main"]).main([a.event]))
    return p


def main(argv=None) -> int:
    p = build_parser()
    a = p.parse_args(argv)
    if not getattr(a, "fn", None):
        p.print_help()
        return 0
    try:
        return int(a.fn(a) or 0)
    except LookupError as e:
        print("tracepack: %s" % e, file=sys.stderr)
        return 2


if __name__ == "__main__":
    sys.exit(main())
