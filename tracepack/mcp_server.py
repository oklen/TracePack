"""tracepack.mcp_server -- a dependency-free MCP server (stdio, JSON-RPC 2.0).

Tools (all read-only):
  recall(query, budget_tokens?, include_recent?, session?)   exact records about something
  expand(line, start_line?, max_tokens?, session?)           the full text of one record, paged
  status(session?)                                           what TracePack sees, and the last injection

Claude Code starts it from the plugin's `.mcp.json`; any MCP client can run it with `tracepack mcp`.
Which transcript it reads: the `session` argument if given, else the session that is calling
(see `tracepack.session.bound_transcript`). Nothing leaves the machine.
"""
from __future__ import annotations

import json
import os
import sys
import traceback

from tracepack import __version__
from tracepack import session as S

SUPPORTED = ("2025-11-25", "2025-06-18", "2025-03-26", "2024-11-05")

INSTRUCTIONS = (
    "TracePack reads this session's transcript on disk. After a compaction, `recall` returns the exact, "
    "verbatim records (command output, error text, file contents read, numbers, paths, ids, what the user "
    "said) that are no longer in context, each labelled with tool, time and transcript line; `expand` "
    "returns one record in full.")

_SESSION = {"type": "string", "description": "Optional session id or transcript path; defaults to this session."}

TOOLS = [
    {
        "name": "recall",
        "title": "Recall exact session records",
        "description": (
            "Exact, verbatim records from earlier in this session that are no longer in context after "
            "compaction: command output, error messages, file contents that were read, numbers, paths, ids, "
            "and what the user said. Each record is labelled with tool, time and transcript line; a tool call "
            "comes with its output; the answer stays under budget_tokens. Useful before re-running a slow or "
            "non-repeatable command or asking the user to repeat something."),
        "inputSchema": {
            "type": "object",
            "properties": {
                "query": {"type": "string", "description": "What is needed, in plain words, e.g. 'exact error "
                                                           "from the last pytest run', 'p95 latency before the "
                                                           "fix', 'the DB port the user gave'."},
                "budget_tokens": {"type": "integer", "minimum": S.MIN_BUDGET, "maximum": S.MAX_BUDGET,
                                  "default": S.DEFAULT_RECALL_BUDGET, "description": "Hard cap on the answer size."},
                "include_recent": {"type": "boolean", "default": False,
                                   "description": "Also search records that are still in context."},
                "session": _SESSION,
            },
            "required": ["query"],
            "additionalProperties": False,
        },
        "annotations": {"readOnlyHint": True, "destructiveHint": False, "idempotentHint": True,
                        "openWorldHint": False, "title": "Recall exact session records"},
    },
    {
        "name": "expand",
        "title": "Show one record in full",
        "description": "The full text of one record from recall (by its transcript line), paged by lines; "
                       "for records that recall marked as excerpts.",
        "inputSchema": {
            "type": "object",
            "properties": {
                "line": {"type": "string", "description": "The record's transcript line, e.g. '417'."},
                "start_line": {"type": "integer", "minimum": 1, "default": 1,
                               "description": "First line of the record's text to return."},
                "max_tokens": {"type": "integer", "minimum": 100, "maximum": S.MAX_BUDGET, "default": 2000},
                "session": _SESSION,
            },
            "required": ["line"],
            "additionalProperties": False,
        },
        "annotations": {"readOnlyHint": True, "destructiveHint": False, "idempotentHint": True,
                        "openWorldHint": False, "title": "Show one record in full"},
    },
    {
        "name": "status",
        "title": "TracePack status",
        "description": "What TracePack sees for this session (transcript, records, compactions) and what it "
                       "added after the last compaction. For when the user asks about TracePack.",
        "inputSchema": {
            "type": "object",
            "properties": {"show_last": {"type": "boolean", "default": False,
                                         "description": "Include the full text added after the last compaction."},
                           "session": _SESSION},
            "additionalProperties": False,
        },
        "annotations": {"readOnlyHint": True, "destructiveHint": False, "idempotentHint": True,
                        "openWorldHint": False, "title": "TracePack status"},
    },
]


def _log(msg: str) -> None:
    sys.stderr.write("[tracepack] %s\n" % msg)
    sys.stderr.flush()


def _project_dir() -> str:
    for name in ("TRACEPACK_PROJECT_DIR", "CLAUDE_PROJECT_DIR"):
        v = os.environ.get(name, "")
        if v and "${" not in v:
            return v
    return os.getcwd()


def _int_env(name: str, default: int) -> int:
    try:
        return int(os.environ.get(name, "") or default)
    except ValueError:
        return default


def _transcript(args: dict) -> str:
    s = str(args.get("session") or "").strip()
    return S.resolve_transcript(s, _project_dir()) if s else S.bound_transcript(_project_dir())


def _text(t: str, err: bool = False) -> dict:
    return {"content": [{"type": "text", "text": t}], "isError": err}


def _status_text(args: dict) -> str:
    path = _transcript(args)
    st = S.status(path, max_mb=_int_env("TRACEPACK_MAX_MB", S.DEFAULT_MAX_MB))
    last = S.last_injection()
    lines = ["TracePack %s" % __version__,
             "session: %s" % os.path.basename(path)[:-6],
             "transcript: %s (%.1f MB%s)" % (path, st["bytes"] / 1048576.0,
                                             ", last %.0f MB searched" % (st["bytes_read"] / 1048576.0) if st["truncated"] else ""),
             "records: %d (%d tool calls/outputs), compactions: %d, recallable now: %d"
             % (st["events"], st["tool_records"], st["compactions"], st["recallable"])]
    if last:
        lines.append("last added after a compaction: %s · %s records · %s tokens · %s ms"
                     % (last.get("at"), last.get("records"), last.get("tokens"), last.get("elapsed_ms")))
        if args.get("show_last") and last.get("text"):
            lines += ["", last["text"]]
    else:
        lines.append("nothing added yet (it happens right after a compaction)")
    return "\n".join(lines)


def call_tool(name: str, args: dict) -> dict:
    try:
        if name == "recall":
            q = args.get("query")
            if not isinstance(q, str) or not q.strip():
                return _text("recall needs a non-empty `query`.", True)
            res = S.recall(_transcript(args), q,
                           budget=int(args.get("budget_tokens") or _int_env("TRACEPACK_RECALL_BUDGET", S.DEFAULT_RECALL_BUDGET)),
                           include_recent=bool(args.get("include_recent")),
                           max_mb=_int_env("TRACEPACK_MAX_MB", S.DEFAULT_MAX_MB))
            return _text(res["text"])
        if name == "expand":
            res = S.expand(_transcript(args), str(args.get("line") or ""), start_line=int(args.get("start_line") or 1),
                           max_tokens=int(args.get("max_tokens") or 2000),
                           max_mb=_int_env("TRACEPACK_MAX_MB", S.DEFAULT_MAX_MB))
            return _text(res["text"], not res.get("found"))
        if name == "status":
            return _text(_status_text(args))
        return _text("unknown tool %r" % name, True)
    except LookupError as e:
        return _text("TracePack: %s" % e, True)
    except Exception as e:                         # never take the session down with us
        _log(traceback.format_exc())
        return _text("TracePack failed: %s: %s" % (type(e).__name__, e), True)


def handle(msg: dict):
    """One JSON-RPC message in, one response dict out (None for notifications)."""
    mid = msg.get("id")
    method = msg.get("method")
    params = msg.get("params") or {}
    if mid is None:                                # notifications: initialized, cancelled, ...
        return None
    if method == "initialize":
        want = params.get("protocolVersion")
        return {"jsonrpc": "2.0", "id": mid, "result": {
            "protocolVersion": want if want in SUPPORTED else SUPPORTED[1],
            "capabilities": {"tools": {"listChanged": False}},
            "serverInfo": {"name": "tracepack", "title": "TracePack", "version": __version__},
            "instructions": INSTRUCTIONS}}
    if method == "ping":
        return {"jsonrpc": "2.0", "id": mid, "result": {}}
    if method == "tools/list":
        return {"jsonrpc": "2.0", "id": mid, "result": {"tools": TOOLS}}
    if method == "tools/call":
        return {"jsonrpc": "2.0", "id": mid, "result": call_tool(params.get("name"), params.get("arguments") or {})}
    if method in ("resources/list", "prompts/list", "resources/templates/list"):
        key = {"resources/list": "resources", "prompts/list": "prompts",
               "resources/templates/list": "resourceTemplates"}[method]
        return {"jsonrpc": "2.0", "id": mid, "result": {key: []}}
    return {"jsonrpc": "2.0", "id": mid, "error": {"code": -32601, "message": "method not found: %s" % method}}


def serve(stdin=None, stdout=None) -> None:
    stdin = stdin or sys.stdin
    stdout = stdout or sys.stdout
    for line in stdin:
        line = line.strip()
        if not line:
            continue
        try:
            msg = json.loads(line)
        except ValueError:
            stdout.write(json.dumps({"jsonrpc": "2.0", "id": None,
                                     "error": {"code": -32700, "message": "parse error"}}) + "\n")
            stdout.flush()
            continue
        batch = msg if isinstance(msg, list) else [msg]
        replies = [r for r in (handle(m) for m in batch if isinstance(m, dict)) if r is not None]
        if replies:
            stdout.write(json.dumps(replies if isinstance(msg, list) else replies[0], ensure_ascii=False) + "\n")
            stdout.flush()


def main() -> None:
    try:
        sys.stdin.reconfigure(encoding="utf-8", errors="replace")
        sys.stdout.reconfigure(encoding="utf-8")
    except AttributeError:
        pass
    _log("TracePack %s MCP server ready" % __version__)
    try:
        serve()
    except KeyboardInterrupt:
        pass


if __name__ == "__main__":
    main()
