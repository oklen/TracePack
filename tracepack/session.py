"""tracepack.session -- exact recall for Claude Code sessions.

This is the path the plugin, the MCP server, the hooks and the CLI all use:

    from tracepack import session
    res = session.recall("~/.claude/projects/<project>/<id>.jsonl", "exact error from the last pytest run")
    print(res["text"])

What `recall` does:

1. reads the transcript, or only its last `max_mb` megabytes when it is very large;
2. parses it with the Claude Code adapter: tool calls paired with their results by id, sub-agent rows
   dropped, and compaction rows recognised;
3. by default searches only what is **no longer in the model's context**: everything before the last
   compaction, minus what that compaction preserved verbatim. It never spends the budget re-serving
   text the model can already see;
4. seeds with the hybrid router (BM25 plus offline hashing; no model, no network), follows native
   dependency edges, and packs **verbatim** records under a hard token budget. A tool call is never
   separated from its output, and an oversized record is excerpted with a label rather than cut silently;
5. labels every record with where it came from (tool, time, transcript line) and masks
   credential-shaped strings (`tracepack.redact`).

Right after a compaction (`post_compact_packet`) the summary is not in the transcript yet -- Claude
Code writes it after the SessionStart hook has run -- so everything before the compaction counts as
out of context, and the query is built from the user's `/compact` instructions and the latest work.

Stdlib only. Deterministic for a given transcript, query and budget.
"""
from __future__ import annotations

import glob
import json
import os
import re
import time

from tracepack import redact as R
from tracepack.adapters.claude_code import ClaudeCodeAdapter
from tracepack.core.assembler import AssemblerConfig, BudgetAssembler, estimate_text_tokens
from tracepack.core.closure import ClosureConfig, TypedClosure
from tracepack.core.graph import TraceGraph
from tracepack.core.router import RouterConfig, make_router
from tracepack.core.schema import Seed

DEFAULT_RECALL_BUDGET = 2000
DEFAULT_INJECT_BUDGET = 1500
DEFAULT_MAX_MB = 64
DEFAULT_K = 8
MIN_BUDGET, MAX_BUDGET = 200, 12000
INJECT_MAX_CHARS = 8000          # Claude Code moves additionalContext over 10,000 chars into a file
RECENT_PAIRS = 3                 # after a compaction, the latest tool outputs are pinned in
SEP = "\n\x1f\n"                 # entry separator inside the packer; never shown to the model
HEADER_ALLOWANCE = 22            # tokens reserved per entry for the provenance line

__all__ = ["recall", "expand", "post_compact_packet", "status", "resolve_transcript", "bound_transcript",
           "load_graph", "projects_dir", "project_key", "data_dir", "record_session", "last_injection",
           "DEFAULT_RECALL_BUDGET", "DEFAULT_INJECT_BUDGET", "DEFAULT_MAX_MB"]


# --------------------------------------------------------------------------- where things live

def projects_dir() -> str:
    """Where Claude Code keeps transcripts: `$CLAUDE_CONFIG_DIR/projects`, else `~/.claude/projects`."""
    base = os.environ.get("CLAUDE_CONFIG_DIR") or os.path.join(os.path.expanduser("~"), ".claude")
    return os.path.join(base, "projects")


def data_dir() -> str:
    """TracePack's own small state (session bindings, the last injection). Never inside the user's repo."""
    d = (os.environ.get("TRACEPACK_DATA") or os.environ.get("CLAUDE_PLUGIN_DATA")
         or os.path.join(os.path.expanduser("~"), ".tracepack"))
    if "${" in d:                                    # an unsubstituted placeholder
        d = os.path.join(os.path.expanduser("~"), ".tracepack")
    os.makedirs(d, exist_ok=True)
    return d


def project_key(cwd: str) -> str:
    """Claude Code names a project's folder after its path with every non-alphanumeric character as `-`."""
    return re.sub(r"[^A-Za-z0-9]", "-", os.path.abspath(os.path.expanduser(cwd)))


def project_transcripts(cwd: str) -> list:
    """This project's transcripts, newest first."""
    paths = glob.glob(os.path.join(projects_dir(), project_key(cwd), "*.jsonl"))
    return sorted(paths, key=lambda p: os.path.getmtime(p), reverse=True)


_UUID = re.compile(r"^[0-9a-f]{8}-[0-9a-f]{4}-[0-9a-f]{4}-[0-9a-f]{4}-[0-9a-f]{12}$", re.I)


def _by_session_id(sid: str, cwd: str = "") -> str:
    if cwd:
        p = os.path.join(projects_dir(), project_key(cwd), sid + ".jsonl")
        if os.path.isfile(p):
            return p
    hits = glob.glob(os.path.join(projects_dir(), "*", sid + ".jsonl"))
    return max(hits, key=os.path.getmtime) if hits else ""


def resolve_transcript(session: str = "", cwd: str = "") -> str:
    """A transcript path from what the caller knows.

    `session` may be a path to a .jsonl file, a session id, or empty / "current" / "latest" (the newest
    transcript of the project at `cwd`).
    """
    s = (session or "").strip()
    cwd = cwd or os.environ.get("CLAUDE_PROJECT_DIR") or os.getcwd()
    if s and s not in ("current", "latest"):
        p = os.path.abspath(os.path.expanduser(s))
        if os.path.isfile(p):
            return p
        if _UUID.match(s):
            hit = _by_session_id(s, cwd)
            if hit:
                return hit
        raise LookupError("no transcript found for session %r" % s)
    paths = project_transcripts(cwd)
    if not paths:
        raise LookupError("no Claude Code transcript for %s (looked in %s)"
                          % (cwd, os.path.join(projects_dir(), project_key(cwd))))
    return paths[0]


def record_session(pid: str, session_id: str, transcript_path: str, cwd: str = "") -> None:
    """Remember which session a Claude Code process is on (the hooks call this on every SessionStart)."""
    if not pid:
        return
    sdir = os.path.join(data_dir(), "sessions")
    os.makedirs(sdir, exist_ok=True)
    with open(os.path.join(sdir, "%s.json" % re.sub(r"[^0-9A-Za-z_-]", "_", str(pid))), "w", encoding="utf-8") as fh:
        json.dump({"pid": pid, "session_id": session_id, "transcript_path": transcript_path, "cwd": cwd,
                   "at": time.time()}, fh)
    cutoff = time.time() - 7 * 86400                 # bindings of long-gone processes
    for p in glob.glob(os.path.join(sdir, "*.json")):
        try:
            if os.path.getmtime(p) < cutoff:
                os.remove(p)
        except OSError:
            pass


def bound_transcript(cwd: str = "") -> str:
    """The transcript of the Claude Code session that is calling us (used by the MCP server).

    1. the binding the SessionStart hook recorded for our parent process (`CLAUDE_PID`): follows /clear
       and /resume, which start a new session inside the same process;
    2. `CLAUDE_CODE_SESSION_ID`, which Claude Code sets for its MCP servers;
    3. the newest transcript of the project.
    """
    cwd = cwd or os.environ.get("CLAUDE_PROJECT_DIR") or os.getcwd()
    pid = os.environ.get("CLAUDE_PID", "")
    if pid:
        try:
            with open(os.path.join(data_dir(), "sessions", "%s.json" % pid), encoding="utf-8") as fh:
                rec = json.load(fh)
            if rec.get("transcript_path") and os.path.isfile(rec["transcript_path"]):
                return rec["transcript_path"]
        except (OSError, ValueError):
            pass
    sid = os.environ.get("CLAUDE_CODE_SESSION_ID", "")
    if sid and _UUID.match(sid):
        hit = _by_session_id(sid, cwd)
        if hit:
            return hit
    return resolve_transcript("", cwd)


# --------------------------------------------------------------------------- loading, cached

_CACHE: dict = {}        # path -> (key, graph, info); the MCP server lives for a whole session


def _read_rows(path: str, max_bytes: int, upto: int = 0):
    size = os.path.getsize(path)
    end = min(size, upto) if upto else size
    start = max(0, end - max_bytes)
    with open(path, "rb") as fh:
        fh.seek(start)
        data = fh.read(end - start)
    if start > 0:
        nl = data.find(b"\n")
        data = data[nl + 1:] if nl >= 0 else b""
    rows = []
    for line in data.splitlines():
        if not line.strip():
            continue
        try:
            row = json.loads(line)
        except ValueError:
            continue                           # a torn last line while the session is writing
        if isinstance(row, dict):
            rows.append(row)
    return rows, start > 0, len(data), size


def load_graph(path: str, max_mb: float = DEFAULT_MAX_MB, upto_bytes: int = 0):
    """`(graph, info)` for a transcript; re-parsed only when the file changed.

    `upto_bytes` reads the transcript as it was at that size (the hook uses the size recorded at PreCompact).
    """
    path = os.path.abspath(os.path.expanduser(path))
    st = os.stat(path)
    max_bytes = int(max(1.0, float(max_mb)) * 1024 * 1024)
    key = (st.st_size, st.st_mtime_ns, max_bytes, int(upto_bytes or 0))
    hit = _CACHE.get(path)
    if hit and hit[0] == key:
        return hit[1], hit[2]
    t0 = time.time()
    rows, truncated, nbytes, size = _read_rows(path, max_bytes, int(upto_bytes or 0))
    graph = ClaudeCodeAdapter(link_values=True).normalize(rows) if rows else None
    n_comp = 0
    if graph is not None:
        n_comp = sum(1 for e in graph.events if _is_summary(e))
    info = {"path": path, "bytes": size, "bytes_read": nbytes, "truncated": truncated,
            "rows": len(rows), "events": len(graph.events) if graph is not None else 0,
            "compactions": n_comp, "load_ms": int((time.time() - t0) * 1000)}
    _CACHE.clear()                             # one live transcript at a time keeps memory flat
    _CACHE[path] = (key, graph, info)
    return graph, info


# --------------------------------------------------------------------------- what the model can see

def _is_summary(ev) -> bool:
    return ev.kind == "summary" and (ev.meta or {}).get("compaction") == "1"


_NOISE_PREFIX = ("<local-command-", "<command-name>", "<command-message>", "<command-args>",
                 "Caveat: The messages below were generated by the user while running local commands")


def _is_noise(ev) -> bool:
    """Bookkeeping rows that are not part of the work: slash-command echoes, meta rows, summaries."""
    if ev.kind == "summary":
        return True
    if (ev.meta or {}).get("is_meta") == "1":
        return True
    t = (ev.text or "").lstrip()
    return ev.kind == "user" and t.startswith(_NOISE_PREFIX)


def in_context_ids(graph) -> set:
    """Events the model still has in context: everything from the last compaction summary on, plus what
    that compaction preserved verbatim (events in its window that its summary does not replace)."""
    ids = list(graph.event_ids)
    last = None
    for i, eid in enumerate(ids):
        if _is_summary(graph.event(eid)):
            last = i
    if last is None:
        return set(ids)
    keep = set(ids[last:])
    prev = None
    for i in range(last - 1, -1, -1):
        if _is_summary(graph.event(ids[i])):
            prev = i
            break
    replaced = {d.dst_id for d in graph.parents(ids[last], "MATERIALIZES")}
    start = 0 if prev is None else prev + 1
    if replaced:                                # an old-style boundary that lists what it kept
        keep.update(eid for eid in ids[start:last] if eid not in replaced)
    return keep


def _restrict(graph, drop: set):
    if not drop:
        return graph
    evs = [e for e in graph.events if e.event_id not in drop]
    if not evs:
        return None
    eds = [d for d in graph.edges if d.src_id not in drop and d.dst_id not in drop]
    return TraceGraph(evs, eds)


# --------------------------------------------------------------------------- packing and rendering

def _pair(seeds, graph):
    """A seed tool call brings its own output, and a seed output its call."""
    have = {s.event_id for s in seeds}
    out = list(seeds)
    for s in seeds:
        ev = graph.event(s.event_id)
        if ev.kind == "tool_call":
            linked = [d.src_id for d in graph.children(s.event_id, "RESULT_OF")]
        elif ev.kind == "tool_result":
            linked = [d.dst_id for d in graph.parents(s.event_id, "RESULT_OF")]
        else:
            linked = []
        for eid in linked:
            if eid not in have:
                have.add(eid)
                out.append(Seed(event_id=eid, score=s.score, source=s.source, rank=s.rank, pinned=s.pinned))
    return out


def _when(ms: int) -> str:
    if not ms:
        return ""
    return time.strftime("%Y-%m-%d %H:%M", time.localtime(ms / 1000.0))


def _line_of(event_id: str) -> str:
    parts = event_id.split(":")
    return str(int(parts[1])) if len(parts) >= 2 and parts[1].isdigit() else ""


def _tool_of(graph, ev) -> str:
    tool = (ev.meta or {}).get("tool", "")
    if not tool and ev.kind == "tool_result":
        for d in graph.parents(ev.event_id, "RESULT_OF"):
            tool = (graph.event(d.dst_id).meta or {}).get("tool", "")
            if tool:
                break
    return tool


def _pretty_call(tool: str, text: str) -> str:
    """`Bash {"command": ...}` reads better as `$ ...`; anything else stays as recorded."""
    if not tool or not text.startswith(tool + " {"):
        return text
    try:
        args = json.loads(text[len(tool) + 1:])
    except ValueError:
        return text
    if tool == "Bash" and isinstance(args.get("command"), str):
        return "$ " + args["command"]
    if tool == "Read" and isinstance(args.get("file_path"), str):
        return "Read " + args["file_path"]
    return text


def _label(graph, ev, changed_later: bool) -> str:
    kind = {"tool_call": "call", "tool_result": "output", "user": "user said", "assistant": "assistant said",
            "summary": "summary"}.get(ev.kind, ev.kind)
    tool = _tool_of(graph, ev)
    bits = [("%s %s" % (tool, kind)) if tool and ev.kind in ("tool_call", "tool_result") else kind]
    w = _when(ev.timestamp)
    if w:
        bits.append(w)
    ln = _line_of(ev.event_id)
    if ln:
        bits.append("line " + ln)
    if changed_later:
        bits.append("changed later")
    return " · ".join(bits)


def _pack(sub, graph, query: str, seeds, budget: int, k: int):
    """-> (entries, blocks, manifest, n_redacted) for seeds already chosen."""
    seeds = _pair(seeds, sub)
    closure = TypedClosure(ClosureConfig(mode="native")).close(query, seeds, sub, query_mode="lookup")
    cfg = AssemblerConfig(repr_policy="source_only", pack="evidence_first", evidence_hops=2,
                          evidence_share=0.5, cost_order=False, unit_cap_share=None, excerpt=True,
                          separator=SEP)
    inner = max(MIN_BUDGET // 2, budget - HEADER_ALLOWANCE * (k + 4))
    packet = BudgetAssembler(cfg).assemble(query, closure, sub, inner, seeds=seeds, query_mode="lookup")
    m = packet.manifest
    parts = packet.context.split(SEP) if packet.context else []
    if len(parts) != len(m.entries):            # defensive: never mis-attribute provenance
        parts = [sub.event(e.event_id).text for e in m.entries]
    do_redact = R.enabled()
    entries, blocks, n_red = [], [], 0
    for i, (e, part) in enumerate(zip(m.entries, parts), 1):
        ev = sub.event(e.event_id)
        changed = bool(graph.children(e.event_id, "SUPERSEDES"))
        tool = _tool_of(graph, ev)
        body = _pretty_call(tool, part) if ev.kind == "tool_call" else part
        if do_redact:
            body, k_red = R.redact(body)
            n_red += k_red
        excerpt = "excerpt" in (e.repr_kind or "")
        blocks.append("[%d] %s%s\n%s" % (i, _label(graph, ev, changed), " · excerpt" if excerpt else "", body))
        entries.append({"n": i, "event_id": e.event_id, "kind": ev.kind, "tool": tool, "time": _when(ev.timestamp),
                        "line": _line_of(e.event_id), "tokens": e.token_cost, "reason": e.reason,
                        "excerpt": excerpt, "changed_later": changed})
    return entries, blocks, m, n_red


def _footer(m, entries, n_red: int) -> str:
    foot = []
    if m.incomplete:
        foot.append("More matching records did not fit the budget.")
    if any(x["excerpt"] for x in entries):
        foot.append("\"excerpt\" records are shortened; the full text is available with expand(line).")
    if any(x["changed_later"] for x in entries):
        foot.append("\"changed later\" marks a value that a later step overwrote.")
    if n_red:
        foot.append("%d credential-like value%s masked as [redacted]." % (n_red, "" if n_red == 1 else "s"))
    return " ".join(foot)


def _result(found: bool, text: str, budget: int, info: dict, **kw) -> dict:
    out = {"ok": True, "found": found, "text": text, "tokens": estimate_text_tokens(text), "budget": budget,
           "entries": [], "incomplete": False, "searched": 0, "info": info}
    out.update(kw)
    return out


def _norm(text: str) -> str:
    return " ".join((text or "").split())


def _searchable(graph, include_recent: bool):
    drop = {e.event_id for e in graph.events if _is_noise(e)}
    if not include_recent:
        live = in_context_ids(graph)
        drop |= live
        seen = {_norm(graph.event(eid).text) for eid in live}
        seen.discard("")
        # the same text still in context (a re-sent message, a preserved copy) is not worth re-serving
        drop |= {e.event_id for e in graph.events
                 if e.event_id not in drop and len(_norm(e.text)) >= 20 and _norm(e.text) in seen}
    return _restrict(graph, drop)


def _trailing_reply_ids(graph) -> set:
    """The assistant's last reply (after the last tool output or user message): Claude Code keeps it
    verbatim through a compaction, so the hook does not need to add it back."""
    out = set()
    for ev in reversed(graph.events):
        if _is_noise(ev):
            continue
        if ev.kind != "assistant":
            break
        out.add(ev.event_id)
    return out


# --------------------------------------------------------------------------- recall

def recall(transcript: str, query: str, budget: int = DEFAULT_RECALL_BUDGET, k: int = DEFAULT_K,
           include_recent: bool = False, max_mb: float = DEFAULT_MAX_MB) -> dict:
    """Verbatim records relevant to `query`, under `budget` tokens, labelled with where they came from."""
    t0 = time.time()
    query = " ".join((query or "").split())
    if not query:
        raise ValueError("query is empty")
    budget = int(min(MAX_BUDGET, max(MIN_BUDGET, int(budget))))
    graph, info = load_graph(transcript, max_mb)
    if graph is None or not graph.events:
        return _result(False, "TracePack: this session has no recorded messages yet.", budget, info)
    if not include_recent and info["compactions"] == 0:
        return _result(False, "TracePack: this session has not been compacted, so all of it is still in your "
                              "context. (include_recent=true searches it anyway.)", budget, info)
    sub = _searchable(graph, include_recent)
    if sub is None:
        return _result(False, "TracePack: no records outside your current context.", budget, info)
    kk = max(1, min(int(k), 24))
    seeds = list(make_router("hybrid", RouterConfig(k=kk)).retrieve(query, sub, kk))
    entries, blocks, m, n_red = _pack(sub, graph, query, seeds, budget, kk)
    if not entries:
        return _result(False, "TracePack: nothing earlier in this session matches that.", budget, info,
                       searched=len(sub.events))
    scope = "this session" if include_recent else "before the last compaction"
    head = "TracePack: %d verbatim record%s from %s." % (len(entries), "" if len(entries) == 1 else "s", scope)
    if info.get("truncated"):
        head += " (Searched the last %d MB of a %d MB transcript.)" % (
            info["bytes_read"] // (1024 * 1024), info["bytes"] // (1024 * 1024))
    foot = _footer(m, entries, n_red)
    text = head + "\n\n" + "\n\n".join(blocks) + ("\n\n" + foot if foot else "")
    return _result(True, text, budget, info, entries=entries, incomplete=bool(m.incomplete),
                   searched=len(sub.events), redacted=n_red, elapsed_ms=int((time.time() - t0) * 1000))


# --------------------------------------------------------------------------- expand one record

_REF = re.compile(r"(\d+)")


def expand(transcript: str, ref: str, start_line: int = 1, max_tokens: int = 2000,
           max_mb: float = DEFAULT_MAX_MB) -> dict:
    """The full text of one record (`ref` = its transcript line, e.g. "417" or "line 417", or an event id),
    paged by lines so a long tool output can be read in parts."""
    graph, info = load_graph(transcript, max_mb)
    if graph is None:
        return _result(False, "TracePack: this session has no recorded messages yet.", max_tokens, info)
    ref = (ref or "").strip()
    ev = None
    if ref in graph.event_ids:
        ev = graph.event(ref)
    else:
        m = _REF.search(ref)
        if m:
            want = int(m.group(1))
            cands = [e for e in graph.events if _line_of(e.event_id) == str(want)]
            # a row with a call and text has several events; prefer the tool output, then the call
            cands.sort(key=lambda e: {"tool_result": 0, "tool_call": 1}.get(e.kind, 2))
            ev = cands[0] if cands else None
    if ev is None:
        return _result(False, "TracePack: no record at %r." % ref, max_tokens, info)
    lines = (ev.text or "").split("\n")
    start = max(1, int(start_line))
    out, used = [], 0
    end = start - 1
    for i in range(start - 1, len(lines)):
        cost = estimate_text_tokens(lines[i]) + 1
        if out and used + cost > max_tokens:
            break
        out.append(lines[i])
        used += cost
        end = i + 1
    body = "\n".join(out)
    n_red = 0
    if R.enabled():
        body, n_red = R.redact(body)
    more = end < len(lines)
    head = "TracePack: %s, lines %d-%d of %d%s." % (_label(graph, ev, bool(graph.children(ev.event_id, "SUPERSEDES"))),
                                                  start, end, len(lines),
                                                  "; call again with start_line=%d for more" % (end + 1) if more else "")
    text = head + "\n\n" + body + ("\n\n%d credential-like value(s) masked." % n_red if n_red else "")
    return _result(True, text, max_tokens, info, more=more, next_line=end + 1 if more else None)


# --------------------------------------------------------------------------- right after a compaction

_SECTION = re.compile(r"(?im)^\s*(?:\d+\.\s*|#+\s*)?(primary request(?: and intent)?|current work|pending tasks|"
                      r"(?:optional )?next step|key technical concepts|files and code sections|problem solving|"
                      r"errors and fixes)\b[^\n]*\n")


def _summary_query(text: str, max_chars: int = 3000) -> str:
    marks = [(m.start(), m.group(1).lower()) for m in _SECTION.finditer(text)]
    if marks:
        wanted = ("primary request", "current work", "pending tasks", "next step", "optional next step")
        chunks = [text[pos:(marks[i + 1][0] if i + 1 < len(marks) else len(text))]
                  for i, (pos, name) in enumerate(marks) if name.startswith(wanted)]
        if chunks:
            text = "\n".join(chunks)
    return " ".join(text.split())[:max_chars]


def _recent_query(graph, n_user: int = 2, n_assistant: int = 3, max_chars: int = 3000) -> str:
    users, assistants = [], []
    for ev in reversed(graph.events):
        if _is_noise(ev):
            continue
        if ev.kind == "user" and len(users) < n_user:
            users.append(ev.text or "")
        elif ev.kind == "assistant" and len(assistants) < n_assistant:
            assistants.append(ev.text or "")
        if len(users) >= n_user and len(assistants) >= n_assistant:
            break
    return " ".join(" ".join(users + assistants).split())[:max_chars]


def _recent_seeds(sub, n_pairs: int):
    """The latest tool outputs (with their calls): what the work in flight was looking at."""
    seeds = []
    for ev in reversed(sub.events):
        if ev.kind == "tool_result":
            seeds.append(Seed(event_id=ev.event_id, score=1.0, source="pin", rank=len(seeds), pinned=True))
            if len(seeds) >= n_pairs:
                break
    return seeds


def post_compact_packet(transcript: str, budget: int = DEFAULT_INJECT_BUDGET, custom_instructions: str = "",
                        upto_bytes: int = 0, max_mb: float = DEFAULT_MAX_MB, max_chars: int = INJECT_MAX_CHARS) -> dict:
    """What the SessionStart(compact) hook adds: the latest tool outputs, plus the records that the user's
    `/compact` instructions and the most recent requests point at, under `budget` tokens and `max_chars`."""
    t0 = time.time()
    graph, info = load_graph(transcript, max_mb, upto_bytes)
    if graph is None or not graph.events:
        return _result(False, "", budget, info)
    events = list(graph.events)
    summary_last = bool(events) and any(_is_summary(e) for e in events[-6:])
    if summary_last:                              # a Claude Code that writes the summary before the hook
        sub = _searchable(graph, include_recent=False)
        q = _summary_query([e for e in events if _is_summary(e)][-1].text or "")
    else:                                         # today: the summary is not written yet; all of it is gone
        sub = _searchable(graph, include_recent=True)
        tail = _trailing_reply_ids(graph)
        if sub is not None and tail:
            sub = _restrict(sub, tail)
        q = _recent_query(graph)
    q = " ".join(((custom_instructions or "") + " " + q).split())
    if sub is None or not q:
        return _result(False, "", budget, info)
    kk = DEFAULT_K
    recent = _recent_seeds(sub, RECENT_PAIRS)
    have = {s.event_id for s in recent}
    hybrid = [s for s in make_router("hybrid", RouterConfig(k=kk)).retrieve(q, sub, kk) if s.event_id not in have]
    seeds = recent + [Seed(event_id=s.event_id, score=s.score, source=s.source, rank=len(recent) + i, pinned=False)
                      for i, s in enumerate(hybrid)]
    b = int(min(MAX_BUDGET, max(MIN_BUDGET, int(budget))))
    for _ in range(4):
        entries, blocks, m, n_red = _pack(sub, graph, q, seeds, b, kk)
        if not entries:
            return _result(False, "", budget, info)
        head = ("TracePack: %d verbatim record%s from this session before the compaction, in the order they "
                "happened. The summary above may have shortened them. Other records can be looked up with the "
                "TracePack recall tool." % (len(entries), "" if len(entries) == 1 else "s"))
        foot = _footer(m, entries, n_red)
        text = head + "\n\n" + "\n\n".join(blocks) + ("\n\n" + foot if foot else "")
        if len(text) <= max_chars or b <= MIN_BUDGET:
            break
        b = max(MIN_BUDGET, int(b * max_chars / float(len(text)) * 0.9))
    if len(text) > max_chars:                     # last resort: cut at a record boundary
        keep = []
        for blk in blocks:
            if len(head) + sum(len(x) + 2 for x in keep) + len(blk) + 2 > max_chars - 200:
                break
            keep.append(blk)
        entries = entries[:len(keep)]
        text = head + "\n\n" + "\n\n".join(keep) + "\n\nMore records did not fit; use the TracePack recall tool."
    return _result(True, text, budget, info, entries=entries, incomplete=bool(m.incomplete), searched=len(sub.events),
                   redacted=n_red, query=q[:300], elapsed_ms=int((time.time() - t0) * 1000))


# --------------------------------------------------------------------------- status

def last_injection() -> dict:
    try:
        with open(os.path.join(data_dir(), "last_inject.json"), encoding="utf-8") as fh:
            rec = json.load(fh)
    except (OSError, ValueError):
        return {}
    try:
        with open(os.path.join(data_dir(), "last_inject.txt"), encoding="utf-8") as fh:
            rec["text"] = fh.read()
    except OSError:
        rec["text"] = ""
    return rec


def status(transcript: str, max_mb: float = DEFAULT_MAX_MB) -> dict:
    graph, info = load_graph(transcript, max_mb)
    if graph is None:
        return dict(info, recallable=0, tool_records=0)
    sub = _searchable(graph, include_recent=False) if info["compactions"] else None
    tools = sum(1 for e in graph.events if e.kind in ("tool_call", "tool_result"))
    return dict(info, recallable=len(sub.events) if sub is not None else 0, tool_records=tools)
