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
from tracepack.core.assembler import AssemblerConfig, BudgetAssembler


def estimate_text_tokens(text: str) -> int:
    """Tokens as the rest of TracePack counts them: about 4 characters per token (the adapters' estimate,
    close to Claude's and GPT's tokenizers on English text and code)."""
    return (len(text) + 3) // 4 if text else 0
from tracepack.core.closure import ClosureConfig, TypedClosure
from tracepack.core.graph import TraceGraph
from tracepack.core.router import RouterConfig, make_router
from tracepack.core.schema import Seed

DEFAULT_RECALL_BUDGET = 2000
DEFAULT_INJECT_BUDGET = 1500
DEFAULT_MAX_MB = 64
DEFAULT_K = 8
DEFAULT_ROUTER = "lexical"       # BM25; CodeMemo odd half: beats the BM25+hashing hybrid (34.6% vs 30.8%)
DEFAULT_CLOSURE = "off"          # dependency closure cost evidence on CodeMemo (structured packer only)
DEFAULT_PAIR = True              # a tool call comes with its output (and an output with its call)
DEFAULT_EXCERPT = True           # a record too big for what is left is served as a labelled excerpt
DEFAULT_PACKER = "greedy"        # whole records in rank order until the budget is full ("structured": closure packer)
GREEDY_CANDIDATES = 300          # how deep the ranking goes when filling the budget
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


def _when_short(ms: int) -> str:
    """`10-08 17:02`: the label repeats on every record, so it stays short."""
    if not ms:
        return ""
    return time.strftime("%m-%d %H:%M", time.localtime(ms / 1000.0))


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
    w = _when_short(ev.timestamp)
    if w:
        bits.append(w)
    ln = _line_of(ev.event_id)
    if ln:
        bits.append("L" + ln)
    if changed_later:
        bits.append("changed later")
    return " · ".join(bits)


def _pack(sub, graph, query: str, seeds, budget: int, k: int, closure_mode: str = "native",
          pair: bool = True, excerpt: bool = True):
    """-> (entries, blocks, manifest, n_redacted) for seeds already chosen, labels included in `budget`."""
    if pair:
        seeds = _pair(seeds, sub)
    closure = TypedClosure(ClosureConfig(mode=closure_mode)).close(query, seeds, sub, query_mode="lookup")
    cfg = AssemblerConfig(repr_policy="source_only", pack="evidence_first", evidence_hops=2,
                          evidence_share=0.5, cost_order=False, unit_cap_share=None, excerpt=excerpt,
                          separator=SEP)
    packet = BudgetAssembler(cfg).assemble(query, closure, sub, max(50, int(budget)), seeds=seeds, query_mode="lookup")
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


def _fit(sub, graph, query: str, seeds, budget: int, k: int, closure_mode: str, render,
         pair: bool = True, excerpt: bool = True):
    """Pack, render with `render(entries, blocks, m, n_red) -> text`, and shrink the packer's budget until
    the whole rendered answer (labels, header and footer included) is within `budget` tokens."""
    inner = int(budget * 0.85)
    best = None
    for _ in range(6):
        entries, blocks, m, n_red = _pack(sub, graph, query, seeds, inner, k, closure_mode, pair, excerpt)
        text = render(entries, blocks, m, n_red)
        tok = estimate_text_tokens(text)
        best = (entries, blocks, m, n_red, text)
        if tok <= budget or inner <= 50:
            break
        inner = max(50, int(inner * budget / float(tok) * 0.95))
    entries, blocks, m, n_red, text = best
    while entries and estimate_text_tokens(text) > budget:       # last resort: drop the last record
        entries, blocks = entries[:-1], blocks[:-1]
        text = render(entries, blocks, m, n_red)
    return entries, blocks, m, n_red, text


_STOP = frozenset("""a an and are as at be been but by can did do does done for from had has have how i if in into is
it its it's me my no not of on or our out over so than that the their them then there these they this those to up
us was we were what when where which who why will with would you your about after again all also any because before
between both could each few further here just more most much only other own same should some such too under until
very while""".split())
_TERM = re.compile(r"[A-Za-z0-9_][A-Za-z0-9_./:\-]{2,}")


def _terms(query: str):
    return [w for w in (m.group(0).lower().strip(".:-/") for m in _TERM.finditer(query or "")) if w and w not in _STOP]


def _excerpt(text: str, terms, max_tokens: int) -> str:
    """The lines of `text` that mention the query's words (with one line of context), in order, `…` at gaps.
    A single huge line is cut to a window around the first match."""
    if max_tokens <= 0:
        return ""
    lines = (text or "").split("\n")
    hits = [i for i, l in enumerate(lines) if any(w in l.lower() for w in terms)] or [0]
    keep = sorted({j for i in hits for j in range(max(0, i - 1), min(len(lines), i + 2))})
    out, used, prev = [], 0, None
    for j in keep:
        piece = lines[j]
        if len(piece) > 600:
            low = piece.lower()
            pos = min([low.find(w) for w in terms if low.find(w) >= 0] or [0])
            piece = ("…" if pos > 200 else "") + piece[max(0, pos - 200):pos + 400] + "…"
        cost = estimate_text_tokens(piece) + 1
        if used + cost > max_tokens:
            break
        if prev is not None and j != prev + 1:
            out.append("…")
        out.append(piece)
        used += cost
        prev = j
    return "\n".join(out)


def _call_hint(graph, ev) -> str:
    """For a tool output: the call it answers, in a few words (`$ pytest -q`, `parse.py`)."""
    for d in graph.parents(ev.event_id, "RESULT_OF"):
        call = graph.event(d.dst_id)
        tool = (call.meta or {}).get("tool", "")
        txt = _pretty_call(tool, call.text or "")
        if txt.startswith("$ "):
            return "`%s`" % (txt[2:82] + ("…" if len(txt) > 82 else ""))
        try:
            args = json.loads((call.text or "")[len(tool) + 1:]) if tool else {}
        except ValueError:
            args = {}
        for key in ("file_path", "path", "pattern", "url", "query", "notebook_path"):
            if isinstance(args.get(key), str):
                return "`%s`" % args[key][-80:]
        return ""
    return ""


class _Greedy:
    incomplete = False


def _greedy(sub, graph, query: str, ranked_ids, budget: int, excerpt: bool = True, reserve: int = 70,
            pinned=()):
    """Fill `budget` with whole records in rank order, then excerpt the best-ranked records that did not fit.

    A ranked tool call is served as the call (its content is often the answer: an Edit's new text, a
    command) with a short output attached; a ranked tool output names its call in the label. `pinned`
    records go first. -> (entries, blocks, state, n_redacted), blocks in the order the records happened."""
    terms = _terms(query)
    do_redact = R.enabled()
    chosen, skipped, used, n_red = {}, [], 0, 0
    state = _Greedy()
    cap = budget - reserve

    covered, hinted = set(), set()          # outputs already shown under their call; calls named in a label

    def block_for(ev, attach=True):
        tool = _tool_of(graph, ev)
        changed = bool(graph.children(ev.event_id, "SUPERSEDES"))
        label = _label(graph, ev, changed)
        if ev.kind == "tool_result":
            hint = _call_hint(graph, ev)
            if hint:
                label = label.replace("output", "output of " + hint, 1)
            body = ev.text or ""
        elif ev.kind == "tool_call":
            body = _pretty_call(tool, ev.text or "")
            outs = [d.src_id for d in sub.children(ev.event_id, "RESULT_OF")]
            if outs and attach and outs[0] not in chosen:
                out = sub.event(outs[0]).text or ""
                if estimate_text_tokens(out) <= 40:
                    body += "\n→ " + out.strip()
                    return tool, changed, label, body, outs[0]
        else:
            body = ev.text or ""
        return tool, changed, label, body, None

    order = list(pinned) + [e for e in ranked_ids if e not in set(pinned)]
    for eid in order:
        if used >= cap - 30:
            state.incomplete = True
            break
        if eid in chosen or eid in covered or eid not in sub:
            continue
        ev = sub.event(eid)
        if ev.kind == "tool_call" and eid in hinted:
            if (ev.meta or {}).get("tool", "") in ("Bash", "Read", "Grep", "Glob", "LS"):
                continue                            # the output's label already names this call
        tool, changed, label, body, folded = block_for(ev, attach=eid not in hinted)
        cost = estimate_text_tokens("[99] " + label) + 2 + estimate_text_tokens(body)
        if used + cost > cap:
            state.incomplete = True
            skipped.append(eid)
            continue
        if do_redact:
            body, k = R.redact(body)
            n_red += k
        used += cost
        chosen[eid] = (ev, label, body, tool, changed, False, cost)
        if folded:
            covered.add(folded)
        if ev.kind == "tool_result":
            hinted.update(d.dst_id for d in sub.parents(eid, "RESULT_OF"))
    if excerpt:                                     # second pass: the best-ranked records that were too big
        for eid in skipped[:3]:
            room = cap - used - 4
            if room < 120:
                break
            if eid in covered:
                continue
            ev = sub.event(eid)
            tool, changed, label, body, folded = block_for(ev, attach=False)
            head_cost = estimate_text_tokens("[99] " + label + " · excerpt") + 2
            body = _excerpt(body, terms, room - head_cost)
            if not body:
                continue
            if do_redact:
                body, k = R.redact(body)
                n_red += k
            cost = head_cost + estimate_text_tokens(body)
            used += cost
            chosen[eid] = (ev, label + " · excerpt", body, tool, changed, True, cost)
    pos = {e: i for i, e in enumerate(sub.event_ids)}
    entries, blocks = [], []
    for i, eid in enumerate(sorted(chosen, key=lambda e: pos.get(e, 0)), 1):
        ev, label, body, tool, changed, is_excerpt, cost = chosen[eid]
        blocks.append("[%d] %s\n%s" % (i, label, body))
        entries.append({"n": i, "event_id": eid, "kind": ev.kind, "tool": tool, "time": _when(ev.timestamp),
                        "line": _line_of(eid), "tokens": cost, "reason": "ranked", "excerpt": is_excerpt,
                        "changed_later": changed})
    return entries, blocks, state, n_red


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


_SEARCH: dict = {}       # (graph object, include_recent) -> (searchable sub-graph, router); one graph at a time


def _searcher(graph, include_recent: bool, router_name: str):
    """The searchable sub-graph and a router that keeps its index across queries on the same graph."""
    key = (id(graph), bool(include_recent), router_name)
    hit = _SEARCH.get(key)
    if hit is None or hit[0] is not graph:
        if any(v[0] is not graph for v in _SEARCH.values()):
            _SEARCH.clear()
        sub = _searchable(graph, include_recent)
        hit = (graph, sub, make_router(router_name, RouterConfig(k=DEFAULT_K)))
        _SEARCH[key] = hit
    return hit[1], hit[2]


def auto_k(budget: int) -> int:
    """How many records to rank for a budget: enough to fill it (about one per 100 tokens), 8..64."""
    return max(DEFAULT_K, min(64, int(budget) // 100))


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

def recall(transcript: str, query: str, budget: int = DEFAULT_RECALL_BUDGET, k: int = 0,
           include_recent: bool = False, max_mb: float = DEFAULT_MAX_MB, router: str = "", closure: str = "",
           pair=None, excerpt=None, packer: str = "") -> dict:
    """Verbatim records relevant to `query`, under `budget` tokens, labelled with where they came from.

    `k` (records ranked; 0 = enough to fill the budget), `router` and `closure` are tuning knobs; the
    defaults are the measured ones (README, "Benchmark")."""
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
    kk = int(k) if k else auto_k(budget)
    sub, rt = _searcher(graph, include_recent, router or DEFAULT_ROUTER)
    if sub is None:
        return _result(False, "TracePack: no records outside your current context.", budget, info)
    seeds = list(rt.retrieve(query, sub, kk)) if (packer or DEFAULT_PACKER) != "greedy" else []
    scope = "this session" if include_recent else "before the last compaction"
    trunc = ""
    if info.get("truncated"):
        trunc = " (Searched the last %d MB of a %d MB transcript.)" % (
            info["bytes_read"] // (1024 * 1024), info["bytes"] // (1024 * 1024))

    def render(entries, blocks, m, n_red):
        head = "TracePack: %d verbatim record%s from %s.%s" % (len(entries), "" if len(entries) == 1 else "s",
                                                             scope, trunc)
        foot = _footer(m, entries, n_red)
        return head + "\n\n" + "\n\n".join(blocks) + ("\n\n" + foot if foot else "")

    use_excerpt = DEFAULT_EXCERPT if excerpt is None else bool(excerpt)
    if (packer or DEFAULT_PACKER) == "greedy":
        ranked = [s.event_id for s in rt.retrieve(query, sub, min(len(sub.events), GREEDY_CANDIDATES))]
        entries, blocks, m, n_red = _greedy(sub, graph, query, ranked, budget, use_excerpt)
        text = render(entries, blocks, m, n_red)
        while entries and estimate_text_tokens(text) > budget:      # estimate drift: drop the last record
            entries, blocks = entries[:-1], blocks[:-1]
            text = render(entries, blocks, m, n_red)
    else:
        entries, blocks, m, n_red, text = _fit(sub, graph, query, seeds, budget, kk, closure or DEFAULT_CLOSURE,
                                               render, DEFAULT_PAIR if pair is None else bool(pair), use_excerpt)
    if not entries:
        return _result(False, "TracePack: nothing earlier in this session matches that.", budget, info,
                       searched=len(sub.events))
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
    recent = [s.event_id for s in _recent_seeds(sub, RECENT_PAIRS)]
    ranked = [s.event_id for s in make_router(DEFAULT_ROUTER, RouterConfig(k=DEFAULT_K)).retrieve(
        q, sub, min(len(sub.events), GREEDY_CANDIDATES))]

    def render(entries, blocks, m, n_red):
        head = ("TracePack: %d verbatim record%s from this session before the compaction, in the order they "
                "happened. The summary above may have shortened them. Other records can be looked up with the "
                "TracePack recall tool." % (len(entries), "" if len(entries) == 1 else "s"))
        foot = _footer(m, entries, n_red)
        return head + "\n\n" + "\n\n".join(blocks) + ("\n\n" + foot if foot else "")

    b = int(min(MAX_BUDGET, max(MIN_BUDGET, int(budget))))
    for _ in range(4):
        entries, blocks, m, n_red = _greedy(sub, graph, q, ranked, b, DEFAULT_EXCERPT, pinned=recent)
        if not entries:
            return _result(False, "", budget, info)
        text = render(entries, blocks, m, n_red)
        if (len(text) <= max_chars and estimate_text_tokens(text) <= budget) or b <= MIN_BUDGET:
            break
        b = max(MIN_BUDGET, int(b * min(max_chars / float(len(text)), budget / float(estimate_text_tokens(text))) * 0.9))
    while entries and (len(text) > max_chars or estimate_text_tokens(text) > budget):
        entries, blocks = entries[:-1], blocks[:-1]
        text = render(entries, blocks, m, n_red)
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
