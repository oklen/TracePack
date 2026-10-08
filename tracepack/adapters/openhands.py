"""tracepack.adapters.openhands -- third adapter (WP4): OpenHands function-calling trajectories -> the IR.

Source: `nvidia/SWE-Hero-openhands-trajectories` (SWE-rebench subset; `WP4_sources.md` §4.2).  A
trajectory is an OpenAI-style message list `[{role, content, tool_calls}]`; `eval/wp4/export_swehero.py`
writes each selected trajectory as one `.jsonl` file so the rest of the pipeline can treat it like a
transcript path:

    {"tracepack_format": "openhands", "instance_id": ..., "repo": ..., "dataset": ..., "trajectory_id": ...}
    {"id": "m0000", "role": "system",    "content": "..."}
    {"id": "m0001", "role": "user",      "content": "<uploaded_files>..."}
    {"id": "m0002", "role": "assistant", "content": "...", "tool_calls": [{"id": "chatcmpl-tool-...", "type": "function", "function": {"name": "str_replace_editor", "arguments": "{...json...}"}}]}
    {"id": "m0003", "role": "tool",      "content": "OBSERVATION..."}

Mapping
-------
| native                                     | IR                                                             |
|--------------------------------------------|----------------------------------------------------------------|
| `system`                                   | dropped (boilerplate the harness prepends; counted)             |
| `user` (the task, plus any follow-ups)     | `kind="user"`                                                   |
| `assistant.content` (non-empty)            | `kind="assistant"`                                              |
| `assistant.tool_calls[j]`                  | `kind="tool_call"`, `tool_call_id = tool_calls[j].id`           |
| `tool`                                     | `kind="tool_result"`, paired to the OLDEST still-open call      |

**Pairing is by position, and the module says so.**  These `tool` messages carry no
`tool_call_id` (measured: 0/1,769 rows in the SWE-rebench shard have one), and every assistant
turn issues exactly one call (0/99 turns with more in the sample), so "the oldest still-open call"
is deterministic and is the only pairing the format licenses.  The predicate on the edge is
`"position"` rather than `"tool_call_id"`, and `expected_strong_edges` re-derives the same rule
in a second pass -- which makes contract #8's declared-edge check weaker here than for Claude Code
or pi (it cannot catch an order-vs-id disagreement because the format has no id to disagree with);
the atomic-group and native-id halves of the audit still apply in full.  A `tool` message with no
open call gets its event and no edge (`unpaired_tool_results`).

Edit tools: `str_replace_editor` with `command` in `create` / `str_replace` / `insert` targets
`path` (SUPERSEDES by path) and quotes `old_str` / `new_str` / `file_text` (the 40-char literal
rule, ranking and cap are the reference adapter's).  `think` is a no-op tool whose reply is a
constant; it is kept as an ordinary call/result pair (the text is the agent's own reasoning, which
`datasets.py` may use as a question source) and flagged `meta.noop="1"`.

There are no timestamps and no parent ids: `timestamp` is the message index (so `TraceGraph`'s
`(timestamp, event_id)` order is transcript order) and no CONTROL edges are emitted.  `native_ref`
is the message id (`m%04d`), assigned by the exporter from the list position.
"""
from __future__ import annotations

import json
import os
from bisect import bisect_left
from collections import Counter, deque

try:  # normal import path
    from ..core.schema import RepresentationRef, TraceEdge, TraceEvent
    from ..core.graph import TraceGraph
    from ..core.values import extract_values
    from .base import AdapterError, build_graph, roundtrip_check, open_trace
    from .claude_code import _estimate_tokens, literal_overlap
except ImportError:  # pragma: no cover - direct execution
    import sys

    sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))
    from tracepack.core.schema import RepresentationRef, TraceEdge, TraceEvent  # type: ignore[no-redef]
    from tracepack.core.graph import TraceGraph  # type: ignore[no-redef]
    from tracepack.core.values import extract_values  # type: ignore[no-redef]
    from tracepack.adapters.base import AdapterError, build_graph, roundtrip_check, open_trace  # type: ignore[no-redef]
    from tracepack.adapters.claude_code import _estimate_tokens, literal_overlap  # type: ignore[no-redef]

__all__ = ["OpenHandsAdapter", "EDIT_TOOLS", "QUOTE_KEYS", "messages_to_rows"]

HEADER_KEY = "tracepack_format"
EDIT_TOOLS = frozenset({"str_replace_editor", "str_replace_based_edit_tool", "edit_file", "file_editor"})   # file_editor: OpenHands SDK >= 1.x (pilot round 2)
WRITE_COMMANDS = frozenset({"create", "str_replace", "insert"})
QUOTE_KEYS = ("old_str", "new_str", "file_text", "content", "old_string", "new_string")
PATH_KEY = "path"
NOOP_TOOLS = frozenset({"think"})
AGENT_REPLY_TOOLS = frozenset({"finish"})   # the observation is the agent's own final message, not a tool record


def messages_to_rows(messages, header: "dict | None" = None):
    """Parquet `trajectory` list -> the row form this adapter reads (ids from position)."""
    rows = []
    if header is not None:
        h = dict(header)
        h[HEADER_KEY] = "openhands"
        rows.append(h)
    for i, m in enumerate(messages):
        if not isinstance(m, dict):
            continue
        row = {"id": "m%04d" % i, "role": m.get("role"), "content": m.get("content")}
        if m.get("tool_calls"):
            row["tool_calls"] = m["tool_calls"]
        if m.get("tool_call_id"):
            row["tool_call_id"] = m["tool_call_id"]
        rows.append(row)
    return rows


def _args_of(tc) -> "tuple[str, dict]":
    fn = tc.get("function") if isinstance(tc.get("function"), dict) else {}
    name = fn.get("name") or tc.get("name") or ""
    raw = fn.get("arguments") if fn else tc.get("arguments")
    if isinstance(raw, str):
        try:
            args = json.loads(raw)
        except ValueError:
            args = {"_raw": raw}
    elif isinstance(raw, dict):
        args = raw
    else:
        args = {}
    return str(name), args if isinstance(args, dict) else {"_raw": str(args)}


def _collect_quotes(value, out, *, depth: int = 0) -> None:
    if depth > 6:
        return
    if isinstance(value, dict):
        for k, v in value.items():
            if k in QUOTE_KEYS and isinstance(v, str):
                out.append(v)
            else:
                _collect_quotes(v, out, depth=depth + 1)
    elif isinstance(value, list):
        for v in value:
            _collect_quotes(v, out, depth=depth + 1)


def _text(content) -> str:
    if isinstance(content, str):
        return content
    if isinstance(content, list):
        parts = []
        for p in content:
            if isinstance(p, dict) and p.get("type") in ("text", None):
                parts.append(p.get("text") or "")
            elif isinstance(p, str):
                parts.append(p)
        return "\n".join(parts)
    return "" if content is None else str(content)


class OpenHandsAdapter:
    """`TraceAdapter` for exported OpenHands trajectories.  Stateless; `normalize` is pure."""

    def __init__(self, *, source_id: str = "oh", link_edits: bool = True, link_writes: bool = True,
                 link_values: bool = False, value_max_sources: int = 3, max_value_edges: int = 4,
                 value_lookback: int = 0, value_require_result_first: bool = True,
                 quote_min_chars: int = 40, quote_lookback: int = 120, quote_scan_cap: int = 120_000,
                 quote_len_cap: int = 20_000, max_quote_edges: int = 2, emit_temporal: bool = True) -> None:
        if not isinstance(source_id, str) or not source_id or ":" in source_id:
            raise AdapterError("source_id must be a non-empty string without ':' (got %r)" % (source_id,))
        if quote_min_chars < 2:
            raise AdapterError("quote_min_chars must be >= 2 (DESIGN_FROZEN §2 froze it at 40)")
        for name, val in (("quote_lookback", quote_lookback), ("quote_scan_cap", quote_scan_cap),
                          ("quote_len_cap", quote_len_cap), ("max_quote_edges", max_quote_edges),
                          ("value_max_sources", value_max_sources), ("max_value_edges", max_value_edges),
                          ("value_lookback", value_lookback)):
            if not isinstance(val, int) or val < 0:
                raise AdapterError("%s must be a non-negative int (got %r)" % (name, val))
        self.source_id = source_id
        self.link_edits = bool(link_edits)
        self.link_writes = bool(link_writes)
        self.link_values = bool(link_values)
        self.value_max_sources = value_max_sources
        self.max_value_edges = max_value_edges
        self.value_lookback = value_lookback
        self.value_require_result_first = bool(value_require_result_first)
        self.quote_min_chars = quote_min_chars
        self.quote_lookback = quote_lookback
        self.quote_scan_cap = quote_scan_cap
        self.quote_len_cap = quote_len_cap
        self.max_quote_edges = max_quote_edges
        self.emit_temporal = bool(emit_temporal)

    def __repr__(self) -> str:  # pragma: no cover
        return "OpenHandsAdapter(source_id=%r, link_edits=%s)" % (self.source_id, self.link_edits)

    # -------------------------------------------------------------- row input

    def _rows(self, native_trace):
        if isinstance(native_trace, (str, bytes, os.PathLike)):
            path = os.fspath(native_trace)
            if not os.path.isfile(path):
                raise AdapterError("trajectory not found: %s" % (path,))
            with open_trace(path) as fh:
                for n, line in enumerate(fh, 1):
                    if not line.strip():
                        continue
                    try:
                        row = json.loads(line)
                    except ValueError:
                        continue
                    if isinstance(row, dict):
                        yield n, row
            return
        if isinstance(native_trace, dict):
            raise AdapterError("native_trace is a single dict; pass a path or a sequence of rows")
        try:
            iterator = enumerate(native_trace, 1)
        except TypeError:
            raise AdapterError("native_trace is neither a path nor an iterable of rows: %r" % (type(native_trace).__name__,))
        for n, row in iterator:
            if not isinstance(row, dict):
                raise AdapterError("trajectory row %d is %r, expected a JSON object" % (n, type(row).__name__))
            yield n, row

    def _blocks(self, row):
        """Row -> `[(kind, tool_call_id_or_None, text, meta_extra, raw_args)]`."""
        role = row.get("role")
        out = []
        if role == "user":
            text = _text(row.get("content"))
            if text.strip():
                out.append(("user", None, text, {}, None))
        elif role == "assistant":
            text = _text(row.get("content"))
            if text.strip():
                out.append(("assistant", None, text, {}, None))
            for tc in (row.get("tool_calls") if isinstance(row.get("tool_calls"), list) else []):
                if not isinstance(tc, dict):
                    continue
                name, args = _args_of(tc)
                text = "%s %s" % (name, json.dumps(args, sort_keys=True, ensure_ascii=False))
                extra = {"tool": name}
                if name in NOOP_TOOLS:
                    extra["noop"] = "1"
                if name in AGENT_REPLY_TOOLS:
                    extra["agent_authored"] = "1"
                out.append(("tool_call", tc.get("id"), text, extra, args))
        elif role == "tool":
            text = _text(row.get("content"))
            out.append(("tool_result", row.get("tool_call_id"), text, {}, None))
        return out

    # ---------------------------------------------------------------- public

    def normalize(self, native_trace) -> TraceGraph:
        return self.normalize_with_stats(native_trace)[0]

    def normalize_with_stats(self, native_trace):
        events = []
        edges_result, edges_temporal, edges_depends = [], [], []
        open_calls = deque()        # (tool_call_id, event_id, tool name) not yet answered, FIFO
        tool_results, edit_calls, consumers, write_targets = [], [], [], []
        counts = Counter()
        header = {}
        ts = 0

        for line_no, row in self._rows(native_trace):
            if row.get(HEADER_KEY):
                header = {k: v for k, v in row.items() if k != HEADER_KEY and isinstance(v, (str, int, bool))}
                continue
            role = row.get("role")
            rid = row.get("id") if isinstance(row.get("id"), str) else None
            if role == "system":
                counts["rows_filtered_system"] += 1
                continue
            blocks = self._blocks(row)
            if not blocks:
                counts["rows_no_blocks"] += 1
                continue
            ts += 1
            for bidx, (kind, tool_id, text, extra, raw_args) in enumerate(blocks):
                eid = "%s:%08d:%03d" % (self.source_id, line_no, bidx)
                meta = {"row_type": str(role)}
                meta.update(extra)
                group = None
                if kind == "tool_call":
                    if not tool_id:
                        tool_id = "%s-%d-%d" % (self.source_id, line_no, bidx)   # synthetic but deterministic
                        counts["tool_calls_without_id"] += 1
                    group = "%s:tool:%s" % (self.source_id, tool_id)
                elif kind == "tool_result":
                    if open_calls:
                        tool_id, call_eid, tname = open_calls.popleft()
                        group = "%s:tool:%s" % (self.source_id, tool_id)
                        meta["tool"] = tname
                        if tname in NOOP_TOOLS:
                            meta["noop"] = "1"
                        if tname in AGENT_REPLY_TOOLS:
                            meta["agent_authored"] = "1"
                    else:
                        tool_id = None
                        counts["unpaired_tool_results"] += 1
                cost = _estimate_tokens(text)
                events.append(TraceEvent(
                    event_id=eid, kind=kind, text=text, timestamp=ts, step_id="%08d" % line_no,
                    tool_call_id=tool_id, native_ref=rid, token_cost=cost, atomic_group=group, meta=meta,
                    representations=(RepresentationRef(kind="raw_text", token_cost=cost, text=text),),
                ))
                counts["kind_" + kind] += 1
                if self.link_values and (kind in ("assistant", "tool_call") or (kind == "tool_result" and meta.get("agent_authored") == "1")):
                    consumers.append((len(events) - 1, eid, text))   # agent-authored text consumes values, never sources them
                if kind == "tool_call":
                    open_calls.append((tool_id, eid, extra.get("tool") or ""))
                    if extra.get("tool") in EDIT_TOOLS and isinstance(raw_args, dict):
                        cmd = raw_args.get("command")
                        fp = raw_args.get(PATH_KEY)
                        if (cmd in WRITE_COMMANDS or cmd is None) and isinstance(fp, str) and fp.strip():
                            write_targets.append((len(events) - 1, eid, fp.strip()))
                        if self.link_edits:
                            quotes = []
                            _collect_quotes(raw_args, quotes)
                            quotes = [q[:self.quote_len_cap] for q in quotes if len(q) >= self.quote_min_chars]
                            if quotes:
                                edit_calls.append((len(events) - 1, eid, quotes))
                elif kind == "tool_result":
                    if meta.get("agent_authored") != "1":
                        tool_results.append((len(events) - 1, eid, text))
                    if tool_id is not None:
                        edges_result.append(TraceEdge(src_id=eid, dst_id=call_eid, edge_type="RESULT_OF",
                                                      predicate="position", provenance="native"))
                        counts["result_of_edges"] += 1
        counts["calls_left_open"] = len(open_calls)

        if self.emit_temporal:
            for i in range(1, len(events)):
                edges_temporal.append(TraceEdge(src_id=events[i].event_id, dst_id=events[i - 1].event_id,
                                                edge_type="TEMPORAL", predicate="transcript_order", provenance="native"))

        # ---- DEPENDS_ON: >=40-char literal quote of an earlier tool_result (reference policy) ----
        anchor = max(1, self.quote_min_chars // 2)
        rank_cap = 4 * self.quote_min_chars
        if self.link_edits and edit_calls and tool_results:
            for call_idx, call_eid, quotes in edit_calls:
                budget = self.quote_scan_cap
                scored = []
                for res_idx, res_eid, res_text in reversed(tool_results):
                    if res_idx >= call_idx:
                        continue
                    if call_idx - res_idx > self.quote_lookback:
                        break
                    if budget <= 0:
                        counts["quote_scan_truncated"] += 1
                        break
                    text = res_text[:budget]
                    budget -= len(text)
                    best, best_q = 0, ""
                    for quote in quotes:
                        run = literal_overlap(quote, text, anchor=anchor, min_len=self.quote_min_chars, stop_at=rank_cap)
                        if run > best:
                            best, best_q = run, quote
                        if best >= rank_cap:
                            break
                    if best >= self.quote_min_chars:
                        scored.append((best, res_idx, res_eid, best_q))
                if scored:
                    counts["depends_on_calls"] += 1
                scored.sort(key=lambda t: (-t[0], -t[1]))
                for run, _idx, res_eid, best_q in scored[:self.max_quote_edges]:
                    bucket = self.quote_min_chars
                    for step in (2, 4):
                        if run >= step * self.quote_min_chars:
                            bucket = step * self.quote_min_chars
                    edges_depends.append(TraceEdge(src_id=call_eid, dst_id=res_eid, edge_type="DEPENDS_ON",
                                                   predicate="literal_quote>=%d" % bucket, provenance="inferred",
                        meta={"match": "literal_quote", "run": str(run), "quote": best_q[:120], "n_candidates": str(len(scored))}))
                    counts["depends_on_edges"] += 1
                    counts["depends_on_run>=%d" % bucket] += 1

        # ---- DEPENDS_ON(value): reference policy (phase 2, WP3a) ----
        edges_values = []
        if self.link_values and tool_results and consumers:
            res_values, res_eid_at = {}, {}
            for r_idx, r_eid, r_text in tool_results:
                res_eid_at[r_idx] = r_eid
                for _typ, v in extract_values(r_text):
                    res_values.setdefault(v, []).append(r_idx)
            first_consumer, cons_vals = {}, []
            for c_idx, c_eid, c_text in consumers:
                vals = extract_values(c_text)
                cons_vals.append((c_idx, c_eid, vals))
                for _typ, v in vals:
                    first_consumer.setdefault(v, c_idx)
            for c_idx, c_eid, vals in cons_vals:
                cands = []
                for typ, v in vals:
                    srcs = res_values.get(v)
                    if not srcs or len(srcs) > self.value_max_sources:
                        counts["value_skip_common" if srcs else "value_skip_nosource"] += 1
                        continue
                    if self.value_require_result_first and first_consumer.get(v, c_idx) < srcs[0]:
                        counts["value_skip_agent_wrote_it"] += 1
                        continue
                    j = bisect_left(srcs, c_idx) - 1
                    if j < 0:
                        continue
                    src_idx = srcs[j]
                    if self.value_lookback and c_idx - src_idx > self.value_lookback:
                        counts["value_skip_lookback"] += 1
                        continue
                    cands.append((len(srcs), -src_idx, typ, v, src_idx, j + 1))
                cands.sort()
                seen_dst = set()
                for _n, _neg, typ, v, src_idx, n_before in cands:
                    dst = res_eid_at[src_idx]
                    if dst in seen_dst:
                        continue
                    seen_dst.add(dst)
                    edges_values.append(TraceEdge(src_id=c_eid, dst_id=dst, edge_type="DEPENDS_ON",
                                                  predicate="value:%s" % typ, provenance="inferred",
                        meta={"match": "value", "type": typ, "value": v, "n_sources": str(n_before), "n_sources_total": str(_n)}))
                    counts["value_edges"] += 1
                    counts["value_edges_%s" % typ] += 1
                    if len(seen_dst) >= self.max_value_edges:
                        break
                if seen_dst:
                    counts["value_consumers_linked"] += 1

        # ---- SUPERSEDES: consecutive write-ish calls to the SAME path ----
        edges_supersede = []
        if self.link_writes:
            by_path = {}
            for idx, eid, fp in write_targets:
                prev = by_path.get(fp)
                if prev is not None:
                    edges_supersede.append(TraceEdge(src_id=eid, dst_id=prev, edge_type="SUPERSEDES",
                                                     predicate="path=%s" % fp[-80:], provenance="inferred"))
                    counts["supersedes_edges"] += 1
                by_path[fp] = eid
            counts["write_targets"] = len(write_targets)
            counts["write_paths"] = len(by_path)

        edges = edges_result + edges_depends + edges_values + edges_supersede + edges_temporal
        graph = build_graph(events, edges)
        stats = dict(sorted(counts.items()))
        stats["n_events"] = len(events)
        stats["n_edges"] = len(edges)
        stats["total_tokens"] = sum(e.token_cost for e in events)
        stats.update({"header_" + k: v for k, v in header.items()})
        return graph, stats

    def render(self, packet) -> str:
        context = getattr(packet, "context", None)
        if not isinstance(context, str):
            raise AdapterError("render expects a MemoryPacket with a str context, got %r" % (type(packet).__name__,))
        return context

    # ------------------------------------------- round-trip audit declarations

    def native_ids(self, native_trace):
        out = []
        for _, row in self._rows(native_trace):
            if row.get(HEADER_KEY) or row.get("role") == "system":
                continue
            rid = row.get("id")
            if isinstance(rid, str) and self._blocks(row):
                out.append(rid)
        return tuple(out)

    def expected_strong_edges(self, native_trace):
        """`(tool_msg_id, assistant_msg_id, "RESULT_OF")` by the same FIFO position rule (second pass)."""
        open_calls, pairs = deque(), []
        for _, row in self._rows(native_trace):
            if row.get(HEADER_KEY):
                continue
            role, rid = row.get("role"), row.get("id")
            if role == "assistant":
                for tc in (row.get("tool_calls") if isinstance(row.get("tool_calls"), list) else []):
                    if isinstance(tc, dict):
                        open_calls.append(rid)
            elif role == "tool":
                if open_calls and isinstance(rid, str):
                    src = open_calls.popleft()
                    if src is not None:
                        pairs.append((rid, src, "RESULT_OF"))
        return tuple(pairs)


# ------------------------------------------------------------------ selfcheck


def _expect(exc, fn, *a, **kw) -> None:
    try:
        fn(*a, **kw)
    except exc:
        return
    except Exception as e:  # noqa: BLE001
        raise AssertionError("expected %s, got %s: %s" % (exc.__name__, type(e).__name__, e))
    raise AssertionError("expected %s, nothing raised" % exc.__name__)


def synthetic_trajectory():
    view = "Here's the result of running `cat -n` on /workspace/app/config.py:\n     1\tSAFETY_MARGIN = 0.15  # tuned after run 4471\n     2\tTIMEOUT = 30\n"
    tc = lambda i, name, args: {"id": "chatcmpl-tool-%d" % i, "type": "function", "function": {"name": name, "arguments": json.dumps(args)}}
    return [
        {"role": "system", "content": "You are OpenHands agent."},
        {"role": "user", "content": "<uploaded_files>\n/workspace/app\n</uploaded_files>\nFix the margin."},
        {"role": "assistant", "content": "Let me think first.", "tool_calls": [tc(1, "think", {"thought": "look at config first"})]},
        {"role": "tool", "content": "Your thought has been logged."},
        {"role": "assistant", "content": "", "tool_calls": [tc(2, "str_replace_editor", {"command": "view", "path": "/workspace/app/config.py"})]},
        {"role": "tool", "content": view},
        {"role": "assistant", "content": "Bumping it.", "tool_calls": [tc(3, "str_replace_editor", {
            "command": "str_replace", "path": "/workspace/app/config.py",
            "old_str": "SAFETY_MARGIN = 0.15  # tuned after run 4471", "new_str": "SAFETY_MARGIN = 0.20  # tuned after run 4471"})]},
        {"role": "tool", "content": "The file /workspace/app/config.py has been edited."},
        {"role": "assistant", "content": "", "tool_calls": [tc(4, "str_replace_editor", {"command": "create", "path": "/workspace/app/config.py", "file_text": "SAFETY_MARGIN = 0.20\n"})]},
        {"role": "tool", "content": "File created successfully at: /workspace/app/config.py"},
        {"role": "assistant", "content": "Done."},
    ]


def _selfcheck() -> None:
    rows = messages_to_rows(synthetic_trajectory(), {"instance_id": "demo__app-1", "repo": "demo/app", "dataset": "nebius/SWE-rebench"})
    ad = OpenHandsAdapter()
    graph, stats = ad.normalize_with_stats(rows)
    ref = {e.event_id: e.native_ref for e in graph.events}
    kinds = Counter(e.kind for e in graph.events)
    assert kinds == {"user": 1, "assistant": 3, "tool_call": 4, "tool_result": 4}, kinds   # two assistant turns have empty content
    pairs = {ref[e.src_id]: ref[e.dst_id] for e in graph.edges if e.edge_type == "RESULT_OF"}
    assert pairs == {"m0003": "m0002", "m0005": "m0004", "m0007": "m0006", "m0009": "m0008"}, pairs
    res = {e.tool_call_id: e for e in graph.events if e.kind == "tool_result"}
    calls = {e.tool_call_id: e for e in graph.events if e.kind == "tool_call"}
    for tid in calls:
        assert res[tid].atomic_group == calls[tid].atomic_group == "oh:tool:%s" % tid
    assert res["chatcmpl-tool-1"].meta.get("noop") == "1" and res["chatcmpl-tool-2"].meta.get("tool") == "str_replace_editor"
    dep = [(ref[e.src_id], ref[e.dst_id], e.predicate) for e in graph.edges if e.edge_type == "DEPENDS_ON"]
    assert dep == [("m0006", "m0005", "literal_quote>=40")], dep
    sup = [(ref[e.src_id], ref[e.dst_id]) for e in graph.edges if e.edge_type == "SUPERSEDES"]
    assert sup == [("m0008", "m0006")], sup           # view is not a write; str_replace then create are
    assert stats["rows_filtered_system"] == 1 and stats["header_instance_id"] == "demo__app-1", stats
    assert [e.timestamp for e in graph.events] == sorted(e.timestamp for e in graph.events)
    rt = roundtrip_check(ad, rows)
    assert rt["ok"] and rt["native_ids_source"] == "adapter" and rt["n_declared_strong"] == 4, rt
    assert rt["n_native_ids_expected"] == 10 == rt["n_native_ids_present"], rt

    # a result with no open call: event kept, no edge, counted
    extra = rows + [{"id": "m0011", "role": "tool", "content": "stray"}]
    g2, st2 = ad.normalize_with_stats(extra)
    assert st2["unpaired_tool_results"] == 1 and st2["result_of_edges"] == 4, st2
    assert roundtrip_check(ad, extra)["ok"]

    # fault injection: an adapter that splits the atomic group is caught by the audit
    class SplitGroups(OpenHandsAdapter):
        def normalize(self, native_trace):
            g = super().normalize(native_trace)
            evs = [TraceEvent(event_id=e.event_id, kind=e.kind, text=e.text, timestamp=e.timestamp, step_id=e.step_id,
                              tool_call_id=e.tool_call_id, native_ref=e.native_ref, token_cost=e.token_cost,
                              representations=e.representations,
                              atomic_group=(None if e.kind == "tool_result" else e.atomic_group), meta=e.meta)
                   for e in g.events]
            return build_graph(evs, list(g.edges))

    _expect(AdapterError, roundtrip_check, SplitGroups(), rows)
    # purity
    g3 = ad.normalize(rows)
    assert [(e.event_id, e.text) for e in g3.events] == [(e.event_id, e.text) for e in graph.events]
    _expect(AdapterError, OpenHandsAdapter, source_id="a:b")
    _expect(AdapterError, ad.normalize, "/nonexistent/traj.jsonl")
    print("[openhands] ok  events=%d edges=%d strong=%d groups=%d" % (rt["n_events"], rt["n_edges"], rt["n_strong_edges"], rt["n_atomic_groups"]))
    print("OPENHANDS_SELFCHECK_OK")


if __name__ == "__main__":
    import sys

    if len(sys.argv) > 1:
        for p in sys.argv[1:]:
            g, st = OpenHandsAdapter().normalize_with_stats(p)
            rt = roundtrip_check(OpenHandsAdapter(), p, strict=False)
            print("%s: events=%d edges=%d ok=%s %s" % (os.path.basename(p)[:60], st["n_events"], st["n_edges"], rt["ok"],
                                                       {k: v for k, v in st.items() if k.startswith(("kind_", "result_of", "unpaired", "calls_left", "depends_on_edges", "supersedes", "rows_"))}))
            if not rt["ok"]:
                print("   failures:", rt["failures"])
    else:
        _selfcheck()
