"""tracepack.adapters.pi -- second adapter (WP4): pi coding-agent session (`pi session format`) -> the IR.

The pi harness (https://pi.dev, badlogic/pi-mono) writes one newline-delimited JSON entry per event.
`WP4_sources.md` §4.1 surveyed the format on 1,291 real human sessions
(`MaxDevv/real-pi-coding-agent-traces-sessions`); this module contains *only* format knowledge
(§7.1) and follows the reference adapter's edge policy (`claude_code.py`, DESIGN_FROZEN §2)
verbatim, so a graph built from a pi session and one built from a Claude Code transcript carry the
same edge types with the same provenance and the same thresholds.

Native entries (each has `type`, `id`, `parentId`, `timestamp`)
------------------------------------------------------------------
| entry                                               | IR                                                      |
|-----------------------------------------------------|---------------------------------------------------------|
| `type=session` (header: version, cwd)               | dropped; `cwd` kept in the adapter stats                 |
| `message.role=user`, text block(s) / string          | `kind="user"`                                            |
| `message.role=assistant`, `text` block               | `kind="assistant"`                                       |
| `message.role=assistant`, `thinking` block           | `kind="assistant"`, `meta.block="thinking"` (opt-in)     |
| `message.role=assistant`, `toolCall` block           | `kind="tool_call"`, `tool_call_id = block.id`            |
| `message.role=toolResult`                            | `kind="tool_result"`, `tool_call_id = message.toolCallId`|
| `type=compaction` (`summary`, `firstKeptEntryId`)    | `kind="summary"`, `meta.compaction="1"`                  |
| `type=branch_summary` (another branch, summarized)   | `kind="summary"`, `meta.branch_summary="1"`, no edges    |
| `type=custom_message` (e.g. subagent-notify)         | `kind="user"`, `meta.custom_type=<customType>`           |
| `message.role=bashExecution` (user-run `!cmd`)       | `kind="user"`, `meta.block="bashExecution"` (`$ cmd` + output; dropped when `excludeFromContext`) |
| `model_change` / `thinking_level_change` / `custom` / `session_info` | dropped (state, not conversation)        |

Differences from Claude Code that matter here
--------------------------------------------
* **A tool result is its own entry** (role `toolResult`), not a block inside a user row, and it
  names its call with `toolCallId`.  Pairing is by that id, never by order (contract #8).
* **Compaction is one entry**, not a boundary + summary pair: `firstKeptEntryId` names the first
  entry that stayed in the context; everything from the previous cut up to (not including) that
  entry was replaced by `summary`.  Those events get `summary --MATERIALIZES--> event`
  (`provenance="native"`, same predicate as Claude Code's compaction).  **The replaced entries
  are still in the file** -- pi appends, it never rewrites -- so the graph holds the full
  history plus the carrier, exactly the situation the closure rules were written for.
  A compaction whose `firstKeptEntryId` is unknown replaces everything since the previous cut
  and is counted as `compactions_without_kept_id`.
* **Edit tools are lower-case and quote different keys**: `edit{path, oldText, newText}` and
  `write{path, content}`.  `EDIT_TOOLS` / `QUOTE_KEYS` / the path key are the pi names; the
  40-char literal-quote rule, the ranking, the `max_quote_edges` cap, the value-source edges and
  the SUPERSEDES-by-path rule are the reference adapter's, unchanged.
* **Every entry has an id**, so `native_ref` is the entry id and `CONTROL` edges follow
  `parentId` (walked through dropped entries, labelled `parentId:transitive` when they had to).
* `step_id` / `event_id` are line-scoped exactly like the reference adapter
  (`"<source_id>:<line:08d>:<block:03d>"`), and `token_cost` uses the same estimator
  (`len//4`, or the exact tokenizer when `TRACEPACK_TOKENIZER` is set).

Pure and dependency-free; the only I/O is reading the session path.
"""
from __future__ import annotations

import json
import os
from bisect import bisect_left
from collections import Counter

try:  # normal import path
    from ..core.schema import RepresentationRef, TraceEdge, TraceEvent
    from ..core.graph import TraceGraph
    from ..core.values import extract_values
    from .base import AdapterError, build_graph, roundtrip_check, open_trace
    from .claude_code import _COMPACTION_PREDICATE, _estimate_tokens, _parse_timestamp, literal_overlap
except ImportError:  # pragma: no cover - direct execution
    import sys

    sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))
    from tracepack.core.schema import RepresentationRef, TraceEdge, TraceEvent  # type: ignore[no-redef]
    from tracepack.core.graph import TraceGraph  # type: ignore[no-redef]
    from tracepack.core.values import extract_values  # type: ignore[no-redef]
    from tracepack.adapters.base import AdapterError, build_graph, roundtrip_check, open_trace  # type: ignore[no-redef]
    from tracepack.adapters.claude_code import (  # type: ignore[no-redef]
        _COMPACTION_PREDICATE,
        _estimate_tokens,
        _parse_timestamp,
        literal_overlap,
    )

__all__ = ["PiAdapter", "EDIT_TOOLS", "QUOTE_KEYS", "is_pi_session"]

#: entry types that carry conversation content (measured on 66 real sessions, WP4_sources.md §4.1)
CONTENT_TYPES = frozenset({"message", "compaction", "branch_summary", "custom_message"})
#: entry types dropped on sight (harness state, not conversation)
SKIP_TYPES = frozenset({"session", "model_change", "thinking_level_change", "custom", "session_info", "label", "branch"})
#: pi tools whose arguments can literally quote an earlier tool result
EDIT_TOOLS = frozenset({"edit", "write"})
#: argument keys searched for that literal quote
QUOTE_KEYS = ("oldText", "newText", "content", "old_string", "new_string")
#: argument key naming the file a write-ish call targets (SUPERSEDES)
PATH_KEY = "path"


def is_pi_session(path) -> bool:
    """Cheap sniff: the first non-empty line is a `type=session` header with a `version`."""
    try:
        with open_trace(path) as fh:
            for line in fh:
                if not line.strip():
                    continue
                row = json.loads(line)
                return isinstance(row, dict) and row.get("type") == "session" and "version" in row
    except (OSError, ValueError):
        return False
    return False


def _text_of(content) -> "tuple[str, bool]":
    """content (str | list of blocks) -> (joined text, had_non_text_part)."""
    if isinstance(content, str):
        return content, False
    if isinstance(content, list):
        parts, other = [], False
        for part in content:
            if isinstance(part, dict) and part.get("type") == "text":
                parts.append(part.get("text") or "")
            elif isinstance(part, str):
                parts.append(part)
            else:
                other = True
        return "\n".join(parts), other
    return "", content is not None


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


class PiAdapter:
    """`TraceAdapter` for pi session `.jsonl` files.  Stateless; `normalize` is pure."""

    def __init__(self, *, source_id: str = "pi", include_thinking: bool = False,
                 link_edits: bool = True, link_writes: bool = True, link_values: bool = False,
                 value_max_sources: int = 3, max_value_edges: int = 4, value_lookback: int = 0,
                 value_require_result_first: bool = True,
                 quote_min_chars: int = 40, quote_lookback: int = 120,
                 quote_scan_cap: int = 120_000, quote_len_cap: int = 20_000,
                 max_quote_edges: int = 2, emit_control: bool = True, emit_temporal: bool = True) -> None:
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
        self.include_thinking = bool(include_thinking)
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
        self.emit_control = bool(emit_control)
        self.emit_temporal = bool(emit_temporal)

    def __repr__(self) -> str:  # pragma: no cover
        return "PiAdapter(source_id=%r, thinking=%s, link_edits=%s)" % (
            self.source_id, self.include_thinking, self.link_edits)

    # -------------------------------------------------------------- row input

    def _rows(self, native_trace):
        if isinstance(native_trace, (str, bytes, os.PathLike)):
            path = os.fspath(native_trace)
            if not os.path.isfile(path):
                raise AdapterError("session not found: %s" % (path,))
            with open(path, encoding="utf-8", errors="replace") as fh:
                for n, line in enumerate(fh, 1):
                    if not line.strip():
                        continue
                    try:
                        row = json.loads(line)
                    except ValueError:
                        continue          # a torn line: skipped, never guessed at
                    if isinstance(row, dict):
                        yield n, row
            return
        if isinstance(native_trace, dict):
            raise AdapterError("native_trace is a single dict; pass a path or a sequence of rows")
        try:
            iterator = enumerate(native_trace, 1)
        except TypeError:
            raise AdapterError("native_trace is neither a path nor an iterable of rows: %r"
                               % (type(native_trace).__name__,))
        for n, row in iterator:
            if not isinstance(row, dict):
                raise AdapterError("session row %d is %r, expected a JSON object" % (n, type(row).__name__))
            yield n, row

    def _blocks(self, row):
        """Row -> `[(kind, tool_call_id, text, meta_extra, raw_args)]` in native order."""
        rtype = row.get("type")
        out = []
        if rtype == "compaction":
            text = row.get("summary")
            text = text if isinstance(text, str) else ""
            out.append(("summary", None, text, {"compaction": "1"}, None))
            return out
        if rtype == "branch_summary":
            text = row.get("summary")
            if isinstance(text, str) and text.strip():
                out.append(("summary", None, text, {"branch_summary": "1"}, None))
            return out
        if rtype == "custom_message":
            text = row.get("content")
            if isinstance(text, str) and text.strip():
                out.append(("user", None, text, {"custom_type": str(row.get("customType") or "")}, None))
            return out
        if rtype != "message":
            return out
        message = row.get("message")
        message = message if isinstance(message, dict) else {}
        role = message.get("role")
        content = message.get("content")
        if role == "user":
            text, other = _text_of(content)
            if text.strip():
                extra = {"non_text_payload": "1"} if other else {}
                out.append(("user", None, text, extra, None))
            return out
        if role == "assistant":
            if isinstance(content, str):
                if content.strip():
                    out.append(("assistant", None, content, {}, None))
                return out
            for block in (content if isinstance(content, list) else []):
                if not isinstance(block, dict):
                    continue
                btype = block.get("type")
                if btype == "text":
                    text = block.get("text") or ""
                    if text.strip():
                        out.append(("assistant", None, text, {}, None))
                elif btype == "thinking":
                    if not self.include_thinking:
                        continue
                    text = block.get("thinking") or block.get("text") or ""
                    if text.strip():
                        out.append(("assistant", None, text, {"block": "thinking"}, None))
                elif btype == "toolCall":
                    name = block.get("name") or ""
                    args = block.get("arguments")
                    args = args if isinstance(args, dict) else {}
                    text = "%s %s" % (name, json.dumps(args, sort_keys=True, ensure_ascii=False))
                    out.append(("tool_call", block.get("id"), text, {"tool": name}, args))
            return out
        if role == "bashExecution":
            if message.get("excludeFromContext"):
                return out
            cmd = message.get("command") if isinstance(message.get("command"), str) else ""
            outp = message.get("output") if isinstance(message.get("output"), str) else ""
            text = ("$ %s\n%s" % (cmd, outp)).rstrip()
            if text.strip():
                extra = {"block": "bashExecution"}
                if message.get("cancelled"):
                    extra["cancelled"] = "1"
                out.append(("user", None, text, extra, None))
            return out
        if role == "toolResult":
            text, other = _text_of(content)
            extra = {}
            if message.get("toolName"):
                extra["tool"] = str(message.get("toolName"))
            if message.get("isError"):
                extra["is_error"] = "1"
            if other:
                extra["non_text_payload"] = "1"
            out.append(("tool_result", message.get("toolCallId"), text, extra, None))
            return out
        return out

    # ---------------------------------------------------------------- public

    def normalize(self, native_trace) -> TraceGraph:
        return self.normalize_with_stats(native_trace)[0]

    def normalize_with_stats(self, native_trace):
        events, event_line = [], []
        edges_result, edges_mat, edges_control, edges_temporal, edges_depends = [], [], [], [], []
        call_event_of = {}
        row_span = {}               # entry id -> (first_event_id, last_event_id)
        parent_links = []           # (child_first_event_id, parent entry id)
        parent_of = {}              # entry id -> parentId, for EVERY row
        line_of_id = {}             # entry id -> line_no, for EVERY row
        tool_results, edit_calls, consumers, write_targets = [], [], [], []
        prev_cut = 0
        last_ts = 0
        counts = Counter()
        cwd = None

        for line_no, row in self._rows(native_trace):
            rtype = row.get("type")
            rid, rparent = row.get("id"), row.get("parentId")
            if isinstance(rid, str):
                line_of_id.setdefault(rid, line_no)
                if isinstance(rparent, str):
                    parent_of.setdefault(rid, rparent)
            if rtype == "session":
                cwd = row.get("cwd") if isinstance(row.get("cwd"), str) else cwd
                counts["rows_filtered"] += 1
                continue
            if rtype not in CONTENT_TYPES:
                counts["rows_filtered"] += 1
                if rtype in SKIP_TYPES:
                    counts["rows_filtered_type"] += 1
                else:
                    counts["rows_unknown_type"] += 1
                continue
            blocks = self._blocks(row)
            if not blocks:
                counts["rows_no_blocks"] += 1
                continue

            native_ts = _parse_timestamp(row.get("timestamp"))
            ts = last_ts if native_ts is None else native_ts
            if native_ts is None:
                counts["rows_without_timestamp"] += 1
            adjusted = None
            if ts < last_ts:
                adjusted = ts
                ts = last_ts
                counts["timestamps_clamped"] += 1
            last_ts = ts

            first_eid = None
            for bidx, (kind, tool_id, text, extra, raw_args) in enumerate(blocks):
                eid = "%s:%08d:%03d" % (self.source_id, line_no, bidx)
                meta = {"row_type": str(rtype)}
                meta.update(extra)
                if adjusted is not None:
                    meta["timestamp_native"] = str(adjusted)
                group = ("%s:tool:%s" % (self.source_id, tool_id)) if tool_id else None
                cost = _estimate_tokens(text)
                events.append(TraceEvent(
                    event_id=eid, kind=kind, text=text, timestamp=ts, step_id="%08d" % line_no,
                    tool_call_id=tool_id, native_ref=rid if isinstance(rid, str) else None,
                    token_cost=cost, atomic_group=group, meta=meta,
                    representations=(RepresentationRef(kind="raw_text", token_cost=cost, text=text),),
                ))
                event_line.append(line_no)
                counts["kind_" + kind] += 1
                if self.link_values and kind in ("assistant", "tool_call"):
                    consumers.append((len(events) - 1, eid, text))
                if first_eid is None:
                    first_eid = eid

                if kind == "tool_call":
                    if tool_id:
                        if tool_id in call_event_of:
                            counts["duplicate_tool_call_ids"] += 1
                        else:
                            call_event_of[tool_id] = eid
                    else:
                        counts["tool_calls_without_id"] += 1
                    if extra.get("tool") in EDIT_TOOLS and isinstance(raw_args, dict):
                        fp = raw_args.get(PATH_KEY)
                        if isinstance(fp, str) and fp.strip():
                            write_targets.append((len(events) - 1, eid, fp.strip()))
                        if self.link_edits:
                            quotes = []
                            _collect_quotes(raw_args, quotes)
                            quotes = [q[:self.quote_len_cap] for q in quotes if len(q) >= self.quote_min_chars]
                            if quotes:
                                edit_calls.append((len(events) - 1, eid, quotes))
                elif kind == "tool_result":
                    tool_results.append((len(events) - 1, eid, text))
                    if tool_id and tool_id in call_event_of:
                        edges_result.append(TraceEdge(src_id=eid, dst_id=call_event_of[tool_id],
                                                      edge_type="RESULT_OF", predicate="toolCallId",
                                                      provenance="native"))
                        counts["result_of_edges"] += 1
                    else:
                        counts["unpaired_tool_results"] += 1
                elif kind == "summary":
                    counts["compactions"] += 1
                    cut = len(events) - 1
                    kept = row.get("firstKeptEntryId")
                    kept_line = line_of_id.get(kept) if isinstance(kept, str) else None
                    if kept_line is None:
                        counts["compactions_without_kept_id"] += 1
                    replaced = 0
                    for idx in range(prev_cut, cut):
                        if kept_line is not None and event_line[idx] >= kept_line:
                            continue
                        edges_mat.append(TraceEdge(src_id=eid, dst_id=events[idx].event_id,
                                                   edge_type="MATERIALIZES",
                                                   predicate=_COMPACTION_PREDICATE, provenance="native"))
                        replaced += 1
                    counts["materializes_edges"] += replaced
                    prev_cut = cut

            if isinstance(rid, str) and first_eid is not None:
                if rid in row_span:
                    counts["duplicate_entry_ids"] += 1
                else:
                    row_span[rid] = (first_eid, events[-1].event_id)
                if self.emit_control and isinstance(rparent, str):
                    parent_links.append((first_eid, rparent))

        # ---- CONTROL (parentId) and TEMPORAL (line order): weak, native ----
        if self.emit_control:
            for child_eid, parent_id in parent_links:
                pid, hops, seen = parent_id, 0, set()
                while pid is not None and pid not in row_span and hops < 64:
                    if pid in seen:
                        pid = None
                        break
                    seen.add(pid)
                    pid = parent_of.get(pid)
                    hops += 1
                span = row_span.get(pid) if pid is not None else None
                if span is None or span[1] == child_eid:
                    counts["control_unresolved"] += 1
                    continue
                edges_control.append(TraceEdge(src_id=child_eid, dst_id=span[1], edge_type="CONTROL",
                                               predicate="parentId" if hops == 0 else "parentId:transitive",
                                               provenance="native"))
                counts["control_direct" if hops == 0 else "control_transitive"] += 1
        if self.emit_temporal:
            for i in range(1, len(events)):
                edges_temporal.append(TraceEdge(src_id=events[i].event_id, dst_id=events[i - 1].event_id,
                                                edge_type="TEMPORAL", predicate="transcript_order",
                                                provenance="native"))

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
                        run = literal_overlap(quote, text, anchor=anchor, min_len=self.quote_min_chars,
                                              stop_at=rank_cap)
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

        edges = (edges_result + edges_mat + edges_depends + edges_values + edges_supersede
                 + edges_control + edges_temporal)
        graph = build_graph(events, edges)
        stats = dict(sorted(counts.items()))
        stats["n_events"] = len(events)
        stats["n_edges"] = len(edges)
        stats["total_tokens"] = sum(e.token_cost for e in events)
        if cwd is not None:
            stats["cwd"] = cwd
        return graph, stats

    def render(self, packet) -> str:
        context = getattr(packet, "context", None)
        if not isinstance(context, str):
            raise AdapterError("render expects a MemoryPacket with a str context, got %r" % (type(packet).__name__,))
        return context

    # ------------------------------------------- round-trip audit declarations

    def native_ids(self, native_trace):
        """Entry ids this adapter promises to keep as `native_ref`; a second, simpler pass."""
        out = []
        for _, row in self._rows(native_trace):
            rid = row.get("id")
            if not isinstance(rid, str) or row.get("type") not in CONTENT_TYPES:
                continue
            if self._row_has_content(row):
                out.append(rid)
        return tuple(out)

    def _row_has_content(self, row) -> bool:
        rtype = row.get("type")
        if rtype == "compaction":
            return True
        if rtype == "branch_summary":
            return isinstance(row.get("summary"), str) and bool(row["summary"].strip())
        if rtype == "custom_message":
            return isinstance(row.get("content"), str) and bool(row["content"].strip())
        message = row.get("message")
        message = message if isinstance(message, dict) else {}
        role, content = message.get("role"), message.get("content")
        if role == "toolResult":
            return True
        if role == "bashExecution":
            if message.get("excludeFromContext"):
                return False
            return bool(("%s%s" % (message.get("command") or "", message.get("output") or "")).strip())
        if role not in ("user", "assistant"):
            return False
        if isinstance(content, str):
            return bool(content.strip())
        if not isinstance(content, list):
            return False
        for block in content:
            if isinstance(block, str) and block.strip():
                return True
            if not isinstance(block, dict):
                continue
            btype = block.get("type")
            if btype == "toolCall":
                return True
            if btype == "text" and (block.get("text") or "").strip():
                return True
            if btype == "thinking" and self.include_thinking and (block.get("thinking") or block.get("text") or "").strip():
                return True
        return False

    def expected_strong_edges(self, native_trace):
        """`(result_entry_id, call_entry_id, "RESULT_OF")` resolved BY `toolCallId` -- independent of normalize."""
        call_entry_of, pairs = {}, []
        for _, row in self._rows(native_trace):
            if row.get("type") != "message":
                continue
            rid = row.get("id")
            message = row.get("message")
            message = message if isinstance(message, dict) else {}
            role = message.get("role")
            if role == "assistant":
                for block in (message.get("content") if isinstance(message.get("content"), list) else []):
                    if isinstance(block, dict) and block.get("type") == "toolCall" and block.get("id"):
                        call_entry_of.setdefault(block["id"], rid)
            elif role == "toolResult":
                tid = message.get("toolCallId")
                if tid in call_entry_of and isinstance(rid, str) and call_entry_of[tid] is not None:
                    pairs.append((rid, call_entry_of[tid], "RESULT_OF"))
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


def synthetic_session():
    """A pi session whose three tool results come back C, A, B (order pairing gets every pair
    wrong), with an `edit` quoting an earlier result, two `write`s to one path, and a compaction
    that keeps only the last two entries."""
    ts = lambda s: "2026-09-04T00:00:%02d.000Z" % s
    payload_a = "SAFETY_MARGIN = 0.15  # tuned on 2026-08-30 after the regression in run 4471"
    payload_b = "def load_config(path):\n    return yaml.safe_load(open(path))\n" * 2
    payload_c = "no matches found for NOT_THERE"
    return [
        {"type": "session", "version": 3, "id": "s0", "timestamp": ts(0), "cwd": "/work/demo"},
        {"type": "message", "id": "u1", "parentId": None, "timestamp": ts(1),
         "message": {"role": "user", "content": [{"type": "text", "text": "audit the three checks"}]}},
        {"type": "message", "id": "a1", "parentId": "u1", "timestamp": ts(2),
         "message": {"role": "assistant", "content": [
             {"type": "thinking", "thinking": "plan"},
             {"type": "toolCall", "id": "call_A", "name": "bash", "arguments": {"command": "grep -n SAFETY_MARGIN src"}},
             {"type": "toolCall", "id": "call_B", "name": "read", "arguments": {"path": "src/config.py"}}]}},
        {"type": "message", "id": "a1c", "parentId": "a1", "timestamp": ts(2),
         "message": {"role": "assistant", "content": [
             {"type": "toolCall", "id": "call_C", "name": "bash", "arguments": {"command": "grep NOT_THERE src"}}]}},
        {"type": "message", "id": "r1", "parentId": "a1c", "timestamp": ts(3),
         "message": {"role": "toolResult", "toolCallId": "call_C", "toolName": "bash", "isError": True,
                     "content": [{"type": "text", "text": payload_c}]}},
        {"type": "message", "id": "r2", "parentId": "r1", "timestamp": ts(4),
         "message": {"role": "toolResult", "toolCallId": "call_A", "toolName": "bash",
                     "content": [{"type": "text", "text": payload_a}]}},
        {"type": "message", "id": "r3", "parentId": "r2", "timestamp": ts(5),
         "message": {"role": "toolResult", "toolCallId": "call_B", "toolName": "read",
                     "content": [{"type": "text", "text": payload_b}]}},
        {"type": "model_change", "id": "m1", "parentId": "r3", "timestamp": ts(6), "provider": "x", "modelId": "y"},
        {"type": "message", "id": "a2", "parentId": "m1", "timestamp": ts(7),
         "message": {"role": "assistant", "content": [
             {"type": "text", "text": "The margin is 0.15; I will bump it."},
             {"type": "toolCall", "id": "call_D", "name": "edit",
              "arguments": {"path": "src/config.py", "oldText": payload_a, "newText": payload_a.replace("0.15", "0.20")}}]}},
        {"type": "message", "id": "r4", "parentId": "a2", "timestamp": ts(8),
         "message": {"role": "toolResult", "toolCallId": "call_D", "toolName": "edit",
                     "content": [{"type": "text", "text": "ok"}]}},
        {"type": "message", "id": "a3", "parentId": "r4", "timestamp": ts(9),
         "message": {"role": "assistant", "content": [
             {"type": "toolCall", "id": "call_E", "name": "write",
              "arguments": {"path": "src/config.py", "content": "SAFETY_MARGIN = 0.20\n"}}]}},
        {"type": "message", "id": "r5", "parentId": "a3", "timestamp": ts(10),
         "message": {"role": "toolResult", "toolCallId": "call_E", "toolName": "write",
                     "content": [{"type": "text", "text": "written"}]}},
        {"type": "compaction", "id": "c1", "parentId": "r5", "timestamp": ts(11),
         "summary": "## Goal\nbump SAFETY_MARGIN to 0.20 (done)", "firstKeptEntryId": "a3", "tokensBefore": 1234},
        {"type": "message", "id": "u2", "parentId": "c1", "timestamp": ts(12),
         "message": {"role": "user", "content": "thanks"}},
    ]


def _selfcheck() -> None:
    rows = synthetic_session()
    ad = PiAdapter()
    graph, stats = ad.normalize_with_stats(rows)
    ref = {e.event_id: e.native_ref for e in graph.events}
    kinds = Counter(e.kind for e in graph.events)
    assert kinds == {"user": 2, "tool_call": 5, "tool_result": 5, "assistant": 1, "summary": 1}, kinds
    # id pairing, never order: results came back C, A, B
    pairs = {ref[e.src_id]: ref[e.dst_id] for e in graph.edges if e.edge_type == "RESULT_OF"}
    assert pairs == {"r1": "a1c", "r2": "a1", "r3": "a1", "r4": "a2", "r5": "a3"}, pairs
    by_tid = {e.tool_call_id: e for e in graph.events if e.kind == "tool_result"}
    assert by_tid["call_A"].text.startswith("SAFETY_MARGIN") and by_tid["call_C"].meta.get("is_error") == "1"
    calls = {e.tool_call_id: e for e in graph.events if e.kind == "tool_call"}
    for tid in ("call_A", "call_B", "call_C", "call_D", "call_E"):
        assert calls[tid].atomic_group == by_tid[tid].atomic_group == "pi:tool:%s" % tid
    # the edit quotes result A (>=40 chars) -> DEPENDS_ON (inferred), and nothing else
    dep = [(ref[e.src_id], ref[e.dst_id], e.predicate) for e in graph.edges if e.edge_type == "DEPENDS_ON"]
    assert dep == [("a2", "r2", "literal_quote>=40")], dep
    sup = [(ref[e.src_id], ref[e.dst_id]) for e in graph.edges if e.edge_type == "SUPERSEDES"]
    assert sup == [("a3", "a2")], sup
    # compaction: everything before entry a3 is materialized by the summary; a3/r5 are kept
    mat = sorted({ref[e.dst_id] for e in graph.edges if e.edge_type == "MATERIALIZES"})
    assert mat == ["a1", "a1c", "a2", "r1", "r2", "r3", "r4", "u1"], mat
    assert stats["compactions"] == 1 and stats["materializes_edges"] == 10, stats   # u1 + 3 calls + 3 results + a2 (2) + r4
    # CONTROL walks through the dropped model_change entry
    ctrl = {(ref[e.src_id], ref[e.dst_id]): e.predicate for e in graph.edges if e.edge_type == "CONTROL"}
    assert ctrl[("a2", "r3")] == "parentId:transitive" and ctrl[("a1c", "a1")] == "parentId", ctrl
    assert stats["rows_filtered_type"] == 1 and stats["cwd"] == "/work/demo"
    # thinking is opt-in
    assert Counter(e.kind for e in PiAdapter(include_thinking=True).normalize(rows).events)["assistant"] == 2
    # round trip: the real audit passes ...
    rt = roundtrip_check(ad, rows)
    assert rt["ok"] and rt["native_ids_source"] == "adapter" and rt["n_declared_strong"] == 5, rt
    assert rt["n_native_ids_expected"] == 12 == rt["n_native_ids_present"], rt   # 14 entries minus session + model_change

    # ... and rejects an order-pairing regression
    class OrderPairing(PiAdapter):
        def normalize(self, native_trace):
            g = super().normalize(native_trace)
            calls_ = [e for e in g.events if e.kind == "tool_call"]
            results = [e for e in g.events if e.kind == "tool_result"]
            kept = [e for e in g.edges if e.edge_type != "RESULT_OF"]
            for res, call in zip(results, calls_):
                kept.append(TraceEdge(res.event_id, call.event_id, "RESULT_OF", predicate="order", provenance="native"))
            return build_graph(list(g.events), kept)

    _expect(AdapterError, roundtrip_check, OrderPairing(), rows)
    soft = roundtrip_check(OrderPairing(), rows, strict=False)
    # r1->call_A crosses entries a1c/a1 and r3->call_C crosses a1/a1c (declared-edge check);
    # r2->call_B stays inside entry a1 and is caught by the atomic-group check instead
    assert not soft["ok"] and len(soft["missing_strong"]) == 2, soft
    assert any("atomic_group" in f for f in soft["failures"]), soft

    # purity
    g2 = ad.normalize(rows)
    assert [(e.event_id, e.text, e.timestamp) for e in g2.events] == [(e.event_id, e.text, e.timestamp) for e in graph.events]
    assert [(e.src_id, e.dst_id, e.edge_type) for e in g2.edges] == [(e.src_id, e.dst_id, e.edge_type) for e in graph.edges]
    # guards
    _expect(AdapterError, PiAdapter, source_id="a:b")
    _expect(AdapterError, ad.normalize, {"type": "session"})
    _expect(AdapterError, ad.normalize, "/nonexistent/session.jsonl")
    print("[pi] ok  events=%d edges=%d strong=%d groups=%d" % (rt["n_events"], rt["n_edges"], rt["n_strong_edges"], rt["n_atomic_groups"]))
    print("[pi] fault injection: order-pairing rejected")
    print("PI_SELFCHECK_OK")


if __name__ == "__main__":
    import sys

    if len(sys.argv) > 1:
        for p in sys.argv[1:]:
            g, st = PiAdapter().normalize_with_stats(p)
            rt = roundtrip_check(PiAdapter(), p, strict=False)
            print("%s: events=%d edges=%d ok=%s %s" % (os.path.basename(p)[:60], st["n_events"], st["n_edges"], rt["ok"],
                                                       {k: v for k, v in st.items() if k.startswith(("kind_", "result_of", "unpaired", "compaction", "materializes", "depends_on_edges", "supersedes", "control_", "rows_"))}))
            if not rt["ok"]:
                print("   failures:", rt["failures"])
    else:
        _selfcheck()
