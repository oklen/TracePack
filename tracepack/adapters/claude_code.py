"""tracepack.adapters.claude_code -- the reference adapter: Claude Code transcript -> the IR.

Implements `TraceAdapter` (proposal §4.2) for the one native format this project actually has at
scale: `~/.claude/projects/<project>/<session>.jsonl`, one JSON object per line.  Per §7.1 it
contains *only* format knowledge -- no retrieval, no closure rules, no LLM inference, no carrier
verification.  Edge policy follows DESIGN_FROZEN §2 verbatim.

Parsing traps (all of them are inherited from `kvmemory/ca_extract.py`, which already solved them
on 12 real transcripts; this module is a re-implementation of that survey logic against the IR)
-----------------------------------------------------------------------------------------------

1. **`toolUseResult` is a duplicate.**  A `type:"user"` row carrying a tool result stores the
   payload TWICE: structured in the top-level `toolUseResult` field and again as a
   `tool_result` block inside `message.content`.  We read `message.content` ONLY.  Reading both
   silently doubles every tool-result token count -- and tool results are the bulk of a coding
   trace, so the budget experiments would be measuring a fantasy.
2. **`file-history-snapshot` rows are 24.5% of transcript bytes** and are editor state, not
   conversation.  They and the other non-message record types (`attachment`, `ai-title`,
   `queue-operation`, ...) are dropped -- see `SKIP_ROW_TYPES`, copied from ca_extract.
3. **`isSidechain: true` rows are a different agent's trace** (a Task/subagent run) interleaved
   into the same file.  Mixing them into the main graph invents dependencies across two
   independent contexts, so they are dropped by default (`include_sidechain=True` keeps them and
   marks `meta["sidechain"]="1"`; it never merges the two id spaces because event ids stay
   line-scoped).
4. **Compaction is two rows, not one**: a `type:"system", subtype:"compact_boundary"` row
   (carrying `compactMetadata.preservedMessages.uuids`) immediately followed by a
   `isCompactSummary:true` user row whose `message.content` is the summary string.  The boundary
   row is structure, not an event; the summary row is the carrier.
5. `logicalParentUuid` is unreliable for segmentation -- ca_extract's survey mandates positional
   segmentation between boundary lines, which is what `_compaction` does here.

Mapping to the IR (§4.1)
------------------------

| native                                        | IR                                                    |
|-----------------------------------------------|-------------------------------------------------------|
| `assistant` row, `text` block                 | `kind="assistant"`                                     |
| `assistant` row, `thinking` block             | `kind="assistant"`, `meta.block="thinking"` (opt-in)   |
| `assistant` row, `tool_use` block             | `kind="tool_call"`, `tool_call_id = block.id`          |
| `user` row, `tool_result` block               | `kind="tool_result"`, `tool_call_id = block.tool_use_id`|
| `user` row, string/`text` content             | `kind="user"`                                          |
| `user` row with `isCompactSummary`            | `kind="summary"`                                       |

Edges, and what grounds each one (nothing is emitted that a native field does not license):

* `tool_result --RESULT_OF--> tool_call`, `provenance="native"`, **paired by
  `tool_use_id` -> `id`, never by order.**  DESIGN_FROZEN §0 measured order-pairing on this
  corpus: 96.53% agreement overall (1,114 mispairs) and 31.8% agreement in the worst single
  session.  Order pairing raises no error; it just silently attaches results to the wrong call.
  A result whose call is not in this file (compacted away, or a resumed session) gets its event
  and its `tool_call_id`, but **no edge** -- counted as `unpaired_tool_results`.
  The pair shares one `atomic_group` (`"<source>:tool:<tool_use_id>"`, §3.4), derived from the
  native tool id so it is stable whether or not both halves are present.
* `summary --MATERIALIZES(predicate="compaction_summary")--> replaced event`,
  `provenance="native"`.  "Replaced" = every event between the previous boundary and this one
  whose row uuid is NOT in `compactMetadata.preservedMessages.uuids`.  **No `SUPERSEDES` edge is
  emitted**: compaction drops events from the context window, it does not invalidate them, and
  `SUPERSEDES` would tell a state query to prefer the summary over the source -- the exact
  carrier-first mistake §3.1 forbids.  `MATERIALIZES` is also not in `REQUIRED_EDGES`, so
  closure will not treat the summary as a licence to skip the source until §4.5 verification
  says so.
* `Edit/Write tool_call --DEPENDS_ON--> earlier tool_result`, `provenance="inferred"`, emitted
  **only** when a `>= 40`-character literal run is shared between the call's
  `old_string`/`new_string`/`content`/`new_source` and the result payload.  The 40-char
  threshold is DESIGN_FROZEN §2 (= EC3's `ca_written.LCS_MIN`): shorter literal overlaps in code
  traces are almost all path fragments and common tokens, which destroys precision.  Below the
  threshold *no edge at all* is emitted -- a weak edge would still be an invented claim.
  Candidates that clear the threshold are then **ranked by shared-run length** and only the top
  `max_quote_edges` (default 2) are linked, because 40 chars is not a high bar in practice: on a
  real transcript a 42-character absolute path clears it, and taking the most *recent* matches
  spent every slot on such paths while the actual source shared a 227-character code run.
  Whitespace-only runs never count at any length.  The evidence strength is carried on the edge
  as `predicate="literal_quote>=40|80|160"` (bucketed, so the value set stays small), because
  measuring it after the fact means re-deriving the match.  **Measured on a real transcript: 99%
  of these edges rest on a run under 80 characters, and the median run (42) is one filesystem
  path.**  `DEPENDS_ON` is a `REQUIRED_EDGE`, so closure follows it; the buckets let the eval
  harness see what that costs, and let a policy demand `>=80` without re-parsing anything.
* `event --CONTROL(predicate="parentUuid")--> parent row's last event` and
  `event --TEMPORAL(predicate="transcript_order")--> previous event`, both `provenance="native"`
  (parent-child and line order, DESIGN_FROZEN §2).  Both are WEAK: ordering and audit only.
  The `parentUuid` chain is walked *through* dropped rows (thinking-only turns are 19% of rows
  in this corpus, and stopping at the first one fragmented the control backbone on 30% of
  events); a link that stepped over one is labelled `predicate="parentUuid:transitive"` so the
  audit can tell a native adjacency from a native reachability.

Identity and cost
-----------------

* `step_id` = the transcript line number, **1-based** (so `sed -n '<step>p'` shows the row) and
  zero-padded to 8 digits.  Padding is not cosmetic: `TraceGraph` sorts by
  `(timestamp, event_id)`, many rows share a timestamp or have none, and unpadded `"9" > "10"`
  would scramble them.  (ca_extract prints 0-based `line_no`; add 1 to cross-reference.)
* `event_id` = `"<source_id>:<line:08d>:<block:03d>"` -- one line can hold several content
  blocks, so the block index is part of identity.  It is line-scoped rather than uuid-scoped
  because system rows have no uuid and a resumed session can repeat one.  `native_ref` carries
  the row `uuid`, which is what `roundtrip_check` audits.
* `token_cost = len(text) // 4`, and one `raw_text` representation with the same estimate.  This
  is a deterministic stand-in so the adapter stays tokenizer-free; the eval harness re-costs
  every event with the target model's tokenizer before any budget claim is made (§3.4 requires
  exact counts, and DESIGN_FROZEN §1 froze the budgets against this same `len//4` measure).
* `timestamp` = epoch milliseconds parsed from the ISO `timestamp` field, **clamped to
  non-decreasing**.  A row without one inherits the last seen value.  The clamp is not cosmetic:
  `TraceGraph` ranks events by `(timestamp, event_id)`, and a resumed or compacted session
  replays ancestor rows carrying their *original* timestamps -- 0.1%-0.8% of rows per file in
  this corpus, and a compaction summary can be weeks older than the file it sits in.  Left raw,
  that summary sorts to the front of the graph, ahead of every event it materializes, and the
  assembler serves a carrier before its sources.  Transcript position is the order the context
  actually had, so it wins; the displaced native value is preserved in
  `meta["timestamp_native"]` and counted as `timestamps_clamped`.

Pure and dependency-free: no clock, no RNG, no network, no numpy/torch.  The only I/O is reading
the transcript path the caller passes.
"""
from __future__ import annotations

import json
import os
from collections import Counter

try:  # normal import path
    from ..core.schema import RepresentationRef, TraceEdge, TraceEvent
    from ..core.graph import TraceGraph
    from .base import AdapterError, build_graph, roundtrip_check
except ImportError:  # pragma: no cover - direct execution
    import sys

    sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))
    from tracepack.core.schema import (  # type: ignore[no-redef]
        RepresentationRef,
        TraceEdge,
        TraceEvent,
    )
    from tracepack.core.graph import TraceGraph  # type: ignore[no-redef]
    from tracepack.adapters.base import (  # type: ignore[no-redef]
        AdapterError,
        build_graph,
        roundtrip_check,
    )

__all__ = ["ClaudeCodeAdapter", "SKIP_ROW_TYPES", "literal_overlap"]

#: non-message record types.  Copied from kvmemory/ca_extract.py (survey-verified); the first
#: entry alone is 24.5% of transcript bytes.
SKIP_ROW_TYPES = frozenset({
    "file-history-snapshot", "file-history-delta", "attachment", "custom-title", "ai-title",
    "mode", "permission-mode", "last-prompt", "queue-operation", "pr-link", "agent-name",
    "frame-link", "atis-latch", "cost-state", "worktree-state", "agent-setting",
})

#: only these carry conversation content
MESSAGE_ROW_TYPES = frozenset({"user", "assistant"})

#: tool calls whose arguments can literally quote an earlier tool result (DESIGN_FROZEN §2)
EDIT_TOOLS = frozenset({"Edit", "MultiEdit", "Write", "NotebookEdit", "Update"})

#: argument keys searched for that literal quote, at any nesting depth (MultiEdit nests them)
QUOTE_KEYS = ("old_string", "new_string", "content", "new_source")

_COMPACTION_PREDICATE = "compaction_summary"


# ------------------------------------------------------------------- helpers


_TOK = None
_TOK_PATH = os.environ.get("TRACEPACK_TOKENIZER", "")
_TIK = None

try:
    from ..core.values import extract_values
except ImportError:  # pragma: no cover - script-mode import
    from tracepack.core.values import extract_values


def _tokenizer():
    """The real tokenizer, loaded once, only when TRACEPACK_TOKENIZER names a tokenizer.json.

    Kept behind an env switch on purpose: the frozen budgets (DESIGN_FROZEN §1) and every number
    published before 2026-09-02 are on the ``len//4`` scale, and the contract tests assert that
    scale.  Exact accounting is a separate, explicitly labelled run -- red team round 2 §10.2
    measured ``len//4`` at ~2x under on this corpus, and the error is not a constant, so it can
    change which events fit and therefore the relative order of methods.
    """
    global _TOK
    if _TOK is None:
        from tokenizers import Tokenizer      # local import: the default path never needs it
        _TOK = Tokenizer.from_file(os.path.expanduser(_TOK_PATH))
    return _TOK


def _tiktoken_enc():
    global _TIK
    if _TIK is None:
        import tiktoken
        _TIK = tiktoken.get_encoding(_TOK_PATH.split(":", 1)[1] or "cl100k_base")
    return _TIK


def _estimate_tokens(text: str) -> int:
    """``len//4`` by default (the frozen scale); exact tokens when TRACEPACK_TOKENIZER is set.

    Two forms: a path to a ``tokenizer.json`` (HF ``tokenizers``), or ``tiktoken:<encoding>`` for the
    OpenAI-family BPE the API readers actually use.  Both are exact; which one is *right* depends on
    the reader, which is why neither is the default.
    """
    if not text:
        return 0
    if _TOK_PATH.startswith("tiktoken:"):
        return len(_tiktoken_enc().encode(text, disallowed_special=()))   # transcripts contain "<|endoftext|>" as literal text
    if _TOK_PATH:
        return len(_tokenizer().encode(text, add_special_tokens=False).ids)
    return len(text) // 4


def _parse_timestamp(raw) -> "int | None":
    """ISO-8601 (`2026-08-22T06:16:23.173Z`) -> epoch milliseconds; None if unusable."""
    if not isinstance(raw, str) or not raw:
        return None
    from datetime import datetime, timezone

    t = raw.strip()
    if t.endswith("Z"):
        t = t[:-1] + "+00:00"
    try:
        dt = datetime.fromisoformat(t)
    except ValueError:
        return None
    if dt.tzinfo is None:
        dt = dt.replace(tzinfo=timezone.utc)
    return int(dt.timestamp() * 1000)


def _result_text(block) -> "tuple[str, bool]":
    """`tool_result` block -> (text, had_non_text_part).  Never touches `toolUseResult` (trap 1)."""
    content = block.get("content")
    if isinstance(content, str):
        return content, False
    if isinstance(content, list):
        parts, other = [], False
        for part in content:
            if isinstance(part, dict) and part.get("type") == "text":
                parts.append(part.get("text") or "")
            else:
                other = True          # image / structured payload: recorded, not invented
        return "\n".join(parts), other
    return "", content is not None


def literal_overlap(quote: str, text: str, *, anchor: int = 20, min_len: int = 40,
                    max_occ: int = 8, stop_at: "int | None" = None) -> int:
    """Length of the longest literal common substring, early-exiting at `stop_at` (def. `min_len`).

    Anchors are the non-overlapping `anchor`-char windows of `quote`; any common substring of
    length `>= 2*anchor` fully contains one of them, so with `anchor = min_len // 2` this cannot
    miss a qualifying match.  `str.find` does the scanning in C, which is what makes this cheap
    enough to run against every earlier tool result (0.2s over 4k real transcript rows).

    Two deliberate conservatisms, both costing recall and never precision -- which is the right
    trade for an edge that closure is allowed to follow:

    * a whitespace-only run does not count, at any length: 40 aligned blanks in an indented file
      ground nothing;
    * if an anchor occurs more than `max_occ` times in `text` before the extendable occurrence,
      the match can be missed on very repetitive payloads.

    `stop_at` above `min_len` turns the boolean test into a comparable score, so callers can rank
    candidate sources by how much text they actually share.
    """
    if anchor < 1 or min_len < anchor:
        raise AdapterError("literal_overlap: need 1 <= anchor <= min_len")
    limit = min_len if stop_at is None else max(min_len, stop_at)
    if len(quote) < min_len or len(text) < min_len:
        return 0
    best = 0
    for i in range(0, len(quote) - anchor + 1, anchor):
        needle = quote[i:i + anchor]
        start, occ = 0, 0
        while occ < max_occ:
            j = text.find(needle, start)
            if j < 0:
                break
            occ += 1
            start = j + 1
            left = 0
            while i - left - 1 >= 0 and j - left - 1 >= 0 and quote[i - left - 1] == text[j - left - 1]:
                left += 1
            right = 0
            while (i + anchor + right < len(quote) and j + anchor + right < len(text)
                   and quote[i + anchor + right] == text[j + anchor + right]):
                right += 1
            total = left + anchor + right
            if total > best and quote[i - left:i + anchor + right].strip():
                best = total
            if best >= limit:
                return best
    return best


def _collect_quotes(value, out, *, depth: int = 0) -> None:
    """Recursively pull `QUOTE_KEYS` string values out of a tool_use input dict."""
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


# --------------------------------------------------------------- the adapter


class ClaudeCodeAdapter:
    """`TraceAdapter` for Claude Code `.jsonl` transcripts.  Stateless; `normalize` is pure.

    Options are constructor-only so that one adapter instance always produces one mapping; two
    different policies are two objects, which keeps the packet hash traceable to a policy.
    """

    def __init__(self, *, source_id: str = "cc", include_sidechain: bool = False,
                 include_thinking: bool = False, link_edits: bool = True,
                 link_writes: bool = True, link_values: bool = False,
                 value_max_sources: int = 3, max_value_edges: int = 4, value_lookback: int = 0,
                 value_require_result_first: bool = True,
                 quote_min_chars: int = 40, quote_lookback: int = 120,
                 quote_scan_cap: int = 120_000, quote_len_cap: int = 20_000,
                 max_quote_edges: int = 2, emit_control: bool = True, emit_temporal: bool = True,
                 strict_compaction: bool = False, max_summary_gap: int = 3) -> None:
        if not isinstance(source_id, str) or not source_id or ":" in source_id:
            raise AdapterError("source_id must be a non-empty string without ':' (got %r)"
                               % (source_id,))
        if quote_min_chars < 2:
            raise AdapterError("quote_min_chars must be >= 2 (DESIGN_FROZEN §2 froze it at 40)")
        for name, val in (("quote_lookback", quote_lookback), ("quote_scan_cap", quote_scan_cap),
                          ("quote_len_cap", quote_len_cap), ("max_quote_edges", max_quote_edges),
                          ("max_summary_gap", max_summary_gap),
                          ("value_max_sources", value_max_sources),
                          ("max_value_edges", max_value_edges), ("value_lookback", value_lookback)):
            if not isinstance(val, int) or val < 0:
                raise AdapterError("%s must be a non-negative int (got %r)" % (name, val))
        self.source_id = source_id
        self.include_sidechain = bool(include_sidechain)
        self.include_thinking = bool(include_thinking)
        self.link_edits = bool(link_edits)
        self.link_writes = bool(link_writes)
        #: phase 2 / WP3a -- OFF by default so every graph built before it is byte-identical.
        self.link_values = bool(link_values)
        self.value_max_sources = value_max_sources
        self.max_value_edges = max_value_edges
        self.value_lookback = value_lookback
        #: when False, a value the agent wrote before any tool output showed it (a path passed
        #: to Read, say) may still link a later mention to the output that holds it -- the
        #: output is then the file's CONTENT, which is evidence for questions about that path.
        #: Phase A measured the strict rule linking 0 of the 27 reachable-in-principle items.
        self.value_require_result_first = bool(value_require_result_first)
        self.quote_min_chars = quote_min_chars
        self.quote_lookback = quote_lookback
        self.quote_scan_cap = quote_scan_cap
        self.quote_len_cap = quote_len_cap
        self.max_quote_edges = max_quote_edges
        self.emit_control = bool(emit_control)
        self.emit_temporal = bool(emit_temporal)
        self.strict_compaction = bool(strict_compaction)
        self.max_summary_gap = max_summary_gap

    def __repr__(self) -> str:  # pragma: no cover - debugging aid
        return ("ClaudeCodeAdapter(source_id=%r, sidechain=%s, thinking=%s, link_edits=%s)"
                % (self.source_id, self.include_sidechain, self.include_thinking, self.link_edits))

    # -------------------------------------------------------------- row input

    def _rows(self, native_trace):
        """Yield `(line_no_1based, row_dict)` from a path or from already-parsed rows."""
        if isinstance(native_trace, (str, bytes, os.PathLike)):
            path = os.fspath(native_trace)
            if not os.path.isfile(path):
                raise AdapterError("transcript not found: %s" % (path,))
            with open(path, encoding="utf-8", errors="replace") as fh:
                for n, line in enumerate(fh, 1):
                    if not line.strip():
                        continue
                    try:
                        row = json.loads(line)
                    except ValueError:
                        continue          # a torn last line: skipped, never guessed at
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
                raise AdapterError("transcript row %d is %r, expected a JSON object"
                                   % (n, type(row).__name__))
            yield n, row

    def _keep_row(self, row) -> bool:
        """Type / sidechain filter (traps 2 and 3).  Compaction rows are handled separately."""
        rtype = row.get("type")
        if rtype in SKIP_ROW_TYPES or rtype not in MESSAGE_ROW_TYPES:
            return False
        if row.get("isSidechain") and not self.include_sidechain:
            return False
        return True

    def _blocks(self, row):
        """Row -> `[(kind, tool_call_id, text, meta_extra)]` in native block order."""
        message = row.get("message")
        message = message if isinstance(message, dict) else {}
        role = message.get("role") or row.get("type")
        content = message.get("content")
        out = []

        if row.get("isCompactSummary"):
            # the summary is always a bare string on this row; anything else is a format change
            # we would rather record as an empty carrier than silently mis-read
            text = content if isinstance(content, str) else ""
            out.append(("summary", None, text, {"compaction": "1"}))
            return out

        if isinstance(content, str):
            if content.strip():
                out.append(("user" if role == "user" else "assistant", None, content, {}))
            return out
        if not isinstance(content, list):
            return out

        for block in content:
            if not isinstance(block, dict):
                continue
            btype = block.get("type")
            if btype == "text":
                text = block.get("text") or ""
                if text.strip():
                    out.append(("user" if role == "user" else "assistant", None, text, {}))
            elif btype == "thinking":
                if not self.include_thinking:
                    continue
                text = block.get("thinking") or block.get("text") or ""
                if text.strip():
                    out.append(("assistant", None, text, {"block": "thinking"}))
            elif btype == "tool_use":
                name = block.get("name") or ""
                args = block.get("input")
                args = args if isinstance(args, dict) else {}
                text = "%s %s" % (name, json.dumps(args, sort_keys=True, ensure_ascii=False))
                out.append(("tool_call", block.get("id"), text, {"tool": name}))
            elif btype == "tool_result":
                text, had_other = _result_text(block)
                extra = {}
                if block.get("is_error"):
                    extra["is_error"] = "1"
                if had_other:
                    extra["non_text_payload"] = "1"
                out.append(("tool_result", block.get("tool_use_id"), text, extra))
        return out

    # ---------------------------------------------------------------- public

    def normalize(self, native_trace) -> TraceGraph:
        """Native transcript -> `TraceGraph`.  Pure: same bytes in, same graph out."""
        return self.normalize_with_stats(native_trace)[0]

    def normalize_with_stats(self, native_trace):
        """`(graph, stats)`; stats is returned rather than stashed so `normalize` stays pure."""
        events = []
        edges_result, edges_mat, edges_control, edges_temporal, edges_depends = [], [], [], [], []
        call_event_of = {}          # tool_use id -> event_id  (id pairing, never order)
        row_span = {}               # row uuid -> (first_event_id, last_event_id)
        parent_links = []           # (child_first_event_id, parent_uuid)
        tool_results = []           # (index_in_events, event_id, text) for quote search
        edit_calls = []             # (index_in_events, event_id, [quote strings])
        consumers = []              # (index_in_events, event_id, text) assistant/tool_call, for value edges
        write_targets = []          # (index_in_events, event_id, file_path) for SUPERSEDES
        parent_of = {}              # row uuid -> parentUuid, for EVERY row (dropped ones too)
        pending_boundary = None     # (line_no, preserved_uuids, cut_index)
        prev_cut = 0
        last_ts = 0
        counts = Counter()

        for line_no, row in self._rows(native_trace):
            rtype = row.get("type")

            # recorded for EVERY row, including the ones we drop: a row we chose not to
            # represent (a thinking-only turn -- 19% of rows here) still carries the native
            # parent link its children need to reach a retained ancestor.
            ruuid, rparent = row.get("uuid"), row.get("parentUuid")
            if isinstance(ruuid, str) and isinstance(rparent, str):
                parent_of.setdefault(ruuid, rparent)

            if rtype == "system" and row.get("subtype") == "compact_boundary":
                if pending_boundary is not None:
                    counts["dangling_boundaries"] += 1
                meta = row.get("compactMetadata")
                meta = meta if isinstance(meta, dict) else {}
                pres = meta.get("preservedMessages")
                uuids = pres.get("uuids") if isinstance(pres, dict) else None
                if not isinstance(uuids, list):
                    # no preserved list => every pre-boundary event was replaced.  Recorded,
                    # because "0 preserved" and "the field moved" look identical downstream.
                    uuids = []
                    counts["boundaries_without_preserved_list"] += 1
                pending_boundary = (line_no, frozenset(u for u in uuids if isinstance(u, str)),
                                    len(events))
                counts["compact_boundaries"] += 1
                continue

            if not self._keep_row(row):
                counts["rows_filtered"] += 1
                if rtype in SKIP_ROW_TYPES:
                    counts["rows_filtered_type"] += 1
                elif row.get("isSidechain"):
                    counts["rows_filtered_sidechain"] += 1
                continue

            blocks = self._blocks(row)
            if not blocks:
                counts["rows_no_blocks"] += 1
                continue

            uuid = row.get("uuid")
            native_ts = _parse_timestamp(row.get("timestamp"))
            ts = last_ts if native_ts is None else native_ts
            if native_ts is None:
                counts["rows_without_timestamp"] += 1
            adjusted = None
            if ts < last_ts:
                # Resumed / compacted sessions replay ancestor rows with their ORIGINAL
                # timestamps (measured: 0.1-0.8% of rows per file, and a compaction summary can
                # be weeks older than the file it lives in).  `TraceGraph` ranks by
                # (timestamp, event_id), so leaving those raw sorts a summary in front of the
                # events it materializes.  Clamp to non-decreasing -- transcript position is the
                # order the model's context actually had -- and keep the native value in meta.
                adjusted = ts
                ts = last_ts
                counts["timestamps_clamped"] += 1
            last_ts = ts

            first_eid = None
            for bidx, (kind, tool_id, text, extra) in enumerate(blocks):
                if bidx >= 1000:
                    counts["blocks_over_cap"] += 1
                eid = "%s:%08d:%03d" % (self.source_id, line_no, bidx)
                meta = {"row_type": str(rtype)}
                meta.update(extra)
                if adjusted is not None:
                    meta["timestamp_native"] = str(adjusted)
                if row.get("isSidechain"):
                    meta["sidechain"] = "1"
                if row.get("isMeta"):
                    meta["is_meta"] = "1"
                group = None
                if tool_id:
                    group = "%s:tool:%s" % (self.source_id, tool_id)
                cost = _estimate_tokens(text)
                events.append(TraceEvent(
                    event_id=eid, kind=kind, text=text, timestamp=ts,
                    step_id="%08d" % line_no, tool_call_id=tool_id, native_ref=uuid,
                    token_cost=cost, atomic_group=group, meta=meta,
                    representations=(RepresentationRef(kind="raw_text", token_cost=cost,
                                                       text=text),),
                ))
                counts["kind_" + kind] += 1
                if self.link_values and kind in ("assistant", "tool_call"):
                    consumers.append((len(events) - 1, eid, text))
                if first_eid is None:
                    first_eid = eid

                if kind == "tool_call":
                    if tool_id:
                        if tool_id in call_event_of:
                            counts["duplicate_tool_use_ids"] += 1
                        else:
                            call_event_of[tool_id] = eid
                    else:
                        counts["tool_calls_without_id"] += 1
                    if extra.get("tool") in EDIT_TOOLS:
                        # target path of a write-ish call -- basis for SUPERSEDES (see below)
                        _msg = row.get("message")
                        _raw = (_msg or {}).get("content") if isinstance(_msg, dict) else None
                        for _b in (_raw if isinstance(_raw, list) else []):
                            if (isinstance(_b, dict) and _b.get("type") == "tool_use"
                                    and _b.get("id") == tool_id):
                                _inp = _b.get("input")
                                _fp = _inp.get("file_path") if isinstance(_inp, dict) else None
                                if isinstance(_fp, str) and _fp.strip():
                                    write_targets.append((len(events) - 1, eid, _fp.strip()))
                    if self.link_edits and extra.get("tool") in EDIT_TOOLS:
                        quotes = []
                        message = row.get("message")
                        raw = (message or {}).get("content") if isinstance(message, dict) else None
                        for block in (raw if isinstance(raw, list) else []):
                            if (isinstance(block, dict) and block.get("type") == "tool_use"
                                    and block.get("id") == tool_id):
                                _collect_quotes(block.get("input"), quotes)
                        quotes = [q[:self.quote_len_cap] for q in quotes
                                  if len(q) >= self.quote_min_chars]
                        if quotes:
                            edit_calls.append((len(events) - 1, eid, quotes))
                elif kind == "tool_result":
                    tool_results.append((len(events) - 1, eid, text))
                    if tool_id and tool_id in call_event_of:
                        edges_result.append(TraceEdge(
                            src_id=eid, dst_id=call_event_of[tool_id], edge_type="RESULT_OF",
                            predicate="tool_use_id", provenance="native"))
                        counts["result_of_edges"] += 1
                    else:
                        # call not in this file (compacted away / resumed session): no edge,
                        # because there is nothing native to point at.
                        counts["unpaired_tool_results"] += 1
                elif kind == "summary":
                    counts["summaries"] += 1
                    cut = len(events) - 1
                    if pending_boundary is None or (line_no - pending_boundary[0]) > self.max_summary_gap:
                        # a resumed session can open with the summary; without a boundary we do
                        # not know what it replaced, so we invent nothing.
                        counts["unanchored_summaries"] += 1
                        if self.strict_compaction:
                            raise AdapterError(
                                "line %d: isCompactSummary without an adjacent compact_boundary"
                                % (line_no,))
                        pending_boundary = None
                        prev_cut = cut
                        continue
                    preserved = pending_boundary[1]
                    replaced = 0
                    for older in events[prev_cut:pending_boundary[2]]:
                        if older.native_ref is not None and older.native_ref in preserved:
                            continue
                        edges_mat.append(TraceEdge(
                            src_id=eid, dst_id=older.event_id, edge_type="MATERIALIZES",
                            predicate=_COMPACTION_PREDICATE, provenance="native"))
                        replaced += 1
                    counts["materializes_edges"] += replaced
                    pending_boundary = None
                    prev_cut = cut

            if uuid is not None and first_eid is not None:
                if uuid in row_span:
                    counts["duplicate_row_uuids"] += 1
                else:
                    row_span[uuid] = (first_eid, events[-1].event_id)
                parent_uuid = row.get("parentUuid")
                if self.emit_control and isinstance(parent_uuid, str):
                    parent_links.append((first_eid, parent_uuid))

        if pending_boundary is not None:
            counts["dangling_boundaries"] += 1
            if self.strict_compaction:
                raise AdapterError("line %d: compact_boundary with no summary row"
                                   % (pending_boundary[0],))

        # ---- CONTROL (parentUuid) and TEMPORAL (line order): weak, native ----
        if self.emit_control:
            for child_eid, parent_uuid in parent_links:
                # walk up the native chain until a RETAINED ancestor is found; a link that had
                # to step over dropped rows says so in its predicate rather than pretending to
                # be direct.  Bounded + visited-guarded: a corrupt file must not hang the parse.
                uuid_, hops, seen = parent_uuid, 0, set()
                while uuid_ is not None and uuid_ not in row_span and hops < 64:
                    if uuid_ in seen:
                        uuid_ = None
                        break
                    seen.add(uuid_)
                    uuid_ = parent_of.get(uuid_)
                    hops += 1
                span = row_span.get(uuid_) if uuid_ is not None else None
                if span is None or span[1] == child_eid:
                    counts["control_unresolved"] += 1
                    continue
                edges_control.append(TraceEdge(
                    src_id=child_eid, dst_id=span[1], edge_type="CONTROL",
                    predicate="parentUuid" if hops == 0 else "parentUuid:transitive",
                    provenance="native"))
                counts["control_direct" if hops == 0 else "control_transitive"] += 1
        if self.emit_temporal:
            for i in range(1, len(events)):
                edges_temporal.append(TraceEdge(
                    src_id=events[i].event_id, dst_id=events[i - 1].event_id,
                    edge_type="TEMPORAL", predicate="transcript_order", provenance="native"))

        # ---- DEPENDS_ON: >=40-char literal quote of an earlier tool_result ----
        # Candidates are RANKED by how many characters they actually share, not by recency.
        # Measured on a real transcript: the frozen 40-char threshold (DESIGN_FROZEN §2) admits
        # a 42-char absolute path -- exactly the "path fragment" noise that section predicts --
        # while the true source shared a 227-char code run.  Recency-first linking spent all 4
        # slots on the path matches; strength-first puts the real source in slot 1.  Ranking is a
        # tie-break over candidates that already cleared the frozen threshold, not a new one.
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
                        run = literal_overlap(quote, text, anchor=anchor,
                                              min_len=self.quote_min_chars, stop_at=rank_cap)
                        if run > best:
                            best, best_q = run, quote
                        if best >= rank_cap:
                            break
                    if best >= self.quote_min_chars:
                        scored.append((best, res_idx, res_eid, best_q))
                if scored:
                    counts["depends_on_calls"] += 1
                scored.sort(key=lambda t: (-t[0], -t[1]))   # strongest, then most recent
                for run, _idx, res_eid, best_q in scored[:self.max_quote_edges]:
                    bucket = self.quote_min_chars
                    for step in (2, 4):
                        if run >= step * self.quote_min_chars:
                            bucket = step * self.quote_min_chars
                    edges_depends.append(TraceEdge(
                        src_id=call_eid, dst_id=res_eid, edge_type="DEPENDS_ON",
                        predicate="literal_quote>=%d" % bucket, provenance="inferred",
                        meta={"match": "literal_quote", "run": str(run), "quote": best_q[:120], "n_candidates": str(len(scored))}))
                    counts["depends_on_edges"] += 1
                    counts["depends_on_run>=%d" % bucket] += 1

        # ---- DEPENDS_ON(value): "this value came from that tool output" (phase 2, WP3a) ----
        # The shipped DEPENDS_ON above links a >=40-char verbatim quote; the dependencies the
        # questions ask about are VALUES (a path, a line number, a hash, a count) that an
        # assistant turn or a later call repeats from an earlier tool output.  Rules, all
        # structural and gold-blind:
        #   * a value counts only if it ENTERED the session through a tool_result: if an
        #     assistant / tool_call mentioned it before any output showed it, the agent wrote it
        #     and the output merely echoes it -- no edge;
        #   * the value must occur in at most `value_max_sources` tool_results in the whole
        #     session (common numbers and paths are not evidence of anything);
        #   * the edge goes to the NEAREST earlier tool_result holding the value (the current
        #     reading of it), at most `max_value_edges` per consumer, rarest values first.
        edges_values = []
        if self.link_values and tool_results and consumers:
            from bisect import bisect_left
            res_values = {}
            res_eid_at = {}
            for r_idx, r_eid, r_text in tool_results:
                res_eid_at[r_idx] = r_eid
                for _typ, v in extract_values(r_text):
                    res_values.setdefault(v, []).append(r_idx)
            first_consumer = {}
            cons_vals = []
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
                    edges_values.append(TraceEdge(
                        src_id=c_eid, dst_id=dst, edge_type="DEPENDS_ON",
                        predicate="value:%s" % typ, provenance="inferred",
                        meta={"match": "value", "type": typ, "value": v, "n_sources": str(n_before), "n_sources_total": str(_n)}))
                    counts["value_edges"] += 1
                    counts["value_edges_%s" % typ] += 1
                    if len(seen_dst) >= self.max_value_edges:
                        break
                if seen_dst:
                    counts["value_consumers_linked"] += 1

        # ---- SUPERSEDES: consecutive write-ish calls to the SAME file_path ----
        # Grounded in a native field (the call's input.file_path), but the *claim* -- that the
        # later write invalidates the earlier one as "the current content of that file" -- is an
        # inference, so provenance is "inferred" and the predicate names the target.  Compaction
        # deliberately does NOT get a SUPERSEDES edge (see the module docstring): dropping context
        # is not invalidating it.  Chains of N writes give N-1 edges, newest -> previous.
        edges_supersede = []
        if self.link_writes:
            by_path = {}
            for idx, eid, fp in write_targets:
                prev = by_path.get(fp)
                if prev is not None:
                    edges_supersede.append(TraceEdge(
                        src_id=eid, dst_id=prev, edge_type="SUPERSEDES",
                        predicate="file_path=%s" % fp[-80:], provenance="inferred"))
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
        return graph, stats

    def render(self, packet) -> str:
        """`MemoryPacket` -> native context.  Deliberately trivial (§7.1: no policy here).

        The assembler already fixed the content, the order and the budget; re-formatting here
        would change the served string without changing the manifest hash that certifies it.
        """
        context = getattr(packet, "context", None)
        if not isinstance(context, str):
            raise AdapterError("render expects a MemoryPacket with a str context, got %r"
                               % (type(packet).__name__,))
        return context

    # ------------------------------------------- round-trip audit declarations

    def native_ids(self, native_trace):
        """Row uuids this adapter promises to preserve as `native_ref` (see base.py).

        Deliberately computed by a second, simpler pass: it decides retention from the row's
        *block types* only, without running the conversion, so `roundtrip_check` compares two
        code paths instead of comparing the converter with itself.
        """
        out = []
        for _, row in self._rows(native_trace):
            uuid = row.get("uuid")
            if uuid is None or not self._keep_row(row):
                continue
            if self._row_has_content(row):
                out.append(uuid)
        return tuple(out)

    def _row_has_content(self, row) -> bool:
        if row.get("isCompactSummary"):
            return True
        message = row.get("message")
        message = message if isinstance(message, dict) else {}
        content = message.get("content")
        if isinstance(content, str):
            return bool(content.strip())
        if not isinstance(content, list):
            return False
        for block in content:
            if not isinstance(block, dict):
                continue
            btype = block.get("type")
            if btype in ("tool_use", "tool_result"):
                return True
            if btype == "text" and (block.get("text") or "").strip():
                return True
            if btype == "thinking" and self.include_thinking and (
                    block.get("thinking") or block.get("text") or "").strip():
                return True
        return False

    def expected_strong_edges(self, native_trace):
        """`(result_uuid, call_uuid, "RESULT_OF")` triples, resolved BY `tool_use_id`.

        Independent of `normalize`; this is what catches an order-pairing regression.
        """
        call_uuid_of = {}
        pairs = []
        for _, row in self._rows(native_trace):
            if not self._keep_row(row):
                continue
            uuid = row.get("uuid")
            message = row.get("message")
            message = message if isinstance(message, dict) else {}
            content = message.get("content")
            if not isinstance(content, list):
                continue
            for block in content:
                if not isinstance(block, dict):
                    continue
                if block.get("type") == "tool_use" and block.get("id"):
                    call_uuid_of.setdefault(block["id"], uuid)
                elif block.get("type") == "tool_result":
                    tid = block.get("tool_use_id")
                    if tid in call_uuid_of and uuid is not None and call_uuid_of[tid] is not None:
                        pairs.append((uuid, call_uuid_of[tid], "RESULT_OF"))
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


def _synthetic_transcript():
    """8 rows: a filtered snapshot, a sidechain row, two calls whose results come back CROSSED,
    a duplicated `toolUseResult`, and an Edit quoting one of the results."""
    payload_b = ("def compute_budget(ctx_len, sys_len, out_reserve):\n"
                 "    return ctx_len - sys_len - out_reserve - SAFETY_MARGIN\n")
    return [
        # 1
        {"type": "user", "uuid": "u-1", "isSidechain": False,
         "timestamp": "2026-09-01T00:00:01.000Z",
         "message": {"role": "user", "content": "fix the budget helper"}},
        # 2  -- 24.5% of real transcript bytes; must never become an event
        {"type": "file-history-snapshot", "uuid": "snap-1", "snapshot": {"files": {"a": "b"}},
         "messageId": "m1"},
        # 3
        {"type": "assistant", "uuid": "a-1", "isSidechain": False,
         "timestamp": "2026-09-01T00:00:02.000Z",
         "message": {"role": "assistant", "content": [
             {"type": "text", "text": "Let me read both files."},
             {"type": "tool_use", "id": "toolu_A", "name": "Read",
              "input": {"file_path": "/tmp/notes.md"}}]}},
        # 4
        {"type": "assistant", "uuid": "a-2", "isSidechain": False, "parentUuid": "a-1",
         "timestamp": "2026-09-01T00:00:03.000Z",
         "message": {"role": "assistant", "content": [
             {"type": "tool_use", "id": "toolu_B", "name": "Read",
              "input": {"file_path": "/tmp/budget.py"}}]}},
        # 5  -- B answers FIRST: order pairing would attach it to call A
        {"type": "user", "uuid": "u-2", "isSidechain": False, "parentUuid": "a-2",
         "timestamp": "2026-09-01T00:00:04.000Z",
         "message": {"role": "user", "content": [
             {"type": "tool_result", "tool_use_id": "toolu_B",
              "content": [{"type": "text", "text": payload_b}]}]},
         # the duplicate the survey warns about: DIFFERENT text, must be ignored
         "toolUseResult": {"stdout": "DUPLICATE-MUST-NOT-BE-READ" * 40, "stderr": ""}},
        # 6
        {"type": "user", "uuid": "u-3", "isSidechain": False, "parentUuid": "u-2",
         "timestamp": "2026-09-01T00:00:05.000Z",
         "message": {"role": "user", "content": [
             {"type": "tool_result", "tool_use_id": "toolu_A",
              "content": "notes: remember the safety margin"}]}},
        # 7  -- quotes >=40 chars of tool result B verbatim
        {"type": "assistant", "uuid": "a-3", "isSidechain": False, "parentUuid": "u-3",
         "timestamp": "2026-09-01T00:00:06.000Z",
         "message": {"role": "assistant", "content": [
             {"type": "tool_use", "id": "toolu_C", "name": "Edit",
              "input": {"file_path": "/tmp/budget.py",
                        "old_string": "    return ctx_len - sys_len - out_reserve - SAFETY_MARGIN\n",
                        "new_string": "    return max(0, ctx_len - sys_len - out_reserve)\n"}}]}},
        # 8  -- a sub-agent's trace sharing the file
        {"type": "assistant", "uuid": "s-1", "isSidechain": True,
         "timestamp": "2026-09-01T00:00:07.000Z",
         "message": {"role": "assistant", "content": [
             {"type": "text", "text": "sidechain: searching for the caller"}]}},
    ]


def _compaction_transcript():
    """boundary + summary, with one pre-boundary row preserved."""
    return [
        {"type": "user", "uuid": "c-1", "timestamp": "2026-09-01T00:00:01.000Z",
         "message": {"role": "user", "content": "old question one"}},
        {"type": "assistant", "uuid": "c-2", "timestamp": "2026-09-01T00:00:02.000Z",
         "message": {"role": "assistant", "content": [{"type": "text", "text": "old answer"}]}},
        {"type": "user", "uuid": "c-3", "timestamp": "2026-09-01T00:00:03.000Z",
         "message": {"role": "user", "content": "keep me"}},
        {"type": "system", "subtype": "compact_boundary", "uuid": "b-1",
         "compactMetadata": {"trigger": "auto", "preTokens": 900,
                             "preservedMessages": {"uuids": ["c-3"]}}},
        # the summary carries the ANCESTOR session's timestamp -- weeks older than this file
        {"type": "user", "uuid": "c-4", "isCompactSummary": True,
         "timestamp": "2026-07-24T16:11:35.395Z",
         "message": {"role": "user", "content": "Summary: the user asked about budgets."}},
        {"type": "assistant", "uuid": "c-5", "timestamp": "2026-09-01T00:00:05.000Z",
         "message": {"role": "assistant", "content": [{"type": "text", "text": "carrying on"}]}},
    ]


def _selfcheck() -> None:
    rows = _synthetic_transcript()
    adapter = ClaudeCodeAdapter()
    graph, stats = adapter.normalize_with_stats(rows)
    by_id = {e.event_id: e for e in graph.events}
    ref = {e.event_id: e.native_ref for e in graph.events}

    # --- 1. filtering (traps 2 and 3) --------------------------------------
    refs = {e.native_ref for e in graph.events}
    assert "snap-1" not in refs, "file-history-snapshot row leaked into the graph"
    assert "s-1" not in refs, "isSidechain row leaked into the main trace"
    assert stats["rows_filtered"] == 2, stats
    keeping = ClaudeCodeAdapter(include_sidechain=True).normalize(rows)
    assert "s-1" in {e.native_ref for e in keeping.events}, "include_sidechain must keep it"
    assert len(keeping.events) == len(graph.events) + 1

    # --- 2. toolUseResult duplication (trap 1) ------------------------------
    res_b = [e for e in graph.events if e.native_ref == "u-2"][0]
    assert "DUPLICATE-MUST-NOT-BE-READ" not in res_b.text, "read toolUseResult instead of content"
    assert res_b.text.startswith("def compute_budget"), res_b.text[:40]
    if not _TOK_PATH:
        assert res_b.token_cost == len(res_b.text) // 4, "token estimate must be len//4"
    assert res_b.representation("raw_text").text == res_b.text

    # --- 3. RESULT_OF pairs BY ID, and order pairing would differ -----------
    result_of = [e for e in graph.edges if e.edge_type == "RESULT_OF"]
    assert len(result_of) == 2, result_of
    got = {(ref[e.src_id], ref[e.dst_id]) for e in result_of}
    assert got == {("u-2", "a-2"), ("u-3", "a-1")}, got
    # what pairing by order would have produced, computed independently:
    calls = [e for e in graph.events if e.kind == "tool_call" and e.meta.get("tool") == "Read"]
    results = [e for e in graph.events if e.kind == "tool_result"]
    by_order = {(ref[r.event_id], ref[c.event_id]) for r, c in zip(results, calls)}
    assert by_order == {("u-2", "a-1"), ("u-3", "a-2")}, by_order
    assert by_order != got, "the fixture must actually distinguish id pairing from order pairing"
    for e in result_of:
        assert e.provenance == "native" and e.predicate == "tool_use_id"
        assert by_id[e.src_id].atomic_group == by_id[e.dst_id].atomic_group is not None
    assert graph.atomic_group("cc:tool:toolu_B") and len(graph.atomic_group("cc:tool:toolu_B")) == 2

    # --- 4. DEPENDS_ON only on a >=40-char literal quote, provenance inferred
    dep = [e for e in graph.edges if e.edge_type == "DEPENDS_ON"]
    assert len(dep) == 1, dep
    assert (ref[dep[0].src_id], ref[dep[0].dst_id]) == ("a-3", "u-2"), dep
    assert dep[0].provenance == "inferred", dep
    assert dep[0].predicate.startswith("literal_quote>="), dep[0].predicate
    assert int(dep[0].predicate.split(">=")[1]) >= 40, dep[0].predicate
    short = ClaudeCodeAdapter(quote_min_chars=4000).normalize(rows)
    assert not [e for e in short.edges if e.edge_type == "DEPENDS_ON"], \
        "a quote below threshold must produce NO edge"
    assert not [e for e in ClaudeCodeAdapter(link_edits=False).normalize(rows).edges
                if e.edge_type == "DEPENDS_ON"]
    # the matcher itself: 39 shared chars is not enough, 40 is
    base = "0123456789" * 6
    assert literal_overlap("x" + base[:39] + "y", "z" + base[:39] + "w") == 39
    assert literal_overlap("x" + base[:40] + "y", "z" + base[:40] + "w") >= 40
    assert literal_overlap(" " * 80, " " * 80) == 0, "whitespace-only runs must not count"
    assert literal_overlap(base * 4, base * 4, stop_at=200) >= 200, "stop_at must score, not gate"
    _expect(AdapterError, literal_overlap, "a" * 50, "a" * 50, anchor=0)

    # ranking: a long code quote must outrank a bare path fragment that only just clears 40
    path = "/home/someone/workspaces/projects/tracepack/core/assembler.py"
    body = ("def assemble(query, closure, budget):\n"
            "    entries, spent = [], 0\n"
            "    for eid in closure.required:\n"
            "        spent += cost_of(eid)\n")
    rank_rows = [
        {"type": "assistant", "uuid": "r-1", "timestamp": "2026-09-01T00:00:01.000Z",
         "message": {"role": "assistant", "content": [
             {"type": "tool_use", "id": "t_body", "name": "Read", "input": {"file_path": path}}]}},
        {"type": "user", "uuid": "r-2", "timestamp": "2026-09-01T00:00:02.000Z",
         "message": {"role": "user", "content": [
             {"type": "tool_result", "tool_use_id": "t_body", "content": body}]}},
        {"type": "assistant", "uuid": "r-3", "timestamp": "2026-09-01T00:00:03.000Z",
         "message": {"role": "assistant", "content": [
             {"type": "tool_use", "id": "t_ls", "name": "Bash", "input": {"command": "ls"}}]}},
        {"type": "user", "uuid": "r-4", "timestamp": "2026-09-01T00:00:04.000Z",
         "message": {"role": "user", "content": [
             {"type": "tool_result", "tool_use_id": "t_ls", "content": path}]}},   # 61-char path
        {"type": "assistant", "uuid": "r-5", "timestamp": "2026-09-01T00:00:05.000Z",
         "message": {"role": "assistant", "content": [
             {"type": "tool_use", "id": "t_edit", "name": "Edit",
              # new_string mentions the path, so BOTH earlier results clear 40 chars
              "input": {"file_path": path, "old_string": body,
                        "new_string": body.replace("spent", "used") + "# see " + path + "\n"}}]}},
    ]
    rg = ClaudeCodeAdapter(max_quote_edges=1).normalize(rank_rows)
    rref = {e.event_id: e.native_ref for e in rg.events}
    rdep = [e for e in rg.edges if e.edge_type == "DEPENDS_ON"]
    assert len(rdep) == 1 and rref[rdep[0].dst_id] == "r-2", \
        "the 227-char body must outrank the 61-char path fragment: %s" % (
            [(rref[e.src_id], rref[e.dst_id]) for e in rdep],)
    assert len([e for e in ClaudeCodeAdapter(max_quote_edges=2).normalize(rank_rows).edges
                if e.edge_type == "DEPENDS_ON"]) == 2, "both candidates do clear the threshold"

    # --- 5. ids, steps, ordering -------------------------------------------
    a1 = [e for e in graph.events if e.native_ref == "a-1"]
    assert len(a1) == 2 and [e.kind for e in a1] == ["assistant", "tool_call"], a1
    assert a1[0].event_id == "cc:00000003:000" and a1[1].event_id == "cc:00000003:001"
    assert a1[0].step_id == "%08d" % 3, a1[0].step_id
    assert list(graph.event_ids) == sorted(graph.event_ids), "padding must keep ids sortable"
    assert [ref[i] for i in graph.event_ids] == ["u-1", "a-1", "a-1", "a-2", "u-2", "u-3", "a-3"]

    # --- 6. weak edges are grounded ----------------------------------------
    control = [e for e in graph.edges if e.edge_type == "CONTROL"]
    assert control and all(e.provenance == "native" and e.predicate.startswith("parentUuid")
                           for e in control)
    assert ("a-2", "a-1") in {(ref[e.src_id], ref[e.dst_id]) for e in control}
    # the parent chain must survive a dropped (thinking-only) row, and say that it stepped over one
    thinking_gap = list(rows)
    thinking_gap.insert(3, {"type": "assistant", "uuid": "t-1", "parentUuid": "a-1",
                            "timestamp": "2026-09-01T00:00:02.500Z",
                            "message": {"role": "assistant",
                                        "content": [{"type": "thinking", "thinking": "hmm"}]}})
    thinking_gap[4] = dict(thinking_gap[4], parentUuid="t-1")   # a-2's parent is now dropped
    tg, tstats = adapter.normalize_with_stats(thinking_gap)
    tref = {e.event_id: e.native_ref for e in tg.events}
    tctl = {(tref[e.src_id], tref[e.dst_id]): e.predicate
            for e in tg.edges if e.edge_type == "CONTROL"}
    assert tstats["rows_no_blocks"] == 1, tstats
    assert tctl.get(("a-2", "a-1")) == "parentUuid:transitive", tctl
    assert tstats["control_transitive"] == 1, tstats
    temporal = [e for e in graph.edges if e.edge_type == "TEMPORAL"]
    assert len(temporal) == len(graph.events) - 1
    assert not [e for e in ClaudeCodeAdapter(emit_control=False, emit_temporal=False)
                .normalize(rows).edges if e.edge_type in ("CONTROL", "TEMPORAL")]

    # --- 7. round-trip -----------------------------------------------------
    rt = roundtrip_check(adapter, rows)
    assert rt["ok"] and rt["native_ids_source"] == "adapter", rt
    assert rt["n_native_ids_expected"] == 6 and not rt["missing_native_ids"], rt
    assert rt["n_declared_strong"] == 2 and not rt["missing_strong"], rt
    assert roundtrip_check(adapter, iter(rows))["n_events"] == rt["n_events"]

    # --- 8. fault injection: an adapter that pairs by ORDER is rejected ------
    class OrderPairingAdapter(ClaudeCodeAdapter):
        """The silent corruption DESIGN_FROZEN §0 measured at 3.5% (68% worst session)."""

        def normalize_with_stats(self, native_trace):
            graph_, stats_ = ClaudeCodeAdapter.normalize_with_stats(self, native_trace)
            calls_ = [e for e in graph_.events if e.kind == "tool_call"]
            results_ = [e for e in graph_.events if e.kind == "tool_result"]
            keep = [e for e in graph_.edges if e.edge_type != "RESULT_OF"]
            crossed = [TraceEdge(src_id=r.event_id, dst_id=c.event_id, edge_type="RESULT_OF",
                                 predicate="position", provenance="native")
                       for r, c in zip(results_, calls_)]
            fixed = []
            for e in graph_.events:
                grp = e.atomic_group
                for r, c in zip(results_, calls_):
                    if e.event_id == r.event_id:
                        grp = c.atomic_group
                fixed.append(TraceEvent(
                    event_id=e.event_id, kind=e.kind, text=e.text, timestamp=e.timestamp,
                    step_id=e.step_id, tool_call_id=e.tool_call_id, native_ref=e.native_ref,
                    token_cost=e.token_cost, representations=e.representations,
                    atomic_group=grp, meta=dict(e.meta)))
            return build_graph(fixed, keep + crossed), stats_

        def normalize(self, native_trace):
            return self.normalize_with_stats(native_trace)[0]

    bad = OrderPairingAdapter()
    assert len(bad.normalize(rows).edges) == len(graph.edges)   # a structurally valid graph ...
    _expect(AdapterError, roundtrip_check, bad, rows)           # ... that the audit rejects
    soft = roundtrip_check(bad, rows, strict=False)
    assert len(soft["missing_strong"]) == 2, soft

    # --- 9. malformed input is rejected, never guessed at -------------------
    _expect(AdapterError, adapter.normalize, 12345)
    _expect(AdapterError, adapter.normalize, rows[:1] + ["not a dict"])
    _expect(AdapterError, adapter.normalize, "/nonexistent/transcript.jsonl")
    _expect(AdapterError, adapter.normalize, {"type": "user"})
    _expect(AdapterError, ClaudeCodeAdapter, source_id="a:b")
    _expect(AdapterError, ClaudeCodeAdapter, quote_lookback=-1)
    _expect(AdapterError, adapter.render, object())

    # --- 10. compaction -----------------------------------------------------
    crows = _compaction_transcript()
    cgraph, cstats = adapter.normalize_with_stats(crows)
    cref = {e.event_id: e.native_ref for e in cgraph.events}
    summary = [e for e in cgraph.events if e.kind == "summary"]
    assert len(summary) == 1 and summary[0].native_ref == "c-4", summary
    mat = [e for e in cgraph.edges if e.edge_type == "MATERIALIZES"]
    assert {(cref[e.src_id], cref[e.dst_id]) for e in mat} == {("c-4", "c-1"), ("c-4", "c-2")}, mat
    assert all(e.predicate == _COMPACTION_PREDICATE and e.provenance == "native" for e in mat)
    assert "c-3" not in {cref[e.dst_id] for e in mat}, "preservedMessages uuid must be excluded"
    assert not [e for e in cgraph.edges if e.edge_type == "SUPERSEDES"], \
        "compaction must not emit SUPERSEDES (§3.1: it drops context, it does not invalidate)"
    assert cstats["compact_boundaries"] == 1 and cstats["materializes_edges"] == 2, cstats
    assert roundtrip_check(adapter, crows)["ok"]

    # the summary's replayed ancestor timestamp must NOT sort it ahead of what it materializes
    assert cstats["timestamps_clamped"] == 1, cstats
    assert summary[0].meta["timestamp_native"] == str(_parse_timestamp("2026-07-24T16:11:35.395Z"))
    order = list(cgraph.event_ids)
    assert order == sorted(order), "clamping must keep graph order == transcript order"
    for e in mat:
        assert order.index(e.src_id) > order.index(e.dst_id), \
            "carrier served before its source: %s" % (e,)

    # a summary with no boundary invents nothing; strict mode refuses it
    orphan = [crows[0], crows[4]]
    ograph, ostats = adapter.normalize_with_stats(orphan)
    assert ostats["unanchored_summaries"] == 1
    assert not [e for e in ograph.edges if e.edge_type == "MATERIALIZES"]
    _expect(AdapterError, ClaudeCodeAdapter(strict_compaction=True).normalize, orphan)
    _expect(AdapterError, ClaudeCodeAdapter(strict_compaction=True).normalize, crows[:4])

    # --- 11. determinism + render ------------------------------------------
    g2, s2 = ClaudeCodeAdapter().normalize_with_stats(rows)
    assert [e.event_id for e in g2.events] == [e.event_id for e in graph.events]
    assert [(e.src_id, e.dst_id, e.edge_type) for e in g2.edges] == \
           [(e.src_id, e.dst_id, e.edge_type) for e in graph.edges]
    assert s2 == stats

    class _FakePacket:
        context = "EVIDENCE\n---\nx"

    assert adapter.render(_FakePacket()) == "EVIDENCE\n---\nx"

    print("[claude_code] synthetic ok  events=%d edges=%s tokens=%d"
          % (rt["n_events"], rt["edge_types"], rt["total_tokens"]))
    print("[claude_code] fault injection: order-pairing rejected, sub-threshold quote -> no edge, "
          "snapshot+sidechain filtered, malformed rows rejected")

    _real_transcript_smoke()
    print("CLAUDE_CODE_SELFCHECK_OK")


def _real_transcript_smoke(max_rows: int = 4000) -> None:
    """Run against real transcripts if this machine has any.  Never fails the selfcheck."""
    try:
        root = os.path.expanduser("~/.claude/projects")
        if not os.path.isdir(root):
            print("[claude_code] real transcripts: none on this machine")
            return
        paths = []
        for proj in sorted(os.listdir(root)):
            d = os.path.join(root, proj)
            if not os.path.isdir(d):
                continue
            for f in sorted(os.listdir(d)):
                if f.endswith(".jsonl"):
                    paths.append(os.path.join(d, f))
        if not paths:
            print("[claude_code] real transcripts: none found")
            return
        adapter = ClaudeCodeAdapter()
        shown = 0
        for path in paths:
            rows = []
            with open(path, encoding="utf-8", errors="replace") as fh:
                for i, line in enumerate(fh):
                    if i >= max_rows:
                        break
                    line = line.strip()
                    if not line:
                        continue
                    try:
                        row = json.loads(line)
                    except ValueError:
                        continue
                    if isinstance(row, dict):
                        rows.append(row)
            if len(rows) < 20:
                continue
            st = roundtrip_check(adapter, rows)
            _, ns = adapter.normalize_with_stats(rows)
            assert st["ok"], st
            print("[claude_code] REAL %-40s rows=%-5d events=%-5d edges=%s"
                  % (os.path.basename(path)[:40], len(rows), st["n_events"], st["edge_types"]))
            print("               native_ids=%d/%d  declared RESULT_OF=%d missing=%d  "
                  "unpaired=%d  summaries=%d  materializes=%d  depends_on=%d  tokens=%d"
                  % (st["n_native_ids_present"], st["n_native_ids_expected"],
                     st["n_declared_strong"], len(st["missing_strong"]),
                     ns.get("unpaired_tool_results", 0), ns.get("summaries", 0),
                     ns.get("materializes_edges", 0), ns.get("depends_on_edges", 0),
                     st["total_tokens"]))
            shown += 1
            if shown >= 3:
                break
        if not shown:
            print("[claude_code] real transcripts: all candidates too short")
    except Exception as e:  # noqa: BLE001 - a machine without transcripts must still pass
        print("[claude_code] real-transcript smoke skipped: %s: %s" % (type(e).__name__, e))


if __name__ == "__main__":
    _selfcheck()
