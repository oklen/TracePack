"""tracepack.adapters.registry -- pick the adapter for a transcript path by sniffing its first line.

WP4 adds transcripts in two more native formats next to Claude Code's.  Every eval script used to
write `ClaudeCodeAdapter().normalize(tr)`; they now call `adapter_for(tr).normalize(tr)`.  For a
Claude Code transcript this returns a default `ClaudeCodeAdapter()`, so every graph built before
WP4 is byte-identical (contract #13's golden digests are the regression test for that).

Detection is by content, not by path or extension, because the three formats all live in
`.jsonl` files:

| first non-empty line                                     | adapter              |
|----------------------------------------------------------|----------------------|
| `{"type": "session", "version": ...}`                    | `PiAdapter`          |
| `{"tracepack_format": "openhands", ...}` (our export)    | `OpenHandsAdapter`   |
| anything else (`{"type": "user"|"assistant"|...}`)       | `ClaudeCodeAdapter`  |

Adapters are constructed with defaults; a caller that needs options (`link_values=True`, say)
passes them through `adapter_for(path, **options)` -- the option names are the reference
adapter's, and a format that has no use for one ignores it.
"""
from __future__ import annotations

import json
import os

try:
    from .claude_code import ClaudeCodeAdapter
    from .pi import PiAdapter
    from .openhands import OpenHandsAdapter
    from .base import open_trace
except ImportError:  # pragma: no cover - direct execution
    import sys

    sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))
    from tracepack.adapters.claude_code import ClaudeCodeAdapter  # type: ignore[no-redef]
    from tracepack.adapters.pi import PiAdapter  # type: ignore[no-redef]
    from tracepack.adapters.openhands import OpenHandsAdapter  # type: ignore[no-redef]
    from tracepack.adapters.base import open_trace  # type: ignore[no-redef]

__all__ = ["adapter_for", "format_of", "FORMATS"]

FORMATS = ("claude_code", "pi", "openhands")


def format_of(path) -> str:
    """`"pi"` / `"openhands"` / `"claude_code"` from the first non-empty line; unreadable -> claude_code."""
    try:
        with open_trace(path) as fh:
            for line in fh:
                if not line.strip():
                    continue
                try:
                    row = json.loads(line)
                except ValueError:
                    return "claude_code"
                if not isinstance(row, dict):
                    return "claude_code"
                if row.get("type") == "session" and "version" in row:
                    return "pi"
                if row.get("tracepack_format") == "openhands":
                    return "openhands"
                return "claude_code"
    except OSError:
        return "claude_code"
    return "claude_code"


_ACCEPTED = {
    "claude_code": ("source_id", "include_sidechain", "include_thinking", "link_edits", "link_writes",
                    "link_values", "value_max_sources", "max_value_edges", "value_lookback",
                    "value_require_result_first", "quote_min_chars", "quote_lookback", "quote_scan_cap",
                    "quote_len_cap", "max_quote_edges", "emit_control", "emit_temporal",
                    "strict_compaction", "max_summary_gap"),
    "pi": ("source_id", "include_thinking", "link_edits", "link_writes", "link_values", "value_max_sources",
           "max_value_edges", "value_lookback", "value_require_result_first", "quote_min_chars",
           "quote_lookback", "quote_scan_cap", "quote_len_cap", "max_quote_edges", "emit_control",
           "emit_temporal"),
    "openhands": ("source_id", "link_edits", "link_writes", "link_values", "value_max_sources",
                  "max_value_edges", "value_lookback", "value_require_result_first", "quote_min_chars",
                  "quote_lookback", "quote_scan_cap", "quote_len_cap", "max_quote_edges", "emit_temporal"),
}


def adapter_for(path, **options):
    """The adapter for `path`, with `options` filtered to what that format's adapter accepts."""
    fmt = format_of(path)
    opts = {k: v for k, v in options.items() if k in _ACCEPTED[fmt]}
    if fmt == "pi":
        return PiAdapter(**opts)
    if fmt == "openhands":
        return OpenHandsAdapter(**opts)
    return ClaudeCodeAdapter(**opts)
