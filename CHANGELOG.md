# Changelog

## 0.2.0 — 2026-10-08

TracePack becomes a Claude Code plugin.

- **Plugin.** This repository is now a plugin and its own marketplace:
  - Install with `/plugin marketplace add oklen/TracePack` and then `/plugin install tracepack@tracepack`.
  - PreCompact hook: notes the `/compact` instructions.
  - SessionStart hook: right after a compaction, adds back the exact records the summary is likely to
    drop. By default that is at most 1,500 tokens and 8,000 characters, with a one-line notice to the user.
- **MCP server** with no dependencies, offering three read-only tools:
  - `recall`: verbatim records about something;
  - `expand`: one record in full, page by page;
  - `status`.

  It binds to the session that calls it, and follows `/clear` and `/resume`.
- **Commands:** `/tracepack:recall`, `/tracepack:status`, `/tracepack:last`.
- **`tracepack` command line:** `recall`, `expand`, `status`, `last`, `sessions`, `demo`, `doctor`, `mcp`.
- **`tracepack.session` API:**
  - searches only records that are no longer in the model's context;
  - labels each record with tool, time and transcript line;
  - for a very large transcript, reads only its newest part.
- **Secrets masked** in everything handed back (`TRACEPACK_REDACT=0` turns this off).
- **Fixed:** `tracepack.pack()` now reads Claude Code transcripts. Version 0.1.0 parsed them as pi
  sessions and found no events.
- **Tests:** contract 18, with 17 plugin tests, all on synthetic sessions. 293 tests in total.
- The research notes and the library reference moved to `docs/RESEARCH.md`.

## 0.1.0 — 2026-10-08

- First code release of the research library: trace adapters (Claude Code, pi, OpenHands), router,
  typed closure, budget assembler, profiler, OpenHands condenser, LlamaIndex memory block, and the
  evaluation harness.
