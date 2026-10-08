# Changelog

## 0.3.0 — 2026-10-08

Recall is now measured on a public benchmark, and tuned so that it matches a BM25 search engine.

- **Benchmark.** `tracepack bench codememo` runs a model-free evidence benchmark: CodeMemo, 153 questions
  over 66 real Claude Code sessions. At 2,000 tokens, TracePack delivers the answer's turn for 34.6% of
  questions and the exact value for 47%. Grep over the transcript gets 13.7% / 17%; the newest
  context gets 3.9% / 12%. See the README.
- **New defaults**, chosen on the odd-numbered questions and checked on the even-numbered ones:
  - BM25 ranking; the BM25 + hashing hybrid was worse.
  - No dependency closure; it cost evidence.
  - Records packed whole, best first, until the budget is full, with a labelled excerpt for a top
    record that doesn't fit. Before this, 8 records were ranked regardless of budget, so a larger
    budget changed nothing.
- **Records are not served twice.** A call carries its short output instead of repeating it, and a
  tool output's label names its call.
- **The budget holds for the whole answer.** Labels, header and footer now count against it, and all
  token counts use one estimate (about 4 characters per token).
- **Repeated recalls are about 4× faster.** The search index is kept between queries.
- **Shorter labels:** `Bash output of \`pytest -q\` · 10-08 13:14 · L22`.
- **README leads with the results.** First the compaction study's comparison with Codex (LongMemEval-S, 32k: +17.5 points;
  research harness, not in this repository; table in `docs/RESEARCH.md`), then the CodeMemo recall benchmark.

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
