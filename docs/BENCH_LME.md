# LongMemEval with compaction: `tracepack bench lme`

This is the harness behind the compaction numbers in the README. Each LongMemEval-S question comes with a
chat history of about 50 sessions (≈115k tokens). The history is streamed into a context window session
by session, oldest first. Whenever the context passes the window (32,768 tokens by default), the arm
compacts it. When the whole history is in, the question is asked once, with no tools. A judge model grades
the answer with LongMemEval's official per-type prompts.

## Arms

| Arm | What it keeps when the context fills up |
|---|---|
| `tracepack` | A note-taking memory summary (capped at 0.15 × window). Next to it, the user's own messages: a model picks them by number, and the harness copies them verbatim and append-only (0.2 × window, oldest dropped first). |
| `codex` | Codex CLI's local compaction, re-implemented from its source. Its hand-off summary prompt and prefix, then the most recent user messages verbatim, up to min(20,000, 0.3 × window) tokens, counted as Codex counts them (UTF-8 bytes / 4). |
| `notes` | The same note-taking summary with the whole memory budget (0.35 × window), and nothing verbatim. |
| `full` | No compaction: the whole history. This needs a model with a ~128k window. |
| `claude-code` | Real Claude Code. The history is written as a Claude Code transcript, compacted with `claude -p /compact`, and the question is asked with `claude -p … --tools ""`. |
| `claude-code+tracepack` | The same, with this repository loaded as a plugin (`--plugin-dir`): note-taking compaction plus your words restored. |
| `claude-code+notes`, `claude-code+restore` | The plugin with one half switched off (`TRACEPACK_INJECT_BUDGET=0` or `TRACEPACK_NOTES=0`). |

The harness arms use the study's code paths unchanged:
- the same prompts, byte for byte (`tracepack/tests/test_bench_lme.py` checks their SHA-256 against the
  study's files);
- the same temperatures: 0.7 for summaries, 0 for the first pick and for the answer;
- the same budgets and the same rule for rejecting and retrying a summary;
- the same token estimate (cl100k, calibrated on 21 full-history calls).

Fed the study's own verdicts, `report` prints the study's table: tracepack 86.0%, codex 68.5%,
+17.5 [+13.3, +21.7].

## Run it

```bash
pip install tiktoken                               # token counts (cl100k), as in the study
tracepack bench lme download                       # LongMemEval-S cleaned, 277 MB, sha256-checked

export TRACEPACK_LLM_BASE_URL=https://…/v1         # any OpenAI-compatible /chat/completions
export TRACEPACK_LLM_API_KEY=…
export TRACEPACK_LLM_MODEL=…                       # reader, summarizer and picker
export TRACEPACK_JUDGE_MODEL=…                     # judge (base URL and key fall back to the LLM ones)

tracepack bench lme run --arms tracepack,codex --out runs/lme --workers 32
tracepack bench lme report --out runs/lme
```

- **It is resumable.** Each finished question is one JSON file under `runs/lme/units/`; a rerun skips it.
- **Partial runs:** `--limit 20` runs the first 20 questions. `--ids first200` runs the study's first batch
  (a stratified sample), and `--ids all` runs all 500.
- **A dry run makes no model calls:** `--dry-run --limit 3` uses a stub model and checks the whole
  pipeline offline.
- **Cost:** each compacting arm makes about 3 summaries of a ~33k-token context per question, plus the
  pick calls for `tracepack` and one answer. That is roughly 150k input tokens per question per arm, or
  about 70M for 480 questions. Output per question in the study: about 4k tokens for `codex`, 32k for
  `tracepack` and 46k for `notes`.
- **Rate limits:** retries back off from 5 s to 60 s. `TRACEPACK_LLM_MAX_INFLIGHT` caps the requests open
  at once.

## The Claude Code arms

```bash
tracepack bench lme run --arms claude-code,claude-code+tracepack --out runs/lme --workers 32 \
    --claude $(which claude)
```

- **Fully isolated.** Each question gets its own `CLAUDE_CONFIG_DIR` and `HOME`, so your settings, hooks,
  plugins and MCP servers are never loaded. Claude Code's own transcript rows are kept under
  `runs/lme/cc/<arm>/<question>/`.
- **Same model as the other arms.** By default Claude Code talks to a small built-in bridge
  (`tracepack.bench.anthropic_bridge`), which serves the Anthropic Messages API from your
  `TRACEPACK_LLM_*` model. The bridge keeps the text and the message roles. It drops Claude Code's tool
  definitions and thinking settings, so the model answers in text. To run against Anthropic instead,
  pass `--anthropic-base-url https://api.anthropic.com` with `ANTHROPIC_API_KEY` set, and optionally
  `TRACEPACK_CLAUDE_MODEL`.
- **Compaction runs as `/compact`.** It happens at exactly the points where the other arms compact: the
  conversation is counted the same way, without Claude Code's own system prompt. A manual `/compact`
  prints the PreCompact output into the conversation, so in the plugin arms the note-taking
  instructions are visible after the first compaction.
- **Dates differ.** Claude Code tells the model the machine's date, and LongMemEval states each question's
  own date in the question. Every Claude Code arm sees the same mismatch. Shifting Claude Code's clock back
  by years makes it spin on several CPU cores, so the harness does not do it.

## Output

`report` prints three things:
- accuracy per arm;
- paired differences with a bootstrap over questions (4,000 resamples, seed 20261004; an ungraded unit
  counts as missing for its own arm only);
- accuracy by question type.

`--json` gives the same data as JSON.

Each unit file holds the full record:
- for the harness arms, every model call (purpose, attempt, prompt and completion tokens) and every
  compaction, with the summary and the verbatim block;
- for the Claude Code arms, Claude Code's summary and TracePack's restore at each compaction, and the
  final context the model answered from;
- for every arm, the judge's verdict.
