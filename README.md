# TracePack

**Compaction that keeps what you said. After `/compact`, the exact lines come back.**

[![License: MIT](https://img.shields.io/badge/license-MIT-blue.svg)](LICENSE)
![Python 3.9+](https://img.shields.io/badge/python-3.9%2B-blue.svg)
![Claude Code plugin](https://img.shields.io/badge/Claude%20Code-plugin-d97757.svg)
![No dependencies](https://img.shields.io/badge/dependencies-none-brightgreen.svg)

## Results

### Claude Code with TracePack: +43.7 points on LongMemEval

Real Claude Code (2.1.294) answered the same 480 LongMemEval-S questions twice, once as shipped and once
with this plugin. For each question:
- the history (about 50 chat sessions, ~115k tokens) is replayed into a Claude Code session;
- whenever the conversation passes 32k tokens, it is compacted with `/compact` (three times for nearly
  every question);
- the question is then answered from what is left, with no tools.

Every arm used the same model (DeepSeek-V4-Pro, through the built-in bridge) and the same compaction
points. LongMemEval's official prompts graded the answers.

| 480 questions, same model | Accuracy |
|---|---|
| **Claude Code + TracePack** | **75.8%** |
| Claude Code + TracePack, notes only | 70.2% |
| Claude Code + TracePack, your words only | 53.8% |
| Claude Code | 32.1% |
| *Reference: Codex's compaction* | *68.1%* |

- **+43.7 points, 95% CI [+39.0, +48.5].** It wins 219 questions and loses 9, and it is higher on all
  six question types: multi-session 21% → 73%, temporal reasoning 19% → 71%, facts the user stated
  42% → 94%.
- **Both halves pay on their own.** The dated notes add +38.1. Restoring your words adds +21.7.
- **Why stock Claude Code loses so much here.** Its compaction prompt is written for coding: request,
  files, errors, next step. After three compactions most personal facts are gone, and it answered
  "I don't have that information" to 271 of the 480 questions (80 with TracePack). Unlike Codex, it
  keeps none of your messages verbatim.
- **What it costs:** summaries about 45% longer (median 5.5k vs 3.8k tokens), plus about 2.2k tokens
  restored after each compaction.
- **What is still left.** The study's own method, run in the harness with the same model, scores
  81.9%. Its verbatim part is three times larger (6.5k tokens, picked by the model), against the
  plugin's 2k tokens picked locally.

This setting is deliberately hard: a 32k window forces three compactions of a 115k-token history. Claude
Code's real window is 200k, so this happens only in much longer sessions. Two more things differ from
everyday use:
- the model is DeepSeek-V4-Pro, not Claude;
- Claude Code shows the model the machine's date, while each question states its own date.

Reproduce it with `tracepack bench lme run --arms claude-code,claude-code+tracepack`
([how](docs/BENCH_LME.md)).

### Compaction that keeps what can't be re-read: +17.5 points over Codex

When the context fills up, Codex's compaction writes a hand-off summary of the task's progress and
bets that the agent can re-read any detail later. That works for files. It fails for what the user
said, which exists nowhere else: Codex's hand-off summary kept the needed fact in 4 of the 305
cases where it was on screen.

The TracePack study keeps those facts explicitly. A note-taking summary is paired with the user's own
messages, picked by content and kept verbatim, never rewritten. On **LongMemEval-S** (long dialogues,
histories of about 115k tokens), with a 32k window so that each question's history is compacted about
three times:

| 480 questions, same time window, same reader | Accuracy |
|---|---|
| **TracePack's compaction** (note-taking summary + the user's words, kept verbatim) | **86.0%** |
| Codex's compaction (re-implemented from Codex's source) | 68.5% |

- **+17.5 points, 95% CI [+13.3, +21.7].** It is higher on all six question types, and **+19.3 on
  280 fresh questions**.
- **The same budget spent only on a note-taking summary ties it** (+1.0 [−2.3, +4.2]). TracePack
  generates 30% fewer tokens for that result.
- **On SWE-bench Verified (32k)**, putting forgotten file views back after each compaction makes the
  agent hit the step limit **10.8 points less often** than OpenHands' own compaction (−10.8
  [−18.6, −3.0]). The solve rate is tied.

These are pre-registered, paired runs from our research harness. Its LongMemEval part is now in this
repository as `tracepack bench lme`: the prompts are byte-identical to the study's, and fed the study's
verdicts, its report prints this table. Re-run with DeepSeek-V4-Pro as the model, the same method beats
Codex's compaction by +13.7 [+8.8, +18.5]. Details and the other benchmarks are in
[the research notes](docs/RESEARCH.md).

### Recall after compaction: 2.5× more of the right lines than grep

On **CodeMemo**, a public benchmark of 153 questions over 66 real Claude Code sessions (up to 430 MB of
history per project), this is how often the turns that hold the answer come back, verbatim, within
the same 2,000-token budget:

| Within 2,000 tokens | Answer's turn delivered | Exact value delivered |
|---|---|---|
| **TracePack** | **34.6%** | **47%** |
| Grep over the transcript (what Claude does without it) | 13.7% | 17% |
| What's still in context (the newest 2,000 tokens) | 3.9% | 12% |
| *Reference: a BM25 search engine over messages* | *34.0%* | *47%* |

- **2.5× the evidence and 2.8× the exact values of grepping the transcript**, and about 9× what
  is still in context.
- **4.7× on debugging history.** For "how did we diagnose and fix X" questions it is 48%, against 10%
  for grep.
- **As good as a BM25 search engine, and packaged for an agent.** Whole records, each labelled with
  tool, time and transcript line; a call with its short output; a hard budget.
- **No model, no network.** About 0.4 s per question over histories of up to 52,000 records.
- **Works live in Claude Code.** A random key printed before `/compact` came back exactly, in one
  turn, without re-running anything.

The test is model-free and deterministic: the turns CodeMemo marks as each answer's evidence have to come
back inside the budget. Run `tracepack bench codememo --data <CodeMemo folder>` to reproduce it.
TracePack's defaults were chosen on the odd-numbered questions. On the held-out even-numbered half it
ties BM25 on evidence (34.7% vs 34.7%) and leads on exact values (43% vs 40%). Full tables are in
[Benchmark](#benchmark).

When Claude Code compacts a long session, the summary keeps the plan and drops the specifics: the
exact error text, the path it created, the number a benchmark printed, the port you mentioned two
hours ago. TracePack hands those back **verbatim**, inside a token budget, and labels each one with
where it came from.

It reads the transcript Claude Code already keeps on disk. It makes no model calls of its own: the
notes are written by Claude Code's own compaction. It needs no API key, runs no background process, and
nothing leaves your machine.

## Install

In Claude Code:

```
/plugin marketplace add oklen/TracePack
/plugin install tracepack@tracepack
```

Then restart Claude Code, or run `/reload-plugins`. Check that it works with `/tracepack:status`.

It needs Python 3.9 or newer on your `PATH` as `python3`. macOS and most Linux distributions already
have it. Nothing else gets installed.

## Try it in 60 seconds

1. Ask Claude to run this, then reply "done":
   `python3 -c "import secrets; [print('key%02d' % i, secrets.token_hex(8)) for i in range(30)]"`
2. Run `/compact`.
3. Ask: *"What was key17? Don't run anything."*

Random keys cannot be regenerated, so only the original output has the answer. Right after the
compaction you see *TracePack restored N exact records*, and `/tracepack:last` shows exactly what was
put back. Claude then answers from the restored output in one turn, without re-running anything.

In a session this short, Claude Code's own summary may keep the keys as well. The difference shows in
long sessions, where the summary has to drop most outputs. There, Claude would otherwise re-run the
command (and get different keys) or grep its own transcript file.

## What you get

- **Claude Code's compaction keeps notes.** Every compaction, manual or automatic, also writes a
  dated "Notes" section: one line per fact you stated (requirements, constraints, preferences, names,
  numbers, paths, versions), per answer Claude gave you, and per change, with the old and new value.
  Values are copied as stated, and notes carry over from one compaction to the next.
- **Your own words come back after every compaction.** Within ~2,000 tokens, TracePack adds back:
  - your sentences that state facts, picked by content from everything you said, verbatim and dated;
  - the latest tool outputs;
  - the records your `/compact` instructions and recent requests point at.

  They come in the order they happened. Each is labelled, e.g. `Bash output of \`pytest -q\` · 10-08 13:14 · L22`.
- **A `recall` tool** for anything else from before the compaction. Claude asks in plain words
  (*"exact error from the last pytest run"*, *"the DB port the user gave"*) and gets verbatim records
  under a budget, each labelled with the call that produced it.
- **An `expand` tool** for the full text of one long record, page by page.
- **Commands:**
  - `/tracepack:recall <question>`
  - `/tracepack:status`
  - `/tracepack:last`, which shows exactly what was added after the last compaction.

This is what Claude sees after a compaction (the built-in demo session; run `tracepack demo`):

```
TracePack: the user's own words from before the compaction, verbatim. These sentences were picked
because they state facts (names, numbers, dates, preferences, decisions, constraints); "…" marks
where a message was shortened. The summary above may have paraphrased them.

[u1] user said · 2026-10-08 17:00 · L1
… Profile it and keep the numbers; the staging DB is at port 6543.

[u2] user said · 2026-10-08 17:10 · L11
… Also the migration id we must not touch is 2026_09_30_add_ledger_index.

TracePack: 7 verbatim records from this session before the compaction, in the order they happened.
The summary above may have shortened them. Other records can be looked up with the TracePack recall tool.

[2] Bash output of `python bench.py --rows 200000` · 10-08 17:02 · L3
rows=200000 elapsed=41.7s p95_latency=812ms
hot path: parse_rows 63% of time
...
[5] Edit call · 10-08 17:06 · L7
Edit {"file_path": "/work/demo/bench/parse.py", "new_string": "for line in buf.splitlines():", ...}
...
[7] Bash output of `python bench.py --rows 200000` · 10-08 17:09 · L10
rows=200000 elapsed=29.3s p95_latency=530ms
hot path: parse_rows 41% of time
```

## How it works

- **It reads, it doesn't record.** Claude Code already writes every message and tool output to
  `~/.claude/projects/<project>/<session>.jsonl`. TracePack only reads that file.
- **It knows what Claude can no longer see.**
  - It pairs each tool call with its output by id.
  - It recognises compactions, including the messages a compaction keeps verbatim.
  - It searches only what dropped out of context, so it doesn't spend the budget repeating what
    Claude already has.
- **It ranks without a model.** It scores records against the question with BM25.
- **It fills the budget with whole records, best first.**
  - A tool output is labelled with the call that produced it.
  - A call carries its short output.
  - If a top record is too large for what's left, a labelled excerpt of its matching lines takes
    its place; nothing is cut silently.
  - The records are then shown in the order they happened.
- **It masks secrets.** Strings that look like credentials (API keys, tokens, passwords) are masked
  in everything it hands back.
- **It picks your words without a model.** Each sentence you wrote is scored for facts: numbers,
  dates, names, identifiers, first-person statements, preferences, plans, decisions, constraints such
  as "never" or "must". Questions, thanks and pasted material score low. The sentences with the most
  facts per token come back, so the one fact inside a long message can return without the rest.
- **Two hooks, both cheap:**
  - PreCompact notes your `/compact` instructions and prints the note-taking instructions. Claude
    Code appends whatever a PreCompact hook prints to its own compaction prompt, after your
    `/compact` instructions.
  - SessionStart, right after a compaction, adds your words and the records back.

  There are no per-tool-call hooks, no daemon and no ports. Claude Code writes the compaction
  summary only after these hooks run. So TracePack builds its query from your `/compact`
  instructions and the latest work, and it pins the latest tool outputs first.

## When it helps, and when it doesn't

It helps with details you can't get back cheaply:
- values printed once: random ids, timings, benchmark numbers;
- output of slow, remote or side-effecting commands;
- things you told Claude in chat;
- earlier versions of a file that has since changed.

If a detail can be re-read cheaply (the file is still there), Claude re-reading it works just as
well. Without TracePack, Claude can sometimes still dig a value out of its raw transcript file with
grep, if the summary points it there. That costs an extra tool call over JSON lines.

Our research measurements found that most details lost in compaction were cheap to re-read. On
SWE-bench, adding the packet did not change the resolve rate. TracePack is for the cases where
re-reading isn't an option; see [the research notes](docs/RESEARCH.md).

## Privacy and cost

- **Local only.** It reads files on your machine and writes a few small JSON files to the plugin's
  data folder (`~/.claude/plugins/data/tracepack…`). It makes no network calls, collects no
  telemetry and calls no model.
- **Cost.** The only cost is the tokens it adds: the notes in each compaction summary, and at most
  ~2,000 tokens after each compaction. Nothing otherwise. Tool outputs from `recall` are sized by
  the budget Claude asks for.
- **Your files stay as they are.** It never writes into your repository or your settings.
- **Transcripts expire.** They belong to Claude Code, which deletes them after `cleanupPeriodDays`
  (30 days by default). TracePack can recall only what still exists.

## Settings

The defaults need no setup. To change them, set environment variables, for example in
`~/.claude/settings.json` under `"env"`:

| Variable | Default | Effect |
|---|---|---|
| `TRACEPACK_NOTES` | `1` | Ask each compaction to keep dated notes. `0` turns this off. |
| `TRACEPACK_INJECT_BUDGET` | `2000` | Tokens added after each compaction (8,000 characters at most). `0` turns this off. |
| `TRACEPACK_USER_WORDS` | `1` | Include your own sentences in what is added. `0` adds only records. |
| `TRACEPACK_RECALL_BUDGET` | `2000` | Default size of a `recall` answer. |
| `TRACEPACK_MAX_MB` | `64` | For a very large transcript, read only its newest part. |
| `TRACEPACK_REDACT` | `1` | Mask credential-like strings. `0` turns masking off. |

## Command line

The same engine is available outside Claude Code:

```bash
uv tool install git+https://github.com/oklen/TracePack     # or: pipx install git+https://github.com/oklen/TracePack
tracepack recall "exact error from the last pytest run"     # newest session of the current project
tracepack recall "the port I gave you" --session <id or path>
tracepack expand 417                                        # one record in full
tracepack status | last | sessions | demo | doctor
tracepack bench codememo --data <CodeMemo folder>          # the recall benchmark (no model)
tracepack bench lme run --arms tracepack,codex --out runs   # LongMemEval with compaction (needs a model)
```

## Other agents (MCP)

`tracepack mcp` is a standard stdio MCP server with no dependencies, exposing `recall`, `expand` and
`status`. Any MCP client can use it:

```json
{ "mcpServers": { "tracepack": { "command": "tracepack", "args": ["mcp"] } } }
```

The Python API also reads pi sessions and OpenHands traces (`tracepack.pack`). It ships an
OpenHands condenser and a LlamaIndex memory block; see [the library reference](docs/RESEARCH.md).

## Benchmark

[CodeMemo](https://github.com/laynepenney/codememo-benchmark) (MIT) has 158 questions over 66 Claude Code
sessions from three software projects; 153 of them have evidence turns that can be located. Each
project's sessions are joined in order into one long history, as if it were one session compacted many
times. For every question and budget, a method hands back records, and we check two things:
- **Answer's turn delivered:** at least one of the turns CodeMemo lists as the answer's evidence is among them.
- **Exact value delivered** (in parentheses): the specific value in the short answer appears verbatim.
  This counts only the questions whose short answer contains a number, version, path or identifier.

All methods count tokens the same way (about 4 characters per token).

| Method | 1,000 tokens | 2,000 tokens | 4,000 tokens |
|---|---|---|---|
| **TracePack** | 26.1% (37%) | 34.6% (47%) | 39.2% (56%) |
| BM25 over messages | 29.4% (36%) | 34.0% (47%) | 40.5% (55%) |
| grep over the transcript | 10.5% (12%) | 13.7% (17%) | 17.0% (21%) |
| newest context only | 3.3% (8%) | 3.9% (12%) | 5.2% (16%) |

- **TracePack** is `recall` with its defaults: BM25 ranking, whole records in rank order, labels, a
  call's short output folded in, and excerpts for top records that don't fit.
- **BM25** is the same ranking without labels or folding: the strongest simple baseline.
- **grep** reads the raw JSON lines that contain the question's two most specific words, in file
  order, each cut to 2,000 characters.
- **Newest context** is the last records that fit.

By question category, at 2,000 tokens (answer's turn delivered):

| Category | TracePack | BM25 | grep | newest context |
|---|---|---|---|---|
| Factual | 37% | 37% | 17% | 6% |
| Debug | 48% | 45% | 10% | 0% |
| Architecture | 32% | 32% | 7% | 4% |
| Temporal | 24% | 24% | 14% | 5% |
| Convention | 35% | 35% | 20% | 5% |
| Cross-session | 25% | 25% | 15% | 5% |

The defaults (BM25 rather than BM25 plus a hashing embedding, no dependency closure, rank-order packing)
were chosen on the odd-numbered questions and checked on the even-numbered ones, at 2,000 tokens:

| Half | Method | Answer's turn | Exact value |
|---|---|---|---|
| odd (n=78) | tracepack | 34.6% | 51% |
| odd (n=78) | bm25 | 33.3% | 53% |
| even (n=75) | tracepack | 34.7% | 43% |
| even (n=75) | bm25 | 34.7% | 40% |

Two things this does not measure:
- **Answer accuracy.** That also depends on the model reading the records.
- **The automatic restore after `/compact`.** That uses the same packer with the latest tool outputs
  pinned first. The LongMemEval result above measures the restore and the notes end to end.

The earlier research measurements, including where dependency closure did and did not pay, are in
[the research notes](docs/RESEARCH.md).

### LongMemEval with compaction

`tracepack bench lme` is the harness behind the compaction results above. Each LongMemEval-S
history (about 50 sessions, ~115k tokens) is streamed into a 32k context, compacted by the arm whenever
it fills up, and the question is asked once at the end, with no tools; LongMemEval's official prompts
grade the answer. The arms:

- `tracepack`: the study's method (a note-taking summary plus the user's own messages, picked by content
  and kept verbatim).
- `codex`: Codex CLI's compaction, re-implemented from its source.
- `notes`: the note-taking summary alone, with the same total budget.
- `full`: no compaction.
- `claude-code` and `claude-code+tracepack`: real Claude Code, without and with this plugin, plus two
  ablations.

Prompts, budgets and statistics are the study's, and the tests check the prompts byte for byte. Any
OpenAI-compatible model works:

```bash
pip install tiktoken
export TRACEPACK_LLM_BASE_URL=… TRACEPACK_LLM_API_KEY=… TRACEPACK_LLM_MODEL=… TRACEPACK_JUDGE_MODEL=…
tracepack bench lme run --arms tracepack,codex --out runs/lme      # resumable; --dry-run makes no calls
tracepack bench lme report --out runs/lme
```

Roughly 150k input tokens per question per arm. Setup, the Claude Code arms and the output format are
described in [docs/BENCH_LME.md](docs/BENCH_LME.md).

## Troubleshooting

- **Run `/tracepack:status` or `tracepack doctor`.** The doctor checks Python, finds your
  transcripts, and runs the whole path on a built-in example session.
- **`python3: command not found`:** install Python 3.9 or newer.
- **Nothing restored after `/compact`:** either nothing before the compaction qualified (no tool
  output and no sentence of yours with a fact), or `TRACEPACK_INJECT_BUDGET` is `0`.
- **The note-taking text shows up after a manual `/compact`:** Claude Code prints every PreCompact
  hook's output in the `/compact` result. `TRACEPACK_NOTES=0` turns the notes off.
- **Hook errors:** they never interrupt Claude. They are logged to `hook_errors.log` in the data
  folder, and `tracepack doctor` reports them.

## Uninstall

```
/plugin uninstall tracepack@tracepack
/plugin marketplace remove tracepack
```

Uninstalling removes the plugin's data folder. Your transcripts are not touched.

## Development

```bash
python3 tracepack/tests/run_all.py                 # 310 tests, contract by contract
claude plugin validate . --strict                  # manifest, marketplace, hooks, MCP config
claude --plugin-dir .                              # try the working copy in Claude Code
```

Releases are listed in the [CHANGELOG](CHANGELOG.md).

## License

MIT © 2026 Zefeng Cai — see [LICENSE](LICENSE).
