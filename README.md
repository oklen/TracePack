# TracePack

**After `/compact`, get the exact lines back.**

[![License: MIT](https://img.shields.io/badge/license-MIT-blue.svg)](LICENSE)
![Python 3.9+](https://img.shields.io/badge/python-3.9%2B-blue.svg)
![Claude Code plugin](https://img.shields.io/badge/Claude%20Code-plugin-d97757.svg)
![No dependencies](https://img.shields.io/badge/dependencies-none-brightgreen.svg)

Zefeng Cai · independent researcher

When Claude Code compacts a long session, the summary keeps the plan and drops the specifics: the
exact error text, the path it created, the number a benchmark printed, the port you mentioned two
hours ago. TracePack hands those back **verbatim**, inside a token budget, and labels each one with
where it came from.

It reads the transcript Claude Code already keeps on disk. It makes no model calls and needs no API
key. There is no background process, and nothing leaves your machine.

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

- **Records come back automatically after every compaction.** Within ~1,500 tokens, TracePack adds
  back:
  - the latest tool outputs;
  - the records your `/compact` instructions and recent requests point at.

  They come in the order they happened. Each is labelled `Bash output · 2026-10-08 13:14 · line 22`.
- **A `recall` tool** for anything else from before the compaction. Claude asks in plain words
  (*"exact error from the last pytest run"*, *"the DB port the user gave"*) and gets verbatim records
  under a budget. A tool call always comes with its output.
- **An `expand` tool** for the full text of one long record, page by page.
- **Commands:**
  - `/tracepack:recall <question>`
  - `/tracepack:status`
  - `/tracepack:last`, which shows exactly what was added after the last compaction.

This is what Claude sees after a compaction (the built-in demo session; run `tracepack demo`):

```
TracePack: 12 verbatim records from this session before the compaction, in the order they happened.
The summary above may have shortened them. Other records can be looked up with the TracePack recall tool.

[3] Bash call · 2026-10-08 17:01 · line 2
$ python bench.py --rows 200000

[4] Bash output · 2026-10-08 17:02 · line 3
rows=200000 elapsed=41.7s p95_latency=812ms
hot path: parse_rows 63% of time
...
[11] Bash output · 2026-10-08 17:09 · line 10
rows=200000 elapsed=29.3s p95_latency=530ms
hot path: parse_rows 41% of time

[12] user said · 2026-10-08 17:10 · line 11
Good. Also the migration id we must not touch is 2026_09_30_add_ledger_index.
```

## How it works

- **It reads, it doesn't record.** Claude Code already writes every message and tool output to
  `~/.claude/projects/<project>/<session>.jsonl`. TracePack only reads that file.
- **It knows what Claude can no longer see.**
  - It pairs each tool call with its output by id.
  - It recognises compactions, including the messages a compaction keeps verbatim.
  - It searches only what dropped out of context, so it doesn't spend the budget repeating what
    Claude already has.
- **It ranks without a model.** It scores records against the question with BM25 plus a hashing
  embedding.
- **It packs whole records.** It follows dependency links (a call to its output, a value to where
  it came from) and packs whole records into a hard token budget. A record too large for the budget
  is excerpted and labelled as such, never cut silently.
- **It masks secrets.** Strings that look like credentials (API keys, tokens, passwords) are masked
  in everything it hands back.
- **Two hooks, both cheap:**
  - PreCompact notes your `/compact` instructions.
  - SessionStart, right after a compaction, adds the records back.

  There are no per-tool-call hooks, no daemon and no ports. Claude Code writes the compaction
  summary only after these hooks run. So TracePack builds its query from your `/compact`
  instructions and the latest work, and it always includes the latest tool outputs.

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
- **Cost.** The only cost is the tokens it adds: at most ~1,500 per compaction by default, and none
  otherwise. Tool outputs from `recall` are sized by the budget Claude asks for.
- **Your files stay as they are.** It never writes into your repository or your settings.
- **Transcripts expire.** They belong to Claude Code, which deletes them after `cleanupPeriodDays`
  (30 days by default). TracePack can recall only what still exists.

## Settings

The defaults need no setup. To change them, set environment variables, for example in
`~/.claude/settings.json` under `"env"`:

| Variable | Default | Effect |
|---|---|---|
| `TRACEPACK_INJECT_BUDGET` | `1500` | Tokens added after each compaction. `0` turns this off. |
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
```

## Other agents (MCP)

`tracepack mcp` is a standard stdio MCP server with no dependencies, exposing `recall`, `expand` and
`status`. Any MCP client can use it:

```json
{ "mcpServers": { "tracepack": { "command": "tracepack", "args": ["mcp"] } } }
```

The Python API also reads pi sessions and OpenHands traces (`tracepack.pack`). It ships an
OpenHands condenser and a LlamaIndex memory block; see [the library reference](docs/RESEARCH.md).

## Troubleshooting

- **Run `/tracepack:status` or `tracepack doctor`.** The doctor checks Python, finds your
  transcripts, and runs the whole path on a built-in example session.
- **`python3: command not found`:** install Python 3.9 or newer.
- **Nothing restored after `/compact`:** either there was no tool output before the compaction, or
  `TRACEPACK_INJECT_BUDGET` is `0`.
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
python3 tracepack/tests/run_all.py                 # 293 tests, contract by contract
claude plugin validate . --strict                  # manifest, marketplace, hooks, MCP config
claude --plugin-dir .                              # try the working copy in Claude Code
```

Releases are listed in the [CHANGELOG](CHANGELOG.md).

## License

MIT © 2026 Zefeng Cai — see [LICENSE](LICENSE).
