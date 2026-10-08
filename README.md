# TracePack

Turn an agent's execution trace into a budgeted, auditable evidence packet.

Zefeng Cai · independent researcher

When a coding agent's context is compacted, the summary keeps the plan and drops the details: the
exact path, number or tool output that a later step needs. TracePack keeps the trace, and when a
question or instruction arrives it serves back a **verbatim** packet of the relevant events under a
hard token budget.

```
trace ──▶ IR: events + typed edges ──▶ router: seeds ──▶ typed closure ──▶ hard-budget assembler ──▶ packet
```

- **IR.** An adapter turns a Claude Code transcript, a pi session or OpenHands message rows into
  events joined by typed edges: `RESULT_OF` (a tool call and its result), `DEPENDS_ON` (a later event
  reuses a literal from an earlier one), `MATERIALIZES` (a compaction summary and the events it
  replaced), `SUPERSEDES` (a newer value replaces an older one), `CONTROL` and `TEMPORAL`.
- **Router.** Picks seed events for the query: `last_n`, `lexical` (BM25), `dense` (offline keyed
  hashing, no model), `hybrid`, or `hybrid_pin`. The evaluation adds an `oracle` arm for ceilings.
- **Closure.** Follows strong edges from the seeds to what they depend on: `off`, `native`,
  `full_ancestor`, or `oracle` for ceilings.
- **Assembler.** Packs under a hard budget. It never cuts an event in half, keeps a tool call with
  its result, and when the budget cannot hold what the closure requires it returns
  `incomplete=True` with the missing events listed instead of dropping them silently.
- **Profiler.** Counterfactual arms that attribute a wrong answer to routing, closure, packing or
  reading.

The core path is stdlib-only and deterministic: the same trace, query and budget give the same
packet bytes in every process.

## Install

```bash
pip install .                    # Python >= 3.9; the core has no third-party dependencies
pip install ".[bm25]"            # rank_bm25 for the lexical router (a built-in BM25 is used otherwise)
pip install ".[openhands]"       # the OpenHands condenser (Python >= 3.12)
pip install ".[llama-index]"     # the LlamaIndex memory block (Python >= 3.10)
pip install ".[eval]"            # the evaluation reader and statistics (torch, transformers, numpy)
```

## Quickstart

```python
import tracepack

trace = [                                   # OpenAI-style rows, or a path to a .jsonl export
    {"tracepack_format": "openhands"},
    {"id": "m0", "role": "user", "content": "the tests fail with a TypeError in units"},
    {"id": "m1", "role": "assistant", "content": "look",
     "tool_calls": [{"id": "c1", "type": "function",
                     "function": {"name": "bash", "arguments": '{"command": "pytest -q"}'}}]},
    {"id": "m2", "role": "tool", "content": "E   TypeError: unsupported operand for Quantity"},
]
pkt = tracepack.pack(trace, query="why does the Quantity comparison fail?", budget=2048)
print(pkt["text"])                          # the packet, ready to append after a summary
print(pkt["tokens"], pkt["n_entries"], pkt["incomplete"])
```

```bash
python3 -m tracepack.examples.quickstart                         # a synthetic trace, two budgets
python3 tracepack/examples/why_end_to_end.py <transcript.jsonl>  # every stage on one real transcript
```

## Trace formats

The registry picks the adapter from the first line of the file, so all three formats can sit in
`.jsonl` files side by side.

| first line | format | adapter |
|---|---|---|
| `{"type": "session", "version": ...}` | pi coding-agent session | `tracepack/adapters/pi.py` |
| `{"tracepack_format": "openhands", ...}` | OpenHands / OpenAI-style message rows | `tracepack/adapters/openhands.py` |
| anything else | Claude Code transcript (`~/.claude/projects/<project>/<session>.jsonl`) | `tracepack/adapters/claude_code.py` |

## Integrations

- **OpenHands.** `tracepack.make_condenser(llm, max_tokens, ...)` returns the stock summarizing
  condenser with a packet appended after the summary. `tracepack/condenser/` is the same recall
  step as a drop-in condenser; its [README](tracepack/condenser/README.md) lists every default
  together with the measurement that set it.
- **LlamaIndex.** `tracepack/adapters/llama_index_memory.py` serves a trace as a memory block that
  takes runtime arguments (budget, k, query mode) and re-assembles at a smaller budget instead of
  being deleted under pressure: `python3 -m tracepack.examples.llama_index_integration <transcript.jsonl>`.
- **HTTP.** `python3 -m tracepack.pilot.tp_serve --port 8765` serves `POST /evidence` packets for
  a live pi session.

## Evaluate on your own traces

The harness builds its questions from your own sessions. Every question asks for a literal value
(a path, number, identifier or version) that the answer depends on, in five slices defined by
dependency shape: `direct_fact`, `explicit_ref`, `decision_source`, `tool_chain`, `correction_stale`.

Model calls go through one runner, `tracepack/eval/llm_runner.py`, which talks to any
OpenAI-compatible endpoint configured by environment variables (no defaults):

```bash
export TRACEPACK_LLM_BASE_URL=https://...   TRACEPACK_LLM_API_KEY=...   TRACEPACK_LLM_MODEL=...
export TRACEPACK_READER_MODEL=/path/to/Qwen3-8B      # local Hugging Face checkpoint for the reader
```

```bash
# 1. items from your Claude Code sessions, frozen so later stages read identical traces
python3 tracepack/eval/datasets.py --transcripts "$HOME/.claude/projects/*/*.jsonl" --out data/items.jsonl
python3 tracepack/eval/freeze_corpus.py --items data/items.jsonl --out data/items_frozen.jsonl --dir data/traces

# 2. one question per item, written by an LLM and screened for answer leaks
python3 tracepack/eval/gen_queries.py --items data/items_frozen.jsonl --out data/items_q.jsonl
python3 tracepack/eval/audit_dataset.py data/items_q.jsonl         # must end with AUDIT_OK

# 3. packets for every arm (CPU, where the traces are), then the reader (GPU)
python3 tracepack/eval/run_eval.py --emit --items data/items_q.jsonl --out data/packets.jsonl
python3 tracepack/eval/read_gen.py --packets data/packets.jsonl --items data/items_q.jsonl --out data/gen.jsonl

# 4. audit the judge, judge the answers, then the tables and the pre-registered gates
python3 tracepack/eval/judge_answers.py --audit data/gen.jsonl --out data/audit_pack.jsonl
python3 -m tracepack.eval.llm_runner --pack data/audit_pack.jsonl --out data/audit_res.jsonl
python3 tracepack/eval/judge_answers.py --audit-score data/audit_pack.jsonl --verdicts data/audit_res.jsonl --out data/audit.json
python3 tracepack/eval/judge_answers.py --dump data/gen.jsonl --out data/judge_pack.jsonl
python3 -m tracepack.eval.llm_runner --pack data/judge_pack.jsonl --out data/judge_res.jsonl
python3 tracepack/eval/judge_answers.py --merge data/gen.jsonl --verdicts data/judge_res.jsonl --out data/eval.jsonl
python3 tracepack/eval/analyze.py data/eval.jsonl
```

`infer_edges.py`, `edge_arms.py` and `run_edges.py` repeat the experiment with LLM-inferred
dependency edges against random edges of the same count and distance. `stats.py` holds the
session-clustered statistics (wild cluster bootstrap) used for every comparison.

## Tests

```bash
python3 tracepack/tests/run_all.py     # 276 tests, reported contract by contract; or: pytest
```

The contracts include: identical inputs give an identical packet hash; the budget is never
exceeded; a satisfiable closure leaves no dangling required event; an unsatisfiable one is declared
`incomplete`; a tool call is never split from its result; a superseded value is never served as the
current one.

## What we measured

These numbers come from our own Claude Code sessions: 10 long sessions and 190 questions; free-form
answers from Qwen3-8B and Qwen3-32B, graded by GPT-5.6-Sol after a 280/280 self-audit, with
statistics clustered by session. **These sessions are private: neither they nor anything derived
from them (questions, packets, answers, verdicts) is released.** The harness above builds the same
kind of dataset from your own sessions.

- **Picking the right seeds matters most.** With an exact tokenizer, perfect seeds beat the default
  router by 0.421 in accuracy at a 2,048-token budget and by 0.305 at 8,192 (Qwen3-8B; 0.342 and
  0.242 with Qwen3-32B).
- **Bringing dependencies along does not pay as implemented, but could.** The shipped closure costs
  accuracy at the default configuration (2,048 tokens: −5.3 pp with 8B, −4.7 pp with 32B). Adding
  each question's required source events to the closure and ranking them first is worth +27 pp with
  lexical seeds and +38 pp with the default seeds, the same order as perfect selection. The loss
  comes from three places: literal edges rarely reach the needed source, the required set is not
  ranked, and closure material pushes seeds out of the packet.
- **Better edges help only at the margin.** LLM-inferred edges beat random edges of the same count
  and distance by a margin that shrinks as the reader gets more accurate: +5.3 pp (two-choice) →
  +4.2 (free-form, 8B) → +3.7 (exact tokenizer) → +2.6 pp with Qwen3-32B (p = 0.125). All of it comes
  from 8 of the 190 questions, where only the LLM edges delivered the answer.
- **The edge effect depends on the trace format.** In a second round over 40 sessions (ours plus
  public pi and OpenHands sessions, judged by DeepSeek-V4-Pro), closure over LLM-inferred edges
  against no closure (lexical seeds) gains +5.8 to +9.6 pp on our Claude Code sessions, significant
  in all four reader × budget cells, and has no significant effect in any cell on the pi or
  OpenHands sessions.
- **End to end**, on SWE-bench Verified with a 27B executor, appending the packet to OpenHands'
  summarizing condenser produced no detectable gain in resolve rate. Two configuration-identical
  runs disagree on 11.7% of instances, more than any effect we measured.

Use it where a needed fact cannot be re-read (it was in a tool output that is gone, or only in the
conversation); it is not a general accuracy gain, and this package does not claim one.

**Not in this release:** our sessions and every file derived from them; the SWE-bench, BeyondSWE,
LOCA and LongMemEval experiment harnesses, which depend on infrastructure we cannot publish; and our
experiment notes. Comments in the code cite those notes (`RESULTS_*.md`, `PLAN_*.md`) as the
provenance of a default or a test.

## License

MIT — see [LICENSE](LICENSE).
