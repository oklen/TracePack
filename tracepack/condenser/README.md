# tracepack.condenser

A recall step for coding agents whose context compaction loses values.

When the harness summarizes and drops events, this keeps the dropped events, and when the next
instruction arrives it puts a budgeted **verbatim** packet of the relevant ones back into the view —
selected by walking the trace's dependency edges, not by text similarity.

```python
from tracepack.condenser.oh_condenser import TracePackCondenser

agent = agent.model_copy(update={
    "condenser": TracePackCondenser(llm=summary_llm, max_size=120, budget=2048),
})
```

That is the whole integration. It is a subclass of `LLMSummarizingCondenser`: same summary, same
trigger, same knobs. No model, no network, no index to maintain — the seeding fuses BM25 with an
offline hashing embedding, and the graph is rebuilt from the archive per query.

## What it does, in four steps

| step | what | why this value |
|---|---|---|
| **seed** | `hybrid` retrieval (BM25 ⊕ offline hashing vectors), k=8 | `RESULTS_seeding.md` §9 — **+38.7 / +38.9** read-out points over lexical seeding on hop ≥ 2 with a tail, p = .016, two readers 0.2 apart |
| **gate** | run the closure only if a seed reaches **≥ 2** dependency hops | `REPORT_abstract.md` §3 — closure is worth **+60** when it does and **0** when it doesn't. A shut gate drops the *closure*, not the packet |
| **close** | typed closure, `max_hops = 12` | `RESULTS_ablation.md` §10 — hop-4 reachability 0/12/38/38% at max_hops 6/8/10/12. Rule: `max_hops ≥ 2n+2` |
| **pack** | `evidence_first` under a hard budget, **cost in the sort key**, per-unit ceiling 0.8, labelled excerpts | `RESULTS_hops.md` §7 — without these the packet gets *worse* as the budget grows: on this corpus the record is lost in **90%** of budget sweeps |

Every default in `recipe.py` carries the reading that fixes it, and a field without one raises
`SchemaError`. That is enforced, not documented: `tests/test_condenser.py` fails if you add a knob
without a citation. The reason is in `RESULTS_ablation.md` §24 — three separate defects in this
project were the same shape, *a default nobody had measured*.

## The packet is built at query time

Condensation **archives**; the instruction **retrieves**. The measured effect comes from retrieving
with the user's instruction as the query, and at condensation time there is no query. A query-free
condense-time packet has never been measured here, so it is not the default.

## Where this does **not** help

Stated up front because it is part of the claim (`REPORT_abstract.md` §3, conclusion three):

- **the harness's summary already keeps values** — then there is nothing to recover, and the packet
  is pure cost. Measured: on a harness with a good summarizer the gain is zero.
- **the workspace persists and the agent will re-read it** — one `ls` beats any retrieval.
- **what you need is one hop from what retrieval already finds** — the gate shuts by itself and you
  get plain verbatim retrieval, which is worth having (+16.7 points over the native summarizer on
  its own) but is not the mechanism this exists for.

The shape it *is* for: the summary dropped a specific value, the source is gone or expensive to
re-derive, and the record you need is reachable only by following what depended on what.

## Cost

| archive size | one recall |
|---|---|
| 100 events | ~6–15 ms |
| 500 events | ~26–62 ms |
| 2,000 events | ~100–240 ms |
| 8,000 events | ~0.4–1.0 s |

Paid once per instruction, not once per step: the packet is cached per query, so the tool loop
inside a turn costs nothing extra. Plus the packet's own tokens (default 2,048) on each LLM call of
that turn.

## Layers

| module | needs | what |
|---|---|---|
| `recipe.py` | python 3.9 | the settled defaults, each citing its reading |
| `core.py` | python 3.9 | `TracePackRecall` — (graph, query) → packet. Harness-agnostic, deterministic |
| `oh_rows.py` | python 3.9 | OpenHands events → adapter rows; also replays an archive offline |
| `oh_condenser.py` | python ≥ 3.12 + `openhands-sdk` | the drop-in condenser |
| `keep.py` | python 3.9 | the keep rule: which forgotten file views to put back ([below](#putting-back-the-forgotten-file-views-oh_keep)) |
| `oh_keep.py` | python ≥ 3.12 + `openhands-sdk` | `KeepCondenser`, the drop-in that puts them back |

Only the `oh_*` modules need the SDK, so the engines and their contracts run anywhere.

## Knobs worth touching

```python
TracePackCondenser(
    llm=summary_llm,
    budget=2048,        # packet ceiling. Bigger is NOT automatically better -- see RESULTS_hops §7
    k=8,                # seeds. k=16 opens the gate more often (68% vs 43% on scripted traces) and
                        # costs budget; that trade is measured at the packet level, not at read-out
    gate_hops=2,        # 0 disables the gate (always close). Not recommended: 0 is what the +60/0
                        # split says is worthless
    enabled=False,      # behave exactly like the stock condenser -- the control arm of an A/B
    archive_path="...", # also write the forgotten events to disk, so a run can be replayed offline
    log_path="...",     # one line per recall: gate state, what was served, latency
)
```

## Auditing a live run

`log_path` gives one JSON line per recall. The three fields worth watching:

- `gate_open` — if it is false most of the time, this deployment is not the shape the mechanism
  needs, and you are paying for verbatim retrieval.
- `n_missing_required` — required evidence the budget could not fit. Declared, never silent
  (contract #1/#4). Persistently non-zero means the budget is too small for these traces.
- `ms` — see the table above.

## Verification

`python3 tracepack/tests/run_all.py` — contract #16 covers this package (26 tests), contract #21 the
keep rule below. Packet-level measurements against the scripted 1–4-hop corpus are in
`RESULTS_condenser.md`.

## Putting back the forgotten file views (`oh_keep`)

A second binding, for a different loss. On repository tasks a compaction rarely loses anything the
agent cannot get back: the files are still on disk. What it loses is steps. Within 10 steps of a
compaction the agent re-reads 46% of the files it had read before (9% without one), to look at them
again rather than to edit them (4% of those re-reads edit the file within 3 steps), and every re-read
fills the context toward the next compaction. `KeepCondenser` puts the agent's own forgotten file views
back after the summary, verbatim, so it does not have to read them again.

```python
from tracepack.condenser.oh_keep import KeepCondenser

agent = agent.model_copy(update={
    "condenser": KeepCondenser(llm=summary_llm, max_size=120),
})
```

Same summary, same trigger, same knobs as `LLMSummarizingCondenser`. No query, no retrieval, no model
call of its own.

**The rule** (`keep.select`: a pure function of the persisted events, identical to the rule the measured
runs used; every constant cites its reading in `keep.PROVENANCE`):

| step | what | why |
|---|---|---|
| candidates | `file_editor` views that a compaction forgot; the latest one per file | if that latest read is still in the context, putting it back would be a duplicate |
| drop | the file was edited since (by the editor, or by a terminal command that writes files); a directory listing; a view over 16,000 characters; a file the summary does not name | a stale view misleads; a listing holds no code; one huge view would take half the budget; the summary names what it still considers live (the latest view of every file covers 67% of the re-reads at ~11.6k tokens per compaction, the named ones 44% at ~4.5k) |
| keep | newest first, at most 4 files and 32,000 characters (about 8k tokens) | a quarter of the 32k-token compaction line |
| put back | each kept view with its whole model response: every action with the same `llm_response_id`, and their observations | the SDK drops a tool call whose model response is partly forgotten |

**Where the views go.** Into every View the condenser returns, right after the summary, until the next
compaction picks again; never into the conversation state. Counted in the state, the put-back views made
the probe compact earlier (0.19 → 0.55 re-compactions in 12 steps). In the View they cost their tokens
on every call but do not move the compaction line.

**Measured** (`PREREG_keep.md`: SWE-bench Verified, OpenHands SDK, executor gpt-5.6-terra, 32k
compaction line; 242 checkpoints taken at real compaction points, 4 runs per checkpoint and arm, against
the stock summarizer continuing from the same checkpoint):

| | stock | keep | difference [95% CI] |
|---|---|---|---|
| runs that hit the 64-step limit | 19.2% | 8.4% | −10.8 [−18.6, −3.0] |
| re-compactions per run | 2.86 | 1.63 | −1.23 [−1.93, −0.54] |
| steps after the checkpoint (post hoc) | 30.7 | 25.2 | −18.1% [−23.6%, −12.2%]; −10.5% on the 115 checkpoints not used to design the rule |
| solved | 46.7% | 50.0% | +3.3 [−1.4, +10.3], tied |
| tokens | | | +12.8% |

The pre-registered gate, a lower cost per solved task, did not pass (ratio 0.928 [0.808, 1.062]). The
drop in step-limit hits comes mostly from 7 long instances, 5 of which were used to design the rule. The
probe before it (166 checkpoints, 12 steps): reads of the put-back files −56% with the real views, +4%
when each view was replaced by a note saying "you read this, unchanged since".

**Where it does not help.** Short tasks that never compact; a needed fact that is gone (the files are
still there, so this saves steps, not correctness: the solve rate did not move); a summary that already
carries the code.

**Where the forgotten events come from.** `events_dir` = the conversation's persisted events directory
(what the measured runs read; it also covers a conversation resumed from disk). Unset, the condenser
keeps its own archive of the events it has seen in a View: the same events in the same order, for a
conversation that started with it. `log_path` writes one line per compaction (files, ranges, what each
rule dropped); `enabled=False` is the control arm; a failure falls back to the stock view and never
raises into the agent loop.

**Checks.** `tests/test_keep.py` (contract #21, python 3.9) covers each clause of the rule.
`python3.12 -m tracepack.condenser.selfcheck_oh_keep` checks the binding wherever the SDK is installed:
the views land right after the summary, their outputs reach the model as tool messages, the
conversation's own view is not changed, failures fall back, and the archive picks the same views as the
persisted events. On the runs' own data (not public), the shipped rule returned exactly what the
measured one did at all 646 compactions of the 249 checkpoint prefixes.
