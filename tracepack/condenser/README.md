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

Only the last one needs the SDK, so the engine and its contracts run anywhere.

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

`python3 tracepack/tests/run_all.py` — contract #16 covers this package (26 tests). Packet-level
measurements against the scripted 1–4-hop corpus are in `RESULTS_condenser.md`.
