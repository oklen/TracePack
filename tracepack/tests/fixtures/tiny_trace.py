"""tracepack.tests.fixtures.tiny_trace -- hand-built traces whose correct answer is known.

Every builder here exists so that a test can assert an *arithmetic* fact ("required costs
exactly 200 tokens", "the head of the chain is s_c") instead of asserting whatever the code
happened to produce.  A fixture whose expected answer is computed by the module under test is
not a fixture, it is a tautology; so each builder ships with module-level ``*_EXPECT`` /
``*_COST`` constants written out by hand from the proposal's rules (§3.2 edge taxonomy, §3.3
query modes, §3.4 hard budget, §4.5 carrier gate).

Design decisions the tests depend on
------------------------------------

* **Costs are declared, not derived.**  ``TraceEvent.token_cost`` and the ``raw_text``
  ``RepresentationRef`` carry the same hand-picked integer, so a budget test can say
  "budget == 200 fits exactly, 199 does not" without owning a tokenizer.  ``estimate_text_tokens``
  is only ever applied to the assembler *header*, which every fixture leaves empty.

* **Every graph carries a distractor.**  A trace where all events are required cannot detect
  over-expansion; each fixture has at least one event reachable only through a WEAK edge
  (``CONTROL``/``TEMPORAL``), which closure must never pull in (§3.2).

* **The broken fixtures return ``(events, edges)``, not a graph.**  ``duplicate_id_events`` and
  ``dangling_edge_events`` are inputs that must *raise* in ``TraceGraph.__init__``; handing back
  a constructed graph would mean the guard already failed inside the fixture.

* **``random_graph(seed)`` is seeded and total.**  It is the many-graphs axis of test contract
  #2 (Budget Violation Rate == 0).  Randomness comes from ``random.Random(seed)`` only -- no
  global RNG, no clock -- and the generator never emits a self edge, a dangling edge, a
  duplicate id, a ``MATERIALIZES`` without a predicate, or a ``SUPERSEDES`` pointing forward in
  time, so every produced graph is constructible by definition.

* **``crossed_tool_rows()`` makes order-pairing and id-pairing disagree on every pair.**  Three
  tool calls A, B, C answered in the order C, A, B.  DESIGN_FROZEN §0 measured order pairing at
  96.53% agreement on the real corpus (31.8% in the worst session) and it never raises -- it
  just attaches results to the wrong call.  This fixture turns that silent 3.5% into a 100%
  mismatch so a test can see it.

Implements fixtures for proposal §8.1 contracts 1-10 plus the adversarial cases in §8's spirit.
Pure stdlib, deterministic, no network / LLM / torch.
"""
from __future__ import annotations

import os
import random
import sys

if __package__ in (None, ""):  # pragma: no cover - direct execution
    sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(
        os.path.dirname(os.path.abspath(__file__))))))

from tracepack.core.graph import TraceGraph
from tracepack.core.schema import (
    CarrierVerification,
    RepresentationRef,
    SchemaError,
    Seed,
    TraceEdge,
    TraceEvent,
)

__all__ = [
    # builders
    "event", "seed", "tool_chain_graph", "supersede_chain_graph", "supersede_cycle_graph",
    "carrier_source_graph", "carrier_verification", "oversized_event_graph", "exact_fit_graph",
    "deep_chain_graph", "cyclic_depends_graph", "pin_vs_topk_graph", "atomic_pair_graph",
    "duplicate_id_events", "dangling_edge_events", "self_edge_events",
    "random_graph", "random_graphs", "crossed_tool_rows", "order_pairing_would_give",
    "id_pairing_truth",
    # known answers
    "TOOL_CHAIN_SEED", "TOOL_CHAIN_WHY_REQUIRED", "TOOL_CHAIN_WHY_COST", "TOOL_CHAIN_DISTRACTOR",
    "SUPERSEDE_LATEST", "SUPERSEDE_STALE", "CARRIER_ID", "CARRIER_SOURCE_ID",
    "CARRIER_PREDICATE", "CARRIER_MODEL_ID", "CARRIER_READOUT", "CARRIER_CONSTRUCTION",
    "CARRIER_SOURCE_COST", "OVERSIZED_ID", "OVERSIZED_COST", "EXACT_FIT_SEED", "EXACT_FIT_SUM",
    "EXACT_FIT_REQUIRED", "DEEP_CHAIN_SEED", "CYCLE_IDS", "PIN_TARGET", "PIN_QUERY",
    "ATOMIC_GROUP", "ATOMIC_PAIR", "ATOMIC_PAIR_COST",
]


# ---------------------------------------------------------------- primitives


def event(event_id: str, kind: str, timestamp: int, text: str, cost: int, *,
          group: str | None = None, tool_call_id: str | None = None,
          step_id: str | None = None, native_ref: str | None = None,
          carrier_text: str | None = None, carrier_cost: int | None = None,
          meta: dict | None = None) -> TraceEvent:
    """One event whose ``raw_text`` representation costs exactly ``cost``.

    ``carrier_text`` adds a second ``materialized_text`` witness so the assembler's
    ``carrier_only`` / ``source_plus_carrier`` policies have something to choose between.
    """
    if cost < 0:
        raise SchemaError("fixture cost must be >= 0, got %d" % cost)
    reps = [RepresentationRef(kind="raw_text", token_cost=cost, text=text)]
    if carrier_text is not None:
        if carrier_cost is None:
            raise SchemaError("carrier_text needs carrier_cost (a witness without a price "
                              "cannot be budgeted)")
        reps.append(RepresentationRef(kind="materialized_text", token_cost=carrier_cost,
                                      text=carrier_text))
    return TraceEvent(
        event_id=event_id, kind=kind, text=text, timestamp=timestamp,
        step_id=step_id, tool_call_id=tool_call_id, native_ref=native_ref,
        token_cost=cost, representations=tuple(reps), atomic_group=group,
        meta=dict(meta or {}),
    )


def seed(event_id: str, rank: int = 0, source: str = "lexical",
         score: float | None = None, pinned: bool = False) -> Seed:
    """A router result.  ``score`` defaults to a rank-monotone value so ties never decide."""
    if score is None:
        score = 1.0 / (1.0 + rank)
    return Seed(event_id=event_id, score=score, source=source, rank=rank, pinned=pinned)


# ---------------------------------------------------------------- 1. tool chain
#
#   u1 (user) <--GROUNDED_IN-- d1 (decision) --DEPENDS_ON--> tr1 --RESULT_OF--> tc1
#   noise1 hangs off d1 by a TEMPORAL edge only: closure must never reach it (§3.2).

TOOL_CHAIN_SEED = "d1"
#: native closure, query_mode="why": the decision, its source, the source's call, the question
TOOL_CHAIN_WHY_REQUIRED = ("u1", "tc1", "tr1", "d1")
#: 5 + 6 + 30 + 8 -- written out by hand, not read back from the events
TOOL_CHAIN_WHY_COST = 49
TOOL_CHAIN_DISTRACTOR = "noise1"
ATOMIC_GROUP = "grp:t1"
ATOMIC_PAIR = ("tc1", "tr1")
ATOMIC_PAIR_COST = 36          # 6 + 30


def tool_chain_graph() -> TraceGraph:
    """A four-event evidence chain plus one weak-edge-only distractor."""
    events = [
        event("u1", "user", 1, "why did you pick vendor B for the reorder", 5),
        event("tc1", "tool_call", 2, 'Read {"file_path": "/tmp/vendors.csv"}', 6,
              group=ATOMIC_GROUP, tool_call_id="t1", step_id="00000002"),
        event("tr1", "tool_result", 3,
              "vendor A unit 940.00 USD lead 21d\nvendor B unit 780.25 USD lead 9d", 30,
              group=ATOMIC_GROUP, tool_call_id="t1", step_id="00000002"),
        event("d1", "decision", 4, "going with vendor B: cheaper and faster", 8),
        event(TOOL_CHAIN_DISTRACTOR, "assistant", 5, "unrelated chatter about the weather", 7),
    ]
    edges = [
        TraceEdge("tr1", "tc1", "RESULT_OF", predicate="tool_use_id"),
        TraceEdge("d1", "tr1", "DEPENDS_ON"),
        TraceEdge("d1", "u1", "GROUNDED_IN"),
        TraceEdge("tc1", "u1", "CONTROL", predicate="parentUuid"),
        TraceEdge(TOOL_CHAIN_DISTRACTOR, "d1", "TEMPORAL", predicate="transcript_order"),
    ]
    return TraceGraph(events, edges)


def atomic_pair_graph() -> TraceGraph:
    """Just the call/result pair plus a cheap decision that depends on the result.

    Sized so that a budget can fit the decision and *half* the pair: the packer must then
    take neither half (§3.4 all-or-nothing), not the result alone.
    """
    events = [
        event("dec", "decision", 3, "shipping from the eu-west warehouse", 10),
        event("call", "tool_call", 1, 'Bash {"command": "warehouse --list"}', 12,
              group="grp:pair", tool_call_id="tp"),
        event("res", "tool_result", 2, "eu-west: 412 units\nus-east: 0 units", 24,
              group="grp:pair", tool_call_id="tp"),
    ]
    edges = [
        TraceEdge("res", "call", "RESULT_OF", predicate="tool_use_id"),
        TraceEdge("dec", "res", "DEPENDS_ON"),
    ]
    return TraceGraph(events, edges)


# ---------------------------------------------------------------- 2. supersede chain

SUPERSEDE_LATEST = "s_c"
SUPERSEDE_STALE = ("s_a", "s_b")


def supersede_chain_graph() -> TraceGraph:
    """Three revisions of one state; ``s_c`` is the only non-stale answer (§3.3, contract #7)."""
    events = [
        event("s_a", "state", 1, "deploy target: us-east-1", 9),
        event("s_b", "state", 2, "correction, deploy target: eu-west-1", 9),
        event("s_c", "state", 3, "final: deploy target is ap-southeast-1", 9),
        event("bystander", "assistant", 4, "noting the deploy target change", 6),
    ]
    edges = [
        TraceEdge("s_b", "s_a", "SUPERSEDES"),
        TraceEdge("s_c", "s_b", "SUPERSEDES"),
        TraceEdge("bystander", "s_c", "TEMPORAL", predicate="transcript_order"),
    ]
    return TraceGraph(events, edges)


def supersede_cycle_graph() -> TraceGraph:
    """A corrupt trace: x -> y -> z -> x.  Constructible, but ``latest_state`` must refuse."""
    events = [event(e, "state", i + 1, "cyclic state %s" % e, 4)
              for i, e in enumerate(("cx", "cy", "cz"))]
    edges = [
        TraceEdge("cy", "cx", "SUPERSEDES"),
        TraceEdge("cz", "cy", "SUPERSEDES"),
        TraceEdge("cx", "cz", "SUPERSEDES"),
    ]
    return TraceGraph(events, edges)


# ---------------------------------------------------------------- 3. carrier + source

CARRIER_ID = "car1"
CARRIER_SOURCE_ID = "src1"
CARRIER_PREDICATE = "vendor_price"
CARRIER_MODEL_ID = "qwen3-8b@sha256:abc"
CARRIER_READOUT = "readout/v1"
CARRIER_CONSTRUCTION = "native"      # == the MATERIALIZES edge's provenance (§4.5)
CARRIER_SOURCE_COST = 120


def carrier_source_graph() -> TraceGraph:
    """``dec2 --DEPENDS_ON--> car1 --MATERIALIZES(vendor_price)--> src1``.

    The carrier is 6 tokens, the source 120: relaxation is worth 114 tokens, which is exactly
    why §4.5 refuses to grant it without a matching verification record.
    """
    events = [
        event(CARRIER_SOURCE_ID, "tool_result", 1,
              "sku Z-9 unit price 780.25 USD, moq 500, lead 9d, incoterm FOB",
              CARRIER_SOURCE_COST, tool_call_id="tv"),
        event(CARRIER_ID, "summary", 2, "vendor B price is about 780 USD", 6,
              carrier_text="vendor B ~780 USD", carrier_cost=4),
        event("dec2", "decision", 3, "ordering from vendor B", 8),
        event("chatter", "assistant", 4, "acknowledged", 3),
    ]
    edges = [
        TraceEdge(CARRIER_ID, CARRIER_SOURCE_ID, "MATERIALIZES",
                  predicate=CARRIER_PREDICATE, provenance=CARRIER_CONSTRUCTION),
        TraceEdge("dec2", CARRIER_ID, "DEPENDS_ON"),
        TraceEdge("chatter", "dec2", "TEMPORAL", predicate="transcript_order"),
    ]
    return TraceGraph(events, edges)


def carrier_verification(**overrides) -> CarrierVerification:
    """The record that DOES license the relaxation; pass a kwarg to break exactly one field.

    Test contract #9 is "all of model + predicate + protocol must match", so the tests build a
    matching record and then mutate one field at a time -- a mismatch table written by hand
    would drift from this fixture the first time a field is renamed.
    """
    fields = dict(
        carrier_id=CARRIER_ID,
        predicate=CARRIER_PREDICATE,
        model_id=CARRIER_MODEL_ID,
        construction_protocol=CARRIER_CONSTRUCTION,
        readout_protocol=CARRIER_READOUT,
        test_version="carrier-probe/2026-09-01",
        ci_low=-0.012,
        ci_high=0.031,
    )
    unknown = set(overrides) - set(fields)
    if unknown:
        raise SchemaError("carrier_verification: unknown field(s) %s" % sorted(unknown))
    fields.update(overrides)
    return CarrierVerification(**fields)


# ---------------------------------------------------------------- 4. oversized event

OVERSIZED_ID = "huge"
OVERSIZED_COST = 100_000


def oversized_event_graph() -> TraceGraph:
    """A required event that fits no budget the project will ever run (§3.4 stress=1024).

    Real transcripts do this: DESIGN_FROZEN §0 measured a max event cost of 29,213 tokens
    against a 4,096-token relaxed budget.  The only correct behaviour is to report it, never to
    truncate it into a half-quoted tool result the reader cannot tell was cut.
    """
    payload = "PAYLOAD-BEGIN " + ("0123456789abcdef " * 500) + "PAYLOAD-END"
    events = [
        event(OVERSIZED_ID, "tool_result", 1, payload, OVERSIZED_COST, tool_call_id="th"),
        event("tiny", "decision", 2, "the dump above says the checksum is fine", 4),
    ]
    edges = [TraceEdge("tiny", OVERSIZED_ID, "DEPENDS_ON")]
    return TraceGraph(events, edges)


# ---------------------------------------------------------------- 5. exact fit

EXACT_FIT_SEED = "e_root"
#: 40 (e_root) + 100 (e_call) + 60 (e_mid) -- the off-by-one boundary for contract #2
EXACT_FIT_SUM = 200
EXACT_FIT_REQUIRED = ("e_call", "e_mid", "e_root")


def exact_fit_graph() -> TraceGraph:
    """Required evidence costing exactly 200 tokens, with an atomic pair worth 160 of it."""
    events = [
        event("e_call", "tool_call", 1, 'Grep {"pattern": "SAFETY_MARGIN"}', 100,
              group="grp:e1", tool_call_id="te"),
        event("e_mid", "tool_result", 2, "budget.py:41: SAFETY_MARGIN = 256", 60,
              group="grp:e1", tool_call_id="te"),
        event(EXACT_FIT_SEED, "decision", 3, "the margin is 256 tokens", 40),
        event("e_far", "assistant", 4, "unrelated follow-up", 25),
    ]
    edges = [
        TraceEdge("e_mid", "e_call", "RESULT_OF", predicate="tool_use_id"),
        TraceEdge(EXACT_FIT_SEED, "e_mid", "DEPENDS_ON"),
        TraceEdge("e_far", EXACT_FIT_SEED, "TEMPORAL", predicate="transcript_order"),
    ]
    return TraceGraph(events, edges)


# ---------------------------------------------------------------- 6. deep chain

DEEP_CHAIN_SEED = "ch00"


def deep_chain_graph(length: int = 10) -> TraceGraph:
    """``ch00 --DEPENDS_ON--> ch01 --DEPENDS_ON--> ... `` -- longer than ``max_hops``.

    The head is the newest event, so ``ch{i+1}`` is always older than ``ch{i}``: this is what a
    real "the answer rests on the thing before it" chain looks like.
    """
    if not isinstance(length, int) or isinstance(length, bool) or length < 2:
        raise SchemaError("deep_chain_graph(length) needs an int >= 2, got %r" % (length,))
    events = [event("ch%02d" % i, "decision" if i == 0 else "tool_result",
                    length - i, "chain link %02d" % i, 3) for i in range(length)]
    edges = [TraceEdge("ch%02d" % i, "ch%02d" % (i + 1), "DEPENDS_ON")
             for i in range(length - 1)]
    return TraceGraph(events, edges)


# ---------------------------------------------------------------- 7. closure cycle

CYCLE_IDS = ("cy_a", "cy_b", "cy_c")


def cyclic_depends_graph() -> TraceGraph:
    """``a -> b -> c -> a`` over DEPENDS_ON: closure must terminate, not spin (§3.3)."""
    events = [event(e, "decision", i + 1, "mutually justified claim %s" % e, 5)
              for i, e in enumerate(CYCLE_IDS)]
    edges = [
        TraceEdge("cy_a", "cy_b", "DEPENDS_ON"),
        TraceEdge("cy_b", "cy_c", "DEPENDS_ON"),
        TraceEdge("cy_c", "cy_a", "DEPENDS_ON"),
    ]
    return TraceGraph(events, edges)


# ---------------------------------------------------------------- 8. pin vs top-k

PIN_TARGET = "addressed"
#: names step 17 explicitly; every *word* in it points at the decoys instead
PIN_QUERY = 'in step 17, which retry budget threshold did we settle on?'


def pin_vs_topk_graph() -> TraceGraph:
    """One explicitly addressed event with deliberately non-matching text, plus 6 decoys.

    The decoys repeat every content word of ``PIN_QUERY``, so an unpinned retriever ranks the
    addressed event last -- which is what makes contract #5 falsifiable rather than vacuous.
    """
    decoy_text = ("retry budget threshold settle retry budget threshold "
                  "we settled the retry budget threshold in this turn")
    events = [event(PIN_TARGET, "tool_result", 1,
                    "kappa equals 0.24 and the ledger balanced", 12,
                    step_id="17", tool_call_id="tp17")]
    for i in range(6):
        events.append(event("decoy%d" % i, "assistant", 2 + i,
                            "%s (%d)" % (decoy_text, i), 12, step_id="%08d" % (90 + i)))
    return TraceGraph(events, [])


# ---------------------------------------------------------------- 9. malformed inputs


def duplicate_id_events():
    """``(events, edges)`` with the same ``event_id`` twice -- ``TraceGraph`` must raise."""
    return ([event("dup", "user", 1, "first", 4), event("dup", "user", 2, "second", 4)], [])


def dangling_edge_events():
    """``(events, edges)`` whose edge points at an event that does not exist -- must raise."""
    return ([event("only", "user", 1, "lonely", 4)],
            [TraceEdge("only", "ghost", "DEPENDS_ON")])


def self_edge_events():
    """An event depending on itself: rejected by ``TraceEdge.__post_init__`` at build time."""
    return ([event("solo", "decision", 1, "I justify myself", 4)],
            [("solo", "solo", "DEPENDS_ON")])          # tuple: TraceEdge() itself must raise


# ---------------------------------------------------------------- 10. seeded random graphs

_KINDS_PLAIN = ("user", "assistant")


def random_graph(rng_seed: int, n_events: int = 24, *, tag: str | None = None) -> TraceGraph:
    """Deterministic pseudo-random trace: tool pairs, decisions, states, carriers, distractors.

    Same ``rng_seed`` -> byte-identical graph, in this process and any other (``random.Random``
    is seeded explicitly; the module never touches the global RNG or the clock).  One event in
    ~14 is deliberately huge (2,000-5,000 tokens) so the budget sweep of contract #2 keeps
    hitting the "does not fit at all" branch instead of only the comfortable one.
    """
    if not isinstance(rng_seed, int) or isinstance(rng_seed, bool):
        raise SchemaError("random_graph(rng_seed) must be an int, got %r" % (rng_seed,))
    if not isinstance(n_events, int) or isinstance(n_events, bool) or n_events < 4:
        raise SchemaError("random_graph(n_events) must be an int >= 4, got %r" % (n_events,))
    rng = random.Random(rng_seed)
    pre = "g%d" % rng_seed if tag is None else tag
    events: list[TraceEvent] = []
    edges: list[TraceEdge] = []
    results: list[str] = []
    states: list[str] = []
    costly: list[str] = []
    ts = 0
    i = 0

    def cost() -> int:
        return rng.randint(2000, 5000) if rng.random() < 0.07 else rng.randint(3, 220)

    while len(events) < n_events:
        i += 1
        ts += 1
        roll = rng.random()
        if roll < 0.40:
            gid = "%s:grp%03d" % (pre, i)
            tid = "%s:tool%03d" % (pre, i)
            call = "%s:call%03d" % (pre, i)
            res = "%s:res%03d" % (pre, i)
            events.append(event(call, "tool_call", ts, "call %03d payload" % i, cost(),
                                group=gid, tool_call_id=tid))
            ts += 1
            events.append(event(res, "tool_result", ts, "result %03d payload" % i, cost(),
                                group=gid, tool_call_id=tid))
            edges.append(TraceEdge(res, call, "RESULT_OF", predicate="tool_use_id"))
            results.append(res)
            costly.append(res)
        elif roll < 0.62:
            dec = "%s:dec%03d" % (pre, i)
            events.append(event(dec, "decision", ts, "decision %03d" % i, cost()))
            for src in rng.sample(results, min(len(results), rng.randint(0, 2))):
                edges.append(TraceEdge(dec, src, "DEPENDS_ON"))
            costly.append(dec)
        elif roll < 0.78:
            sid = "%s:st%03d" % (pre, i)
            events.append(event(sid, "state", ts, "state %03d" % i, cost()))
            if states and rng.random() < 0.7:
                # always points BACKWARDS in time -> the SUPERSEDES relation stays acyclic
                edges.append(TraceEdge(sid, states[-1], "SUPERSEDES"))
            states.append(sid)
        elif roll < 0.88 and costly:
            car = "%s:car%03d" % (pre, i)
            events.append(event(car, "summary", ts, "summary %03d" % i, rng.randint(3, 30),
                                carrier_text="carrier %03d" % i, carrier_cost=rng.randint(2, 20)))
            edges.append(TraceEdge(car, rng.choice(costly), "MATERIALIZES",
                                   predicate="predicate_%03d" % i, provenance="native"))
        else:
            plain = "%s:msg%03d" % (pre, i)
            events.append(event(plain, rng.choice(_KINDS_PLAIN), ts, "message %03d" % i, cost()))
        if len(events) >= 2:
            edges.append(TraceEdge(events[-1].event_id, events[-2].event_id, "TEMPORAL",
                                   predicate="transcript_order"))
    return TraceGraph(events, edges)


def random_graphs(n: int = 12, *, first_seed: int = 0, n_events: int = 24):
    """``[(rng_seed, graph), ...]`` -- the many-graphs axis of test contract #2."""
    if not isinstance(n, int) or isinstance(n, bool) or n < 1:
        raise SchemaError("random_graphs(n) must be a positive int, got %r" % (n,))
    return [(s, random_graph(s, n_events)) for s in range(first_seed, first_seed + n)]


# ---------------------------------------------------------------- 11. adapter rows

_CROSSED_PAYLOADS = {
    "toolu_A": "alpha: 3 files changed, 412 insertions",
    "toolu_B": "beta: SAFETY_MARGIN = 256 in budget.py line 41",
    "toolu_C": "gamma: no matches found for the pattern",
}


def crossed_tool_rows():
    """8 transcript rows where results come back C, A, B -- order pairing gets all three wrong.

    Also carries the two traps the adapter is built around: a ``file-history-snapshot`` row
    (24.5% of real transcript bytes, must never become an event) and a ``toolUseResult`` field
    duplicating a result payload with *different* text (reading it would double every
    tool-result token count).
    """
    rows = [
        {"type": "user", "uuid": "x-u1", "timestamp": "2026-09-01T00:00:01.000Z",
         "message": {"role": "user", "content": "audit the three checks"}},
        {"type": "file-history-snapshot", "uuid": "x-snap", "messageId": "m1",
         "snapshot": {"files": {"a": "b"}}},
        {"type": "assistant", "uuid": "x-a1", "parentUuid": "x-u1",
         "timestamp": "2026-09-01T00:00:02.000Z",
         "message": {"role": "assistant", "content": [
             {"type": "tool_use", "id": "toolu_A", "name": "Bash",
              "input": {"command": "git diff --stat"}}]}},
        {"type": "assistant", "uuid": "x-a2", "parentUuid": "x-a1",
         "timestamp": "2026-09-01T00:00:03.000Z",
         "message": {"role": "assistant", "content": [
             {"type": "tool_use", "id": "toolu_B", "name": "Grep",
              "input": {"pattern": "SAFETY_MARGIN"}}]}},
        {"type": "assistant", "uuid": "x-a3", "parentUuid": "x-a2",
         "timestamp": "2026-09-01T00:00:04.000Z",
         "message": {"role": "assistant", "content": [
             {"type": "tool_use", "id": "toolu_C", "name": "Grep",
              "input": {"pattern": "NOT_THERE"}}]}},
        # results come back OUT OF ORDER: C first, then A, then B
        {"type": "user", "uuid": "x-r1", "parentUuid": "x-a3",
         "timestamp": "2026-09-01T00:00:05.000Z",
         "message": {"role": "user", "content": [
             {"type": "tool_result", "tool_use_id": "toolu_C",
              "content": [{"type": "text", "text": _CROSSED_PAYLOADS["toolu_C"]}]}]},
         "toolUseResult": {"stdout": "DUPLICATE-MUST-NOT-BE-READ" * 40}},
        {"type": "user", "uuid": "x-r2", "parentUuid": "x-r1",
         "timestamp": "2026-09-01T00:00:06.000Z",
         "message": {"role": "user", "content": [
             {"type": "tool_result", "tool_use_id": "toolu_A",
              "content": _CROSSED_PAYLOADS["toolu_A"]}]}},
        {"type": "user", "uuid": "x-r3", "parentUuid": "x-r2",
         "timestamp": "2026-09-01T00:00:07.000Z",
         "message": {"role": "user", "content": [
             {"type": "tool_result", "tool_use_id": "toolu_B",
              "content": _CROSSED_PAYLOADS["toolu_B"]}]}},
    ]
    return rows


def id_pairing_truth():
    """``{result_row_uuid: call_row_uuid}`` -- the ONLY correct pairing (by ``tool_use_id``)."""
    return {"x-r1": "x-a3", "x-r2": "x-a1", "x-r3": "x-a2"}


def order_pairing_would_give():
    """``{result_row_uuid: call_row_uuid}`` an order-pairing adapter produces instead.

    Disjoint from :func:`id_pairing_truth` on every entry, so a test that asserts "the graph
    matches id pairing" also proves "the graph is not order-paired".
    """
    return {"x-r1": "x-a1", "x-r2": "x-a2", "x-r3": "x-a3"}


# ---------------------------------------------------------------- selfcheck


def _expect(exc, fn, *args, **kwargs) -> None:
    """Fault injection: the guard must fire, and fire as a TracePackError subclass."""
    try:
        fn(*args, **kwargs)
    except exc:
        return
    except Exception as e:  # noqa: BLE001
        raise AssertionError("expected %s from %s, got %s: %s"
                             % (exc.__name__, getattr(fn, "__name__", fn), type(e).__name__, e))
    raise AssertionError("expected %s from %s(%r)"
                         % (exc.__name__, getattr(fn, "__name__", fn), args))


def _selfcheck() -> None:
    # --- the hand-written constants must match the graphs they describe -------------------
    g = tool_chain_graph()
    by_id = {e.event_id: e for e in g.events}
    assert sum(by_id[e].token_cost for e in TOOL_CHAIN_WHY_REQUIRED) == TOOL_CHAIN_WHY_COST, \
        "TOOL_CHAIN_WHY_COST is stale -- a fixture whose constants drift is worse than no fixture"
    assert by_id["tc1"].atomic_group == by_id["tr1"].atomic_group == ATOMIC_GROUP
    assert sum(by_id[e].token_cost for e in ATOMIC_PAIR) == ATOMIC_PAIR_COST
    assert TOOL_CHAIN_DISTRACTOR not in TOOL_CHAIN_WHY_REQUIRED
    # the distractor really is unreachable over STRONG edges from the seed
    strong_out = {(e.src_id, e.dst_id) for e in g.strong_subgraph()}
    assert not any(s == TOOL_CHAIN_DISTRACTOR or d == TOOL_CHAIN_DISTRACTOR
                   for s, d in strong_out)

    ex = exact_fit_graph()
    ex_by_id = {e.event_id: e for e in ex.events}
    assert sum(ex_by_id[e].token_cost for e in EXACT_FIT_REQUIRED) == EXACT_FIT_SUM
    assert ex_by_id["e_call"].atomic_group == ex_by_id["e_mid"].atomic_group

    sup = supersede_chain_graph()
    assert sup.latest_state("s_a") == SUPERSEDE_LATEST
    assert set(sup.superseded_ids()) == set(SUPERSEDE_STALE)
    _expect(SchemaError, supersede_cycle_graph().latest_state, "cx")

    car = carrier_source_graph()
    mat = [e for e in car.edges if e.edge_type == "MATERIALIZES"]
    assert len(mat) == 1 and mat[0].predicate == CARRIER_PREDICATE
    assert mat[0].provenance == CARRIER_CONSTRUCTION
    v = carrier_verification()
    assert v.matches(predicate=CARRIER_PREDICATE, model_id=CARRIER_MODEL_ID,
                     construction_protocol=CARRIER_CONSTRUCTION,
                     readout_protocol=CARRIER_READOUT)
    assert not carrier_verification(model_id="other").matches(
        predicate=CARRIER_PREDICATE, model_id=CARRIER_MODEL_ID,
        construction_protocol=CARRIER_CONSTRUCTION, readout_protocol=CARRIER_READOUT)
    _expect(SchemaError, carrier_verification, not_a_field=1)

    over = oversized_event_graph()
    assert {e.event_id for e in over.events} == {OVERSIZED_ID, "tiny"}
    assert over.event(OVERSIZED_ID).token_cost == OVERSIZED_COST > 4096

    deep = deep_chain_graph(10)
    assert len(deep.events) == 10 and len(deep.edges) == 9
    assert deep.event(DEEP_CHAIN_SEED).timestamp == 10, "the seed must be the NEWEST link"
    _expect(SchemaError, deep_chain_graph, 1)

    cyc = cyclic_depends_graph()
    assert len(cyc.edges) == 3 and {e.event_id for e in cyc.events} == set(CYCLE_IDS)

    pin = pin_vs_topk_graph()
    assert pin.event(PIN_TARGET).step_id == "17"
    assert "17" in PIN_QUERY and "retry" not in pin.event(PIN_TARGET).text, \
        "the addressed event must NOT be lexically findable, or contract #5 proves nothing"

    # --- determinism of the seeded generator --------------------------------------------
    a, b = random_graph(7), random_graph(7)
    assert a.event_ids == b.event_ids and len(a.edges) == len(b.edges)
    assert [e.token_cost for e in a.events] == [e.token_cost for e in b.events]
    assert random_graph(8).event_ids != a.event_ids, "different seeds must differ"
    assert len(random_graph(3, 40).events) >= 40
    seeds_seen = {s for s, _ in random_graphs(5)}
    assert seeds_seen == {0, 1, 2, 3, 4}
    _expect(SchemaError, random_graph, "seven")
    _expect(SchemaError, random_graph, 1, 2)
    _expect(SchemaError, random_graphs, 0)
    # the generator must never emit a graph that TraceGraph rejects
    for s in range(20):
        random_graph(s, 12)

    # --- adapter fixture: the two pairings really do disagree everywhere -----------------
    truth, wrong = id_pairing_truth(), order_pairing_would_give()
    assert set(truth) == set(wrong)
    assert all(truth[k] != wrong[k] for k in truth), \
        "a fixture where order pairing accidentally agrees would test nothing"
    rows = crossed_tool_rows()
    assert any(r.get("type") == "file-history-snapshot" for r in rows)
    assert any("toolUseResult" in r for r in rows)

    # --- malformed fixtures must be rejected downstream, not here -----------------------
    _expect(SchemaError, TraceGraph, *duplicate_id_events())
    _expect(SchemaError, TraceGraph, *dangling_edge_events())
    _expect(SchemaError, TraceEdge, *self_edge_events()[1][0])
    _expect(SchemaError, event, "neg", "user", 1, "text", -1)
    _expect(SchemaError, event, "nocost", "user", 1, "t", 3, carrier_text="c")

    print("tiny_trace selfcheck OK: 11 fixtures, hand-written costs verified "
          "(tool_chain=%d, exact_fit=%d), 20 random graphs constructible, "
          "9 fault injections caught" % (TOOL_CHAIN_WHY_COST, EXACT_FIT_SUM))


if __name__ == "__main__":
    _selfcheck()
