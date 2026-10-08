"""tracepack.eval.baselines -- the ten §6.2 baselines as ready-made (router, closure, repr) specs.

§6.2 lists ten arms to compare "under the same token budget".  This module is the single place
where each arm's *configuration* lives, so that an arm is a data record (:class:`BaselineSpec`)
rather than a code path.  Two consequences the harness depends on: the §6.3 three-factor grid is
just a filter over these specs, and an arm's identity can be hashed into a run id
(:meth:`BaselineSpec.digest`) instead of being described in prose in a results table.

Design decisions others depend on
=================================

1. **A spec is a configuration, never a behaviour.**  ``build_packet`` reads the spec and calls
   the frozen core (``make_router`` -> ``TypedClosure`` -> ``BudgetAssembler``); it contains no
   arm-specific branching except the one arm that is genuinely not a closure at all
   (``full_trace``).  Adding an arm must be a new record, not a new ``if``.

2. **``full_trace`` is a LONG-CONTEXT BASELINE, NOT AN ORACLE.**  §5.2 is explicit ("Full trace
   只能作为长上下文 baseline，不能称为 oracle"), and DESIGN_FROZEN §6 repeats it.  It serves the
   whole trace truncated to the budget **from the END** (the newest contiguous window), which is
   what a serving stack actually does, and it is subject to distraction and lost-in-the-middle.
   See :data:`FULL_TRACE_CAVEAT`, which the results table should quote verbatim.

3. **Tail truncation keeps a CONTIGUOUS suffix and stops at the first event that does not fit.**
   It does not skip an oversized event to squeeze in older, smaller ones.  Greedy skipping would
   quietly turn ``full_trace`` into a *selective* arm -- i.e. a (bad) retriever -- and it would no
   longer be "the whole trace truncated".  The real consequence is recorded rather than hidden:
   the corpus has events up to 29,213 tokens (DESIGN_FROZEN §0), so one oversized newest event
   can starve the packet to empty.  ``full_trace_tail_ids`` is public so this is inspectable.

4. **The budget cap is enforced by the assembler for every arm, ``full_trace`` included.**  The
   tail is chosen with the same cost function the assembler charges (``event.cost_of("raw_text")``
   == ``assembler._source_ref(...).token_cost``), so the packet fits by construction *and* is
   re-checked by the assembler's own hard-budget contract (§3.4).

5. **Oracle arms are seeded from the annotation, in annotation order.**  ``Seed.source="oracle"``,
   ``rank`` = position in ``item.gold_seed``, all scores 1.0: a gold seed list has no retrieval
   score, and inventing a decreasing one would leak a fake ranking signal into the assembler's
   priority order.  An oracle seed naming an event that is not in the graph raises -- a silently
   dropped gold seed turns the ceiling arm into an ordinary one.

6. **Baseline 7 (``full_ancestor``) reuses baseline 6's router.**  §6.2 lists it directly after
   "hybrid + pinning + typed closure", so the contrast that is being drawn is closure policy at
   fixed routing.  Stated because the proposal does not say it outright.

7. **No arm sets a header.**  Per DESIGN_FROZEN §1 the budget covers *evidence* tokens only; the
   reader prompt is fixed and identical across arms, so charging it here would add a constant to
   every arm and shrink the evidence they can actually serve.

8. **Determinism.**  Everything downstream is deterministic already; the only freedom here is the
   dense arm's embedder, which defaults to the router's offline keyed-hash embedding.  Injecting
   ``embed_fn`` is allowed and is recorded by ``build_arm``'s result, never silently.

Implements §6.2 (all ten baselines), §6.3 (the three factors, as spec fields), and the §5.2
arm-to-baseline mapping.  Pure stdlib + the frozen core: no network, no torch, no LLM.
"""
from __future__ import annotations

import hashlib
import json
from dataclasses import dataclass, field
from typing import Any, Callable, Mapping, Sequence

try:  # normal package import
    from tracepack.core.assembler import AssemblerConfig, BudgetAssembler
    from tracepack.core.closure import CLOSURE_MODES, ClosureConfig, TypedClosure
    from tracepack.core.router import ROUTER_NAMES, RouterConfig, make_router
    from tracepack.core.schema import (
        QUERY_MODES,
        BudgetError,
        CarrierVerification,
        ClosureStep,
        EvidenceClosure,
        MemoryPacket,
        SchemaError,
        Seed,
        TraceEvent,
    )
    from tracepack.eval.metrics import item_field, item_id_list
except ImportError:  # pragma: no cover - direct `python3 tracepack/eval/baselines.py`
    import os as _os
    import sys as _sys

    _sys.path.insert(
        0, _os.path.dirname(_os.path.dirname(_os.path.dirname(_os.path.abspath(__file__))))
    )
    from tracepack.core.assembler import AssemblerConfig, BudgetAssembler  # noqa: E402
    from tracepack.core.closure import CLOSURE_MODES, ClosureConfig, TypedClosure  # noqa: E402
    from tracepack.core.router import ROUTER_NAMES, RouterConfig, make_router  # noqa: E402
    from tracepack.core.schema import (  # noqa: E402
        QUERY_MODES,
        BudgetError,
        CarrierVerification,
        ClosureStep,
        EvidenceClosure,
        MemoryPacket,
        SchemaError,
        Seed,
        TraceEvent,
    )
    from tracepack.eval.metrics import item_field, item_id_list  # noqa: E402

__all__ = [
    "BaselineSpec", "ArmOutput", "BASELINE_NAMES", "SEED_SOURCES", "FULL_TRACE_CAVEAT",
    "NO_CLOSURE", "FULL_TRACE_EDGE", "FULL_TRACE_RULE",
    "baseline_specs", "get_spec", "build_packet", "build_arm", "full_trace_tail_ids",
    "oracle_seeds",
]

#: ``closure_mode`` value for the one arm that runs no closure at all (§6.2 #10)
NO_CLOSURE = "none"
#: pseudo edge type / rule recorded for a full-trace entry so the manifest still says *why*
FULL_TRACE_EDGE = "FULL_TRACE"
FULL_TRACE_RULE = "full_trace_tail_truncation"

#: how an arm gets its seeds
SEED_SOURCES = ("router", "oracle", "none")

#: quote this next to any full_trace number (§5.2, DESIGN_FROZEN §6)
FULL_TRACE_CAVEAT = (
    "full_trace is a LONG-CONTEXT BASELINE, not an oracle: it is the whole trace truncated to the "
    "budget from the END, so it is subject to distraction and lost-in-the-middle, it can miss "
    "evidence that fell outside the newest window, and its incomplete_packet_rate is 0.0 by "
    "construction (it declares no required set) and is therefore not comparable with the closure "
    "arms."
)

_ORACLE_SEED_SCORE = 1.0


# ============================================================ the spec


@dataclass(frozen=True)
class BaselineSpec:
    """One §6.2 arm: which router, which closure policy, which representation.

    ``closure_mode`` is a :data:`~tracepack.core.closure.CLOSURE_MODES` value, or
    :data:`NO_CLOSURE` for ``full_trace``.  ``seed_source`` says where seeds come from:
    ``"router"`` (run ``router``), ``"oracle"`` (the item's ``gold_seed``), ``"none"``.
    ``factors`` is the (Router x Closure x Representation) cell of §6.3 this arm occupies, so a
    results table can be grouped without re-deriving it.
    """

    name: str
    router: "str | None"
    closure_mode: str
    repr_policy: str
    seed_source: str
    k: int = 8
    router_kwargs: Mapping[str, Any] = field(default_factory=dict)
    is_long_context: bool = False
    factors: str = ""
    notes: str = ""

    def __post_init__(self):
        if not isinstance(self.name, str) or not self.name:
            raise SchemaError("BaselineSpec.name must be a non-empty str")
        if self.seed_source not in SEED_SOURCES:
            raise SchemaError("unknown seed_source %r (want one of %s)"
                              % (self.seed_source, list(SEED_SOURCES)))
        if self.router is not None and self.router not in ROUTER_NAMES:
            raise SchemaError("unknown router %r (want one of %s or None)"
                              % (self.router, list(ROUTER_NAMES)))
        if self.seed_source == "router" and self.router is None:
            raise SchemaError("%s: seed_source='router' needs a router name" % self.name)
        if self.seed_source != "router" and self.router is not None:
            raise SchemaError("%s: router %r is set but seeds come from %r -- an arm that names a "
                              "router it never runs is a mislabelled result"
                              % (self.name, self.router, self.seed_source))
        if self.closure_mode not in CLOSURE_MODES and self.closure_mode != NO_CLOSURE:
            raise SchemaError("unknown closure_mode %r (want one of %s or %r)"
                              % (self.closure_mode, list(CLOSURE_MODES), NO_CLOSURE))
        if not isinstance(self.k, int) or isinstance(self.k, bool) or self.k <= 0:
            raise SchemaError("BaselineSpec.k must be a positive int, got %r" % (self.k,))
        if not isinstance(self.router_kwargs, Mapping):
            raise SchemaError("BaselineSpec.router_kwargs must be a mapping")
        # copy, so a caller mutating the dict it passed cannot retro-actively change an arm
        object.__setattr__(self, "router_kwargs", dict(self.router_kwargs))
        # AssemblerConfig owns the repr_policy vocabulary; validate by construction
        AssemblerConfig(repr_policy=self.repr_policy)

    def digest(self) -> str:
        """Deterministic id for this arm -- put it in the run id, not a prose description."""
        payload = {
            "name": self.name, "router": self.router, "closure": self.closure_mode,
            "repr": self.repr_policy, "seeds": self.seed_source, "k": self.k,
            "router_kwargs": {k: repr(v) for k, v in sorted(self.router_kwargs.items())},
        }
        blob = json.dumps(payload, sort_keys=True, ensure_ascii=False, separators=(",", ":"))
        return hashlib.sha256(blob.encode("utf-8")).hexdigest()[:16]

    def with_repr(self, repr_policy: str) -> "BaselineSpec":
        """The same arm at another representation level (§6.3's third factor)."""
        return BaselineSpec(
            name="%s+%s" % (self.name, repr_policy), router=self.router,
            closure_mode=self.closure_mode, repr_policy=repr_policy,
            seed_source=self.seed_source, k=self.k, router_kwargs=dict(self.router_kwargs),
            is_long_context=self.is_long_context,
            factors=self.factors.rsplit(" x ", 1)[0] + " x " + repr_policy,
            notes=self.notes)


@dataclass(frozen=True)
class ArmOutput:
    """Everything one arm produced for one item.

    ``closure`` is carried out separately because ``metrics.dependency_closure_recall`` /
    ``closure_precision`` measure the closure stage and refuse to reconstruct it from the
    manifest.  ``seeds`` is kept so router-level analyses do not have to re-run the router.
    """

    spec: BaselineSpec
    packet: MemoryPacket
    closure: EvidenceClosure
    seeds: "tuple[Seed, ...]"
    budget: int


# ============================================================ the ten arms (§6.2)


_SPECS: "tuple[BaselineSpec, ...]" = (
    BaselineSpec(
        name="last_n", router="last_n", closure_mode="off", repr_policy="source_only",
        seed_source="router", factors="last_n x off x source_only",
        notes="§6.2 #1. No retrieval at all: the k newest events. The recency control."),
    BaselineSpec(
        name="bm25", router="lexical", closure_mode="off", repr_policy="source_only",
        seed_source="router", factors="lexical x off x source_only",
        notes="§6.2 #2. BM25 over event text only; identifiers are unreachable here by "
              "design (router design note 5), so this is pure lexical matching."),
    BaselineSpec(
        name="dense", router="dense", closure_mode="off", repr_policy="source_only",
        seed_source="router", factors="dense x off x source_only",
        notes="§6.2 #3. Cosine over the offline keyed-hash embedding unless embed_fn is "
              "injected at build time."),
    BaselineSpec(
        name="hybrid", router="hybrid", closure_mode="off", repr_policy="source_only",
        seed_source="router", factors="hybrid x off x source_only",
        notes="§6.2 #4. RRF of the lexical and dense arms."),
    BaselineSpec(
        name="hybrid_pin", router="hybrid_pin", closure_mode="off", repr_policy="source_only",
        seed_source="router", factors="hybrid_pin x off x source_only",
        notes="§6.2 #5. Adds explicit step/tool/path pinning. Pins are additive and are never "
              "displaced by k, so this arm can return more than k seeds (§4.4)."),
    BaselineSpec(
        name="hybrid_pin_closure", router="hybrid_pin", closure_mode="native",
        repr_policy="source_only", seed_source="router",
        factors="hybrid_pin x native x source_only",
        notes="§6.2 #6. THE PROPOSED SYSTEM. Its contrast with hybrid_pin is G_closure "
              "(§5.2) at fixed routing and fixed budget."),
    BaselineSpec(
        name="full_ancestor", router="hybrid_pin", closure_mode="full_ancestor",
        repr_policy="source_only", seed_source="router",
        factors="hybrid_pin x full_ancestor x source_only",
        notes="§6.2 #7. Deliberately over-expanding: every STRONG edge, no query-mode rules, "
              "no carrier gate. Same router as #6 so the contrast is closure policy alone "
              "(design note 6). Expect high evidence recall and low closure precision."),
    BaselineSpec(
        name="oracle_seed_native", router=None, closure_mode="native",
        repr_policy="source_only", seed_source="oracle",
        factors="oracle x native x source_only",
        notes="§6.2 #8. Gold seeds + the real closure. Its gap to #6 is L_router (§5.2): how "
              "much of the remaining error is retrieval rather than expansion."),
    BaselineSpec(
        name="oracle_dependency", router=None, closure_mode="oracle",
        repr_policy="source_only", seed_source="oracle",
        factors="oracle x oracle x source_only",
        notes="§6.2 #9. Gold seeds + the annotated gold minimal evidence, expanded no further. "
              "Its gap to #8 is L_closure (§5.2): edge-coverage misses. Failure here is "
              "task_or_reader_failure (§5.4) -- the ceiling of the evidence pipeline."),
    BaselineSpec(
        name="full_trace", router=None, closure_mode=NO_CLOSURE, repr_policy="source_only",
        seed_source="none", is_long_context=True, factors="none x none x source_only",
        notes="§6.2 #10. " + FULL_TRACE_CAVEAT),
)

#: arm names in §6.2 order
BASELINE_NAMES = tuple(s.name for s in _SPECS)


def baseline_specs() -> "list[BaselineSpec]":
    """The ten §6.2 baselines, in the proposal's order.  Fresh list; specs are frozen."""
    return list(_SPECS)


def get_spec(name: str) -> BaselineSpec:
    """One arm by name.  Unknown names raise instead of falling back to a default arm."""
    if not isinstance(name, str):
        raise SchemaError("baseline name must be a str, got %r" % (type(name),))
    for s in _SPECS:
        if s.name == name:
            return s
    raise SchemaError("unknown baseline %r (known: %s)" % (name, ", ".join(BASELINE_NAMES)))


# ============================================================ graph access


def _events_chronological(graph: Any) -> "list[TraceEvent]":
    """Events in the canonical ``(timestamp, event_id)`` order (``graph.py``'s ``_order_key``).

    Duck-typed like every other consumer of the graph, and re-sorted rather than trusted: a
    ``TraceGraph`` is already in this order, and a mapping or bare sequence is not.
    """
    if graph is None:
        raise SchemaError("graph is None")
    raw = getattr(graph, "events", None)
    if callable(raw):
        raw = raw()
    if raw is None:
        if isinstance(graph, Mapping):
            raw = list(graph.values())
        elif isinstance(graph, (list, tuple)):
            raw = list(graph)
        else:
            raise SchemaError("graph exposes no .events and is not a mapping/sequence: %r"
                              % (type(graph),))
    if isinstance(raw, Mapping):
        raw = list(raw.values())
    events = list(raw)
    for ev in events:
        if not isinstance(ev, TraceEvent):
            raise SchemaError("graph contains a non-TraceEvent: %r" % (type(ev),))
    events.sort(key=lambda e: (e.timestamp, e.event_id))
    return events


def _known_ids(graph: Any) -> "frozenset[str]":
    return frozenset(e.event_id for e in _events_chronological(graph))


# ============================================================ seeds


def oracle_seeds(item: Any, graph: Any) -> "tuple[Seed, ...]":
    """Gold seeds for the oracle arms: ``item.gold_seed`` in annotation order.

    ``source="oracle"``, ``rank`` = annotation position, ``score`` = 1.0 for all of them.  A gold
    list carries no retrieval score; fabricating a decreasing one would feed a fake ranking into
    the assembler's priority order and quietly make the "oracle" arm order-sensitive.
    An id that is not in the graph raises -- silently dropping a gold seed turns the ceiling arm
    into an ordinary one (design note 5).
    """
    ids = item_id_list(item, "gold_seed")
    known = _known_ids(graph)
    missing = [i for i in ids if i not in known]
    if missing:
        raise SchemaError("oracle seeds name events that are not in this graph: %s "
                          "(wrong trace for this item?)" % (missing[:3],))
    return tuple(Seed(event_id=eid, score=_ORACLE_SEED_SCORE, source="oracle", rank=i)
                 for i, eid in enumerate(ids))


def _router_seeds(spec: BaselineSpec, query: str, graph: Any,
                  embed_fn: "Callable[..., Any] | None") -> "tuple[Seed, ...]":
    cfg = RouterConfig(k=spec.k)
    kwargs = dict(spec.router_kwargs)
    if embed_fn is not None:
        # inject the encoder wherever the arm actually has a dense component
        if spec.router == "dense":
            kwargs.setdefault("embed_fn", embed_fn)
        elif spec.router in ("hybrid", "hybrid_pin"):
            dense = make_router("dense", cfg, embed_fn=embed_fn)
            hybrid = make_router("hybrid", cfg, dense=dense)
            if spec.router == "hybrid":
                return tuple(hybrid.retrieve(query, graph, spec.k))
            kwargs.setdefault("hybrid", hybrid)
        else:
            raise SchemaError("embed_fn was passed but arm %r has no dense component"
                              % (spec.name,))
    router = make_router(spec.router, cfg, **kwargs)
    return tuple(router.retrieve(query, graph, spec.k))


# ============================================================ full trace (§6.2 #10)


def full_trace_tail_ids(graph: Any, budget: int) -> "tuple[str, ...]":
    """The newest contiguous window of events that fits in ``budget``, in chronological order.

    Walks the trace backwards from the last event, charging ``event.cost_of("raw_text")`` -- the
    same number ``assembler._source_ref`` charges -- and **stops at the first event that does not
    fit**.  It does not skip an oversized event to fit older, smaller ones: that would make this
    arm a (bad) retriever rather than a truncation, and §5.2 needs it to be a truncation.
    Returns ``()`` when even the newest event exceeds the budget; with events up to 29,213 tokens
    in the corpus (DESIGN_FROZEN §0) that is a real outcome, not a defect, and it is exactly what
    tail truncation does to a serving stack.
    """
    if isinstance(budget, bool) or not isinstance(budget, int):
        raise BudgetError("budget must be an int, got %r" % (type(budget),))
    if budget < 0:
        raise BudgetError("negative budget: %d" % budget)
    room = budget
    picked: "list[str]" = []
    for ev in reversed(_events_chronological(graph)):
        cost = ev.cost_of("raw_text")
        if cost > room:
            break
        room -= cost
        picked.append(ev.event_id)
    picked.reverse()
    return tuple(picked)


def _full_trace_closure(graph: Any, budget: int, query_mode: str) -> EvidenceClosure:
    """Wrap the tail as an EvidenceClosure so the *same* assembler serves every arm.

    Every tail event is ``required`` (they all fit by construction, so nothing is reported
    missing and ``incomplete`` stays False -- see :data:`FULL_TRACE_CAVEAT`), and each carries a
    ``ClosureStep`` with edge type :data:`FULL_TRACE_EDGE` so the manifest's inclusion reason
    reads ``closure:FULL_TRACE`` instead of the misleading ``closure:required``.  ``child_id`` is
    empty: there is no child that pulled the event in, and inventing one would be a fabricated
    audit trail (``TypedClosure._oracle`` uses the same empty-anchor convention).
    """
    tail = full_trace_tail_ids(graph, budget)
    steps = tuple(ClosureStep(child_id="", parent_id=eid, edge_type=FULL_TRACE_EDGE,
                              rule=FULL_TRACE_RULE) for eid in tail)
    return EvidenceClosure(seeds=(), required=tail, optional=(), steps=steps,
                           query_mode=query_mode)


# ============================================================ build


def build_arm(spec: BaselineSpec, item: Any, graph: Any, budget: int, *,
              verifications: "Sequence[CarrierVerification]" = (),
              model_id: "str | None" = None,
              readout_protocol: "str | None" = None,
              oracle_required: "Mapping[str, Sequence[str]] | None" = None,
              embed_fn: "Callable[..., Any] | None" = None,
              query: "str | None" = None,
              order: str = "chronological") -> ArmOutput:
    """Run one arm end to end for one item and return packet + closure + seeds.

    ``build_packet`` is the thin wrapper that returns only the packet; use this one when the
    closure is needed too -- ``metrics.dependency_closure_recall`` and ``closure_precision``
    require it and will not reconstruct it from the manifest.

    ``query`` overrides ``item.query``.  It exists because a forced-choice dataset
    (``datasets.py``) stores ``gold`` and ``distractors`` and renders the question text at read
    time; the router and the exact-payload detector both need that text, so the harness passes
    the string it will actually show the reader.  Whatever is used here MUST be the string the
    reader sees -- routing on one question and asking another silently decouples every retrieval
    metric from every task metric.

    ``oracle_required`` overrides the gold dependency list for the ``oracle`` closure mode; by
    default it is the item's ``gold_minimal_packet``, falling back to
    ``gold_seed + required_sources`` (the same definition as ``metrics.gold_evidence``, in
    annotation order).  ``gold_seed`` alone is refused: the arm would collapse to seeds-only and
    still be reported as the ceiling.
    """
    if not isinstance(spec, BaselineSpec):
        raise SchemaError("spec must be a BaselineSpec, got %r" % (type(spec),))
    if isinstance(budget, bool) or not isinstance(budget, int):
        raise BudgetError("budget must be an int, got %r" % (type(budget),))
    if budget < 0:
        raise BudgetError("negative budget: %d" % budget)

    if query is None:
        query = item_field(item, "query")
    if not isinstance(query, str):
        raise SchemaError("query must be a str, got %r -- pass query= when the dataset renders "
                          "the question at read time" % (type(query),))
    query_mode = item_field(item, "query_mode", "lookup")
    if query_mode not in QUERY_MODES:
        raise SchemaError("item field 'query_mode' is %r (want one of %s)"
                          % (query_mode, list(QUERY_MODES)))

    # ---- seeds
    if spec.seed_source == "router":
        seeds = _router_seeds(spec, query, graph, embed_fn)
    elif spec.seed_source == "oracle":
        if embed_fn is not None:
            raise SchemaError("embed_fn was passed to the oracle-seeded arm %r, which never "
                              "runs a router" % (spec.name,))
        seeds = oracle_seeds(item, graph)
    else:
        if embed_fn is not None:
            raise SchemaError("embed_fn was passed to arm %r, which never runs a router"
                              % (spec.name,))
        seeds = ()

    # ---- closure
    if spec.closure_mode == NO_CLOSURE:
        closure = _full_trace_closure(graph, budget, query_mode)
    else:
        gold: "Mapping[str, Sequence[str]] | None" = None
        if spec.closure_mode == "oracle":
            if oracle_required is not None:
                gold = oracle_required
            else:
                gmp = item_id_list(item, "gold_minimal_packet", required=False)
                req = item_id_list(item, "required_sources", required=False)
                if gmp:
                    ids = gmp
                elif req:
                    sd = item_id_list(item, "gold_seed", required=False)
                    ids = sd + tuple(e for e in req if e not in sd)
                else:
                    # gold_seed alone is NOT enough: the oracle-dependency arm would silently
                    # collapse to a seeds-only arm and still be reported as the ceiling.
                    raise SchemaError(
                        "arm %r needs a gold evidence list: annotate gold_minimal_packet or "
                        "required_sources on item %r"
                        % (spec.name, item_field(item, "item_id", "<no id>")))
                gold = {query: list(ids)}
        closure = TypedClosure(ClosureConfig(mode=spec.closure_mode)).close(
            query, seeds, graph, query_mode=query_mode, verifications=verifications,
            model_id=model_id, readout_protocol=readout_protocol, oracle_required=gold)

    # ---- assembly (§3.4 hard budget; header is empty for every arm -- design note 7)
    assembler = BudgetAssembler(AssemblerConfig(repr_policy=spec.repr_policy, order=order))
    packet = assembler.assemble(query, closure, graph, budget, seeds=seeds,
                                query_mode=query_mode, verifications=verifications)
    return ArmOutput(spec=spec, packet=packet, closure=closure, seeds=tuple(seeds),
                     budget=budget)


def build_packet(spec: BaselineSpec, item: Any, graph: Any, budget: int, **kw) -> MemoryPacket:
    """One arm's :class:`MemoryPacket` for one item.  See :func:`build_arm` for the kwargs."""
    return build_arm(spec, item, graph, budget, **kw).packet


# ============================================================ selfcheck


def _expect(exc, fn, *args, **kwargs) -> None:
    try:
        fn(*args, **kwargs)
    except exc:
        return
    raise AssertionError("expected %s from %r(%r, %r)" % (exc.__name__, fn, args, kwargs))


def _ev(eid, kind, ts, text, cost, **kw) -> TraceEvent:
    return TraceEvent(event_id=eid, kind=kind, text=text, timestamp=ts, token_cost=cost, **kw)


def _fixture():
    """A tiny trace whose every arm's answer is computable by hand.

        u1    user question                                   cost 10
        tc1   tool_call  read_config          group g1        cost 10   (t1)
        tr1   tool_result "port = 8080"       group g1        cost 30   --RESULT_OF--> tc1
        a1    assistant "the port is 8080"                    cost 20   --DEPENDS_ON--> tr1
        tc2   tool_call  unrelated grep                       cost 10   (t2)
        tr2   tool_result "no matches"                        cost 40   --RESULT_OF--> tc2
        u2    user follow-up                                  cost 10
    """
    from tracepack.core.graph import TraceGraph
    from tracepack.core.schema import TraceEdge

    events = [
        _ev("u1", "user", 1, "what port does the service use", 10),
        _ev("tc1", "tool_call", 2, "read_config config.yaml", 10,
            tool_call_id="t1", atomic_group="g1", step_id="s1"),
        _ev("tr1", "tool_result", 3, "port = 8080 in config.yaml", 30,
            tool_call_id="t1", atomic_group="g1", step_id="s1"),
        _ev("a1", "decision", 4, "the service listens on port 8080", 20, step_id="s2"),
        _ev("tc2", "tool_call", 5, "grep unrelated pattern", 10, tool_call_id="t2"),
        _ev("tr2", "tool_result", 6, "no matches found for the unrelated pattern", 40,
            tool_call_id="t2"),
        _ev("u2", "user", 7, "thanks, and what about logging", 10),
    ]
    edges = [
        TraceEdge("tr1", "tc1", "RESULT_OF"),
        TraceEdge("a1", "tr1", "DEPENDS_ON", provenance="inferred"),
        TraceEdge("tr2", "tc2", "RESULT_OF"),
        TraceEdge("tc1", "u1", "CONTROL"),
    ]
    return TraceGraph(events, edges)


def _item(**kw):
    base = {
        "item_id": "i1", "session_id": "sess1", "slice": "decision_source",
        "query": "which port did the config tool report?",
        "query_mode": "why",
        "gold_seed": ["a1"],
        "required_sources": ["a1", "tr1", "tc1"],
        "gold_minimal_packet": ["a1", "tr1", "tc1"],
        "gold_answer": "8080",
    }
    base.update(kw)
    return base


def _selfcheck() -> None:
    graph = _fixture()
    total_cost = sum(e.token_cost for e in graph.events)   # 10+10+30+20+10+40+10 = 130
    assert total_cost == 130, total_cost

    # ---------------------------------------------------------------- the ten specs
    specs = baseline_specs()
    assert len(specs) == 10, len(specs)
    assert [s.name for s in specs] == [
        "last_n", "bm25", "dense", "hybrid", "hybrid_pin", "hybrid_pin_closure",
        "full_ancestor", "oracle_seed_native", "oracle_dependency", "full_trace"]
    assert len({s.digest() for s in specs}) == 10, "every arm must have a distinct digest"
    assert baseline_specs() is not _SPECS and baseline_specs()[0] is specs[0]
    assert get_spec("bm25").router == "lexical"
    assert sum(1 for s in specs if s.is_long_context) == 1
    assert get_spec("full_trace").is_long_context and get_spec("full_trace").closure_mode == "none"
    # the §6.3 third factor is a spec transform, not a new code path
    car = get_spec("hybrid_pin_closure").with_repr("carrier_only")
    assert car.repr_policy == "carrier_only" and car.closure_mode == "native"
    assert car.digest() != get_spec("hybrid_pin_closure").digest()

    # ---------------------------------------------------------------- last_n: known answer
    # k=3 newest events are tc2(10), tr2(40), u2(10) = 60 tokens, all fit in 2048
    last3 = BaselineSpec(name="last_n3", router="last_n", closure_mode="off",
                         repr_policy="source_only", seed_source="router", k=3)
    out = build_arm(last3, _item(), graph, 2048)
    assert set(out.packet.event_ids) == {"tc2", "tr2", "u2"}, out.packet.event_ids
    assert out.packet.manifest.total_tokens == 60
    assert not out.packet.incomplete

    # ---------------------------------------------------------------- closure actually closes
    # seed the decision a1; native closure must pull tr1 (DEPENDS_ON) and then tc1 (RESULT_OF).
    seeded = BaselineSpec(name="probe", router=None, closure_mode="native",
                          repr_policy="source_only", seed_source="oracle")
    got = build_arm(seeded, _item(), graph, 2048)
    assert set(got.closure.required) == {"a1", "tr1", "tc1"}, got.closure.required
    assert set(got.packet.event_ids) == {"tc1", "tr1", "a1"}
    # ... and the no-closure arm at the same seed serves the seed alone
    seed_only = BaselineSpec(name="probe_off", router=None, closure_mode="off",
                             repr_policy="source_only", seed_source="oracle")
    off = build_arm(seed_only, _item(), graph, 2048)
    assert set(off.packet.event_ids) == {"a1"}, off.packet.event_ids
    assert off.packet.manifest.total_tokens == 20

    # oracle_dependency: exactly the annotated gold list, expanded no further
    orc = build_arm(get_spec("oracle_dependency"),
                    _item(gold_minimal_packet=["a1", "tr1"]), graph, 2048)
    assert set(orc.closure.required) == {"a1", "tr1"}, orc.closure.required
    assert set(orc.packet.event_ids) == {"tr1", "a1"}
    # ... and an explicit override wins over the annotation
    ovr = build_arm(get_spec("oracle_dependency"), _item(), graph, 2048,
                    oracle_required={_item()["query"]: ["tr2"]})
    assert set(ovr.closure.required) == {"a1", "tr2"}, ovr.closure.required

    # full_ancestor over-expands relative to native: same router, strictly larger required set
    nat = build_arm(get_spec("hybrid_pin_closure"), _item(), graph, 2048)
    fan = build_arm(get_spec("full_ancestor"), _item(), graph, 2048)
    assert set(nat.closure.required) <= set(fan.closure.required)
    assert nat.seeds and [s.event_id for s in nat.seeds] == [s.event_id for s in fan.seeds]
    # design note 6: #7 must differ from #6 in the closure factor ALONE.  Asserted on the specs,
    # not on the seeds: two routers can coincide on a small trace and hide the difference.
    a6, a7 = get_spec("hybrid_pin_closure"), get_spec("full_ancestor")
    assert (a6.router, a6.k, a6.repr_policy, a6.seed_source) == \
           (a7.router, a7.k, a7.repr_policy, a7.seed_source), (a6, a7)
    assert a6.closure_mode != a7.closure_mode

    # ---- integration with metrics.py: the DESIGN_FROZEN closure gate is computable end to end.
    # required_sources = {a1, tr1, tc1}, gold_seed = {a1}, so deps = {tr1, tc1}.
    from tracepack.eval.metrics import (EvalRecord, dependency_closure_recall,
                                        evidence_recall_at_budget)
    r_off = EvalRecord(item=_item(), packet=off.packet, closure=off.closure)
    r_nat = EvalRecord(item=_item(), packet=got.packet, closure=got.closure)
    assert evidence_recall_at_budget(r_off) == 1.0 / 3.0
    assert evidence_recall_at_budget(r_nat) == 1.0
    assert dependency_closure_recall(r_off) == 0.0
    assert dependency_closure_recall(r_nat) == 1.0

    # ---- a forced-choice dataset item: no question text, gold_seed is a bare id
    ds = {"item_id": "sess#direct_fact#0", "session": "sess1", "slice": "direct_fact",
          "query_mode": "lookup", "gold": "8080", "gold_seed": "tr1",
          "required_sources": ["tc1", "tr1"]}
    _expect(SchemaError, build_packet, get_spec("hybrid_pin_closure"), ds, graph, 2048)
    rendered = "which port did the tool report, 8080 or 9090?"
    pq = build_packet(get_spec("hybrid_pin_closure"), ds, graph, 2048, query=rendered)
    assert pq.manifest.query == rendered, "the manifest must record the query that was routed on"
    orc_ds = build_arm(get_spec("oracle_seed_native"), ds, graph, 2048, query=rendered)
    assert [s.event_id for s in orc_ds.seeds] == ["tr1"], "a bare gold_seed id is ONE seed"
    assert set(orc_ds.packet.event_ids) == {"tc1", "tr1"}

    # ---------------------------------------------------------------- full_trace: tail, not head
    ft = get_spec("full_trace")
    # budget 60 fits exactly the newest three (u2 10 + tr2 40 + tc2 10); a4th (a1, 20) does not
    assert full_trace_tail_ids(graph, 60) == ("tc2", "tr2", "u2")
    assert full_trace_tail_ids(graph, 59) == ("tr2", "u2")
    assert full_trace_tail_ids(graph, 1000) == graph.event_ids, "whole trace when it fits"
    p60 = build_packet(ft, _item(), graph, 60)
    assert p60.event_ids == ("tc2", "tr2", "u2"), p60.event_ids
    assert p60.manifest.total_tokens == 60 and not p60.incomplete
    assert [e.reason for e in p60.manifest.entries] == ["closure:FULL_TRACE"] * 3, \
        "a full_trace entry must say why it is there, not 'closure:required'"
    # it is a TAIL: the head of the trace is dropped first
    assert "u1" not in p60.event_ids and "u2" in p60.event_ids

    # a bare (unsorted) event sequence must give the same tail: "newest" is a property of the
    # timestamps, never of the caller's iteration order
    shuffled = [graph.event(e) for e in ("tr2", "u1", "u2", "a1", "tc2", "tr1", "tc1")]
    assert full_trace_tail_ids(shuffled, 60) == ("tc2", "tr2", "u2")

    # contiguity: an oversized newest event starves the packet instead of being skipped over
    from tracepack.core.graph import TraceGraph
    fat = TraceGraph([_ev("small", "user", 1, "tiny", 5),
                      _ev("huge", "tool_result", 2, "x" * 400, 400)], [])
    assert full_trace_tail_ids(fat, 100) == (), \
        "tail truncation must not skip the oversized newest event to reach an older one"
    empty_pkt = build_packet(ft, _item(), fat, 100)
    assert empty_pkt.event_ids == () and empty_pkt.context == ""
    assert empty_pkt.manifest.total_tokens == 0

    # ---------------------------------------------------------------- budget is hard, every arm
    for spec in specs:
        for budget in (0, 25, 60, 1024, 2048):
            pkt = build_packet(spec, _item(), graph, budget)
            m = pkt.manifest
            assert m.total_tokens <= budget, (spec.name, budget, m.total_tokens)
            assert m.budget == budget
            if m.missing_required:
                assert m.incomplete, (spec.name, budget)
            # determinism: same inputs -> same digest (test contract #1)
            again = build_packet(spec, _item(), graph, budget)
            assert again.manifest.digest() == m.digest(), (spec.name, budget)
    # a budget too small for any single event yields an empty, non-violating packet
    tiny = build_packet(get_spec("hybrid_pin_closure"), _item(), graph, 5)
    assert tiny.event_ids == () and tiny.manifest.total_tokens == 0

    # ---------------------------------------------------------------- oracle seeds
    sds = oracle_seeds(_item(gold_seed=["tr1", "a1", "tr1"]), graph)
    assert [s.event_id for s in sds] == ["tr1", "a1"], "annotation order, de-duplicated"
    assert [s.rank for s in sds] == [0, 1]
    assert {s.source for s in sds} == {"oracle"}
    assert {s.score for s in sds} == {1.0}, "no fabricated ranking score"

    # ---------------------------------------------------------------- injected embedder
    calls = []

    def fake_embed(texts):
        calls.append(len(texts))
        return [[1.0] + [0.0] * 255 for _ in texts]

    d = build_arm(get_spec("dense"), _item(), graph, 2048, embed_fn=fake_embed)
    assert calls and d.seeds, "embed_fn must actually be used by the dense arm"
    calls.clear()
    build_arm(get_spec("hybrid_pin"), _item(), graph, 2048, embed_fn=fake_embed)
    assert calls, "embed_fn must reach the dense component of a hybrid arm"

    # ---------------------------------------------------------------- fault injection
    _expect(SchemaError, get_spec, "not_a_baseline")
    _expect(SchemaError, get_spec, 7)
    # a spec that names a router it never runs is a mislabelled arm
    _expect(SchemaError, BaselineSpec, "bad", "hybrid", "native", "source_only", "oracle")
    _expect(SchemaError, BaselineSpec, "bad", None, "native", "source_only", "router")
    _expect(SchemaError, BaselineSpec, "bad", "nope", "native", "source_only", "router")
    _expect(SchemaError, BaselineSpec, "bad", "hybrid", "sideways", "source_only", "router")
    _expect(SchemaError, BaselineSpec, "bad", "hybrid", "native", "carrier_maybe", "router")
    _expect(SchemaError, BaselineSpec, "bad", "hybrid", "native", "source_only", "psychic")
    _expect(SchemaError, BaselineSpec, "", "hybrid", "native", "source_only", "router")
    _expect(SchemaError, lambda: BaselineSpec("bad", "hybrid", "native", "source_only",
                                              "router", k=0))
    # router_kwargs is copied, so a later mutation cannot retro-actively redefine an arm
    kw = {"backend": "internal"}
    sp = BaselineSpec("lex", "lexical", "off", "source_only", "router", router_kwargs=kw)
    kw["backend"] = "auto"
    assert sp.router_kwargs == {"backend": "internal"}
    assert build_packet(sp, _item(), graph, 2048).manifest.total_tokens > 0

    # budget guards
    _expect(BudgetError, build_packet, get_spec("last_n"), _item(), graph, -1)
    _expect(BudgetError, build_packet, get_spec("last_n"), _item(), graph, 2.5)
    _expect(BudgetError, full_trace_tail_ids, graph, -5)
    _expect(BudgetError, full_trace_tail_ids, graph, True)

    # item guards
    _expect(SchemaError, build_packet, get_spec("last_n"), {"item_id": "x"}, graph, 100)
    _expect(SchemaError, build_packet, get_spec("last_n"),
            _item(query_mode="telepathy"), graph, 100)
    _expect(SchemaError, build_packet, get_spec("last_n"), _item(query=42), graph, 100)
    # an oracle seed that is not in this graph must raise, not be dropped
    _expect(SchemaError, build_packet, get_spec("oracle_seed_native"),
            _item(gold_seed=["ghost"]), graph, 100)
    # oracle closure with nothing annotated must raise, not fall back to "seeds only"
    _expect(SchemaError, build_packet, get_spec("oracle_dependency"),
            _item(gold_minimal_packet=[], required_sources=[], gold_seed=["a1"]), graph, 100)
    # embed_fn where there is no router / no dense arm
    _expect(SchemaError, build_arm, get_spec("full_trace"), _item(), graph, 100,
            embed_fn=fake_embed)
    _expect(SchemaError, build_arm, get_spec("oracle_seed_native"), _item(), graph, 100,
            embed_fn=fake_embed)
    _expect(SchemaError, build_arm, get_spec("bm25"), _item(), graph, 100, embed_fn=fake_embed)
    _expect(SchemaError, build_arm, "last_n", _item(), graph, 100)
    # graph guards
    _expect(SchemaError, full_trace_tail_ids, None, 100)
    _expect(SchemaError, full_trace_tail_ids, ["not an event"], 100)

    print("baselines.py selfcheck OK: %d arms, full_trace tail=%s, native closure required=%s"
          % (len(specs), full_trace_tail_ids(graph, 60), tuple(sorted(got.closure.required))))


if __name__ == "__main__":
    _selfcheck()
