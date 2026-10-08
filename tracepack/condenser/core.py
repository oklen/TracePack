"""Harness-agnostic recall: (events already dropped from context, current instruction) -> packet.

This is the deliverable's engine.  It imports only this repo, runs on python 3.9, and does not know
what a condenser is -- :mod:`tracepack.condenser.oh_condenser` binds it to a harness.

The pipeline is four steps and each one is a component with a pre-registered reading behind it
(see :mod:`tracepack.condenser.recipe`)::

    seeds   = hybrid(query, forgotten events, k=8)      + a seed tool_call brings its own result
    gate    = can the closure walk >= 2 DEPENDS_ON hops from a seed?
    closure = native typed closure, max_hops=12          (skipped when the gate is shut)
    packet  = evidence_first under a hard budget, cost in the sort key, per-unit ceiling, excerpts

Three functions here (``paired``, ``in_context_ids``, ``scope``) are deliberate copies of
``pilot/tp_serve.py``, which stays pinned to the legacy recipe so every published number still
reproduces byte-for-byte.  This module is the maintained copy; that one is a fossil.
"""
from __future__ import annotations

import time
from dataclasses import dataclass, field

from ..core.assembler import AssemblerConfig, BudgetAssembler
from ..core.closure import ClosureConfig, TypedClosure
from ..core.graph import TraceGraph
from ..core.router import HybridRouter, LexicalRouter, RouterConfig, make_router
from ..core.schema import Seed, SchemaError
from .recipe import RECIPE, Recipe


# ---------------------------------------------------------------- graph helpers


def paired(seeds, graph):
    """A seed tool_call brings its own tool_result (frozen since Phase A, `eval/emit_phase_a.py`)."""
    have = {s.event_id for s in seeds}
    out = list(seeds)
    for s in seeds:
        ev = graph.event(s.event_id)
        if getattr(ev, "kind", "") != "tool_call":
            continue
        for ed in graph.children(s.event_id, "RESULT_OF"):
            rid = ed.src_id
            if rid not in have:
                have.add(rid)
                out.append(Seed(event_id=rid, score=s.score, source=s.source, rank=s.rank,
                                pinned=s.pinned))
    return out


def in_context_ids(graph, query: str) -> set:
    """Events the model already has: the compaction summary, everything after it, and the query.

    Serving those back is pure waste -- the first full pilot run spent two of six packet entries on
    the summary and the question.  In the OpenHands binding this is usually a no-op, because the
    corpus handed in is already exactly the forgotten events; it matters when a caller hands in a
    whole trace.
    """
    ids = list(graph.event_ids)
    ex = set()
    last_sum = None
    for i, eid in enumerate(ids):
        ev = graph.event(eid)
        if ev.kind == "summary" and (ev.meta or {}).get("compaction") == "1":
            last_sum = i
    if last_sum is not None:
        ex.update(ids[last_sum:])
    q = " ".join(query.split())
    for eid in ids:
        ev = graph.event(eid)
        if ev.kind == "user" and " ".join((ev.text or "").split()) == q:
            ex.add(eid)
    return ex


def without(graph, ex: set):
    """A graph restricted to the events not in ``ex`` (edges touching them dropped)."""
    if not ex:
        return graph
    evs = [e for e in graph.events if e.event_id not in ex]
    eds = [d for d in graph.edges if d.src_id not in ex and d.dst_id not in ex]
    return TraceGraph(evs, eds)


def scope(graph, query: str):
    """(searchable graph, ids excluded as already-in-context)."""
    ex = in_context_ids(graph, query)
    return without(graph, ex), ex


def seed_depth(graph, seeds) -> int:
    """How many DEPENDS_ON hops the closure can walk from these seeds (consumer -> source).

    The gate is a statement about *the seeds the packet is actually built from*, so this takes them
    rather than re-deriving them: measured 78/384 (20%) disagreement when the caller re-derived with
    a different router, i.e. one gate reading in five was about a different retrieval than the packet
    (`pilot/hops_packets.py`).  With `router="hybrid"` that mistake would be the common case.
    """
    dep, res_call = {}, {}
    for d in graph.edges:
        if d.edge_type == "DEPENDS_ON":
            dep.setdefault(d.src_id, []).append(d.dst_id)
        elif d.edge_type == "RESULT_OF":
            res_call[d.src_id] = d.dst_id
    best = 0
    for s in seeds:
        seen = {s.event_id}
        frontier = [(s.event_id, 0)]
        while frontier:
            cur, h = frontier.pop()
            if h >= 8:
                continue
            for src in dep.get(cur, []):
                if src in seen:
                    continue
                seen.add(src)
                best = max(best, h + 1)
                call = res_call.get(src)
                if call and call not in seen:
                    seen.add(call)
                    frontier.append((call, h + 1))
            call = res_call.get(cur)
            if call and call not in seen:
                seen.add(call)
                frontier.append((call, h))
    return best


# ---------------------------------------------------------------- result


@dataclass(frozen=True)
class Recall:
    """What one recall produced, plus everything needed to audit it after the fact.

    ``text`` is empty when there was nothing to serve; callers should treat empty as "inject
    nothing", never as an error.
    """

    text: str
    tokens: int
    event_ids: tuple = ()
    gate_open: bool = False
    depth: int = 0
    incomplete: bool = False
    #: required evidence the budget could not fit.  Declared, never silent -- contract #1/#4.
    missing_required: tuple = ()
    n_events: int = 0
    n_seeds: int = 0
    n_excluded: int = 0
    ms: int = 0
    recipe: Recipe = field(default=RECIPE)

    def __bool__(self) -> bool:
        return bool(self.text)

    def text_sha1(self) -> str:
        """Digest of the served text.  Logged instead of the text itself: it makes a live packet
        checkable against an offline replay of the same archive (PLAN_condenser G4a) without writing
        trace content to a shared disk."""
        import hashlib
        return hashlib.sha1(self.text.encode("utf-8")).hexdigest()

    def as_dict(self) -> dict:
        return {"sha1": self.text_sha1(),
                "tokens": self.tokens, "n_entries": len(self.event_ids), "gate_open": self.gate_open,
                "depth": self.depth, "incomplete": self.incomplete,
                "n_missing_required": len(self.missing_required), "n_events": self.n_events,
                "n_seeds": self.n_seeds, "n_excluded": self.n_excluded, "ms": self.ms,
                "router": self.recipe.router, "max_hops": self.recipe.max_hops,
                "budget": self.recipe.budget, "served": list(self.event_ids)}


EMPTY = Recall(text="", tokens=0)


# ---------------------------------------------------------------- the engine


class TracePackRecall:
    """Build a verbatim packet for ``query`` out of events the harness has dropped.

    Stateless and deterministic: same graph + same query + same recipe -> byte-identical packet.
    Hold one per process; it costs nothing to keep.
    """

    def __init__(self, recipe: Recipe = RECIPE):
        if not isinstance(recipe, Recipe):
            raise SchemaError("recipe must be a Recipe, got %r" % (type(recipe).__name__,))
        self.recipe = recipe
        self._router = self._build_router(recipe)

    @staticmethod
    def _build_router(recipe: Recipe):
        """Build the seed router with the BM25 backend PINNED.

        `LexicalRouter`'s default is `backend="auto"`, which silently picks rank_bm25 when it happens
        to be installed.  That made the packet depend on the environment: measured, the two backends
        disagree on 44% of seed sets and 40% of packets (RESULTS_condenser §7).  It was found by G4a
        reporting a mismatch between a live packet and its offline replay on a machine where one had
        rank_bm25 and the other did not.
        """
        cfg = RouterConfig(k=recipe.k)
        if recipe.router == "lexical":
            return LexicalRouter(cfg, backend=recipe.bm25_backend)
        if recipe.router == "hybrid":
            return HybridRouter(cfg, lexical=LexicalRouter(cfg, backend=recipe.bm25_backend))
        return make_router(recipe.router, cfg)

    # -- the four steps, each overridable without touching the others ---------------------------

    def seeds_for(self, query: str, graph):
        seeds = list(self._router.retrieve(query, graph, self.recipe.k))
        if self.recipe.pair_call_and_result:
            seeds = paired(seeds, graph)
        return seeds

    def gate(self, graph, seeds) -> "tuple[bool, int]":
        depth = seed_depth(graph, seeds)
        return depth >= self.recipe.gate_hops, depth

    def assembler_config(self, gate_open: bool, budget: int) -> AssemblerConfig:
        r = self.recipe
        header = r.header % budget if "%d" in (r.header or "") else r.header
        if not gate_open:
            # A shut gate means the CLOSURE is worth 0, not that the packet is.  Verbatim retrieval
            # on its own was +16.7 points over the native arm in round 2.
            return AssemblerConfig(repr_policy=r.repr_policy, pack=r.pack_closed, header=header)
        return AssemblerConfig(repr_policy=r.repr_policy, pack=r.pack_open,
                               evidence_hops=r.evidence_hops, evidence_share=r.evidence_share,
                               cost_order=r.cost_order, unit_cap_share=r.unit_cap_share,
                               excerpt=r.excerpt, header=header)

    # -- one call -------------------------------------------------------------------------------

    def recall(self, graph, query: str, budget: "int | None" = None,
               already_scoped: bool = True) -> Recall:
        """``graph``: a TraceGraph over the events the harness dropped.

        ``already_scoped=False`` asks for in-context events to be removed first (pass a whole trace).
        """
        if not isinstance(query, str) or not query.strip():
            raise SchemaError("query must be a non-empty str")
        t0 = time.time()
        r = self.recipe
        budget = int(r.budget if budget is None else budget)
        ex: set = set()
        if not already_scoped:
            graph, ex = scope(graph, query)
        n_events = len(list(graph.events))
        if n_events == 0:
            return Recall(text="", tokens=0, n_excluded=len(ex), recipe=r,
                          ms=int(1000 * (time.time() - t0)))

        seeds = self.seeds_for(query, graph)
        gate_open, depth = self.gate(graph, seeds)
        mode = "native" if gate_open else "off"
        closure = TypedClosure(ClosureConfig(mode=mode, max_hops=r.max_hops)).close(
            query, seeds, graph, query_mode="lookup")
        packet = BudgetAssembler(self.assembler_config(gate_open, budget)).assemble(
            query, closure, graph, budget, seeds=seeds, query_mode="lookup")
        m = packet.manifest
        return Recall(text=packet.context if m.entries else "", tokens=m.total_tokens,
                      event_ids=tuple(e.event_id for e in m.entries), gate_open=gate_open,
                      depth=depth, incomplete=bool(m.incomplete),
                      missing_required=tuple(m.missing_required), n_events=n_events,
                      n_seeds=len(seeds), n_excluded=len(ex), recipe=r,
                      ms=int(1000 * (time.time() - t0)))


def _selfcheck() -> None:
    """A small hand-built trace where the answer is two hops from anything the query matches."""
    from ..adapters.base import build_graph
    from ..core.schema import TraceEdge, TraceEvent

    def ev(eid, kind, text, ts, **kw):
        return TraceEvent(event_id=eid, kind=kind, text=text, timestamp=ts,
                          token_cost=max(1, len(text) // 4), **kw)

    # chain: the agent restates a job id -> the job id came from a status call -> whose output
    # quotes the port that only the original deploy log ever printed.
    events = [
        ev("e1", "tool_call", "bash {\"cmd\": \"deploy --emit\"}", 1, tool_call_id="c1",
           atomic_group="g1"),
        ev("e2", "tool_result", "deploy finished\nlistening on port 8813\njob id 4417", 2,
           tool_call_id="c1", atomic_group="g1"),
        ev("e3", "tool_call", "bash {\"cmd\": \"status 4417\"}", 3, tool_call_id="c2",
           atomic_group="g2"),
        ev("e4", "tool_result", "job id 4417 healthy, bound to port 8813", 4, tool_call_id="c2",
           atomic_group="g2"),
        ev("e5", "assistant", "status for job id 4417 looks fine", 5),
    ]
    edges = [
        TraceEdge(src_id="e2", dst_id="e1", edge_type="RESULT_OF", predicate="position",
                  provenance="native"),
        TraceEdge(src_id="e4", dst_id="e3", edge_type="RESULT_OF", predicate="position",
                  provenance="native"),
        TraceEdge(src_id="e3", dst_id="e2", edge_type="DEPENDS_ON", predicate="value",
                  provenance="native"),
        TraceEdge(src_id="e5", dst_id="e4", edge_type="DEPENDS_ON", predicate="value",
                  provenance="native"),
    ]
    g = build_graph(events, edges)

    rec = TracePackRecall()
    out = rec.recall(g, "which job id did the status check use")
    assert out.gate_open and out.depth >= 2, (out.gate_open, out.depth)
    assert "8813" in out.text, "the two-hop source never made it into the packet"
    assert out.tokens <= RECIPE.budget

    # determinism: same inputs, byte-identical packet
    assert rec.recall(g, "which job id did the status check use").text == out.text

    # a shut gate still serves a packet -- that is the design call, so it gets a test
    flat = build_graph(events, [e for e in edges if e.edge_type != "DEPENDS_ON"])
    shut = rec.recall(flat, "which job id did the status check use")
    assert not shut.gate_open and shut.text, "a shut gate must still serve verbatim retrieval"

    # empty corpus -> empty recall, never an exception
    assert not TracePackRecall().recall(build_graph([], []), "anything").text

    print("core selfcheck ok  gate=%s depth=%d tokens=%d entries=%d"
          % (out.gate_open, out.depth, out.tokens, len(out.event_ids)))


if __name__ == "__main__":
    _selfcheck()
