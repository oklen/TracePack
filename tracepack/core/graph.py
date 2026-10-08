"""tracepack.core.graph -- the indexed trace graph (proposal §3.2, §3.3, §3.4, §4.1/§4.2).

`TraceGraph` is the read-only, fully indexed view of one normalized trace.  It is what the
adapter *produces* (`TraceAdapter.normalize -> TraceGraph`, §4.2) and what the router, the
closure policy, the budget assembler and the profiler all *consume*.  It owns exactly two
jobs: validate the (events, edges) pair once, and answer neighbourhood/lookup questions in
O(1) so that closure can run its fixed-point expansion without ever scanning the edge list.

Design decisions the rest of the package depends on
---------------------------------------------------

* **Edge direction is the schema's, not intuition's.**  `src --EDGE--> dst` means "the current
  event rests on dst" (§3.2), so `tool_result --RESULT_OF--> tool_call`.  Therefore
  `parents(x)` returns the edges with ``src_id == x`` (what x rests on -- the things closure
  must pull in) and `children(x)` returns the edges with ``dst_id == x`` (what rests on x).
  The names follow the *evidence* DAG, not wall-clock order.

* **Chronological order is `(timestamp, event_id)`, computed once.**  `events` is sorted;
  `edges` keeps adapter insertion order.  Ties are broken by `event_id` so two adapters that
  emit the same trace with the same timestamps produce the same packet hash (test contract #1).
  Every ordered accessor (`by_step`, `by_tool_call`, `atomic_group`, `chronological`) reuses
  that single canonical rank -- there is no second sort key anywhere in the package.

* **Exact-duplicate edges are collapsed, but counted.**  Adapters that merge native metadata
  with annotated edges emit the same `(src, dst, type, predicate, provenance)` twice; left
  alone it double-counts in `stats()` and duplicates `ClosureStep`s in the manifest.  They are
  dropped keeping the first occurrence, and `stats()["n_duplicate_edges_dropped"]` reports it,
  so the normalization is auditable instead of silent.

* **SUPERSEDES is resolved lazily, never at construction.**  §3.3 says a *state* query prefers
  the newest valid event, so `latest_state` walks the SUPERSEDES relation **backwards**
  (dst=old -> src=new) to the head of the chain.  A cyclic chain is a corrupt trace, but the
  graph still has to be constructible so a human can inspect it; so the cycle raises
  `SchemaError` from `latest_state` (the query that cannot be answered), not from `__init__`.
  A fork (two events superseding one state) is resolved deterministically to the newest
  terminal by chronological rank rather than raising -- branch-then-correct is a legal trace.

* **Guards are loud.**  Unknown ids, unknown edge types, an empty type filter and `None`
  group keys all raise `SchemaError`.  `types=[]` returning "nothing" and `by_step(None)`
  bucketing every step-less event together are the two silent-corruption modes here.

* **Closure policy lives in closure.py, not here.**  This module exposes typed neighbourhoods
  (`parents(x, REQUIRED_EDGES)`) and the strong-edge subgraph; it deliberately does not know
  what a query mode is, and never follows an edge on its own except in `latest_state`.

Pure and dependency-free: no clock, no RNG, no I/O, no numpy/torch.
"""
from __future__ import annotations

from typing import Iterable, Sequence

try:  # normal import path
    from .schema import (
        EDGE_TYPES,
        SchemaError,
        TraceEdge,
        TraceEvent,
        validate_events_edges,
    )
except ImportError:  # pragma: no cover - direct execution: `python3 tracepack/core/graph.py`
    import os
    import sys

    sys.path.insert(
        0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
    )
    from tracepack.core.schema import (  # type: ignore[no-redef]
        EDGE_TYPES,
        SchemaError,
        TraceEdge,
        TraceEvent,
        validate_events_edges,
    )

__all__ = ["TraceGraph"]

_EDGE_TYPE_SET = frozenset(EDGE_TYPES)
_SUPERSEDES = "SUPERSEDES"


def _order_key(event: TraceEvent) -> tuple[int, str]:
    """The one canonical event order in the package: time, then id to break ties."""
    return (event.timestamp, event.event_id)


def _edge_key(edge: TraceEdge) -> tuple[str, str, str, str | None, str]:
    return (edge.src_id, edge.dst_id, edge.edge_type, edge.predicate, edge.provenance)


def _normalize_types(types) -> tuple[str, ...] | None:
    """Accept ``None`` (= every type), a single edge type, or an iterable of them."""
    if types is None:
        return None
    if isinstance(types, str):
        types = (types,)
    try:
        candidates = list(types)
    except TypeError:
        raise SchemaError(
            "edge type filter must be a string or an iterable of strings: %r" % (types,)
        )
    out: list[str] = []
    for t in candidates:
        if not isinstance(t, str):
            raise SchemaError("edge type filter must contain strings, got %r" % (t,))
        if t not in _EDGE_TYPE_SET:
            raise SchemaError(
                "unknown edge_type filter %r (closed set: %s)" % (t, ", ".join(EDGE_TYPES))
            )
        if t not in out:
            out.append(t)
    if not out:
        # an empty filter silently returning "no dependencies" would make closure look complete
        raise SchemaError("empty edge type filter; pass types=None to mean 'every type'")
    return tuple(out)


class TraceGraph:
    """Immutable indexed view over a normalized trace.

    Attributes
    ----------
    events : tuple[TraceEvent, ...]
        chronological, `(timestamp, event_id)`.
    edges : tuple[TraceEdge, ...]
        adapter insertion order, exact duplicates removed.
    """

    __slots__ = (
        "events",
        "edges",
        "_by_id",
        "_rank",
        "_out",
        "_in",
        "_out_typed",
        "_in_typed",
        "_by_step",
        "_by_tool_call",
        "_by_group",
        "_superseded",
        "_strong",
        "_dupe_edges",
    )

    # ------------------------------------------------------------------ build

    def __init__(self, events: Sequence[TraceEvent], edges: Sequence[TraceEdge]) -> None:
        if events is None or edges is None:
            raise SchemaError("TraceGraph(events, edges): both arguments are required")
        try:
            raw_events = tuple(events)
            raw_edges = tuple(edges)
        except TypeError:
            raise SchemaError("TraceGraph(events, edges): both arguments must be iterable")

        for ev in raw_events:
            if not isinstance(ev, TraceEvent):
                raise SchemaError("not a TraceEvent: %r" % (ev,))
        for ed in raw_edges:
            if not isinstance(ed, TraceEdge):
                raise SchemaError("not a TraceEdge: %r" % (ed,))

        # duplicate ids / dangling edges -- the two mistakes that corrupt every number downstream
        validate_events_edges(raw_events, raw_edges)

        self.events = tuple(sorted(raw_events, key=_order_key))

        kept: list[TraceEdge] = []
        seen_edges: set[tuple] = set()
        dupes = 0
        for ed in raw_edges:
            k = _edge_key(ed)
            if k in seen_edges:
                dupes += 1
                continue
            seen_edges.add(k)
            kept.append(ed)
        self.edges = tuple(kept)
        self._dupe_edges = dupes

        self._by_id: dict[str, TraceEvent] = {}
        self._rank: dict[str, int] = {}
        by_step: dict[str, list[TraceEvent]] = {}
        by_tool_call: dict[str, list[TraceEvent]] = {}
        by_group: dict[str, list[TraceEvent]] = {}
        for i, ev in enumerate(self.events):
            self._by_id[ev.event_id] = ev
            self._rank[ev.event_id] = i
            if ev.step_id:
                by_step.setdefault(ev.step_id, []).append(ev)
            if ev.tool_call_id:
                by_tool_call.setdefault(ev.tool_call_id, []).append(ev)
            if ev.atomic_group:
                by_group.setdefault(ev.atomic_group, []).append(ev)
        # built from self.events, so every bucket is already chronological
        self._by_step = {k: tuple(v) for k, v in by_step.items()}
        self._by_tool_call = {k: tuple(v) for k, v in by_tool_call.items()}
        self._by_group = {k: tuple(v) for k, v in by_group.items()}

        out: dict[str, list[TraceEdge]] = {}
        inn: dict[str, list[TraceEdge]] = {}
        out_typed: dict[str, dict[str, list[TraceEdge]]] = {}
        in_typed: dict[str, dict[str, list[TraceEdge]]] = {}
        superseded: set[str] = set()
        strong: list[TraceEdge] = []
        for ed in self.edges:
            out.setdefault(ed.src_id, []).append(ed)
            inn.setdefault(ed.dst_id, []).append(ed)
            out_typed.setdefault(ed.src_id, {}).setdefault(ed.edge_type, []).append(ed)
            in_typed.setdefault(ed.dst_id, {}).setdefault(ed.edge_type, []).append(ed)
            if ed.edge_type == _SUPERSEDES:
                superseded.add(ed.dst_id)
            if ed.is_strong:
                strong.append(ed)
        self._out = {k: tuple(v) for k, v in out.items()}
        self._in = {k: tuple(v) for k, v in inn.items()}
        self._out_typed = {k: {t: tuple(v) for t, v in d.items()} for k, d in out_typed.items()}
        self._in_typed = {k: {t: tuple(v) for t, v in d.items()} for k, d in in_typed.items()}
        self._superseded = frozenset(superseded)
        self._strong = tuple(strong)

    # ------------------------------------------------------------ small stuff

    def __len__(self) -> int:
        return len(self.events)

    def __iter__(self):
        return iter(self.events)

    def __contains__(self, event_id: object) -> bool:
        return isinstance(event_id, str) and event_id in self._by_id

    def __repr__(self) -> str:
        return "TraceGraph(events=%d, edges=%d)" % (len(self.events), len(self.edges))

    @property
    def event_ids(self) -> tuple[str, ...]:
        """Chronological id list; the canonical iteration order for assembly (§4.3)."""
        return tuple(ev.event_id for ev in self.events)

    # ------------------------------------------------------------ lookups

    def has(self, event_id: str) -> bool:
        return isinstance(event_id, str) and event_id in self._by_id

    def event(self, event_id: str) -> TraceEvent:
        """Never raises `KeyError`: an unknown id is a schema violation, not a dict miss."""
        if not isinstance(event_id, str):
            raise SchemaError("event_id must be a string, got %r" % (event_id,))
        ev = self._by_id.get(event_id)
        if ev is None:
            raise SchemaError("unknown event_id: %s" % event_id)
        return ev

    def _require_id(self, event_id: str, where: str) -> None:
        if not isinstance(event_id, str):
            raise SchemaError("%s: event_id must be a string, got %r" % (where, event_id))
        if event_id not in self._by_id:
            raise SchemaError("%s: unknown event_id: %s" % (where, event_id))

    @staticmethod
    def _require_key(key, where: str) -> str:
        if key is None:
            raise SchemaError(
                "%s: key is None -- an event without this field must not be looked up by it"
                % where
            )
        if not isinstance(key, str):
            raise SchemaError("%s: key must be a string, got %r" % (where, key))
        if not key:
            raise SchemaError("%s: key must be non-empty" % where)
        return key

    # ------------------------------------------------------------ neighbours

    def parents(self, event_id: str, types=None) -> tuple[TraceEdge, ...]:
        """Edges with ``src_id == event_id``: what this event *rests on* (§3.2).

        `closure` expands `parents(x, REQUIRED_EDGES)`.  O(1) dict hit; the returned order is
        the adapter's edge insertion order, which keeps closure steps stable in the manifest.
        """
        self._require_id(event_id, "parents")
        return self._neighbours(self._out, self._out_typed, event_id, types)

    def children(self, event_id: str, types=None) -> tuple[TraceEdge, ...]:
        """Edges with ``dst_id == event_id``: what rests on this event."""
        self._require_id(event_id, "children")
        return self._neighbours(self._in, self._in_typed, event_id, types)

    @staticmethod
    def _neighbours(flat, typed, event_id: str, types) -> tuple[TraceEdge, ...]:
        wanted = _normalize_types(types)
        if wanted is None:
            return flat.get(event_id, ())
        buckets = typed.get(event_id)
        if not buckets:
            return ()
        if len(wanted) == 1:
            return buckets.get(wanted[0], ())
        keep = frozenset(wanted)
        # filter the flat list, not the buckets, so insertion order survives multi-type filters
        return tuple(e for e in flat.get(event_id, ()) if e.edge_type in keep)

    # ------------------------------------------------------------ groupings

    def by_step(self, step_id: str) -> tuple[TraceEvent, ...]:
        """Events of one step, chronological.  Unknown step -> empty tuple (a legal query)."""
        return self._by_step.get(self._require_key(step_id, "by_step"), ())

    def by_tool_call(self, tool_call_id: str) -> tuple[TraceEvent, ...]:
        """The tool call and its result(s) -- the native atomic pair of §3.4."""
        return self._by_tool_call.get(self._require_key(tool_call_id, "by_tool_call"), ())

    def atomic_group(self, group: str) -> tuple[TraceEvent, ...]:
        """All-or-nothing packing unit (§3.4), chronological.

        A group named by a single event is legal (a dangling group is just a group of one);
        an unknown group name returns `()` so the assembler can treat "no group" uniformly.
        """
        return self._by_group.get(self._require_key(group, "atomic_group"), ())

    # ------------------------------------------------------------ supersession

    def superseded_ids(self) -> frozenset[str]:
        """Every `dst` of a SUPERSEDES edge, i.e. every *old* state (§3.3, test contract #7)."""
        return self._superseded

    def latest_state(self, event_id: str) -> str:
        """Head of `event_id`'s supersession chain -- the newest valid event (§3.3).

        Walks SUPERSEDES **backwards** (dst=old -> src=new) transitively.  Returns `event_id`
        itself when nothing supersedes it.  Forks are resolved to the chronologically newest
        terminal; a cycle raises `SchemaError` instead of spinning forever.
        """
        self._require_id(event_id, "latest_state")

        def successors(node: str) -> tuple[str, ...]:
            seen: list[str] = []
            for e in self._in_typed.get(node, {}).get(_SUPERSEDES, ()):
                if e.src_id not in seen:
                    seen.append(e.src_id)
            return tuple(seen)

        cache: dict[str, tuple[str, ...]] = {}
        terminals: list[str] = []
        done: set[str] = set()
        on_path: set[str] = {event_id}
        stack: list[tuple[str, int]] = [(event_id, 0)]
        budget = 4 * (len(self.events) + len(self.edges)) + 16  # belt-and-braces halting bound
        while stack:
            budget -= 1
            if budget < 0:  # pragma: no cover - unreachable while `done` is honoured
                raise SchemaError("latest_state did not terminate from %s" % event_id)
            node, i = stack[-1]
            kids = cache.get(node)
            if kids is None:
                kids = successors(node)
                cache[node] = kids
            if i == 0 and not kids:
                terminals.append(node)
            if i < len(kids):
                stack[-1] = (node, i + 1)
                child = kids[i]
                if child in on_path:
                    raise SchemaError(
                        "SUPERSEDES cycle: %s -> %s (a state cannot supersede its own successor)"
                        % (node, child)
                    )
                if child in done:
                    continue
                on_path.add(child)
                stack.append((child, 0))
            else:
                stack.pop()
                on_path.discard(node)
                done.add(node)

        if not terminals:  # pragma: no cover - a DAG walk always ends on some terminal
            raise SchemaError("no head found for supersession chain of %s" % event_id)
        return max(terminals, key=lambda nid: self._rank[nid])

    # ------------------------------------------------------------ ordering

    def chronological(self, ids: Iterable[str]) -> list[str]:
        """Sort ids into the canonical order, de-duplicated.

        De-duplication is deliberate: closure returns a *set* of evidence and the assembler
        must not emit an event twice (it would be double-charged against the budget, §3.4).
        """
        if ids is None:
            raise SchemaError("chronological(ids): ids is required")
        if isinstance(ids, str):
            raise SchemaError("chronological(ids): pass an iterable of ids, not a single string")
        seen: set[str] = set()
        picked: list[str] = []
        for eid in ids:
            self._require_id(eid, "chronological")
            if eid in seen:
                continue
            seen.add(eid)
            picked.append(eid)
        picked.sort(key=lambda e: self._rank[e])
        return picked

    # ------------------------------------------------------------ views/stats

    def strong_subgraph(self) -> tuple[TraceEdge, ...]:
        """Evidence-bearing edges only (§3.2 whitelist); CONTROL/TEMPORAL are excluded.

        Closure must never follow a weak edge, and adapter round-trip (test contract #8) is
        checked against this view.
        """
        return self._strong

    def stats(self) -> dict:
        """Deterministic, JSON-serialisable counts for the manifest and the eval harness.

        `edge_types` always carries all seven closed-set keys (stable shape across traces);
        `kinds` and `representations` only carry what is present, since those vocabularies are
        adapter-extensible.
        """
        kinds: dict[str, int] = {}
        reprs: dict[str, int] = {}
        total_tokens = 0
        for ev in self.events:
            kinds[ev.kind] = kinds.get(ev.kind, 0) + 1
            total_tokens += ev.token_cost
            for r in ev.representations:
                reprs[r.kind] = reprs.get(r.kind, 0) + 1
        edge_types = {t: 0 for t in EDGE_TYPES}
        n_strong = 0
        for ed in self.edges:
            edge_types[ed.edge_type] += 1
            if ed.is_strong:
                n_strong += 1
        timespan = [self.events[0].timestamp, self.events[-1].timestamp] if self.events else None
        return {
            "n_events": len(self.events),
            "n_edges": len(self.edges),
            "n_strong_edges": n_strong,
            "n_weak_edges": len(self.edges) - n_strong,
            "kinds": {k: kinds[k] for k in sorted(kinds)},
            "edge_types": {t: edge_types[t] for t in sorted(edge_types)},
            "representations": {k: reprs[k] for k in sorted(reprs)},
            "n_steps": len(self._by_step),
            "n_tool_calls": len(self._by_tool_call),
            "n_atomic_groups": len(self._by_group),
            "n_superseded_events": len(self._superseded),
            "n_duplicate_edges_dropped": self._dupe_edges,
            "total_token_cost": total_tokens,
            "timespan": timespan,
        }


# ---------------------------------------------------------------- self check


def _expect(exc, fn, *args, **kwargs) -> None:
    """Fault injection: assert the guard actually fires, and fires as a TracePackError."""
    try:
        fn(*args, **kwargs)
    except exc:
        return
    raise AssertionError("expected %s from %r(%r, %r)" % (exc.__name__, fn, args, kwargs))


def _ev(eid, kind, ts, **kw) -> TraceEvent:
    return TraceEvent(event_id=eid, kind=kind, text="%s text" % eid, timestamp=ts, **kw)


def _selfcheck() -> None:
    # ---- tiny synthetic trace -------------------------------------------------
    #   u1  (user)
    #   tc1 (tool_call,   step s1, tool_call_id t1, group g1)
    #   tr1 (tool_result, step s1, tool_call_id t1, group g1)  --RESULT_OF--> tc1
    #   c1  (summary carrier)  --MATERIALIZES(p=vendor_price)--> tr1
    #   d1  (decision)         --DEPENDS_ON--> tr1, --GROUNDED_IN--> u1
    #   st1 <- st2 <- st3 <- st4  (three SUPERSEDES hops)
    events = [
        _ev("u1", "user", 1),
        _ev("tc1", "tool_call", 2, step_id="s1", tool_call_id="t1", atomic_group="g1"),
        _ev("tr1", "tool_result", 3, step_id="s1", tool_call_id="t1",
            atomic_group="g1", token_cost=40),
        _ev("c1", "summary", 4, token_cost=8, atomic_group="g_solo"),
        _ev("d1", "decision", 5, step_id="s2", token_cost=12),
        _ev("st1", "state", 6),
        _ev("st2", "state", 7),
        _ev("st3", "state", 8),
        _ev("st4", "state", 9),
    ]
    edges = [
        TraceEdge("tr1", "tc1", "RESULT_OF"),
        TraceEdge("d1", "tr1", "DEPENDS_ON"),
        TraceEdge("d1", "u1", "GROUNDED_IN"),
        TraceEdge("c1", "tr1", "MATERIALIZES", predicate="vendor_price"),
        TraceEdge("tc1", "u1", "CONTROL"),
        TraceEdge("d1", "c1", "TEMPORAL"),
        TraceEdge("st2", "st1", "SUPERSEDES"),
        TraceEdge("st3", "st2", "SUPERSEDES"),
        TraceEdge("st4", "st3", "SUPERSEDES"),
        TraceEdge("tr1", "tc1", "RESULT_OF"),  # exact duplicate -> collapsed, counted
    ]
    g = TraceGraph(events, edges)

    # ---- construction / ordering ---------------------------------------------
    assert g.event_ids == ("u1", "tc1", "tr1", "c1", "d1", "st1", "st2", "st3", "st4")
    assert len(g) == 9 and "d1" in g and "nope" not in g
    assert len(g.edges) == 9, "exact duplicate edge must be collapsed"
    assert g.stats()["n_duplicate_edges_dropped"] == 1
    # order is insensitive to input order (determinism, test contract #1)
    g2 = TraceGraph(list(reversed(events)), list(reversed(edges)))
    assert g2.event_ids == g.event_ids
    assert g2.stats()["edge_types"] == g.stats()["edge_types"]

    # ---- lookups & unknown-id rejection --------------------------------------
    assert g.event("d1").kind == "decision"
    assert g.has("d1") and not g.has("ghost")
    _expect(SchemaError, g.event, "ghost")
    _expect(SchemaError, g.parents, "ghost")
    _expect(SchemaError, g.children, "ghost")
    _expect(SchemaError, g.latest_state, "ghost")
    _expect(SchemaError, g.chronological, ["d1", "ghost"])
    _expect(SchemaError, g.event, 7)

    # ---- direction: parents = what the event rests on -------------------------
    assert tuple(e.dst_id for e in g.parents("d1")) == ("tr1", "u1", "c1")  # insertion order
    assert tuple(e.dst_id for e in g.parents("d1", "DEPENDS_ON")) == ("tr1",)
    assert tuple(e.dst_id for e in g.parents("d1", ("GROUNDED_IN", "DEPENDS_ON"))) == \
        ("tr1", "u1"), "multi-type filter must keep edge insertion order, not filter order"
    assert g.parents("u1") == (), "a root event rests on nothing"
    assert tuple(e.src_id for e in g.children("tr1")) == ("d1", "c1")
    assert tuple(e.src_id for e in g.children("tc1", "RESULT_OF")) == ("tr1",)
    assert g.children("tc1", "GROUNDED_IN") == ()
    # type-filter guards
    _expect(SchemaError, g.parents, "d1", ("NOT_AN_EDGE",))
    _expect(SchemaError, g.parents, "d1", ())  # empty filter must not silently mean "nothing"
    _expect(SchemaError, g.children, "tr1", [3])

    # ---- groupings ------------------------------------------------------------
    assert tuple(e.event_id for e in g.by_step("s1")) == ("tc1", "tr1")
    assert tuple(e.event_id for e in g.by_tool_call("t1")) == ("tc1", "tr1")
    assert tuple(e.event_id for e in g.atomic_group("g1")) == ("tc1", "tr1")
    assert tuple(e.event_id for e in g.atomic_group("g_solo")) == ("c1",), \
        "a group named by one event only is legal"
    assert g.atomic_group("g_missing") == () and g.by_step("s_missing") == ()
    _expect(SchemaError, g.by_step, None)  # an event without step_id must not bucket
    _expect(SchemaError, g.atomic_group, "")

    # ---- supersession ---------------------------------------------------------
    assert g.superseded_ids() == frozenset({"st1", "st2", "st3"})
    assert "st4" not in g.superseded_ids(), "the head of the chain is never superseded"
    assert g.latest_state("st1") == "st4", "must follow all three hops"
    assert g.latest_state("st2") == "st4"
    assert g.latest_state("st4") == "st4"
    assert g.latest_state("d1") == "d1", "non-state events are their own latest"
    assert g.latest_state("tc1") == "tc1", "RESULT_OF/DEPENDS_ON must not be followed here"

    # fork: two corrections of one state -> newest terminal, deterministically
    fork = TraceGraph(
        [_ev("a", "state", 1), _ev("b", "state", 2), _ev("c", "state", 3)],
        [TraceEdge("b", "a", "SUPERSEDES"), TraceEdge("c", "a", "SUPERSEDES")],
    )
    assert fork.latest_state("a") == "c"
    fork_rev = TraceGraph(
        [_ev("c", "state", 3), _ev("b", "state", 2), _ev("a", "state", 1)],
        [TraceEdge("c", "a", "SUPERSEDES"), TraceEdge("b", "a", "SUPERSEDES")],
    )
    assert fork_rev.latest_state("a") == "c", "fork resolution must not depend on input order"

    # diamond: a revisit is not a cycle
    diamond = TraceGraph(
        [_ev(x, "state", i + 1) for i, x in enumerate(("a", "b", "c", "d"))],
        [
            TraceEdge("b", "a", "SUPERSEDES"),
            TraceEdge("c", "a", "SUPERSEDES"),
            TraceEdge("d", "b", "SUPERSEDES"),
            TraceEdge("d", "c", "SUPERSEDES"),
        ],
    )
    assert diamond.latest_state("a") == "d"

    # cycle: constructible (so it can be audited) but the query refuses to spin
    cyc = TraceGraph(
        [_ev("x", "state", 1), _ev("y", "state", 2), _ev("z", "state", 3)],
        [
            TraceEdge("y", "x", "SUPERSEDES"),
            TraceEdge("z", "y", "SUPERSEDES"),
            TraceEdge("x", "z", "SUPERSEDES"),
        ],
    )
    _expect(SchemaError, cyc.latest_state, "x")
    _expect(SchemaError, cyc.latest_state, "z")

    # ---- ordering -------------------------------------------------------------
    assert g.chronological(["d1", "u1", "tr1"]) == ["u1", "tr1", "d1"]
    assert g.chronological({"st4", "st1"}) == ["st1", "st4"]
    assert g.chronological(["d1", "d1", "u1"]) == ["u1", "d1"], "must de-duplicate"
    assert g.chronological([]) == []
    _expect(SchemaError, g.chronological, "d1")  # a bare string is 1-char ids, never intended
    # ties are broken by event_id, not by input order
    tied = TraceGraph([_ev("zz", "user", 5), _ev("aa", "user", 5)], [])
    assert tied.chronological(["zz", "aa"]) == ["aa", "zz"]

    # ---- strong subgraph & stats ----------------------------------------------
    strong = g.strong_subgraph()
    assert all(e.is_strong for e in strong)
    assert {e.edge_type for e in strong} == {
        "RESULT_OF", "DEPENDS_ON", "GROUNDED_IN", "MATERIALIZES", "SUPERSEDES"}
    assert not any(e.edge_type in ("CONTROL", "TEMPORAL") for e in strong)
    assert len(strong) == 7 and len(g.edges) - len(strong) == 2

    st = g.stats()
    assert st["n_events"] == 9 and st["n_edges"] == 9
    assert st["n_strong_edges"] == 7 and st["n_weak_edges"] == 2
    assert st["kinds"] == {"decision": 1, "state": 4, "summary": 1,
                           "tool_call": 1, "tool_result": 1, "user": 1}
    assert set(st["edge_types"]) == set(EDGE_TYPES), "stable shape: all closed-set keys present"
    assert st["edge_types"]["SUPERSEDES"] == 3 and st["edge_types"]["CONTROL"] == 1
    assert st["n_steps"] == 2 and st["n_tool_calls"] == 1 and st["n_atomic_groups"] == 2
    assert st["n_superseded_events"] == 3
    assert st["total_token_cost"] == 60 and st["timespan"] == [1, 9]
    import json as _json
    assert _json.dumps(st, sort_keys=True) == _json.dumps(g.stats(), sort_keys=True)

    # ---- empty graph is legal --------------------------------------------------
    empty = TraceGraph([], [])
    assert len(empty) == 0 and empty.stats()["timespan"] is None and empty.strong_subgraph() == ()
    assert empty.superseded_ids() == frozenset() and empty.chronological([]) == []

    # ---- construction-time fault injection -------------------------------------
    _expect(SchemaError, TraceGraph, [_ev("dup", "user", 1), _ev("dup", "user", 2)], [])
    _expect(SchemaError, TraceGraph, [_ev("a", "user", 1)], [TraceEdge("a", "ghost", "DEPENDS_ON")])
    _expect(SchemaError, TraceGraph, [_ev("a", "user", 1)], [TraceEdge("ghost", "a", "DEPENDS_ON")])
    _expect(SchemaError, TraceGraph, ["not an event"], [])
    _expect(SchemaError, TraceGraph, [_ev("a", "user", 1)], ["not an edge"])
    _expect(SchemaError, TraceGraph, None, [])

    print("graph.py selfcheck OK: %r, %d strong edges, chain head=%s"
          % (g, len(g.strong_subgraph()), g.latest_state("st1")))


if __name__ == "__main__":
    _selfcheck()
