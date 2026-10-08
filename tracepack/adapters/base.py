"""tracepack.adapters.base -- the adapter contract (proposal §4.2, §7.1) and its round-trip audit.

An adapter is the *only* place in TracePack that is allowed to know a framework's native
transcript format.  Proposal §7.1 fixes its job at exactly two functions::

    normalize(native_trace) -> TraceGraph        # native format  -> frozen IR
    render(packet)          -> NativeContext     # IR packet      -> what the reader sees

and forbids everything else inside it: no retrieval, no closure rules, no LLM-based dependency
inference, no carrier verification, no business prompts.  This module therefore contains no
policy at all -- it holds the Protocol, one construction helper, and the audit that decides
whether a concrete adapter kept its promise.

Design decisions the rest of the package depends on
---------------------------------------------------

* **`build_graph` is the single construction point.**  Adapters do not call `TraceGraph`
  directly; they hand `(events, edges)` here.  `validate_events_edges` runs first, so the two
  silent-corruption modes (duplicate `event_id`, edge pointing at a missing event) are rejected
  by the adapter layer with the schema's own error, before an indexed graph is ever built.

* **Round-trip is checked against an *independent* declaration, not against the conversion.**
  Test contract #8 ("every native id survives") is worthless if the expected id set is derived
  from the graph the conversion just produced -- that is a tautology and passes even when the
  converter drops half the transcript.  So an adapter may additionally implement two optional
  methods (`NativeTraceIntrospection` below):

      native_ids(native_trace)            -> ids the adapter PROMISES to preserve as native_ref
      expected_strong_edges(native_trace) -> (src_native_id, dst_native_id, edge_type) triples

  computed by a second, simpler pass over the native rows.  `roundtrip_check` compares the two
  paths.  That is what makes it a real audit: an adapter that pairs tool results by *order*
  instead of by id produces a perfectly valid graph and is still caught, because the declared
  `RESULT_OF` pairs are id-based (see `claude_code.py::_selfcheck`, and DESIGN_FROZEN §0: order
  pairing mismatches 3.5% of pairs overall and up to 68% within a single session).
  An adapter that implements neither method still gets the structural half of the audit, and
  the returned stats say so via `native_ids_source`.

* **`roundtrip_check` raises, it does not warn.**  A silent "0 ids survived" is exactly the
  failure this audit exists to prevent, so `strict=True` (the default) raises `AdapterError`.
  `strict=False` returns the same stats dict with `ok=False` for callers that want to report
  many transcripts at once.

* **The native trace is materialized once.**  `native_trace` may be a generator; calling
  `native_ids` and then `normalize` on it would silently hand the second call an exhausted
  iterator (and "0 events" would look like an adapter bug).  Anything that is not a path is
  turned into a list before use.

* **Errors are `TracePackError` subclasses.**  `AdapterError` is defined here rather than in the
  frozen schema; callers that do not care about the layer catch `TracePackError`.

Pure and dependency-free: no clock, no RNG, no network, no numpy/torch.  The only I/O is an
adapter opening the path a caller passed it.
"""
from __future__ import annotations

import gzip
import os
from collections import Counter
from typing import Iterable, Protocol, Sequence, runtime_checkable

try:  # normal import path
    from ..core.schema import (
        STRONG_EDGES,
        MemoryPacket,
        SchemaError,
        TraceEdge,
        TraceEvent,
        TracePackError,
        validate_events_edges,
    )
    from ..core.graph import TraceGraph
except ImportError:  # pragma: no cover - direct execution: `python3 tracepack/adapters/base.py`
    import sys

    sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))
    from tracepack.core.schema import (  # type: ignore[no-redef]
        STRONG_EDGES,
        MemoryPacket,
        SchemaError,
        TraceEdge,
        TraceEvent,
        TracePackError,
        validate_events_edges,
    )
    from tracepack.core.graph import TraceGraph  # type: ignore[no-redef]

__all__ = [
    "AdapterError",
    "TraceAdapter",
    "NativeTraceIntrospection",
    "build_graph",
    "materialize_native",
    "roundtrip_check",
]

#: how many offending ids a failure message / stats dict carries before truncating
_MAX_REPORTED = 12


class AdapterError(TracePackError):
    """Raised by any adapter guard and by `roundtrip_check`."""


# ---------------------------------------------------------------- the contract


@runtime_checkable
class TraceAdapter(Protocol):
    """Proposal §4.2.  Two methods, no policy (§7.1)."""

    def normalize(self, native_trace) -> "TraceGraph":
        """Native transcript (a path, or already-parsed rows) -> the frozen IR.

        Must be pure and deterministic: same bytes in, same graph out, including event order,
        event ids and edge order.  Must never invent an edge it cannot ground in a native
        field; `TraceEdge.provenance` says which field ("native") or which heuristic
        ("inferred") licensed it.
        """

    def render(self, packet: MemoryPacket) -> str:
        """`MemoryPacket` -> the native context string handed to the reader.

        The assembler already decided *what* is in the packet and in which order; render must
        not re-select, re-order, or truncate -- doing so would break the budget contract
        (§3.4) and the packet hash (§3.5).
        """


@runtime_checkable
class NativeTraceIntrospection(Protocol):
    """Optional capability that upgrades `roundtrip_check` from structural to real (see above)."""

    def native_ids(self, native_trace) -> "tuple[str, ...]":
        """Ids the adapter promises will appear as some event's `native_ref`."""

    def expected_strong_edges(self, native_trace) -> "tuple[tuple[str, str, str], ...]":
        """`(src_native_id, dst_native_id, edge_type)` triples the adapter promises to emit."""


# ---------------------------------------------------------------- construction


def open_trace(path):
    """Text handle for a trajectory file; `.gz` is read transparently (the pilot keeps its exports compressed
    so that an agent grepping the filesystem cannot read them as text)."""
    p = os.fspath(path)
    if p.endswith(".gz"):
        return gzip.open(p, "rt", encoding="utf-8", errors="replace")
    return open(p, encoding="utf-8", errors="replace")


def build_graph(events: Sequence[TraceEvent], edges: Sequence[TraceEdge]) -> TraceGraph:
    """The one place adapters turn `(events, edges)` into a `TraceGraph`.

    Validates first (`validate_events_edges`), so duplicate ids and dangling edges surface as
    `SchemaError` from the adapter layer rather than as a confusing index error later.
    """
    ev = tuple(events)
    ed = tuple(edges)
    for e in ev:
        if not isinstance(e, TraceEvent):
            raise AdapterError("build_graph: not a TraceEvent: %r" % (e,))
    for e in ed:
        if not isinstance(e, TraceEdge):
            raise AdapterError("build_graph: not a TraceEdge: %r" % (e,))
    validate_events_edges(ev, ed)   # raises SchemaError on the two corrupting mistakes
    return TraceGraph(ev, ed)


def materialize_native(native_trace):
    """Return something both `native_ids` and `normalize` can consume twice.

    Paths are returned unchanged (re-reading a file is cheap and keeps normalize streaming);
    anything else is listed, because a one-shot generator would make the second consumer see an
    empty trace and report it as an adapter bug.
    """
    if isinstance(native_trace, (str, bytes, os.PathLike)):
        return native_trace
    if isinstance(native_trace, (list, tuple)):
        return native_trace
    if isinstance(native_trace, Iterable):
        return list(native_trace)
    raise AdapterError("native_trace is neither a path nor an iterable of rows: %r"
                       % (type(native_trace).__name__,))


# ---------------------------------------------------------------------- audit


def roundtrip_check(adapter, native_trace, *, strict: bool = True) -> dict:
    """normalize -> every native id survived -> every declared strong edge survived -> stats.

    Returns a stats dict (always); raises `AdapterError` on failure when `strict`.

    Checked, in order:

    1. `normalize` returns a `TraceGraph` (so `build_graph`'s validation has run: no duplicate
       ids, no dangling edges);
    2. every id the adapter declared via `native_ids` appears as some event's `native_ref`
       (test contract #8).  Without that method the check degrades to "every event carries a
       `native_ref`", and `native_ids_source` reports `"graph"` so nobody mistakes the weaker
       result for the real one;
    3. every strong edge in the graph resolves to two events that exist and carry native refs
       (a strong edge whose endpoints lost their provenance is unauditable, §3.5);
    4. every triple the adapter declared via `expected_strong_edges` is present in the graph,
       mapped through `native_ref`.  This is the id-vs-order pairing trap (DESIGN_FROZEN §0);
    5. atomic groups are intact: a `RESULT_OF` pair must share one `atomic_group` (§3.4), since
       a group that got split cannot be packed all-or-nothing.
    """
    if not hasattr(adapter, "normalize") or not hasattr(adapter, "render"):
        raise AdapterError("object is not a TraceAdapter (needs normalize + render): %r"
                           % (type(adapter).__name__,))

    native = materialize_native(native_trace)
    graph = adapter.normalize(native)
    if not isinstance(graph, TraceGraph):
        raise AdapterError("normalize must return a TraceGraph, got %r"
                           % (type(graph).__name__,))

    events = tuple(graph.events)
    edges = tuple(graph.edges)
    by_id = {e.event_id: e for e in events}

    ref_of = {e.event_id: e.native_ref for e in events}
    refs_present = set(r for r in ref_of.values() if r is not None)

    failures = []

    # (2) native ids ---------------------------------------------------------
    declared_ids = None
    if hasattr(adapter, "native_ids"):
        declared_ids = tuple(adapter.native_ids(native))
    if declared_ids is not None:
        missing_ids = sorted(set(declared_ids) - refs_present)
        n_expected = len(set(declared_ids))
        ids_source = "adapter"
    else:
        missing_ids = sorted(e.event_id for e in events if not e.native_ref)
        n_expected = len(events)
        ids_source = "graph"
    if missing_ids:
        failures.append("%d native id(s) did not survive normalize: %s"
                        % (len(missing_ids), missing_ids[:_MAX_REPORTED]))

    # (3) strong edges are resolvable ---------------------------------------
    strong = tuple(e for e in edges if e.edge_type in STRONG_EDGES)
    unresolved = [(e.src_id, e.dst_id, e.edge_type) for e in strong
                  if e.src_id not in by_id or e.dst_id not in by_id]
    if unresolved:   # build_graph should have caught this; belt and braces
        failures.append("%d strong edge(s) point at missing events: %s"
                        % (len(unresolved), unresolved[:_MAX_REPORTED]))
    unprovenanced = [(e.src_id, e.dst_id, e.edge_type) for e in strong
                     if ref_of.get(e.src_id) is None or ref_of.get(e.dst_id) is None]

    # (4) declared strong edges survived ------------------------------------
    declared_strong = None
    missing_strong = []
    if hasattr(adapter, "expected_strong_edges"):
        declared_strong = tuple(adapter.expected_strong_edges(native))
        present = set()
        for e in strong:
            s, d = ref_of.get(e.src_id), ref_of.get(e.dst_id)
            if s is not None and d is not None:
                present.add((s, d, e.edge_type))
        missing_strong = sorted(set(tuple(t) for t in declared_strong) - present)
        if missing_strong:
            failures.append("%d declared strong edge(s) missing from the graph: %s"
                            % (len(missing_strong), missing_strong[:_MAX_REPORTED]))

    # (5) atomic groups intact ----------------------------------------------
    split_groups = []
    for e in strong:
        if e.edge_type != "RESULT_OF":
            continue
        a, b = by_id.get(e.src_id), by_id.get(e.dst_id)
        if a is None or b is None:
            continue
        if a.atomic_group is None or a.atomic_group != b.atomic_group:
            split_groups.append((e.src_id, e.dst_id))
    if split_groups:
        failures.append("%d RESULT_OF pair(s) do not share an atomic_group: %s"
                        % (len(split_groups), split_groups[:_MAX_REPORTED]))

    kinds = Counter(e.kind for e in events)
    edge_types = Counter(e.edge_type for e in edges)
    provenance = Counter(e.provenance for e in edges)

    stats = {
        "ok": not failures,
        "failures": tuple(failures),
        "adapter": type(adapter).__name__,
        "n_events": len(events),
        "n_edges": len(edges),
        "n_strong_edges": len(strong),
        "edge_types": dict(sorted(edge_types.items())),
        "edge_provenance": dict(sorted(provenance.items())),
        "kinds": dict(sorted(kinds.items())),
        "native_ids_source": ids_source,
        "n_native_ids_expected": n_expected,
        "n_native_ids_present": len(refs_present),
        "missing_native_ids": tuple(missing_ids[:_MAX_REPORTED]),
        "n_declared_strong": 0 if declared_strong is None else len(set(map(tuple, declared_strong))),
        "missing_strong": tuple(missing_strong[:_MAX_REPORTED]),
        "n_strong_without_provenance": len(unprovenanced),
        "n_atomic_groups": len({e.atomic_group for e in events if e.atomic_group}),
        "total_tokens": sum(e.token_cost for e in events),
    }
    if strict and failures:
        raise AdapterError("roundtrip_check failed for %s: %s"
                           % (stats["adapter"], " | ".join(failures)))
    return stats


# ------------------------------------------------------------------ selfcheck


def _expect(exc, fn, *a, **kw) -> None:
    try:
        fn(*a, **kw)
    except exc:
        return
    except Exception as e:  # noqa: BLE001 - the point is that it must be *this* type
        raise AssertionError("expected %s, got %s: %s" % (exc.__name__, type(e).__name__, e))
    raise AssertionError("expected %s, nothing raised" % exc.__name__)


def _selfcheck() -> None:
    """Tiny two-call/two-result trace + fault injection (an adapter that pairs by order)."""

    # rows: (native_id, kind, tool_id, text)
    ROWS = [
        ("n0", "user", None, "find the config"),
        ("n1", "call", "T1", "grep config"),
        ("n2", "call", "T2", "grep secret"),
        ("n3", "result", "T2", "secret=hunter2"),     # T2 answers FIRST -- order != id
        ("n4", "result", "T1", "config=/etc/app.yml"),
    ]

    KIND = {"user": "user", "call": "tool_call", "result": "tool_result"}

    class ToyAdapter:
        """Pairs results to calls BY ID (correct)."""

        pair_by_order = False

        def normalize(self, native_trace):
            rows = list(native_trace)
            events, edges = [], []
            call_of = {}                       # tool_id -> event_id
            calls_in_order, results_in_order = [], []
            for i, (nid, kind, tid, text) in enumerate(rows):
                eid = "e%02d" % i
                if kind == "call":
                    call_of[tid] = eid
                    calls_in_order.append((eid, tid))
                elif kind == "result":
                    results_in_order.append((eid, tid))
            for n, (eid, tid) in enumerate(results_in_order):
                if self.pair_by_order:
                    # the classic silent corruption: nth result <- nth call
                    tid_used = calls_in_order[n][1]
                else:
                    tid_used = tid
                edges.append((eid, call_of[tid_used]))
            pair_tid = {}
            for n, (eid, tid) in enumerate(results_in_order):
                pair_tid[eid] = calls_in_order[n][1] if self.pair_by_order else tid
            for i, (nid, kind, tid, text) in enumerate(rows):
                eid = "e%02d" % i
                gid = pair_tid.get(eid, tid)
                events.append(TraceEvent(event_id=eid, kind=KIND[kind], text=text, timestamp=i,
                                         step_id="%04d" % i, tool_call_id=gid, native_ref=nid,
                                         token_cost=len(text) // 4,
                                         atomic_group=("tool:%s" % gid) if gid else None))
            edge_objs = [TraceEdge(src_id=s, dst_id=d, edge_type="RESULT_OF", provenance="native")
                         for s, d in edges]
            return build_graph(events, edge_objs)

        def render(self, packet):
            return packet.context

        def native_ids(self, native_trace):
            return tuple(r[0] for r in native_trace)

        def expected_strong_edges(self, native_trace):
            rows = list(native_trace)
            calls = {tid: nid for nid, kind, tid, _ in rows if kind == "call"}
            return tuple((nid, calls[tid], "RESULT_OF")
                         for nid, kind, tid, _ in rows if kind == "result" and tid in calls)

    toy = ToyAdapter()
    assert isinstance(toy, TraceAdapter), "structural Protocol check must accept the toy adapter"
    assert isinstance(toy, NativeTraceIntrospection)

    stats = roundtrip_check(toy, ROWS)
    assert stats["ok"], stats
    assert stats["n_events"] == 5 and stats["n_strong_edges"] == 2, stats
    assert stats["native_ids_source"] == "adapter"
    assert stats["n_native_ids_expected"] == 5 == stats["n_native_ids_present"], stats
    assert stats["n_declared_strong"] == 2 and not stats["missing_strong"], stats
    assert stats["edge_provenance"] == {"native": 2}, stats
    assert stats["n_atomic_groups"] == 2, stats

    # a generator input must survive being consumed twice
    stats_gen = roundtrip_check(toy, (r for r in ROWS))
    assert stats_gen["n_events"] == 5, stats_gen

    # --- fault injection 1: pair by ORDER instead of by id -------------------
    broken = ToyAdapter()
    broken.pair_by_order = True
    g = broken.normalize(ROWS)          # a perfectly valid, internally consistent graph ...
    assert len(g.events) == 5 and len(g.edges) == 2
    _expect(AdapterError, roundtrip_check, broken, ROWS)     # ... that the audit rejects
    soft = roundtrip_check(broken, ROWS, strict=False)
    assert soft["ok"] is False and soft["missing_strong"], soft
    assert len(soft["missing_strong"]) == 2, soft   # both pairs are crossed

    # --- fault injection 2: dropped native ids ------------------------------
    class DroppingAdapter(ToyAdapter):
        def normalize(self, native_trace):
            return ToyAdapter.normalize(self, list(native_trace)[:-1])

    _expect(AdapterError, roundtrip_check, DroppingAdapter(), ROWS)

    # --- fault injection 3: the two graph-corrupting mistakes ---------------
    e1 = TraceEvent(event_id="x", kind="user", text="a", timestamp=0)
    e2 = TraceEvent(event_id="x", kind="user", text="b", timestamp=1)
    _expect(SchemaError, build_graph, [e1, e2], [])
    e3 = TraceEvent(event_id="y", kind="user", text="c", timestamp=2)
    _expect(SchemaError, build_graph, [e1, e3],
            [TraceEdge(src_id="y", dst_id="ghost", edge_type="DEPENDS_ON")])
    _expect(AdapterError, build_graph, ["not an event"], [])

    # --- guards -------------------------------------------------------------
    _expect(AdapterError, roundtrip_check, object(), ROWS)
    _expect(AdapterError, materialize_native, 42)

    class NoIntrospection:
        def normalize(self, native_trace):
            return toy.normalize(native_trace)

        def render(self, packet):
            return packet.context

    weak = roundtrip_check(NoIntrospection(), ROWS)
    assert weak["native_ids_source"] == "graph" and weak["ok"], weak

    print("[base] ok  events=%d edges=%d strong=%d groups=%d tokens=%d"
          % (stats["n_events"], stats["n_edges"], stats["n_strong_edges"],
             stats["n_atomic_groups"], stats["total_tokens"]))
    print("[base] fault injection: order-pairing rejected, dropped ids rejected, "
          "dup ids + dangling edge rejected")
    print("BASE_SELFCHECK_OK")


if __name__ == "__main__":
    _selfcheck()
