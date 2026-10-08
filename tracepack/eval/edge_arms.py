"""tracepack.eval.edge_arms -- swap a graph's DEPENDS_ON edges, and audit the swap.

The closure code is untouched: an alternative edge set is installed by rebuilding the
:class:`TraceGraph` with different edges, so every arm runs the *same* algorithm and the only
thing that differs is which dependencies exist.  That is exactly the contrast the main experiment
could not make (RESULTS_tracepack_main.md §5.1).

Four edge sets:

  ``native``  what the adapter inferred from >=40-char literal overlap (the shipped behaviour)
  ``llm``     gold-blind LLM inference (tracepack/eval/infer_edges.py)
  ``random``  **the control**: the same NUMBER of DEPENDS_ON edges as ``llm``, wired at random
              between the same pair of events-per-distance distribution.  Without it, "more edges
              -> more events served -> more chances the answer is in there" is a confound that
              would masquerade as "better edges help".
  ``none``    DEPENDS_ON removed entirely -- the floor for this axis.

`audit()` scores an edge set against the dataset's own chains *after the fact*: it never feeds
that information back into edge construction.
"""
from __future__ import annotations

import json
import os
import random
import sys
from collections import Counter, defaultdict, deque

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

from tracepack.core.graph import TraceGraph
from tracepack.core.schema import TraceEdge

DEP = "DEPENDS_ON"


def load_edges(path):
    """``{session: [(src, dst, predicate)]}``.

    Keyed by session on purpose: the adapter numbers events by POSITION with no session prefix,
    so ``cc:00001158:000`` exists in every graph.  Loading edges as a flat list silently applies
    one session's dependencies to all ten -- measured: 82,653 inferred edges became 134,212
    applied ones before this was caught.
    """
    out = defaultdict(list)
    n_no_sess = 0
    for line in open(os.path.expanduser(path), encoding="utf-8"):
        try:
            r = json.loads(line)
        except ValueError:
            continue
        s = r.get("session")
        if not s:
            n_no_sess += 1
            continue
        out[s].append((r["src_id"], r["dst_id"], r.get("predicate") or ""))
    if n_no_sess:
        raise SystemExit("%d edges carry no session field -- refusing an edge file that cannot be "
                         "scoped (cross-session contamination)" % n_no_sess)
    return out


def with_edges(graph, mode, llm_edges=None, seed=1234):
    """Return a TraceGraph whose DEPENDS_ON set is replaced according to ``mode``."""
    keep = [e for e in graph.edges if e.edge_type != DEP]
    ids = {e.event_id for e in graph.events}
    rank = {e.event_id: i for i, e in enumerate(graph.events)}

    if mode == "native":
        return graph
    if mode == "none":
        return TraceGraph(graph.events, keep)
    if mode == "llm":
        add = []
        seen = set()
        for src, dst, pred in (llm_edges or ()):
            if src in ids and dst in ids and src != dst and (src, dst) not in seen:
                if rank[dst] >= rank[src]:          # parent must precede child
                    continue
                seen.add((src, dst))
                add.append(TraceEdge(src_id=src, dst_id=dst, edge_type=DEP,
                                     predicate=(pred or "llm")[:80], provenance="inferred"))
        return TraceGraph(graph.events, keep + add)
    if mode == "random":
        # density- AND distance-matched: for every llm edge inside this graph, draw a new child at
        # random and a parent at the SAME index distance.  Matching distance matters: a random
        # edge set that is mostly short-range would be an easier control than the llm set.
        rng = random.Random(seed)
        ev = list(graph.events)
        dists = []
        for src, dst, _ in (llm_edges or ()):
            if src in rank and dst in rank and rank[dst] < rank[src]:
                dists.append(rank[src] - rank[dst])
        add, seen = [], set()
        for d in dists:
            for _ in range(8):
                c = rng.randrange(d, len(ev))
                p = c - d
                pair = (ev[c].event_id, ev[p].event_id)
                if pair not in seen:
                    seen.add(pair)
                    add.append(TraceEdge(src_id=pair[0], dst_id=pair[1], edge_type=DEP,
                                         predicate="random-control", provenance="inferred"))
                    break
        return TraceGraph(graph.events, keep + add)
    raise ValueError("unknown edge mode %r" % (mode,))


def adjacency(graph, types=(DEP, "RESULT_OF")):
    """child -> [parent] once per graph; scanning graph.edges per node is O(E) and these graphs
    have up to 87k edges."""
    out = defaultdict(list)
    for e in graph.edges:
        if e.edge_type in types:
            out[e.src_id].append(e.dst_id)
    return out


def reachable(adj, start, max_hops=3):
    """Ids reachable from ``start`` along the given adjacency within ``max_hops``."""
    seen, q = {start}, deque([(start, 0)])
    while q:
        node, h = q.popleft()
        if h >= max_hops:
            continue
        for p in adj.get(node, ()):
            if p not in seen:
                seen.add(p)
                q.append((p, h + 1))
    return seen


def audit(graph, items, max_hops=3):
    """Can the item's answer events be REACHED from its seed, under this edge set?

    This is the honest scoring of an edge set: it is computed after the fact and never used to
    build edges.  ``coverage`` is what the closure could in principle pull in; ``n_dep`` is what
    it costs.
    """
    adj = adjacency(graph)
    hit = tot = 0
    for it in items:
        s = it["gold_seed"]
        if not graph.has(s):
            continue
        r = reachable(adj, s, max_hops)
        need = set(it["required_sources"])
        tot += 1
        hit += int(need <= r)
    n_dep = sum(1 for e in graph.edges if e.edge_type == DEP)
    return {"reached": hit, "n": tot, "coverage": hit / max(1, tot), "n_dep_edges": n_dep}
