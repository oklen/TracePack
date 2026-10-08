#!/usr/bin/env python3
"""One builder for every arm in PLAN_baselines.md -- selection x assembly, plus the neighbour window.

Why one file: the arms must differ in exactly the factor under test.  Round 2 already cost us a day
to the opposite mistake ([[cross-arm-same-tool]]), so B and C here are REQUIRED to reproduce
``tp_serve.evidence`` byte-for-byte; ``selfcheck()`` asserts it and refuses to import cleanly if not.

Factors
-------
selection : which records
    bm25            k lexical seeds.  No pairing, no closure.   (= published arm B)
    bm25pair        + a seed tool_call brings its own tool_result
    bm25pair_nb<w>  + every event within w positions of a seed
    closure         paired seeds + the native typed closure       (= published arm C)
    closure_h       the same, seeded with the `hybrid` router      (= arm CX)
assembly  : how they are organised
    chrono            flat chronological                          (= published arm B)
    evidence_first    the legacy TracePack packer                 (= published arm C)
    evidence_first_x  the same packer with the excerpt tier on    (= arm CX)

Neighbours are added as extra *seeds* under ``ClosureConfig(mode="off")``, which makes them
required entries without touching core/: mode="off" sets required_set = seed_ids.
"""
from __future__ import annotations

import os
import re
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(os.path.dirname(HERE))
if ROOT not in sys.path:
    sys.path.insert(0, ROOT)

from tracepack.core.assembler import AssemblerConfig, BudgetAssembler   # noqa: E402
from tracepack.core.closure import ClosureConfig, TypedClosure          # noqa: E402
from tracepack.core.router import RouterConfig, make_router             # noqa: E402
from tracepack.core.schema import Seed                                  # noqa: E402
from tracepack.pilot import tp_serve                                    # noqa: E402

#: Every published round-2 number ran with these two on (gate_deps.py sets the same).
LINK_VALUES = True
EVIDENCE_HOPS = 2

SELECTIONS = ("bm25", "bm25pair", "bm25pair_nb1", "bm25pair_nb2", "bm25pair_nb4",
              "bm25id1", "bm25id2", "bm25id4", "closure", "closure_h")

#: Every selection above seeds with `lexical`, which is what every published number ran with.
#: `closure_h` is the one exception: RESULTS_seeding §9 measured `hybrid` at +38.7 / +38.9 points
#: (WCR p=.016) and adopted it into the recipe, but the pilot call sites stayed pinned to `lexical`
#: to keep those published numbers reproducible.  Naming the seeder per selection is what lets an
#: arm use the adopted recipe without moving the published diagonal.
SEEDERS = {"closure_h": "hybrid"}
ASSEMBLIES = ("chrono", "evidence_first", "evidence_first_x")

#: The arms PLAN_baselines §2 names, as (selection, assembly).
ARMS = {
    "B":  ("bm25", "chrono"),                   # published
    "C":  ("closure", "evidence_first"),        # published
    "CX": ("closure_h", "evidence_first_x"),    # the ADOPTED recipe: hybrid seeding + the excerpt tier
    "CXs": ("closure", "evidence_first_x"),     # ablation: C + the excerpt tier only
    "CXh": ("closure_h", "evidence_first"),     # ablation: C + hybrid seeding only
    "S1": ("bm25", "evidence_first"),           # selection x assembly: off-diagonal
    "S2": ("closure", "chrono"),                # selection x assembly: off-diagonal
    "P1": ("bm25pair_nb1", "chrono"),           # control 1, w=1
    "P2": ("bm25pair_nb2", "chrono"),           # control 1, w=2
    "P4": ("bm25pair_nb4", "chrono"),           # control 1, w=4
    "P0": ("bm25pair", "chrono"),               # control 1 with pairing only (ablation §21 factor 9)
    "S3": ("bm25pair_nb2", "evidence_first"),   # the packer's best shot at the stronger baseline
    "I1": ("bm25id1", "chrono"),                # identifier chasing, 1 round
    "I2": ("bm25id2", "chrono"),                # identifier chasing, 2 rounds
    "I4": ("bm25id4", "chrono"),                # identifier chasing, 4 rounds -- the B2 baseline
    "I4E": ("bm25id4", "evidence_first"),       # ... with the closure packer, for the swap grid
}


# ---------------------------------------------------------------- selection


def neighbours(seeds, graph, w: int):
    """Every event within ``w`` positions of a seed, added as a seed itself.

    Position is the graph's own event order, which for an OpenHands export is the order the
    messages were emitted -- i.e. "nearby events" in the sense the control asks for.
    """
    if w <= 0:
        return list(seeds)
    ids = list(graph.event_ids)
    pos = {eid: i for i, eid in enumerate(ids)}
    have = {s.event_id for s in seeds}
    out = list(seeds)
    for s in list(seeds):
        i = pos.get(s.event_id)
        if i is None:
            continue
        for j in range(max(0, i - w), min(len(ids), i + w + 1)):
            nid = ids[j]
            if nid in have:
                continue
            have.add(nid)
            out.append(Seed(event_id=nid, score=s.score, source="pin", rank=s.rank,
                            pinned=s.pinned))
    return out


# ---------------------------------------------------------------- identifier chasing (arm I*)

#: Generic identifier shapes.  Deliberately NOT tuned to this corpus: hex blobs, uuids, dotted
#: versions, file-ish paths, and long alphanumeric tokens that contain a digit.  A rule written to
#: match `4b7d21c9` and nothing else would make this baseline a straw man, and the whole point of the
#: arm is to be the strongest thing a retrieval system can do WITHOUT a pre-built dependency graph.
_ID_PATTERNS = (
    re.compile(r"\b[0-9a-f]{8}-[0-9a-f]{4}-[0-9a-f]{4}-[0-9a-f]{4}-[0-9a-f]{12}\b"),   # uuid
    re.compile(r"\b[0-9a-f]{6,40}\b"),                                                 # hex blob / sha
    re.compile(r"\b\d+\.\d+(?:\.\d+)*\b"),                                           # dotted version
    re.compile(r"\b[\w./-]+\.[A-Za-z]{1,5}\b"),                                        # path-ish
    re.compile(r"\b(?=[A-Za-z0-9_-]*\d)[A-Za-z0-9_-]{6,}\b"),                          # token with a digit
)
#: Words that look like identifiers but carry no reference (they would match everything).
_ID_STOP = frozenset(("00000000", "11111111", "0.0.0", "1.0.0", "0.0.0.0", "127.0.0.1"))


def identifiers(text: str, cap: int = 64) -> list:
    """Identifier-looking tokens in ``text``.  Order here is arbitrary; ranking happens in
    :func:`id_expand`, which can see the corpus and therefore how selective each token is."""
    out = set()
    for pat in _ID_PATTERNS:
        for m in pat.findall(text or "")[:400]:
            t = m if isinstance(m, str) else m[0]
            if len(t) >= 5 and t.lower() not in _ID_STOP:
                out.add(t)
    return sorted(out)[:cap]


def _df(graph):
    """event-frequency of every identifier in the corpus -- the selectivity signal.

    Ranking identifiers by LENGTH (the obvious first guess) makes this baseline a straw man: a
    repository trace is full of long file paths, and an 8-character key loses to `scripts/probe_
    backends.sh` in every round.  Rarity is the standard retrieval answer and it is not tuned to any
    particular answer: an identifier occurring in two events is a reference, one occurring in thirty
    is vocabulary.
    """
    df = {}
    for ev in graph.events:
        for t in set(identifiers(getattr(ev, "text", "") or "")):
            df[t] = df.get(t, 0) + 1
    return df


_WORD = re.compile(r"[A-Za-z][A-Za-z0-9_]{2,}")
_STOPW = frozenset(("the", "and", "for", "with", "that", "this", "you", "its", "was", "are", "not",
                    "can", "any", "has", "have", "from", "what", "exact", "write", "does", "file",
                    "docs", "them", "their", "rely", "gone", "good", "source", "reason"))


def _words(text: str) -> list:
    return [w.lower() for w in _WORD.findall(text or "") if w.lower() not in _STOPW]


def id_expand(seeds, graph, query: str, rounds: int, per_round: int = 8, add_per_round: int = 6,
              total_cap: int = 24, df_max: int = 8, per_event_cap: int = 40):
    """Query-time dependency chasing WITHOUT a pre-built graph.  The B2 baseline.

    Each round: read identifiers out of the text retrieved so far, look them up by exact match over
    the corpus, add the best few hits.  It reads only text it has already retrieved -- no gold, no
    answer list, no TracePack edge.  On a two-hop chain it can in principle walk
    `tls-apply --plan A` -> the output that printed A -> `tls-plan --id K` -> the record that
    printed K, one round per hop.  Whether it does is the measurement.

    Two ranking choices decide whether this arm is a real baseline or a straw man, so both are
    stated:

    * identifiers are ranked by **whether they sit on a line that talks about the question**, then
      by rarity.  Ranking by LENGTH (the obvious first guess) was measured and is a straw man: a
      repository trace carries dozens of long file paths, and an 8-character key loses every round
      to `scripts/probe_backends.sh`.  Rarity alone is not enough either -- this fixture plants
      decoy hex ids that are just as rare as the real one.
    * among events matching equally many identifiers, the **earliest** wins: a value is produced
      before it is used, so chasing an identifier backwards is the direction that finds a source.

    Caps (4 rounds, 8 identifiers, 6 events per round, 24 total) are fixed here, not per task.
    """
    have = {s.event_id for s in seeds}
    out = list(seeds)
    known = set()
    added = 0
    df = _df(graph)
    qw = set(_words(query))
    for _ in range(max(0, rounds)):
        best = {}
        for s in out:
            ev = graph.event(s.event_id)
            got = 0
            for line in (getattr(ev, "text", "") or "").splitlines():
                near = len(set(_words(line)) & qw)
                for t in identifiers(line, cap=8):
                    if t in known:
                        continue
                    if best.get(t, -1) < near:
                        best[t] = near
                    got += 1
                if got >= per_event_cap:
                    break            # one 900-line file dump must not decide the round
        uniq = [t for t in best if 2 <= df.get(t, 0) <= df_max]
        if not uniq:
            break
        picked = sorted(uniq, key=lambda t: (-best[t], df.get(t, 10 ** 6), -len(t), t))[:per_round]
        known.update(picked)
        scored = []
        for i, ev in enumerate(graph.events):
            if ev.event_id in have:
                continue
            txt = getattr(ev, "text", "") or ""
            n = sum(1 for t in picked if t in txt)
            if n:
                scored.append((-n, i, ev.event_id))
        if not scored:
            continue
        scored.sort()
        gained = 0
        for _, _, eid in scored:
            if gained >= add_per_round or added >= total_cap:
                break
            have.add(eid)
            # Both priorities were measured on the 192 archived corpora, and the baseline is
            # reported at its better one.  rank=0 (chased events packed ahead of the seeds)
            # looks like the fairer choice and is WORSE: it floods the front of the packet and
            # evicts the seed that carried the agent's own restatement of the record
            # (E_served 23.4 -> 20.3%, answer-in-packet 57.3 -> 30.7%).  So: appended.
            out.append(Seed(event_id=eid, score=0.0, source="pin", rank=len(out)))
            gained += 1
            added += 1
        if gained == 0:
            break
    return out


def select(kind: str, query: str, graph, k: int):
    """-> (seeds, closure).  The only place an arm's record set is decided."""
    seeds = list(make_router(SEEDERS.get(kind, "lexical"), RouterConfig(k=k)).retrieve(query, graph, k))
    if kind == "bm25":
        pass
    elif kind in ("closure", "closure_h"):
        seeds = tp_serve.paired(seeds, graph)
    elif kind == "bm25pair":
        seeds = tp_serve.paired(seeds, graph)
    elif kind.startswith("bm25pair_nb"):
        seeds = neighbours(tp_serve.paired(seeds, graph), graph, int(kind[len("bm25pair_nb"):]))
    elif kind.startswith("bm25id"):
        seeds = id_expand(tp_serve.paired(seeds, graph), graph, query,
                          int(kind[len("bm25id"):]))
    else:
        raise ValueError("unknown selection %r (want one of %s)" % (kind, SELECTIONS))
    mode = "native" if kind in ("closure", "closure_h") else "off"
    closure = TypedClosure(ClosureConfig(mode=mode)).close(query, seeds, graph, query_mode="lookup")
    return seeds, closure


# ---------------------------------------------------------------- assembly


def assembler_config(kind: str) -> AssemblerConfig:
    """The two packers, copied from ``tp_serve.evidence`` so the diagonal stays byte-identical.

    ``evidence_first`` is deliberately the LEGACY packer (cost_order off, no unit cap, no excerpt):
    every published TracePack number was produced with it, and an arm that quietly used the fixed
    packer would be comparing against a different C than the one in RESULTS_*.
    """
    if kind == "chrono":
        return AssemblerConfig(repr_policy="source_only", pack="chrono")
    if kind == "evidence_first":
        return AssemblerConfig(repr_policy="source_only", pack="evidence_first",
                               evidence_hops=EVIDENCE_HOPS, evidence_share=0.5,
                               cost_order=False, unit_cap_share=None, excerpt=False)
    if kind == "evidence_first_x":
        # ONE factor away from the line above: the excerpt tier.  Not cost_order, not unit_cap --
        # e80a483 measured those two as NOT better on top-unit loss (69% vs 65%), while the excerpt
        # tier is 2-4x (65% -> 12% real, 100% -> 7% scripted, at exactly this 2048 budget).  Keeping
        # the other two off is what makes a CX-vs-C difference attributable to the excerpt tier.
        return AssemblerConfig(repr_policy="source_only", pack="evidence_first",
                               evidence_hops=EVIDENCE_HOPS, evidence_share=0.5,
                               cost_order=False, unit_cap_share=None, excerpt=True)
    raise ValueError("unknown assembly %r (want one of %s)" % (kind, ASSEMBLIES))


# ---------------------------------------------------------------- one packet


def build(session_file: str, query: str, arm: str, budget: int = 2048, k: int = 8) -> dict:
    """Build one arm's packet over one archived forgotten-events corpus."""
    import time
    if arm not in ARMS:
        raise ValueError("unknown arm %r (want one of %s)" % (arm, sorted(ARMS)))
    sel, asm = ARMS[arm]
    t0 = time.time()
    tp_serve.LINK_VALUES = LINK_VALUES
    tp_serve.EVIDENCE_HOPS = EVIDENCE_HOPS
    full = tp_serve.adapter_for(session_file).normalize(session_file)
    graph, ex, _ = tp_serve.universe(full, query)
    t_graph = time.time()
    seeds, closure = select(sel, query, graph, k)
    packet = BudgetAssembler(assembler_config(asm)).assemble(query, closure, graph, budget,
                                                             seeds=seeds, query_mode="lookup")
    m = packet.manifest
    return {"arm": arm, "selection": sel, "assembly": asm,
            "text": packet.context, "tokens": m.total_tokens, "n_entries": len(m.entries),
            "served": [e.event_id for e in m.entries], "incomplete": bool(m.incomplete),
            "n_events": len(graph.events), "n_seeds": len(seeds), "n_in_context_excluded": len(ex),
            "n_edges_depends": sum(1 for d in graph.edges if d.edge_type == "DEPENDS_ON"),
            # cost: graph construction is RULE-BASED here -- zero LLM calls, which is the number
            # ContextWeaver has to be compared against (PLAN_baselines §5).
            "graph_ms": int(1000 * (t_graph - t0)), "graph_llm_calls": 0, "graph_llm_tokens": 0,
            "ms": int(1000 * (time.time() - t0))}


# ---------------------------------------------------------------- selfcheck


def selfcheck(session_file: str, query: str, budget: int = 2048, k: int = 8) -> None:
    """B and C must reproduce ``tp_serve.evidence`` byte-for-byte, and the factors must bite.

    Without the first assertion this file would be a second ruler, and a cross-arm comparison with
    two rulers is the failure mode [[cross-arm-same-tool]] is about.
    """
    tp_serve.LINK_VALUES = LINK_VALUES
    tp_serve.EVIDENCE_HOPS = EVIDENCE_HOPS
    for arm, mode in (("B", "bm25"), ("C", "tracepack")):
        mine = build(session_file, query, arm, budget, k)
        theirs = tp_serve.evidence(session_file, query, mode, budget, k)
        assert mine["text"] == theirs["text"], (
            "arm %s does not reproduce tp_serve %r byte-for-byte (%d vs %d chars)"
            % (arm, mode, len(mine["text"]), len(theirs["text"])))
        assert mine["served"] == theirs["served"], "arm %s serves a different entry set" % arm

    # the factors have to actually do something, or a null result here means nothing
    b = build(session_file, query, "B", budget, k)
    p0 = build(session_file, query, "P0", budget, k)
    p2 = build(session_file, query, "P2", budget, k)
    assert p0["n_seeds"] >= b["n_seeds"], "pairing removed seeds"
    assert p2["n_seeds"] > p0["n_seeds"], "the neighbour window added nothing (w=2)"
    i4 = build(session_file, query, "I4", budget, k)
    assert i4["n_seeds"] > p0["n_seeds"], "identifier chasing added nothing"
    assert identifiers("id: 4b7d21c9 build 2.14.0 svc/config.py"), "the identifier rule matches nothing"
    print("cw_arms selfcheck ok  B/C byte-identical to tp_serve; seeds B=%d P0=%d P2=%d I4=%d"
          % (b["n_seeds"], p0["n_seeds"], p2["n_seeds"], i4["n_seeds"]))


if __name__ == "__main__":
    import argparse

    ap = argparse.ArgumentParser()
    ap.add_argument("session_file")
    ap.add_argument("--query", default="what did the one-shot record say")
    ap.add_argument("--selfcheck", action="store_true")
    ap.add_argument("--arm", default="")
    a = ap.parse_args()
    if a.selfcheck:
        selfcheck(a.session_file, a.query)
    elif a.arm:
        r = build(a.session_file, a.query, a.arm)
        print({kk: vv for kk, vv in r.items() if kk != "text"})
        print(r["text"][:1200])
    else:
        for arm in sorted(ARMS):
            r = build(a.session_file, a.query, arm)
            print("%-3s %-14s %-14s seeds %3d entries %2d tokens %5d"
                  % (arm, r["selection"], r["assembly"], r["n_seeds"], r["n_entries"], r["tokens"]))
