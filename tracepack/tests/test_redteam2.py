"""Regression contracts for red team round 2 (2026-09-02).

Round 1 found seven problems and each got a standing gate.  Round 2's findings are the same
species -- the instrument doing something other than what its name says -- so they get the same
treatment.  Every test here fails on the code as it stood before the round-2 fixes.

  (8)  a compaction carrier MATERIALIZES the whole session, and source-first on an unverified
       carrier therefore makes the whole session `required`  -> pinned, and the diagnostic that
       has to fire is pinned with it
  (9)  the WCR p-value has a hard floor 2^(1-k) set by how many clusters have a non-zero score;
       when the floor exceeds alpha the contrast is undecidable at ANY effect size
  (10) the CI was the convex hull of accepted GRID POINTS, and the grid never contained 0, so it
       could exclude a value its own test accepts
  (12) `gold_in_context` was computed with a bare substring test -- a third matcher for the
       semantics round 1 unified into exactly one
"""
from __future__ import annotations

import os
import re
import sys
import unittest

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

import numpy as np

from tracepack.core.closure import ClosureConfig, TypedClosure
from tracepack.core.graph import TraceGraph
from tracepack.core.schema import TraceEdge
from tracepack.core.textmatch import contains_value
from tracepack.eval.stats import (contrast, crve, informative, p_floor, sign_matrix, wcr_ci,
                                  wcr_p)
from tracepack.tests.fixtures.tiny_trace import event, seed

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))


def _sparse(diffs_per_cluster):
    """{cluster: [per-item differences]} from a list of (n_items, net_sum) pairs."""
    out = {}
    for i, (n, net) in enumerate(diffs_per_cluster):
        vals = [0] * n
        step = 1 if net >= 0 else -1
        for j in range(abs(int(net))):
            vals[j] = step
        out["s%d" % i] = vals
    return out


class Contract09StatisticalRuler(unittest.TestCase):
    """The ruler must declare when it cannot decide, instead of saying 'not significant'."""

    def test_informative_counts_nonzero_cluster_scores(self):
        self.assertEqual(informative(np.array([0.0, -1.0, 0.0, 2.0])), 2)
        self.assertEqual(informative(np.array([0.0, 0.0, 0.0])), 0)

    def test_p_floor_is_two_to_the_one_minus_k(self):
        for k, want in ((1, 1.0), (2, 0.5), (4, 0.125), (6, 0.03125), (10, 0.001953125)):
            self.assertAlmostEqual(p_floor(k), want, places=9)
        self.assertEqual(p_floor(0), 1.0)

    def test_a_huge_one_sided_effect_on_four_clusters_still_cannot_reach_alpha(self):
        # every informative session agrees, and agrees hugely.  The ruler still cannot reject,
        # because with k=4 the smallest attainable p is 0.125.  This is the shape of TracePack's
        # own auto-closure contrast, which the document reported as "p = 0.125, not significant".
        st = contrast(_sparse([(20, 12), (20, 0), (20, 15), (20, 0), (20, 11),
                               (20, 0), (20, 0), (20, 14), (20, 0), (20, 0)]))
        self.assertEqual(st["k_informative"], 4)
        self.assertAlmostEqual(st["p_floor"], 0.125, places=9)
        self.assertFalse(st["decidable"])
        self.assertGreater(st["wcr_p"], 0.05, "a k=4 contrast must never come out significant")

    def test_ten_informative_clusters_are_decidable(self):
        st = contrast(_sparse([(20, 6)] * 10))
        self.assertEqual(st["k_informative"], 10)
        self.assertTrue(st["decidable"])
        self.assertLessEqual(st["wcr_p"], 0.05)


class Contract09IntervalMatchesItsTest(unittest.TestCase):
    """Finding (10): a test-inversion interval must contain every value its test accepts."""

    SHAPES = [
        [(21, -1), (22, 0), (16, -2), (17, 0), (18, -1), (18, -1), (19, 0), (20, 0), (21, 0),
         (18, 0)],                                                   # the real auto-closure shape
        [(19, 2), (18, 0), (20, 1), (17, 0), (21, 3), (16, 0), (18, 1), (20, 0), (19, 1),
         (22, 0)],
        [(20, 5)] * 10,
        [(20, 0)] * 9 + [(20, 4)],
    ]

    def test_zero_is_inside_the_interval_whenever_the_test_accepts_zero(self):
        for shape in self.SHAPES:
            st = contrast(_sparse(shape), with_percentile=False)
            inside = st["wcr_lo"] <= 0.0 <= st["wcr_hi"]
            self.assertEqual(st["wcr_p"] > 0.05, inside,
                             "p=%.4f but CI=[%+.5f,%+.5f] for %r"
                             % (st["wcr_p"], st["wcr_lo"], st["wcr_hi"], shape))

    def test_the_grid_contains_zero_even_when_dhat_is_far_from_it(self):
        d = _sparse([(20, 5)] * 10)
        keys = sorted(d)
        S = np.array([float(sum(d[k])) for k in keys])
        ng = np.array([float(len(d[k])) for k in keys])
        dhat = float(S.sum() / ng.sum())
        se = crve(S, ng, dhat)
        signs = sign_matrix(len(S))
        lo, hi = wcr_ci(S, ng, signs, dhat, se)
        p0 = wcr_p(S, ng, 0.0, signs, dhat, se)
        self.assertEqual(p0 > 0.05, lo <= 0.0 <= hi)

    def test_contrast_refuses_to_return_a_ci_that_contradicts_its_p(self):
        # the guard itself must be live: monkeypatch the interval to a wrong one and require a raise
        import tracepack.eval.stats as st_mod
        real = st_mod.wcr_ci
        st_mod.wcr_ci = lambda *a, **k: (-1.0, -0.5)
        try:
            with self.assertRaises(AssertionError):
                contrast(_sparse([(20, 0)] * 8 + [(20, 1), (20, -1)]), with_percentile=False)
        finally:
            st_mod.wcr_ci = real


def _carrier_graph(n_sources: int) -> TraceGraph:
    """One compaction summary that MATERIALIZES ``n_sources`` earlier events -- the real shape."""
    events = [event("src%d" % i, "tool_result", i, "payload %d" % i, 10)
              for i in range(n_sources)]
    events.append(event("sum", "summary", n_sources, "a compaction summary", 10))
    events.append(event("q", "decision", n_sources + 1, "the decision under test", 10))
    edges = [TraceEdge(src_id="sum", dst_id="src%d" % i, edge_type="MATERIALIZES",
                       provenance="native", predicate="summarised")
             for i in range(n_sources)]
    edges.append(TraceEdge(src_id="q", dst_id="sum", edge_type="DEPENDS_ON",
                           provenance="inferred"))
    return TraceGraph(events, edges)


class Contract08CarrierFanout(unittest.TestCase):
    """Finding (8): source-first on an unverified compaction carrier requires the whole session."""

    def test_one_carrier_seed_makes_every_summarised_event_required(self):
        n = 200
        g = _carrier_graph(n)
        # the router returns the compaction summary itself -- it is long and topical, so it wins
        # lexical retrieval.  Live: 1 of the 8 seeds was a carrier on 5 of 8 sampled items.
        cl = TypedClosure(ClosureConfig(mode="native")).close(
            "what did the run write", [seed("sum")], g, query_mode="lookup")
        # the decision, the summary, and every one of the n events the summary stands in for
        self.assertGreaterEqual(len(cl.required), n,
                                "the carrier blow-up is the documented behaviour; if this now "
                                "collapses, the closure policy changed and every closure number "
                                "in RESULTS must be re-run, not just re-read")

    def test_deleting_the_materializes_edges_collapses_it(self):
        n = 200
        g = _carrier_graph(n)
        g2 = TraceGraph(list(g.events),
                        [e for e in g.edges if e.edge_type != "MATERIALIZES"])
        cl = TypedClosure(ClosureConfig(mode="native")).close(
            "what did the run write", [seed("sum")], g2, query_mode="lookup")
        self.assertLess(len(cl.required), 5,
                        "with MATERIALIZES gone the same closure must be tiny -- that is the "
                        "counterfactual that identifies the carrier as the sole cause")

    def test_the_blow_up_scales_with_fan_out_not_with_hops(self):
        small = TypedClosure(ClosureConfig(mode="native")).close(
            "q", [seed("sum")], _carrier_graph(10), query_mode="lookup")
        big = TypedClosure(ClosureConfig(mode="native")).close(
            "q", [seed("sum")], _carrier_graph(400), query_mode="lookup")
        self.assertGreater(len(big.required) - len(small.required), 380)


class Contract03OneMatcher(unittest.TestCase):
    """Findings (3) and (12): 'is this value in this text' has exactly one implementation."""

    def test_the_boundary_case_that_the_bare_substring_test_got_wrong(self):
        ctx = 'cd /home/b/work/audit_50_20260729/code && head -80 x.py'
        self.assertIn("20260729", ctx)                       # a bare `in` says yes
        self.assertFalse(contains_value(ctx, "20260729"),    # the canonical matcher says no
                         "'20260729' inside 'audit_50_20260729' is not an occurrence of the value")

    def test_no_module_tests_gold_membership_with_a_bare_substring(self):
        pat = re.compile(r'\[["\']gold["\']\]\s+in\s+|gold\s+in\s+\w*\.?context')
        offenders = []
        for dirpath, _dirs, files in os.walk(ROOT):
            if os.sep + "tests" in dirpath:
                continue
            for fn in files:
                if not fn.endswith(".py"):
                    continue
                path = os.path.join(dirpath, fn)
                for i, line in enumerate(open(path, encoding="utf-8"), 1):
                    if line.lstrip().startswith("#"):
                        continue
                    if pat.search(line):
                        offenders.append("%s:%d %s" % (os.path.relpath(path, ROOT), i,
                                                       line.strip()[:70]))
        self.assertEqual(offenders, [],
                         "gold membership must go through core.textmatch.contains_value; a "
                         "second implementation of the same semantics is how round 1 finding "
                         "(3) happened, and round 2 finding (12) is the same hole reopened")


if __name__ == "__main__":
    unittest.main(verbosity=2)
