"""Phase-2 contracts (PLAN_phase2.md §5): the closure-ranking packs and the byte-identity of the
packs that shipped before them.

  #11  under `tiered` / `evidence_first`, material reached only through weak edges (the compaction
       summary's MATERIALIZES fan-out) never packs ahead of a seed; strong near evidence packs
       where the policy says (right behind its seed / ahead of all seeds); pinned still leads
  #13  `chrono` and `bundle` produce byte-identical packets to the pre-phase-2 assembler
       (fixtures/golden_digests_pre_phase2.json, minted from git HEAD before the change), so the
       old arms' readouts remain comparable without being re-run

The fixture is drawn so the four packs disagree:

    s0 (rank 0) --DEPENDS_ON--> p0r --RESULT_OF--> p0c        strong, hops 1 and 2, atomic pair
    s1 (rank 1) --DEPENDS_ON--> sum --MATERIALIZES--> w1,w2,w3  strong hop 1, then WEAK fan-out
    s2 (rank 2)                                              a seed with no evidence of its own

Timestamps put the fan-out FIRST in time (w1..w3 at t=1..3) so `chrono`, which packs non-seed
required events oldest-first, spends the budget on the fan-out and drops the evidence pair --
the defect measured in RESULTS_tracepack_edges2.md §1.6 (E bucket).
"""
from __future__ import annotations

import hashlib
import json
import os
import sys
import unittest

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

from tracepack.core import assembler as A
from tracepack.core.assembler import PACKS, AssemblerConfig, BudgetAssembler
from tracepack.core.closure import ClosureConfig, TypedClosure
from tracepack.core.graph import TraceGraph
from tracepack.core.router import RouterConfig, make_router
from tracepack.core.schema import SchemaError, TraceEdge
from tracepack.tests.fixtures import tiny_trace as F

HERE = os.path.dirname(os.path.abspath(__file__))
GOLDEN = os.path.join(HERE, "fixtures", "golden_digests_pre_phase2.json")

SEEDS = ("s0", "s1", "s2")
STRONG = ("p0c", "p0r", "sum")
WEAK = ("w1", "w2", "w3")
COST = dict(s0=10, s1=10, s2=10, p0c=5, p0r=30, sum=8, w1=6, w2=6, w3=6)
QUERY = "why did the deploy target change"


def ranking_graph() -> TraceGraph:
    ev = [
        F.event("w1", "tool_result", 1, "old log line one", COST["w1"]),
        F.event("w2", "tool_result", 2, "old log line two", COST["w2"]),
        F.event("w3", "tool_result", 3, "old log line three", COST["w3"]),
        F.event("sum", "summary", 5, "summary of the earlier logs", COST["sum"]),
        F.event("p0c", "tool_call", 19, 'Read {"file_path": "/etc/deploy.yaml"}', COST["p0c"],
                group="grp:p0", tool_call_id="tp0"),
        F.event("p0r", "tool_result", 20, "target: ap-southeast-1\nreplicas: 3", COST["p0r"],
                group="grp:p0", tool_call_id="tp0"),
        F.event("s0", "assistant", 50, "the deploy target is read from deploy.yaml", COST["s0"]),
        F.event("s1", "assistant", 60, "per the summary the logs were clean", COST["s1"]),
        F.event("s2", "assistant", 70, "unrelated remark that also matched", COST["s2"]),
    ]
    edges = [
        TraceEdge("p0r", "p0c", "RESULT_OF", predicate="tool_use_id"),
        TraceEdge("s0", "p0r", "DEPENDS_ON"),
        TraceEdge("s1", "sum", "DEPENDS_ON"),
        TraceEdge("sum", "w1", "MATERIALIZES", predicate="summary_of"),
        TraceEdge("sum", "w2", "MATERIALIZES", predicate="summary_of"),
        TraceEdge("sum", "w3", "MATERIALIZES", predicate="summary_of"),
    ]
    return TraceGraph(ev, edges)


def seeds(pinned=()):
    return [F.seed(s, rank=i, pinned=(s in pinned)) for i, s in enumerate(SEEDS)]


def closure_for(graph, sd):
    return TypedClosure(ClosureConfig(mode="native")).close(QUERY, sd, graph, query_mode="why")


def unit_order(pack, graph, sd, closure, **cfg):
    """White-box: the packing order the assembler will walk, as tuples of member ids."""
    asm = BudgetAssembler(AssemblerConfig(pack=pack, **cfg))
    gv = A._GraphView(graph)
    cands = asm._candidates(closure, gv, asm._best_seed_per_event(sd))
    return [u.member_ids for u in asm._units(cands)]


def flat(order):
    return [eid for members in order for eid in members]


class Contract11EvidenceRanking(unittest.TestCase):
    """PLAN_phase2 WP1: strong near evidence may outrank low seeds; weak fan-out never may."""

    def setUp(self):
        self.graph = ranking_graph()
        self.seeds = seeds()
        self.closure = closure_for(self.graph, self.seeds)
        # the fixture must actually produce what the docstring claims, else the test is vacuous
        req = set(self.closure.required)
        self.assertTrue(set(SEEDS) | set(STRONG) | set(WEAK) <= req, req)

    def test_11a_weak_fanout_never_precedes_a_seed(self):
        for pack in ("tiered", "evidence_first"):
            with self.subTest(pack=pack):
                order = flat(unit_order(pack, self.graph, self.seeds, self.closure))
                last_seed = max(order.index(s) for s in SEEDS)
                first_weak = min(order.index(w) for w in WEAK)
                self.assertGreater(first_weak, last_seed, order)

    def test_11b_tiered_packs_evidence_right_behind_its_own_seed(self):
        order = unit_order("tiered", self.graph, self.seeds, self.closure)
        self.assertEqual(order[:5], [("s0",), ("p0c", "p0r"), ("s1",), ("sum",), ("s2",)], order)
        self.assertEqual(set(flat(order[5:])), set(WEAK))

    def test_11c_evidence_first_packs_all_strong_evidence_ahead_of_every_seed(self):
        order = unit_order("evidence_first", self.graph, self.seeds, self.closure)
        self.assertEqual(order[:5], [("p0c", "p0r"), ("sum",), ("s0",), ("s1",), ("s2",)], order)
        self.assertEqual(set(flat(order[5:])), set(WEAK))

    def test_11d_the_shipped_orders_really_do_misplace_the_evidence(self):
        """Documents the defect the new packs fix; if this starts failing, the goldens moved."""
        chrono = flat(unit_order("chrono", self.graph, self.seeds, self.closure))
        self.assertEqual(chrono[:3], list(SEEDS))
        self.assertLess(chrono.index("w3"), chrono.index("p0r"))      # fan-out before evidence
        bundle = flat(unit_order("bundle", self.graph, self.seeds, self.closure))
        self.assertLess(bundle.index("w3"), bundle.index("s2"))       # fan-out before a seed

    def test_11e_black_box_at_a_budget_that_fits_seeds_plus_evidence(self):
        """73 tokens = s0 + pair + s1 + sum + s2: tiered serves exactly that, chrono spends it on
        the fan-out and drops the pair (first-fit skips what no longer fits)."""
        for pack, want in (("tiered", set(SEEDS) | set(STRONG)),
                           ("chrono", set(SEEDS) | set(WEAK) | {"sum"})):
            with self.subTest(pack=pack):
                pkt = BudgetAssembler(AssemblerConfig(pack=pack)).assemble(
                    QUERY, self.closure, self.graph, 73, seeds=self.seeds, query_mode="why")
                self.assertEqual(set(pkt.event_ids), want, pkt.event_ids)
                self.assertLessEqual(pkt.manifest.total_tokens, 73)

    def test_11f_evidence_first_black_box_drops_the_lowest_seed_before_the_evidence(self):
        pkt = BudgetAssembler(AssemblerConfig(pack="evidence_first")).assemble(
            QUERY, self.closure, self.graph, 63, seeds=self.seeds, query_mode="why")
        self.assertEqual(set(pkt.event_ids), {"p0c", "p0r", "sum", "s0", "s1"}, pkt.event_ids)
        self.assertIn("s2", pkt.manifest.missing_required)

    def test_11g_pinned_still_leads_under_both_packs(self):
        sd = seeds(pinned=("s2",))
        cl = closure_for(self.graph, sd)
        for pack in ("tiered", "evidence_first"):
            with self.subTest(pack=pack):
                self.assertEqual(unit_order(pack, self.graph, sd, cl)[0], ("s2",))

    def test_11h_evidence_hops_bounds_what_counts_as_near(self):
        """With evidence_hops=1 the call (hop 2) is still packed with its result: the atomic
        pair takes the best origin of its members, so atomicity beats the hop cap."""
        order = unit_order("tiered", self.graph, self.seeds, self.closure, evidence_hops=1)
        self.assertEqual(order[:2], [("s0",), ("p0c", "p0r")], order)
        order0 = flat(unit_order("tiered", self.graph, self.seeds, self.closure, evidence_hops=0))
        # hop 0 means nothing is near: every non-seed drops to the weak tier, behind all seeds
        self.assertEqual(order0[:3], list(SEEDS), order0)

    def test_11i_budget_contract_and_atomicity_hold_across_a_sweep(self):
        for pack in ("tiered", "evidence_first"):
            for budget in range(0, 100):
                pkt = BudgetAssembler(AssemblerConfig(pack=pack)).assemble(
                    QUERY, self.closure, self.graph, budget, seeds=self.seeds, query_mode="why")
                self.assertLessEqual(pkt.manifest.total_tokens, budget)
                got = {"p0c", "p0r"} & set(pkt.event_ids)
                self.assertIn(got, (set(), {"p0c", "p0r"}), (pack, budget, pkt.event_ids))

    def test_11j_config_is_validated(self):
        with self.assertRaises(SchemaError):
            AssemblerConfig(pack="ranked")
        with self.assertRaises(SchemaError):
            AssemblerConfig(pack="tiered", evidence_hops=-1)
        with self.assertRaises(SchemaError):
            AssemblerConfig(pack="tiered", evidence_hops=True)
        self.assertEqual(PACKS, ("chrono", "bundle", "tiered", "evidence_first"))


class Contract11bEvidenceShare(unittest.TestCase):
    """The evidence tier may take at most `evidence_share` of the budget ahead of the seeds; the
    rest is deferred behind the seeds, never dropped for being evidence."""

    def setUp(self):
        self.graph = ranking_graph()
        self.seeds = seeds()
        self.closure = closure_for(self.graph, self.seeds)

    def test_11k_share_caps_the_evidence_tier_and_defers_the_rest(self):
        # evidence_first, budget 63, share 0.25 -> cap 16: the 35-token pair exceeds the cap and
        # is deferred; `sum` (8) fits under it.  Seeds then take 30 -> 38 used; the deferred pair
        # (35) no longer fits, the weak fan-out (6 each) does.
        pkt = BudgetAssembler(AssemblerConfig(pack="evidence_first", evidence_share=0.25)).assemble(
            QUERY, self.closure, self.graph, 63, seeds=self.seeds, query_mode="why")
        self.assertEqual(set(pkt.event_ids), {"sum", "s0", "s1", "s2", "w1", "w2", "w3"}, pkt.event_ids)
        self.assertEqual(sorted(pkt.manifest.missing_required), ["p0c", "p0r"])
        # with a budget that has room after the seeds, the deferred pair is still served
        pkt = BudgetAssembler(AssemblerConfig(pack="evidence_first", evidence_share=0.25)).assemble(
            QUERY, self.closure, self.graph, 73, seeds=self.seeds, query_mode="why")
        self.assertEqual(set(pkt.event_ids), {"sum", "s0", "s1", "s2", "p0c", "p0r"}, pkt.event_ids)

    def test_11l_share_one_is_the_uncapped_order(self):
        for pack in ("tiered", "evidence_first"):
            for budget in (45, 63, 73, 91):
                a = BudgetAssembler(AssemblerConfig(pack=pack, evidence_share=1.0)).assemble(
                    QUERY, self.closure, self.graph, budget, seeds=self.seeds, query_mode="why")
                b = BudgetAssembler(AssemblerConfig(pack=pack)).assemble(
                    QUERY, self.closure, self.graph, budget, seeds=self.seeds, query_mode="why")
                self.assertEqual(a.manifest.digest(), b.manifest.digest(), (pack, budget))

    def test_11m_seeds_survive_a_flood_of_evidence_under_the_cap(self):
        """Ten hop-1 parents of seed 0, 20 tokens each, budget 100, share 0.5: at most 50 tokens
        of evidence precede the seeds, so every seed is still served."""
        ev = [F.event("s%d" % i, "assistant", 100 + i, "seed %d" % i, 10) for i in range(3)]
        ev += [F.event("p%d" % j, "tool_result", 10 + j, "parent %d" % j, 20) for j in range(10)]
        edges = [TraceEdge("s0", "p%d" % j, "DEPENDS_ON") for j in range(10)]
        g = TraceGraph(ev, edges)
        sd = seeds()
        cl = closure_for(g, sd)
        capped = BudgetAssembler(AssemblerConfig(pack="evidence_first", evidence_share=0.5)).assemble(
            QUERY, cl, g, 100, seeds=sd, query_mode="why")
        self.assertTrue(set(SEEDS) <= set(capped.event_ids), capped.event_ids)
        self.assertEqual(sum(1 for e in capped.event_ids if e.startswith("p")), 3)   # 60 + 30 = 90
        flooded = BudgetAssembler(AssemblerConfig(pack="evidence_first")).assemble(
            QUERY, cl, g, 100, seeds=sd, query_mode="why")
        self.assertEqual(set(flooded.event_ids) & set(SEEDS), set(), flooded.event_ids)

    def test_11n_share_is_validated(self):
        for bad in (-0.1, 1.5, "half", True):
            with self.assertRaises(SchemaError):
                AssemblerConfig(pack="tiered", evidence_share=bad)


def starving_graph() -> TraceGraph:
    """One fat top-priority unit and a cheap chain whose tail carries the record.

    `fat` is strong hop-1 evidence of the rank-0 seed, so it packs FIRST, and it costs 88 -- 88% of
    the 100 budget below, the same share as the 3,618-of-4,096 unit that produced the real defect.
    The cheap tail is 28 tokens in total, so both budgets have room for all of it.  The
    record `rec` sits 3 DEPENDS_ON hops out -- past the default evidence_hops -- so it packs LAST.
    That is the shape RESULTS_hops §7 found on the 4-hop tasks: the deeper the chain, the later
    the record sorts, so it is the first thing a fat unit starves.
    """
    ev = [
        F.event("fat", "tool_result", 10, "a very long tool output " * 5, 88),
        F.event("c1", "tool_result", 30, "cheap one", 6),
        F.event("c2", "tool_result", 31, "cheap two", 6),
        F.event("rec", "tool_result", 32, "id: 555001", 6),
        F.event("s0", "assistant", 50, "the deploy target is read from deploy.yaml", 10),
    ]
    edges = [
        TraceEdge("s0", "fat", "DEPENDS_ON"),
        TraceEdge("s0", "c1", "DEPENDS_ON"),
        TraceEdge("c1", "c2", "DEPENDS_ON"),
        TraceEdge("c2", "rec", "DEPENDS_ON"),
    ]
    return TraceGraph(ev, edges)


class Contract11cUnitCap(unittest.TestCase):
    """`unit_cap_share`: no single unit may preempt the units below it.

    Without it the packet is NOT monotone in budget -- a fat unit that does not fit a small budget
    (so the packet serves the whole cheap tail, record included) fits a larger one, takes it, and
    starves the tail.  Measured on the hop sweep: E served 38% at 2,048 and 0% at 4,096 (hop4),
    33% -> 21% (hop3), reproduced by both readers (RESULTS_hops §7).
    """

    def setUp(self):
        self.graph = starving_graph()
        self.seeds = [F.seed("s0", rank=0)]
        self.closure = TypedClosure(ClosureConfig(mode="native", max_hops=6)).close(
            QUERY, self.seeds, self.graph, query_mode="why")

    def _served(self, budget, share=None, cost_order=False):
        """evidence_share=1.0 so this class measures `unit_cap_share` alone, not the tier cap."""
        pkt = BudgetAssembler(AssemblerConfig(pack="evidence_first", evidence_share=1.0,
                                              unit_cap_share=share, cost_order=cost_order)).assemble(
            QUERY, self.closure, self.graph, budget, seeds=self.seeds, query_mode="why")
        return set(pkt.event_ids)

    def test_11o_the_defect_reproduces_a_bigger_budget_serves_less(self):
        small, large = self._served(60), self._served(100)
        self.assertIn("rec", small, small)          # fat (88) cannot fit 60, so the tail gets in
        self.assertNotIn("fat", small, small)
        self.assertIn("fat", large, large)          # fat fits 100, takes 88% of it, starves the tail
        self.assertNotIn("rec", large, large)

    def test_11p_the_cap_restores_monotonicity(self):
        for budget in (60, 100, 120):
            self.assertIn("rec", self._served(budget, share=0.8), budget)
        # the fat unit is HELD, not dropped: give the packet room for everything and it is served
        self.assertIn("fat", self._served(200, share=0.8))

    def test_11q_the_legacy_flags_restore_the_shipped_packer(self):
        """Every published TracePack number was produced with cost_order=False, unit_cap_share=None.
        That pair must keep reproducing the pre-fix packet, defect included."""
        self.assertIn("rec", self._served(60, share=None, cost_order=False))
        self.assertNotIn("rec", self._served(100, share=None, cost_order=False))

    def test_11t_the_default_is_the_fixed_packer(self):
        cfg = AssemblerConfig(pack="evidence_first")
        self.assertTrue(cfg.cost_order)
        self.assertEqual(cfg.unit_cap_share, 0.8)
        for budget in (60, 100, 120, 200):
            pkt = BudgetAssembler(AssemblerConfig(pack="evidence_first", evidence_share=1.0)).assemble(
                QUERY, self.closure, self.graph, budget, seeds=self.seeds, query_mode="why")
            self.assertIn("rec", set(pkt.event_ids), budget)

    def test_11r_the_cap_never_breaks_the_budget_or_hides_a_miss(self):
        for budget in (60, 100, 120):
            pkt = BudgetAssembler(AssemblerConfig(pack="evidence_first", evidence_share=1.0,
                                                  unit_cap_share=0.8)).assemble(
                QUERY, self.closure, self.graph, budget, seeds=self.seeds, query_mode="why")
            self.assertLessEqual(pkt.manifest.total_tokens, budget, budget)
            if "fat" not in set(pkt.event_ids):
                self.assertIn("fat", pkt.manifest.missing_required, budget)

    def test_11u_declared_omissions_reach_the_reader_not_just_the_manifest(self):
        """Contract #1 refuses a silent cut because the reader cannot tell.  The declaration it offers
        instead lands in the manifest, which the reader never sees -- so by the contract's own reasoning
        a dropped required event is just as silent.  `declare_omissions` closes that, and stays inside
        the budget while doing it."""
        base = dict(pack="evidence_first", evidence_share=1.0, unit_cap_share=None, cost_order=False)
        quiet = BudgetAssembler(AssemblerConfig(**base)).assemble(
            QUERY, self.closure, self.graph, 60, seeds=self.seeds, query_mode="why")
        loud = BudgetAssembler(AssemblerConfig(declare_omissions=True, **base)).assemble(
            QUERY, self.closure, self.graph, 60, seeds=self.seeds, query_mode="why")
        self.assertTrue(quiet.manifest.missing_required, "fixture must actually drop something")
        self.assertNotIn("tracepack-omitted", quiet.context)      # today: silent to the reader
        self.assertIn("tracepack-omitted", loud.context)          # with the flag: visible
        self.assertLessEqual(loud.manifest.total_tokens, 60)      # and charged, not free
        self.assertEqual(set(quiet.event_ids), set(loud.event_ids))   # same evidence, only the note added

    def test_11v_declare_omissions_is_silent_when_nothing_is_missing(self):
        big = dict(pack="evidence_first", evidence_share=1.0, declare_omissions=True)
        pkt = BudgetAssembler(AssemblerConfig(**big)).assemble(
            QUERY, self.closure, self.graph, 4096, seeds=self.seeds, query_mode="why")
        self.assertEqual(pkt.manifest.missing_required, ())
        self.assertNotIn("tracepack-omitted", pkt.context)

    def test_11s_unit_cap_share_is_validated(self):
        for bad in (-0.1, 0.0, 1.5, "half", True):
            with self.assertRaises(SchemaError):
                AssemblerConfig(pack="evidence_first", unit_cap_share=bad)
        for bad in (1, "yes", None):
            with self.assertRaises(SchemaError):
                AssemblerConfig(pack="evidence_first", declare_omissions=bad)


def _cost(text: str) -> int:
    return max(1, len(text) // 4)


def excerpt_graph():
    """A 40-line tool result whose VALUE line shares no token with the query, plus a child that
    quotes one other line, plus a short seed."""
    lines = ["$ cat /srv/app/deploy.yaml"]
    lines += ["  option_%02d: value_%02d" % (i, i) for i in range(1, 11)]
    lines += ["  secret_token: hunter2xyz"]                 # the "gold": no query overlap, far from
    lines += ["  option_%02d: value_%02d" % (i, i) for i in range(11, 30)]   # any selected line
    lines += ["  region: ap-southeast-1 (primary)"]        # quoted by the child below
    lines += ["  trailing_%d: x" % i for i in range(9)]
    big = "\n".join(lines)
    ev = [
        F.event("big", "tool_result", 1, big, _cost(big), group="grp:b", tool_call_id="tb"),
        F.event("call", "tool_call", 0, 'Bash {"command": "cat /srv/app/deploy.yaml"}', 12,
                group="grp:b", tool_call_id="tb"),
        F.event("child", "assistant", 5, "we use region: ap-southeast-1 (primary) for the rollout", 14),
        F.event("s", "assistant", 9, "the deploy yaml lists the options", 8),
    ]
    edges = [TraceEdge("big", "call", "RESULT_OF", predicate="tool_use_id"),
             TraceEdge("child", "big", "DEPENDS_ON"), TraceEdge("s", "big", "DEPENDS_ON")]
    return TraceGraph(ev, edges), big


class Contract12ExcerptIsALabelledLastResort(unittest.TestCase):
    """PLAN_phase2 WP2 / core/excerpt.py: the excerpt contract."""

    EXQ = "which options does the deploy yaml list"

    def setUp(self):
        from tracepack.core.excerpt import make_excerpt_fn
        self.graph, self.big = excerpt_graph()
        # graph-level quoting: the texts of every DEPENDS_ON child of an event (the child is
        # usually NOT in the closure -- edges point child -> parent)
        quoting = {}
        for e in self.graph.edges:
            if e.edge_type == "DEPENDS_ON":
                quoting.setdefault(e.dst_id, []).append(self.graph.event(e.src_id).text)
        self.fn = make_excerpt_fn(_cost, quoting_of=lambda ev: quoting.get(ev.event_id, ()))
        self.seeds = [F.seed("s", rank=0)]
        self.closure = TypedClosure(ClosureConfig(mode="native")).close(
            self.EXQ, self.seeds, self.graph, query_mode="why")
        self.assertIn("big", self.closure.required)
        self.whole = self.graph.event("big").token_cost + 12   # pair cost

    def _asm(self, **cfg):
        return BudgetAssembler(AssemblerConfig(**cfg), excerpt_fn=self.fn)

    def test_12a_whole_event_wins_when_it_fits(self):
        pkt = self._asm().assemble(self.EXQ, self.closure, self.graph, 10_000, seeds=self.seeds,
                                   query_mode="why")
        kinds = {e.event_id: e.repr_kind for e in pkt.manifest.entries}
        self.assertEqual(kinds["big"], "raw_text")
        self.assertIn(self.big, pkt.context)

    def test_12b_excerpt_is_taken_only_when_the_whole_does_not_fit_and_is_labelled(self):
        budget = self.whole + 8 - 1          # seed (8) + whole pair no longer fits
        pkt = self._asm().assemble(self.EXQ, self.closure, self.graph, budget, seeds=self.seeds,
                                   query_mode="why")
        kinds = {e.event_id: e.repr_kind for e in pkt.manifest.entries}
        self.assertIn("big", kinds, pkt.manifest.missing_required)
        self.assertTrue(kinds["big"].startswith("excerpt["), kinds["big"])
        self.assertRegex(kinds["big"], r"^excerpt\[[0-9,\-]+/41\]$")
        self.assertIn("[excerpt: lines", pkt.context)
        self.assertLessEqual(pkt.manifest.total_tokens, budget)
        self.assertFalse(pkt.incomplete)
        # provenance is exact: every excerpt line is a real line of the event
        body = [ln for ln in pkt.context.split("\n") if ln and not ln.startswith("[excerpt")]
        real = set(self.big.split("\n")) | {"...", "we use region: ap-southeast-1 (primary) for the rollout",
                                             "the deploy yaml lists the options",
                                             'Bash {"command": "cat /srv/app/deploy.yaml"}'}
        self.assertTrue(all(ln in real for ln in body), [ln for ln in body if ln not in real][:3])

    def test_12c_gold_blind_the_value_line_is_not_selected_without_query_or_quote_support(self):
        budget = self.whole + 8 - 1
        pkt = self._asm().assemble(self.EXQ, self.closure, self.graph, budget, seeds=self.seeds,
                                   query_mode="why")
        self.assertNotIn("hunter2xyz", pkt.context)
        self.assertIn("$ cat /srv/app/deploy.yaml", pkt.context)        # header line kept

    def test_12d_a_quoting_child_pulls_its_quoted_line_in(self):
        budget = self.whole + 8 - 1
        pkt = self._asm().assemble(self.EXQ, self.closure, self.graph, budget, seeds=self.seeds,
                                   query_mode="why")
        self.assertIn("region: ap-southeast-1 (primary)", pkt.context)

    def test_12e_query_terms_select_lines(self):
        budget = self.whole + 8 - 1
        pkt = self._asm().assemble("what was option_17 set to", self.closure, self.graph, budget,
                                   seeds=self.seeds, query_mode="why")
        self.assertIn("option_17: value_17", pkt.context)
        self.assertNotIn("option_03: value_03", pkt.context)

    def test_12f_without_excerpt_fn_nothing_changes(self):
        budget = self.whole + 8 - 1
        plain = BudgetAssembler(AssemblerConfig()).assemble(self.EXQ, self.closure, self.graph,
                                                            budget, seeds=self.seeds, query_mode="why")
        self.assertNotIn("big", plain.event_ids)
        self.assertIn("big", plain.manifest.missing_required)

    def test_12g_carrier_only_never_excerpts_and_bad_excerpts_are_rejected(self):
        pkt = self._asm(repr_policy="carrier_only").assemble(
            self.EXQ, self.closure, self.graph, self.whole + 8 - 1, seeds=self.seeds, query_mode="why")
        self.assertTrue(all(not e.repr_kind.startswith("excerpt") for e in pkt.manifest.entries))
        bad = BudgetAssembler(AssemblerConfig(), excerpt_fn=lambda ev, q, qt: [("cut", "x", 1)])
        with self.assertRaises(SchemaError):
            bad.assemble(self.EXQ, self.closure, self.graph, self.whole + 8 - 1, seeds=self.seeds,
                         query_mode="why")
        with self.assertRaises(SchemaError):
            BudgetAssembler(AssemblerConfig(), excerpt_fn="not-callable")

    def test_12h_the_digest_sees_the_excerpt(self):
        budget = self.whole + 8 - 1
        a = self._asm().assemble(self.EXQ, self.closure, self.graph, budget, seeds=self.seeds,
                                 query_mode="why")
        b = self._asm().assemble("what was option_17 set to", self.closure, self.graph, budget,
                                 seeds=self.seeds, query_mode="why")
        self.assertNotEqual(a.manifest.digest(), b.manifest.digest())


def _row(uuid, parent, role, content, ts="2026-09-03T10:00:%02dZ"):
    return {"uuid": uuid, "parentUuid": parent, "type": role, "timestamp": ts % (int(uuid[1:]) % 60),
            "message": {"role": "assistant" if role == "assistant" else "user", "content": content}}


def value_trace():
    """Six rows: an `ls` whose output first shows `cfg/deploy_v2.yaml`, an assistant turn that
    repeats it (-> value edge), a Bash call that names a NEW path the agent made up (`out/report_7.md`),
    its result echoing that path (must NOT become a source), and a final assistant turn mentioning
    both, plus a number `409600` that appears in FOUR tool results (too common: no edge)."""
    rows = [
        _row("u1", None, "user", "list the config dir"),
        _row("a1", "u1", "assistant", [{"type": "tool_use", "id": "t_ls", "name": "Bash",
                                        "input": {"command": "ls cfg"}}]),
        _row("u2", "a1", "user", [{"type": "tool_result", "tool_use_id": "t_ls",
                                   "content": "cfg/deploy_v2.yaml\ncfg/old.yaml\nsize 409600"}]),
        _row("a2", "u2", "assistant", [{"type": "text", "text": "the live file is cfg/deploy_v2.yaml"}]),
        _row("a3", "a2", "assistant", [{"type": "tool_use", "id": "t_w", "name": "Bash",
                                        "input": {"command": "echo done > out/report_7.md"}}]),
        _row("u3", "a3", "user", [{"type": "tool_result", "tool_use_id": "t_w",
                                   "content": "wrote out/report_7.md 409600"}]),
        _row("a4", "u3", "assistant", [{"type": "tool_use", "id": "t_c", "name": "Bash",
                                        "input": {"command": "cat cfg/deploy_v2.yaml"}}]),
        _row("u4", "a4", "user", [{"type": "tool_result", "tool_use_id": "t_c",
                                   "content": "replicas: 409600\nregion: x"}]),
        _row("a5", "u4", "assistant", [{"type": "tool_use", "id": "t_d", "name": "Bash",
                                        "input": {"command": "du out"}}]),
        _row("u5", "a5", "user", [{"type": "tool_result", "tool_use_id": "t_d",
                                   "content": "409600 out\n409600 total"}]),
        _row("a6", "u5", "assistant", [{"type": "text",
                                        "text": "summary: cfg/deploy_v2.yaml and out/report_7.md, size 409600"}]),
    ]
    return "\n".join(json.dumps(r) for r in rows) + "\n"


class Contract14ValueSourceEdges(unittest.TestCase):
    """PLAN_phase2 WP3a: DEPENDS_ON(value:*) edges from the adapter, opt-in and structural."""

    def setUp(self):
        import tempfile
        from tracepack.adapters.claude_code import ClaudeCodeAdapter
        self.tmp = tempfile.NamedTemporaryFile("w", suffix=".jsonl", delete=False, encoding="utf-8")
        self.tmp.write(value_trace())
        self.tmp.close()
        self.A = ClaudeCodeAdapter
        self.g_off, self.s_off = self.A().normalize_with_stats(self.tmp.name)
        self.g_on, self.s_on = self.A(link_values=True).normalize_with_stats(self.tmp.name)

    def tearDown(self):
        os.unlink(self.tmp.name)

    def _values(self, g):
        return {(e.src_id, e.dst_id, e.predicate) for e in g.edges
                if e.edge_type == "DEPENDS_ON" and (e.predicate or "").startswith("value:")}

    def _text(self, g, needle):
        return [e.event_id for e in g.events if needle in (e.text or "")]

    def test_14a_off_by_default_and_the_default_graph_is_unchanged(self):
        self.assertEqual(self._values(self.g_off), set())
        self.assertEqual([(e.src_id, e.dst_id, e.edge_type) for e in self.g_off.edges],
                         [(e.src_id, e.dst_id, e.edge_type) for e in self.A().normalize(self.tmp.name).edges])
        self.assertNotIn("value_edges", self.s_off)

    def test_14b_a_value_that_entered_through_a_tool_result_links_the_repeating_turn_to_it(self):
        ls_result = self._text(self.g_on, "cfg/old.yaml")[0]
        a2 = self._text(self.g_on, "the live file is")[0]
        self.assertIn((a2, ls_result, "value:path"), self._values(self.g_on))

    def test_14c_a_value_the_agent_wrote_first_never_gets_a_source(self):
        """`out/report_7.md` was invented in the agent's own Bash call; the result only echoes it."""
        echo = self._text(self.g_on, "wrote out/report_7.md")[0]
        bad = [e for e in self._values(self.g_on) if e[1] == echo]
        self.assertEqual(bad, [], bad)
        self.assertGreaterEqual(self.s_on.get("value_skip_agent_wrote_it", 0), 1)

    def test_14d_a_common_value_is_not_evidence(self):
        """`409600` occurs in four tool results (> value_max_sources=3): no edge carries it."""
        self.assertFalse([e for e in self.g_on.edges if e.edge_type == "DEPENDS_ON"
                          and (e.predicate or "") == "value:number"])
        self.assertGreaterEqual(self.s_on.get("value_skip_common", 0), 1)
        # with a looser cap the same value IS linked -- the rule is the knob, not an accident
        g_loose = self.A(link_values=True, value_max_sources=10).normalize(self.tmp.name)
        self.assertTrue([e for e in g_loose.edges if (e.predicate or "") == "value:number"])

    def test_14e_the_edge_points_at_the_nearest_earlier_reading(self):
        """`a6` repeats cfg/deploy_v2.yaml; the nearest earlier result holding it is the `cat`
        call's result?  No -- that result does not contain the path; the `ls` result does, and a4's
        call (which names it) is a consumer, not a source.  So a6 -> ls result."""
        a6 = self._text(self.g_on, "summary: cfg/deploy_v2.yaml")[0]
        ls_result = self._text(self.g_on, "cfg/old.yaml")[0]
        self.assertIn((a6, ls_result, "value:path"), self._values(self.g_on))

    def test_14f_edges_are_inferred_strong_and_closure_follows_them(self):
        from tracepack.core.closure import ClosureConfig, TypedClosure
        for e in self.g_on.edges:
            if (e.predicate or "").startswith("value:"):
                self.assertEqual(e.provenance, "inferred")
                self.assertTrue(e.is_strong and e.is_required)
        a2 = self._text(self.g_on, "the live file is")[0]
        ls_result = self._text(self.g_on, "cfg/old.yaml")[0]
        cl = TypedClosure(ClosureConfig(mode="native")).close(
            "which file is live", [F.seed(a2, rank=0)], self.g_on, query_mode="lookup")
        self.assertIn(ls_result, cl.required)
        cl0 = TypedClosure(ClosureConfig(mode="native")).close(
            "which file is live", [F.seed(a2, rank=0)], self.g_off, query_mode="lookup")
        self.assertNotIn(ls_result, cl0.required)

    def test_14g_options_are_validated(self):
        from tracepack.adapters.base import AdapterError
        for kw in (dict(value_max_sources=-1), dict(max_value_edges=-2), dict(value_lookback="x")):
            with self.assertRaises(AdapterError):
                self.A(link_values=True, **kw)


class Contract13ShippedPacksUnchanged(unittest.TestCase):
    """The pre-phase-2 assembler's packets, byte for byte, for every chrono/bundle combination."""

    QUERIES = {"why": "why did we choose that branch",
               "lookup": "what exact value did the tool print",
               "state": "what is the current deploy target"}

    def _graphs(self):
        graphs = [("tool_chain", F.tool_chain_graph()), ("atomic_pair", F.atomic_pair_graph()),
                  ("supersede", F.supersede_chain_graph()), ("carrier", F.carrier_source_graph())]
        graphs += [("rand%d" % i, g) for i, (_, g) in enumerate(F.random_graphs(6, n_events=24))]
        return dict(graphs)

    def test_13a_golden_digests_reproduce(self):
        rows = json.load(open(GOLDEN, encoding="utf-8"))
        self.assertGreater(len(rows), 1000, "golden file is too small to mean anything")
        graphs = self._graphs()
        cache = {}
        mismatches = []
        for r in rows:
            g = graphs[r["graph"]]
            ck = (r["graph"], r["router"], r["qm"], r["closure"])
            if ck not in cache:
                q = self.QUERIES[r["qm"]]
                sd = make_router(r["router"], RouterConfig(k=5)).retrieve(q, g, 5)
                cl = TypedClosure(ClosureConfig(mode=r["closure"])).close(q, sd, g, query_mode=r["qm"])
                cache[ck] = (q, sd, cl)
            q, sd, cl = cache[ck]
            pkt = BudgetAssembler(AssemblerConfig(repr_policy=r["repr"], order=r["order"],
                                                  pack=r["pack"])).assemble(
                q, cl, g, r["budget"], seeds=sd, query_mode=r["qm"])
            ctx = hashlib.sha256(pkt.context.encode("utf-8")).hexdigest()[:16]
            if pkt.manifest.digest() != r["digest"] or ctx != r["ctx"]:
                mismatches.append(r)
        self.assertEqual(mismatches[:3], [], "%d of %d golden packets changed" % (len(mismatches), len(rows)))

    def test_13b_the_new_packs_do_change_something(self):
        """If tiered/evidence_first were no-ops the goldens would prove nothing about isolation."""
        g = ranking_graph()
        sd = seeds()
        cl = closure_for(g, sd)
        # 45 tokens: chrono -> {s0,s1,s2,w1,w2}; tiered -> {s0,p0c,p0r}; evidence_first -> {p0c,p0r,sum}
        digests = {pack: BudgetAssembler(AssemblerConfig(pack=pack)).assemble(
            QUERY, cl, g, 45, seeds=sd, query_mode="why").manifest.digest() for pack in PACKS}
        self.assertEqual(len({digests["chrono"], digests["tiered"], digests["evidence_first"]}), 3,
                         digests)


if __name__ == "__main__":
    unittest.main()
