"""tracepack.tests.test_budget -- proposal §8.1 contracts 2, 4, 5, 6.

| §8.1 | claim                                                        | class here                    |
|------|--------------------------------------------------------------|-------------------------------|
| #2   | Budget Violation Rate == 0                                    | `Contract02BudgetViolationRate` |
| #4   | unsatisfiable => `incomplete` + non-empty `missing_required`  | `Contract04Incomplete`        |
| #5   | a pinned event is never displaced by ordinary top-k results   | `Contract05PinnedNotDisplaced`|
| #6   | a tool_call/tool_result atomic group is never split           | `Contract06Atomicity`         |

Design decisions in these tests
-------------------------------

* **The sweep proves its own coverage.**  A budget sweep where nothing ever fits has a Budget
  Violation Rate of 0 and tests nothing.  ``test_02a`` therefore asserts *four* things about the
  sweep itself -- that it produced complete packets, incomplete packets, packets that dropped an
  optional, and packets that hit the "does not fit at all" branch -- before it reports BVR.  See
  ``_Sweep.assert_covered``.

* **The axes are the frozen experiment factors (DESIGN_FROZEN §4), not arbitrary knobs.**  The
  sweep runs ``router x closure x representation`` over seeded random graphs so a regression in
  any one arm shows up here rather than in the profiler run that produces the paper number.
  The three pre-registered budgets 1024/2048/4096 are always included, alongside pathological
  ones (0, 1, off-by-one around the exact-fit sum, and one above the whole graph).

* **Contract #5 is checked where it can fail.**  Asserting "the pin came back" from a router
  that had room for everything is vacuous, so ``pin_vs_topk_graph`` is built so the unpinned
  arms rank the addressed event *last*: the test first proves the plain routers drop it, then
  proves the pinned router keeps it.

* **Contract #6 is checked as an invariant, not an example.**  ``_split_groups`` scans every
  packet in the sweep for a served, strict, non-empty subset of an atomic group that the closure
  selected.  Contrast with the ``off`` arm, where the group is legitimately never selected whole
  -- documented in ``test_06d`` so it is not mistaken for a leak.

* **"Never truncated" is asserted on the served string, not on the manifest.**  A manifest that
  omits an event and a context that quietly contains half of it would pass every count-based
  check; ``test_04a`` greps the context for the payload marker.

Pure stdlib + unittest; no network, no LLM, no torch.
"""
from __future__ import annotations

import os
import sys
import unittest

if __package__ in (None, ""):  # pragma: no cover - direct execution
    sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(
        os.path.abspath(__file__)))))

from tracepack.core.assembler import (
    REPR_POLICIES,
    AssemblerConfig,
    BudgetAssembler,
    estimate_text_tokens,
)
from tracepack.core.closure import ClosureConfig, TypedClosure
from tracepack.core.router import RouterConfig, make_router
from tracepack.core.schema import BudgetError, SchemaError
from tracepack.tests.fixtures import tiny_trace as F

#: DESIGN_FROZEN §1 (pre-registered) plus the pathological neighbours of the fixtures' costs
BUDGETS = (0, 1, 3, 9, 45, 46, 47, 199, 200, 201, 512,
           1024, 2048, 4096, 8192, 1_000_000)
QUERIES = ("what exact value did the tool print",
           "why did we choose that branch",
           "what is the current deploy target")


# ---------------------------------------------------------------- instruments


def _violation(packet, budget: int):
    """The one number contract #2 is about: served tokens minus the hard budget, when > 0."""
    total = packet.manifest.total_tokens
    return None if total <= budget else (total, budget)


def _recount(packet) -> int:
    """Independent re-derivation of ``total_tokens`` -- never read back from the manifest field.

    §3.4's equation is ``header + sum(entry costs)``; if the assembler's own bookkeeping drifts
    from its entries, this catches it even when the field itself is <= budget.
    """
    return sum(e.token_cost for e in packet.manifest.entries)


def _split_groups(packet, closure, graph):
    """Atomic groups the packet served only part of, counting only members the closure picked."""
    selected = set(closure.required) | set(closure.optional)
    served = set(packet.event_ids)
    wanted = {}
    for eid in selected:
        grp = graph.event(eid).atomic_group
        if grp:
            wanted.setdefault(grp, set()).add(eid)
    bad = []
    for grp, members in sorted(wanted.items()):
        got = members & served
        if got and got != members:
            bad.append((grp, tuple(sorted(got)), tuple(sorted(members))))
    return bad


def _seed_mix(graph, router_seeds):
    """Router seeds + one decision, one state and one carrier, when the graph has them.

    Without this the sweep never seeds a ``decision`` or a superseded ``state``, so the two
    branches that *produce* optional evidence (§3.3 ``why_needs_sources`` in lookup mode and
    ``state_prefers_latest``) never fire and "no optional was ever dropped" would be an artefact
    of the query wording rather than a property of the packer.
    """
    seeds = list(router_seeds)
    taken = {s.event_id for s in seeds}
    rank = len(seeds)
    for kind in ("decision", "state", "summary"):
        for ev in graph.events:
            if ev.kind == kind and ev.event_id not in taken:
                seeds.append(F.seed(ev.event_id, rank=rank, source="lexical"))
                taken.add(ev.event_id)
                rank += 1
                break
    return seeds


class _Sweep:
    """Accumulates what a sweep actually exercised, so the sweep can be audited too."""

    def __init__(self):
        self.n = 0
        self.violations = []
        self.miscounts = []
        self.split = []
        self.n_complete = 0
        self.n_incomplete = 0
        self.n_dropped_optional = 0
        self.n_nothing_fit = 0

    def record(self, packet, budget, closure, graph):
        self.n += 1
        v = _violation(packet, budget)
        if v is not None:
            self.violations.append((budget, v))
        expect = _recount(packet)
        header = estimate_text_tokens("")
        if packet.manifest.total_tokens != expect + header:
            self.miscounts.append((budget, packet.manifest.total_tokens, expect))
        self.split.extend(_split_groups(packet, closure, graph))
        if packet.manifest.incomplete:
            self.n_incomplete += 1
        else:
            self.n_complete += 1
        if packet.manifest.omitted_optional:
            self.n_dropped_optional += 1
        if not packet.manifest.entries and (closure.required or closure.optional):
            self.n_nothing_fit += 1

    def assert_covered(self, case: unittest.TestCase):
        case.assertGreater(self.n, 500, "the sweep is too small to mean anything")
        case.assertGreater(self.n_complete, 0, "no packet ever fit: BVR==0 would be vacuous")
        case.assertGreater(self.n_incomplete, 0, "no packet was ever over budget")
        case.assertGreater(self.n_dropped_optional, 0, "the drop-optional branch never ran")
        case.assertGreater(self.n_nothing_fit, 0, "the nothing-fits branch never ran")


# ---------------------------------------------------------------- contract #2


class Contract02BudgetViolationRate(unittest.TestCase):
    """§8.1 #2 / §3.4: the hard budget is a system contract, checked across the frozen factors."""

    def test_02a_bvr_is_zero_across_routers_closures_representations_and_budgets(self):
        sweep = _Sweep()
        for rng_seed, graph in F.random_graphs(6, n_events=20):
            for router_name in ("lexical", "hybrid_pin"):
                router = make_router(router_name, RouterConfig(k=5))
                for query in QUERIES[:2]:
                    seeds = _seed_mix(graph, router.retrieve(query, graph, 5))
                    for closure_mode in ("off", "native", "full_ancestor"):
                        for query_mode in ("lookup", "why", "state"):
                            closure = TypedClosure(ClosureConfig(mode=closure_mode)).close(
                                query, seeds, graph, query_mode=query_mode)
                            for policy in REPR_POLICIES:
                                asm = BudgetAssembler(AssemblerConfig(repr_policy=policy))
                                for budget in BUDGETS:
                                    pkt = asm.assemble(query, closure, graph, budget,
                                                       seeds=seeds, query_mode=query_mode)
                                    sweep.record(pkt, budget, closure, graph)
        self.assertEqual([], sweep.violations,
                         "%d/%d packets exceeded their budget" % (len(sweep.violations), sweep.n))
        self.assertEqual([], sweep.miscounts, "manifest.total_tokens disagrees with its entries")
        self.assertEqual([], sweep.split, "an atomic group was split during the sweep")
        sweep.assert_covered(self)

    def test_02b_the_header_is_charged_and_dropped_rather_than_truncated(self):
        graph = F.tool_chain_graph()
        closure = TypedClosure().close("q", [F.seed("d1")], graph, query_mode="why")
        header = "EVIDENCE PACKET -- do not quote the framing"
        cost = estimate_text_tokens(header)
        self.assertGreater(cost, 0)
        asm = BudgetAssembler(AssemblerConfig(header=header))
        tight = asm.assemble("q", closure, graph, cost - 1, seeds=[F.seed("d1")],
                             query_mode="why")
        self.assertNotIn(header[:12], tight.context)
        self.assertTrue(tight.incomplete, "a dropped header is a real degradation, not a detail")
        self.assertLessEqual(tight.manifest.total_tokens, cost - 1)
        roomy = asm.assemble("q", closure, graph, 1000, seeds=[F.seed("d1")], query_mode="why")
        self.assertIn(header, roomy.context)
        self.assertEqual(cost + _recount(roomy), roomy.manifest.total_tokens)

    def test_02c_budget_zero_yields_an_empty_packet_not_an_exception(self):
        graph = F.tool_chain_graph()
        closure = TypedClosure().close("q", [F.seed("d1")], graph, query_mode="why")
        pkt = BudgetAssembler().assemble("q", closure, graph, 0, seeds=[F.seed("d1")],
                                         query_mode="why")
        self.assertEqual((), pkt.event_ids)
        self.assertEqual("", pkt.context)
        self.assertEqual(0, pkt.manifest.total_tokens)
        self.assertTrue(pkt.incomplete)
        self.assertEqual(set(F.TOOL_CHAIN_WHY_REQUIRED), set(pkt.manifest.missing_required))

    def test_02d_off_by_one_at_the_exact_required_sum(self):
        """The boundary the sweep can only sample: required costs exactly 200 tokens."""
        graph = F.exact_fit_graph()
        seeds = [F.seed(F.EXACT_FIT_SEED)]
        closure = TypedClosure().close("q", seeds, graph, query_mode="why")
        self.assertEqual(set(F.EXACT_FIT_REQUIRED), set(closure.required))
        asm = BudgetAssembler()

        exact = asm.assemble("q", closure, graph, F.EXACT_FIT_SUM, seeds=seeds, query_mode="why")
        self.assertFalse(exact.incomplete)
        self.assertEqual(F.EXACT_FIT_SUM, exact.manifest.total_tokens)
        self.assertEqual(set(F.EXACT_FIT_REQUIRED), set(exact.event_ids))

        short = asm.assemble("q", closure, graph, F.EXACT_FIT_SUM - 1, seeds=seeds,
                             query_mode="why")
        self.assertTrue(short.incomplete)
        self.assertLess(short.manifest.total_tokens, F.EXACT_FIT_SUM)
        self.assertTrue(short.manifest.missing_required)

        over = asm.assemble("q", closure, graph, F.EXACT_FIT_SUM + 1, seeds=seeds,
                            query_mode="why")
        # the digest deliberately hashes the budget itself, so compare what was SERVED
        self.assertEqual(exact.context, over.context,
                         "one spare token must not change what is served")
        self.assertEqual([(e.event_id, e.token_cost) for e in exact.manifest.entries],
                         [(e.event_id, e.token_cost) for e in over.manifest.entries])

    def test_02e_illegal_budgets_raise_instead_of_being_coerced(self):
        graph = F.tool_chain_graph()
        closure = TypedClosure().close("q", [F.seed("d1")], graph, query_mode="why")
        asm = BudgetAssembler()
        for bad in (-1, -1024, True, 2.5, "2048", None):
            with self.subTest(budget=bad):
                with self.assertRaises(BudgetError):
                    asm.assemble("q", closure, graph, bad, query_mode="why")

    def test_02f_every_served_entry_costs_what_its_representation_says(self):
        """The assembler must never re-tokenize; the eval harness owns the tokenizer (§3.4)."""
        for rng_seed, graph in F.random_graphs(4, n_events=16):
            seeds = make_router("hybrid", RouterConfig(k=4)).retrieve(QUERIES[0], graph, 4)
            closure = TypedClosure().close(QUERIES[0], seeds, graph, query_mode="why")
            for policy in REPR_POLICIES:
                pkt = BudgetAssembler(AssemblerConfig(repr_policy=policy)).assemble(
                    QUERIES[0], closure, graph, 4096, seeds=seeds, query_mode="why")
                for entry in pkt.manifest.entries:
                    ev = graph.event(entry.event_id)
                    with self.subTest(seed=rng_seed, policy=policy, event=entry.event_id):
                        if entry.repr_kind == "raw_text+materialized_text":
                            expect = (ev.cost_of("raw_text")
                                      + ev.representation("materialized_text").token_cost)
                        else:
                            expect = ev.cost_of(entry.repr_kind)
                        self.assertEqual(expect, entry.token_cost)


# ---------------------------------------------------------------- contract #4


class Contract04Incomplete(unittest.TestCase):
    """§8.1 #4: an event that cannot fit is reported, never cut down to size."""

    def test_04a_an_event_larger_than_every_budget_is_reported_not_truncated(self):
        graph = F.oversized_event_graph()
        seeds = [F.seed("tiny")]
        closure = TypedClosure().close("q", seeds, graph, query_mode="why")
        for budget in (1024, 2048, 4096):        # the frozen budgets, DESIGN_FROZEN §1
            with self.subTest(budget=budget):
                pkt = BudgetAssembler().assemble("q", closure, graph, budget, seeds=seeds,
                                                 query_mode="why")
                self.assertTrue(pkt.incomplete)
                self.assertEqual((F.OVERSIZED_ID,), pkt.manifest.missing_required)
                self.assertNotIn("PAYLOAD-BEGIN", pkt.context)
                self.assertNotIn("PAYLOAD-END", pkt.context)
                # the rest of the evidence still gets served: a missing dependency is not a crash
                self.assertIn("tiny", pkt.event_ids)

    def test_04b_a_partially_served_packet_says_exactly_what_is_missing(self):
        graph = F.atomic_pair_graph()
        seeds = [F.seed("dec")]
        closure = TypedClosure().close("q", seeds, graph, query_mode="why")
        pkt = BudgetAssembler().assemble("q", closure, graph, 30, seeds=seeds, query_mode="why")
        self.assertTrue(pkt.incomplete)
        self.assertEqual({"call", "res"}, set(pkt.manifest.missing_required))
        self.assertEqual(("dec",), pkt.event_ids)

    def test_04c_missing_required_and_incomplete_move_together(self):
        for rng_seed, graph in F.random_graphs(5, n_events=18):
            seeds = make_router("hybrid", RouterConfig(k=4)).retrieve(QUERIES[1], graph, 4)
            closure = TypedClosure().close(QUERIES[1], seeds, graph, query_mode="why")
            for budget in (0, 64, 512, 4096, 1_000_000):
                pkt = BudgetAssembler().assemble(QUERIES[1], closure, graph, budget,
                                                 seeds=seeds, query_mode="why")
                with self.subTest(seed=rng_seed, budget=budget):
                    if pkt.manifest.missing_required:
                        self.assertTrue(pkt.incomplete)
                    # served + missing + omitted must account for the whole closure
                    accounted = (set(pkt.event_ids) | set(pkt.manifest.missing_required)
                                 | set(pkt.manifest.omitted_optional))
                    self.assertEqual(set(closure.required) | set(closure.optional), accounted)

    def test_04d_a_complete_packet_never_claims_incompleteness(self):
        graph = F.tool_chain_graph()
        seeds = [F.seed("d1")]
        closure = TypedClosure().close("q", seeds, graph, query_mode="why")
        pkt = BudgetAssembler().assemble("q", closure, graph, 4096, seeds=seeds, query_mode="why")
        self.assertFalse(pkt.incomplete)
        self.assertEqual((), pkt.manifest.missing_required)
        self.assertEqual(F.TOOL_CHAIN_WHY_COST, pkt.manifest.total_tokens)


# ---------------------------------------------------------------- contract #5


class Contract05PinnedNotDisplaced(unittest.TestCase):
    """§8.1 #5 / §4.4: an explicit address outranks a retrieval score."""

    def setUp(self):
        self.graph = F.pin_vs_topk_graph()
        self.query = F.PIN_QUERY

    def test_05a_the_unpinned_arms_really_do_drop_the_addressed_event(self):
        """Without this, contract #5 would be satisfied by an easy graph rather than by pinning."""
        for name in ("lexical", "dense", "hybrid"):
            with self.subTest(router=name):
                seeds = make_router(name, RouterConfig(k=3)).retrieve(self.query, self.graph, 3)
                self.assertNotIn(F.PIN_TARGET, [s.event_id for s in seeds])

    def test_05b_the_pinned_router_keeps_it_at_every_k(self):
        for k in (1, 2, 3, 5):
            with self.subTest(k=k):
                seeds = make_router("hybrid_pin", RouterConfig(k=k)).retrieve(
                    self.query, self.graph, k)
                ids = [s.event_id for s in seeds]
                self.assertIn(F.PIN_TARGET, ids)
                pin = [s for s in seeds if s.event_id == F.PIN_TARGET][0]
                self.assertTrue(pin.pinned)
                self.assertEqual("pin", pin.source)
                self.assertEqual(0, pin.rank, "pin_first puts addresses at the head")

    def test_05c_pins_survive_even_when_they_outnumber_k(self):
        """§4.4: the token budget is the assembler's contract, not the router's."""
        query = 'step 17 and step 90000000 and "%s"' % "17"
        seeds = make_router("hybrid_pin", RouterConfig(k=1)).retrieve(query, self.graph, 1)
        pinned = [s for s in seeds if s.pinned]
        self.assertGreaterEqual(len(seeds), len(pinned))
        self.assertIn(F.PIN_TARGET, [s.event_id for s in seeds])

    def test_05d_ranks_are_contiguous_so_the_manifest_order_is_total(self):
        seeds = make_router("hybrid_pin", RouterConfig(k=4)).retrieve(self.query, self.graph, 4)
        self.assertEqual(list(range(len(seeds))), [s.rank for s in seeds])

    def test_05e_in_the_packet_a_pinned_unit_is_packed_before_a_better_ranked_one(self):
        graph = F.tool_chain_graph()
        seeds = [F.seed("noise1", rank=0, source="lexical"),
                 F.seed("d1", rank=1, source="pin", pinned=True)]
        closure = TypedClosure(ClosureConfig(mode="off")).close(
            "q", seeds, graph, query_mode="lookup")
        # d1 costs 8, noise1 costs 7; at 8 tokens only one of them can be served
        pkt = BudgetAssembler().assemble("q", closure, graph, 8, seeds=seeds, query_mode="lookup")
        self.assertEqual(("d1",), pkt.event_ids)
        self.assertEqual("seed:pin", pkt.manifest.entries[0].reason)
        self.assertEqual(("noise1",), pkt.manifest.missing_required)

    def test_05f_a_pin_that_does_not_fit_is_reported_not_truncated(self):
        """Priority is not a licence to break §3.4: the pin loses to the hard budget, loudly."""
        graph = F.tool_chain_graph()
        seeds = [F.seed("noise1", rank=0), F.seed("d1", rank=1, source="pin", pinned=True)]
        closure = TypedClosure(ClosureConfig(mode="off")).close(
            "q", seeds, graph, query_mode="lookup")
        pkt = BudgetAssembler().assemble("q", closure, graph, 7, seeds=seeds, query_mode="lookup")
        self.assertNotIn("d1", pkt.event_ids)
        self.assertIn("d1", pkt.manifest.missing_required)
        self.assertTrue(pkt.incomplete)

    def test_05g_an_unresolvable_address_produces_no_seed_at_all(self):
        """A pin is an address; an address that points nowhere is a bug, not evidence."""
        seeds = make_router("hybrid_pin", RouterConfig(k=3)).retrieve(
            "what happened in step 4242?", self.graph, 3)
        self.assertEqual([], [s for s in seeds if s.pinned])


# ---------------------------------------------------------------- contract #6


class Contract06Atomicity(unittest.TestCase):
    """§8.1 #6 / §3.4: a tool_call and its tool_result are packed all-or-nothing."""

    def setUp(self):
        self.graph = F.atomic_pair_graph()
        self.seeds = [F.seed("dec")]
        self.closure = TypedClosure().close("q", self.seeds, self.graph, query_mode="why")

    def test_06a_the_group_is_never_half_served(self):
        asm = BudgetAssembler()
        saw_none, saw_both = False, False
        for budget in range(0, 60):
            pkt = asm.assemble("q", self.closure, self.graph, budget, seeds=self.seeds,
                               query_mode="why")
            served = set(pkt.event_ids) & {"call", "res"}
            with self.subTest(budget=budget):
                self.assertIn(served, ({"call", "res"}, set()),
                              "budget %d served exactly %s" % (budget, sorted(served)))
            saw_none |= not served
            saw_both |= served == {"call", "res"}
        self.assertTrue(saw_none and saw_both, "the sweep must exercise both outcomes")

    def test_06b_a_budget_that_fits_only_the_result_takes_neither(self):
        """`res` alone costs 24 and would fit at 34; the pair costs 36 and does not."""
        pkt = BudgetAssembler().assemble("q", self.closure, self.graph, 34, seeds=self.seeds,
                                         query_mode="why")
        self.assertEqual(("dec",), pkt.event_ids)
        self.assertEqual({"call", "res"}, set(pkt.manifest.missing_required))
        self.assertNotIn("eu-west: 412 units", pkt.context)

    def test_06c_the_first_budget_that_fits_the_pair_takes_all_of_it(self):
        pkt = BudgetAssembler().assemble("q", self.closure, self.graph, 46, seeds=self.seeds,
                                         query_mode="why")
        self.assertEqual({"call", "res", "dec"}, set(pkt.event_ids))
        self.assertEqual(46, pkt.manifest.total_tokens)

    def test_06d_the_off_arm_never_selects_the_group_whole_and_says_so(self):
        """DOCUMENTED BOUNDARY, not a leak: atomicity is resolved over the closure's candidate
        set (assembler [CONTRACT] #2).  With `closure=off` the call was never selected, so the
        packet serves a lone tool_result -- which is precisely the deficiency the `off` baseline
        exists to measure (§6.2).  The manifest still names the reason, so it is auditable.
        """
        closure = TypedClosure(ClosureConfig(mode="off")).close(
            "q", [F.seed("res")], self.graph, query_mode="why")
        self.assertEqual(("res",), closure.required)
        pkt = BudgetAssembler().assemble("q", closure, self.graph, 4096, seeds=[F.seed("res")],
                                         query_mode="why")
        self.assertEqual(("res",), pkt.event_ids)
        self.assertNotIn("call", pkt.event_ids)
        self.assertEqual("seed:lexical", pkt.manifest.entries[0].reason)
        # and the native policy does NOT have this hole:
        native = TypedClosure().close("q", [F.seed("res")], self.graph, query_mode="why")
        self.assertEqual({"res", "call"}, set(native.required))

    def test_06e_group_integrity_holds_across_the_random_sweep(self):
        bad = []
        for rng_seed, graph in F.random_graphs(6, n_events=20):
            seeds = make_router("hybrid", RouterConfig(k=5)).retrieve(QUERIES[0], graph, 5)
            for query_mode in ("lookup", "why"):
                closure = TypedClosure().close(QUERIES[0], seeds, graph, query_mode=query_mode)
                for budget in (0, 64, 256, 1024, 4096, 65536):
                    pkt = BudgetAssembler().assemble(QUERIES[0], closure, graph, budget,
                                                     seeds=seeds, query_mode=query_mode)
                    bad.extend((rng_seed, budget) + s for s in _split_groups(pkt, closure, graph))
        self.assertEqual([], bad)


# ---------------------------------------------------------------- adversarial


class AdversarialBudget(unittest.TestCase):
    """Mistakes that would make every budget number in the paper a fantasy."""

    def test_the_context_contains_exactly_what_the_manifest_claims(self):
        graph = F.tool_chain_graph()
        seeds = [F.seed("d1")]
        closure = TypedClosure().close("q", seeds, graph, query_mode="why")
        pkt = BudgetAssembler().assemble("q", closure, graph, 4096, seeds=seeds, query_mode="why")
        for entry in pkt.manifest.entries:
            self.assertIn(graph.event(entry.event_id).text, pkt.context)
        for eid in set(graph.event_ids) - set(pkt.event_ids):
            self.assertNotIn(graph.event(eid).text, pkt.context)

    def test_entries_are_never_duplicated_even_when_an_event_is_seeded_twice(self):
        graph = F.tool_chain_graph()
        seeds = [F.seed("d1", rank=0, source="lexical"), F.seed("d1", rank=1, source="dense")]
        closure = TypedClosure().close("q", seeds, graph, query_mode="why")
        pkt = BudgetAssembler().assemble("q", closure, graph, 4096, seeds=seeds, query_mode="why")
        ids = list(pkt.event_ids)
        self.assertEqual(len(ids), len(set(ids)), "a double-seeded event is double-charged")
        self.assertEqual(F.TOOL_CHAIN_WHY_COST, pkt.manifest.total_tokens)

    def test_a_seed_for_a_nonexistent_event_is_a_router_bug_and_raises(self):
        graph = F.tool_chain_graph()
        closure = TypedClosure().close("q", [F.seed("d1")], graph, query_mode="why")
        with self.assertRaises(SchemaError):
            BudgetAssembler().assemble("q", closure, graph, 4096,
                                       seeds=[F.seed("ghost")], query_mode="why")

    def test_config_rejects_unknown_policies_and_orders(self):
        for kwargs in ({"repr_policy": "kv_only"}, {"order": "random"},
                       {"header": 7}, {"separator": None}):
            with self.subTest(**kwargs):
                with self.assertRaises(SchemaError):
                    AssemblerConfig(**kwargs)

    def test_seed_first_ordering_changes_the_context_but_not_the_budget(self):
        graph = F.tool_chain_graph()
        seeds = [F.seed("d1")]
        closure = TypedClosure().close("q", seeds, graph, query_mode="why")
        chrono = BudgetAssembler(AssemblerConfig(order="chronological")).assemble(
            "q", closure, graph, 4096, seeds=seeds, query_mode="why")
        seed_first = BudgetAssembler(AssemblerConfig(order="seed_first")).assemble(
            "q", closure, graph, 4096, seeds=seeds, query_mode="why")
        self.assertEqual(chrono.manifest.total_tokens, seed_first.manifest.total_tokens)
        self.assertEqual(set(chrono.event_ids), set(seed_first.event_ids))
        self.assertEqual("d1", seed_first.event_ids[0])
        self.assertNotEqual(chrono.manifest.digest(), seed_first.manifest.digest())

    def test_an_empty_closure_produces_an_empty_but_legal_packet(self):
        graph = F.tool_chain_graph()
        closure = TypedClosure(ClosureConfig(mode="off")).close(
            "q", [], graph, query_mode="lookup")
        pkt = BudgetAssembler().assemble("q", closure, graph, 4096, query_mode="lookup")
        self.assertEqual((), pkt.event_ids)
        self.assertFalse(pkt.incomplete)
        self.assertEqual(0, pkt.manifest.total_tokens)


# ---------------------------------------------------------------- selfcheck


def _selfcheck() -> None:
    """Run the suite, then prove the three instruments can actually report a fault."""

    class _FakeManifest:
        def __init__(self, total, entries=(), incomplete=False, omitted=()):
            self.total_tokens = total
            self.entries = entries
            self.incomplete = incomplete
            self.omitted_optional = omitted
            self.missing_required = ()

    class _FakePacket:
        def __init__(self, manifest, ids=()):
            self.manifest = manifest
            self.event_ids = tuple(ids)

    # ---- instrument 1: the violation detector must fire on an over-budget packet
    assert _violation(_FakePacket(_FakeManifest(2049)), 2048) == (2049, 2048), \
        "_violation missed an injected budget overflow"
    assert _violation(_FakePacket(_FakeManifest(2048)), 2048) is None
    assert _violation(_FakePacket(_FakeManifest(0)), 0) is None

    # ---- instrument 2: the split-group detector must fire on a half-served pair
    graph = F.atomic_pair_graph()
    closure = TypedClosure().close("q", [F.seed("dec")], graph, query_mode="why")
    half = _FakePacket(_FakeManifest(24), ids=("res",))
    found = _split_groups(half, closure, graph)
    assert found and found[0][1] == ("res",), \
        "_split_groups missed an injected half-served atomic group: %r" % (found,)
    whole = _FakePacket(_FakeManifest(46), ids=("res", "call", "dec"))
    assert _split_groups(whole, closure, graph) == []

    # ---- instrument 3: the coverage auditor must reject a vacuous sweep
    empty = _Sweep()
    try:
        empty.assert_covered(unittest.TestCase("run"))
    except AssertionError:
        pass
    else:  # pragma: no cover
        raise AssertionError("_Sweep.assert_covered accepted a sweep that ran nothing")

    # ---- and the real assembler must survive the same injections unchanged
    real = BudgetAssembler().assemble("q", closure, graph, 46, seeds=[F.seed("dec")],
                                      query_mode="why")
    assert _violation(real, 46) is None and _split_groups(real, closure, graph) == []

    suite = unittest.TestLoader().loadTestsFromModule(sys.modules[__name__])
    result = unittest.TextTestRunner(verbosity=1, stream=sys.stdout).run(suite)
    if not result.wasSuccessful():
        raise SystemExit("test_budget: %d failure(s), %d error(s)"
                         % (len(result.failures), len(result.errors)))
    print("test_budget selfcheck OK: %d tests, 3 instruments fault-injected, "
          "%d budgets x %d representation policies swept"
          % (result.testsRun, len(BUDGETS), len(REPR_POLICIES)))


if __name__ == "__main__":
    _selfcheck()
