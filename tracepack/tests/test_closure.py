"""tracepack.tests.test_closure -- proposal §8.1 contracts 3, 4, 7, 9, 10.

| §8.1 | claim                                                            | class here                  |
|------|------------------------------------------------------------------|-----------------------------|
| #3   | satisfiable => no dangling required dependency in the packet      | `Contract03NoDanglingRequired` |
| #4   | unsatisfiable => `incomplete` and non-empty `missing_required`    | `Contract04Unsatisfiable`   |
| #7   | a SUPERSEDES test never returns the stale state (`query_mode="state"`) | `Contract07Supersedes` |
| #9   | carrier verification must match model + predicate + protocol      | `Contract09CarrierVerification` |
| #10  | an UNVERIFIED carrier never triggers source removal               | `Contract10NoUnverifiedRemoval` |

Contracts 1, 2, 5, 6 and 8 live in `test_determinism.py`, `test_budget.py` and `test_adapter.py`.

Design decisions in these tests
-------------------------------

* **"Dangling" is measured against the policy that ran, and separately against the schema.**
  ``REQUIRED_EDGES`` is the schema's unconditional whitelist, but §3.3 lets the *policy* demote a
  decision's sources to optional in ``query_mode="lookup"``.  So contract #3 is checked twice:
  ``_dangling_under_policy`` (every parent the closure itself classified as required is in
  ``required``) holds in every mode, and ``_dangling_under_schema`` (every ``REQUIRED_EDGES``
  parent of a required event is required) is asserted only in the modes that promise it -- why,
  audit and state.  Collapsing the two would either hide a real bug or fail on documented policy.

* **Fault injection targets the instrument first.**  ``_dangling_under_policy`` is fed a
  deliberately broken closure in ``_selfcheck`` and must report the dangling id; a checker that
  cannot fail is not evidence that the code is right.

* **The carrier mismatch table is generated, not typed out.**  ``carrier_verification(**{field:
  bad})`` mutates exactly one of the five §4.5 fields per subtest, so a renamed field breaks the
  test instead of silently dropping a case.

* **Both original known gaps are CLOSED (2026-09-01) and now assert the fixed behaviour.**
  See
  ``test_03d_max_hops_truncation_...``: writing the *current* behaviour into an assertion would
  freeze the bug into the suite.

Pure stdlib + unittest; no network, no LLM, no torch.
"""
from __future__ import annotations

import os
import sys
import unittest

if __package__ in (None, ""):  # pragma: no cover - direct execution
    sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(
        os.path.abspath(__file__)))))

from tracepack.core.assembler import AssemblerConfig, BudgetAssembler
from tracepack.core.closure import ClosureConfig, RULES, TypedClosure
from tracepack.core.graph import TraceGraph
from tracepack.core.schema import (
    REQUIRED_EDGES,
    EvidenceClosure,
    SchemaError,
    Seed,
)
from tracepack.tests.fixtures import tiny_trace as F

GENEROUS = 1_000_000          # a budget nothing in the fixtures can exhaust


# ---------------------------------------------------------------- instruments


def _dangling_under_policy(closure: EvidenceClosure, graph: TraceGraph):
    """Ids the closure's OWN steps say are required parents, yet left out of ``required``.

    Reads ``closure.steps``: every expansion the policy actually performed is recorded there
    with the rule that licensed it, so this checker never has to re-derive the policy.
    """
    required = set(closure.required)
    served = required | set(closure.optional)
    bad = []
    for st in closure.steps:
        if st.rule in ("max_hops_truncated", "cycle_guard", "state_prefers_latest"):
            continue
        if st.child_id in required and st.edge_type in REQUIRED_EDGES:
            if st.parent_id not in served:
                bad.append((st.child_id, st.parent_id, st.edge_type, st.rule))
    return bad


def _dangling_under_schema(closure: EvidenceClosure, graph: TraceGraph):
    """Ids that a REQUIRED_EDGES parent of a required event, and are not required themselves."""
    required = set(closure.required)
    bad = []
    for eid in sorted(required):
        for edge in graph.parents(eid, REQUIRED_EDGES):
            if edge.dst_id not in required:
                bad.append((eid, edge.dst_id, edge.edge_type))
    return bad


def _dangling_in_packet(packet, graph: TraceGraph):
    """Events served whose REQUIRED_EDGES parents were not served."""
    served = set(packet.event_ids)
    bad = []
    for eid in sorted(served):
        for edge in graph.parents(eid, REQUIRED_EDGES):
            if edge.dst_id not in served:
                bad.append((eid, edge.dst_id, edge.edge_type))
    return bad


def _close(graph, seed_ids, *, mode="native", query_mode="why", query="why did we do that",
           **kw) -> EvidenceClosure:
    seeds = [F.seed(e, rank=i) for i, e in enumerate(seed_ids)]
    return TypedClosure(ClosureConfig(mode=mode)).close(
        query, seeds, graph, query_mode=query_mode, **kw)


# ---------------------------------------------------------------- contract #3


class Contract03NoDanglingRequired(unittest.TestCase):
    """§8.1 #3: when the evidence is satisfiable, nothing required is left dangling."""

    def test_03a_tool_chain_pulls_the_whole_evidence_path(self):
        g = F.tool_chain_graph()
        c = _close(g, [F.TOOL_CHAIN_SEED], query_mode="why")
        self.assertEqual(tuple(sorted(c.required)), tuple(sorted(F.TOOL_CHAIN_WHY_REQUIRED)))
        self.assertEqual([], _dangling_under_policy(c, g))
        self.assertEqual([], _dangling_under_schema(c, g))

    def test_03b_weak_edges_are_never_followed(self):
        """§3.2: CONTROL/TEMPORAL are execution ancestry, not evidence."""
        g = F.tool_chain_graph()
        for query_mode in ("lookup", "why", "state", "audit"):
            with self.subTest(query_mode=query_mode):
                c = _close(g, [F.TOOL_CHAIN_SEED], query_mode=query_mode)
                reached = set(c.required) | set(c.optional)
                self.assertNotIn(F.TOOL_CHAIN_DISTRACTOR, reached)

    def test_03c_satisfiable_packet_has_no_dangling_required_parent(self):
        for name, graph, seed_ids in (
            ("tool_chain", F.tool_chain_graph(), [F.TOOL_CHAIN_SEED]),
            ("atomic_pair", F.atomic_pair_graph(), ["dec"]),
            ("exact_fit", F.exact_fit_graph(), [F.EXACT_FIT_SEED]),
            ("carrier", F.carrier_source_graph(), ["dec2"]),
        ):
            for query_mode in ("why", "audit", "state"):
                with self.subTest(graph=name, query_mode=query_mode):
                    c = _close(graph, seed_ids, query_mode=query_mode)
                    self.assertEqual([], _dangling_under_schema(c, graph))
                    seeds = [F.seed(e, rank=i) for i, e in enumerate(seed_ids)]
                    pkt = BudgetAssembler().assemble(
                        "q", c, graph, GENEROUS, seeds=seeds, query_mode=query_mode)
                    self.assertFalse(pkt.incomplete)
                    self.assertEqual((), pkt.manifest.missing_required)
                    self.assertEqual([], _dangling_in_packet(pkt, graph))

    def test_03d_random_graphs_never_dangle_under_their_own_policy(self):
        for rng_seed, g in F.random_graphs(8, n_events=20):
            ids = list(g.event_ids)
            picks = [ids[0], ids[len(ids) // 2], ids[-1]]
            for mode in ("native", "full_ancestor"):
                for query_mode in ("lookup", "why", "state", "audit"):
                    with self.subTest(seed=rng_seed, mode=mode, query_mode=query_mode):
                        c = _close(g, picks, mode=mode, query_mode=query_mode)
                        self.assertEqual([], _dangling_under_policy(c, g))
                        self.assertEqual(set(), set(c.required) & set(c.optional))

    def test_03e_lookup_mode_demotes_decision_sources_to_optional_not_to_the_void(self):
        """§3.3 policy: a plain lookup does not pay for a decision's sources -- but it must
        still SEE them, so they land in `optional`, never dropped."""
        g = F.tool_chain_graph()
        c = _close(g, [F.TOOL_CHAIN_SEED], query_mode="lookup")
        self.assertEqual(("d1",), c.required)
        self.assertEqual(set(F.TOOL_CHAIN_WHY_REQUIRED) - {"d1"}, set(c.optional))
        self.assertEqual([], _dangling_under_policy(c, g))
        # ... which is a *policy* divergence from the schema whitelist, and it is visible:
        self.assertNotEqual([], _dangling_under_schema(c, g))

    def test_03f_max_hops_truncation_must_not_silently_dangle(self):
        """GAP CLOSED 2026-09-01 (assembler frontier reporting).

        Original finding: `closure._expand` recorded a `max_hops_truncated` step and dropped the
        parent from BOTH tiers, so the assembler minted `incomplete=False, missing_required=()`
        for a packet whose last served event had an unserved DEPENDS_ON parent -- a *silent*
        dangling dependency.

        The assertion was rewritten because the property it encoded ("complete AND non-dangling")
        is unachievable by construction: a 10-long chain under max_hops=6 cannot be closed, so
        contract #4 (declare it) is what applies, not contract #3 (satisfy it).  What #3 really
        forbids is SILENCE, and that is what is asserted now: the truncated frontier appears in
        `missing_required` and the packet declares itself incomplete, so the dangle is auditable.
        `test_03h` covers the other half: with hops enough, nothing dangles and the packet is
        complete.
        """
        g = F.deep_chain_graph(10)
        c = TypedClosure(ClosureConfig(mode="native", max_hops=6)).close(
            "q", [F.seed(F.DEEP_CHAIN_SEED)], g, query_mode="why")
        truncated = [s for s in c.steps if s.rule == "max_hops_truncated"]
        self.assertTrue(truncated, "fixture must actually exceed max_hops")
        pkt = BudgetAssembler().assemble(
            "q", c, g, GENEROUS, seeds=[F.seed(F.DEEP_CHAIN_SEED)], query_mode="why")
        self.assertTrue(pkt.incomplete, "a truncated closure may not claim completeness")
        frontier = {st.parent_id for st in truncated}
        self.assertTrue(frontier <= set(pkt.manifest.missing_required),
                        "the unexpanded frontier must be reported, not dropped")
        # every dangling parent is either served or named in missing_required -- never silent.
        # _dangling_in_packet yields (child, parent, edge_type) triples; missing_required holds
        # bare ids, so compare the PARENT id, not the triple.
        flagged = set(pkt.manifest.missing_required)
        unflagged = [d for d in _dangling_in_packet(pkt, g) if d[1] not in flagged]
        self.assertEqual([], unflagged, "dangling dependency with nothing flagging it")

    def test_03h_enough_hops_gives_a_complete_non_dangling_packet(self):
        """The other half of #3: when the closure CAN finish, it must, and nothing dangles."""
        g = F.deep_chain_graph(10)
        c = TypedClosure(ClosureConfig(mode="native", max_hops=32)).close(
            "q", [F.seed(F.DEEP_CHAIN_SEED)], g, query_mode="why")
        self.assertFalse([s for s in c.steps if s.rule == "max_hops_truncated"])
        pkt = BudgetAssembler().assemble(
            "q", c, g, GENEROUS, seeds=[F.seed(F.DEEP_CHAIN_SEED)], query_mode="why")
        self.assertFalse(pkt.incomplete)
        self.assertEqual([], _dangling_in_packet(pkt, g))

    def test_03g_max_hops_truncation_is_at_least_recorded_in_the_audit_trail(self):
        """The half of #3f that DOES hold today: the loss is auditable, just not flagged."""
        g = F.deep_chain_graph(10)
        c = TypedClosure(ClosureConfig(mode="native", max_hops=6)).close(
            "q", [F.seed(F.DEEP_CHAIN_SEED)], g, query_mode="why")
        self.assertEqual(7, len(c.required), "seed + max_hops parents")
        truncated = [s for s in c.steps if s.rule == "max_hops_truncated"]
        self.assertEqual([("ch06", "ch07")], [(s.child_id, s.parent_id) for s in truncated])
        self.assertNotIn("ch07", set(c.required) | set(c.optional))


# ---------------------------------------------------------------- contract #4


class Contract04Unsatisfiable(unittest.TestCase):
    """§8.1 #4: when the evidence does not fit, the packet says so out loud."""

    def test_04a_oversized_required_event_is_reported_never_truncated(self):
        g = F.oversized_event_graph()
        c = _close(g, ["tiny"], query_mode="why")
        self.assertIn(F.OVERSIZED_ID, c.required)
        for budget in (0, 1, 1024, 2048, 4096):
            with self.subTest(budget=budget):
                pkt = BudgetAssembler().assemble(
                    "q", c, g, budget, seeds=[F.seed("tiny")], query_mode="why")
                self.assertTrue(pkt.incomplete)
                self.assertIn(F.OVERSIZED_ID, pkt.manifest.missing_required)
                self.assertNotIn(F.OVERSIZED_ID, pkt.event_ids)
                self.assertNotIn("PAYLOAD-BEGIN", pkt.context)
                self.assertLessEqual(pkt.manifest.total_tokens, budget)

    def test_04b_incomplete_implies_a_named_cause(self):
        """`incomplete=True` with an empty `missing_required` and no dropped header would be an
        unexplainable flag; §3.5 requires the packet to name what it lost."""
        g = F.oversized_event_graph()
        c = _close(g, ["tiny"], query_mode="why")
        pkt = BudgetAssembler().assemble("q", c, g, 100, seeds=[F.seed("tiny")], query_mode="why")
        self.assertTrue(pkt.incomplete)
        self.assertTrue(pkt.manifest.missing_required)

    def test_04c_satisfiable_packet_is_not_flagged_incomplete(self):
        g = F.exact_fit_graph()
        c = _close(g, [F.EXACT_FIT_SEED], query_mode="why")
        pkt = BudgetAssembler().assemble(
            "q", c, g, F.EXACT_FIT_SUM, seeds=[F.seed(F.EXACT_FIT_SEED)], query_mode="why")
        self.assertFalse(pkt.incomplete)
        self.assertEqual((), pkt.manifest.missing_required)
        self.assertEqual(F.EXACT_FIT_SUM, pkt.manifest.total_tokens)


# ---------------------------------------------------------------- contract #7


class Contract07Supersedes(unittest.TestCase):
    """§8.1 #7 / §3.3: a state query prefers the newest valid event."""

    def test_07a_any_point_of_the_chain_promotes_to_the_head(self):
        g = F.supersede_chain_graph()
        for entry in ("s_a", "s_b", "s_c"):
            with self.subTest(seed=entry):
                c = _close(g, [entry], query_mode="state", query="what is the deploy target")
                self.assertEqual((F.SUPERSEDE_LATEST,), c.required)

    def test_07b_stale_states_are_optional_never_required_and_never_dropped(self):
        g = F.supersede_chain_graph()
        c = _close(g, ["s_a"], query_mode="state", query="what is the deploy target")
        for stale in F.SUPERSEDE_STALE:
            self.assertNotIn(stale, c.required)
            self.assertIn(stale, c.optional)
        rules = {s.rule for s in c.steps}
        self.assertIn("state_prefers_latest", rules)
        self.assertLessEqual(rules, set(RULES))

    def test_07c_under_pressure_only_the_latest_survives(self):
        g = F.supersede_chain_graph()
        c = _close(g, ["s_a"], query_mode="state", query="what is the deploy target")
        pkt = BudgetAssembler().assemble(
            "q", c, g, 9, seeds=[F.seed("s_a")], query_mode="state")
        self.assertEqual((F.SUPERSEDE_LATEST,), pkt.event_ids)
        self.assertIn("ap-southeast-1", pkt.context)
        self.assertNotIn("us-east-1", pkt.context)

    def test_07d_when_a_stale_state_is_also_served_the_latest_is_served_too(self):
        g = F.supersede_chain_graph()
        c = _close(g, ["s_a"], query_mode="state", query="what is the deploy target")
        for budget in range(0, 40):
            with self.subTest(budget=budget):
                pkt = BudgetAssembler().assemble(
                    "q", c, g, budget, seeds=[F.seed("s_a")], query_mode="state")
                served = set(pkt.event_ids)
                if served & set(F.SUPERSEDE_STALE):
                    self.assertIn(F.SUPERSEDE_LATEST, served,
                                  "a stale state may accompany the answer, never replace it")

    def test_07e_fork_resolution_is_deterministic_and_order_free(self):
        base = [F.event(e, "state", i + 1, "branch %s" % e, 5)
                for i, e in enumerate(("f_a", "f_b", "f_c"))]
        fwd = TraceGraph(base, [F.TraceEdge("f_b", "f_a", "SUPERSEDES"),
                                F.TraceEdge("f_c", "f_a", "SUPERSEDES")])
        rev = TraceGraph(list(reversed(base)),
                         [F.TraceEdge("f_c", "f_a", "SUPERSEDES"),
                          F.TraceEdge("f_b", "f_a", "SUPERSEDES")])
        a = _close(fwd, ["f_a"], query_mode="state")
        b = _close(rev, ["f_a"], query_mode="state")
        self.assertEqual(("f_c",), a.required)
        self.assertEqual(a.required, b.required)
        self.assertEqual(a.optional, b.optional)

    def test_07f_supersedes_cycle_is_refused_not_spun(self):
        g = F.supersede_cycle_graph()
        with self.assertRaises(SchemaError):
            g.latest_state("cx")
        # closure has its own guard and must terminate with a bounded audit trail
        c = _close(g, ["cx"], query_mode="state")
        self.assertLessEqual(len(c.steps), len(g.edges) + len(g.events))
        self.assertTrue(set(c.required) | set(c.optional))

    def test_07g_baseline_arms_do_not_apply_the_rule_which_is_what_makes_them_baselines(self):
        """`off` and `full_ancestor` exist to measure what the native policy buys (§6.2).

        They return the seed as-is, stale and all -- recorded here so the number the profiler
        reports for those arms is understood rather than mistaken for a bug.
        """
        g = F.supersede_chain_graph()
        for mode in ("off", "full_ancestor"):
            with self.subTest(mode=mode):
                c = _close(g, ["s_a"], mode=mode, query_mode="state")
                self.assertEqual(("s_a",), c.required)
                self.assertNotIn(F.SUPERSEDE_LATEST, c.required)

    def test_07i_a_served_stale_state_must_be_labelled_as_superseded(self):
        """KNOWN GAP (see the module report): the manifest entry for a demoted state reads
        `seed:lexical`, exactly like fresh evidence.

        `assembler._candidates` gives `seed:<source>` the highest reason precedence, so the
        `state_prefers_latest` demotion that moved `s_a` out of `required` never reaches
        `PacketEntry.reason`.  In the served packet the stale state is rendered *before* the
        authoritative one (chronological order) and the manifest offers no way to tell them
        apart without cross-referencing `closure_steps` -- §3.5 asks the manifest to name why
        each event is present, and here it names the wrong thing.
        """
        g = F.supersede_chain_graph()
        c = _close(g, ["s_a"], query_mode="state", query="what is the deploy target")
        pkt = BudgetAssembler().assemble("q", c, g, 30, seeds=[F.seed("s_a")],
                                         query_mode="state")
        reasons = {e.event_id: e.reason for e in pkt.manifest.entries}
        self.assertIn("s_a", reasons, "fixture must actually serve the stale state here")
        self.assertNotEqual("seed:lexical", reasons["s_a"])

    def test_07h_assembler_refuses_to_relabel_a_closure_built_for_another_mode(self):
        g = F.supersede_chain_graph()
        c = _close(g, ["s_a"], query_mode="lookup")
        with self.assertRaises(SchemaError):
            BudgetAssembler().assemble("q", c, g, 100, query_mode="state")


# ---------------------------------------------------------------- contract #9


class Contract09CarrierVerification(unittest.TestCase):
    """§8.1 #9 / §4.5: a relaxation needs carrier id + predicate + model + both protocols."""

    def _close_with(self, verification=None, *, model_id=F.CARRIER_MODEL_ID,
                    readout=F.CARRIER_READOUT, query="why did we buy from vendor B"):
        g = F.carrier_source_graph()
        kw = {}
        if verification is not None:
            kw = dict(verifications=[verification], model_id=model_id, readout_protocol=readout)
        return g, _close(g, ["dec2"], query_mode="why", query=query, **kw)

    def test_09a_a_fully_matching_record_licenses_the_relaxation(self):
        g, c = self._close_with(F.carrier_verification())
        self.assertIn(F.CARRIER_ID, c.required)
        self.assertNotIn(F.CARRIER_SOURCE_ID, c.required)
        self.assertIn(F.CARRIER_SOURCE_ID, c.optional)
        self.assertEqual(("%s|%s" % (F.CARRIER_ID, F.CARRIER_PREDICATE),), c.relaxations)

    def test_09b_breaking_any_single_field_revokes_it(self):
        table = {
            "carrier_id": "some_other_carrier",
            "predicate": "shipping_lead_time",
            "model_id": "qwen3-8b@sha256:DIFFERENT",
            "construction_protocol": "annotated",
            "readout_protocol": "readout/v2",
        }
        for field, bad in sorted(table.items()):
            with self.subTest(field=field):
                _, c = self._close_with(F.carrier_verification(**{field: bad}))
                self.assertIn(F.CARRIER_SOURCE_ID, c.required,
                              "a mismatched %s must not free the source" % field)
                self.assertEqual((), c.relaxations)

    def test_09c_runtime_model_and_readout_must_match_too(self):
        """The record can be perfect and still not apply: it is bound to the *serving* model."""
        for kwargs in ({"model_id": "another-checkpoint"}, {"readout": "readout/v9"}):
            with self.subTest(**kwargs):
                _, c = self._close_with(F.carrier_verification(), **kwargs)
                self.assertIn(F.CARRIER_SOURCE_ID, c.required)
                self.assertEqual((), c.relaxations)

    def test_09d_an_unbound_verification_is_a_hard_error(self):
        g = F.carrier_source_graph()
        for kw in ({}, {"model_id": F.CARRIER_MODEL_ID},
                   {"readout_protocol": F.CARRIER_READOUT}):
            with self.subTest(**kw):
                with self.assertRaises(SchemaError):
                    TypedClosure().close("q", [F.seed("dec2")], g, query_mode="why",
                                         verifications=[F.carrier_verification()], **kw)

    def test_09e_verification_records_are_type_checked(self):
        g = F.carrier_source_graph()
        with self.assertRaises(SchemaError):
            TypedClosure().close("q", [F.seed("dec2")], g, query_mode="why",
                                 verifications=[{"carrier_id": F.CARRIER_ID}],
                                 model_id=F.CARRIER_MODEL_ID,
                                 readout_protocol=F.CARRIER_READOUT)

    def test_09f_a_materializes_edge_without_a_predicate_cannot_be_built(self):
        """§4.5 binds verification to the predicate, so an unlabelled carrier edge is illegal."""
        with self.assertRaises(SchemaError):
            F.TraceEdge(F.CARRIER_ID, F.CARRIER_SOURCE_ID, "MATERIALIZES")


# ---------------------------------------------------------------- contract #10


class Contract10NoUnverifiedRemoval(unittest.TestCase):
    """§8.1 #10 / §3.1: source-first is the default; a carrier earns relief, never assumes it."""

    def test_10a_no_verifications_at_all_keeps_the_source_required(self):
        g = F.carrier_source_graph()
        c = _close(g, ["dec2"], query_mode="why")
        self.assertIn(F.CARRIER_SOURCE_ID, c.required)
        self.assertEqual((), c.relaxations)
        rules = {s.rule for s in c.steps}
        self.assertIn("unverified_carrier_needs_source", rules)

    def test_10b_the_source_is_actually_served_when_the_budget_allows(self):
        g = F.carrier_source_graph()
        c = _close(g, ["dec2"], query_mode="why")
        pkt = BudgetAssembler().assemble(
            "q", c, g, GENEROUS, seeds=[F.seed("dec2")], query_mode="why")
        self.assertIn(F.CARRIER_SOURCE_ID, pkt.event_ids)
        self.assertIn("780.25", pkt.context)

    def test_10c_and_is_reported_missing_rather_than_dropped_when_it_does_not_fit(self):
        g = F.carrier_source_graph()
        c = _close(g, ["dec2"], query_mode="why")
        pkt = BudgetAssembler().assemble(
            "q", c, g, 40, seeds=[F.seed("dec2")], query_mode="why")
        self.assertIn(F.CARRIER_SOURCE_ID, pkt.manifest.missing_required)
        self.assertTrue(pkt.incomplete)

    def test_10d_carrier_only_rendering_still_does_not_remove_the_source_event(self):
        """§5.3: `carrier_only` chooses a cheaper *witness*, it does not delete evidence."""
        g = F.carrier_source_graph()
        c = _close(g, ["dec2"], query_mode="why")
        pkt = BudgetAssembler(AssemblerConfig(repr_policy="carrier_only")).assemble(
            "q", c, g, GENEROUS, seeds=[F.seed("dec2")], query_mode="why")
        self.assertIn(F.CARRIER_SOURCE_ID, pkt.event_ids)

    def test_10e_an_exact_payload_lookup_ignores_even_a_valid_verification(self):
        """§3.3: a predicate-only carrier cannot answer 'what was the exact number'."""
        g = F.carrier_source_graph()
        # seeded on the carrier itself: in `lookup` a *decision*'s sources are demoted (§3.3),
        # which would make this pass for the wrong reason.
        c = _close(g, [F.CARRIER_ID], query_mode="lookup",
                   query="what was the exact unit price number for sku Z-9?",
                   verifications=[F.carrier_verification()],
                   model_id=F.CARRIER_MODEL_ID, readout_protocol=F.CARRIER_READOUT)
        self.assertIn(F.CARRIER_SOURCE_ID, c.required)
        self.assertEqual((), c.relaxations)
        self.assertIn("exact_payload_no_carrier_stop", {s.rule for s in c.steps})
        # the same verification DOES apply once the query stops asking for a literal payload
        soft = _close(g, [F.CARRIER_ID], query_mode="lookup",
                      query="roughly how expensive was that vendor",
                      verifications=[F.carrier_verification()],
                      model_id=F.CARRIER_MODEL_ID, readout_protocol=F.CARRIER_READOUT)
        self.assertNotIn(F.CARRIER_SOURCE_ID, soft.required)


# ---------------------------------------------------------------- adversarial


class AdversarialClosure(unittest.TestCase):
    """The mistakes this project cares about, in closure's half of the pipeline."""

    def test_duplicate_event_ids_are_rejected(self):
        with self.assertRaises(SchemaError):
            TraceGraph(*F.duplicate_id_events())

    def test_dangling_edges_are_rejected(self):
        with self.assertRaises(SchemaError):
            TraceGraph(*F.dangling_edge_events())

    def test_self_edges_are_rejected_at_edge_construction(self):
        src, dst, edge_type = F.self_edge_events()[1][0]
        with self.assertRaises(SchemaError):
            F.TraceEdge(src, dst, edge_type)

    def test_a_dependency_cycle_terminates_with_a_bounded_audit_trail(self):
        g = F.cyclic_depends_graph()
        c = _close(g, ["cy_a"], query_mode="why")
        self.assertEqual(set(F.CYCLE_IDS), set(c.required))
        self.assertLessEqual(len(c.steps), len(g.edges),
                             "each node is scanned once; more steps than edges means spinning")

    def test_seeds_must_exist_in_the_graph(self):
        g = F.tool_chain_graph()
        with self.assertRaises(SchemaError):
            TypedClosure().close("q", [F.seed("not_an_event")], g, query_mode="why")

    def test_unknown_modes_are_rejected(self):
        with self.assertRaises(SchemaError):
            ClosureConfig(mode="magic")
        with self.assertRaises(SchemaError):
            TypedClosure().close("q", [], F.tool_chain_graph(), query_mode="telepathy")
        with self.assertRaises(SchemaError):
            ClosureConfig(max_hops=-1)

    def test_seed_records_are_type_checked(self):
        g = F.tool_chain_graph()
        with self.assertRaises(SchemaError):
            TypedClosure().close("q", ["d1"], g, query_mode="why")
        with self.assertRaises(SchemaError):
            Seed(event_id="d1", score=1.0, source="vibes")

    def test_oracle_mode_refuses_to_invent_events(self):
        g = F.tool_chain_graph()
        tc = TypedClosure(ClosureConfig(mode="oracle"))
        with self.assertRaises(SchemaError):
            tc.close("q", [F.seed("d1")], g, query_mode="why")
        with self.assertRaises(SchemaError):
            tc.close("q", [F.seed("d1")], g, query_mode="why",
                     oracle_required={"q": ["ghost"]})
        c = tc.close("q", [F.seed("d1")], g, query_mode="why",
                     oracle_required={"q": ["tr1"]})
        self.assertEqual(("tr1", "d1"), c.required)

    def test_required_and_optional_are_always_disjoint_and_ordered(self):
        for rng_seed, g in F.random_graphs(6, n_events=18):
            ids = list(g.event_ids)
            for query_mode in ("lookup", "why", "state", "audit"):
                with self.subTest(seed=rng_seed, query_mode=query_mode):
                    c = _close(g, [ids[0], ids[-1]], query_mode=query_mode)
                    self.assertEqual(set(), set(c.required) & set(c.optional))
                    chrono = {e: i for i, e in enumerate(g.event_ids)}
                    self.assertEqual(list(c.required), sorted(c.required, key=chrono.get))
                    self.assertEqual(list(c.optional), sorted(c.optional, key=chrono.get))


# ---------------------------------------------------------------- selfcheck


def _selfcheck() -> None:
    """Run this module's suite, then fault-inject the instruments it trusts.

    A checker that cannot report a fault proves nothing about the code it checks, so
    `_dangling_under_policy`, `_dangling_under_schema` and `_dangling_in_packet` are each fed a
    known-broken input and must name the missing id.
    """
    from tracepack.core.schema import ClosureStep

    graph = F.tool_chain_graph()

    # ---- instrument 1: a closure whose step says "required" but whose set forgot the parent
    broken = EvidenceClosure(
        seeds=("d1",), required=("d1",), optional=(),
        steps=(ClosureStep(child_id="d1", parent_id="tr1", edge_type="DEPENDS_ON",
                           rule="native_required_edge"),),
        query_mode="why")
    found = _dangling_under_policy(broken, graph)
    assert found and found[0][:2] == ("d1", "tr1"), \
        "_dangling_under_policy failed to report an injected dangling parent: %r" % (found,)
    good = _close(graph, ["d1"], query_mode="why")
    assert _dangling_under_policy(good, graph) == []

    # ---- instrument 2: schema-level checker
    assert _dangling_under_schema(broken, graph), "_dangling_under_schema missed the injection"
    assert _dangling_under_schema(good, graph) == []

    # ---- instrument 3: packet-level checker, fed a packet with a parent removed
    class _FakePacket:
        event_ids = ("d1",)
    assert _dangling_in_packet(_FakePacket(), graph), "_dangling_in_packet missed the injection"
    pkt = BudgetAssembler().assemble("q", good, graph, GENEROUS, seeds=[F.seed("d1")],
                                     query_mode="why")
    assert _dangling_in_packet(pkt, graph) == []

    # ---- the fixture's own claim must be true before any test leans on it
    assert set(F.TOOL_CHAIN_WHY_REQUIRED) == set(good.required)

    suite = unittest.TestLoader().loadTestsFromModule(sys.modules[__name__])
    result = unittest.TextTestRunner(verbosity=1, stream=sys.stdout).run(suite)
    if not result.wasSuccessful():
        raise SystemExit("test_closure: %d failure(s), %d error(s)"
                         % (len(result.failures), len(result.errors)))
    print("test_closure selfcheck OK: %d tests, %d expected failure(s) (known gaps), "
          "3 instruments fault-injected" % (result.testsRun, len(result.expectedFailures)))


if __name__ == "__main__":
    _selfcheck()
