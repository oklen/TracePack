"""tracepack.tests.test_adapter -- proposal §8.1 contract 8 (+ the parsing traps of §4.2/§7.1).

| §8.1 | claim                                                          | class here                |
|------|----------------------------------------------------------------|---------------------------|
| #8   | adapter round-trip preserves native event IDs and strong edges  | `Contract08RoundTrip`     |

Contracts 1-7, 9 and 10 live in the other three test modules.

Design decisions in these tests
-------------------------------

* **The order-vs-id pairing trap is tested with a fixture where the two DISAGREE on every
  pair.**  DESIGN_FROZEN §0 measured order pairing at 96.53% agreement over 32,134 real pairs
  (31.8% in the worst single session) -- which means a regression to order pairing passes a
  spot check 96 times out of 100 and never raises.  ``crossed_tool_rows`` answers three calls in
  the order C, A, B, so id pairing and order pairing share *no* correct answer and the test
  cannot pass by luck.  ``test_08c`` asserts both directions: the graph equals id pairing AND
  differs from order pairing.

* **The auditor is audited.**  ``test_08j`` builds an adapter that pairs by order, feeds it to
  ``roundtrip_check``, and requires an ``AdapterError``.  A green ``roundtrip_check`` on the good
  adapter means nothing unless the same check is known to go red on a bad one -- this is the
  single most valuable test in the file.

* **Duplication is measured in tokens, not in strings.**  Trap 1 (`toolUseResult` repeats the
  payload) inflates exactly the events that dominate a coding trace.  ``test_08e`` asserts the
  decoy text is absent *and* that the total token cost is small, because a future adapter could
  read the duplicate into ``meta`` and pass a substring check while doubling the budget numbers.

* **Round-trip means all the way to a packet.**  ``test_08k`` runs rows -> graph -> router ->
  closure -> assembler and checks the §3.4 budget on the result, which is the DESIGN_FROZEN §5
  "Adapter gate" verbatim.

Pure stdlib + unittest; the transcript is a list of dicts, so no file I/O and no network.
"""
from __future__ import annotations

import os
import sys
import unittest

if __package__ in (None, ""):  # pragma: no cover - direct execution
    sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(
        os.path.abspath(__file__)))))

from tracepack.adapters.base import AdapterError, build_graph, roundtrip_check
from tracepack.adapters.claude_code import ClaudeCodeAdapter
from tracepack.core.assembler import BudgetAssembler
from tracepack.core.closure import ClosureConfig, TypedClosure
from tracepack.core.graph import TraceGraph
from tracepack.core.router import RouterConfig, make_router
from tracepack.core.schema import STRONG_EDGES, SchemaError, TraceEdge
from tracepack.tests.fixtures import tiny_trace as F


# ---------------------------------------------------------------- instruments


def result_of_by_native_ref(graph: TraceGraph) -> dict:
    """``{result_row_uuid: call_row_uuid}`` read back out of the graph's RESULT_OF edges.

    Expressed in *native* ids on both sides so it can be compared directly with the fixture's
    hand-written pairing tables; comparing internal event ids would only prove the graph is
    self-consistent.
    """
    ref = {e.event_id: e.native_ref for e in graph.events}
    out = {}
    for edge in graph.edges:
        if edge.edge_type != "RESULT_OF":
            continue
        src, dst = ref.get(edge.src_id), ref.get(edge.dst_id)
        if src is None or dst is None:
            raise AssertionError("RESULT_OF edge lost its native provenance: %r" % (edge,))
        out[src] = dst
    return out


def graph_signature(graph: TraceGraph) -> tuple:
    """Everything about a normalized graph a caller can observe -- for purity checks."""
    return (
        tuple((e.event_id, e.kind, e.text, e.timestamp, e.step_id, e.tool_call_id,
               e.native_ref, e.token_cost, e.atomic_group, tuple(sorted(e.meta.items())))
              for e in graph.events),
        tuple((e.src_id, e.dst_id, e.edge_type, e.predicate, e.provenance) for e in graph.edges),
    )


class OrderPairingAdapter(ClaudeCodeAdapter):
    """FAULT INJECTION: the same adapter, but RESULT_OF edges rewired by arrival order.

    This is the regression DESIGN_FROZEN §0 forbids, written out so ``roundtrip_check`` can be
    shown to catch it.  ``expected_strong_edges`` is inherited unchanged (it resolves by
    ``tool_use_id``), which is exactly the independence that makes the audit work.
    """

    def normalize(self, native_trace) -> TraceGraph:
        graph = super().normalize(native_trace)
        calls = [e for e in graph.events if e.kind == "tool_call"]
        results = [e for e in graph.events if e.kind == "tool_result"]
        kept = [e for e in graph.edges if e.edge_type != "RESULT_OF"]
        for res, call in zip(results, calls):          # <-- the bug: positional pairing
            kept.append(TraceEdge(res.event_id, call.event_id, "RESULT_OF",
                                  predicate="arrival_order", provenance="native"))
        return build_graph(list(graph.events), kept)


class DroppingRefAdapter(ClaudeCodeAdapter):
    """FAULT INJECTION: normalization that forgets ``native_ref`` -- contract #8's other half."""

    def normalize(self, native_trace) -> TraceGraph:
        graph = super().normalize(native_trace)
        stripped = [
            F.TraceEvent(event_id=e.event_id, kind=e.kind, text=e.text, timestamp=e.timestamp,
                         step_id=e.step_id, tool_call_id=e.tool_call_id, native_ref=None,
                         token_cost=e.token_cost, representations=e.representations,
                         atomic_group=e.atomic_group, meta=e.meta)
            for e in graph.events]
        return build_graph(stripped, list(graph.edges))


# ---------------------------------------------------------------- contract #8


class Contract08RoundTrip(unittest.TestCase):
    """§8.1 #8: native ids and strong edges survive `normalize`."""

    def setUp(self):
        self.rows = F.crossed_tool_rows()
        self.adapter = ClaudeCodeAdapter()
        self.graph = self.adapter.normalize(self.rows)

    def test_08a_roundtrip_check_passes_on_the_reference_adapter(self):
        stats = roundtrip_check(self.adapter, self.rows)
        self.assertTrue(stats["ok"])
        self.assertEqual((), stats["failures"])
        self.assertEqual("adapter", stats["native_ids_source"],
                         "the strong form of the check needs adapter.native_ids()")
        self.assertEqual(stats["n_native_ids_expected"], stats["n_native_ids_present"])
        self.assertEqual((), stats["missing_native_ids"])
        self.assertEqual((), stats["missing_strong"])
        self.assertEqual(0, stats["n_strong_without_provenance"])

    def test_08b_every_retained_native_uuid_appears_as_a_native_ref(self):
        declared = set(self.adapter.native_ids(self.rows))
        present = {e.native_ref for e in self.graph.events}
        self.assertTrue(declared)
        self.assertTrue(declared <= present)
        # ... and nothing was invented: every native_ref came from a real row
        row_uuids = {r["uuid"] for r in self.rows if "uuid" in r}
        self.assertTrue(present <= row_uuids)

    def test_08c_result_of_edges_are_paired_by_id_and_not_by_order(self):
        actual = result_of_by_native_ref(self.graph)
        self.assertEqual(F.id_pairing_truth(), actual)
        wrong = F.order_pairing_would_give()
        self.assertNotEqual(wrong, actual)
        # every single pair differs, so this cannot pass by coincidence
        for result_uuid, call_uuid in wrong.items():
            self.assertNotEqual(call_uuid, actual[result_uuid])

    def test_08d_each_call_result_pair_shares_one_atomic_group(self):
        by_id = {e.event_id: e for e in self.graph.events}
        pairs = [e for e in self.graph.edges if e.edge_type == "RESULT_OF"]
        self.assertEqual(3, len(pairs))
        for edge in pairs:
            res, call = by_id[edge.src_id], by_id[edge.dst_id]
            self.assertIsNotNone(res.atomic_group)
            self.assertEqual(res.atomic_group, call.atomic_group)
            self.assertEqual(res.tool_call_id, call.tool_call_id)

    def test_08e_the_duplicated_toolUseResult_payload_is_never_read(self):
        """Trap 1: reading both copies silently doubles every tool-result token count."""
        for ev in self.graph.events:
            self.assertNotIn("DUPLICATE-MUST-NOT-BE-READ", ev.text)
            for value in ev.meta.values():
                self.assertNotIn("DUPLICATE-MUST-NOT-BE-READ", str(value))
        total = sum(e.token_cost for e in self.graph.events)
        self.assertLess(total, 200, "the decoy payload alone would be ~250 tokens")

    def test_08f_snapshot_rows_never_become_events(self):
        """Trap 2: `file-history-snapshot` is 24.5% of real transcript bytes, and is editor
        state rather than conversation."""
        self.assertNotIn("x-snap", {e.native_ref for e in self.graph.events})
        self.assertEqual(7, len(self.graph.events))

    def test_08g_sidechain_rows_are_dropped_by_default_and_marked_when_kept(self):
        """Trap 3: a Task/subagent run interleaved into the same file is a different context."""
        rows = list(self.rows) + [
            {"type": "assistant", "uuid": "x-side", "isSidechain": True,
             "timestamp": "2026-09-01T00:00:08.000Z",
             "message": {"role": "assistant",
                         "content": [{"type": "text", "text": "subagent chatter"}]}}]
        default_graph = ClaudeCodeAdapter().normalize(rows)
        self.assertNotIn("x-side", {e.native_ref for e in default_graph.events})
        kept = ClaudeCodeAdapter(include_sidechain=True).normalize(rows)
        side = [e for e in kept.events if e.native_ref == "x-side"]
        self.assertEqual(1, len(side))
        self.assertEqual("1", side[0].meta.get("sidechain"))

    def test_08h_normalize_is_pure(self):
        again = ClaudeCodeAdapter().normalize(self.rows)
        self.assertEqual(graph_signature(self.graph), graph_signature(again))
        third = self.adapter.normalize(list(self.rows))
        self.assertEqual(graph_signature(self.graph), graph_signature(third))

    def test_08i_strong_edges_only_carry_evidence_types(self):
        strong = self.graph.strong_subgraph()
        self.assertTrue(strong)
        for edge in strong:
            self.assertIn(edge.edge_type, STRONG_EDGES)
        self.assertFalse([e for e in strong if e.edge_type in ("CONTROL", "TEMPORAL")])
        # and the weak edges really are present -- they are just not strong
        types = {e.edge_type for e in self.graph.edges}
        self.assertIn("CONTROL", types)
        self.assertIn("TEMPORAL", types)

    def test_08j_the_roundtrip_auditor_catches_an_order_pairing_adapter(self):
        """VERIFY THE INSTRUMENT.  If this passes silently, `test_08a` proves nothing."""
        broken = OrderPairingAdapter()
        wrong_graph = broken.normalize(self.rows)
        self.assertEqual(F.order_pairing_would_give(), result_of_by_native_ref(wrong_graph))
        with self.assertRaises(AdapterError):
            roundtrip_check(broken, self.rows)
        soft = roundtrip_check(broken, self.rows, strict=False)
        self.assertFalse(soft["ok"])
        self.assertEqual(3, len(soft["missing_strong"]))

    def test_08k_the_auditor_also_catches_a_lost_native_ref(self):
        broken = DroppingRefAdapter()
        with self.assertRaises(AdapterError):
            roundtrip_check(broken, self.rows)
        soft = roundtrip_check(broken, self.rows, strict=False)
        self.assertFalse(soft["ok"])
        self.assertTrue(soft["missing_native_ids"])

    def test_08l_native_trace_to_packet_is_a_complete_round_trip(self):
        """DESIGN_FROZEN §5 adapter gate: native trace -> IR -> packet -> native context."""
        query = "what did the SAFETY_MARGIN grep return?"
        seeds = make_router("hybrid_pin", RouterConfig(k=4)).retrieve(query, self.graph, 4)
        self.assertTrue(seeds)
        closure = TypedClosure(ClosureConfig(mode="native")).close(
            query, seeds, self.graph, query_mode="why")
        for budget in (1024, 2048, 4096):
            with self.subTest(budget=budget):
                packet = BudgetAssembler().assemble(query, closure, self.graph, budget,
                                                    seeds=seeds, query_mode="why")
                self.assertLessEqual(packet.manifest.total_tokens, budget)
                self.assertEqual(packet.context, self.adapter.render(packet))
                for entry in packet.manifest.entries:
                    self.assertIn(self.graph.event(entry.event_id).text, packet.context)


# ---------------------------------------------------------------- adversarial


class AdversarialAdapter(unittest.TestCase):
    """Malformed native input, and the two mistakes that corrupt every number downstream."""

    def setUp(self):
        self.rows = F.crossed_tool_rows()

    def test_duplicate_event_ids_and_dangling_edges_are_rejected_at_build(self):
        with self.assertRaises(SchemaError):
            build_graph(*F.duplicate_id_events())
        with self.assertRaises(SchemaError):
            build_graph(*F.dangling_edge_events())

    def test_a_result_whose_call_is_absent_gets_no_invented_edge(self):
        """Compacted-away or resumed sessions: the event survives, the edge does not."""
        orphan = [
            {"type": "user", "uuid": "o-1", "timestamp": "2026-09-01T00:00:01.000Z",
             "message": {"role": "user", "content": [
                 {"type": "tool_result", "tool_use_id": "toolu_GONE",
                  "content": "result of a call that is not in this file"}]}}]
        graph, stats = ClaudeCodeAdapter().normalize_with_stats(orphan)
        self.assertEqual(1, len(graph.events))
        self.assertEqual(1, stats.get("unpaired_tool_results"))
        self.assertEqual([], [e for e in graph.edges if e.edge_type == "RESULT_OF"])
        self.assertEqual("toolu_GONE", graph.events[0].tool_call_id)

    def test_malformed_rows_raise_instead_of_being_guessed_at(self):
        adapter = ClaudeCodeAdapter()
        for bad in (42, "a string row", None, ["nested list"]):
            with self.subTest(row=bad):
                with self.assertRaises(AdapterError):
                    adapter.normalize([bad])
        with self.assertRaises(AdapterError):
            adapter.normalize({"type": "user"})           # a single dict is not a transcript
        with self.assertRaises(AdapterError):
            adapter.normalize("/no/such/transcript.jsonl")

    def test_a_content_free_row_is_skipped_and_counted_not_turned_into_an_empty_event(self):
        """An event with no text would be charged against the budget and answer nothing."""
        graph, stats = ClaudeCodeAdapter().normalize_with_stats(
            [{"type": "user", "uuid": "empty-1"},
             {"type": "assistant", "uuid": "empty-2",
              "message": {"role": "assistant", "content": [{"type": "text", "text": "   "}]}}])
        self.assertEqual(0, len(graph.events))
        self.assertEqual(2, stats.get("rows_no_blocks"))

    def test_adapter_options_are_validated_at_construction(self):
        for kwargs in ({"source_id": ""}, {"source_id": "has:colon"}, {"quote_min_chars": 1},
                       {"max_quote_edges": -1}):
            with self.subTest(**kwargs):
                with self.assertRaises(AdapterError):
                    ClaudeCodeAdapter(**kwargs)

    def test_roundtrip_check_rejects_a_non_adapter(self):
        with self.assertRaises(AdapterError):
            roundtrip_check(object(), self.rows)

    def test_step_ids_are_zero_padded_so_line_order_survives_sorting(self):
        """Unpadded "9" > "10" would scramble events that share a timestamp."""
        graph = ClaudeCodeAdapter().normalize(self.rows)
        widths = {len(e.step_id) for e in graph.events}
        self.assertEqual({8}, widths)
        self.assertEqual(sorted(graph.event_ids), list(graph.event_ids),
                         "padded ids must already be in transcript order")

    def test_the_graph_the_adapter_produces_satisfies_the_ir_invariants(self):
        graph = ClaudeCodeAdapter().normalize(self.rows)
        ids = [e.event_id for e in graph.events]
        self.assertEqual(len(ids), len(set(ids)))
        known = set(ids)
        for edge in graph.edges:
            self.assertIn(edge.src_id, known)
            self.assertIn(edge.dst_id, known)
            self.assertNotEqual(edge.src_id, edge.dst_id)
        # rebuilding from the same parts must be accepted, i.e. the output is a legal input
        TraceGraph(list(graph.events), list(graph.edges))


# ---------------------------------------------------------------- selfcheck


def _selfcheck() -> None:
    """Run the suite, then fault-inject the instruments this file trusts."""
    rows = F.crossed_tool_rows()
    good = ClaudeCodeAdapter().normalize(rows)

    # ---- instrument 1: the pairing reader must see the difference between the two policies
    truth = result_of_by_native_ref(good)
    wrong = result_of_by_native_ref(OrderPairingAdapter().normalize(rows))
    assert truth == F.id_pairing_truth(), "result_of_by_native_ref misread the good adapter"
    assert wrong == F.order_pairing_would_give(), \
        "result_of_by_native_ref cannot see an order-paired graph: %r" % (wrong,)
    assert truth != wrong

    # ---- instrument 2: roundtrip_check must go red on the injected regression
    try:
        roundtrip_check(OrderPairingAdapter(), rows)
    except AdapterError:
        pass
    else:  # pragma: no cover
        raise AssertionError("roundtrip_check accepted an order-pairing adapter")
    try:
        roundtrip_check(DroppingRefAdapter(), rows)
    except AdapterError:
        pass
    else:  # pragma: no cover
        raise AssertionError("roundtrip_check accepted an adapter that dropped native_ref")
    assert roundtrip_check(ClaudeCodeAdapter(), rows)["ok"]

    # ---- instrument 3: graph_signature must distinguish two genuinely different graphs
    assert graph_signature(good) == graph_signature(ClaudeCodeAdapter().normalize(rows))
    assert graph_signature(good) != graph_signature(
        ClaudeCodeAdapter(include_thinking=True, source_id="other").normalize(rows)), \
        "graph_signature ignores the adapter policy that produced the graph"

    suite = unittest.TestLoader().loadTestsFromModule(sys.modules[__name__])
    result = unittest.TextTestRunner(verbosity=1, stream=sys.stdout).run(suite)
    if not result.wasSuccessful():
        raise SystemExit("test_adapter: %d failure(s), %d error(s)"
                         % (len(result.failures), len(result.errors)))
    print("test_adapter selfcheck OK: %d tests, id-vs-order pairing disagrees on 3/3 pairs, "
          "3 instruments fault-injected" % result.testsRun)


if __name__ == "__main__":
    _selfcheck()
