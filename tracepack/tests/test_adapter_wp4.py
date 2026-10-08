"""tracepack.tests.test_adapter_wp4 -- contract 15: the WP4 adapters (pi, OpenHands) pass the same
round-trip audit as the reference adapter, and the registry routes every format to its adapter
without touching Claude Code graphs.

| §8.1 | claim                                                                 | class here                  |
|------|-----------------------------------------------------------------------|-----------------------------|
| #15  | second-format adapters: id pairing, atomic groups, compaction, audit   | `Contract15SecondAdapters`  |

Design decisions
----------------
* **The pi fixture crosses tool results across entries AND inside one entry** (A and B share an
  assistant entry, C has its own; results come back C, A, B).  Order pairing therefore fails the
  declared-edge check on two pairs and the atomic-group check on the third -- both halves of the
  audit are exercised by one fixture, and the test asserts the audit goes red on the fault
  injection before trusting that it is green on the real adapter.
* **OpenHands has no ids to pair by**, so the test does not pretend otherwise: it asserts the
  position rule, the `predicate="position"` label, and that the audit still catches a split
  atomic group (the half of contract #8 that remains meaningful there).
* **The registry is tested by content, not by name**: a temp file whose first line is a pi
  header goes to `PiAdapter`, an exported OpenHands header to `OpenHandsAdapter`, a Claude Code
  row to `ClaudeCodeAdapter` -- and the Claude Code graph the registry builds is identical to the
  one the reference adapter builds directly (the byte-identity contract #13 depends on).
"""
from __future__ import annotations

import json
import os
import sys
import tempfile
import unittest
from collections import Counter

if __package__ in (None, ""):  # pragma: no cover - direct execution
    sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

from tracepack.adapters.base import AdapterError, build_graph, roundtrip_check
from tracepack.adapters.claude_code import ClaudeCodeAdapter
from tracepack.adapters.openhands import OpenHandsAdapter, messages_to_rows, synthetic_trajectory
from tracepack.adapters.pi import PiAdapter, synthetic_session
from tracepack.adapters.registry import adapter_for, format_of
from tracepack.core.schema import TraceEdge, TraceEvent
from tracepack.tests.fixtures import tiny_trace as F


def _ref(graph):
    return {e.event_id: e.native_ref for e in graph.events}


class Contract15SecondAdapters(unittest.TestCase):
    """§8.1 #15: pi / OpenHands adapters keep native ids, id-pair results, and survive the audit."""

    def setUp(self):
        self.pi_rows = synthetic_session()
        self.pi = PiAdapter()
        self.oh_rows = messages_to_rows(synthetic_trajectory(), {"instance_id": "demo__app-1", "repo": "demo/app"})
        self.oh = OpenHandsAdapter()

    # ---------------------------------------------------------------- pi

    def test_15a_pi_roundtrip_audit_is_green_in_its_strong_form(self):
        st = roundtrip_check(self.pi, self.pi_rows)
        self.assertTrue(st["ok"], st)
        self.assertEqual("adapter", st["native_ids_source"])
        self.assertEqual(st["n_native_ids_expected"], st["n_native_ids_present"])
        self.assertEqual(5, st["n_declared_strong"])

    def test_15b_pi_pairs_results_by_toolCallId_not_by_order(self):
        g = self.pi.normalize(self.pi_rows)
        ref = _ref(g)
        pairs = {ref[e.src_id]: ref[e.dst_id] for e in g.edges if e.edge_type == "RESULT_OF"}
        self.assertEqual({"r1": "a1c", "r2": "a1", "r3": "a1", "r4": "a2", "r5": "a3"}, pairs)
        order_would_give = {"r1": "a1", "r2": "a1", "r3": "a1c"}
        for r, wrong in order_would_give.items():
            self.assertNotEqual(wrong, pairs[r] if r != "r2" else "a1c")
        for e in g.edges:
            if e.edge_type == "RESULT_OF":
                self.assertEqual("toolCallId", e.predicate)

    def test_15c_pi_call_and_result_share_one_atomic_group(self):
        g = self.pi.normalize(self.pi_rows)
        calls = {e.tool_call_id: e for e in g.events if e.kind == "tool_call"}
        results = {e.tool_call_id: e for e in g.events if e.kind == "tool_result"}
        self.assertEqual(set(calls), set(results))
        for tid, c in calls.items():
            self.assertIsNotNone(c.atomic_group)
            self.assertEqual(c.atomic_group, results[tid].atomic_group)

    def test_15d_pi_compaction_materializes_everything_before_firstKeptEntryId(self):
        g = self.pi.normalize(self.pi_rows)
        ref = _ref(g)
        summ = [e for e in g.events if e.kind == "summary"]
        self.assertEqual(1, len(summ))
        self.assertEqual("1", summ[0].meta.get("compaction"))
        mat = sorted({ref[e.dst_id] for e in g.edges if e.edge_type == "MATERIALIZES"})
        self.assertEqual(["a1", "a1c", "a2", "r1", "r2", "r3", "r4", "u1"], mat)
        self.assertNotIn("a3", mat)          # kept entries are not materialized
        self.assertFalse([e for e in g.edges if e.edge_type == "SUPERSEDES" and e.src_id == summ[0].event_id],
                         "compaction must never SUPERSEDE its sources (§3.1)")

    def test_15e_pi_edit_quote_and_write_path_edges_follow_the_reference_policy(self):
        g = self.pi.normalize(self.pi_rows)
        ref = _ref(g)
        dep = [(ref[e.src_id], ref[e.dst_id], e.predicate, e.provenance) for e in g.edges if e.edge_type == "DEPENDS_ON"]
        self.assertEqual([("a2", "r2", "literal_quote>=40", "inferred")], dep)
        sup = [(ref[e.src_id], ref[e.dst_id], e.provenance) for e in g.edges if e.edge_type == "SUPERSEDES"]
        self.assertEqual([("a3", "a2", "inferred")], sup)

    def test_15f_pi_thinking_is_opt_in_and_state_entries_are_dropped(self):
        default = Counter(e.kind for e in self.pi.normalize(self.pi_rows).events)
        with_thinking = Counter(e.kind for e in PiAdapter(include_thinking=True).normalize(self.pi_rows).events)
        self.assertEqual(default["assistant"] + 1, with_thinking["assistant"])
        _, stats = self.pi.normalize_with_stats(self.pi_rows)
        self.assertEqual(1, stats["rows_filtered_type"])       # the model_change entry
        self.assertEqual(0, stats.get("rows_unknown_type", 0))

    def test_15g_the_audit_rejects_an_order_pairing_pi_adapter(self):
        class OrderPairing(PiAdapter):
            def normalize(self, native_trace):
                g = super().normalize(native_trace)
                calls = [e for e in g.events if e.kind == "tool_call"]
                results = [e for e in g.events if e.kind == "tool_result"]
                kept = [e for e in g.edges if e.edge_type != "RESULT_OF"]
                for res, call in zip(results, calls):
                    kept.append(TraceEdge(res.event_id, call.event_id, "RESULT_OF", predicate="order", provenance="native"))
                return build_graph(list(g.events), kept)

        with self.assertRaises(AdapterError):
            roundtrip_check(OrderPairing(), self.pi_rows)
        soft = roundtrip_check(OrderPairing(), self.pi_rows, strict=False)
        self.assertFalse(soft["ok"])
        self.assertEqual(2, len(soft["missing_strong"]))                 # the two cross-entry pairs
        self.assertTrue(any("atomic_group" in f for f in soft["failures"]))   # the within-entry pair

    def test_15h_pi_normalize_is_pure(self):
        a = self.pi.normalize(self.pi_rows)
        b = self.pi.normalize(list(self.pi_rows))
        self.assertEqual([(e.event_id, e.kind, e.text, e.timestamp, e.atomic_group) for e in a.events],
                         [(e.event_id, e.kind, e.text, e.timestamp, e.atomic_group) for e in b.events])
        self.assertEqual([(e.src_id, e.dst_id, e.edge_type, e.predicate) for e in a.edges],
                         [(e.src_id, e.dst_id, e.edge_type, e.predicate) for e in b.edges])

    # ---------------------------------------------------------- openhands

    def test_15i_openhands_pairs_by_position_and_says_so(self):
        st = roundtrip_check(self.oh, self.oh_rows)
        self.assertTrue(st["ok"], st)
        g = self.oh.normalize(self.oh_rows)
        ref = _ref(g)
        pairs = {ref[e.src_id]: ref[e.dst_id] for e in g.edges if e.edge_type == "RESULT_OF"}
        self.assertEqual({"m0003": "m0002", "m0005": "m0004", "m0007": "m0006", "m0009": "m0008"}, pairs)
        self.assertTrue(all(e.predicate == "position" for e in g.edges if e.edge_type == "RESULT_OF"))
        calls = {e.tool_call_id: e for e in g.events if e.kind == "tool_call"}
        results = {e.tool_call_id: e for e in g.events if e.kind == "tool_result"}
        for tid in calls:
            self.assertEqual(calls[tid].atomic_group, results[tid].atomic_group)

    def test_15j_openhands_edit_edges_and_unpaired_results(self):
        g, st = self.oh.normalize_with_stats(self.oh_rows)
        ref = _ref(g)
        self.assertEqual([("m0006", "m0005")], [(ref[e.src_id], ref[e.dst_id]) for e in g.edges if e.edge_type == "DEPENDS_ON"])
        self.assertEqual([("m0008", "m0006")], [(ref[e.src_id], ref[e.dst_id]) for e in g.edges if e.edge_type == "SUPERSEDES"])
        self.assertEqual(1, st["rows_filtered_system"])
        g2, st2 = self.oh.normalize_with_stats(self.oh_rows + [{"id": "m0011", "role": "tool", "content": "stray"}])
        self.assertEqual(1, st2["unpaired_tool_results"])
        self.assertEqual(len(g.edges) - len([e for e in g.edges if e.edge_type == "TEMPORAL"]),
                         len(g2.edges) - len([e for e in g2.edges if e.edge_type == "TEMPORAL"]))

    def test_15k_the_audit_rejects_a_split_atomic_group_openhands_adapter(self):
        class SplitGroups(OpenHandsAdapter):
            def normalize(self, native_trace):
                g = super().normalize(native_trace)
                evs = [TraceEvent(event_id=e.event_id, kind=e.kind, text=e.text, timestamp=e.timestamp, step_id=e.step_id,
                                  tool_call_id=e.tool_call_id, native_ref=e.native_ref, token_cost=e.token_cost,
                                  representations=e.representations,
                                  atomic_group=(None if e.kind == "tool_result" else e.atomic_group), meta=e.meta)
                       for e in g.events]
                return build_graph(evs, list(g.edges))

        with self.assertRaises(AdapterError):
            roundtrip_check(SplitGroups(), self.oh_rows)

    # ----------------------------------------------------------- registry

    def test_15l_registry_routes_by_content_and_leaves_claude_code_graphs_untouched(self):
        with tempfile.TemporaryDirectory() as d:
            pi_p, oh_p, cc_p = [os.path.join(d, n) for n in ("a.jsonl", "b.jsonl", "c.jsonl")]
            with open(pi_p, "w") as fo:
                for r in self.pi_rows:
                    fo.write(json.dumps(r) + "\n")
            with open(oh_p, "w") as fo:
                for r in self.oh_rows:
                    fo.write(json.dumps(r) + "\n")
            with open(cc_p, "w") as fo:
                for r in F.crossed_tool_rows():
                    fo.write(json.dumps(r) + "\n")
            self.assertEqual(("pi", "openhands", "claude_code"), (format_of(pi_p), format_of(oh_p), format_of(cc_p)))
            self.assertIsInstance(adapter_for(pi_p), PiAdapter)
            self.assertIsInstance(adapter_for(oh_p), OpenHandsAdapter)
            self.assertIsInstance(adapter_for(cc_p), ClaudeCodeAdapter)
            via_registry = adapter_for(cc_p).normalize(cc_p)
            direct = ClaudeCodeAdapter().normalize(cc_p)
            self.assertEqual([(e.event_id, e.text, e.timestamp, e.atomic_group) for e in direct.events],
                             [(e.event_id, e.text, e.timestamp, e.atomic_group) for e in via_registry.events])
            self.assertEqual([(e.src_id, e.dst_id, e.edge_type) for e in direct.edges],
                             [(e.src_id, e.dst_id, e.edge_type) for e in via_registry.edges])
            # options are filtered per format, never rejected
            self.assertTrue(adapter_for(pi_p, link_values=True, strict_compaction=True).link_values)
            self.assertTrue(roundtrip_check(adapter_for(pi_p), pi_p)["ok"])
            self.assertTrue(roundtrip_check(adapter_for(oh_p), oh_p)["ok"])


class Contract15bOpenHandsSdkTools(unittest.TestCase):
    """Pilot round 2 (RESULTS_pilot defect 11): the OpenHands SDK names its editor `file_editor`. The quote /
    write-path policy must treat it exactly like `str_replace_editor`, and with value links on, a one-shot
    record's id must link the command that consumes it back to the record. Round 1 ran with neither."""

    HEADER = {"instance_id": "demo__app-1", "repo": "demo/app"}

    def _rows(self, tool_name):
        msgs = json.loads(json.dumps(synthetic_trajectory()))
        for m in msgs:
            for tc in m.get("tool_calls") or []:
                if tc["function"]["name"] == "str_replace_editor":
                    tc["function"]["name"] = tool_name
        return messages_to_rows(msgs, self.HEADER)

    @staticmethod
    def _typed_edges(g):
        ref = _ref(g)
        return sorted((ref[e.src_id], ref[e.dst_id], e.edge_type, e.predicate)
                      for e in g.edges if e.edge_type in ("DEPENDS_ON", "SUPERSEDES"))

    def test_15i_file_editor_quote_and_write_path_edges_match_str_replace_editor(self):
        legacy = self._typed_edges(OpenHandsAdapter().normalize(self._rows("str_replace_editor")))
        sdk = self._typed_edges(OpenHandsAdapter().normalize(self._rows("file_editor")))
        self.assertEqual(legacy, sdk)
        self.assertIn(("m0006", "m0005", "DEPENDS_ON", "literal_quote>=40"), sdk)
        self.assertTrue(any(t == "SUPERSEDES" for _, _, t, _ in sdk))
        # an editor name the adapter does not know yields no typed edge at all: the round-1 failure mode
        unknown = self._typed_edges(OpenHandsAdapter().normalize(self._rows("some_future_editor")))
        self.assertEqual([], unknown)

    def test_15k_a_gzip_compressed_export_reads_identically(self):
        """Round-2 lanes keep exports as .jsonl.gz so an agent grepping the filesystem cannot read them as text;
        the registry and the adapter must see the same graph through the compressed file."""
        import gzip
        rows = self._rows("file_editor")
        with tempfile.TemporaryDirectory() as td:
            plain = os.path.join(td, "oh_forgotten.jsonl")
            packed = plain + ".gz"
            head = json.dumps({"tracepack_format": "openhands", "instance_id": "x", "repo": "demo/app"})
            body = "\n".join([head] + [json.dumps(r) for r in rows]) + "\n"
            open(plain, "w", encoding="utf-8").write(body)
            with gzip.open(packed, "wt", encoding="utf-8") as fh:
                fh.write(body)
            self.assertEqual(format_of(plain), format_of(packed))
            g1, g2 = adapter_for(plain).normalize(plain), adapter_for(packed).normalize(packed)
            self.assertEqual([(e.kind, e.text) for e in g1.events], [(e.kind, e.text) for e in g2.events])
            self.assertEqual([(e.src_id, e.dst_id, e.edge_type) for e in g1.edges], [(e.src_id, e.dst_id, e.edge_type) for e in g2.edges])

    def test_15j_value_edge_links_a_one_shot_id_to_the_command_that_consumes_it(self):
        tc = lambda i, name, args: {"id": "chatcmpl-tool-%d" % i, "type": "function",
                                    "function": {"name": name, "arguments": json.dumps(args)}}
        msgs = [
            {"role": "system", "content": "You are OpenHands agent."},
            {"role": "user", "content": "Run the one-shot script once and register the id it prints."},
            {"role": "assistant", "content": "", "tool_calls": [tc(1, "execute_bash", {"command": "bash scripts/os_d03.sh"})]},
            {"role": "tool", "content": "schema change 0042 evaluated\nid: 7a3e90f1\noutcome: skipped, column sr already NOT NULL\n"
                                        "applied-at marker: 2026-09-01T03:12Z\nthe evaluation log is not retained\n"},
            {"role": "assistant", "content": "", "tool_calls": [tc(2, "execute_bash", {"command": "python scripts/ops.py migration-status"})]},
            {"role": "tool", "content": "migration registry: no records\n"},
            {"role": "assistant", "content": "Registering it.", "tool_calls": [tc(3, "execute_bash", {"command": "python scripts/ops.py migration-register --id 7a3e90f1"})]},
            {"role": "tool", "content": "registered 7a3e90f1\n"},
            {"role": "assistant", "content": "Done."},
        ]
        rows = messages_to_rows(msgs, self.HEADER)
        off = OpenHandsAdapter().normalize(rows)                       # round-1 service setting: no value links
        self.assertEqual([], [e for e in off.edges if e.edge_type == "DEPENDS_ON"])
        on = OpenHandsAdapter(link_values=True).normalize(rows)
        ref = _ref(on)
        dep = [(ref[e.src_id], ref[e.dst_id], e.predicate, e.provenance) for e in on.edges if e.edge_type == "DEPENDS_ON"]
        self.assertEqual(1, len(dep), dep)
        src, dst, pred, prov = dep[0]
        self.assertEqual(("m0006", "m0003", "inferred"), (src, dst, prov))   # register call -> one-shot record
        self.assertTrue(pred.startswith("value:"), pred)
        # the evidence behind the edge travels with it (review 09-08: "how did this edge come about?")
        meta = next(e.meta for e in on.edges if e.edge_type == "DEPENDS_ON")
        self.assertEqual({"match": "value", "value": "7a3e90f1", "n_sources": "1"}, {k: meta[k] for k in ("match", "value", "n_sources")})
        self.assertEqual(pred, "value:" + meta["type"])
        q = next(e.meta for e in OpenHandsAdapter().normalize(self._rows("file_editor")).edges if e.edge_type == "DEPENDS_ON")
        self.assertEqual("literal_quote", q["match"])
        self.assertTrue(int(q["run"]) >= 40 and q["quote"].startswith("SAFETY_MARGIN = 0.15"), q)
        # round-2 premise: the consuming call is what BM25 finds for a phase-2 query, the record is not;
        # the adapter only guarantees the edge, which is what lets a 2-hop pack reach the record
        self.assertTrue(roundtrip_check(OpenHandsAdapter(link_values=True), rows)["ok"])


if __name__ == "__main__":
    unittest.main(verbosity=2)
