"""Contract tests for the LlamaIndex integration's framework-free half.

Split on purpose: `TracePackRetriever` carries all the behaviour and has no third-party
dependency, so it is tested here and runs everywhere.  The class that actually subclasses
LlamaIndex's ``BaseMemoryBlock`` is exercised by
``tracepack/examples/llama_index_integration.py`` against the real framework (needs Python >=3.10,
which llama-index-core requires and this repo's default interpreter is not).

The properties below are the ones the upstream PR would claim, so they are asserted, not assumed.
"""
from __future__ import annotations

import os
import sys
import unittest

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

from tracepack.adapters.llama_index_memory import (AssemblerPinDropped, TracePackRetriever,
                                                   _query_from_messages)
from tracepack.tests.fixtures.tiny_trace import tool_chain_graph


class _Msg:
    def __init__(self, role, content):
        self.role, self.content = role, content


class LlamaIndexRetriever(unittest.TestCase):

    def setUp(self):
        self.graph = tool_chain_graph()
        self.r = TracePackRetriever(graph=self.graph, budget=512, k=4)
        self.q = "why was supplier B rejected"

    # ---- construction is validated, not trusted -------------------------------------------
    def test_needs_a_graph(self):
        with self.assertRaises(ValueError):
            TracePackRetriever(graph=None)

    def test_budget_must_be_a_positive_int(self):
        for bad in (0, -1, True, "512"):
            with self.assertRaises(ValueError):
                TracePackRetriever(graph=self.graph, budget=bad)

    def test_unknown_default_query_mode_is_rejected(self):
        with self.assertRaises(ValueError):
            TracePackRetriever(graph=self.graph, default_query_mode="guess")

    # ---- the hard budget is the whole point ------------------------------------------------
    def test_budget_is_never_exceeded(self):
        for b in (64, 128, 256, 512, 1024):
            self.assertLessEqual(self.r.retrieve(self.q, budget=b).tokens, b)

    def test_zero_budget_raises_instead_of_falling_back(self):
        # `budget or self.budget` would silently serve the default here -- caught in review
        with self.assertRaises(ValueError):
            self.r.retrieve(self.q, budget=0)

    def test_negative_budget_and_bad_k_raise(self):
        with self.assertRaises(ValueError):
            self.r.retrieve(self.q, budget=-1)
        for bad in (0, -3, True):
            with self.assertRaises(ValueError):
                self.r.retrieve(self.q, k=bad)

    def test_empty_query_raises(self):
        for bad in ("", "   ", None):
            with self.assertRaises(ValueError):
                self.r.retrieve(bad)

    def test_unknown_query_mode_raises(self):
        with self.assertRaises(ValueError):
            self.r.retrieve(self.q, query_mode="vibes")

    # ---- determinism (contract #1 seen from the adapter side) ------------------------------
    def test_same_inputs_same_packet(self):
        a, b = self.r.retrieve(self.q), self.r.retrieve(self.q)
        self.assertEqual(a.digest, b.digest)
        self.assertEqual(a.text, b.text)

    # ---- runtime conditioning: the claim the upstream PR makes -----------------------------
    def test_query_mode_changes_the_packet(self):
        seen = {m: self.r.retrieve(self.q, query_mode=m).digest
                for m in ("lookup", "why", "state", "audit")}
        self.assertGreater(len(set(seen.values())), 1,
                           "all four query modes produced the same packet: the axis is dead")

    # ---- pinning: contract #5, and it must fail loudly -------------------------------------
    def test_a_pinned_event_is_served(self):
        eid = self.graph.events[0].event_id
        res = self.r.retrieve(self.q, budget=1024, pin=[eid])
        self.assertIn(eid, res.served)

    def test_a_pin_that_does_not_fit_raises_rather_than_vanishing(self):
        eid = max(self.graph.events, key=lambda e: e.cost_of("raw_text")).event_id
        cost = self.graph.event(eid).cost_of("raw_text")
        if cost <= 8:
            self.skipTest("fixture has no event large enough to overflow a tiny budget")
        with self.assertRaises(AssemblerPinDropped):
            self.r.retrieve(self.q, budget=max(1, cost // 4), pin=[eid])

    def test_an_unknown_pin_id_is_ignored_not_fabricated(self):
        res = self.r.retrieve(self.q, pin=["no-such-event"])
        self.assertNotIn("no-such-event", res.served)


class LlamaIndexMessageParsing(unittest.TestCase):

    def test_takes_the_last_user_message(self):
        msgs = [_Msg("user", "first"), _Msg("assistant", "reply"), _Msg("user", "second")]
        self.assertEqual(_query_from_messages(msgs), "second")

    def test_no_user_message_yields_empty_not_an_assistant_turn(self):
        self.assertEqual(_query_from_messages([_Msg("assistant", "hello")]), "")

    def test_empty_and_none(self):
        self.assertEqual(_query_from_messages([]), "")
        self.assertEqual(_query_from_messages(None), "")

    def test_falls_back_to_content_blocks(self):
        class B:
            text = "from a block"

        class M2:
            role = "user"
            content = None
            blocks = [B()]
        self.assertEqual(_query_from_messages([M2()]), "from a block")


if __name__ == "__main__":
    unittest.main(verbosity=2)
