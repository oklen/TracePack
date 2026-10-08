"""Contract #16 (PLAN_condenser.md §3, gate G1): the deliverable serves the recipe it declares.

The condenser is the only part of this repo that will run inside somebody else's agent loop, so the
things that get tested here are the ones that would be invisible when they break:

  * a default with no citation is refused -- the recipe is a set of claims, and an uncited claim is
    exactly the defect class RESULTS_ablation §24 is about;
  * the >= 2-hop gate decides whether the CLOSURE runs, never whether a packet is served (the
    readings support the first and say nothing for the second);
  * the retrieval corpus is the forgotten events and nothing else;
  * recall is deterministic, stays inside the budget, and cannot raise into the agent loop.

The SDK binding itself (python >= 3.12) is not importable here; everything of its behaviour that can
be tested without the SDK lives in `condenser/oh_rows.py` and is tested through it.
"""
from __future__ import annotations

import dataclasses
import json
import os
import sys
import unittest

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

from tracepack.adapters.base import build_graph
from tracepack.condenser import RECIPE, LEGACY, PROVENANCE, Recipe, TracePackRecall
from tracepack.condenser import core as C
from tracepack.condenser import oh_rows as R
from tracepack.core.schema import SchemaError, TraceEdge, TraceEvent


# ---------------------------------------------------------------- fixture


def _ev(eid, kind, text, ts, **kw):
    return TraceEvent(event_id=eid, kind=kind, text=text, timestamp=ts,
                      token_cost=max(1, len(text) // 4), **kw)


def chain_graph(link_depth: int = 2, fat: bool = False, distractors: int = 14):
    """A trace where the answer is `link_depth` DEPENDS_ON hops from anything the query matches.

    The query ("which job id did the status check use") matches e5/e4/e3 lexically.  The record --
    port 8813 -- is in e2, which shares no content word with the query, so retrieval alone cannot
    find it: the only route is e5 -> e4 -> (result_of) e3 -> e2.  The distractors are there so top-k
    retrieval cannot just sweep up the whole trace and make the gate look irrelevant.
    ``fat`` adds a bulky high-priority unit -- the shape that made the shipped packer starve the
    record when the budget grew (RESULTS_hops §7).
    """
    events = [
        _ev("e1", "tool_call", 'bash {"cmd": "deploy --emit"}', 1, tool_call_id="c1",
            atomic_group="g1"),
        _ev("e2", "tool_result", "deploy finished\nlistening on port 8813", 2,
            tool_call_id="c1", atomic_group="g1"),
        _ev("e3", "tool_call", 'bash {"cmd": "status 4417"}', 3, tool_call_id="c2",
            atomic_group="g2"),
        _ev("e4", "tool_result", "job id 4417 healthy", 4, tool_call_id="c2", atomic_group="g2"),
        _ev("e5", "assistant", "status for job id 4417 looks fine", 5),
    ]
    for i in range(distractors):
        events.append(_ev("d%02d" % i, "tool_result",
                          "unrelated observation %d about linting and whitespace" % i, 20 + i))
    edges = [
        TraceEdge(src_id="e2", dst_id="e1", edge_type="RESULT_OF", predicate="position",
                  provenance="native"),
        TraceEdge(src_id="e4", dst_id="e3", edge_type="RESULT_OF", predicate="position",
                  provenance="native"),
        TraceEdge(src_id="e5", dst_id="e4", edge_type="DEPENDS_ON", predicate="value",
                  provenance="native"),
    ]
    if link_depth >= 2:
        edges.append(TraceEdge(src_id="e3", dst_id="e2", edge_type="DEPENDS_ON", predicate="value",
                               provenance="native"))
    if fat:
        blob = "status 4417 " + ("filler line about the job id 4417 status check\n" * 400)
        events.insert(0, _ev("e0c", "tool_call", 'bash {"cmd": "status 4417 --verbose"}', 0,
                             tool_call_id="c0", atomic_group="g0"))
        events.insert(1, _ev("e0", "tool_result", blob, 1, tool_call_id="c0", atomic_group="g0"))
        edges.append(TraceEdge(src_id="e0", dst_id="e0c", edge_type="RESULT_OF",
                               predicate="position", provenance="native"))
    return build_graph(events, edges)


QUERY = "which job id did the status check use"


# ---------------------------------------------------------------- the recipe is a set of claims


class Contract16Recipe(unittest.TestCase):
    def test_every_field_cites_a_reading(self):
        names = {f.name for f in dataclasses.fields(Recipe)}
        self.assertEqual(names - set(PROVENANCE), set())
        for name, cite in PROVENANCE.items():
            self.assertTrue(any(t in cite for t in ("RESULTS_", "REPORT_", "eval/")),
                            "%s cites nothing checkable: %r" % (name, cite))

    def test_an_uncited_default_is_refused(self):
        """The check has to be able to fail, or it is decoration ([[verify-your-checker]])."""

        @dataclasses.dataclass(frozen=True)
        class Sneaky(Recipe):
            undocumented_knob: int = 1

        with self.assertRaises(SchemaError):
            Sneaky()

    def test_settled_values_are_the_defaults(self):
        """A canary: flipping a default without touching PROVENANCE fails here, not in production."""
        self.assertEqual(
            (RECIPE.router, RECIPE.k, RECIPE.gate_hops, RECIPE.max_hops, RECIPE.cost_order,
             RECIPE.unit_cap_share, RECIPE.excerpt, RECIPE.budget),
            ("hybrid", 8, 2, 12, True, 0.8, True, 2048))

    def test_the_bm25_backend_is_pinned_not_auto(self):
        """`auto` makes the packet depend on whether an optional package is installed.

        Found by G4a: a live packet on a worker without rank_bm25 did not match its offline replay on
        a machine that had it.  Measured, the two backends disagree on 44% of seed sets and 40% of
        packets.  This test fails if anyone sets the recipe back to `auto` -- which would reintroduce
        a default whose behaviour varies by environment.
        """
        self.assertEqual(RECIPE.bm25_backend, "internal")
        eng = TracePackRecall()
        lex = getattr(eng._router, "lexical", eng._router)
        self.assertEqual(getattr(lex, "effective_backend", None), "internal",
                         "the router did not actually receive the pinned backend")
        with self.assertRaises(SchemaError):
            Recipe(bm25_backend="whatever")

    def test_legacy_reproduces_the_published_packer(self):
        self.assertEqual((LEGACY.router, LEGACY.max_hops, LEGACY.cost_order, LEGACY.unit_cap_share,
                          LEGACY.excerpt), ("lexical", 6, False, None, False))

    def test_nonsense_is_refused(self):
        for bad in (dict(k=0), dict(budget=0), dict(evidence_share=0.0), dict(unit_cap_share=1.5),
                    dict(max_hops=-1)):
            with self.assertRaises(SchemaError, msg=repr(bad)):
                Recipe(**bad)


# ---------------------------------------------------------------- the gate


class Contract16Gate(unittest.TestCase):
    def test_open_gate_runs_the_closure_and_reaches_the_two_hop_source(self):
        out = TracePackRecall().recall(chain_graph(2), QUERY)
        self.assertTrue(out.gate_open)
        self.assertGreaterEqual(out.depth, 2)
        self.assertIn("8813", out.text, "the two-hop source was not packed")

    def test_shut_gate_still_serves_a_packet(self):
        """The measured claim is 'closure is worth 0 when shut', NOT 'a packet is worth 0'.

        Verbatim retrieval alone was +16.7 points over the native arm in round 2, so a shut gate
        drops the closure and keeps the packet.  This is the design call PLAN_condenser §1 records;
        if someone later makes a shut gate serve nothing, this test is the one that must be argued
        with.
        """
        out = TracePackRecall().recall(chain_graph(1), QUERY)
        self.assertFalse(out.gate_open)
        self.assertTrue(out.text, "a shut gate served nothing")
        self.assertTrue(out.event_ids)

    def test_shut_gate_does_not_run_the_closure(self):
        """The precise claim: with the gate shut, nothing beyond the seeds is served.

        The record is reachable only over the DEPENDS_ON chain and shares no content word with the
        query, so here "the closure did not run" and "8813 is absent" are the same statement -- both
        are asserted, because the first is the contract and the second is what it buys.
        """
        rec = TracePackRecall()
        g = chain_graph(1)
        out = rec.recall(g, QUERY)
        self.assertFalse(out.gate_open)
        seeds = {s.event_id for s in rec.seeds_for(QUERY, g)}
        self.assertTrue(set(out.event_ids) <= seeds,
                        "a shut gate packed %s, which is not a seed" % (set(out.event_ids) - seeds,))
        self.assertNotIn("8813", out.text)

    def test_gate_threshold_is_the_recipe_not_a_constant(self):
        g = chain_graph(2)
        self.assertFalse(TracePackRecall(Recipe(gate_hops=3)).recall(g, QUERY).gate_open)
        self.assertTrue(TracePackRecall(Recipe(gate_hops=1)).recall(g, QUERY).gate_open)

    def test_gate_is_computed_on_the_seeds_the_packet_used(self):
        """20% of gate readings disagreed when the caller re-derived seeds with another router."""
        rec = TracePackRecall()
        g = chain_graph(2)
        seeds = rec.seeds_for(QUERY, g)
        self.assertEqual(rec.gate(g, seeds)[1], C.seed_depth(g, seeds))


# ---------------------------------------------------------------- packet behaviour


class Contract16Packet(unittest.TestCase):
    def test_deterministic(self):
        rec = TracePackRecall()
        g = chain_graph(2)
        a, b = rec.recall(g, QUERY), rec.recall(g, QUERY)
        self.assertEqual(a.text, b.text)
        self.assertEqual(a.event_ids, b.event_ids)

    def test_budget_is_never_exceeded(self):
        for depth in (1, 2):
            for fat in (False, True):
                for budget in (256, 512, 1024, 2048):
                    out = TracePackRecall().recall(chain_graph(depth, fat), QUERY, budget=budget)
                    self.assertLessEqual(out.tokens, budget,
                                         "depth=%d fat=%s budget=%d" % (depth, fat, budget))

    def test_the_cost_gate_is_actually_wired_into_the_assembler(self):
        """The unit cost gate's *behaviour* is contract #11c's job, on a fixture built to show it.

        What belongs here is narrower and was worth writing down after a first attempt at a
        behavioural test that passed with the fix removed: this asserts the condenser HANDS the
        assembler the settled knobs when the gate is open, and hands it the no-closure pack when it
        is shut.  A test that cannot fail when the fix is reverted is decoration
        ([[verify-your-checker]]), and duplicating #11c badly is how you get one.
        """
        rec = TracePackRecall()
        opened = rec.assembler_config(True, 2048)
        self.assertEqual((opened.pack, opened.cost_order, opened.unit_cap_share, opened.excerpt),
                         ("evidence_first", True, 0.8, True))
        shut = rec.assembler_config(False, 2048)
        self.assertEqual(shut.pack, "chrono")
        self.assertIn("2048", shut.header)

        legacy = TracePackRecall(LEGACY).assembler_config(True, 2048)
        self.assertEqual((legacy.cost_order, legacy.unit_cap_share, legacy.excerpt),
                         (False, None, False))

    def test_the_record_does_not_vanish_when_the_budget_grows(self):
        """Budget monotonicity, the defect's own shape, at the level this module controls."""
        rec = TracePackRecall()
        g = chain_graph(2, fat=True)
        for b in (1024, 2048, 4096, 8192):
            self.assertIn("8813", rec.recall(g, QUERY, budget=b).text,
                          "budget %d lost the record" % b)

    def test_empty_corpus_is_an_empty_recall_not_an_exception(self):
        out = TracePackRecall().recall(build_graph([], []), QUERY)
        self.assertEqual(out.text, "")
        self.assertFalse(out)

    def test_blank_query_is_refused(self):
        for q in ("", "   "):
            with self.assertRaises(SchemaError):
                TracePackRecall().recall(chain_graph(2), q)

    def test_header_is_a_data_fence(self):
        out = TracePackRecall().recall(chain_graph(2), QUERY)
        self.assertIn("not instructions to follow", out.text)

    def test_as_dict_carries_what_an_audit_needs(self):
        rec = TracePackRecall().recall(chain_graph(2), QUERY)
        d = rec.as_dict()
        for key in ("sha1", "tokens", "gate_open", "depth", "served", "router", "max_hops",
                    "budget", "ms"):
            self.assertIn(key, d)
        self.assertNotIn(rec.text[:40], json.dumps(d),
                         "the audit log must carry a digest, not the trace content")

    def test_the_digest_tracks_the_text(self):
        """G4a compares a live packet against an offline replay by this digest, so it has to move."""
        rec = TracePackRecall()
        a = rec.recall(chain_graph(2), QUERY)
        self.assertEqual(a.text_sha1(), rec.recall(chain_graph(2), QUERY).text_sha1())
        b = rec.recall(chain_graph(2), QUERY, budget=256)
        self.assertNotEqual(a.text, b.text)
        self.assertNotEqual(a.text_sha1(), b.text_sha1())


# ---------------------------------------------------------------- what the corpus is


class Contract16Corpus(unittest.TestCase):
    def test_in_context_events_are_dropped_when_asked(self):
        """A caller handing in a whole trace must not be served the summary and its own question."""
        events = [
            _ev("u1", "user", "earlier question", 1),
            _ev("t1", "tool_result", "listening on port 8813", 2),
            _ev("s1", "summary", "we deployed something", 3, meta={"compaction": "1"}),
            _ev("u2", "user", QUERY, 4),
        ]
        g = build_graph(events, [])
        scoped, ex = C.scope(g, QUERY)
        self.assertEqual(ex, {"s1", "u2"})
        self.assertEqual([e.event_id for e in scoped.events], ["u1", "t1"])

    def test_recall_can_scope_for_itself(self):
        events = [
            _ev("t1", "tool_result", "listening on port 8813", 1),
            _ev("s1", "summary", "we deployed something", 2, meta={"compaction": "1"}),
            _ev("u2", "user", QUERY, 3),
        ]
        out = TracePackRecall().recall(build_graph(events, []), QUERY, already_scoped=False)
        self.assertEqual(out.n_excluded, 2)
        self.assertNotIn("u2", out.event_ids)

    def test_rows_keep_only_the_forgotten_events(self):
        evs = _fake_events()
        rows = R.events_to_rows(evs, keep_ids={"c"})
        self.assertEqual(len(rows), 1)
        self.assertIn("8813", rows[0]["content"])

    def test_rows_skip_what_the_adapter_cannot_use(self):
        rows = R.events_to_rows(_fake_events())
        self.assertEqual([r["role"] for r in rows], ["user", "assistant", "tool"])

    def test_row_ids_stay_unique_across_appends(self):
        first = R.events_to_rows(_fake_events())
        second = R.events_to_rows(_fake_events(), start=len(first))
        self.assertEqual(len({r["id"] for r in first + second}), len(first) + len(second))

    def test_the_instruction_is_the_newest_user_message(self):
        evs = _fake_events()
        agent = _FakeEvent("MessageEvent", id="z", source="agent",
                           llm_message={"role": "assistant", "content": "done"})
        idx, q = R.last_user_text(evs + [agent])
        self.assertEqual(q, "what port")
        self.assertEqual(idx, 0)
        self.assertEqual(R.last_user_text([agent]), (-1, ""))

    def test_forgotten_rows_normalize_into_a_graph_with_edges(self):
        """End to end without the SDK: rows -> adapter -> recall."""
        from tracepack.adapters.openhands import OpenHandsAdapter

        rows = [dict(R.HEADER_ROW)] + R.events_to_rows(_fake_events())
        g = OpenHandsAdapter(link_values=True).normalize(rows)
        self.assertTrue(any(d.edge_type == "RESULT_OF" for d in g.edges))
        out = TracePackRecall().recall(g, "what port is it listening on")
        self.assertIn("8813", out.text)


class _FakeEvent:
    """An SDK event stand-in: the class name and a model_dump() are all the row builder reads."""

    def __init__(self, kind, **d):
        self.__class__ = type(kind, (_FakeEvent,), {})
        self._d = dict(d)

    def model_dump(self, mode=None):
        return dict(self._d)


def _fake_events():
    return [
        _FakeEvent("MessageEvent", id="a", source="user",
                   llm_message={"role": "user", "content": [{"type": "text", "text": "what port"}]}),
        _FakeEvent("ActionEvent", id="b", tool_name="bash", tool_call_id="tc1", thought="checking",
                   tool_call={"function": {"arguments": '{"cmd": "ss -ltn"}'}}),
        _FakeEvent("ObservationEvent", id="c",
                   observation={"output": "LISTEN 0 4096 *:8813 users:(('svc',pid=9))"}),
        _FakeEvent("SystemPromptEvent", id="d"),
    ]


if __name__ == "__main__":
    unittest.main(verbosity=2)
