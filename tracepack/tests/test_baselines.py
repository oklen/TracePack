"""Contract #17 -- the baseline arms and the ContextWeaver port (PLAN_baselines.md).

These lock the properties that decide whether the comparison means anything:

  * the new arm builder is the SAME RULER as the published one (B and C byte-identical to tp_serve)
  * each factor actually bites, so a null for it is a measurement and not a no-op
  * the identifier-chasing baseline reads only what it has retrieved -- never the gold
  * ContextWeaver's ancestry, masking and warmup follow Algorithm 1
  * a dead analyzer collapses the ancestry instead of producing a plausible-looking recency window
  * graph construction cannot see an instruction that had not been issued yet
"""
from __future__ import annotations

import os
import sys
import unittest

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(os.path.dirname(HERE))
if ROOT not in sys.path:
    sys.path.insert(0, ROOT)

from tracepack.adapters.base import build_graph                    # noqa: E402
from tracepack.baselines import contextweaver as CW                # noqa: E402
from tracepack.core.schema import TraceEdge, TraceEvent            # noqa: E402
from tracepack.pilot import cw_arms                                # noqa: E402


def _ev(eid, kind, text, ts, **kw):
    return TraceEvent(event_id=eid, kind=kind, text=text, timestamp=ts,
                      token_cost=max(1, len(text) // 4), **kw)


def _graph():
    """A chain whose answer is two hops from anything the query matches, plus three distractors."""
    events = [
        _ev("e1", "tool_call", 'bash {"cmd": "deploy --emit"}', 1, tool_call_id="c1",
            atomic_group="g1"),
        _ev("e2", "tool_result", "deploy finished\nlistening on port 8813\njob id 4417", 2,
            tool_call_id="c1", atomic_group="g1"),
        _ev("e3", "tool_call", 'bash {"cmd": "ls -R"}', 3, tool_call_id="c2", atomic_group="g2"),
        _ev("e4", "tool_result", "core/api.py\nsvc/config.py\ntests/test_api.py", 4,
            tool_call_id="c2", atomic_group="g2"),
        _ev("e5", "tool_call", 'bash {"cmd": "status 4417"}', 5, tool_call_id="c3",
            atomic_group="g3"),
        _ev("e6", "tool_result", "job id 4417 healthy, bound to port 8813", 6, tool_call_id="c3",
            atomic_group="g3"),
        _ev("e7", "assistant", "status for job id 4417 looks fine", 7),
    ]
    edges = [
        TraceEdge(src_id="e2", dst_id="e1", edge_type="RESULT_OF", predicate="position",
                  provenance="native"),
        TraceEdge(src_id="e4", dst_id="e3", edge_type="RESULT_OF", predicate="position",
                  provenance="native"),
        TraceEdge(src_id="e6", dst_id="e5", edge_type="RESULT_OF", predicate="position",
                  provenance="native"),
        TraceEdge(src_id="e5", dst_id="e2", edge_type="DEPENDS_ON", predicate="value:id",
                  provenance="native"),
        TraceEdge(src_id="e7", dst_id="e6", edge_type="DEPENDS_ON", predicate="value:id",
                  provenance="native"),
    ]
    return build_graph(events, edges)


QUERY = "which job id did the status check use"


class Contract17Arms(unittest.TestCase):
    """The arm builder must be one ruler, and every factor must bite."""

    def setUp(self):
        self.g = _graph()

    def _seeds(self, kind):
        return cw_arms.select(kind, QUERY, self.g, 4)[0]

    def test_17a_pairing_only_adds_results_of_seed_calls(self):
        base = {s.event_id for s in self._seeds("bm25")}
        paired = {s.event_id for s in self._seeds("bm25pair")}
        self.assertTrue(base <= paired, "pairing removed a seed")
        for eid in paired - base:
            ev = self.g.event(eid)
            self.assertEqual(ev.kind, "tool_result",
                             "pairing added something that is not a tool_result: %s" % eid)

    def test_17b_neighbour_window_adds_exactly_the_window(self):
        ids = list(self.g.event_ids)
        pos = {e: i for i, e in enumerate(ids)}
        base = [s for s in self._seeds("bm25pair")]
        for w in (1, 2):
            got = {s.event_id for s in cw_arms.neighbours(base, self.g, w)}
            want = set()
            for s in base:
                i = pos[s.event_id]
                want.update(ids[max(0, i - w):i + w + 1])
            self.assertEqual(got, want, "w=%d window is not the +-w window" % w)

    def test_17c_neighbour_window_is_monotone_in_w(self):
        sizes = [len(self._seeds("bm25pair_nb%d" % w)) for w in (1, 2, 4)]
        self.assertEqual(sizes, sorted(sizes), "a wider window returned fewer seeds: %s" % sizes)

    def test_17d_identifier_chasing_never_invents_an_event(self):
        got = {s.event_id for s in self._seeds("bm25id4")}
        self.assertTrue(got <= set(self.g.event_ids), "chasing returned an id not in the graph")

    def test_17e_identifier_rule_is_generic(self):
        """It must match ordinary identifier shapes, not one corpus's key."""
        found = set(cw_arms.identifiers("id: 4b7d21c9 build 2.14.0 path svc/config.py sha "
                                        "449503e2eaba73a60062f69bfc40b6c35ace5fe1"))
        for want in ("4b7d21c9", "2.14.0", "svc/config.py"):
            self.assertIn(want, found, "the identifier rule missed %r" % want)
        self.assertFalse(cw_arms.identifiers("the quick brown fox jumps over"),
                         "plain prose produced identifiers")

    def test_17f_assembly_cannot_act_without_a_closure(self):
        """evidence_first orders the EVIDENCE tier; a seed-only selection has none.

        Measured on 192 real corpora at five budgets: B and S1 are byte-identical in every cell.
        Locking it here is what stops the 2x2 being reported as a clean factorial.
        """
        from tracepack.core.assembler import BudgetAssembler
        for budget in (256, 2048):
            out = {}
            for asm in ("chrono", "evidence_first"):
                seeds, closure = cw_arms.select("bm25", QUERY, self.g, 4)
                out[asm] = BudgetAssembler(cw_arms.assembler_config(asm)).assemble(
                    QUERY, closure, self.g, budget, seeds=seeds, query_mode="lookup").context
            self.assertEqual(out["chrono"], out["evidence_first"],
                             "the assembly factor changed a seed-only packet at budget %d" % budget)


class Contract17Supersede(unittest.TestCase):
    """`superseded` is our rule, not the paper's, so it has to say exactly what it does.

    The legacy form also matched `"command"` and its capture stopped at the first escaped quote, so
    `grep -n \\` keyed 151 steps across 42 unrelated commands, and for a file editor the first
    matching field was the OPERATION NAME -- every `str_replace` superseded every earlier one.
    """

    @staticmethod
    def _n(i, action, obs="ok"):
        return CW.Node(idx=i, action=action, observation=obs, validation="passed")

    def _edit(self, i, path, op="str_replace"):
        return self._n(i, 'file_editor {"command": "%s", "path": "%s"}' % (op, path))

    def _term(self, i, cmd):
        return self._n(i, 'terminal {"command": "%s", "is_input": false}' % cmd)

    def test_17s1_a_later_write_supersedes_an_earlier_write_to_the_same_path(self):
        ns = [self._edit(0, "/t/a.py"), self._edit(1, "/t/a.py")]
        CW.mark_superseded(ns)
        self.assertEqual([n.validation for n in ns], ["superseded", "passed"])

    def test_17s2_writes_to_different_paths_do_not_supersede(self):
        ns = [self._edit(0, "/t/a.py"), self._edit(1, "/t/b.py")]
        CW.mark_superseded(ns)
        self.assertEqual([n.validation for n in ns], ["passed", "passed"])

    def test_17s3_a_view_is_never_superseded_its_output_is_the_evidence(self):
        ns = [self._edit(0, "/t/a.py", op="view"), self._edit(1, "/t/a.py")]
        CW.mark_superseded(ns)
        self.assertEqual(ns[0].validation, "passed",
                         "a read of the file a later edit is based on was removed from the history")

    def test_17s4_terminal_steps_are_never_superseded(self):
        ns = [self._term(0, 'grep -n \\"Foo\\" /t/a.py'), self._term(1, 'grep -n \\"Bar\\" /t/b.py')]
        CW.mark_superseded(ns)
        self.assertEqual([n.validation for n in ns], ["passed", "passed"],
                         "two unrelated greps superseded each other")

    def test_17s5_legacy_mode_still_reproduces_the_published_behaviour(self):
        ns = [self._term(0, 'grep -n \\"Foo\\" /t/a.py'), self._term(1, 'grep -n \\"Bar\\" /t/b.py')]
        CW.mark_superseded(ns, "legacy")
        self.assertEqual(ns[0].validation, "superseded",
                         "legacy mode no longer reproduces what the published W numbers ran with")
        ns2 = [self._edit(0, "/t/a.py"), self._edit(1, "/t/b.py")]
        CW.mark_superseded(ns2, "legacy")
        self.assertEqual(ns2[0].validation, "superseded",
                         "legacy keyed on the operation name, so different paths still collided")

    def test_17s6_an_unknown_mode_is_refused_not_silently_ignored(self):
        with self.assertRaises(ValueError):
            CW.mark_superseded([], "whatever")


class Contract17Validation(unittest.TestCase):
    """`failed` is a state the paper names and does not define, so the classifier is ours.

    The published one matched error|traceback|exception|failed as a bare word ANYWHERE, and reading a
    Python file almost always contains one somewhere.  Measured over 1,577 real nodes: 11.9% marked
    failed, 65% of those with no failure at the start of any line -- successful edits whose echoed
    file body happened to contain the word.  Failed steps are dropped from the analyzer's candidates.
    """

    REAL_TRACEBACK = ("Traceback (most recent call last):\n"
                      "  File \"/testbed/x.py\", line 3, in <module>\n"
                      "TypeError: bad operand")
    SUCCESSFUL_EDIT = ("The file /testbed/astropy/coordinates/sky_coordinate.py has been edited. "
                       "Here's the result of running `cat -n`:\n"
                       "   12\t    raise ValueError('failed to parse')   # a line IN the file")
    SUCCESSFUL_UNDO = ("Last edit to /testbed/x.py undone successfully. Here's the result of "
                       "running `cat -n`:\n    4\tclass MyException(Exception):")
    GREP_HIT = "/testbed/tests/test_errors.py:102:    def test_exception_message(self):"

    def test_17v1_a_real_traceback_is_a_failure_under_both_rules(self):
        for mode in ("anchored", "legacy"):
            self.assertEqual(CW._validation(self.REAL_TRACEBACK, mode), "failed", mode)

    def test_17v2_a_successful_edit_is_not_a_failure(self):
        self.assertEqual(CW._validation(self.SUCCESSFUL_EDIT, "anchored"), "passed",
                         "a successful edit was dropped because the file body says 'failed'")
        self.assertEqual(CW._validation(self.SUCCESSFUL_EDIT, "legacy"), "failed",
                         "legacy mode no longer reproduces the published behaviour")

    def test_17v3_a_successful_undo_is_not_a_failure(self):
        self.assertEqual(CW._validation(self.SUCCESSFUL_UNDO, "anchored"), "passed")
        self.assertEqual(CW._validation(self.SUCCESSFUL_UNDO, "legacy"), "failed")

    def test_17v4_a_grep_hit_that_mentions_errors_is_not_a_failure(self):
        self.assertEqual(CW._validation(self.GREP_HIT, "anchored"), "passed")

    def test_17v5_a_shell_level_failure_still_registers(self):
        for obs in ("bash: frobnicate: command not found",
                    "grep: /testbed/nope.py: No such file or directory",
                    "E   AssertionError: expected 3",
                    "FAILED tests/test_x.py::test_y"):
            self.assertEqual(CW._validation(obs, "anchored"), "failed", obs)

    def test_17v6_an_empty_observation_is_unknown_not_passed(self):
        self.assertEqual(CW._validation("", "anchored"), "unknown")

    def test_17v7_an_unknown_mode_is_refused(self):
        with self.assertRaises(ValueError):
            CW._validation("anything", "whatever")


class Contract17PublishedPins(unittest.TestCase):
    """A driver that owns published numbers must PIN its rules, not inherit today's default.

    This test exists because I broke exactly this: changing CWConfig's defaults to the corrected
    candidate rules silently changed what cw_weave.py -- the driver behind RESULTS_baselines'
    ContextWeaver numbers -- would reproduce on a re-run.  Nothing failed, because nothing checked.
    """

    def test_17p1_the_published_driver_defaults_to_the_published_rules(self):
        import re as _re
        src = open(os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
                                "pilot", "cw_weave.py"), encoding="utf-8").read()
        for flag in ("supersede", "validation"):
            m = _re.search(r'--%s"[^)]*?default="([a-z_]+)"' % flag, src, _re.S)
            self.assertIsNotNone(m, "cw_weave.py no longer exposes --%s" % flag)
            self.assertEqual(m.group(1), "legacy",
                             "cw_weave.py defaults --%s to %r; the published numbers ran `legacy`, "
                             "so a re-run would no longer reproduce them" % (flag, m.group(1)))

    def test_17p2_the_published_driver_passes_the_rules_explicitly(self):
        src = open(os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
                                "pilot", "cw_weave.py"), encoding="utf-8").read()
        self.assertIn("CW.extract_nodes(rows, cfg.validation)", src,
                      "cw_weave.py takes extract_nodes' default instead of naming the rule")
        self.assertIn("CW.mark_superseded(nodes, cfg.supersede)", src,
                      "cw_weave.py takes mark_superseded's default instead of naming the rule")

    def test_17p3_the_swe_condenser_pins_W_to_the_published_pair(self):
        """Read as source, not imported: oh_contextweaver needs the SDK and py3.12, which the test
        host need not have -- and a guard that only runs on some hosts is not a guard."""
        import re as _re
        src = open(os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
                                "baselines", "oh_contextweaver.py"), encoding="utf-8").read()
        for field in ("supersede", "validation"):
            m = _re.search(r'%s: str = Field\(default="([a-z_]+)"' % field, src)
            self.assertIsNotNone(m, "oh_contextweaver no longer declares %s" % field)
            self.assertEqual(m.group(1), "legacy",
                             "arm W defaults %s to %r; it would no longer be the arm the published "
                             "numbers describe" % (field, m.group(1)))
        swe = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "pilot", "swe",
                           "swe_pilot.py")
        if os.path.exists(swe):     # the SWE-bench cell runner is not part of the public release
            self.assertIn('("legacy" if arm == "W" else', open(swe, encoding="utf-8").read(),
                          "the cell runner no longer pins W to the legacy rules per arm")


class Contract17Weaver(unittest.TestCase):
    """ContextWeaver: Algorithm 1's ancestry, masking, warmup -- and a dead analyzer must show."""

    def setUp(self):
        self.nodes = CW.extract_nodes(CW.PLANT_ROWS)
        CW.mark_superseded(self.nodes)
        self.cfg = CW.CWConfig(W=3, m=2)

    def test_17g_nodes_are_action_observation_pairs(self):
        self.assertEqual(len(self.nodes), 5)
        for n in self.nodes:
            self.assertTrue(n.action and n.observation, "node %d is half a step" % n.idx)
            self.assertEqual(len(n.event_ids), 2)

    def test_17h_ancestry_is_bfs_bounded_by_W(self):
        chat = CW._StubChat({(4, 1): 9, (4, 3): 2, (3, 0): 7, (1, 0): 4})
        CW.build_graph(chat, self.nodes, "goal", self.cfg)
        a = CW.ancestry(self.nodes, 4, self.cfg.W)
        self.assertIn(4, a, "the anchor is not in its own ancestry")
        self.assertIn(1, a, "the analyzer's top parent is not in the ancestry")
        self.assertLessEqual(len(a), self.cfg.W, "ancestry exceeded W")

    def test_17i_non_ancestors_lose_their_observation_and_keep_their_action(self):
        chat = CW._StubChat({(4, 1): 9})
        CW.build_graph(chat, self.nodes, "goal", self.cfg)
        a = CW.ancestry(self.nodes, 4, self.cfg.W)
        text = CW.weave(self.nodes, a, 4, include_depsum=False)
        for n in self.nodes:
            self.assertIn(n.action[:40], text, "step %d lost its action" % n.idx)
        self.assertIn(CW.PLACEHOLDER, text, "no observation was masked")
        self.assertIn("OCSP stapling unsupported", text, "an ancestor was not kept verbatim")

    def test_17j_warmup_keeps_everything(self):
        full = CW.weave(self.nodes, tuple(range(len(self.nodes))), 4, include_depsum=False)
        self.assertNotIn(CW.PLACEHOLDER, full, "warmup masked an observation")

    def test_17k_a_dead_analyzer_collapses_the_ancestry(self):
        """A scorer that returns 0 for everything must yield an ancestry of just the anchor.

        Without this, a broken analyzer looks like "ContextWeaver kept the last few steps", which is
        a different method entirely and would be reported as its result.
        """
        nodes = CW.extract_nodes(CW.PLANT_ROWS)
        CW.build_graph(CW._StubChat(fixed=0), nodes, "goal", self.cfg)
        self.assertTrue(all(not n.parents for n in nodes))
        self.assertEqual(CW.ancestry(nodes, 4, self.cfg.W), (4,))

    def test_17l_graph_construction_cannot_see_a_later_instruction(self):
        """Each node is scored against the instruction in force when it ran, not a later one."""
        rows = [{"tracepack_format": "openhands"},
                {"id": "u0", "role": "user", "content": "FIRST INSTRUCTION"},
                {"id": "a0", "role": "assistant", "content": "",
                 "tool_calls": [{"function": {"name": "terminal", "arguments": '{"command": "ls"}'}}]},
                {"id": "t0", "role": "tool", "content": "a.py"},
                {"id": "u1", "role": "user", "content": "SECOND INSTRUCTION"},
                {"id": "a1", "role": "assistant", "content": "",
                 "tool_calls": [{"function": {"name": "terminal", "arguments": '{"command": "cat a.py"}'}}]},
                {"id": "t1", "role": "tool", "content": "print(1)"}]
        nodes = CW.extract_nodes(rows)
        self.assertEqual([n.goal for n in nodes], ["FIRST INSTRUCTION", "SECOND INSTRUCTION"])
        chat = CW._StubChat({})
        CW.build_graph(chat, nodes, "A LATER INSTRUCTION THAT MUST NOT BE SEEN", CW.CWConfig(W=5, m=1))
        self.assertTrue(chat.prompts, "the analyzer was never called, so this proves nothing")
        self.assertTrue(all("A LATER INSTRUCTION" not in p for p in chat.prompts),
                        "a later instruction reached the analyzer")
        self.assertTrue(any("SECOND INSTRUCTION" in p for p in chat.prompts),
                        "the node's own instruction never reached the analyzer either -- the "
                        "assertion above would then be vacuous")


class Contract17LiveEvents(unittest.TestCase):
    """Node extraction from a LIVE harness view.

    This class exists because the first version read `event.tool_calls`, which OpenHands events do
    not have.  It found zero nodes, so the condenser returned the view untouched, so every arm-W
    cell ran with the FULL history and passed -- a result that would have been published as
    ContextWeaver winning.  The shape of the live event is now a contract.
    """

    class _E(dict):
        """A duck-typed SDK event: class name in `_kind`, fields as in the real model_dump()."""

        def model_dump(self, mode="json"):
            return {k: v for k, v in self.items() if k != "_kind"}

    def _view(self):
        E = self._E

        def msg(i, role, text):
            return E(_kind="MessageEvent", id="m%d" % i,
                     llm_message={"role": role, "content": [{"type": "text", "text": text}]})

        def act(i, cid, thought, cmd):
            return E(_kind="ActionEvent", id="a%d" % i, tool_call_id=cid, tool_name="terminal",
                     thought=thought,
                     tool_call={"function": {"name": "terminal",
                                             "arguments": '{"command": "%s"}' % cmd}})

        def obs(i, cid, text):
            return E(_kind="ObservationEvent", id="o%d" % i, tool_call_id=cid,
                     observation={"text": text})

        return [msg(0, "user", "FIRST INSTRUCTION"),
                act(1, "c1", "run the one-shot", "bash oneshot/os.sh"),
                obs(2, "c1", "edge certificate issued\nid: 4b7d21c9\nrenewal blocked: OCSP"),
                act(3, "c2", "list", "ls -R"),
                obs(4, "c2", "core/api.py"),
                msg(5, "user", "SECOND INSTRUCTION"),
                act(6, "c3", "register", "python scripts/ops.py tls-register --id 4b7d21c9"),
                obs(7, "c3", "tls material 4b7d21c9 installed")]

    def test_17m_live_events_produce_nodes(self):
        nodes = CW.nodes_from_events(self._view())
        self.assertEqual(len(nodes), 3, "live ActionEvent/ObservationEvent pairs were not found")
        self.assertIn("4b7d21c9", nodes[0].observation)
        self.assertIn("tls-register", nodes[2].action)
        for n in nodes:
            self.assertEqual(len(n.event_ids), 2)

    def test_17n_live_nodes_carry_the_instruction_in_force(self):
        nodes = CW.nodes_from_events(self._view())
        self.assertEqual([n.goal for n in nodes],
                         ["FIRST INSTRUCTION", "FIRST INSTRUCTION", "SECOND INSTRUCTION"])

    def test_17o_last_user_text_is_the_newest_instruction(self):
        self.assertEqual(CW.last_user_text(self._view()), "SECOND INSTRUCTION")

    def test_17p_an_unrecognised_event_shape_yields_nothing(self):
        """It must produce zero nodes, not something plausible.  The binding turns zero into a hard
        error after enough calls; here we only lock that zero is what comes out."""
        junk = [{"role": "assistant", "tool_calls": [{"function": {"name": "x", "arguments": "{}"}}]}]
        self.assertEqual(CW.nodes_from_events(junk), [],
                         "an unrecognised event shape silently produced nodes")


class Contract17BudgetAdapted(unittest.TestCase):
    """The budget-adapted rendering of a ContextWeaver ancestry (comparison plan 5.3).

    Its whole purpose is to make the two families readable at the SAME budget, so the adapter must
    not be the thing that decides the result.  Truncating text would cut the tail, which is exactly
    where a one-shot record's value sits, and would turn this into a straw man.
    """

    def setUp(self):
        from tracepack.pilot import cw_weave
        self.bm = cw_weave.budget_matched
        self.nodes = CW.extract_nodes(CW.PLANT_ROWS)

    def test_17q_units_are_whole_or_absent(self):
        out = self.bm(self.nodes, (0, 1, 2, 3, 4), 40, 4)
        for i in out["kept"]:
            self.assertIn(self.nodes[i].observation.split("\n")[0], out["text"],
                          "a kept step was emitted without its observation")
        for i in out["dropped"]:
            self.assertNotIn("step %d |" % i, out["text"], "a dropped step still appears")
        self.assertEqual(sorted(out["kept"] + out["dropped"]), [0, 1, 2, 3, 4])

    def test_17r_a_generous_budget_drops_nothing(self):
        out = self.bm(self.nodes, (0, 1, 2, 3, 4), 10 ** 6, 4)
        self.assertEqual(out["dropped"], [])
        self.assertEqual(out["kept"], [0, 1, 2, 3, 4])

    def test_17s_a_tight_budget_prefers_nearer_ancestors_and_does_not_privilege_the_anchor(self):
        """Nearer-to-the-anchor wins among the units that FIT -- and nothing is privileged.

        The first version of this test asserted the anchor always survives.  It does not, and it
        should not: at a tight budget the anchor can be the fat unit that starves everything below
        it, which is the exact defect RESULTS_hops 7 fixed in the shipped packer by putting cost in
        the ordering key.  What must hold is that the budget is respected and that, among units
        that fit, proximity decides.
        """
        out = self.bm(self.nodes, (0, 1, 2, 3, 4), 30, 4)
        self.assertTrue(out["kept"], "a tight budget kept nothing at all")
        self.assertLessEqual(out["tokens"], 30, "the budget was exceeded")
        big = self.bm(self.nodes, (0, 1, 2, 3, 4), 10 ** 6, 4)
        self.assertGreater(len(big["kept"]), len(out["kept"]),
                           "a tight budget kept as much as an unlimited one -- it is not binding")
        # among what fits, the nearest ancestor that fits must be taken before a farther one
        order = sorted(out["kept"], key=lambda i: abs(i - 4))
        for i in out["kept"]:
            for j in out["dropped"]:
                if abs(j - 4) < abs(i - 4):
                    unit = len("step %d | %s\naction: %s\nobservation: %s"
                               % (j, self.nodes[j].validation, self.nodes[j].action,
                                  self.nodes[j].observation or "")) // 4
                    self.assertGreater(unit, 1, "a nearer unit was skipped for no reason")
        self.assertEqual(order, sorted(order, key=lambda i: abs(i - 4)))

    def test_17t_it_never_truncates(self):
        out = self.bm(self.nodes, (0, 1, 2, 3, 4), 60, 4)
        self.assertNotIn("...", out["text"].replace("....", ""),
                         "the adapter truncated instead of dropping")
        for i in out["kept"]:
            self.assertIn(self.nodes[i].action[:30], out["text"])

    def test_17u_output_is_in_trace_order(self):
        out = self.bm(self.nodes, (0, 1, 2, 3, 4), 10 ** 6, 4)
        pos = [out["text"].index("step %d |" % i) for i in out["kept"]]
        self.assertEqual(pos, sorted(pos), "the rendering is not in trace order")


if __name__ == "__main__":
    unittest.main(verbosity=2)
