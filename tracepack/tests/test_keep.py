"""Contract #21 -- the "keep" rule (tracepack.condenser.keep): which forgotten file views come back.

Synthetic OpenHands-shaped event dicts (the persisted JSON shape), no SDK.  Each test isolates one
clause of the rule; the equivalence with the measured implementation on real runs is checked outside
the repository (the runs' events are not public).
"""
from __future__ import annotations

import copy
import os
import sys
import unittest

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(os.path.dirname(HERE))
if ROOT not in sys.path:
    sys.path.insert(0, ROOT)

from tracepack.condenser import keep  # noqa: E402


class _Trace:
    """Builds an event list: views, edits, terminal commands, compactions."""

    def __init__(self):
        self.events = []
        self.n = 0

    def _id(self, prefix):
        self.n += 1
        return "%s%d" % (prefix, self.n)

    def act(self, tool, action, resp=None, thought=""):
        aid = self._id("a")
        ev = {"kind": "ActionEvent", "id": aid, "tool_name": tool, "action": action,
              "llm_response_id": resp or self._id("r"), "tool_call_id": "tc-" + aid,
              "thought": [{"type": "text", "text": thought}] if thought else []}
        self.events.append(ev)
        return ev

    def obs(self, action_ev, text):
        oid = self._id("o")
        ev = {"kind": "ObservationEvent", "id": oid, "action_id": action_ev["id"],
              "tool_call_id": action_ev["tool_call_id"],
              "observation": {"content": [{"cache_prompt": False, "type": "text", "text": text}]}}
        self.events.append(ev)
        return ev

    def view(self, path, text="x = 1\n", rng=None, resp=None, thought=""):
        action = {"command": "view", "path": path}
        if rng:
            action["view_range"] = rng
        a = self.act("file_editor", action, resp=resp, thought=thought)
        o = self.obs(a, text)
        return a, o

    def edit(self, path, command="str_replace"):
        a = self.act("file_editor", {"command": command, "path": path})
        self.obs(a, "edited")
        return a

    def term(self, command, resp=None):
        a = self.act("terminal", {"command": command}, resp=resp)
        self.obs(a, "ok")
        return a

    def compact(self, forget, summary):
        ev = {"kind": "Condensation", "id": self._id("c"),
              "forgotten_event_ids": [e["id"] for e in forget], "summary": summary}
        self.events.append(ev)
        return len(self.events) - 1


def _paths(sel):
    return [it["path"] for it in sel["items"]]


class Contract21KeepRule(unittest.TestCase):

    def test_21a_forgotten_view_named_in_summary_comes_back_whole(self):
        t = _Trace()
        a, o = t.view("/testbed/pkg/core.py", "def f():\n    return 1\n")
        up = t.compact([a, o], "Read core.py; f returns 1.")
        sel = keep.select(t.events, up)
        self.assertEqual(_paths(sel), ["pkg/core.py"])           # relative to /testbed
        self.assertEqual(sel["unforget"], sorted([a["id"], o["id"]]))
        # the caps measure every string in the observation record, as the measured runs did
        self.assertEqual(sel["chars"], len(keep._obs_text(o)))
        self.assertEqual(sel["items"][0]["range"], [1, -1])      # whole file

    def test_21b_latest_read_still_in_context_is_not_put_back(self):
        t = _Trace()
        a1, o1 = t.view("/testbed/pkg/core.py")
        a2, o2 = t.view("/testbed/pkg/core.py")                  # the later read stays in view
        up = t.compact([a1, o1], "core.py")
        sel = keep.select(t.events, up)
        self.assertEqual(sel["items"], [])
        self.assertEqual(sel["dropped"], {"latest_in_context": 1})

    def test_21c_edited_after_the_read_is_stale(self):
        t = _Trace()
        a, o = t.view("/testbed/pkg/core.py")
        e = t.edit("/testbed/pkg/core.py")
        up = t.compact([a, o, e], "core.py edited")
        self.assertEqual(keep.select(t.events, up)["dropped"], {"edited_since": 1})
        for cmd in ("sed -i 's/1/2/' pkg/core.py", "echo x > pkg/core.py", "git checkout pkg/core.py"):
            t = _Trace()
            a, o = t.view("/testbed/pkg/core.py")
            t.term(cmd)
            up = t.compact([a, o], "core.py")
            with self.subTest(cmd=cmd):
                self.assertEqual(keep.select(t.events, up)["dropped"], {"edited_since": 1})

    def test_21d_an_edit_before_the_read_does_not_count(self):
        t = _Trace()
        t.edit("/testbed/pkg/core.py")
        t.term("sed -i 's/1/2/' pkg/core.py")
        a, o = t.view("/testbed/pkg/core.py")
        up = t.compact([a, o], "core.py")
        self.assertEqual(_paths(keep.select(t.events, up)), ["pkg/core.py"])

    def test_21e_directory_listing_oversize_unnamed_and_unobserved_are_dropped(self):
        t = _Trace()
        a1, o1 = t.view("/testbed/pkg.d", "Here's the files and directories up to 2 levels deep in /testbed/pkg.d")
        a2, o2 = t.view("/testbed/pkg/big.py", "y" * (keep.MAX_READ_CHARS + 1))
        a3, o3 = t.view("/testbed/pkg/other.py")
        a4 = t.act("file_editor", {"command": "view", "path": "/testbed/pkg/lost.py"})   # no observation
        up = t.compact([a1, o1, a2, o2, a3, o3, a4], "pkg.d big.py lost.py")             # other.py unnamed
        sel = keep.select(t.events, up)
        self.assertEqual(sel["items"], [])
        self.assertEqual(sel["dropped"], {"dir_view": 1, "over_read_cap": 1, "not_in_summary": 1,
                                          "no_observation": 1})

    def test_21f_paths_without_an_extension_are_not_file_views(self):
        t = _Trace()
        a, o = t.view("/testbed/pkg", "Here's the files and directories up to 2 levels deep in /testbed/pkg")
        up = t.compact([a, o], "pkg")
        sel = keep.select(t.events, up)
        self.assertEqual((sel["items"], sel["dropped"], sel["n_candidates_latest"]), ([], {}, 0))

    def test_21g_newest_first_and_the_file_cap(self):
        t = _Trace()
        evs, names = [], []
        for k in range(keep.MAX_FILES + 2):
            a, o = t.view("/testbed/m%d.py" % k, "v%d\n" % k)
            evs += [a, o]
            names.append("m%d.py" % k)
        up = t.compact(evs, " ".join(names))
        sel = keep.select(t.events, up)
        self.assertEqual(_paths(sel), ["m5.py", "m4.py", "m3.py", "m2.py"])
        self.assertEqual(sel["dropped"], {"over_file_cap": 2})

    def test_21h_the_total_cap_skips_and_keeps_going(self):
        t = _Trace()
        sizes = [5000, 12000, 12000, 12000]                      # oldest first
        evs, obs = [], []
        for k, n in enumerate(sizes):
            a, o = t.view("/testbed/f%d.py" % k, "z" * n)
            evs += [a, o]
            obs.append(o)
        up = t.compact(evs, "f0.py f1.py f2.py f3.py")
        sel = keep.select(t.events, up)
        # newest first: 12000 + 12000; the third 12000 would pass 32000; the 5000 still fits
        self.assertEqual(_paths(sel), ["f3.py", "f2.py", "f0.py"])
        self.assertEqual(sel["chars"], sum(len(keep._obs_text(obs[k])) for k in (3, 2, 0)))
        self.assertEqual(sel["dropped"], {"over_total_cap": 1})

    def test_21i_the_whole_model_response_comes_back(self):
        t = _Trace()
        a, o = t.view("/testbed/pkg/core.py", resp="r-batch")
        sib = t.term("pytest -q", resp="r-batch")                # same model response, not a view
        sib_obs = t.events[-1]
        up = t.compact([a, o, sib, sib_obs], "core.py")
        sel = keep.select(t.events, up)
        self.assertEqual(sel["unforget"], sorted([a["id"], o["id"], sib["id"], sib_obs["id"]]))
        self.assertEqual(sel["batch_siblings"], 1)

    def test_21j_only_forgotten_events_are_put_back(self):
        t = _Trace()
        a, o = t.view("/testbed/pkg/core.py", resp="r-batch")
        sib = t.term("ls", resp="r-batch")                       # sibling still in context
        up = t.compact([a, o], "core.py")
        sel = keep.select(t.events, up)
        self.assertNotIn(sib["id"], sel["unforget"])
        self.assertEqual(sel["unforget"], sorted([a["id"], o["id"]]))

    def test_21k_forgotten_ids_accumulate_over_compactions(self):
        t = _Trace()
        a, o = t.view("/testbed/pkg/core.py")
        t.compact([a, o], "first summary without the file")
        b, p = t.view("/testbed/pkg/util.py")
        up = t.compact([b, p], "core.py util.py")
        sel = keep.select(t.events, up)
        self.assertEqual(_paths(sel), ["pkg/util.py", "pkg/core.py"])
        self.assertEqual(keep.forgotten_ids(t.events, up), {a["id"], o["id"], b["id"], p["id"]})

    def test_21l_pure_and_deterministic(self):
        t = _Trace()
        a, o = t.view("/testbed/pkg/core.py", thought="")
        t.act("think", {"thought": "core.py looks fine"})
        up = t.compact([a, o], "core.py")
        before = copy.deepcopy(t.events)
        s1 = keep.select(t.events, up)
        s2 = keep.select(t.events, up)
        self.assertEqual(s1, s2)
        self.assertEqual(t.events, before)                       # nothing modified
        self.assertEqual(s1["items"][0]["note"], "core.py looks fine")

    def test_21m_upto_must_be_a_condensation(self):
        t = _Trace()
        t.view("/testbed/pkg/core.py")
        with self.assertRaises(ValueError):
            keep.select(t.events, 0)
        self.assertIsNone(keep.latest_condensation(t.events))

    def test_21n_every_constant_cites_a_reading(self):
        for name in ("MAX_FILES", "MAX_READ_CHARS", "MAX_TOTAL_CHARS", "NOTE_CHARS", "WORKSPACE_ROOT",
                     "EDIT_COMMANDS"):
            with self.subTest(name=name):
                self.assertIn(name, keep.PROVENANCE)
        self.assertEqual((keep.MAX_FILES, keep.MAX_READ_CHARS, keep.MAX_TOTAL_CHARS), (4, 16000, 32000))


if __name__ == "__main__":
    unittest.main()
