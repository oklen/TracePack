"""Regressions for the three red-team findings (2026-09-01).

These are not hypotheticals: each one shipped, each one silently changed a headline number, and
each is cheap to reintroduce.  They live in the adversarial bucket because none of them is a
§8.1 packet contract -- they are contracts on the INSTRUMENT.

  #2  the distractor generator must not make gold and distractor separable without evidence
  #3  one word-boundary matcher, shared by builder and profiler
  #4  (found while fixing #2) the path extractor must not harvest slash-joined enumerations
  #5  (found by the profiler's own replay guard) the trace corpus must be immutable, and a
      changed trace must fail loudly rather than drift
"""
from __future__ import annotations

import os
import random
import sys
import unittest
from collections import Counter

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

from tracepack.core.textmatch import boundary_clean, contains_value, wb
from tracepack.eval.freeze_corpus import assert_frozen, sha1_file
from tracepack.eval import attack_surface as atk
from tracepack.eval.datasets import (enforce_global_uniqueness, extract_values, make_distractors,
                                     perturb_same_shape, plausible_path, shape)
from tracepack.profiler.interventions import _wb as profiler_wb


class RedTeamMatcher(unittest.TestCase):
    """#3 -- the profiler used its own matcher with "/" in the boundary class and was blind to
    68% of the path golds, while the builder saw them as present."""

    def test_one_matcher_shared_by_builder_and_profiler(self):
        self.assertIs(profiler_wb, wb)

    def test_a_path_suffix_is_found_inside_its_absolute_path(self):
        self.assertTrue(contains_value("cd /home/b/x.py && ls", "home/b/x.py"))
        self.assertTrue(wb("home/b/x.py").search("/home/b/x.py"))

    def test_fragments_of_longer_tokens_stay_invisible(self):
        self.assertTrue(contains_value("rows=4711003 done", "4711003"))
        self.assertFalse(contains_value("rows=14711003 done", "4711003"))
        self.assertFalse(contains_value("v 4711003.2", "4711003"))
        self.assertFalse(contains_value("open x.jsonl", "x.json"))

    def test_extractor_and_matcher_agree_on_the_boundary(self):
        t = "0.029023746701846966"
        self.assertFalse(boundary_clean(t, 2, len(t)))          # inside a float
        vals = [v for _, v in extract_values(t)]
        self.assertNotIn("029023746701846966", vals)


class RedTeamDistractors(unittest.TestCase):
    """#2 -- a no-evidence arm scored 0.611 (0.763 with the red team's own rule family)."""

    def test_shape_is_preserved_for_every_non_path_type(self):
        rng = random.Random(0)
        for typ, gold in (("number", "0.02902"), ("number", "1048576"),
                          ("hash", "0123456789"), ("hash", "deadbeef"),
                          ("version", "4.51.3"), ("fileline", "a/b/core.py:1204")):
            ds = make_distractors(typ, gold, rng)
            self.assertTrue(ds, "no distractor for %s %r" % (typ, gold))
            for d in ds:
                self.assertEqual(shape(d), shape(gold), "%s: %r vs %r" % (typ, gold, d))
                self.assertNotEqual(d, gold)

    def test_the_two_specific_format_leaks_are_gone(self):
        rng = random.Random(1)
        # decimal gold used to lose its point (re.sub(r"\D","",v))
        for d in make_distractors("number", "3.72", rng):
            self.assertIn(".", d)
        # an all-digit hash gold used to gain letters (hex bump)
        for d in make_distractors("hash", "12345678", rng):
            self.assertTrue(d.isdigit(), d)

    def test_fileline_perturbs_only_the_line_number(self):
        rng = random.Random(2)
        for d in make_distractors("fileline", "tracepack/core/router.py:1051", rng):
            self.assertTrue(d.startswith("tracepack/core/router.py:"), d)

    def test_perturbation_never_touches_the_leading_character(self):
        rng = random.Random(3)
        for v in ("0.5001", "1048576", "00ff00ff"):
            for d in perturb_same_shape(v, rng):
                self.assertEqual(d[0], v[0])

    def test_path_distractors_are_deferred_to_the_global_pass(self):
        # returning a per-session recycled basename here is exactly what caused the leak
        self.assertEqual(make_distractors("path", "a/b/x.py", random.Random(4)), [])

    def test_global_uniqueness_drops_a_repeat_gold_but_only_redraws_a_repeat_distractor(self):
        inv = Counter()
        items = [
            {"item_id": "i1", "gold": "a/b/x.py", "distractors": ["a/b/y.py", "a/b/z.py"]},
            {"item_id": "i2", "gold": "a/b/x.py", "distractors": ["a/b/q.py"]},   # gold repeats
            {"item_id": "i3", "gold": "a/b/w.py", "distractors": ["a/b/y.py", "a/b/v.py"]},
        ]
        kept, lost = enforce_global_uniqueness(items, inv)
        self.assertEqual({it["item_id"] for it in kept}, {"i1", "i3"})
        self.assertEqual(lost, {"i2"})
        i3 = [it for it in kept if it["item_id"] == "i3"][0]
        self.assertEqual(i3["distractors"], ["a/b/v.py"])       # y.py redrawn, item survives

    def test_the_attack_module_detects_a_planted_leak(self):
        """verify-your-checker: the gate must FAIL on a dataset built the old, leaky way."""
        leaky = []
        for i in range(120):
            # distractor is systematically shorter and recycled from a tiny pool
            leaky.append({"item_id": "x%d" % i, "type": "path",
                          "gold": "dir/long_descriptive_name_%03d.py" % i,
                          "distractors": ["dir/a%d.py" % (i % 4)]})
        ok, lines = atk.run(leaky, verbose=False)
        self.assertFalse(ok, "the attack must catch a planted recycling+length leak:\n%s"
                         % "\n".join(lines))

    def test_the_attack_module_passes_a_clean_dataset(self):
        rng = random.Random(5)
        clean = []
        for i in range(120):
            gold = "%08d" % rng.randrange(10 ** 7, 10 ** 8)
            clean.append({"item_id": "y%d" % i, "type": "number", "gold": gold,
                          "distractors": make_distractors("number", gold, rng)[:1]})
        ok, lines = atk.run(clean, verbose=False)
        self.assertTrue(ok, "a shape-matched, never-reused dataset must pass:\n%s"
                        % "\n".join(lines))


class RedTeamCorpusFreeze(unittest.TestCase):
    """#5 -- one of the ten traces is the session that builds the dataset, and it grows while the
    pipeline runs.  Three stages each normalised it independently and got three different graphs;
    nothing raised, because the adapter's ids are positional and appending renumbers nothing."""

    def setUp(self):
        import tempfile
        self.dir = tempfile.mkdtemp(prefix="tp_freeze_")
        self.trace = os.path.join(self.dir, "t.jsonl")
        with open(self.trace, "w", encoding="utf-8") as fo:
            fo.write('{"a": 1}\n')
        self.h = sha1_file(self.trace)
        self.items = [{"item_id": "i1", "transcript": self.trace, "transcript_sha1": self.h}]

    def tearDown(self):
        import shutil
        shutil.rmtree(self.dir, ignore_errors=True)

    def test_an_unchanged_corpus_passes(self):
        self.assertEqual(assert_frozen(self.items), 1)

    def test_an_APPENDED_trace_fails(self):
        # exactly the real failure: append, do not rewrite
        with open(self.trace, "a", encoding="utf-8") as fo:
            fo.write('{"a": 2}\n')
        with self.assertRaises(SystemExit) as cm:
            assert_frozen(self.items)
        self.assertIn("corpus drift", str(cm.exception))

    def test_a_missing_trace_fails(self):
        os.remove(self.trace)
        with self.assertRaises(SystemExit):
            assert_frozen(self.items)

    def test_items_without_a_recorded_hash_are_not_silently_blessed(self):
        # no hash recorded -> nothing verified; the count must say so rather than report success
        self.assertEqual(assert_frozen([{"item_id": "i", "transcript": self.trace}]), 0)


class RedTeamPathExtraction(unittest.TestCase):
    """#4 -- prose enumerations like `a.py/b.py/c.py` were harvested as paths."""

    def test_enumerations_are_rejected(self):
        self.assertFalse(plausible_path("sp50_extract.py/sp50_eval.py/build_review.py"))
        self.assertFalse(plausible_path("q23_a2.py/q23_sys_v2.txt"))

    def test_real_paths_survive(self):
        self.assertTrue(plausible_path("tracepack/core/assembler.py"))
        self.assertTrue(plausible_path("home/b/.claude/jobs/a1b2c3d4/tmp/lf_win.json"))

    def test_the_extractor_applies_it(self):
        text = "see deleak_paraphrase.py/cert_deleak.py and tracepack/core/graph.py"
        vals = [v for t, v in extract_values(text) if t == "path"]
        self.assertIn("tracepack/core/graph.py", vals)
        self.assertNotIn("deleak_paraphrase.py/cert_deleak.py", vals)


if __name__ == "__main__":
    unittest.main(verbosity=2)
