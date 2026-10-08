"""The public API, pinned.

`tracepack.pack` is the one entry point an integrator is meant to call, so the things they will rely
on -- that it takes rows as well as a path, that the budget is a ceiling and not a suggestion, and
that a shortfall is declared rather than swallowed -- are asserted here.  Without this the SDK can
rot silently while every experiment script keeps working through its own sys.path.
"""
from __future__ import annotations

import json
import os
import sys
import tempfile
import unittest

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

import tracepack                                                        # noqa: E402

TRACE = [
    {"tracepack_format": "openhands"},
    {"id": "m0", "role": "user", "content": "the tests fail with a TypeError in units"},
    {"id": "m1", "role": "assistant", "content": "run the suite",
     "tool_calls": [{"id": "c1", "type": "function",
                     "function": {"name": "bash", "arguments": '{"command": "pytest -q astropy/units"}'}}]},
    {"id": "m2", "role": "tool",
     "content": "E   TypeError: unsupported operand for Quantity at astropy/units/quantity.py line 1690"},
    {"id": "m3", "role": "assistant", "content": "read the comparison",
     "tool_calls": [{"id": "c2", "type": "function",
                     "function": {"name": "bash", "arguments": '{"command": "sed -n 1680,1700p astropy/units/quantity.py"}'}}]},
    {"id": "m4", "role": "tool",
     "content": "def __eq__(self, other):\n    other = Quantity(other)   # raises for a string"},
]
QUERY = "why does comparing a Quantity to a string raise?"


class SdkSurface(unittest.TestCase):
    def test_sdk1_pack_accepts_rows_in_memory(self):
        p = tracepack.pack(TRACE, QUERY, budget=2048)
        self.assertTrue(p["text"], "an empty packet from a trace that answers the query")
        self.assertGreater(p["n_entries"], 0)
        self.assertIn("quantity.py", p["text"])

    def test_sdk2_pack_accepts_a_path_and_agrees_with_the_rows_form(self):
        with tempfile.NamedTemporaryFile("w", suffix=".jsonl", delete=False, encoding="utf-8") as fh:
            for r in TRACE:
                fh.write(json.dumps(r) + "\n")
            path = fh.name
        try:
            a = tracepack.pack(TRACE, QUERY, budget=2048)
            b = tracepack.pack(path, QUERY, budget=2048)
            self.assertEqual(a["text"], b["text"], "the same trace packed two ways gave two packets")
        finally:
            os.unlink(path)

    def test_sdk3_budget_is_a_ceiling_not_a_suggestion(self):
        for budget in (40, 120, 400, 2048):
            p = tracepack.pack(TRACE, QUERY, budget=budget)
            self.assertLessEqual(p["tokens"], budget,
                                 "packet spent %d of a %d budget" % (p["tokens"], budget))

    def test_sdk4_a_shortfall_is_declared(self):
        tight = tracepack.pack(TRACE, QUERY, budget=40)
        full = tracepack.pack(TRACE, QUERY, budget=2048)
        self.assertTrue(tight["incomplete"], "a budget that cannot hold the evidence reported complete")
        self.assertFalse(full["incomplete"])

    def test_sdk5_the_adopted_recipe_is_reachable_by_name(self):
        c = tracepack.pack(TRACE, QUERY, budget=2048)
        x = tracepack.pack(TRACE, QUERY, budget=2048,
                           selection="closure_h", assembly="evidence_first_x")
        self.assertEqual(c["selection"], "closure")
        self.assertEqual(x["assembly"], "evidence_first_x")

    def test_sdk6_importing_the_package_costs_no_third_party_import(self):
        """`import tracepack` must not drag in a model stack; the integrations are lazy."""
        import subprocess
        root = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
        code = ("import sys; sys.path.insert(0, %r); import tracepack; "
                "bad = [m for m in sys.modules if m.split('.')[0] in "
                "('openhands', 'torch', 'numpy', 'litellm', 'pydantic')]; "
                "print(','.join(sorted(bad)))" % root)
        out = subprocess.run([sys.executable, "-c", code], capture_output=True, text=True)
        self.assertEqual(out.stdout.strip(), "",
                         "importing tracepack pulled in %s" % out.stdout.strip())


if __name__ == "__main__":
    unittest.main()
