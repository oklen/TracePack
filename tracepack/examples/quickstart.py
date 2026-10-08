#!/usr/bin/env python3
"""The smallest thing that shows what TracePack does.  No model, no network, no dependencies.

    python -m tracepack.examples.quickstart

It builds one evidence packet from a six-row trace and prints the packet plus its manifest, so the
budget contract (`incomplete`) is visible rather than implied.
"""
from __future__ import annotations

import os
import sys

if __package__ in (None, ""):                      # allow `python quickstart.py` from anywhere
    sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

import tracepack                                    # noqa: E402

TRACE = [
    {"tracepack_format": "openhands"},
    {"id": "m0", "role": "user", "content": "the tests fail with a TypeError in units"},
    {"id": "m1", "role": "assistant", "content": "run the suite",
     "tool_calls": [{"id": "c1", "type": "function",
                     "function": {"name": "bash",
                                  "arguments": '{"command": "pytest -q astropy/units"}'}}]},
    {"id": "m2", "role": "tool",
     "content": "E   TypeError: unsupported operand for Quantity\n"
                "    at astropy/units/quantity.py line 1690"},
    {"id": "m3", "role": "assistant", "content": "read the comparison",
     "tool_calls": [{"id": "c2", "type": "function",
                     "function": {"name": "bash",
                                  "arguments": '{"command": "sed -n 1680,1700p astropy/units/quantity.py"}'}}]},
    {"id": "m4", "role": "tool",
     "content": "def __eq__(self, other):\n"
                "    other = Quantity(other)   # raises instead of returning NotImplemented"},
]

QUERY = "why does comparing a Quantity to a string raise instead of returning NotImplemented?"


def main() -> int:
    pkt = tracepack.pack(TRACE, query=QUERY, budget=2048)
    print("tracepack %s" % tracepack.__version__)
    print("  events %s | seeds %s | served %s entries, %s tokens of %s | incomplete=%s"
          % (pkt["n_events"], pkt["n_seeds"], pkt["n_entries"], pkt["tokens"], pkt["budget"],
             pkt["incomplete"]))
    print("  served ids: %s" % pkt["served"])
    print("-" * 78)
    print(pkt["text"])
    print("-" * 78)
    # the same call at a budget too small to hold the evidence: the contract has to SAY so
    tight = tracepack.pack(TRACE, query=QUERY, budget=40)
    print("at budget 40: %s tokens, %s entries, incomplete=%s  <- the packet declares the shortfall"
          % (tight["tokens"], tight["n_entries"], tight["incomplete"]))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
