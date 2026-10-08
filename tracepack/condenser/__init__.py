"""tracepack.condenser -- the deliverable: a recall step for harnesses whose summaries lose values.

`REPORT_abstract.md` §6.  Two layers, on purpose:

* :mod:`tracepack.condenser.core` is harness-agnostic and imports nothing outside this repo, so it
  runs (and is contract-tested) on python 3.9 with no third-party dependency.
* :mod:`tracepack.condenser.oh_condenser` is the OpenHands SDK binding -- the harness the read-out
  numbers were measured on.  It needs python >= 3.12 and `openhands-sdk`, so it is imported lazily.

Scope, stated up front because it is part of the claim (`REPORT_abstract` §3 conclusion three):
this helps when the harness's summary drops values or the source disappears, AND the record you need
is reachable over >= 2 dependency hops from what retrieval finds.  When the gate is shut you get
plain verbatim retrieval, which is worth having but is not the mechanism.
"""
from .core import Recall, TracePackRecall                    # noqa: F401
from .recipe import LEGACY, PROVENANCE, RECIPE, Recipe       # noqa: F401

__all__ = ["Recall", "TracePackRecall", "Recipe", "RECIPE", "LEGACY", "PROVENANCE"]
