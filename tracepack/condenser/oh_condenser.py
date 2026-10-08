"""OpenHands SDK binding: native summary, plus the records the summary dropped, fetched on demand.

    from tracepack.condenser.oh_condenser import TracePackCondenser

    agent = agent.model_copy(update={"condenser": TracePackCondenser(llm=summary_llm, max_size=120)})

A drop-in for ``LLMSummarizingCondenser`` -- same summary, same trigger, same knobs -- with one
addition: when the condenser forgets events it keeps them, and when the next user instruction
arrives it puts a budgeted verbatim packet of the relevant ones back into the view.

**Why the packet is built at query time, not at condensation time.**  The measured effect (+23 points
of agent completion over same-budget verbatim retrieval, +40 over the native summarizer,
`RESULTS_pilot.md` round 2) comes from retrieving *with the instruction as the query*.  At
condensation time there is no query.  A query-free condense-time packet has never been measured, and
this project's most expensive lesson is what happens when a default is a claim nobody measured
(`RESULTS_ablation.md` §24).  So: condensation archives, the instruction retrieves.

**What it costs.**  One packet (default 2,048 tokens) on the first LLM call of each turn that follows
a condensation, and it is cached per instruction so the tool loop inside a turn pays nothing extra.
No model, no network, no index to maintain: the seeding fuses BM25 with an offline hashing embedding,
and the graph is rebuilt from the archive per query.

**Where it does not help** (`REPORT_abstract.md` §3 conclusion three): when the harness's summary
keeps the values, when the source is still on disk and the agent will just re-read it, or when what
you need is one hop from what retrieval already finds.  In the last case the gate shuts by itself and
you get plain verbatim retrieval.

Requires python >= 3.12 and ``openhands-sdk`` (the SDK's own floor).  Everything in
:mod:`tracepack.condenser.core` and :mod:`tracepack.condenser.oh_rows` runs on 3.9 with no dependency.

The file is NOT called ``openhands.py``: run any script that sits in this directory and python puts
the directory first on ``sys.path``, so ``import openhands`` would find this module instead of the
SDK ("'openhands' is not a package").  Caught by selfcheck_oh.py the first time it ran on a machine
that actually had the SDK installed.
"""
from __future__ import annotations

import gzip
import json
import logging
import os
import threading

try:
    from openhands.sdk.context.condenser import LLMSummarizingCondenser
    from openhands.sdk.context.view import View
    from openhands.sdk.event.base import LLMConvertibleEvent
    from openhands.sdk.event.condenser import Condensation, CondensationSummaryEvent
    from openhands.sdk.llm import LLM
    from pydantic import Field, PrivateAttr
except ImportError as exc:                                         # pragma: no cover
    raise ImportError(
        "tracepack.condenser.oh_condenser needs `openhands-sdk` (python >= 3.12). "
        "The engine in tracepack.condenser.core has no such requirement."
    ) from exc

from ..adapters.openhands import OpenHandsAdapter
from .core import Recall, TracePackRecall
from .oh_rows import HEADER_ROW, events_to_rows, last_user_text
from .recipe import RECIPE, Recipe

logger = logging.getLogger(__name__)


class TracePackCondenser(LLMSummarizingCondenser):
    """``LLMSummarizingCondenser`` + query-time recall over the events it forgot.

    Every knob below is a *deployment* choice.  The retrieval recipe itself lives in
    :mod:`tracepack.condenser.recipe`, where each default cites the reading that fixed it.
    """

    budget: int = Field(default=RECIPE.budget, gt=0,
                        description="hard token ceiling for the injected packet")
    router: str = Field(default=RECIPE.router, description="seed router: hybrid | lexical | dense")
    k: int = Field(default=RECIPE.k, gt=0, description="number of seeds")
    max_hops: int = Field(default=RECIPE.max_hops, ge=0, description="closure depth; rule >= 2n+2")
    gate_hops: int = Field(default=RECIPE.gate_hops, ge=0,
                           description="run the closure only when a seed reaches this many hops; "
                                       "below it the packet is plain verbatim retrieval")
    archive_path: str | None = Field(
        default=None, description="optional .jsonl(.gz) the forgotten events are also written to, so "
                                  "a finished or crashed run can be replayed offline")
    log_path: str | None = Field(
        default=None, description="optional .jsonl, one line per recall (no packet text): what the "
                                  "gate said, what was served, how long it took")
    enabled: bool = Field(default=True,
                          description="False = behave exactly like the stock condenser (the control "
                                      "arm of a regression run)")

    _rows: list = PrivateAttr(default_factory=list)
    _seen: set = PrivateAttr(default_factory=set)
    _cache: object = PrivateAttr(default=None)          # (query, Recall) | None
    _engine: object = PrivateAttr(default=None)         # TracePackRecall | None
    _lock: object = PrivateAttr(default_factory=threading.Lock)

    # -- plumbing -------------------------------------------------------------------------------

    @property
    def recipe(self) -> Recipe:
        return Recipe(router=self.router, k=self.k, gate_hops=self.gate_hops,
                      max_hops=self.max_hops, budget=self.budget)

    @property
    def engine(self) -> TracePackRecall:
        if self._engine is None:
            self._engine = TracePackRecall(self.recipe)
        return self._engine

    def archived_rows(self) -> list:
        """The forgotten-event corpus in adapter row form, header row included."""
        return [dict(HEADER_ROW)] + list(self._rows)

    def _archive(self, events, forgotten_ids) -> int:
        """Keep the events this condensation is dropping.  Never keeps anything still in view."""
        fresh = {i for i in forgotten_ids if i not in self._seen}
        if not fresh:
            return 0
        rows = events_to_rows(events, keep_ids=fresh, start=len(self._rows))
        self._rows.extend(rows)
        self._seen |= fresh
        self._cache = None                     # the corpus changed, so any cached packet is stale
        if self.archive_path and rows:
            try:
                opener = gzip.open if self.archive_path.endswith(".gz") else open
                new = not os.path.exists(self.archive_path)
                with opener(self.archive_path, "at", encoding="utf-8") as fh:
                    if new:
                        fh.write(json.dumps(HEADER_ROW, ensure_ascii=False) + "\n")
                    for r in rows:
                        fh.write(json.dumps(r, ensure_ascii=False) + "\n")
            except OSError as exc:                                 # pragma: no cover
                logger.warning("tracepack archive write failed: %s", exc)
        return len(rows)

    def _log(self, query: str, rec: Recall, error: str = "") -> None:
        if not self.log_path:
            return
        row = rec.as_dict()
        row["query_chars"] = len(query)
        row["n_archived_rows"] = len(self._rows)
        if error:
            row["error"] = error
        try:
            with open(self.log_path, "a", encoding="utf-8") as fh:
                fh.write(json.dumps(row, ensure_ascii=False) + "\n")
        except OSError:                                            # pragma: no cover
            pass                                                   # telemetry never breaks a run

    # -- recall ---------------------------------------------------------------------------------

    def recall_for(self, query: str) -> Recall:
        """The packet for ``query`` over everything forgotten so far.  Cached per query."""
        with self._lock:
            cached = self._cache
        if cached is not None and cached[0] == query:
            return cached[1]
        graph = OpenHandsAdapter(link_values=True).normalize(self.archived_rows())
        rec = self.engine.recall(graph, query, budget=self.budget, already_scoped=True)
        with self._lock:
            self._cache = (query, rec)
        self._log(query, rec)
        return rec

    def _inject(self, view: View) -> View:
        """A view with the packet placed just before the newest user instruction."""
        if not self.enabled or not self._rows:
            return view
        idx, query = last_user_text(view.events)
        if idx < 0 or not query.strip():
            return view
        try:
            rec = self.recall_for(query)
        except Exception as exc:                                   # pragma: no cover
            # A recall failure degrades to the stock condenser.  Retrieval is an optimisation; it
            # must never be able to take the agent down.
            logger.warning("tracepack recall failed: %s: %s", type(exc).__name__, exc)
            self._log(query, Recall(text="", tokens=0), error="%s: %s" % (type(exc).__name__, exc))
            return view
        if not rec.text:
            return view
        packet: LLMConvertibleEvent = CondensationSummaryEvent(
            id="tracepack-%d-%d" % (len(self._rows), rec.tokens),
            summary=rec.text, source="environment")
        events = list(view.events)
        events.insert(idx, packet)
        return View(events=events,
                    unhandled_condensation_request=view.unhandled_condensation_request)

    # -- the two SDK entry points ---------------------------------------------------------------

    def condense(self, view: View, agent_llm: "LLM | None" = None) -> "View | Condensation":
        result = super().condense(view, agent_llm=agent_llm)
        if isinstance(result, Condensation):
            self._archive(view.events, set(result.forgotten_event_ids or ()))
            return result
        return self._inject(result)

    async def acondense(self, view: View, agent_llm: "LLM | None" = None) -> "View | Condensation":
        result = await super().acondense(view, agent_llm=agent_llm)
        if isinstance(result, Condensation):
            self._archive(view.events, set(result.forgotten_event_ids or ()))
            return result
        return self._inject(result)
