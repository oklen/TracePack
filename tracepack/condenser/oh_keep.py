"""OpenHands SDK binding of the "keep" rule: the stock summary, plus the agent's own forgotten file views.

    from tracepack.condenser.oh_keep import KeepCondenser

    agent = agent.model_copy(update={"condenser": KeepCondenser(llm=summary_llm, max_size=120)})

A drop-in for ``LLMSummarizingCondenser`` -- same summary, same trigger, same knobs.  At every
compaction it picks, with :func:`tracepack.condenser.keep.select`, which of the file views the
compaction just forgot to put back, and from then on every View it returns carries those events
right after the summary, verbatim and in their original order, until the next compaction picks again.

**Why the views go into the View and not into the conversation state.**  The condenser counts the
state when it decides to compact.  The probe that put the views back into the state compacted earlier
because of them (0.19 -> 0.55 re-compactions in 12 steps, `PREREG_keep_probe.md` §8), and every
re-compaction forgets again.  Inserted into the View only, they cost their tokens on every call
(at most ``keep.MAX_TOTAL_CHARS``, about 8k tokens) but never move the compaction line.  This is the
form that was measured (`PREREG_keep.md` §2, §8).

**Where the forgotten events come from.**  Two sources, in order:

* ``events_dir`` -- the conversation's persisted events (``<persistence dir>/<conversation id>/events``).
  This is what the measured runs read, and it also works for a conversation resumed from disk.
* otherwise the condenser's own archive of every event it has seen in a View.  A View holds every event
  until a compaction forgets it, so for a conversation that started with this condenser the archive
  holds the same events in the same order.  A condenser swapped in mid-conversation cannot put back what
  it never saw.

**Failures fall back to the stock view** and are logged (``log_path``); the condenser never raises
into the agent loop because of the put-back.

Requires python >= 3.12 and ``openhands-sdk``; the rule itself (:mod:`tracepack.condenser.keep`) is
stdlib-only.  Not called ``openhands*.py`` for the reason given in :mod:`tracepack.condenser.oh_condenser`.
"""
from __future__ import annotations

import collections
import json
import logging
import os
import threading
import time

try:
    from openhands.sdk.context.condenser import LLMSummarizingCondenser
    from openhands.sdk.context.view import View
    from openhands.sdk.event.base import Event
    from openhands.sdk.event.condenser import Condensation, CondensationSummaryEvent
    from openhands.sdk.llm import LLM
    from pydantic import Field, PrivateAttr
except ImportError as exc:                                         # pragma: no cover
    raise ImportError(
        "tracepack.condenser.oh_keep needs `openhands-sdk` (python >= 3.12). "
        "The selection rule in tracepack.condenser.keep has no such requirement."
    ) from exc

from . import keep

logger = logging.getLogger(__name__)

__all__ = ["KeepCondenser", "load_events"]


def load_events(events_dir: str) -> list:
    """The persisted events of one conversation, in order, as dicts."""
    out = []
    for name in sorted(os.listdir(events_dir)):
        if not name.startswith("event-"):
            continue
        with open(os.path.join(events_dir, name), encoding="utf-8") as fh:
            out.append(json.load(fh))
    return out


def _as_dict(event) -> dict:
    return json.loads(event.model_dump_json())


class KeepCondenser(LLMSummarizingCondenser):
    """``LLMSummarizingCondenser`` + the forgotten file views put back after the summary."""

    events_dir: str | None = Field(
        default=None, description="the conversation's persisted events directory; when unset the "
                                  "condenser keeps its own archive of the events it has seen")
    log_path: str | None = Field(
        default=None, description="optional .jsonl, one line per compaction: what was put back")
    enabled: bool = Field(default=True,
                          description="False = behave exactly like the stock condenser (the control "
                                      "arm of a comparison)")

    _objs: list = PrivateAttr(default_factory=list)          # archive: event dicts in order
    _by_id: dict = PrivateAttr(default_factory=dict)         # archive: id -> the Event object
    _sel_for: object = PrivateAttr(default=None)             # id of the Condensation selected for
    _kept: list = PrivateAttr(default_factory=list)          # Events to insert after the summary
    _n: int = PrivateAttr(default=0)
    _chars: int = PrivateAttr(default=0)
    _steps_with: int = PrivateAttr(default=0)
    _errors: int = PrivateAttr(default=0)
    _selected: object = PrivateAttr(default_factory=collections.Counter)
    _lock: object = PrivateAttr(default_factory=threading.Lock)

    def stats(self) -> dict:
        """Counters for an audit: compactions selected for, characters put back, calls that carried
        the put-back, errors, and how many files each compaction kept."""
        return {"selections": self._n, "kept_chars_total": self._chars, "steps_with_insert": self._steps_with,
                "errors": self._errors, "selected_files": dict(self._selected)}

    # -- plumbing -------------------------------------------------------------------------------

    def _log(self, row: dict) -> None:
        if not self.log_path:
            return
        try:
            with open(self.log_path, "a", encoding="utf-8") as fh:
                fh.write(json.dumps(row, ensure_ascii=False) + "\n")
        except OSError:                                            # pragma: no cover
            pass                                                   # telemetry never breaks a run

    def _fail(self, exc: Exception) -> None:
        self._errors += 1
        logger.warning("tracepack keep failed: %s: %s", type(exc).__name__, exc)
        self._log({"t": int(time.time()), "error": "%s: %s" % (type(exc).__name__, str(exc)[:300])})

    def _remember(self, view: View) -> None:
        """Archive every event of ``view`` not seen yet (the summary placeholder excluded)."""
        for e in view.events:
            if isinstance(e, CondensationSummaryEvent):
                continue
            eid = str(e.id)
            if eid not in self._by_id:
                self._by_id[eid] = e
                self._objs.append(_as_dict(e))

    def _compute(self, objs: list, cond_id: str) -> None:
        """The put-back for the Condensation ``objs[-1]``; every event before it is in ``objs``."""
        self._n += 1
        sel = keep.select(objs, len(objs) - 1)
        ids = set(sel["unforget"])
        kept = []
        for o in objs[:-1]:
            oid = str(o.get("id"))
            if oid not in ids:
                continue
            ev = self._by_id.get(oid)
            if ev is None:                                 # from disk: validate a copy (validation edits the dict)
                ev = Event.model_validate(json.loads(json.dumps(o)))
            kept.append(ev)
        self._kept = kept
        self._sel_for = cond_id
        self._chars += sel["chars"]
        self._selected[len(sel["items"])] += 1
        self._log({"t": int(time.time()), "condensation": cond_id, "n_selected": len(sel["items"]),
                   "n_events": len(kept), "chars": sel["chars"],
                   "files": [(it["path"], it["range"], it["step"]) for it in sel["items"]],
                   "dropped": sel["dropped"]})

    def _ensure(self) -> None:
        """First call: pick for the last Condensation already on disk (a resumed conversation)."""
        if self._sel_for is not None:
            return
        self._sel_for = ""
        if not self.events_dir:
            return
        try:
            objs = load_events(self.events_dir)
            last = keep.latest_condensation(objs)
            if last is not None:
                self._compute(objs[:last + 1], str(objs[last].get("id")))
        except Exception as exc:
            self._kept = []
            self._fail(exc)

    def _on_condensation(self, cond: Condensation, view: View) -> None:
        try:
            if self.events_dir:
                objs = load_events(self.events_dir)
                have = {str(x.get("id")) for x in objs}
                for e in view.events:                     # events not written to disk yet
                    if str(e.id) not in have and not isinstance(e, CondensationSummaryEvent):
                        objs.append(_as_dict(e))
                objs.append(_as_dict(cond))
            else:
                self._objs.append(_as_dict(cond))
                objs = list(self._objs)
            self._compute(objs, str(cond.id))
        except Exception as exc:
            self._kept = []
            self._fail(exc)

    def _augment(self, view: View) -> View:
        """``view`` with the kept events right after the summary (a copy; the state is untouched)."""
        try:
            if not self._kept:
                return view
            idx = next((i for i, e in enumerate(view.events) if isinstance(e, CondensationSummaryEvent)), None)
            if idx is None:
                return view
            have = {str(e.id) for e in view.events}
            add = [e for e in self._kept if str(e.id) not in have]
            if not add:
                return view
            self._steps_with += 1
            return view.model_copy(update={"events": list(view.events[:idx + 1]) + add + list(view.events[idx + 1:])})
        except Exception as exc:                          # never take the agent down: fall back to stock
            self._fail(exc)
            return view

    # -- the two SDK entry points ---------------------------------------------------------------

    def condense(self, view: View, agent_llm: "LLM | None" = None) -> "View | Condensation":
        if not self.enabled:
            return super().condense(view, agent_llm=agent_llm)
        with self._lock:
            if not self.events_dir:
                self._remember(view)
            self._ensure()
        result = super().condense(view, agent_llm=agent_llm)
        with self._lock:
            if isinstance(result, Condensation):
                self._on_condensation(result, view)
                return result
            return self._augment(result)

    async def acondense(self, view: View, agent_llm: "LLM | None" = None) -> "View | Condensation":
        if not self.enabled:
            return await super().acondense(view, agent_llm=agent_llm)
        with self._lock:
            if not self.events_dir:
                self._remember(view)
            self._ensure()
        result = await super().acondense(view, agent_llm=agent_llm)
        with self._lock:
            if isinstance(result, Condensation):
                self._on_condensation(result, view)
                return result
            return self._augment(result)
