"""Binding smoke test for :mod:`tracepack.condenser.oh_keep`: run it wherever the SDK is installed (python >= 3.12).

    python3.12 -m tracepack.condenser.selfcheck_oh_keep

The rule is contract-tested on python 3.9 (`tests/test_keep.py`); what only the SDK can show is that the
put-back survives the SDK: the kept events land right after the summary, their real outputs reach the
model as tool messages, the conversation's own view is never changed, a failure falls back to the stock
view, and the two sources of forgotten events (the persisted events directory, or the condenser's own
archive) pick the same views.  No LLM is called: the summary step is scripted.
"""
from __future__ import annotations

import json
import os
import sys
import tempfile

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

import openhands.tools.file_editor                                        # noqa: E402,F401  registers the tool types
import openhands.tools.terminal                                           # noqa: E402,F401
from openhands.sdk.context.condenser.base import CondensationRequirement  # noqa: E402
from openhands.sdk.context.view import View                               # noqa: E402
from openhands.sdk.event.base import Event, LLMConvertibleEvent           # noqa: E402
from openhands.sdk.event.condenser import Condensation, CondensationSummaryEvent  # noqa: E402
from openhands.sdk.llm import LLM                                         # noqa: E402
from pydantic import PrivateAttr, SecretStr                               # noqa: E402

from tracepack.condenser import keep                                     # noqa: E402
from tracepack.condenser.oh_keep import KeepCondenser                    # noqa: E402

KINDS = {"file_editor": ("FileEditorAction", "FileEditorObservation"),
         "terminal": ("TerminalAction", "TerminalObservation")}


def action(aid, tool, args, resp):
    call = "call-" + aid
    return {"kind": "ActionEvent", "id": aid, "source": "agent", "thought": [], "thinking_blocks": [],
            "action": dict(args, kind=KINDS[tool][0]), "tool_name": tool, "tool_call_id": call,
            "tool_call": {"id": call, "name": tool, "arguments": json.dumps(args), "origin": "completion"},
            "llm_response_id": resp, "security_risk": "UNKNOWN"}


def observation(oid, act, text):
    tool, args = act["tool_name"], {k: v for k, v in act["action"].items() if k != "kind"}
    extra = {"command": args["command"], "path": args.get("path"), "prev_exist": True} if tool == "file_editor" \
        else {"command": args["command"], "exit_code": 0, "timeout": False}
    return {"kind": "ObservationEvent", "id": oid, "source": "environment", "tool_name": tool,
            "tool_call_id": act["tool_call_id"], "action_id": act["id"],
            "observation": dict({"content": [{"type": "text", "text": text}], "is_error": False,
                                 "kind": KINDS[tool][1]}, **extra)}


def trace():
    """A user message, two file views and two commands, a compaction that forgets the first four events of
    work and whose summary names core.py but not util.py, then one more command."""
    user = {"kind": "MessageEvent", "id": "m0", "source": "user",
            "llm_message": {"role": "user", "content": [{"type": "text", "text": "fix the failing test"}]}}
    a1 = action("a1", "file_editor", {"command": "view", "path": "/testbed/pkg/core.py"}, "r1")
    o1 = observation("o1", a1, "     1\tdef ratio(a, b):\n     2\t    return a / b   # KEEP-MARKER-4417\n")
    a2 = action("a2", "terminal", {"command": "pytest -q tests/test_core.py"}, "r2")
    o2 = observation("o2", a2, "E   ZeroDivisionError: division by zero\n1 failed")
    a3 = action("a3", "file_editor", {"command": "view", "path": "/testbed/pkg/util.py"}, "r3")
    o3 = observation("o3", a3, "     1\tdef helper():\n     2\t    return 0\n")
    a4 = action("a4", "terminal", {"command": "ls tests"}, "r4")
    o4 = observation("o4", a4, "test_core.py")
    cond = {"kind": "Condensation", "id": "c1", "source": "environment",
            "forgotten_event_ids": ["a1", "o1", "a2", "o2", "a3", "o3"],
            "summary": "ratio() in pkg/core.py divides by b; the test fails with ZeroDivisionError.",
            "summary_offset": 1, "llm_response_id": "r-summary"}
    a5 = action("a5", "terminal", {"command": "git status"}, "r5")
    o5 = observation("o5", a5, "modified: nothing")
    return [user, a1, o1, a2, o2, a3, o3, a4, o4, cond, a5, o5]


def as_events(objs):
    return [Event.model_validate(json.loads(json.dumps(o))) for o in objs]


class Scripted(KeepCondenser):
    """The summary step is scripted: the next Condensation is handed in instead of asking an LLM."""
    _next: object = PrivateAttr(default=None)

    def condensation_requirement(self, view, agent_llm=None):
        return CondensationRequirement.HARD if self._next is not None else None

    def get_condensation(self, view, agent_llm=None):
        r, self._next = self._next, None
        return r


def main():
    llm = LLM(model="openai/never-called", base_url="http://127.0.0.1:1/v1",
              api_key=SecretStr("unused"), usage_id="selfcheck")
    objs = trace()
    events = as_events(objs)
    up = keep.latest_condensation(objs)
    sel = keep.select(objs, up)
    assert [it["path"] for it in sel["items"]] == ["pkg/core.py"], sel["items"]
    assert sel["unforget"] == ["a1", "o1"], sel["unforget"]
    assert sel["dropped"] == {"not_in_summary": 1}, sel["dropped"]

    # ---- 1. from the persisted events: the views land right after the summary ------------------
    with tempfile.TemporaryDirectory() as d:
        for n, o in enumerate(objs):
            with open(os.path.join(d, "event-%05d-%s.json" % (n, o["id"])), "w", encoding="utf-8") as fh:
                json.dump(o, fh)
        c = KeepCondenser(llm=llm, max_size=10 ** 6, keep_first=1, events_dir=d)
        view = View.from_events(events)
        before = [str(e.id) for e in view.events]
        out = c.condense(view, agent_llm=llm)
        assert isinstance(out, View)
        assert [str(e.id) for e in view.events] == before, "the conversation's view was changed"
        ids = [str(e.id) for e in out.events]
        idx = next(i for i, e in enumerate(out.events) if isinstance(e, CondensationSummaryEvent))
        assert ids[idx + 1: idx + 3] == ["a1", "o1"], ids
        assert "a3" not in ids and "o3" not in ids, "a file the summary does not name came back"
        msgs = LLMConvertibleEvent.events_to_messages(out.events)
        tool_text = "\n".join((getattr(x, "text", "") or "") for m in msgs if m.role == "tool" for x in m.content)
        assert "KEEP-MARKER-4417" in tool_text, "the kept view's real output did not reach a tool message"

        # ---- 2. the off switch, and failures fall back to the stock view --------------------------
        off = KeepCondenser(llm=llm, max_size=10 ** 6, keep_first=1, events_dir=d, enabled=False)
        assert len(off.condense(view, agent_llm=llm).events) == len(view.events), "enabled=False inserted"
    gone = KeepCondenser(llm=llm, max_size=10 ** 6, keep_first=1, events_dir=os.path.join(d, "missing"))
    r = gone.condense(view, agent_llm=llm)
    assert isinstance(r, View) and len(r.events) == len(view.events), "a broken events_dir did not fall back"
    assert gone.stats()["errors"] == 1, gone.stats()

    # ---- 3. the condenser's own archive picks the same views as the persisted events -----------
    s = Scripted(llm=llm, max_size=10 ** 6, keep_first=1)
    so_far = []
    for o, ev in zip(objs, events):
        if o["kind"] == "Condensation":
            s._next = ev
            r = s.condense(View.from_events(so_far), agent_llm=llm)
            assert isinstance(r, Condensation) and str(r.id) == "c1"
        so_far.append(ev)
        s.condense(View.from_events(so_far), agent_llm=llm)
    assert sorted(str(e.id) for e in s._kept) == sel["unforget"], s.stats()
    final = s.condense(View.from_events(so_far), agent_llm=llm)
    assert [str(e.id) for e in final.events][2:4] == ["a1", "o1"], [str(e.id) for e in final.events]

    print("openhands keep binding selfcheck ok  (%d views put back, %d events, %d chars)"
          % (len(sel["items"]), len(sel["unforget"]), sel["chars"]))


if __name__ == "__main__":
    main()
