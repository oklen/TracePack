"""OpenHands events -> the rows ``tracepack.adapters.openhands`` normalizes.  No SDK import.

Split out of :mod:`tracepack.condenser.oh_condenser` for two reasons: it is the part that decides what
the retrieval corpus actually *is* (so it deserves its own contract tests, on python 3.9, with no
dependency), and it lets an archive written by a live run be replayed offline.

The conversion is byte-identical to ``pilot/oh_pilot.export_openhands_messages``, the exporter every
published round-2 read-out went through -- the deliverable must normalize its input the same way the
measurements did, or the numbers do not transfer.

Duck-typed on purpose: anything with ``model_dump()`` (or a plain dict) and the right class name
works, which is what makes it testable without the SDK installed.
"""
from __future__ import annotations

import json

#: The adapter's header row (``HEADER_KEY`` = ``tracepack_format``).
HEADER_ROW = {"tracepack_format": "openhands", "instance_id": "live", "repo": "live",
              "dataset": "live", "trajectory_id": "live"}


def flatten(x) -> str:
    """Flatten OpenHands content (str / {type:text,text} / list of those / anything) to text."""
    if x is None:
        return ""
    if isinstance(x, str):
        return x
    if isinstance(x, dict):
        for k in ("text", "content", "output", "command", "result"):
            if k in x:
                return flatten(x[k])
        return json.dumps(x, ensure_ascii=False)
    if isinstance(x, (list, tuple)):
        return "\n".join(flatten(i) for i in x)
    return str(x)


def _dump(e) -> dict:
    if hasattr(e, "model_dump"):
        return e.model_dump(mode="json")
    if isinstance(e, dict):
        return dict(e)
    return {}


def _kind(e) -> str:
    k = type(e).__name__
    return k if k != "dict" else str(e.get("_kind", ""))


def events_to_rows(events, keep_ids=None, start: int = 0) -> list:
    """``keep_ids`` restricts to those event ids; ``start`` keeps row ids unique across appends.

    Rows the adapter cannot use (the system prompt, condensation bookkeeping, our own injected
    packet) are skipped rather than passed through as empty text.
    """
    rows = []
    n = start
    for e in events:
        d = _dump(e)
        if keep_ids is not None and d.get("id") not in keep_ids:
            continue
        kind = _kind(e)
        mid = "m%06d" % n
        if kind == "MessageEvent":
            msg = d.get("llm_message") or {}
            rows.append({"id": mid, "role": msg.get("role", "user"),
                         "content": flatten(msg.get("content"))})
        elif kind == "ActionEvent":
            call = d.get("tool_call") or {}
            fn = (call.get("function") or {}) if isinstance(call, dict) else {}
            args = fn.get("arguments")
            if not isinstance(args, str):
                args = json.dumps(d.get("action") or {}, ensure_ascii=False)
            rows.append({"id": mid, "role": "assistant", "content": flatten(d.get("thought")),
                         "tool_calls": [{"id": d.get("tool_call_id") or mid, "type": "function",
                                         "function": {"name": d.get("tool_name") or "tool",
                                                      "arguments": args}}]})
        elif kind == "ObservationEvent":
            rows.append({"id": mid, "role": "tool", "content": flatten(d.get("observation"))})
        else:
            continue
        n += 1
    return rows


def last_user_text(events) -> "tuple[int, str]":
    """(index, text) of the newest user message in the view, or (-1, "").

    The newest user message is the instruction the packet is retrieved for.  Inserting the packet at
    that index (rather than appending) keeps it in place for the whole turn, however many tool steps
    the agent takes before it answers.
    """
    for i in range(len(events) - 1, -1, -1):
        e = events[i]
        if _kind(e) != "MessageEvent":
            continue
        d = _dump(e)
        if d.get("source") != "user":
            continue
        msg = d.get("llm_message") or {}
        return i, flatten(msg.get("content"))
    return -1, ""


def _selfcheck() -> None:
    class Ev:                                    # a stand-in for an SDK event: name + model_dump
        def __init__(self, kind, **d):
            self.__class__ = type(kind, (Ev,), {})
            self._d = dict(d)

        def model_dump(self, mode=None):
            return dict(self._d)

    msg = Ev("MessageEvent", id="a", source="user",
             llm_message={"role": "user", "content": [{"type": "text", "text": "what port"}]})
    act = Ev("ActionEvent", id="b", tool_name="bash", tool_call_id="tc1", thought="checking",
             tool_call={"function": {"arguments": '{"cmd": "ss -ltn"}'}})
    obs = Ev("ObservationEvent", id="c", observation={"output": "LISTEN 0 4096 *:8813"})
    sysm = Ev("SystemPromptEvent", id="d")

    rows = events_to_rows([msg, act, obs, sysm])
    assert [r["role"] for r in rows] == ["user", "assistant", "tool"], rows
    assert rows[0]["content"] == "what port"
    assert rows[1]["tool_calls"][0]["function"]["name"] == "bash"
    assert "8813" in rows[2]["content"]
    assert len(rows) == 3, "the system prompt must not become a row"

    # keep_ids is the whole point: the corpus is the FORGOTTEN events, nothing else
    assert [r["content"] for r in events_to_rows([msg, act, obs], keep_ids={"c"})][0].endswith("8813")
    assert events_to_rows([msg], keep_ids=set()) == []

    # ids stay unique across appends, or the adapter sees duplicate native_refs
    a = events_to_rows([msg, act, obs])
    b = events_to_rows([msg], start=len(a))
    assert len({r["id"] for r in a + b}) == 4

    i, q = last_user_text([act, msg, obs])
    assert (i, q) == (1, "what port"), (i, q)
    assert last_user_text([act, obs]) == (-1, "")
    # an agent message is not an instruction
    agent = Ev("MessageEvent", id="e", source="agent",
               llm_message={"role": "assistant", "content": "done"})
    assert last_user_text([msg, agent])[0] == 0

    print("oh_rows selfcheck ok")


if __name__ == "__main__":
    _selfcheck()
