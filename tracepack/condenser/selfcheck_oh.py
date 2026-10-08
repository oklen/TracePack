"""Binding smoke test: run this wherever the SDK is installed (python >= 3.12).

    python3.12 -m tracepack.condenser.selfcheck_oh

The engine and the row builder are contract-tested on python 3.9 (`tests/test_condenser.py`), but the
part that plugs into the SDK -- archiving forgotten events, building a View with an extra event,
making sure that event converts to a user message -- can only be exercised where the SDK is.  It
needs no LLM and no GPU: `max_size` is set high enough that no condensation is ever requested, and
the archive is filled directly.

Every check here failed at least once while the binding was being written, or is the kind of thing
that fails silently in production (an injected event that converts to an empty message; a packet that
differs from what the offline engine would produce for the same archive).
"""
from __future__ import annotations

import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

from openhands.sdk.context.view import View                               # noqa: E402
from openhands.sdk.event import MessageEvent                              # noqa: E402
from openhands.sdk.event.base import LLMConvertibleEvent                  # noqa: E402
from openhands.sdk.llm import LLM, Message, TextContent                   # noqa: E402
from pydantic import SecretStr                                            # noqa: E402

from tracepack.adapters.openhands import OpenHandsAdapter                 # noqa: E402
from tracepack.condenser import TracePackRecall                           # noqa: E402
from tracepack.condenser.oh_condenser import TracePackCondenser              # noqa: E402

QUERY = "which job id did the status check use"


def msg(role, text, source=None):
    return MessageEvent(source=source or ("user" if role == "user" else "agent"),
                        llm_message=Message(role=role, content=[TextContent(text=text)]))


def main():
    llm = LLM(model="openai/never-called", base_url="http://127.0.0.1:1/v1",
              api_key=SecretStr("unused"), usage_id="selfcheck")
    cond = TracePackCondenser(llm=llm, max_size=4000, keep_first=2, budget=2048,
                              log_path=os.environ.get("TP_SELFCHECK_LOG", "") or None)

    # ---- 1. archiving keeps exactly the forgotten events ------------------------------------
    history = [msg("user", "deploy the service"),
               msg("assistant", "deploy finished, listening on port 8813"),
               msg("assistant", "the status check used job id 4417"),
               msg("user", QUERY)]
    forgotten = {history[1].id, history[2].id}
    n = cond._archive(history, forgotten)
    assert n == 2, "archived %d rows, expected the 2 forgotten events" % n
    rows = cond.archived_rows()
    assert rows[0].get("tracepack_format") == "openhands", rows[0]
    assert any("8813" in (r.get("content") or "") for r in rows[1:]), rows
    assert not any("deploy the service" in (r.get("content") or "") for r in rows[1:]), \
        "an event still in the view was archived"
    assert cond._archive(history, forgotten) == 0, "re-archiving the same ids duplicated the corpus"

    # ---- 2. injection puts a real user message in front of the instruction -------------------
    view = View(events=[history[0], history[3]])
    out = cond._inject(view)
    assert isinstance(out, View)
    assert len(out.events) == len(view.events) + 1, [type(e).__name__ for e in out.events]
    packet = out.events[1]
    assert isinstance(packet, LLMConvertibleEvent)
    m = packet.to_llm_message()
    assert m.role == "user", m.role
    text = "".join(c.text for c in m.content if isinstance(c, TextContent))
    assert "[tracepack]" in text and "8813" in text, text[:400]
    assert out.events[2] is history[3], "the packet must sit BEFORE the instruction"
    assert view.events == [history[0], history[3]], "the input view was mutated (it is read-only)"

    # ---- 3. the packet is what the offline engine produces for the same archive --------------
    graph = OpenHandsAdapter(link_values=True).normalize(cond.archived_rows())
    offline = TracePackRecall(cond.recipe).recall(graph, QUERY, budget=cond.budget)
    assert offline.text == text, "live packet != offline replay of the same archive"

    # ---- 4. the off switches are really off ---------------------------------------------------
    quiet = TracePackCondenser(llm=llm, max_size=4000, keep_first=2, enabled=False)
    quiet._archive(history, forgotten)
    assert len(quiet._inject(view).events) == len(view.events), "enabled=False still injected"

    empty = TracePackCondenser(llm=llm, max_size=4000, keep_first=2)
    assert len(empty._inject(view).events) == len(view.events), "injected with an empty archive"

    no_user = View(events=[msg("assistant", "thinking")])
    assert len(cond._inject(no_user).events) == 1, "injected with no user instruction in view"

    # ---- 5. condense() does not condense when it is not asked to, and still injects ----------
    result = cond.condense(view, agent_llm=llm)
    assert isinstance(result, View) and len(result.events) == 3, \
        "condense() on a small view should pass through + inject, not call the LLM"

    # ---- 6. deterministic, and cached per query ------------------------------------------------
    again = "".join(c.text for c in cond._inject(view).events[1].to_llm_message().content
                    if isinstance(c, TextContent))
    assert again == text, "two recalls for the same query differ"

    # ---- 7. a broken recall degrades to the stock condenser, it does not raise ---------------
    # The fault goes in by SUBCLASSING, not by patching the instance: the condenser is a pydantic
    # model and rejects attribute assignment ("object has no field"), which is how the first version
    # of this check failed on the worker.  Worth knowing for anyone wrapping it.
    class Boom(TracePackCondenser):
        def recall_for(self, query):
            raise RuntimeError("injected fault")

    boom = Boom(llm=llm, max_size=4000, keep_first=2)
    boom._archive(history, forgotten)
    assert len(boom._inject(view).events) == len(view.events), \
        "a failing recall must degrade to the plain view, never propagate"
    assert isinstance(boom.condense(view, agent_llm=llm), View), \
        "a failing recall must not propagate out of condense() either"

    print("openhands binding selfcheck ok  (packet %d tokens, %d entries, gate=%s)"
          % (offline.tokens, len(offline.event_ids), offline.gate_open))


if __name__ == "__main__":
    main()
