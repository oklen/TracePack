"""TracePack -- turn an agent's execution trace into a budgeted, auditable evidence packet.

The core path has **no third-party dependencies**: it is stdlib only, so it can be dropped into an
agent loop without pulling in a model stack.  A model is needed only by the optional integrations
(the OpenHands condenser, the ContextWeaver port), and those are imported lazily.

Quickstart
----------

    import tracepack

    trace = [                                   # OpenAI-style rows, or a path to a .jsonl export
        {"tracepack_format": "openhands"},
        {"id": "m0", "role": "user", "content": "the tests fail with a TypeError in units"},
        {"id": "m1", "role": "assistant", "content": "look",
         "tool_calls": [{"id": "c1", "type": "function",
                         "function": {"name": "bash", "arguments": '{"command": "pytest -q"}'}}]},
        {"id": "m2", "role": "tool", "content": "E   TypeError: unsupported operand for Quantity"},
    ]
    pkt = tracepack.pack(trace, query="why does the Quantity comparison fail?", budget=2048)
    print(pkt["text"])          # the packet, ready to append to a summary
    print(pkt["tokens"], pkt["n_entries"], pkt["incomplete"])

`incomplete` is the honest part: it says the budget could not hold everything the selection asked
for.  A packet that silently drops material is the failure mode this package exists to avoid.

What the numbers say
--------------------
Read the findings in the README before building on this.  On SWE-bench Verified with a 27B executor the
packet did **not** produce a detectable gain in resolve rate over plain summarisation, and §3.8
measures why that is hard to see at all: two configuration-identical systems disagree on 11.7% of
instances, which is larger than every effect measured.  The component is useful where a fact cannot
be re-read; it is not a general accuracy win, and this package does not claim one.
"""
from __future__ import annotations

__version__ = "0.1.0"
__all__ = ["pack", "make_condenser", "__version__"]

_LINK_VALUES = True          # every published number ran with these two on
_EVIDENCE_HOPS = 2


def _adapter_for(trace):
    """Pick the adapter without reading the disk twice, and without guessing for in-memory rows."""
    from tracepack.adapters.openhands import OpenHandsAdapter
    from tracepack.adapters.pi import PiAdapter
    fmt = None
    if isinstance(trace, (list, tuple)) and trace and isinstance(trace[0], dict):
        fmt = trace[0].get("tracepack_format")
    else:
        import json
        from tracepack.adapters.base import open_trace
        try:
            with open_trace(trace) as fh:
                fmt = (json.loads(fh.readline() or "{}") or {}).get("tracepack_format")
        except Exception:
            fmt = None
    return OpenHandsAdapter(link_values=_LINK_VALUES) if fmt == "openhands" else PiAdapter(
        link_values=_LINK_VALUES)


def pack(trace, query: str, budget: int = 2048, k: int = 8, selection: str = "closure",
         assembly: str = "evidence_first") -> dict:
    """Build one evidence packet from a trace.

    `trace` is a path to a .jsonl export or an in-memory sequence of rows.  `budget` is a hard token
    ceiling: the assembler never exceeds it, and reports `incomplete` when it had to stop short.

    `selection` / `assembly` default to the published arm C configuration.  The names and every other
    combination are documented in `tracepack/pilot/cw_arms.py`; two of them ("closure_h",
    "evidence_first_x") switch on settings this project measured and adopted -- hybrid seeding and
    the excerpt tier -- which the published diagonal deliberately pins off so its numbers reproduce.
    """
    from tracepack.pilot import cw_arms, tp_serve
    from tracepack.core.assembler import BudgetAssembler
    tp_serve.LINK_VALUES = _LINK_VALUES
    tp_serve.EVIDENCE_HOPS = _EVIDENCE_HOPS
    full = _adapter_for(trace).normalize(trace)
    graph, excluded, _ = tp_serve.universe(full, query)
    seeds, closure = cw_arms.select(selection, query, graph, k)
    packet = BudgetAssembler(cw_arms.assembler_config(assembly)).assemble(
        query, closure, graph, budget, seeds=seeds, query_mode="lookup")
    m = packet.manifest
    return {"text": packet.context, "tokens": m.total_tokens, "n_entries": len(m.entries),
            "served": [e.event_id for e in m.entries], "incomplete": bool(m.incomplete),
            "n_events": len(graph.events), "n_seeds": len(seeds),
            "n_in_context_excluded": len(excluded),
            "selection": selection, "assembly": assembly, "budget": budget}


def make_condenser(llm, max_tokens: int, arm: str = "C", budget: int = 2048, k: int = 8,
                   keep_first: int = 2, query: str = "", archive_path: str = "",
                   log_path: str = ""):
    """The OpenHands integration: the stock summarising condenser with a packet appended.

    Needs `openhands-sdk` (python >= 3.12); everything above does not.  `arm` selects the retrieval
    configuration -- "C" is the published one, "X" the adopted recipe (hybrid seeding + excerpt
    tier).  The README summarises what each was measured to do.
    """
    from tracepack.baselines.oh_compaction_retrieval import CompactionRetrievalCondenser
    return CompactionRetrievalCondenser(
        llm=llm, max_tokens=max_tokens, keep_first=keep_first, max_size=10 ** 6,
        arm=arm, query=query, budget=budget, k=k,
        archive_path=archive_path, log_path=log_path)
