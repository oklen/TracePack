"""tracepack.adapters.llama_index_memory -- serve a TracePack packet as a LlamaIndex memory block.

This is the *output* side of the control plane: `adapters/claude_code.py` turns a real trace into
the IR, and this turns the IR into something a real agent framework can consume.  It is the
reference integration the proposal's §7 asks for, aimed at
`run-llama/llama_index` issue #22823 (runtime-conditioned Memory retrieval).

Two things make it worth doing rather than a formality:

1. **Runtime conditioning already works, and this proves it.**  ``Memory.aget(input, **kwargs)``
   forwards ``block_kwargs`` to every block's ``_aget``, so a block CAN select on the live query,
   phase and agent state without touching the Workflow-Context -> Memory call chain.  That is the
   open question in #22823, and the thin upstream PR is meant to document + regression-test it,
   not to change the API.

2. **TracePack can give LlamaIndex a truncation it does not have.**  LlamaIndex truncates a block
   by calling ``atruncate``, whose default is to delete the block's whole content; the fallback
   loop then pops entire blocks.  TracePack's assembler instead re-selects under a smaller hard
   budget, never splits an atomic tool_call/tool_result pair, never cuts an event in half, and
   reports what it had to leave out.  So ``atruncate`` here is "re-assemble smaller", and the
   block degrades by dropping the least-relevant EVENTS instead of vanishing.

Import policy: ``llama_index`` is imported lazily, inside :func:`make_memory_block`.  The retrieval
logic lives in :class:`TracePackRetriever`, which has no third-party dependency at all, so the
123-test contract suite still runs in an environment without LlamaIndex (and on Python 3.9, which
llama-index-core no longer supports).
"""
from __future__ import annotations

import os
import sys
from dataclasses import dataclass, field
from typing import Any, Optional, Sequence

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

from tracepack.core.assembler import AssemblerConfig, BudgetAssembler
from tracepack.core.closure import ClosureConfig, TypedClosure
from tracepack.core.router import RouterConfig, make_router
from tracepack.core.schema import Seed

QUERY_MODES = ("lookup", "why", "state", "audit")


class AssemblerPinDropped(RuntimeError):
    """A pinned event did not fit the budget.  Raised rather than returned: a caller that pinned
    an event is asserting it is load-bearing, and quietly returning a packet without it is the
    silent-corruption case the pinning contract exists to prevent."""


@dataclass
class PackedResult:
    """What one retrieval produced -- text plus the accounting the caller may want to log."""

    text: str
    tokens: int
    n_entries: int
    incomplete: bool
    missing_required: tuple
    digest: str
    served: tuple = ()          # event ids actually in the packet, so a caller can verify pins


@dataclass
class TracePackRetriever:
    """Framework-free core: query (+ runtime state) -> packet text under a hard budget.

    Keeping this separate from the LlamaIndex class is not tidiness.  It means the behaviour under
    test is the behaviour that ships, and that a framework upgrade cannot silently change what the
    packet contains.
    """

    graph: Any
    router: str = "hybrid_pin"
    closure_mode: str = "native"
    budget: int = 2048
    k: int = 8
    repr_policy: str = "source_only"
    default_query_mode: str = "lookup"
    _router_cfg: RouterConfig = field(init=False, repr=False)

    def __post_init__(self):
        if self.graph is None or not hasattr(self.graph, "event"):
            raise ValueError("TracePackRetriever needs a TraceGraph")
        if not isinstance(self.budget, int) or isinstance(self.budget, bool) or self.budget <= 0:
            raise ValueError("budget must be a positive int, got %r" % (self.budget,))
        if self.default_query_mode not in QUERY_MODES:
            raise ValueError("query_mode must be one of %s" % (QUERY_MODES,))
        self._router_cfg = RouterConfig(k=self.k)

    def retrieve(self, query: str, *, budget: Optional[int] = None,
                 query_mode: Optional[str] = None, k: Optional[int] = None,
                 pin: Sequence[str] = ()) -> PackedResult:
        """One packet.  ``pin`` names event ids that must not be displaced (the §4.4 pinning
        contract) -- this is how a caller passes agent state such as "the step we are resuming"."""
        if not isinstance(query, str) or not query.strip():
            raise ValueError("query must be a non-empty string")
        mode = query_mode or self.default_query_mode
        if mode not in QUERY_MODES:
            raise ValueError("unknown query_mode %r" % (mode,))
        # `budget or self.budget` would make budget=0 silently fall back to the default -- a
        # caller asking for zero room would get 2048 instead of an error.  Same for k.
        b = int(self.budget if budget is None else budget)
        if b <= 0:
            raise ValueError("budget must be a positive int, got %r" % (budget,))
        if k is not None:
            if not isinstance(k, int) or isinstance(k, bool) or k <= 0:
                raise ValueError("k must be a positive int, got %r" % (k,))
            cfg = RouterConfig(k=k)
        else:
            cfg = self._router_cfg

        seeds = list(make_router(self.router, cfg).retrieve(query, self.graph, cfg.k))
        if pin:
            # The `hybrid_pin` router derives its pins by PARSING the query text (§4.4).  An
            # explicit id list is a different thing -- it is how a caller passes agent state
            # ("the step being resumed") that no wording would name -- so it is merged here, in
            # front, with score 1.0 and pinned=True.  Contract #5: a pin is never displaced.
            have = {x.event_id for x in seeds}
            extra = [Seed(event_id=e, score=1.0, source="pin", rank=0, pinned=True)
                     for e in pin if self.graph.has(e) and e not in have]
            if extra:
                seeds = [Seed(event_id=x.event_id, score=x.score, source=x.source,
                              rank=i, pinned=x.pinned)
                         for i, x in enumerate(extra + seeds)]
        closure = TypedClosure(ClosureConfig(mode=self.closure_mode)).close(
            query, seeds, self.graph, query_mode=mode)
        packet = BudgetAssembler(AssemblerConfig(repr_policy=self.repr_policy)).assemble(
            query, closure, self.graph, b, seeds=seeds, query_mode=mode)
        m = packet.manifest
        served = tuple(e.event_id for e in m.entries)
        if pin:
            # A pin that the assembler could not serve must be visible, not swallowed: contract #5
            # says a pin is never displaced by ordinary results, and contract #4 says an
            # unsatisfiable packet declares what is missing.
            dropped = [e for e in pin if self.graph.has(e) and e not in served]
            if dropped:
                raise AssemblerPinDropped(
                    "pinned events %s did not fit budget=%d; raise the budget or drop the pin "
                    "(silently serving a packet without them would break contract #5)"
                    % (dropped, b))
        return PackedResult(text=packet.context, tokens=m.total_tokens,
                            n_entries=len(m.entries), incomplete=bool(m.incomplete),
                            missing_required=tuple(m.missing_required), digest=m.digest()[:16],
                            served=served)


def _query_from_messages(messages) -> str:
    """Last user-authored text, which is what a memory block is expected to condition on."""
    if not messages:
        return ""
    for msg in reversed(list(messages)):
        role = str(getattr(msg, "role", "") or "")
        if role and "user" not in role.lower():
            continue
        content = getattr(msg, "content", None)
        if isinstance(content, str) and content.strip():
            return content
        blocks = getattr(msg, "blocks", None) or ()
        text = " ".join(str(getattr(b, "text", "")) for b in blocks).strip()
        if text:
            return text
    return ""


def make_memory_block(retriever: TracePackRetriever, *, name: str = "tracepack",
                      priority: int = 1, description: Optional[str] = None):
    """Build a ``TracePackMemoryBlock`` bound to ``retriever``.

    A factory rather than a module-level class: ``BaseMemoryBlock`` is a pydantic model, so
    subclassing it at import time would make ``llama_index`` a hard dependency of this package.
    """
    try:
        from llama_index.core.memory import BaseMemoryBlock
    except ImportError as exc:                                   # pragma: no cover
        raise ImportError(
            "llama-index-core is required for the memory-block integration "
            "(`pip install llama-index-core`); the retrieval logic in TracePackRetriever "
            "works without it") from exc

    # State lives in a closure, not on the model: BaseMemoryBlock is a pydantic model and
    # assigning undeclared attributes to one is either rejected or silently ignored depending on
    # the pydantic config -- neither is something to build on.
    state = {"last_result": None, "last_query": ""}

    class TracePackMemoryBlock(BaseMemoryBlock[str]):
        """A memory block backed by an execution trace, selected per call.

        ``_aget`` honours these runtime kwargs, all optional -- they are exactly the
        "runtime-conditioned retrieval" of issue #22823, and they arrive through the existing
        ``Memory.aget(input, **block_kwargs)`` path with no core change:

            query        override the query (default: last user message)
            query_mode   lookup | why | state | audit
            budget       token budget for THIS call
            k            retrieval breadth for THIS call
            pin          event ids that must be served
        """

        model_config = {"arbitrary_types_allowed": True}

        async def _aget(self, messages=None, **block_kwargs: Any) -> str:
            query = str(block_kwargs.get("query") or _query_from_messages(messages) or "").strip()
            if not query:
                return ""
            res = retriever.retrieve(
                query,
                budget=block_kwargs.get("budget"),
                query_mode=block_kwargs.get("query_mode"),
                k=block_kwargs.get("k"),
                pin=tuple(block_kwargs.get("pin") or ()),
            )
            state["last_result"], state["last_query"] = res, query
            return res.text

        async def _aput(self, messages) -> None:
            """No-op by design.

            The trace is the source of truth and is written by the agent runtime, not by this
            block.  Accepting chat turns here would create a second, divergent copy of history --
            the failure this whole project exists to avoid.
            """
            return None

        async def atruncate(self, content: str, tokens_to_truncate: int) -> Optional[str]:
            """Re-assemble under a smaller budget instead of deleting the block.

            LlamaIndex's default drops the entire content, and its fallback loop pops whole
            blocks.  Here the budget assembler drops the least-relevant EVENTS instead, still
            never splitting an atomic tool_call/tool_result group and still reporting what it had
            to leave out.  If nothing fits, we return "" rather than a half event.
            """
            if not content:
                return content
            if not isinstance(tokens_to_truncate, int) or tokens_to_truncate <= 0:
                return content
            last = state["last_result"]
            current = last.tokens if last else max(1, len(content) // 4)
            new_budget = current - tokens_to_truncate
            if new_budget <= 0:
                return ""
            query = state["last_query"]
            if not query:
                # no query to re-run: fall back to the framework's own semantics rather than
                # inventing a truncation that could cut an event in half
                return ""
            return retriever.retrieve(query, budget=new_budget).text

    return TracePackMemoryBlock(
        name=name, priority=priority,
        description=description or ("TracePack: evidence selected from the agent's own "
                                    "execution trace"))
