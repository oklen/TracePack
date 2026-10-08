"""OpenHands binding for ContextWeaver, so it can be run as an arm of the live agent experiment.

    from tracepack.baselines.oh_contextweaver import ContextWeaverCondenser
    agent = agent.model_copy(update={"condenser": ContextWeaverCondenser(llm=llm, W=5, m=3)})

Faithful to Algorithm 1 in three respects that matter for the comparison:

1. **It re-weaves on every step, not once at a boundary.**  ContextWeaver is an online method: the
   anchor is the newest node, so the ancestry is recomputed as the agent acts.  A binding that wove
   once at the compaction boundary would be a strawman -- it would freeze the anchor before the next
   instruction exists.  The cost of doing it properly (one analyzer pass per new node) is not an
   implementation detail either; it is the number PLAN_baselines §5 asks for.
2. **Nothing is deleted.**  Non-ancestors keep their thought and action and lose their *observation*
   to a placeholder, exactly as the paper describes -- and because an event is not dropped, a step
   that enters the ancestry later comes back in full.  Expressed as a rewritten `View`, not as a
   `Condensation`, because a Condensation's forgetting is irreversible.
3. **Parent scoring is incremental and cached.**  A node's parents are decided once, when it first
   appears, over the candidates alive at that moment -- which is what the algorithm does.

A failure here RAISES.  It must not degrade to "return the view unchanged": that silently turns this
arm into a full-history arm, which is a *better* arm, and the resulting number would look like a
ContextWeaver result while being something else entirely.
"""
from __future__ import annotations

import json
import logging
import os
import threading

try:
    from openhands.sdk.context.condenser import LLMSummarizingCondenser
    from openhands.sdk.context.view import View
    from openhands.sdk.event.condenser import Condensation, CondensationSummaryEvent
    from openhands.sdk.llm import LLM
    from pydantic import Field, PrivateAttr
except ImportError as exc:                                         # pragma: no cover
    raise ImportError(
        "tracepack.baselines.oh_contextweaver needs `openhands-sdk` (python >= 3.12). "
        "The engine in tracepack.baselines.contextweaver has no such requirement."
    ) from exc

from . import contextweaver as CW

logger = logging.getLogger(__name__)

WOVEN_HEADER = (
    "[context] Earlier steps of this session. Steps in the current dependency ancestry are kept in "
    "full; for the others the observation was replaced, and only what the step did remains:\n\n"
)


class ContextWeaverCondenser(LLMSummarizingCondenser):
    """ContextWeaver (arXiv:2604.23069), faithful-as-described.  See the module docstring."""

    W: int = Field(default=5, gt=0, description="warmup length and ancestry cap (the paper's W=5)")
    m: int = Field(default=3, gt=0, description="max parents per node; the paper leaves it open")
    scorer: str = Field(default="pairwise", description="pairwise (faithful) | batched (a deviation)")
    analyzer: str = Field(default="local", description=(
        "which model scores dependencies: `local` uses the same endpoint as the executor (what every "
        "published W number ran with), `remote:<model>` routes the analyzer -- and only the analyzer -- "
        "to the OpenAI-compatible endpoint in `analyzer_url`.  RESULTS_swebench 3.7.8 measured a competent analyzer picking the needed "
        "parent 42% of the time against the 27B's 26%, with the anchor-only collapse falling from "
        "44% to 0%, which is why this switch exists"))
    analyzer_effort: str = Field(default="low", description="reasoning effort the remote analyzer ran at "
                                 "(recorded only; set it on the endpoint)")
    analyzer_url: str = Field(default_factory=lambda: os.environ.get("TRACEPACK_ANALYZER_URL", ""),
                              description=(
        "OpenAI-compatible base URL for a `remote:` analyzer; defaults to the TRACEPACK_ANALYZER_URL "
        "environment variable and has no built-in value"))
    validation: str = Field(default="legacy", description=(
        "failure classifier: `legacy` is the published bare-word match, kept so the W numbers "
        "reproduce; `anchored` requires the failure at the start of a line.  Legacy marked 11.9% of "
        "real nodes failed and 65% of those were successful operations whose echoed file content "
        "merely contained the word (RESULTS_swebench 3.7)"))
    supersede: str = Field(default="legacy", description=(
        "candidate filtering: `legacy` is what the published W numbers ran with and is kept so they "
        "reproduce; `write_path` is the corrected rule (a later file-editor WRITE supersedes an "
        "earlier WRITE to the same path).  Legacy also matched \"command\" with a capture that stops "
        "at the first escaped quote, which removed 43% of every history -- 62% of the steps a later "
        "step demonstrably depended on never reached the analyzer (RESULTS_swebench 3.7)"))
    workers: int = Field(default=8, gt=0, description="concurrency for the independent parent calls")
    summaries: bool = Field(default=True, description="node + dependency summaries (the LLMSUM half)")
    cw_base_url: str = Field(default="", description="analyzer endpoint; defaults to the agent LLM's")
    cw_model: str = Field(default="", description="analyzer model; defaults to the agent LLM's")
    log_path: str | None = Field(default=None, description="one line per weave: cost and ancestry")
    enabled: bool = Field(
        default=True,
        description="False = pass the view through untouched.  The pilot holds this False for "
                    "phase 1 and flips it at the compaction boundary, so every arm enters "
                    "phase 2 from the SAME pre-compaction state (PLAN_baselines §1).  "
                    "ContextWeaver deployed for real is active throughout; that is the "
                    "--cw-online sensitivity arm, reported separately.")

    _nodes: list = PrivateAttr(default_factory=list)
    _parents: dict = PrivateAttr(default_factory=dict)     # node key -> (parents, scores)
    _calls: int = PrivateAttr(default=0)
    _tokens: int = PrivateAttr(default=0)
    _ms: int = PrivateAttr(default=0)
    _weaves: int = PrivateAttr(default=0)
    _calls_seen: int = PrivateAttr(default=0)
    _hard_caps: int = PrivateAttr(default=0)
    #: analyzer replies that came back empty.  A remote endpoint can answer a failed call with an
    #: empty string rather than a 500 so one bad scoring call cannot end a two-hour cell; the
    #: price is that a WHOLLY broken analyzer looks exactly like the anchor-only collapse we
    #: are trying to measure.  This counter is what tells the two apart in the row.
    _empty: int = PrivateAttr(default=0)
    #: scoring calls that raised even after retries and were absorbed as a zero score
    _failed: int = PrivateAttr(default=0)
    _nodes_seen: int = PrivateAttr(default=0)
    _lock: object = PrivateAttr(default_factory=threading.Lock)

    # -- plumbing -------------------------------------------------------------------------------

    def _chat(self, cfg):
        """The analyzer's transport.  `local` is the executor's own endpoint.

        `remote:<model>` points the SAME client at another OpenAI-compatible endpoint
        (`analyzer_url`, or TRACEPACK_ANALYZER_URL; key from TRACEPACK_ANALYZER_API_KEY), so only
        the dependency analyzer changes model and the executor keeps its own.
        """
        if not self.analyzer.startswith("remote:"):
            return CW.Chat(cfg.base_url, cfg.model, cfg.api_key, cfg.timeout)
        if not self.analyzer_url:
            raise ValueError("analyzer=%r needs analyzer_url or TRACEPACK_ANALYZER_URL" % self.analyzer)
        alias = self.analyzer.split(":", 1)[1]
        return CW.Chat(self.analyzer_url, alias, os.environ.get("TRACEPACK_ANALYZER_API_KEY", "local"),
                       max(cfg.timeout, 300))

    @property
    def cfg(self) -> CW.CWConfig:
        base = self.cw_base_url or str(getattr(self.llm, "base_url", "") or "").rstrip("/")
        model = self.cw_model or str(getattr(self.llm, "model", "") or "").split("/")[-1]
        key = getattr(self.llm, "api_key", None)
        key = key.get_secret_value() if hasattr(key, "get_secret_value") else (key or "local")
        return CW.CWConfig(W=self.W, m=self.m, model=model, base_url=base, api_key=key,
                           scorer=self.scorer, max_workers=self.workers, summarize=self.summaries,
                           supersede=self.supersede, validation=self.validation)

    def stats(self) -> dict:
        return {"weaves": self._weaves, "llm_calls": self._calls, "llm_tokens": self._tokens,
                "graph_ms": self._ms, "n_nodes": self._nodes_seen,
                "condense_calls": self._calls_seen, "enabled": self.enabled,
                "hard_cap_hits": self._hard_caps,
                "W": self.W, "m": self.m, "scorer": self.scorer, "supersede": self.supersede,
                "validation": self.validation, "analyzer": self.analyzer,
                "analyzer_empty": self._empty, "analyzer_failed": self._failed}

    def _log(self, row: dict) -> None:
        if not self.log_path:
            return
        try:
            with open(self.log_path, "a", encoding="utf-8") as fh:
                fh.write(json.dumps(row, ensure_ascii=False) + "\n")
        except OSError:                                            # pragma: no cover
            pass                                                   # telemetry never breaks a run

    # -- node extraction from a live View --------------------------------------------------------

    def _sync_nodes(self, events) -> list:
        """Delegates to the SDK-free engine, which is where it can be tested."""
        return CW.nodes_from_events(events)

    def _assert_engaged(self, n_nodes: int, n_events: int) -> None:
        """Turn "quietly did nothing" into a hard failure.

        An enabled condenser that has been consulted many times over a view full of tool calls and
        has never found a node is not warming up -- it is broken, and every cell it touches passes
        because the agent still holds the whole history.  That is the most dangerous shape a bug in
        this file can take, so it raises.
        """
        self._calls_seen += 1
        if n_nodes:
            self._nodes_seen = max(self._nodes_seen, n_nodes)
        if (self._calls_seen >= 20 and not self._nodes_seen and n_events >= 20):
            raise RuntimeError(
                "ContextWeaverCondenser extracted 0 nodes from %d events after %d calls: the event "
                "shape does not match _pairs(), so this arm is silently running with the full "
                "history.  Refusing to continue -- a pass here would be attributed to ContextWeaver."
                % (n_events, self._calls_seen))

    # -- the weave ------------------------------------------------------------------------------

    def _weave_view(self, view: View) -> View:
        import time

        if not self.enabled:
            return view
        events = list(view.events)
        nodes = self._sync_nodes(events)
        self._assert_engaged(len(nodes), len(events))
        if len(nodes) <= self.W:                                   # warmup: keep everything
            return view
        t0 = time.time()
        cfg = self.cfg
        chat = self._chat(cfg)
        goal = CW.last_user_text(events)
        # parents are cached by the parent's EVENT IDS, not by node index: once the hard cap below
        # has forgotten nodes, indices shift and a cached index would point at the wrong step
        key_to_idx = {n.event_ids: n.idx for n in nodes}
        for n in nodes:
            key = n.event_ids
            if key in self._parents:
                pkeys, pscores = self._parents[key]
                n.parents = tuple(key_to_idx[k] for k in pkeys if k in key_to_idx)
                n.scores = {key_to_idx[k]: v for k, v in pscores.items() if k in key_to_idx}
                continue
            cands = [c for c in nodes[:n.idx] if c.validation not in ("failed", "superseded")]
            n.parents, n.scores = CW.select_parents(chat, n, cands, goal, cfg)
            self._parents[key] = (tuple(nodes[i].event_ids for i in n.parents),
                                  {nodes[i].event_ids: v for i, v in (n.scores or {}).items() if 0 <= i < len(nodes)})
        if cfg.summarize:
            CW.summarize(chat, nodes, cfg)
        anchor = len(nodes) - 1
        A = CW.ancestry(nodes, anchor, self.W)
        if getattr(view, "unhandled_condensation_request", False):
            # HARD CAP (PLAN_swebench §2.1, adaptation 2): the woven context no longer fits the
            # model window.  The paper does not say what happens here.  We forget the oldest half
            # of the NON-ancestors outright -- thought, action and all -- through a Condensation,
            # which is the only thing that clears the SDK's request.  Ancestors are never touched.
            # Every hit is counted; a run with zero hits behaved exactly as the paper describes.
            aset = set(A)
            non_anc = [n for n in nodes if n.idx not in aset]
            drop = non_anc[: max(1, len(non_anc) // 2)]
            ids = [eid for n in drop for eid in n.event_ids]
            with self._lock:
                self._hard_caps += 1
                self._calls += chat.calls
                self._empty += chat.empty
                self._failed += chat.failed
                self._tokens += chat.tokens
            self._log({"hard_cap": self._hard_caps, "n_nodes": len(nodes), "dropped_nodes": len(drop),
                       "dropped_events": len(ids), "ancestry": list(A), "ms": int(1000 * (time.time() - t0))})
            return Condensation(forgotten_event_ids=ids)
        keep_ids = set()
        for i in A:
            keep_ids.update(nodes[i].event_ids)
        node_ids = set()
        for n in nodes:
            node_ids.update(n.event_ids)

        woven = CW.weave(nodes, A, anchor)
        carrier = CondensationSummaryEvent(
            id="contextweaver-%d-%d" % (len(nodes), len(A)),
            summary=WOVEN_HEADER + woven, source="environment")
        out = []
        placed = False
        for e in events:
            eid = str(getattr(e, "id", ""))
            if eid in node_ids and eid not in keep_ids:
                if not placed:                                     # the carrier stands in for them
                    out.append(carrier)
                    placed = True
                continue
            out.append(e)
        if not placed:                                             # every node is an ancestor
            return view
        with self._lock:
            self._calls += chat.calls
            self._empty += chat.empty
            self._failed += chat.failed
            self._tokens += chat.tokens
            self._ms += int(1000 * (time.time() - t0))
            self._weaves += 1
        self._log({"weave": self._weaves, "n_nodes": len(nodes), "ancestry": list(A),
                   "kept_events": len(keep_ids), "woven_chars": len(woven),
                   "llm_calls": chat.calls, "llm_tokens": chat.tokens, "empty": chat.empty, "failed": chat.failed,
                   "ms": int(1000 * (time.time() - t0)), "goal_chars": len(goal)})
        return View(events=out, unhandled_condensation_request=False)

    # -- the two SDK entry points ---------------------------------------------------------------

    def condense(self, view: View, agent_llm: "LLM | None" = None):
        return self._weave_view(view)

    async def acondense(self, view: View, agent_llm: "LLM | None" = None):
        return self._weave_view(view)


# ---------------------------------------------------------------- helpers


def _flat(x) -> str:
    if x is None:
        return ""
    if isinstance(x, str):
        return x
    if isinstance(x, (list, tuple)):
        return "\n".join(_flat(i) for i in x)
    if isinstance(x, dict):
        for k in ("text", "content", "output", "command", "arguments", "result"):
            if k in x:
                return _flat(x[k])
        return json.dumps(x, ensure_ascii=False)[:4000]
    for k in ("text", "content", "output", "command", "result"):
        v = getattr(x, k, None)
        if v is not None:
            return _flat(v)
    fn = getattr(x, "function", None)
    if fn is not None:
        return "%s %s" % (getattr(fn, "name", ""), _flat(getattr(fn, "arguments", "")))
    return str(x)[:4000]


def _last_user_text(events) -> str:
    """The instruction in force right now -- read off the view, so it can never be a later one."""
    for e in reversed(events):
        if type(e).__name__ != "MessageEvent":
            continue
        d = e.model_dump(mode="json") if hasattr(e, "model_dump") else (dict(e) if isinstance(e, dict) else {})
        msg = d.get("llm_message") or {}
        if msg.get("role") == "user":
            t = _flat(msg.get("content"))
            if t.strip():
                return t
    return "continue the task"
