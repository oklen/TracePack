"""tracepack.core.closure -- typed dependency closure (proposal §3.2, §3.3, §4.5).

This module turns a *seed set* (whatever the router retrieved) into an
:class:`~tracepack.core.schema.EvidenceClosure`: the set of events that must be served
(``required``), the set that is nice-to-have and dropped first under budget (``optional``), and a
per-expansion audit trail (``steps``) so the manifest can name *why* every event is in the packet
(§3.5).

Design decisions other modules depend on
========================================

1. **"Minimal" is policy-relative, never semantic (§3.3).**  The closure claims only: under the
   declared edge whitelist, query mode and carrier policy, there is no superfluous *required*
   dependency.  Every decision is a named rule string in ``ClosureStep.rule`` -- see :data:`RULES`
   -- so an auditor can replay the policy instead of trusting the output.

2. **Edge direction.**  The IR points from the current event to what it rests on
   (``decision --DEPENDS_ON--> source``), so closure walks *out-edges* to collect parents.  The
   only rule that walks *in-edges* is ``state_prefers_latest``, because superseding runs the other
   way (``new --SUPERSEDES--> old``): to find the successor of an old state you must look at who
   points at it.

3. **WEAK edges are never followed.**  ``CONTROL`` / ``TEMPORAL`` are execution ancestry, not
   evidence dependency (§3.2).  This is enforced in one place (``_classify_*`` returns ``skip``)
   and asserted in ``_selfcheck``.  "Execution ancestry != evidence dependency" is the whole point
   of the typed-edge design, so it must be impossible to leak in by accident.

4. **``REQUIRED_EDGES`` is the *unconditional* whitelist; ``MATERIALIZES`` is a *gated* traversal.**
   The frozen schema deliberately keeps ``MATERIALIZES`` out of ``REQUIRED_EDGES`` ("a carrier
   pointing at its source does not make the source optional").  Reconciled here as: native mode
   follows ``RESULT_OF`` / ``GROUNDED_IN`` / ``DEPENDS_ON`` unconditionally, and follows a
   carrier's ``MATERIALIZES`` edge *only through the §4.5 verification gate* -- which by default
   means the source is pulled in (source-first, §3.1) and is released only when a matching
   ``CarrierVerification`` licenses it.  A carrier can therefore never silently remove a source
   (test contract #10).  ``SUPERSEDES`` is likewise never followed by the generic expansion; it is
   reachable only through the ``state_prefers_latest`` rule.

5. **Optionality is inherited, and computed in a second wave.**  The required wave runs to a fixed
   point first; only then is the optional frontier drained.  Anything discovered *below* an
   optional node is optional too (you cannot be required if the only thing that needs you is
   optional).  Because the required wave completes first, a node can never be "promoted" after the
   fact, which is what keeps ``required``/``optional`` disjoint and the traversal order stable.

6. **Nothing is silently dropped.**  Demoted states (``state_prefers_latest``) and relaxed carrier
   sources (§4.5) land in ``optional``, never in the void: audits need to see them, and the
   assembler is free to include them when budget allows.  ``relaxations`` records only stops that
   *actually took effect* (the source did not end up required through some other path), so the
   field never over-claims.

7. **Determinism (test contract #1).**  Every collection this module iterates is sorted before use
   (out-edges by ``(edge_type, dst_id, predicate)``, seeds by chronological index).  ``required``
   and ``optional`` are emitted in ``graph.chronological`` order.  No wall clock, no RNG, no
   set-iteration order escapes into the output.

8. **The graph object is duck-typed.**  ``core/graph.py`` is a sibling module; this file only
   requires that the object expose (a) its events (``events`` / ``event_map`` / ``by_id`` /
   ``nodes``, as a mapping or an iterable of ``TraceEvent``), (b) its edges (``edges`` /
   ``all_edges`` / ``edge_list``), and optionally (c) ``chronological`` (property or method,
   yielding ``TraceEvent`` objects or bare ids).  A missing ``chronological`` falls back to
   ``(timestamp, event_id)``.  Anything else raises ``SchemaError`` with a message naming the
   contract.

Modes (§6.2 baselines 6/7/8/9)
==============================
``off``            seeds only -- the "no closure" arm.
``native``         the real policy: typed whitelist + query-mode rules + §4.5 carrier gate.
``full_ancestor``  follow *all* STRONG edges transitively -- the deliberately over-expanding arm.
``oracle``         seeds + a caller-supplied gold dependency list -- the ceiling arm.

Implements: §3.1 (source-first), §3.2 (typed edges, strong/weak split), §3.3 (policy-minimal
closure and its four example rules), §3.5 (auditability), §4.5 (counterfactually verified
relaxation).  Pure stdlib, deterministic, no network / LLM / torch.
"""
from __future__ import annotations

import re
from collections import deque
from dataclasses import dataclass
from typing import Any, Callable, Iterable, Mapping, Sequence

from .excerpt import query_terms   # gold-blind query-term overlap, shared with the excerpt (carrier_cap)

try:  # normal package import
    from tracepack.core.schema import (
        QUERY_MODES,
        REQUIRED_EDGES,
        STRONG_EDGES,
        WEAK_EDGES,
        CarrierVerification,
        ClosureStep,
        EvidenceClosure,
        SchemaError,
        Seed,
        TraceEdge,
        TraceEvent,
    )
except ImportError:  # pragma: no cover - direct `python3 tracepack/core/closure.py`
    import os as _os
    import sys as _sys

    _sys.path.insert(
        0, _os.path.dirname(_os.path.dirname(_os.path.dirname(_os.path.abspath(__file__))))
    )
    from tracepack.core.schema import (  # noqa: E402
        QUERY_MODES,
        REQUIRED_EDGES,
        STRONG_EDGES,
        WEAK_EDGES,
        CarrierVerification,
        ClosureStep,
        EvidenceClosure,
        SchemaError,
        Seed,
        TraceEdge,
        TraceEvent,
    )

__all__ = [
    "CLOSURE_MODES",
    "RULES",
    "ClosureConfig",
    "TypedClosure",
    "is_exact_payload_query",
    "explain_exact_payload_query",
]

#: closure policies == the closure factor of the 3-factor experiment (§6.3)
CLOSURE_MODES = ("off", "native", "full_ancestor", "oracle", "all")

# ---------------------------------------------------------------- rule names
# Every ClosureStep.rule is one of these.  They are part of this module's public contract:
# manifest.py renders them as inclusion reasons and the profiler groups failures by them.

RULE_NATIVE_REQUIRED = "native_required_edge"
RULE_TOOL_RESULT = "tool_result_needs_call"
RULE_WHY_SOURCES = "why_needs_sources"
RULE_STATE_LATEST = "state_prefers_latest"
RULE_EXACT_PAYLOAD = "exact_payload_no_carrier_stop"
RULE_VERIFIED_CARRIER = "verified_carrier_or_witness"
RULE_UNVERIFIED_CARRIER = "unverified_carrier_needs_source"
RULE_FULL_ANCESTOR = "full_ancestor_all_strong"
RULE_ORACLE = "oracle_gold_dependency"
RULE_TRUNCATED = "max_hops_truncated"
RULE_CYCLE = "cycle_guard"
RULE_CARRIER_CAPPED = "carrier_fanout_capped"   # ClosureConfig.carrier_cap: ranked out, kept optional

RULES = (
    RULE_NATIVE_REQUIRED,
    RULE_TOOL_RESULT,
    RULE_WHY_SOURCES,
    RULE_STATE_LATEST,
    RULE_EXACT_PAYLOAD,
    RULE_VERIFIED_CARRIER,
    RULE_UNVERIFIED_CARRIER,
    RULE_FULL_ANCESTOR,
    RULE_ORACLE,
    RULE_TRUNCATED,
    RULE_CYCLE,
    RULE_CARRIER_CAPPED,
)

# ---------------------------------------------------------------- exact-payload detector
#
# §3.3: "an exact-payload query must not be stopped by a carrier that only stores a predicate."
# Deciding *what counts as an exact payload* has to be deterministic and inspectable -- an LLM
# classifier here would make the closure non-reproducible, which breaks test contract #1.
#
# The detector is deliberately FAIL-CLOSED: a false positive costs tokens (the source is loaded
# when a verified carrier would have done), a false negative costs a wrong answer (a summary is
# served where a literal number/path/hash was asked for).  Asymmetric costs => over-trigger.
#
# Six signals, each reported by name so the decision is auditable:
#   keyword  -- an English or Chinese word that asks for a literal payload
#   url      -- an http(s)/s3/hdfs locator
#   path     -- a filesystem-looking path (./x, /a/b, C:\x)
#   hash     -- a >=7-char hex run (git sha, md5, sha256, content hash)
#   quoted   -- a quoted span, i.e. the user is asking about verbatim text
#   digit    -- the query itself contains a digit (identifiers, amounts, versions, dates)

_EXACT_EN_WORDS = (
    "exact", "exactly", "verbatim", "literal", "literally", "precise", "precisely",
    "quote", "quotes", "quoted", "quotation", "word for word",
    "number", "numbers", "numeric", "digit", "digits", "value", "values", "amount",
    "figure", "total", "sum", "count", "how many", "how much", "price", "cost",
    "path", "paths", "filename", "filepath", "directory", "url", "uri", "link",
    "hash", "sha", "sha1", "sha256", "md5", "checksum", "digest", "fingerprint",
    "uuid", "guid", "id", "ids", "identifier", "serial", "code", "version",
    "timestamp", "date", "line number", "offset",
)
_EXACT_ZH_WORDS = (
    "精确", "准确", "确切", "原文", "原话", "逐字", "引用", "具体", "数字", "数值",
    "多少", "几个", "几次", "金额", "价格", "路径", "文件名", "目录", "链接",
    "哈希", "校验", "摘要值", "版本号", "编号", "序号", "时间戳", "行号", "总共", "总额",
)

_EXACT_EN_RE = re.compile(r"\b(?:" + "|".join(re.escape(w) for w in _EXACT_EN_WORDS) + r")\b",
                          re.IGNORECASE)
_HASH_RE = re.compile(r"\b[0-9a-fA-F]{7,}\b")
_PATH_RE = re.compile(r"(?:^|\s)(?:\.{0,2}/[\w.\-]|[A-Za-z]:\\)")
_URL_RE = re.compile(r"\b(?:https?|s3|hdfs|ftp)://", re.IGNORECASE)
_QUOTE_RE = re.compile("[\"'\u201c\u2018\u300c\u300e](.{2,}?)[\"'\u201d\u2019\u300d\u300f]")
_DIGIT_RE = re.compile(r"\d")


def explain_exact_payload_query(query: str) -> str | None:
    """Return the *name* of the first signal that marks ``query`` as an exact-payload request.

    Returns ``None`` when no signal fires.  Exposed (rather than folded into the boolean) so the
    profiler can report *why* a relaxation was refused instead of just that it was.
    """
    if not isinstance(query, str):
        raise SchemaError("query must be a str, got %r" % (type(query).__name__,))
    m = _EXACT_EN_RE.search(query)
    if m:
        return "keyword:%s" % m.group(0).lower()
    for w in _EXACT_ZH_WORDS:
        if w in query:
            return "keyword:%s" % w
    if _URL_RE.search(query):
        return "url"
    if _PATH_RE.search(query):
        return "path"
    if _HASH_RE.search(query):
        return "hash"
    if _QUOTE_RE.search(query):
        return "quoted"
    if _DIGIT_RE.search(query):
        return "digit"
    return None


def is_exact_payload_query(query: str) -> bool:
    """True when the query asks for a literal payload (numbers / paths / hashes / verbatim text)."""
    return explain_exact_payload_query(query) is not None


# ---------------------------------------------------------------- graph view (duck-typing)

_EVENT_ATTRS = ("events", "event_map", "by_id", "nodes")
_EDGE_ATTRS = ("edges", "all_edges", "edge_list")


def _fetch(graph: Any, names: Sequence[str]) -> Any:
    """Return the first present attribute among ``names``, calling it when it is a zero-arg method."""
    for name in names:
        if not hasattr(graph, name):
            continue
        value = getattr(graph, name)
        if callable(value):
            try:
                value = value()
            except TypeError:  # takes arguments -> not the accessor we want
                continue
        if value is not None:
            return value
    return None


class _GraphView:
    """Normalised, read-only projection of whatever ``core/graph.py`` hands us.

    Built once per :meth:`TypedClosure.close` call.  Deliberately does *not* re-run
    ``validate_events_edges``: graph.py owns that invariant.  Dangling edge targets are caught
    lazily (and loudly) at the moment closure would have followed them.
    """

    __slots__ = ("events", "edges", "chrono", "chrono_index", "out_edges", "in_edges")

    def __init__(self, events: Mapping[str, TraceEvent], edges: Sequence[TraceEdge],
                 chrono: Sequence[str]):
        self.events = events
        self.edges = tuple(edges)
        self.chrono = tuple(chrono)
        self.chrono_index = {eid: i for i, eid in enumerate(self.chrono)}
        out: dict[str, list[TraceEdge]] = {}
        inc: dict[str, list[TraceEdge]] = {}
        for e in self.edges:
            out.setdefault(e.src_id, []).append(e)
            inc.setdefault(e.dst_id, []).append(e)

        def key_out(e):
            return (e.edge_type, e.dst_id, e.predicate or "")

        def key_in(e):
            return (e.edge_type, e.src_id, e.predicate or "")

        self.out_edges = {k: tuple(sorted(v, key=key_out)) for k, v in out.items()}
        self.in_edges = {k: tuple(sorted(v, key=key_in)) for k, v in inc.items()}

    # -- construction ------------------------------------------------
    @classmethod
    def of(cls, graph: Any) -> "_GraphView":
        if graph is None:
            raise SchemaError("closure needs a graph, got None")
        raw_events = _fetch(graph, _EVENT_ATTRS)
        if raw_events is None:
            raise SchemaError(
                "graph object exposes none of %s; closure needs its events" % (_EVENT_ATTRS,))
        if isinstance(raw_events, Mapping):
            items: Iterable[Any] = list(raw_events.values())
        else:
            try:
                items = list(raw_events)
            except TypeError:
                raise SchemaError(
                    "graph events are not iterable: %r" % (type(raw_events).__name__,))
        events: dict[str, TraceEvent] = {}
        for ev in items:
            eid = getattr(ev, "event_id", None)
            if eid is None:
                raise SchemaError("graph events must be TraceEvent-like (missing .event_id)")
            events[eid] = ev
        if not events:
            raise SchemaError("graph has no events")

        raw_edges = _fetch(graph, _EDGE_ATTRS)
        if raw_edges is None:
            raise SchemaError(
                "graph object exposes none of %s; closure needs its edges" % (_EDGE_ATTRS,))
        edges = tuple(raw_edges)
        for ed in edges:
            if not hasattr(ed, "edge_type") or not hasattr(ed, "src_id"):
                raise SchemaError("graph edges must be TraceEdge-like (missing .edge_type/.src_id)")

        return cls(events, edges, cls._chronological(graph, events))

    @staticmethod
    def _chronological(graph: Any, events: Mapping[str, TraceEvent]) -> tuple[str, ...]:
        raw = _fetch(graph, ("chronological",))
        order: list[str] = []
        placed: set[str] = set()
        if raw is not None:
            for item in raw:
                eid = item if isinstance(item, str) else getattr(item, "event_id", None)
                if eid is None:
                    raise SchemaError("graph.chronological must yield event ids or TraceEvents")
                if eid in events and eid not in placed:
                    order.append(eid)
                    placed.add(eid)
        # graph.chronological is authoritative; anything it omits is appended by the key it would
        # have used, so the output order stays total and deterministic instead of raising on a
        # sibling module's bug.
        missing = sorted((eid for eid in events if eid not in placed),
                         key=lambda i: (events[i].timestamp, i))
        order.extend(missing)
        return tuple(order)

    # -- accessors ---------------------------------------------------
    def kind_of(self, event_id: str) -> str:
        ev = self.events.get(event_id)
        return getattr(ev, "kind", "") if ev is not None else ""

    def out(self, event_id: str) -> tuple[TraceEdge, ...]:
        return self.out_edges.get(event_id, ())

    def incoming(self, event_id: str) -> tuple[TraceEdge, ...]:
        return self.in_edges.get(event_id, ())

    def order(self, ids: Iterable[str]) -> tuple[str, ...]:
        n = len(self.chrono_index)
        return tuple(sorted(set(ids), key=lambda i: (self.chrono_index.get(i, n), i)))


# ---------------------------------------------------------------- config


@dataclass(frozen=True)
class ClosureConfig:
    """Closure policy knobs.  ``mode`` is the experiment factor (§6.3); the other two are guards.

    ``follow_optional`` means *traverse*, not *record*: when it is False, optional parents are
    still written to ``closure.optional`` (the audit trail must not lie about what the policy
    saw) but the closure does not expand through them.
    """

    mode: str = "native"
    max_hops: int = 6
    follow_optional: bool = True
    #: Cap on how many MATERIALIZES sources a carrier (compaction summary) may pull in as
    #: required, ranked by query-term overlap with the source text (gold-blind: it sees the
    #: query and event text only).  0 = unlimited, the frozen behaviour.  Measured 2026-09-05
    #: (eval/wp4/mechanism_decomp.py): a Claude Code summary fans out to a median 873 sources,
    #: the closure's required set reaches a median 1,875 events and a 2,048 budget packs the
    #: gold in only 42% of the items that are reachable this way.  The rest are NOT dropped:
    #: they are recorded as ``carrier_fanout_capped`` steps and offered as optional.
    carrier_cap: int = 0

    def __post_init__(self):
        if self.mode not in CLOSURE_MODES:
            raise SchemaError(
                "unknown closure mode: %r (want one of %s)" % (self.mode, CLOSURE_MODES))
        if not isinstance(self.max_hops, int) or isinstance(self.max_hops, bool):
            raise SchemaError("max_hops must be an int, got %r" % (type(self.max_hops).__name__,))
        if self.max_hops < 0:
            raise SchemaError("max_hops must be >= 0, got %d" % self.max_hops)
        if not isinstance(self.follow_optional, bool):
            raise SchemaError("follow_optional must be a bool")
        if not isinstance(self.carrier_cap, int) or isinstance(self.carrier_cap, bool):
            raise SchemaError("carrier_cap must be an int, got %r" % (type(self.carrier_cap).__name__,))
        if self.carrier_cap < 0:
            raise SchemaError("carrier_cap must be >= 0, got %d" % self.carrier_cap)


# ---------------------------------------------------------------- the closure policy


class TypedClosure:
    """Policy-minimal typed dependency closure (§3.3), with §4.5 carrier relaxation."""

    def __init__(self, cfg: ClosureConfig = ClosureConfig()):
        if not isinstance(cfg, ClosureConfig):
            raise SchemaError("cfg must be a ClosureConfig, got %r" % (type(cfg).__name__,))
        self.cfg = cfg

    # ------------------------------------------------------------ public API
    def close(self, query: str, seeds: Sequence[Seed], graph, *, query_mode: str,
              verifications: Sequence[CarrierVerification] = (),
              model_id: str | None = None,
              readout_protocol: str | None = None,
              oracle_required: Mapping[str, Sequence[str]] | None = None) -> EvidenceClosure:
        """Expand ``seeds`` into an :class:`EvidenceClosure` under the configured policy.

        ``verifications`` are §4.5 records; a relaxation fires only when one of them matches the
        carrier id, the edge predicate, ``model_id``, the edge's ``provenance`` (read as the
        carrier *construction protocol*) and ``readout_protocol``.  Passing verifications without
        a model / readout protocol is a hard error: an unbound verification record is exactly the
        silent-corruption case §4.5 exists to prevent.
        """
        if not isinstance(query, str):
            raise SchemaError("query must be a str, got %r" % (type(query).__name__,))
        if query_mode not in QUERY_MODES:
            raise SchemaError("unknown query_mode: %r (want one of %s)" % (query_mode, QUERY_MODES))
        verifications = tuple(verifications or ())
        for v in verifications:
            if not isinstance(v, CarrierVerification):
                raise SchemaError("verifications must be CarrierVerification, got %r"
                                  % (type(v).__name__,))
        if verifications and (model_id is None or readout_protocol is None):
            raise SchemaError(
                "carrier verifications require model_id and readout_protocol (§4.5 binds a "
                "relaxation to a model checkpoint and a readout protocol)")

        view = _GraphView.of(graph)

        seed_ids: list[str] = []
        for s in seeds or ():
            if not isinstance(s, Seed):
                raise SchemaError("seeds must be Seed instances, got %r" % (type(s).__name__,))
            if s.event_id not in view.events:
                raise SchemaError("seed %r is not an event in the graph" % (s.event_id,))
            if s.event_id not in seed_ids:
                seed_ids.append(s.event_id)

        steps: list[ClosureStep] = []
        relax_candidates: list[tuple[str, str, str]] = []   # (carrier, predicate, source)

        if self.cfg.mode == "off":
            required_set: set[str] = set(seed_ids)
            optional_set: set[str] = set()
        elif self.cfg.mode == "all":
            # BASELINE, not a policy: seeds plus every event in the graph, no edge is consulted.
            # The assembler's first-fit sweep then packs whatever short events fit, in time
            # order.  Red team round 2 found this edge-free limit delivers the gold MORE often
            # than native closure at 2048 (0.590 vs 0.495 lexical) -- so any closure result has
            # to be read against it, not against `off` alone.
            required_set = set(seed_ids) | set(view.events)
            optional_set = set()
        elif self.cfg.mode == "oracle":
            required_set, optional_set = self._oracle(query, seed_ids, view, oracle_required, steps)
        else:
            required_set, optional_set = self._expand(
                query, seed_ids, view, query_mode, verifications, model_id, readout_protocol,
                steps, relax_candidates)

        required = view.order(required_set)
        optional = view.order(optional_set - required_set)
        # Only report relaxations that actually stopped an expansion: if the source ended up
        # required through some other child, no source was removed and claiming one would lie.
        req = set(required)
        relaxations = tuple(sorted({
            "%s|%s" % (carrier, predicate)
            for carrier, predicate, source in relax_candidates if source not in req
        }))
        return EvidenceClosure(
            seeds=tuple(seed_ids),
            required=required,
            optional=optional,
            steps=tuple(steps),
            query_mode=query_mode,
            relaxations=relaxations,
        )

    # ------------------------------------------------------------ oracle baseline
    def _oracle(self, query: str, seed_ids: Sequence[str], view: _GraphView,
                oracle_required: Mapping[str, Sequence[str]] | None,
                steps: list[ClosureStep]) -> tuple[set[str], set[str]]:
        """Gold-dependency arm (§6.2 baseline 9): take the list as given, expand nothing."""
        if oracle_required is None:
            raise SchemaError("mode 'oracle' needs oracle_required")
        if query not in oracle_required:
            raise SchemaError("oracle_required has no entry for query %r" % (query,))
        required = set(seed_ids)
        # A step must name a child; the gold list is attached to the earliest seed so the manifest
        # has a real event to point at.  With no seeds there is nothing to attach to and the child
        # is the empty string -- documented, and never a fabricated event id.
        anchor = view.order(seed_ids)[0] if seed_ids else ""
        for gid in tuple(oracle_required[query]):
            if gid not in view.events:
                raise SchemaError("oracle_required names %r which is not in the graph" % (gid,))
            required.add(gid)
            steps.append(ClosureStep(child_id=anchor, parent_id=gid,
                                     edge_type="ORACLE", rule=RULE_ORACLE))
        return required, set()

    # ------------------------------------------------------------ verification gate (§4.5)
    @staticmethod
    def _verified(edge: TraceEdge, verifications: Sequence[CarrierVerification],
                  model_id: str | None, readout_protocol: str | None) -> bool:
        """All five §4.5 fields must match, plus the carrier id itself.

        ``CarrierVerification.matches`` checks predicate / model / construction / readout but NOT
        ``carrier_id`` -- a record for another carrier must never license this one, so the id is
        compared here (test contract #9).  The construction protocol is read off the edge's
        ``provenance``: the MATERIALIZES edge is the record of how the carrier was built.
        """
        if not verifications or model_id is None or readout_protocol is None:
            return False
        predicate = edge.predicate
        if not predicate:
            return False
        for v in verifications:
            if v.carrier_id != edge.src_id:
                continue
            if v.matches(predicate=predicate, model_id=model_id,
                         construction_protocol=edge.provenance,
                         readout_protocol=readout_protocol):
                return True
        return False

    # ------------------------------------------------------------ edge classification
    def _classify_native(self, edge: TraceEdge, child_kind: str, query_mode: str,
                         exact_payload: bool,
                         verified: Callable[[TraceEdge], bool]) -> tuple[str, str]:
        """Return ``(action, rule)`` with action in {skip, required, optional, relaxed}."""
        et = edge.edge_type
        if et in WEAK_EDGES:
            return ("skip", "")                     # §3.2: control/temporal are never evidence
        if et == "RESULT_OF":
            if child_kind == "tool_result":
                return ("required", RULE_TOOL_RESULT)
            return ("required", RULE_NATIVE_REQUIRED)
        if et in ("GROUNDED_IN", "DEPENDS_ON"):
            if child_kind == "decision":
                # §3.3: "why"/"audit" need the decision's sources; a plain lookup does not have to
                # pay for them, so they are offered as optional rather than dropped.  "state" is
                # treated as source-first too: a state answer that rests on a decision should be
                # able to show what the decision rested on.
                if query_mode == "lookup":
                    return ("optional", RULE_WHY_SOURCES)
                return ("required", RULE_WHY_SOURCES)
            return ("required", RULE_NATIVE_REQUIRED)
        if et == "MATERIALIZES":
            # §3.1 source-first: by default a carrier does NOT free you from its source.
            if query_mode == "lookup" and exact_payload:
                # a predicate-only carrier cannot answer for a literal payload, verified or not
                return ("required", RULE_EXACT_PAYLOAD)
            if verified(edge):
                return ("relaxed", RULE_VERIFIED_CARRIER)
            return ("required", RULE_UNVERIFIED_CARRIER)
        if et == "SUPERSEDES":
            # reachable only through state_prefers_latest; never a generic evidence parent
            return ("skip", "")
        return ("skip", "")

    @staticmethod
    def _classify_full(edge: TraceEdge) -> tuple[str, str]:
        """Over-expanding arm (§6.2 baseline 7): every STRONG edge, no rules, no relaxation."""
        if edge.edge_type in STRONG_EDGES:
            return ("required", RULE_FULL_ANCESTOR)
        return ("skip", "")

    # ------------------------------------------------------------ state rule (§3.3, contract #7)
    def _latest_of_chain(self, seed_id: str, view: _GraphView,
                         steps: list[ClosureStep]) -> tuple[str, list[str]]:
        """Walk ``new --SUPERSEDES--> old`` backwards to the newest event of ``seed_id``'s chain.

        Returns ``(latest_id, superseded_ids)``.  A fork (two events superseding the same state)
        continues down the chronologically latest branch and records the losers as superseded, so
        the walk is total and deterministic instead of arbitrary.  A SUPERSEDES cycle is broken by
        the visited set and recorded as a ``cycle_guard`` step.
        """
        cur = seed_id
        seen = {seed_id}
        superseded: list[str] = []
        n = len(view.chrono_index)
        while True:
            succs = sorted({e.src_id for e in view.incoming(cur)
                            if e.edge_type == "SUPERSEDES" and e.src_id in view.events})
            if not succs:
                return cur, superseded
            nxt = max(succs, key=lambda i: (view.chrono_index.get(i, n), i))
            superseded.extend(s for s in succs if s != nxt)
            if nxt in seen:
                steps.append(ClosureStep(child_id=nxt, parent_id=cur,
                                         edge_type="SUPERSEDES", rule=RULE_CYCLE))
                return cur, superseded
            seen.add(nxt)
            superseded.append(cur)
            cur = nxt

    # ------------------------------------------------------------ the two-wave expansion
    def _expand(self, query: str, seed_ids: Sequence[str], view: _GraphView, query_mode: str,
                verifications: Sequence[CarrierVerification], model_id: str | None,
                readout_protocol: str | None, steps: list[ClosureStep],
                relax_candidates: list[tuple[str, str, str]]) -> tuple[set[str], set[str]]:
        max_hops = self.cfg.max_hops
        full = self.cfg.mode == "full_ancestor"
        exact_payload = is_exact_payload_query(query)

        def verified(edge: TraceEdge) -> bool:
            return self._verified(edge, verifications, model_id, readout_protocol)

        def classify(edge: TraceEdge, child_id: str) -> tuple[str, str]:
            if full:
                return self._classify_full(edge)
            return self._classify_native(edge, view.kind_of(child_id), query_mode,
                                         exact_payload, verified)

        # --- state_prefers_latest runs BEFORE the waves, so a demoted seed is never expanded as
        #     required in the first place (contract #7: never default to the old state).
        demoted: set[str] = set()
        opt_seed_entries: list[tuple[str, int]] = []
        req_seed_ids: list[str] = list(seed_ids)
        if query_mode == "state" and not full:
            promoted: list[str] = []
            for sid in view.order(seed_ids):
                latest, superseded = self._latest_of_chain(sid, view, steps)
                if latest == sid and not superseded:
                    continue
                steps.append(ClosureStep(child_id=sid, parent_id=latest,
                                         edge_type="SUPERSEDES", rule=RULE_STATE_LATEST))
                promoted.append(latest)
                for old in superseded:
                    demoted.add(old)
                    opt_seed_entries.append((old, 0))
            req_seed_ids = [s for s in req_seed_ids if s not in demoted] + list(promoted)

        required: set[str] = set()
        optional: set[str] = set()
        seen_req: set[str] = set()
        seen_opt: set[str] = set()
        rq: deque[tuple[str, int]] = deque()
        oq: deque[tuple[str, int]] = deque(opt_seed_entries)

        for sid in view.order(req_seed_ids):
            if sid in seen_req:
                continue
            seen_req.add(sid)
            required.add(sid)
            rq.append((sid, 0))

        # --- carrier_cap: a summary's MATERIALIZES fan-out is ranked by query-term overlap with
        #     the source text (gold-blind) and only the top `carrier_cap` are followed as required;
        #     the rest are recorded (RULE_CARRIER_CAPPED) and go to the optional frontier.
        cap = self.cfg.carrier_cap
        qterms = query_terms(query) if cap > 0 else frozenset()
        term_cache: dict[str, frozenset[str]] = {}

        def relevance(eid: str) -> int:
            ts = term_cache.get(eid)
            if ts is None:
                ev = view.events.get(eid)
                ts = query_terms(getattr(ev, "text", "") or "") if ev is not None else frozenset()
                term_cache[eid] = ts
            return len(qterms & ts)

        def capped_out(node_id: str) -> set[str]:
            if cap <= 0:
                return set()
            mats = [e for e in view.out(node_id) if e.edge_type == "MATERIALIZES"]
            if len(mats) <= cap:
                return set()
            n = len(view.chrono_index)
            ranked = sorted(mats, key=lambda e: (-relevance(e.dst_id), view.chrono_index.get(e.dst_id, n), e.dst_id))
            return {e.dst_id for e in ranked[cap:]}

        def scan(node_id: str, hop: int, sink: Callable[[str, int, str], None]) -> None:
            """Classify every out-edge of ``node_id`` once, recording a step for each follow."""
            capped = capped_out(node_id)
            for edge in view.out(node_id):
                if capped and edge.edge_type == "MATERIALIZES" and edge.dst_id in capped:
                    steps.append(ClosureStep(child_id=node_id, parent_id=edge.dst_id,
                                             edge_type=edge.edge_type, rule=RULE_CARRIER_CAPPED))
                    sink(edge.dst_id, hop + 1, "optional")
                    continue
                action, rule = classify(edge, node_id)
                if action == "skip":
                    continue
                if edge.dst_id not in view.events:
                    raise SchemaError("edge %s --%s--> %s points at an event not in the graph"
                                      % (edge.src_id, edge.edge_type, edge.dst_id))
                if hop >= max_hops:
                    steps.append(ClosureStep(child_id=node_id, parent_id=edge.dst_id,
                                             edge_type=edge.edge_type, rule=RULE_TRUNCATED))
                    continue
                steps.append(ClosureStep(child_id=node_id, parent_id=edge.dst_id,
                                         edge_type=edge.edge_type, rule=rule))
                if action == "relaxed":
                    relax_candidates.append((edge.src_id, edge.predicate or "", edge.dst_id))
                sink(edge.dst_id, hop + 1, action)

        # --- wave 1: required, to a fixed point (cycle-safe via seen_req) --------------------
        def required_sink(dst: str, nhop: int, action: str) -> None:
            if action == "required":
                if dst not in seen_req:
                    seen_req.add(dst)
                    required.add(dst)
                    rq.append((dst, nhop))
            else:                            # optional / relaxed -> optional frontier
                oq.append((dst, nhop))

        while rq:
            node_id, hop = rq.popleft()
            scan(node_id, hop, required_sink)

        # --- wave 2: optional.  Optionality is inherited: nothing below an optional node can be
        #     required, so no promotion is possible and required/optional stay disjoint. -------
        def optional_sink(dst: str, nhop: int, action: str) -> None:
            oq.append((dst, nhop))

        if self.cfg.follow_optional:
            while oq:
                node_id, hop = oq.popleft()
                if node_id in seen_req or node_id in seen_opt:
                    continue
                seen_opt.add(node_id)
                optional.add(node_id)
                scan(node_id, hop, optional_sink)
        else:
            for node_id, _hop in oq:
                if node_id not in seen_req:
                    optional.add(node_id)

        # An explicit demotion beats any other path that would have made the old state required:
        # otherwise state_prefers_latest would be a no-op exactly when it matters (the old state
        # IS the seed).  It moves to optional, never to the void -- audits need it.
        required -= demoted
        optional |= demoted
        optional -= required
        return required, optional


# ---------------------------------------------------------------- self-check


class _SelfcheckGraph:
    """Minimal graph stand-in: mapping of events + tuple of edges + chronological property."""

    def __init__(self, events: Sequence[TraceEvent], edges: Sequence[TraceEdge]):
        self.events = {e.event_id: e for e in events}
        self.edges = tuple(edges)

    @property
    def chronological(self) -> tuple[TraceEvent, ...]:
        return tuple(sorted(self.events.values(), key=lambda e: (e.timestamp, e.event_id)))


class _IdChronoGraph(_SelfcheckGraph):
    """Same graph, but ``chronological`` yields bare ids -- proves duck-typing handles both."""

    @property
    def chronological(self) -> tuple[str, ...]:
        return tuple(e.event_id for e in
                     sorted(self.events.values(), key=lambda e: (e.timestamp, e.event_id)))


class _NoChronoGraph:
    """Graph without ``chronological`` -- proves the (timestamp, id) fallback works."""

    def __init__(self, events: Sequence[TraceEvent], edges: Sequence[TraceEdge]):
        self.events = tuple(events)
        self.edges = tuple(edges)


def _fixture():
    """A tiny supplier-rejection trace plus a supersession chain and a dependency cycle."""
    ev = [
        TraceEvent("e1", "user", "why was supplier B rejected", 1),
        TraceEvent("e2", "tool_call", "search(vendor_db, 'supplier B')", 2, tool_call_id="tc1"),
        TraceEvent("e3", "tool_result", "supplier B fails the region check", 3, tool_call_id="tc1"),
        TraceEvent("e4", "decision", "reject supplier B", 4),
        TraceEvent("e5", "summary", "supplier B is not region eligible", 5),
        TraceEvent("e6", "state", "shortlist = [A, C]", 6),
        TraceEvent("e7", "state", "shortlist = [A]", 7),
        TraceEvent("e8", "assistant", "we went with supplier A", 8),
        TraceEvent("e9", "decision", "cycle head", 9),
        TraceEvent("e10", "decision", "cycle tail", 10),
        TraceEvent("e11", "assistant", "scratch thought reachable only by CONTROL", 11),
    ]
    ed = [
        TraceEdge("e3", "e2", "RESULT_OF"),
        TraceEdge("e4", "e3", "DEPENDS_ON"),
        TraceEdge("e5", "e3", "MATERIALIZES", predicate="region_eligible", provenance="native"),
        TraceEdge("e7", "e6", "SUPERSEDES"),
        TraceEdge("e8", "e4", "TEMPORAL"),
        TraceEdge("e4", "e11", "CONTROL"),
        TraceEdge("e9", "e10", "DEPENDS_ON"),
        TraceEdge("e10", "e9", "DEPENDS_ON"),
    ]
    return _SelfcheckGraph(ev, ed), ev, ed


def _rules_of(closure: EvidenceClosure) -> set[str]:
    return {s.rule for s in closure.steps}


def _selfcheck() -> None:
    g, ev, ed = _fixture()

    def seed(i, src="lexical"):
        return Seed(event_id=i, score=1.0, source=src)

    good_ver = CarrierVerification(
        carrier_id="e5", predicate="region_eligible", model_id="m1",
        construction_protocol="native", readout_protocol="r1",
        test_version="v1", ci_low=0.91, ci_high=0.97)

    tc = TypedClosure()

    # ---- 0. graph duck-typing: three shapes, identical answer ------------------------------
    base = tc.close("why was supplier B rejected", [seed("e4")], g, query_mode="why")
    for alt in (_IdChronoGraph(ev, ed), _NoChronoGraph(ev, ed)):
        other = tc.close("why was supplier B rejected", [seed("e4")], alt, query_mode="why")
        assert other.required == base.required, (alt, other.required, base.required)
        assert other.steps == base.steps

    # ---- 1. mode off: seeds only -------------------------------------------------------------
    off = TypedClosure(ClosureConfig(mode="off")).close(
        "why was supplier B rejected", [seed("e4"), seed("e3")], g, query_mode="why")
    assert off.required == ("e3", "e4"), off.required      # chronological, not seed order
    assert off.optional == () and off.steps == () and off.relaxations == ()
    assert off.seeds == ("e4", "e3")                       # seeds keep retrieval order

    # ---- 2. tool_result pulls its tool_call ---------------------------------------------------
    tr = tc.close("what did the vendor lookup return", [seed("e3")], g, query_mode="why")
    assert tr.required == ("e2", "e3"), tr.required
    assert RULE_TOOL_RESULT in _rules_of(tr)
    assert any(s.child_id == "e3" and s.parent_id == "e2" and s.edge_type == "RESULT_OF"
               for s in tr.steps)

    # ---- 3. WEAK edges are never followed -----------------------------------------------------
    for mode in ("native", "full_ancestor"):
        c = TypedClosure(ClosureConfig(mode=mode)).close(
            "why was supplier B rejected", [seed("e4")], g, query_mode="audit")
        assert "e11" not in c.required and "e11" not in c.optional, (mode, c)
        assert all(s.edge_type not in WEAK_EDGES for s in c.steps)

    # ---- 4. why/audit vs lookup ---------------------------------------------------------------
    why = tc.close("why was supplier B rejected", [seed("e4")], g, query_mode="why")
    assert why.required == ("e2", "e3", "e4"), why.required
    assert RULE_WHY_SOURCES in _rules_of(why)
    audit = tc.close("show the provenance for supplier B", [seed("e4")], g, query_mode="audit")
    assert audit.required == ("e2", "e3", "e4"), audit.required
    look = tc.close("which supplier did we pick", [seed("e4")], g, query_mode="lookup")
    assert look.required == ("e4",), look.required
    assert look.optional == ("e2", "e3"), look.optional     # offered, never silently dropped
    assert RULE_WHY_SOURCES in _rules_of(look)
    assert not (set(look.required) & set(look.optional))

    # follow_optional=False records the optional parent but does not expand through it
    lean = TypedClosure(ClosureConfig(follow_optional=False)).close(
        "which supplier did we pick", [seed("e4")], g, query_mode="lookup")
    assert lean.required == ("e4",) and lean.optional == ("e3",), (lean.required, lean.optional)

    # ---- 5. carrier: unverified does NOT relax (test contract #10) -----------------------------
    q_pred = "was supplier B eligible for the region"
    assert not is_exact_payload_query(q_pred), explain_exact_payload_query(q_pred)
    unver = tc.close(q_pred, [seed("e5")], g, query_mode="why")
    assert "e3" in unver.required and "e2" in unver.required, unver.required
    assert unver.relaxations == (), unver.relaxations
    assert RULE_UNVERIFIED_CARRIER in _rules_of(unver)

    # ---- 6. carrier: a verified record DOES relax ----------------------------------------------
    ver = tc.close(q_pred, [seed("e5")], g, query_mode="why",
                   verifications=[good_ver], model_id="m1", readout_protocol="r1")
    assert ver.required == ("e5",), ver.required
    assert ver.relaxations == ("e5|region_eligible",), ver.relaxations
    assert ver.optional == ("e2", "e3"), ver.optional        # kept for the audit trail
    assert RULE_VERIFIED_CARRIER in _rules_of(ver)

    # ---- 7. every §4.5 field is load-bearing (test contract #9) --------------------------------
    for kw in (dict(model_id="m2", readout_protocol="r1"),      # wrong model checkpoint
               dict(model_id="m1", readout_protocol="r2")):     # wrong readout protocol
        c = tc.close(q_pred, [seed("e5")], g, query_mode="why", verifications=[good_ver], **kw)
        assert "e3" in c.required and c.relaxations == (), (kw, c.required, c.relaxations)
    for broken in (
        # right predicate, WRONG carrier -- must not license e5
        CarrierVerification("e4", "region_eligible", "m1", "native", "r1", "v1", .9, .9),
        # right carrier, WRONG predicate (§4.5's own example)
        CarrierVerification("e5", "supported_regions", "m1", "native", "r1", "v1", .9, .9),
        # right carrier, WRONG construction protocol
        CarrierVerification("e5", "region_eligible", "m1", "annotated", "r1", "v1", .9, .9),
    ):
        c = tc.close(q_pred, [seed("e5")], g, query_mode="why", verifications=[broken],
                     model_id="m1", readout_protocol="r1")
        assert "e3" in c.required and c.relaxations == (), (broken, c.required)

    # ---- 8. exact payload refuses relaxation even when verified ---------------------------------
    for q in ("what was the exact invoice number", "which sha256 did the build report",
              "what is in the policy file /etc/policy.yaml", "how many suppliers passed",
              "给出原文里的资质结论"):
        assert is_exact_payload_query(q), q
        c = tc.close(q, [seed("e5")], g, query_mode="lookup",
                     verifications=[good_ver], model_id="m1", readout_protocol="r1")
        assert "e3" in c.required, (q, c.required)
        assert c.relaxations == (), (q, c.relaxations)
        assert RULE_EXACT_PAYLOAD in _rules_of(c), q
    # ... and the detector stays quiet on a plain predicate question
    for q in ("was supplier B eligible for the region", "did we reject supplier B"):
        assert not is_exact_payload_query(q), (q, explain_exact_payload_query(q))

    # ---- 9. state prefers the latest, keeps the old one visible (contract #7) --------------------
    st = tc.close("what is the shortlist now", [seed("e6")], g, query_mode="state")
    assert "e7" in st.required and "e6" not in st.required, st.required
    assert "e6" in st.optional, st.optional
    assert RULE_STATE_LATEST in _rules_of(st)
    # a lookup on the same seed must NOT silently jump to the newest state
    st_look = tc.close("which supplier is shortlisted", [seed("e6")], g, query_mode="lookup")
    assert st_look.required == ("e6",), st_look.required
    # seeding the newest state directly is a no-op
    st2 = tc.close("what is the shortlist now", [seed("e7")], g, query_mode="state")
    assert st2.required == ("e7",) and st2.optional == (), (st2.required, st2.optional)

    # ---- 10. cycles terminate --------------------------------------------------------------------
    cyc = tc.close("why the cycle", [seed("e9")], g, query_mode="why")
    assert set(cyc.required) == {"e9", "e10"}, cyc.required
    assert len(cyc.steps) == 2, cyc.steps          # each edge classified exactly once

    # ---- 11. max_hops truncation is recorded, never silent ----------------------------------------
    short = TypedClosure(ClosureConfig(max_hops=1)).close(
        "why was supplier B rejected", [seed("e4")], g, query_mode="why")
    assert short.required == ("e3", "e4"), short.required
    assert RULE_TRUNCATED in _rules_of(short)
    zero = TypedClosure(ClosureConfig(max_hops=0)).close(
        "why was supplier B rejected", [seed("e4")], g, query_mode="why")
    assert zero.required == ("e4",), zero.required
    assert _rules_of(zero) == {RULE_TRUNCATED}

    # ---- 12. full_ancestor over-expands relative to native -----------------------------------------
    fa = TypedClosure(ClosureConfig(mode="full_ancestor")).close(
        q_pred, [seed("e5")], g, query_mode="why",
        verifications=[good_ver], model_id="m1", readout_protocol="r1")
    assert set(fa.required) == {"e5", "e3", "e2"}, fa.required
    assert fa.relaxations == (), fa.relaxations       # the baseline never relaxes
    assert set(ver.required) < set(fa.required)
    fa_state = TypedClosure(ClosureConfig(mode="full_ancestor")).close(
        "shortlist", [seed("e7")], g, query_mode="state")
    assert set(fa_state.required) == {"e7", "e6"}, fa_state.required   # SUPERSEDES followed here

    # ---- 13. oracle takes the gold list as given ----------------------------------------------------
    orc = TypedClosure(ClosureConfig(mode="oracle")).close(
        "why was supplier B rejected", [seed("e5", "oracle")], g, query_mode="why",
        oracle_required={"why was supplier B rejected": ["e2", "e3"]})
    assert orc.required == ("e2", "e3", "e5"), orc.required
    assert _rules_of(orc) == {RULE_ORACLE}

    # ---- 14. determinism: same inputs -> identical closure -------------------------------------------
    a = tc.close(q_pred, [seed("e5"), seed("e4")], g, query_mode="why",
                 verifications=[good_ver], model_id="m1", readout_protocol="r1")
    b = tc.close(q_pred, [seed("e5"), seed("e4")], g, query_mode="why",
                 verifications=[good_ver], model_id="m1", readout_protocol="r1")
    assert a == b
    # e3 is required via e4's DEPENDS_ON, so the carrier stop removed no source ->
    # relaxations must stay empty rather than over-claim.
    assert "e3" in a.required and a.relaxations == (), (a.required, a.relaxations)

    # ---- 15. chronological ordering + disjointness hold everywhere ------------------------------------
    order = {e.event_id: i for i, e in enumerate(g.chronological)}
    for c in (off, tr, why, look, ver, unver, st, cyc, short, fa, orc, a):
        assert list(c.required) == sorted(c.required, key=lambda i: order[i]), c.required
        assert list(c.optional) == sorted(c.optional, key=lambda i: order[i]), c.optional
        assert not (set(c.required) & set(c.optional)), c
        assert all(s.rule in RULES for s in c.steps), c.steps

    # ---- 16. fault injection: broken inputs must be REJECTED -------------------------------------------
    def rejects(fn, label):
        try:
            fn()
        except SchemaError:
            return
        except Exception as exc:                                   # pragma: no cover
            raise AssertionError("%s raised %r, want SchemaError" % (label, exc))
        raise AssertionError("%s was accepted, want SchemaError" % label)

    rejects(lambda: ClosureConfig(mode="magic"), "unknown mode")
    rejects(lambda: ClosureConfig(max_hops=-1), "negative max_hops")
    rejects(lambda: ClosureConfig(follow_optional="yes"), "non-bool follow_optional")
    rejects(lambda: TypedClosure("native"), "cfg that is not a ClosureConfig")
    rejects(lambda: tc.close("q", [seed("e4")], g, query_mode="explain"), "unknown query_mode")
    rejects(lambda: tc.close(42, [seed("e4")], g, query_mode="why"), "non-str query")
    rejects(lambda: tc.close("q", [seed("nope")], g, query_mode="why"), "seed not in graph")
    rejects(lambda: tc.close("q", ["e4"], g, query_mode="why"), "raw str instead of Seed")
    rejects(lambda: tc.close("q", [seed("e4")], None, query_mode="why"), "graph is None")
    rejects(lambda: tc.close("q", [seed("e4")], object(), query_mode="why"), "graph without events")
    rejects(lambda: tc.close(q_pred, [seed("e5")], g, query_mode="why", verifications=[good_ver]),
            "verification without model_id/readout_protocol")
    rejects(lambda: tc.close(q_pred, [seed("e5")], g, query_mode="why",
                             verifications=["not-a-record"], model_id="m1", readout_protocol="r1"),
            "verification of the wrong type")
    rejects(lambda: TypedClosure(ClosureConfig(mode="oracle")).close(
        "q", [seed("e4")], g, query_mode="why"), "oracle without oracle_required")
    rejects(lambda: TypedClosure(ClosureConfig(mode="oracle")).close(
        "q", [seed("e4")], g, query_mode="why", oracle_required={"other": ["e2"]}),
        "oracle_required missing this query")
    rejects(lambda: TypedClosure(ClosureConfig(mode="oracle")).close(
        "q", [seed("e4")], g, query_mode="why", oracle_required={"q": ["ghost"]}),
        "oracle_required naming a ghost event")
    rejects(lambda: explain_exact_payload_query(None), "non-str query to the payload detector")

    # a dangling STRONG edge must be caught the moment closure would follow it
    dangling = _SelfcheckGraph(ev, list(ed) + [TraceEdge("e8", "ghost", "DEPENDS_ON")])
    rejects(lambda: tc.close("why", [seed("e8")], dangling, query_mode="why"), "dangling edge")

    # sanity: the whitelist the module was written against has not drifted
    assert REQUIRED_EDGES == ("RESULT_OF", "GROUNDED_IN", "DEPENDS_ON"), REQUIRED_EDGES

    print("closure._selfcheck: 17 sections OK "
          "(off/native/full_ancestor/oracle, 5 named rules, 18 rejections)")


if __name__ == "__main__":
    _selfcheck()
