"""tracepack.core.assembler -- hard-budget group packing + auditable manifest (§3.4, §3.5).

This module turns an :class:`EvidenceClosure` into a :class:`MemoryPacket`.  It is the place
where the proposal's "hard budget is a system contract" clause (§3.4) is actually enforced, and
where the audit record demanded by §3.5 is minted.

Design decisions (the ones other modules depend on are marked [CONTRACT]):

* **[CONTRACT] The budget is never violated, and required evidence is never SILENTLY truncated.**
  §3.4 forbids a silent cut: a half-quoted tool result the reader cannot tell was cut is a worse
  failure than a declared miss.  It does **not** forbid a *labelled* excerpt, which is exactly
  what contract #12 governs -- the reader can see an excerpt happened.  Read "whole or not at
  all" as "whole, or a labelled excerpt, or declared missing" -- never a silent cut.
  When the mandatory set still does not fit we drop optional first, then report the missing
  required ids in ``manifest.missing_required`` and set ``incomplete=True``.

  This wording used to say "whole or not at all" flat, and the pilots read it as "no excerpting",
  never installed an ``excerpt_fn``, and so ran every unit all-or-nothing.  Measured cost of that
  reading: the packer threw away its own top-ranked evidence in 65% of real-trace packets at
  2,048 tokens and 90% of the required tokens with it (RESULTS_ablation §22).  Dropping the
  highest-ranked evidence because it is long is not a defensible reading of "never truncate".

* **[CONTRACT] Atomicity is resolved over the closure's candidate set, not over the graph.**
  A unit is one event, or all *candidate* events sharing an ``atomic_group``.  The assembler
  never pulls an event into the packet that the closure did not select -- that would quietly
  un-do the policy-minimal closure of §3.3.  The practical consequence for the closure module:
  **if a tool_call and its tool_result must stay together, the closure has to place both in the
  packet's candidate set** (which ``RESULT_OF`` expansion does anyway).  Atomicity here means
  "never split", not "silently re-expand".

* **[CONTRACT] A unit's tier is required if *any* member is required.**  A group that mixes a
  required and an optional member is packed together at the required tier; if it does not fit,
  the required members are reported missing and the optional members omitted.  This is the only
  reading of "all-or-nothing" that cannot split a group.

* **[CONTRACT] Representation choice is uniform within a unit and made at pack time.**
  ``source_plus_carrier`` tries the richer rendering for the whole unit first and falls back to
  source-only for the whole unit; it never mixes, so that "source wins if only one fits" stays
  decidable.  Source always wins the fallback -- §3.1 is source-first.

* **[CONTRACT] ``total_tokens = header_cost + sum(entry.token_cost)``.**  Event costs come from
  the chosen :class:`RepresentationRef` (falling back to ``event.token_cost``); this module
  never re-tokenizes -- the adapter/eval owns the target tokenizer.  The inter-entry
  ``separator`` is treated as constant structural glue absorbed by the ``L_safety_margin`` term
  of §3.4's budget equation, *not* charged per entry, because charging it would require exactly
  the tokenizer we deliberately do not have here.  The header has no RepresentationRef, so its
  cost is the conservative (over-)estimate of :func:`estimate_text_tokens`; over-estimating can
  only make the packet smaller, never violate the budget.

* **Packing order and render order are different things.**  Packing is by priority (pinned
  seeds -> seed-ranked required -> chronological required -> ranked optional, first-fit greedy).
  Rendering is the "chronological/causal reassembly" step of §4.3 and is governed by
  ``cfg.order``.  ``manifest.entries`` mirrors the render order so the digest is a hash over
  what was actually served.

* **The query mode is audit metadata here.**  Mode-dependent evidence selection (SUPERSEDES
  recency, exact-payload carrier bans) belongs to the closure.  The assembler only refuses a
  mode that contradicts ``closure.query_mode``, so a mislabelled packet cannot be minted.

Everything is pure and deterministic: no clock, no RNG, no network, no LLM, no torch.  Given
the same (query, closure, graph, budget, seeds, config) the context string and the manifest
digest are byte-identical.

Not implemented on purpose -- see the module report: the assembler does **not** enforce the §4.5
five-field carrier verification gate.  ``RepresentationRef`` carries a single ``protocol_id``
while :class:`CarrierVerification` requires distinct construction *and* readout protocols, so a
faithful match is not expressible against the frozen IR; the gate lives in the closure
(``EvidenceClosure.relaxations``) and the assembler only carries the records into the manifest
for audit.  ``carrier_only`` is additionally an *experimental arm* (§5.3): the profiler must be
able to force it in order to measure the carrier counterfactual, so gating it here would delete
the measurement.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Mapping, Sequence

try:                                   # normal package import
    from .schema import (
        QUERY_MODES,
        BudgetError,
        CarrierVerification,
        ClosureStep,
        EvidenceClosure,
        MemoryPacket,
        PacketEntry,
        PacketManifest,
        RepresentationRef,
        SchemaError,
        Seed,
        TraceEvent,
    )
except ImportError:                    # `python3 tracepack/core/assembler.py` (selfcheck)
    import os
    import sys

    sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
    from schema import (               # type: ignore[no-redef]
        QUERY_MODES,
        BudgetError,
        CarrierVerification,
        ClosureStep,
        EvidenceClosure,
        MemoryPacket,
        PacketEntry,
        PacketManifest,
        RepresentationRef,
        SchemaError,
        Seed,
        TraceEvent,
    )

__all__ = [
    "REPR_POLICIES",
    "ORDERS",
    "AssemblerConfig",
    "BudgetAssembler",
    "estimate_text_tokens",
]

#: representation policies understood by :class:`AssemblerConfig` (§3.1, §5.3)
REPR_POLICIES = ("source_only", "carrier_only", "source_plus_carrier")
#: packing priorities for non-seed REQUIRED events.  `chrono` (shipped): oldest first, blind to
#: which seed needed them -- with a 900-event closure that is "the start of the session".
#: `bundle`: an event packs right behind the best-ranked seed it was reached from, strong edges
#: (DEPENDS_ON / RESULT_OF / GROUNDED_IN) before MATERIALIZES, nearer hops first.  Red team round 2
#: §1.6 showed the closure's defect is the absence of a ranking, not its size; this is the ranking.
#: `tiered` / `evidence_first` (phase 2, PLAN_phase2.md WP1): the two shipped orders put every
#: non-seed required event AFTER every seed (chrono) or interleave it with the seed it hangs off
#: *including* the compaction-summary fan-out (bundle).  The measured ceiling (`opin`, required
#: sources packed before the seeds) is +12-19pp over both, so the missing ingredient is letting
#: STRONG, NEAR evidence outrank low-ranked seeds while the MATERIALIZES fan-out never does:
#:   tiered         pinned -> seed r, then r's strong evidence within `evidence_hops` (by hop,
#:                  class, time) -> seed r+1 ... -> everything reached only through weak edges,
#:                  by hop then time -> optional
#:   evidence_first pinned -> strong near evidence of ANY seed (by origin seed rank, hop, class,
#:                  time) -> the seeds by rank -> weak-reached required -> optional
PACKS = ("chrono", "bundle", "tiered", "evidence_first")
_EDGE_CLASS = {"DEPENDS_ON": 0, "RESULT_OF": 0, "GROUNDED_IN": 0, "SUPERSEDES": 1, "MATERIALIZES": 2}
#: class table for the phase-2 packs.  ORACLE steps (closure mode `oracle`) stand in for perfect
#: hop-1 strong edges, so they are class 0 here; `bundle` keeps its original table untouched so
#: every packet it ever produced still reproduces byte for byte (contract #13).
_EVIDENCE_CLASS = {"DEPENDS_ON": 0, "RESULT_OF": 0, "GROUNDED_IN": 0, "ORACLE": 0,
                   "SUPERSEDES": 1, "MATERIALIZES": 2}
_EVIDENCE_PACKS = ("tiered", "evidence_first")

#: rendered orderings (§4.3 "chronological/causal reassembly")
ORDERS = ("chronological", "seed_first")

#: sentinel rank for "this event was not a seed" -- keeps sort keys total and int-comparable
_NO_RANK = 1 << 30

_SOURCE_KIND = "raw_text"
_CARRIER_KIND = "materialized_text"
_EXCERPT_KIND = "excerpt"
#: repr_kind recorded when both witnesses are served for one event (source_plus_carrier)
_BOTH_KIND = "raw_text+materialized_text"


def estimate_text_tokens(text: str) -> int:
    """Conservative, tokenizer-free upper estimate for text that has no RepresentationRef.

    Only used for ``cfg.header``.  Deliberately over-estimates (max of a word count and a
    3-chars-per-token count) because the only safe direction of error against a hard budget is
    up: over-estimating shrinks the packet, under-estimating breaks the §3.4 contract.  CJK text
    has no spaces, which is exactly why the character term is there.
    """
    if not text:
        return 0
    return max(1, len(text.split()), (len(text) + 2) // 3)


# ---------------------------------------------------------------- config


@dataclass(frozen=True)
class AssemblerConfig:
    repr_policy: str = "source_only"   # source_only | carrier_only | source_plus_carrier
    header: str = ""                   # optional prefix, counted against the budget
    separator: str = "\n\n"
    order: str = "chronological"       # chronological | seed_first
    pack: str = "chrono"               # chrono | bundle | tiered | evidence_first
    #: `tiered` / `evidence_first` only: how many strong-edge hops from a seed still count as
    #: "near evidence" that may outrank a lower seed.  Beyond it, an event packs in the weak tier.
    evidence_hops: int = 2
    #: `tiered` / `evidence_first` only: the share of the budget that pure-evidence units (no seed,
    #: no pin among their members) may take BEFORE the seeds.  Phase A measured the flood on
    #: LLM-inferred edges: a median of 7 hop-1 evidence units / 4,248 tokens per item against a
    #: 2,048 budget, so an uncapped evidence tier evicts every seed (delivery 0.442 -> 0.116).
    #: Evidence beyond the share is not dropped -- it is DEFERRED to after the seeds and before
    #: the weak tier.  1.0 = no cap.
    evidence_share: float = 1.0
    #: `tiered` / `evidence_first` only: the share of the budget ANY ONE unit may spend on its
    #: first attempt.  Over the ceiling, `_fit_unit` falls to a cheaper witness -- with an
    #: `excerpt_fn` installed that is a labelled excerpt, so the fat unit is still SERVED, just
    #: not at full length; with nothing cheap enough the unit is held and retried at the end
    #: against what survives, so the cap never drops what the uncapped packer would have served
    #: by then.  `evidence_share` caps only *pure-evidence* units, so a fat
    #: unit that is near-strong evidence of the TOP seed escapes it entirely and packs first.
    #: RESULTS_hops §7 measured what that costs: on the 4-hop scripted tasks a 3,618-token unit
    #: at seed rank 0 does not fit a 2,048 budget (so the packet serves 16 cheap units and the
    #: record), fits a 4,096 one, and then starves five units including the 83-token record --
    #: the packet gets WORSE as the budget doubles (E served 38% -> 0%).  The deeper the chain,
    #: the later the record sorts (it is furthest from any seed), so multi-hop is exactly where
    #: this bites.  A unit over the cap is deferred to the very END (after weak + optional), not
    #: dropped: it still goes in if the budget survives.  None = no cap (the shipped behaviour;
    #: every published number was produced with it).
    #: 0.8 is measured, not picked: it catches the unit that ate 88% of the budget (3,618 of 4,096,
    #: RESULTS_hops §7) and leaves alone the 56%-of-budget evidence pair that contract #11f says
    #: must still outrank the lowest seed.  0.5 breaks 11f; 0.9 lets the 88% unit through again
    #: (hop4 recovers to 25% instead of 38%).  None restores the shipped packer exactly.
    unit_cap_share: float | None = 0.8
    #: `tiered` / `evidence_first` only: order units WITHIN a tier by cost, cheapest first, instead
    #: of by origin seed rank then hop.  The tier order is untouched, so contract #11 (weak-reached
    #: material never precedes a seed) still holds -- what changes is that among equally-qualified
    #: evidence the packer takes the many cheap ones before the one expensive one.
    #:
    #: This is the root-cause fix for the family `evidence_share` and `unit_cap_share` each patch
    #: one instance of: the packing key has no cost term, so whatever sorts first can starve
    #: everything below it however much it costs.  Phase A hit it as a tier flooding the seeds
    #: (delivery 0.442 -> 0.116, fixed by capping the tier); the hop sweep hit it as ONE unit
    #: flooding the tail (E served 38% -> 0% when the budget doubled, RESULTS_hops §7).  Two
    #: instances, one cause.
    cost_order: bool = True
    #: `tiered` / `evidence_first` only: when no ``excerpt_fn`` is injected, build the standard
    #: gold-blind line excerpter (core/excerpt.py) so a unit over `unit_cap_share` is served as a
    #: LABELLED excerpt instead of being held out whole.
    #:
    #: Default True because the alternative is indefensible and was measured: with no excerpt tier
    #: every unit is all-or-nothing, and the packer throws away its OWN top-ranked evidence in 65%
    #: of real-trace packets at 2,048 tokens (100% of scripted ones) -- 90-96% of the required
    #: tokens dropped.  Contract #1 forbids a SILENT cut, and says so in as many words ("the reader
    #: cannot tell it happened"); a labelled excerpt is not silent, and contract #12 governs it.
    #: With the excerpt tier on, top-unit loss falls to 12% / 7% and required-token loss to 47% /
    #: 45% (RESULTS_ablation §22).  Set False to reproduce a run made before this default.
    excerpt: bool = True
    #: append a short labelled line to the CONTEXT naming how much required evidence was left out.
    #:
    #: Contract #1 refuses a silent cut because "the reader cannot tell it happened" -- but the
    #: declaration it offers instead goes to ``manifest.missing_required``, and the consumer is
    #: handed only ``packet.context``.  So a whole omitted required event is exactly as invisible
    #: to the reader as the half-quote the contract forbids.  This closes that gap.
    #:
    #: Default False: turning it on changes every packet's bytes and therefore every read-out, so
    #: it is a measured change, not a silent one (RESULTS_ablation §24, suspicion 1).
    declare_omissions: bool = False

    def __post_init__(self):
        if self.pack not in PACKS:
            raise SchemaError("unknown pack: %r (want one of %s)" % (self.pack, list(PACKS)))
        if isinstance(self.evidence_hops, bool) or not isinstance(self.evidence_hops, int) \
                or self.evidence_hops < 0:
            raise SchemaError("evidence_hops must be an int >= 0, got %r" % (self.evidence_hops,))
        if isinstance(self.evidence_share, bool) or not isinstance(self.evidence_share, (int, float)) \
                or not (0.0 <= float(self.evidence_share) <= 1.0):
            raise SchemaError("evidence_share must be a number in [0, 1], got %r"
                              % (self.evidence_share,))
        if self.unit_cap_share is not None and (
                isinstance(self.unit_cap_share, bool)
                or not isinstance(self.unit_cap_share, (int, float))
                or not (0.0 < float(self.unit_cap_share) <= 1.0)):
            raise SchemaError("unit_cap_share must be None or a number in (0, 1], got %r"
                              % (self.unit_cap_share,))
        if not isinstance(self.cost_order, bool):
            raise SchemaError("cost_order must be a bool, got %r" % (self.cost_order,))
        if not isinstance(self.excerpt, bool):
            raise SchemaError("excerpt must be a bool, got %r" % (self.excerpt,))
        if not isinstance(self.declare_omissions, bool):
            raise SchemaError("declare_omissions must be a bool, got %r" % (self.declare_omissions,))
        if self.repr_policy not in REPR_POLICIES:
            raise SchemaError("unknown repr_policy: %r (want one of %s)"
                              % (self.repr_policy, list(REPR_POLICIES)))
        if self.order not in ORDERS:
            raise SchemaError("unknown order: %r (want one of %s)" % (self.order, list(ORDERS)))
        if not isinstance(self.header, str):
            raise SchemaError("header must be a str")
        if not isinstance(self.separator, str):
            raise SchemaError("separator must be a str")


# ---------------------------------------------------------------- internals


@dataclass(frozen=True)
class _Rendered:
    """One event's chosen witness: what will be written, what it costs, what to call it."""

    event_id: str
    repr_kind: str
    text: str
    token_cost: int


@dataclass(frozen=True)
class _Candidate:
    event: TraceEvent
    required: bool
    reason: str
    seed_rank: int
    pinned: bool
    #: bundle origin: (rank of the best seed this event was reached from, edge class, hops).
    #: Seeds carry (own rank, 0, 0); unreachable events carry (_NO_RANK, 9, 99).
    origin: tuple = (_NO_RANK, 9, 99)


@dataclass(frozen=True)
class _Unit:
    """An all-or-nothing packing unit: a lone event, or one atomic group (§3.4)."""

    key: str
    member_ids: tuple[str, ...]
    required: bool
    sort_key: tuple
    #: pure evidence (no seed / pin member) that outranks seeds under an evidence pack; subject
    #: to ``cfg.evidence_share``
    evidence: bool = False


class _GraphView:
    """Duck-typed read-only lookup over whatever ``graph.py`` ends up exposing.

    The assembler needs exactly one capability -- ``event_id -> TraceEvent`` -- so it accepts a
    mapping, a sequence of events, or a graph object with ``event()`` / ``get_event()`` /
    ``events_by_id`` / ``events``.  Keeping the surface this small is deliberate: it stops the
    assembler from growing a dependency on graph internals it has no business reading.
    """

    def __init__(self, graph):
        if graph is None:
            raise SchemaError("assemble() needs a graph to resolve event ids against")
        self._get = None
        self._map: dict[str, TraceEvent] | None = None
        if isinstance(graph, Mapping):
            self._map = dict(graph)
        elif hasattr(graph, "event") and callable(getattr(graph, "event")):
            self._get = graph.event
        elif hasattr(graph, "get_event") and callable(getattr(graph, "get_event")):
            self._get = graph.get_event
        elif hasattr(graph, "events_by_id"):
            self._map = self._as_map(getattr(graph, "events_by_id"))
        elif hasattr(graph, "by_id"):
            self._map = self._as_map(getattr(graph, "by_id"))
        elif hasattr(graph, "events"):
            self._map = self._as_map(getattr(graph, "events"))
        elif isinstance(graph, (list, tuple)):
            self._map = self._as_map(graph)
        else:
            raise SchemaError("graph exposes no event lookup (want mapping, .event(), "
                              ".get_event(), .events_by_id or .events); got %r" % type(graph))

    @staticmethod
    def _as_map(obj) -> dict[str, TraceEvent]:
        if callable(obj):
            obj = obj()
        if isinstance(obj, Mapping):
            return dict(obj)
        try:
            items = list(obj)
        except TypeError as exc:                       # pragma: no cover - defensive
            raise SchemaError("graph events are not iterable: %s" % exc)
        out: dict[str, TraceEvent] = {}
        for ev in items:
            if not isinstance(ev, TraceEvent):
                raise SchemaError("graph events must be TraceEvent, got %r" % type(ev))
            out[ev.event_id] = ev
        return out

    def get(self, event_id: str) -> TraceEvent:
        if self._map is not None:
            ev = self._map.get(event_id)
        else:
            try:
                ev = self._get(event_id)
            except KeyError:
                ev = None
        if ev is None:
            raise SchemaError("event not in graph: %s" % event_id)
        if not isinstance(ev, TraceEvent):
            raise SchemaError("graph returned a non-TraceEvent for %s: %r"
                              % (event_id, type(ev)))
        return ev


def _source_ref(event: TraceEvent) -> _Rendered:
    """The source witness: the explicit raw_text ref if present, else the event's own text.

    ``event.cost_of`` already implements the fallback for the number; we mirror it for the text
    so a graph that never materialised RepresentationRefs still works (§4.1: "V1's core only
    requires raw_text").
    """
    ref = event.representation(_SOURCE_KIND)
    if ref is not None and ref.text is not None:
        return _Rendered(event.event_id, _SOURCE_KIND, ref.text, ref.token_cost)
    return _Rendered(event.event_id, _SOURCE_KIND, event.text, event.token_cost)


def _carrier_ref(event: TraceEvent) -> RepresentationRef | None:
    ref = event.representation(_CARRIER_KIND)
    if ref is None or ref.text is None:
        return None
    return ref


# ---------------------------------------------------------------- the assembler


class BudgetAssembler:
    """Hard-budget group packer (§3.4) + manifest minter (§3.5)."""

    def __init__(self, cfg: AssemblerConfig = AssemblerConfig(), excerpt_fn=None):
        if not isinstance(cfg, AssemblerConfig):
            raise SchemaError("cfg must be an AssemblerConfig, got %r" % type(cfg))
        # re-validate: a frozen dataclass can still be mutated via object.__setattr__
        AssemblerConfig(cfg.repr_policy, cfg.header, cfg.separator, cfg.order, cfg.pack,
                        cfg.evidence_hops, cfg.evidence_share)
        if excerpt_fn is not None and not callable(excerpt_fn):
            raise SchemaError("excerpt_fn must be callable or None")
        self.cfg = cfg
        #: WP2: ``excerpt_fn(event, query, quoting_texts) -> [(label, text, cost), ...]`` offers
        #: query-driven line excerpts of an event that does not fit whole.  Optional and injected
        #: (see core/excerpt.py) so this module still owns no tokenizer.
        if excerpt_fn is None and cfg.excerpt and cfg.pack in _EVIDENCE_PACKS:
            from .excerpt import make_excerpt_fn          # local: keeps the import optional
            excerpt_fn = make_excerpt_fn(estimate_text_tokens)
        self.excerpt_fn = excerpt_fn

    # -------------------------------------------------------- public API

    def assemble(self, query: str, closure: EvidenceClosure, graph, budget: int, *,
                 seeds: Sequence[Seed] = (), query_mode: str = "lookup",
                 verifications: Sequence[CarrierVerification] = ()) -> MemoryPacket:
        cfg = self.cfg
        if not isinstance(query, str):
            raise SchemaError("query must be a str, got %r" % type(query))
        if not isinstance(closure, EvidenceClosure):
            raise SchemaError("closure must be an EvidenceClosure, got %r" % type(closure))
        if isinstance(budget, bool) or not isinstance(budget, int):
            raise BudgetError("budget must be an int, got %r" % type(budget))
        if budget < 0:
            raise BudgetError("negative budget: %d" % budget)
        if query_mode not in QUERY_MODES:
            raise SchemaError("unknown query_mode: %r" % (query_mode,))
        # §3.5: a packet must not be able to lie about the mode that shaped its evidence.
        if query_mode != "lookup" and query_mode != closure.query_mode:
            raise SchemaError("query_mode %r contradicts closure.query_mode %r"
                              % (query_mode, closure.query_mode))
        mode = closure.query_mode

        gv = _GraphView(graph)
        seed_list = self._check_seeds(seeds, gv)
        verifs = self._dedup_verifications(verifications)
        best_seed = self._best_seed_per_event(seed_list)

        candidates = self._candidates(closure, gv, best_seed)
        units = self._units(candidates)

        header_cost = estimate_text_tokens(cfg.header)
        header_dropped = header_cost > budget
        if header_dropped:
            # We cannot serve the header without breaking the hard budget, and we refuse to
            # truncate it.  Dropping it is a real degradation, so it is reported, not hidden.
            header_cost = 0

        remaining = budget - header_cost
        placed: dict[str, _Rendered] = {}
        missing_required: list[str] = []
        omitted_optional: list[str] = []
        # WP2: the texts of events that DEPEND ON a candidate (they quote it); the excerpt
        # selector may look at them, and at the query -- never at anything item-specific.
        quoting: dict[str, list[str]] = {}
        if self.excerpt_fn is not None:
            for st in closure.steps:
                if st.edge_type == "DEPENDS_ON" and st.child_id and st.child_id != st.parent_id:
                    try:
                        quoting.setdefault(st.parent_id, []).append(gv.get(st.child_id).text)
                    except SchemaError:
                        pass

        def record_miss(unit: _Unit) -> None:
            for eid in unit.member_ids:
                if candidates[eid].required:
                    missing_required.append(eid)
                else:
                    omitted_optional.append(eid)

        def place(unit: _Unit, rendering: Mapping[str, _Rendered]) -> int:
            for eid in unit.member_ids:
                placed[eid] = rendering[eid]
            return sum(r.token_cost for r in rendering.values())

        required_units = [u for u in units if u.required]
        optional_units = [u for u in units if not u.required]
        # No single unit may spend more than `unit_cap_share` of the budget on its FIRST attempt:
        # it is fitted under that ceiling, which makes `_fit_unit` fall to a cheaper witness (with
        # an excerpt_fn installed, a labelled excerpt) instead of taking the whole budget.  A unit
        # with nothing cheap enough is HELD and retried at the end against whatever survives, so
        # the cap never drops anything the old packer would have served at that point.
        ucap = (int(round(float(cfg.unit_cap_share) * budget))
                if cfg.pack in _EVIDENCE_PACKS and cfg.unit_cap_share is not None else None)
        _cost_cache: dict[str, int] = {}

        if cfg.cost_order and cfg.pack in _EVIDENCE_PACKS:
            # Cost enters the packing key, INSIDE the tier: the tier order (and with it contract
            # #11) is untouched, but among equally-qualified evidence the cheap units go first, so
            # one expensive unit can no longer starve the tail behind it.
            def _cost_key(u: _Unit) -> tuple:
                if u.key not in _cost_cache:
                    _cost_cache[u.key] = self._unit_min_cost(u, candidates, query, quoting)
                return (u.sort_key[0], _cost_cache[u.key]) + tuple(u.sort_key[1:])
            units = sorted(units, key=_cost_key)
            required_units = [u for u in units if u.required]
            optional_units = [u for u in units if not u.required]

        held: list[_Unit] = []
        if cfg.pack in _EVIDENCE_PACKS and cfg.evidence_share < 1.0:
            # The evidence tier may take at most `evidence_share` of the budget ahead of the
            # seeds; what does not fit under the cap is tried again after the seeds and before
            # the weak tier (tier 2 for `tiered`, 3 for `evidence_first`).
            cap = int(round(float(cfg.evidence_share) * budget))
            weak_tier = 2 if cfg.pack == "tiered" else 3
            strong_units = [u for u in required_units if u.sort_key[0] < weak_tier]
            weak_units = [u for u in required_units if u.sort_key[0] >= weak_tier]
            deferred: list[_Unit] = []
            ev_used = 0
            for unit in strong_units:
                if unit.evidence and ev_used + self._unit_min_cost(unit, candidates, query,
                                                                  quoting) > cap:
                    deferred.append(unit)
                    continue
                lim = remaining if ucap is None else min(remaining, ucap)
                rendering = self._fit_unit(unit, candidates, lim, query, quoting)
                if rendering is None:
                    if lim < remaining:
                        held.append(unit)       # nothing cheap enough; retry at the end
                    else:
                        record_miss(unit)
                    continue
                spent = place(unit, rendering)
                remaining -= spent
                if unit.evidence:
                    ev_used += spent
            order = deferred + weak_units + optional_units
        else:
            # required units first, then optional -- "drop optional first" falls out of the order
            order = required_units + optional_units
        for unit in order:
            lim = remaining if ucap is None else min(remaining, ucap)
            rendering = self._fit_unit(unit, candidates, lim, query, quoting)
            if rendering is None:
                if lim < remaining:
                    held.append(unit)
                else:
                    record_miss(unit)
                continue
            remaining -= place(unit, rendering)
        for unit in held:
            # last resort: the units nothing cheap enough could represent, against what survives
            rendering = self._fit_unit(unit, candidates, remaining, query, quoting)
            if rendering is None:
                record_miss(unit)
                continue
            remaining -= place(unit, rendering)

        entries_order = self._render_order(placed, candidates)
        entries = tuple(
            PacketEntry(event_id=eid,
                        repr_kind=placed[eid].repr_kind,
                        token_cost=placed[eid].token_cost,
                        reason=candidates[eid].reason)
            for eid in entries_order
        )
        total_tokens = header_cost + sum(e.token_cost for e in entries)
        if total_tokens > budget:                      # pragma: no cover - invariant guard
            raise BudgetError("internal packing error: %d > budget %d" % (total_tokens, budget))

        body = cfg.separator.join(placed[eid].text for eid in entries_order)
        if cfg.header and not header_dropped:
            context = cfg.header + cfg.separator + body if body else cfg.header
        else:
            context = body

        # A closure that stopped at max_hops has an UNEXPANDED FRONTIER: the last served event
        # still has a required parent nobody looked at.  Contract #3 (test_03f) caught the old
        # behaviour minting incomplete=False for exactly that packet -- a silent dangling
        # dependency, the one failure mode this whole project exists to prevent.  The frontier is
        # reported as missing_required (it IS required evidence that is not served) and the packet
        # is marked incomplete, whatever the budget allowed.
        frontier = [st.parent_id for st in closure.steps
                    if st.rule == "max_hops_truncated" and st.parent_id not in placed]
        for eid in frontier:
            if eid not in missing_required:
                missing_required.append(eid)

        if cfg.declare_omissions and missing_required:
            note = ("[tracepack-omitted] %d required record(s) did not fit this budget and are NOT "
                    "below: %s" % (len(missing_required), ", ".join(sorted(missing_required)[:8])
                                   + (", ..." if len(missing_required) > 8 else "")))
            # charged, not free: an unbudgeted note would re-introduce the §23 accounting hole
            note_cost = estimate_text_tokens(note)
            if total_tokens + note_cost <= budget:
                context = context + cfg.separator + note if context else note
                total_tokens += note_cost

        manifest = PacketManifest(
            query=query,
            query_mode=mode,
            budget=budget,
            entries=entries,
            seeds=tuple(seed_list),
            closure_steps=tuple(closure.steps),
            omitted_optional=tuple(omitted_optional),
            missing_required=tuple(missing_required),
            carrier_verifications=tuple(verifs),
            incomplete=bool(missing_required) or header_dropped,
            total_tokens=total_tokens,
        )
        return MemoryPacket(context=context, manifest=manifest)

    # -------------------------------------------------------- seeds

    @staticmethod
    def _check_seeds(seeds: Sequence[Seed], gv: _GraphView) -> list[Seed]:
        if isinstance(seeds, (str, bytes)):
            raise SchemaError("seeds must be a sequence of Seed, got a string")
        out: list[Seed] = []
        seen: set[tuple] = set()
        for s in seeds:
            if not isinstance(s, Seed):
                raise SchemaError("seeds must contain Seed objects, got %r" % type(s))
            gv.get(s.event_id)          # a seed for a non-existent event is a router bug
            key = (s.event_id, s.source, s.rank, s.pinned)
            if key in seen:
                continue
            seen.add(key)
            out.append(s)
        return out

    @staticmethod
    def _best_seed_per_event(seeds: Sequence[Seed]) -> dict[str, Seed]:
        """One representative seed per event: pinned wins, then best rank, then source name.

        An event can legitimately be seeded twice (lexical *and* dense).  The reason string and
        the priority ordering need a single answer, and it must not depend on router iteration
        order, hence this total, score-free key.
        """
        best: dict[str, Seed] = {}
        for s in seeds:
            cur = best.get(s.event_id)
            if cur is None:
                best[s.event_id] = s
                continue
            if (0 if s.pinned else 1, s.rank, s.source) < (0 if cur.pinned else 1,
                                                           cur.rank, cur.source):
                best[s.event_id] = s
        return best

    @staticmethod
    def _dedup_verifications(verifications: Sequence[CarrierVerification]
                             ) -> list[CarrierVerification]:
        out: list[CarrierVerification] = []
        seen: set[tuple] = set()
        for v in verifications:
            if not isinstance(v, CarrierVerification):
                raise SchemaError("verifications must contain CarrierVerification, got %r"
                                  % type(v))
            key = (v.carrier_id, v.predicate, v.model_id, v.construction_protocol,
                   v.readout_protocol, v.test_version)
            if key in seen:
                continue
            seen.add(key)
            out.append(v)
        return out

    # -------------------------------------------------------- candidates & units

    def _candidates(self, closure: EvidenceClosure, gv: _GraphView,
                    best_seed: Mapping[str, Seed]) -> dict[str, _Candidate]:
        """Dedup required/optional/seed ids into one ordered candidate table (requirement #4).

        Reason precedence (§3.5 wants *why*, not just *what*):
          seed:<source>  ->  closure:<edge_type>  ->  optional:<rank>  ->  closure:required
        """
        step_edge: dict[str, str] = {}
        for st in closure.steps:
            step_edge.setdefault(st.parent_id, st.edge_type)
        origins = self._origins(
            closure, best_seed,
            _EVIDENCE_CLASS if self.cfg.pack in _EVIDENCE_PACKS else _EDGE_CLASS)

        seed_ids = list(dict.fromkeys(closure.seeds))
        required_ids = list(dict.fromkeys(closure.required))
        required_set = set(required_ids)
        optional_ids = [e for e in dict.fromkeys(closure.optional) if e not in required_set]

        # optional ranking: (seed rank if any, then chronological) -- requirement #3
        opt_events = {eid: gv.get(eid) for eid in optional_ids}
        ranked_optional = sorted(
            optional_ids,
            key=lambda e: (best_seed[e].rank if e in best_seed else _NO_RANK,
                           opt_events[e].timestamp, e),
        )
        opt_rank = {eid: i for i, eid in enumerate(ranked_optional)}

        # events the closure demoted because something supersedes them (rule state_prefers_latest)
        demoted = {st.child_id for st in closure.steps if st.rule == "state_prefers_latest"} \
            & set(ranked_optional)

        out: dict[str, _Candidate] = {}
        for eid in required_ids + ranked_optional:
            if eid in out:
                continue
            ev = gv.get(eid)
            required = eid in required_set
            s = best_seed.get(eid)
            if eid in demoted:
                # `state_prefers_latest` moved this event out of required because a newer event
                # supersedes it.  Seed precedence used to overwrite that with "seed:lexical", so
                # the manifest presented a stale state exactly like fresh evidence (test_07i).
                # The demotion is the load-bearing fact here and wins the label.
                reason = "superseded:demoted"
            elif s is not None:
                reason = "seed:%s" % s.source
            elif eid in seed_ids:
                # a seed the router did not hand us a record for; still not a closure product
                reason = "seed:closure"
            elif eid in step_edge:
                reason = "closure:%s" % step_edge[eid]
            elif not required:
                reason = "optional:%d" % opt_rank[eid]
            else:
                reason = "closure:required"
            out[eid] = _Candidate(event=ev, required=required, reason=reason,
                                  seed_rank=s.rank if s is not None else _NO_RANK,
                                  pinned=bool(s is not None and s.pinned),
                                  origin=origins.get(eid, (_NO_RANK, 9, 99)))
        # A seed the closure listed in neither tier would be evidence nobody owns; that is a
        # closure bug, and swallowing it would make the packet unauditable.
        stray = [e for e in seed_ids if e not in out]
        if stray:
            raise SchemaError("closure seeds absent from required/optional: %s"
                              % sorted(stray)[:3])
        return out

    @staticmethod
    def _origins(closure: EvidenceClosure, best_seed: Mapping[str, Seed],
                 classes: Mapping[str, int] = _EDGE_CLASS) -> dict[str, tuple]:
        """Best (seed rank, edge class, hops) each closure event was reached with.

        A multi-source BFS over the recorded closure steps (child -> parent), seeded at every
        seed with its own rank.  An event reachable from two seeds keeps the lexicographically
        best tuple, so a source shared by seed #0 and seed #5 packs with seed #0's bundle and is
        never charged twice (dedup falls out of the candidate table).
        """
        children: dict[str, list[tuple[str, int]]] = {}
        for st in closure.steps:
            if st.rule == "state_prefers_latest":
                continue
            children.setdefault(st.child_id, []).append(
                (st.parent_id, classes.get(st.edge_type, 3)))
        best: dict[str, tuple] = {}
        frontier: list[tuple[tuple, str]] = []
        for sid in closure.seeds:
            s = best_seed.get(sid)
            o = (s.rank if s is not None else _NO_RANK, 0, 0)
            if o < best.get(sid, (_NO_RANK, 9, 99)):
                best[sid] = o
                frontier.append((o, sid))
        frontier.sort()
        i = 0
        while i < len(frontier):
            o, node = frontier[i]
            i += 1
            if best.get(node) != o:
                continue                       # superseded by a better origin
            for par, cls in children.get(node, ()):
                cand = (o[0], max(o[1], cls), o[2] + 1)
                if cand < best.get(par, (_NO_RANK, 9, 99)):
                    best[par] = cand
                    frontier.append((cand, par))
        return best

    def _units(self, candidates: Mapping[str, _Candidate]) -> list[_Unit]:
        """Collapse candidates into all-or-nothing units and sort them into packing order."""
        groups: dict[str, list[str]] = {}
        for eid, c in candidates.items():
            key = ("g:%s" % c.event.atomic_group) if c.event.atomic_group else ("e:%s" % eid)
            groups.setdefault(key, []).append(eid)

        units: list[_Unit] = []
        for key, members in groups.items():
            members = sorted(members, key=lambda e: (candidates[e].event.timestamp, e))
            required = any(candidates[e].required for e in members)
            pinned = any(candidates[e].pinned for e in members)
            seed_rank = min(candidates[e].seed_rank for e in members)
            first_ts = min(candidates[e].event.timestamp for e in members)
            if self.cfg.pack == "bundle":
                # pinned first (contract #5); then the BUNDLE the unit belongs to (best origin
                # seed rank, strong edges before MATERIALIZES, nearer hops first); then time.
                origin = min(candidates[e].origin for e in members)
                sort_key = (0 if pinned else 1, origin[0], origin[1], origin[2],
                            first_ts, members[0], key)
            elif self.cfg.pack in _EVIDENCE_PACKS:
                sort_key = self._evidence_key(members, candidates, pinned, seed_rank,
                                              first_ts, key)
                evidence = (sort_key[0] == 1 and not pinned and seed_rank == _NO_RANK)
                units.append(_Unit(key=key, member_ids=tuple(members), required=required,
                                   sort_key=sort_key, evidence=evidence))
                continue
            else:
                # pinned first (test contract #5), then seed rank, then chronological, then id
                sort_key = (0 if pinned else 1, seed_rank, first_ts, members[0], key)
            units.append(_Unit(key=key, member_ids=tuple(members),
                               required=required, sort_key=sort_key))
        units.sort(key=lambda u: u.sort_key)
        return units

    def _evidence_key(self, members, candidates, pinned: bool, seed_rank: int, first_ts,
                      key: str) -> tuple:
        """Packing priority of one unit under `tiered` / `evidence_first` (PLAN_phase2 WP1).

        A unit is *near strong evidence* when the best origin any member was reached with is a
        strong edge (class <= 1: DEPENDS_ON / RESULT_OF / GROUNDED_IN / ORACLE / SUPERSEDES)
        within ``cfg.evidence_hops`` of a ranked seed.  Everything else that is not a seed --
        events reached only through MATERIALIZES (the compaction summary's fan-out over the whole
        session) or with no origin at all -- packs in the weak tier, by hop then time.  That is
        the one property both packs share and contract #11 asserts: weak-reached material never
        precedes a seed.
        """
        if pinned:
            return (0, seed_rank, 0, 0, first_ts, members[0], key)
        # The unit's priority is the BEST of its members' priorities, member by member.  Phase A
        # found the unit-level version losing 12/190 gold events: a required tool_result whose
        # call was itself a mid-ranked seed inherited "seed rank 2" for the whole 2,000-token
        # pair and lost to two 96-token seeds -- exactly what the pinned ceiling (`opin`) avoids
        # by packing the evidence first.  Atomicity must not demote evidence.
        tiered = self.cfg.pack == "tiered"
        best = None
        for e in members:
            c = candidates[e]
            o_rank, o_cls, o_hop = c.origin
            is_seed = c.seed_rank != _NO_RANK
            strong = (o_rank != _NO_RANK and o_cls <= 1 and o_hop <= self.cfg.evidence_hops
                      and o_hop > 0)
            weak_hop = o_hop if o_rank != _NO_RANK else 99
            if tiered:
                if is_seed:
                    k = (1, c.seed_rank, 0, 0)
                elif strong:
                    k = (1, o_rank, o_hop, o_cls)
                else:
                    k = (2, weak_hop, 0, 0)
            else:                                   # evidence_first
                if strong:
                    k = (1, o_rank, o_hop, o_cls)
                elif is_seed:
                    k = (2, c.seed_rank, 0, 0)
                else:
                    k = (3, weak_hop, 0, 0)
            if best is None or k < best:
                best = k
        return best + (first_ts, members[0], key)

    # -------------------------------------------------------- representation & fitting

    def _unit_min_cost(self, unit: _Unit, candidates: Mapping[str, _Candidate], query: str,
                       quoting: Mapping[str, Sequence[str]]) -> int:
        """Cheapest whole-unit rendering the packer could choose (uniform tier, requirement #5)."""
        options = {eid: self._variants(candidates[eid].event, query, quoting.get(eid, ()))
                   for eid in unit.member_ids}
        tiers = max(len(v) for v in options.values())
        return min(sum(options[eid][min(t, len(options[eid]) - 1)].token_cost
                       for eid in unit.member_ids) for t in range(tiers))

    def _variants(self, event: TraceEvent, query: str = "",
                  quoting: Sequence[str] = ()) -> list[_Rendered]:
        """Representation options for one event, richest first (requirement #5).

        ``source_plus_carrier`` puts the combined witness first and the source alone second, so
        "source wins if only one fits" falls out of the ordering instead of a special case.
        """
        policy = self.cfg.repr_policy
        src = _source_ref(event)
        car = _carrier_ref(event)
        if policy == "source_only":
            out = [src]
        elif policy == "carrier_only":
            if car is None:
                out = [src]         # no carrier -> serve the source and *record* raw_text
            else:
                out = [_Rendered(event.event_id, _CARRIER_KIND, car.text, car.token_cost)]
        elif car is None:           # source_plus_carrier without a carrier
            out = [src]
        else:
            both = _Rendered(event.event_id, _BOTH_KIND,
                             src.text + self.cfg.separator + car.text,
                             src.token_cost + car.token_cost)
            out = [both, src]
        # WP2: excerpts are the LAST resort, tried only after every whole witness failed to fit.
        # Each one is labelled with its exact line provenance so the manifest (and the digest)
        # say what was served.  Only source-bearing witnesses get excerpted: a carrier is
        # already the compact form.
        if self.excerpt_fn is not None and policy != "carrier_only":
            for label, text, cost in self.excerpt_fn(event, query, tuple(quoting)) or ():
                if not isinstance(label, str) or not label.startswith(_EXCERPT_KIND):
                    raise SchemaError("excerpt_fn must label excerpts '%s[...]', got %r"
                                      % (_EXCERPT_KIND, label))
                if not isinstance(cost, int) or isinstance(cost, bool) or cost < 0:
                    raise SchemaError("excerpt cost must be a non-negative int, got %r" % (cost,))
                if cost >= src.token_cost:
                    continue        # an excerpt no cheaper than the whole is not an excerpt
                out.append(_Rendered(event.event_id, label, text, cost))
        return out

    def _fit_unit(self, unit: _Unit, candidates: Mapping[str, _Candidate],
                  remaining: int, query: str = "",
                  quoting: Mapping[str, Sequence[str]] | None = None) -> dict[str, _Rendered] | None:
        """All-or-nothing fit of one unit; uniform representation tier across its members.

        Returns the per-member rendering, or ``None`` when even the cheapest tier does not fit.
        Never truncates silently: the only levers are *which whole witness* to serve and, when
        an ``excerpt_fn`` is installed, a labelled excerpt as the last tier.
        """
        quoting = quoting or {}
        options = {eid: self._variants(candidates[eid].event, query, quoting.get(eid, ()))
                   for eid in unit.member_ids}
        tiers = max(len(v) for v in options.values())
        for tier in range(tiers):
            rendering = {eid: options[eid][min(tier, len(options[eid]) - 1)]
                         for eid in unit.member_ids}
            if sum(r.token_cost for r in rendering.values()) <= remaining:
                return rendering
        return None

    # -------------------------------------------------------- rendering order

    def _render_order(self, placed: Mapping[str, _Rendered],
                      candidates: Mapping[str, _Candidate]) -> tuple[str, ...]:
        if self.cfg.order == "seed_first":
            def key(eid: str):
                c = candidates[eid]
                return (0 if c.seed_rank != _NO_RANK else 1, 0 if c.pinned else 1,
                        c.seed_rank, c.event.timestamp, eid)
        else:
            def key(eid: str):
                c = candidates[eid]
                return (c.event.timestamp, eid)
        return tuple(sorted(placed, key=key))


# ---------------------------------------------------------------- selfcheck


def _mk_event(eid, ts, text, cost, *, group=None, kind="assistant", carrier=None):
    reps = [RepresentationRef(kind="raw_text", token_cost=cost, text=text)]
    if carrier is not None:
        ctext, ccost = carrier
        reps.append(RepresentationRef(kind="materialized_text", token_cost=ccost, text=ctext,
                                      predicate="summary_of", model_id="m1", protocol_id="p1"))
    return TraceEvent(event_id=eid, kind=kind, text=text, timestamp=ts, token_cost=cost,
                      representations=tuple(reps), atomic_group=group)


def _fixture():
    """Tiny synthetic graph: a pinned decision, an atomic tool_call/tool_result pair, a source
    with a carrier, plus three optional neighbours."""
    events = [
        _mk_event("d1", 10, "DECISION: rejected supplier X", 6),
        _mk_event("tc1", 4, "TOOL_CALL check_supplier(X)", 5, group="g_tool", kind="tool_call"),
        _mk_event("tr1", 5, "TOOL_RESULT status=blacklisted", 7, group="g_tool",
                  kind="tool_result"),
        _mk_event("s1", 2, "SOURCE: supplier registry row for X, long verbatim payload", 12,
                  carrier=("CARRIER: X is blacklisted", 4)),
        _mk_event("o1", 7, "note one", 3),
        _mk_event("o2", 8, "note two", 4),
        _mk_event("o3", 9, "note three, a bit longer", 9),
    ]
    graph = {e.event_id: e for e in events}
    seeds = (
        Seed(event_id="d1", score=9.0, source="pin", rank=0, pinned=True),
        Seed(event_id="tr1", score=3.0, source="lexical", rank=1),
        Seed(event_id="o2", score=1.0, source="dense", rank=2),
    )
    closure = EvidenceClosure(
        seeds=("d1", "tr1", "o2"),
        required=("d1", "tr1", "tc1", "s1"),
        optional=("o1", "o2", "o3"),
        steps=(
            ClosureStep(child_id="tr1", parent_id="tc1", edge_type="RESULT_OF", rule="tool_pair"),
            ClosureStep(child_id="d1", parent_id="s1", edge_type="GROUNDED_IN", rule="why_source"),
        ),
        query_mode="why",
    )
    return graph, closure, seeds


def _selfcheck():
    import random

    graph, closure, seeds = _fixture()
    all_ids = set(graph)
    verifs = (CarrierVerification(carrier_id="s1", predicate="summary_of", model_id="m1",
                                  construction_protocol="c1", readout_protocol="r1",
                                  test_version="v1", ci_low=0.81, ci_high=0.93),)

    # ---- 1. budget never violated, across 200 seeded randomized budgets & configs ----------
    rng = random.Random(20260901)
    configs = [AssemblerConfig(repr_policy=p, header=h, order=o)
               for p in REPR_POLICIES
               for h in ("", "CONTEXT FOR THE READER:")
               for o in ORDERS]
    for i in range(200):
        budget = rng.randint(0, 90)
        cfg = configs[rng.randrange(len(configs))]
        pkt = BudgetAssembler(cfg).assemble("why was supplier X rejected?", closure, graph,
                                            budget, seeds=seeds, query_mode="why",
                                            verifications=verifs)
        m = pkt.manifest
        assert m.total_tokens <= budget, (i, budget, m.total_tokens)
        header_cost = estimate_text_tokens(cfg.header)
        header_dropped = header_cost > budget
        if header_dropped:
            header_cost = 0
        assert m.total_tokens == header_cost + sum(e.token_cost for e in m.entries)

        ids = [e.event_id for e in m.entries]
        # ---- 4. dedup: every event at most once ------------------------------------------
        assert len(ids) == len(set(ids)), (i, ids)
        assert set(ids) <= all_ids
        # ---- 2. atomic group is never split ----------------------------------------------
        assert ({"tc1", "tr1"} & set(ids)) in (set(), {"tc1", "tr1"}), (i, budget, ids)
        # ---- 1b. no silent truncation: every served witness is whole ----------------------
        for e in m.entries:
            ev = graph[e.event_id]
            src = _source_ref(ev)
            car = _carrier_ref(ev)
            allowed = {src.text}
            if car is not None:
                allowed.add(car.text)
                allowed.add(src.text + cfg.separator + car.text)
            assert any(a in pkt.context for a in allowed), (i, e.event_id)
        # ---- accounting: served / missing / omitted exactly partition the candidates -------
        assert set(m.missing_required).isdisjoint(ids)
        assert set(m.omitted_optional).isdisjoint(ids)
        assert set(ids) | set(m.missing_required) | set(m.omitted_optional) == all_ids
        # ---- 3. a complete packet always contains the pinned seed --------------------------
        if not m.incomplete:
            assert "d1" in ids, (i, budget, ids)
        # ---- incomplete iff required is missing (or the header had to be dropped) ----------
        assert m.incomplete == (bool(m.missing_required) or header_dropped)

    # ---- 5. incomplete flag when the required set cannot fit -------------------------------
    tight = BudgetAssembler().assemble("q", closure, graph, 5, seeds=seeds, query_mode="why")
    assert tight.incomplete is True
    assert tight.manifest.missing_required, tight.manifest.missing_required
    assert tight.manifest.total_tokens <= 5
    assert not (set(tight.manifest.omitted_optional) & set(tight.event_ids))

    # generous budget -> complete, nothing missing
    full = BudgetAssembler().assemble("q", closure, graph, 10_000, seeds=seeds, query_mode="why")
    assert full.incomplete is False and not full.manifest.missing_required
    assert set(full.event_ids) == all_ids
    assert full.manifest.total_tokens == sum(graph[e].token_cost for e in all_ids)
    # ---- 6. inclusion reasons --------------------------------------------------------------
    reasons = {e.event_id: e.reason for e in full.manifest.entries}
    assert reasons["d1"] == "seed:pin", reasons
    assert reasons["tr1"] == "seed:lexical", reasons
    assert reasons["tc1"] == "closure:RESULT_OF", reasons
    assert reasons["s1"] == "closure:GROUNDED_IN", reasons
    assert reasons["o1"].startswith("optional:") and reasons["o3"].startswith("optional:")
    assert full.manifest.entries[0].event_id == "s1"   # chronological render (ts=2)

    seed_first = BudgetAssembler(AssemblerConfig(order="seed_first")).assemble(
        "q", closure, graph, 10_000, seeds=seeds, query_mode="why")
    assert seed_first.event_ids[0] == "d1"             # pinned seed leads
    assert set(seed_first.event_ids) == set(full.event_ids)

    # ---- determinism of the digest and of the context --------------------------------------
    a = BudgetAssembler().assemble("q", closure, graph, 40, seeds=seeds, query_mode="why",
                                   verifications=verifs)
    b = BudgetAssembler().assemble("q", closure, graph, 40, seeds=seeds, query_mode="why",
                                   verifications=verifs)
    assert a.manifest.digest() == b.manifest.digest()
    assert a.context == b.context
    # reordering the router's seed list must not change what is served
    c = BudgetAssembler().assemble("q", closure, graph, 40, seeds=tuple(reversed(seeds)),
                                   query_mode="why", verifications=verifs)
    assert c.event_ids == a.event_ids and c.context == a.context
    # a different budget must change the digest (the hash is not degenerate)
    d = BudgetAssembler().assemble("q", closure, graph, 41, seeds=seeds, query_mode="why")
    assert d.manifest.digest() != a.manifest.digest()

    # ---- 5b. representation policies -------------------------------------------------------
    car_only = BudgetAssembler(AssemblerConfig(repr_policy="carrier_only")).assemble(
        "q", closure, graph, 10_000, seeds=seeds, query_mode="why")
    kinds = {e.event_id: e.repr_kind for e in car_only.manifest.entries}
    costs = {e.event_id: e.token_cost for e in car_only.manifest.entries}
    assert kinds["s1"] == "materialized_text" and costs["s1"] == 4   # carrier where one exists
    assert kinds["d1"] == "raw_text" and costs["d1"] == 6            # recorded fallback

    both = BudgetAssembler(AssemblerConfig(repr_policy="source_plus_carrier")).assemble(
        "q", closure, graph, 10_000, seeds=seeds, query_mode="why")
    kb = {e.event_id: e.repr_kind for e in both.manifest.entries}
    cb = {e.event_id: e.token_cost for e in both.manifest.entries}
    assert kb["s1"] == "raw_text+materialized_text" and cb["s1"] == 16
    # source wins when only one of the two fits
    solo = EvidenceClosure(seeds=(), required=("s1",), optional=(), steps=(),
                           query_mode="lookup")
    spc = BudgetAssembler(AssemblerConfig(repr_policy="source_plus_carrier"))
    only_src = spc.assemble("q", solo, graph, 12, seeds=(), query_mode="lookup")
    assert only_src.manifest.entries[0].repr_kind == "raw_text"
    assert only_src.manifest.entries[0].token_cost == 12 and not only_src.incomplete
    assert spc.assemble("q", solo, graph, 16, seeds=(), query_mode="lookup"
                        ).manifest.entries[0].repr_kind == "raw_text+materialized_text"
    # and below even the source cost, nothing is truncated -- it is declared missing
    starved = spc.assemble("q", solo, graph, 11, seeds=(), query_mode="lookup")
    assert starved.event_ids == () and starved.manifest.missing_required == ("s1",)
    assert starved.context == ""

    # ---- 4b. dedup across duplicated / overlapping closure tiers ---------------------------
    dup = EvidenceClosure(seeds=("d1", "d1"), required=("d1", "d1", "s1"), optional=("o1", "o1"),
                          steps=closure.steps, query_mode="why")
    dpk = BudgetAssembler().assemble("q", dup, graph, 10_000,
                                     seeds=(seeds[0], seeds[0]), query_mode="why")
    assert len(dpk.event_ids) == len(set(dpk.event_ids)) == 3, dpk.event_ids
    assert len(dpk.manifest.seeds) == 1                # duplicate Seed records collapsed

    # ---- 2b. atomicity is real: a budget that fits tr1 but not the pair drops both ----------
    pair_only = EvidenceClosure(seeds=("tr1",), required=("tc1", "tr1"), optional=(),
                                steps=closure.steps[:1], query_mode="lookup")
    p1 = BudgetAssembler().assemble("q", pair_only, graph, 7, seeds=(), query_mode="lookup")
    assert p1.event_ids == () and p1.incomplete is True
    assert sorted(p1.manifest.missing_required) == ["tc1", "tr1"]
    p2 = BudgetAssembler().assemble("q", pair_only, graph, 12, seeds=(), query_mode="lookup")
    assert sorted(p2.event_ids) == ["tc1", "tr1"] and p2.incomplete is False

    # ---- header accounting ------------------------------------------------------------------
    hcfg = AssemblerConfig(header="HEADER")
    hp = BudgetAssembler(hcfg).assemble("q", pair_only, graph, 14, seeds=(), query_mode="lookup")
    assert hp.manifest.total_tokens == estimate_text_tokens("HEADER") + 12
    assert hp.context.startswith("HEADER" + hcfg.separator)
    # header alone over budget -> dropped, declared incomplete, budget still honoured
    hp2 = BudgetAssembler(AssemblerConfig(header="H" * 60)).assemble(
        "q", pair_only, graph, 3, seeds=(), query_mode="lookup")
    assert hp2.incomplete is True and hp2.context == "" and hp2.manifest.total_tokens <= 3

    # ---- graph duck-typing: a bare list of events works as well as a mapping ----------------
    lst = BudgetAssembler().assemble("q", pair_only, list(graph.values()), 12, seeds=(),
                                     query_mode="lookup")
    assert lst.context == p2.context

    # ---- FAULT INJECTION: broken inputs must be REJECTED, not silently packed ---------------
    def rejects(exc, fn, *a, **k):
        try:
            fn(*a, **k)
        except exc:
            return True
        raise AssertionError("expected %s from %r %r" % (exc.__name__, fn, a[:2]))

    asm = BudgetAssembler()
    ghost = EvidenceClosure(seeds=(), required=("d1", "NO_SUCH_EVENT"), optional=(),
                            steps=(), query_mode="lookup")
    rejects(SchemaError, asm.assemble, "q", ghost, graph, 100)                  # dangling id
    rejects(BudgetError, asm.assemble, "q", closure, graph, -1, seeds=seeds, query_mode="why")
    rejects(BudgetError, asm.assemble, "q", closure, graph, 3.5, seeds=seeds, query_mode="why")
    rejects(SchemaError, asm.assemble, "q", closure, None, 100)                 # no graph
    rejects(SchemaError, asm.assemble, "q", closure, object(), 100)             # unusable graph
    rejects(SchemaError, asm.assemble, "q", "not-a-closure", graph, 100)
    rejects(SchemaError, asm.assemble, 42, closure, graph, 100)                 # non-str query
    rejects(SchemaError, asm.assemble, "q", closure, graph, 100, query_mode="banana")
    rejects(SchemaError, asm.assemble, "q", closure, graph, 100, seeds=seeds,
            query_mode="audit")                                          # mode contradiction
    rejects(SchemaError, asm.assemble, "q", closure, graph, 100, seeds=("d1",))  # not a Seed
    rejects(SchemaError, asm.assemble, "q", closure, graph, 100,
            seeds=(Seed(event_id="ghost", score=1.0, source="pin", rank=0, pinned=True),))
    rejects(SchemaError, asm.assemble, "q", closure, graph, 100, seeds=seeds, query_mode="why",
            verifications=("not-a-verification",))
    rejects(SchemaError, AssemblerConfig, "carrier_first")                      # bad policy
    rejects(SchemaError, AssemblerConfig, "source_only", "", "\n\n", "random")  # bad order
    rejects(SchemaError, BudgetAssembler, "source_only")                        # bad cfg type
    # a seed the closure never placed in either tier is a closure bug, not a silent drop
    stray = EvidenceClosure(seeds=("o3",), required=("d1",), optional=(), steps=(),
                            query_mode="lookup")
    rejects(SchemaError, asm.assemble, "q", stray, graph, 100)

    print("assembler selfcheck OK: 200 randomized budgets, 0 violations; atomicity, dedup, "
          "determinism, incompleteness and 17 fault injections all asserted")


if __name__ == "__main__":
    _selfcheck()
