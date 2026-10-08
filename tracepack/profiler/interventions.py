"""tracepack.profiler.interventions -- the counterfactual arms (proposal §5.2, §5.3).

One arm = one *served context* for one item.  Everything in this file exists to make the
difference between two arms mean exactly one thing, which is the only reason a difference in
reader accuracy can be read as an attribution (§5.4).

Design decisions the runner and the attribution module depend on
================================================================

1. **An arm is a (plan, content) pair, and the two are separable.**
   The *plan* is which events are served, in what order, at what token cost -- produced by the
   real control plane (router -> :class:`TypedClosure` -> :class:`BudgetAssembler`).  The
   *content* is what text sits in each served slot.  A-arms vary the plan (that is what
   `routing_miss` / `closure_miss` measure).  B-arms **share one frozen plan** and vary only the
   content (that is what `carrier_loss` measures).  Conflating the two is the classic way to
   "prove" a carrier result that is really a budget result.

2. **[CONTRACT] Frozen routing is proved, not promised (§5.3 rule 1).**
   Every B-arm returns the *same* :class:`PacketManifest` -- same event ids, same order, same
   per-entry token cost, same budget -- so ``manifest.digest()`` is byte-identical across
   ``B1..B4``.  :func:`assert_frozen_routing` checks it and raises.  ``A0_shuffled`` is frozen
   against ``A2`` the same way.  A digest mismatch is a bug in this file, never a finding.

3. **[CONTRACT] Removal is equal-length replacement, never deletion (§5.3 rule 2).**
   A suppressed slot keeps its position and its character count; the text is overwritten with a
   deterministic filler.  Deleting it would hand the arm free budget and turn "carrier is worse"
   into "carrier had less competition".  The invariant is asserted per slot
   (``notes["equal_length_ok"]``) and every arm of one item therefore renders a context with the
   *same character length*.

4. **[CONTRACT] Mutation uses a shape-preserving nonce (§5.3 rules 3, 6).**
   ``B4_mutated_source`` replaces the gold value in the *source* slots with a same-length,
   same-shape surrogate that appears nowhere in the served context.  Same length keeps rule 2
   intact; nonce-ness keeps parametric knowledge from answering.  Because a text-only replay
   cannot *re-generate* a carrier, the carrier keeps quoting the old value: those carriers are
   reported in ``notes["stale_carriers"]`` and the arm is explicitly labelled a §5.3-rule-4
   *stale test*, not a source->carrier causal-transmission test.  When the gold value is not
   present in the served source at all the arm is vacuous, and says so
   (``notes["mutation_vacuous"]``) instead of silently reporting a null result.

5. **Surface leak is a flag, not an exception (§5.3 rule 5).**
   A carrier that literally contains the gold answer makes B2 trivially solvable.  We do not
   refuse to build the arm (the dataset builder screens for this, and the profiler must be able
   to *measure* the leak rate); we flag ``notes["surface_leak"]`` and name the leaking carriers,
   and the attribution module uses it to separate `carrier_loss` from `reader_miss`.

6. **Carriers are events, not representations.**  The Claude Code adapter emits compaction
   summaries as ``kind="summary"`` events with ``MATERIALIZES`` edges; it never mints
   ``materialized_text`` representations.  So the source/carrier split here is over *events*
   (``item["carrier_ids"]`` | ``kind in {summary, decision}`` | src of a ``MATERIALIZES`` edge),
   and the assembler's ``repr_policy`` axis is left available for graphs that do carry a
   ``materialized_text`` witness.  ``ArmSpec.repr_policy`` names which *class of events* keeps
   real content in a B-arm, and is passed through to the assembler for A-arms.

7. **Deterministic.**  The only randomness is donor choice and nonce minting; both take an
   explicit ``rng_seed`` and are additionally salted with a stable SHA-256 of the item id (never
   ``hash()``, which is per-process salted for strings).

Implements: §5.2 (the nine arms), §5.3 (every carrier-counterfactual rule a text-only replay can
enforce).

Not implemented on purpose
--------------------------
* §5.3 rule 3 "mutate *before writing*, then regenerate the downstream event/KV" needs a
  generator in the loop.  This module does the text-only replay the task specifies and labels it
  as such (``notes["replay"] == "text_only"``); it never claims causal transmission.
* §5.3 rule 7 (KV swap probe) needs a model, so ``probe_positive`` is an *injected* flag (see
  runner.py), with "the carrier literally contains the answer" as a documented weak fallback.
"""
from __future__ import annotations

import hashlib
import random
import re
from dataclasses import dataclass, field
from typing import Mapping, Sequence

try:                                    # normal package import
    from ..core.assembler import REPR_POLICIES, AssemblerConfig, BudgetAssembler
    from ..core.closure import CLOSURE_MODES, ClosureConfig, TypedClosure
    from ..core.router import RouterConfig, make_router
    from ..core.schema import (
        QUERY_MODES,
        EvidenceClosure,
        MemoryPacket,
        PacketManifest,
        SchemaError,
        Seed,
        TracePackError,
    )
except ImportError:                     # pragma: no cover - direct execution
    import os
    import sys

    sys.path.insert(
        0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))
    from tracepack.core.assembler import (  # type: ignore[no-redef]
        REPR_POLICIES, AssemblerConfig, BudgetAssembler)
    from tracepack.core.closure import (  # type: ignore[no-redef]
        CLOSURE_MODES, ClosureConfig, TypedClosure)
    from tracepack.core.router import RouterConfig, make_router  # type: ignore[no-redef]
    from tracepack.core.schema import (  # type: ignore[no-redef]
        QUERY_MODES, EvidenceClosure, MemoryPacket, PacketManifest, SchemaError, Seed,
        TracePackError)

__all__ = [
    "ProfilerError",
    "ArmSpec",
    "ArmResult",
    "ARMS",
    "ARM_NAMES",
    "DEFAULT_ARMS",
    "SEPARATOR",
    "CARRIER_KINDS",
    "build_arm_context",
    "assert_frozen_routing",
    "carrier_event_ids",
    "contains_value",
    "default_query",
    "equal_length_filler",
    "make_nonce",
]


class ProfilerError(TracePackError):
    """Raised when an intervention cannot be built *honestly*.

    Never raised because an arm produced a bad answer -- that is a finding.  Only for contract
    violations: a broken frozen plan, an unequal-length replacement, a missing annotation the
    arm is defined by.
    """


#: the assembler separator this module assumes; slot reconstruction is verified against it
SEPARATOR = "\n\n"

#: event kinds that are carriers by construction (compaction summaries, written decisions)
CARRIER_KINDS = ("summary", "decision")

#: which slot class keeps its real text
KEEP_MODES = ("both", "source", "carrier")
#: where an arm's seeds come from
SEED_SOURCES = ("none", "router", "gold_seed", "gold_minimal")
#: content interventions
MUTATIONS = (None, "shuffle", "source_value")
#: which serving plan an arm renders
PLAN_KINDS = ("none", "own", "a2", "b")

_FILLER_CHAR = "."


# ---------------------------------------------------------------- small text helpers


# Red-team finding #3 (2026-09-01): this module used to define its own boundary matcher with "/"
# in the boundary class.  The dataset's path values are SUFFIXES of absolute paths, so that
# matcher never fired on 68% of the path golds -- gold-in-context checks, redaction arms and
# stale-conflict labels all read the wrong answer at runtime while the builder saw them as
# present.  ONE definition now, in tracepack.core.textmatch, shared by builder and profiler.
from tracepack.core.textmatch import contains_value, wb as _wb  # noqa: E402,F401


def equal_length_filler(original: str, tag: str = "REDACTED") -> str:
    """Deterministic placeholder with **exactly** ``len(original)`` characters (§5.3 rule 2).

    The tag is included so a human reading a dumped context can see which arm blanked the slot;
    when the slot is shorter than the tag the filler degrades to plain padding rather than
    silently ending up a different length.
    """
    n = len(original)
    if n <= 0:
        return ""
    core = "[%s]" % tag
    if n < len(core):
        return _FILLER_CHAR * n
    return core + _FILLER_CHAR * (n - len(core))


def _stable_salt(*parts: str) -> int:
    blob = "\x00".join(p or "" for p in parts).encode("utf-8")
    return int(hashlib.sha256(blob).hexdigest()[:12], 16)


# ---------------------------------------------------------------- nonce minting (§5.3 rule 6)

_DIGIT_RUN = re.compile(r"\d+")
_HEX = "0123456789abcdef"
_LOWER = "abcdefghijklmnopqrstuvwxyz"
_UPPER = "ABCDEFGHIJKLMNOPQRSTUVWXYZ"


def _shuffle_digits(value: str, rng: random.Random) -> str:
    """Randomise every digit run, preserving length and leading-zero-ness."""

    def one(m):
        run = m.group(0)
        out = []
        for i, ch in enumerate(run):
            if i == 0 and ch == "0":
                out.append("0")                       # keep a leading zero a leading zero
            elif i == 0 and len(run) > 1:
                out.append(rng.choice("123456789"))   # do not shorten the number
            else:
                out.append(rng.choice("0123456789"))
        return "".join(out)

    return _DIGIT_RUN.sub(one, value)


def _shuffle_hex(value: str, rng: random.Random) -> str:
    return "".join(rng.choice(_HEX) if ch in _HEX else ch for ch in value)


def _shuffle_stem(value: str, rng: random.Random) -> str:
    """Randomise the basename stem's alphanumerics, keeping directories and extension."""
    head, sep, base = value.rpartition("/")
    stem, dot, ext = base.rpartition(".")
    if not dot:                       # no extension: the whole basename is the stem
        stem, ext = base, ""
    out = []
    for ch in stem:
        if ch.isdigit():
            out.append(rng.choice("0123456789"))
        elif ch.isalpha() and ord(ch) < 128:
            out.append(rng.choice(_LOWER if ch.islower() else _UPPER))
        else:
            out.append(ch)
    new_base = "".join(out) + (("." + ext) if ext else "")
    return head + sep + new_base


def make_nonce(value: str, value_type: str, rng: random.Random, *,
               forbidden: Sequence = (), max_tries: int = 64) -> str:
    """A same-length, same-shape surrogate for ``value`` absent from every ``forbidden`` text.

    Shape preservation is what keeps §5.3 rule 2 true through a mutation: a same-length
    replacement can neither free nor consume budget.  Nonce-ness (rule 6) is what stops the
    reader answering from parametric knowledge -- the value is a random draw, not a real
    identifier.  ``number``/``version``/``fileline`` randomise digit runs (so the surrogate is
    still a legal value of that type), ``hash`` randomises hex digits, ``path`` randomises the
    basename stem and keeps the extension.

    Raises ``ProfilerError`` rather than returning a colliding or wrong-length value: a mutation
    arm that silently reuses the old value measures nothing.
    """
    if not isinstance(value, str) or not value:
        raise ProfilerError("make_nonce needs a non-empty value, got %r" % (value,))
    kind = (value_type or "").strip().lower()
    if kind == "hash":
        gen = _shuffle_hex
    elif kind in ("number", "version", "fileline"):
        gen = _shuffle_digits
    elif kind == "path":
        gen = _shuffle_stem
    else:                              # unknown type: safest generic shape-preserving draw
        gen = _shuffle_stem if ("/" in value or "." in value) else _shuffle_digits
    for _ in range(max_tries):
        cand = gen(value, rng)
        if cand == value or len(cand) != len(value):
            continue
        if any(contains_value(t, cand) for t in forbidden):
            continue
        return cand
    raise ProfilerError(
        "could not mint a %r nonce for %r that is absent from the served context (tried %d "
        "times); the arm would be unmeasurable" % (kind, value, max_tries))


# ---------------------------------------------------------------- item helpers


def _item_id(item: Mapping) -> str:
    iid = item.get("item_id") or item.get("id")
    if not iid:
        raise ProfilerError("item has no item_id; the profiler keys checkpoints and the "
                            "per-item RNG salt on it")
    return str(iid)


def default_query(item: Mapping) -> str:
    """The item's question.

    The eval harness is expected to supply ``item["query"]`` (or ``["question"]``).  When it does
    not we synthesise a deterministic one from the annotations rather than inventing wording per
    run -- a query that changes between runs changes the router's seeds and therefore silently
    changes every A-arm.  The fallback deliberately carries the explicit reference of the
    ``explicit_ref`` slice, because that slice exists to test pinning (§4.4).
    """
    q = item.get("query") or item.get("question")
    if q:
        if not isinstance(q, str):
            raise ProfilerError("item query must be a str, got %r" % (type(q).__name__,))
        return q
    word = {"path": "file path", "fileline": "file:line location", "hash": "hash",
            "number": "number", "version": "version string"}.get(item.get("type"), "value")
    ref = item.get("explicit_ref") or {}
    bits = []
    if isinstance(ref, Mapping):
        if ref.get("step_id"):
            bits.append("step %s" % ref["step_id"])
        if ref.get("tool_call_id"):
            bits.append("tool call %s" % ref["tool_call_id"])
    tail = (" for " + " ".join(bits)) if bits else ""
    return "What exact %s does this trace record%s?" % (word, tail)


def _query_mode(item: Mapping) -> str:
    mode = item.get("query_mode", "lookup")
    if mode not in QUERY_MODES:
        raise ProfilerError("item query_mode %r is not one of %s" % (mode, list(QUERY_MODES)))
    return mode


def carrier_event_ids(graph, item=None) -> frozenset:
    """Ids of the events acting as *carriers* for this item (§3.1 / §4.5).

    Three sources, unioned: the item's annotation, the compaction/decision event kinds, and the
    src side of any ``MATERIALIZES`` edge.  Everything else served is a *source*.
    """
    out = set()
    for ed in getattr(graph, "edges", ()) or ():
        if getattr(ed, "edge_type", None) == "MATERIALIZES":
            out.add(ed.src_id)
    for ev in getattr(graph, "events", ()) or ():
        if getattr(ev, "kind", None) in CARRIER_KINDS:
            out.add(ev.event_id)
    if item:
        for cid in item.get("carrier_ids") or ():
            if graph.has(cid):
                out.add(cid)
    return frozenset(out)


# ---------------------------------------------------------------- arm specification


@dataclass(frozen=True)
class ArmSpec:
    """One experimental arm (§5.2).

    ``router`` is a name from :data:`tracepack.core.router.ROUTER_NAMES`, or any object with a
    ``retrieve(query, graph, k)`` method (so a stub / oracle router can be injected), or ``None``
    for arms that do not route.  ``repr_policy`` is passed to the assembler for A-arms; for
    B-arms it names which class of events keeps its real text (module docstring, decision 6).
    ``plan`` selects the serving plan: ``own`` (compute it), ``a2`` (freeze against
    ``A2_seed_closure``), ``b`` (the shared frozen source+carrier plan), ``none`` (serve nothing).
    """

    name: str
    router: object
    closure_mode: str
    repr_policy: str
    seed_source: str
    mutate: object          # None | "shuffle" | "source_value"
    plan: str = "own"

    def __post_init__(self):
        if not isinstance(self.name, str) or not self.name:
            raise SchemaError("ArmSpec.name must be a non-empty str")
        if self.closure_mode not in CLOSURE_MODES:
            raise SchemaError("ArmSpec.closure_mode %r not in %s"
                              % (self.closure_mode, list(CLOSURE_MODES)))
        if self.repr_policy not in REPR_POLICIES:
            raise SchemaError("ArmSpec.repr_policy %r not in %s"
                              % (self.repr_policy, list(REPR_POLICIES)))
        if self.seed_source not in SEED_SOURCES:
            raise SchemaError("ArmSpec.seed_source %r not in %s"
                              % (self.seed_source, list(SEED_SOURCES)))
        if self.mutate not in MUTATIONS:
            raise SchemaError("ArmSpec.mutate %r not in %s" % (self.mutate, list(MUTATIONS)))
        if self.plan not in PLAN_KINDS:
            raise SchemaError("ArmSpec.plan %r not in %s" % (self.plan, list(PLAN_KINDS)))
        if self.seed_source == "router" and self.router is None:
            raise SchemaError("ArmSpec %r routes but has no router" % (self.name,))
        if self.plan == "none" and self.mutate is not None:
            raise SchemaError("ArmSpec %r has nothing to mutate (plan='none')" % (self.name,))

    @property
    def keep(self) -> str:
        """Which slot class keeps its real text, derived from ``repr_policy`` (decision 6)."""
        return {"source_only": "source", "carrier_only": "carrier",
                "source_plus_carrier": "both"}[self.repr_policy]


@dataclass(frozen=True)
class ArmResult:
    """What one arm served, plus every flag §5.3 requires us to expose."""

    arm: str
    context: str
    packet: MemoryPacket
    manifest: PacketManifest
    notes: Mapping = field(default_factory=dict)

    @property
    def frozen_key(self) -> str:
        """Digest of the *serving plan*; identical across arms that froze routing together."""
        return self.manifest.digest()

    def stats(self) -> dict:
        """JSON-serialisable per-arm packet stats -- the input of ``attribution.attribute``."""
        m = self.manifest
        out = {
            "arm": self.arm,
            "n_entries": len(m.entries),
            "total_tokens": m.total_tokens,
            "budget": m.budget,
            "incomplete": bool(m.incomplete),
            "missing_required": list(m.missing_required),
            "omitted_optional": list(m.omitted_optional),
            "frozen_key": self.frozen_key,
            "context_chars": len(self.context),
            "context_hash": hashlib.sha256(self.context.encode("utf-8")).hexdigest()[:16],
        }
        out.update({k: v for k, v in self.notes.items() if k != "item_overrides"})
        return out


#: the §5.2 arms.  A0 is split into its two halves, as the failure taxonomy needs both floors.
ARMS = {
    # ---- A ladder: the plan varies, content is always the real text ------------------------
    "A0_null": ArmSpec("A0_null", None, "off", "source_only", "none", None, plan="none"),
    "A0_shuffled": ArmSpec("A0_shuffled", None, "off", "source_only", "none", "shuffle",
                           plan="a2"),
    "A1_seed_only": ArmSpec("A1_seed_only", "hybrid_pin", "off", "source_only", "router", None),
    "A2_seed_closure": ArmSpec("A2_seed_closure", "hybrid_pin", "native", "source_only",
                               "router", None),
    "A3_oracle_seed": ArmSpec("A3_oracle_seed", None, "native", "source_only", "gold_seed", None),
    "A4_gold_minimal": ArmSpec("A4_gold_minimal", None, "off", "source_only", "gold_minimal",
                               None),
    # ---- B axis: ONE frozen plan, only the content varies (§5.3 rule 1) ---------------------
    "B1_source_only": ArmSpec("B1_source_only", "hybrid_pin", "native", "source_only",
                              "router", None, plan="b"),
    "B2_carrier_only": ArmSpec("B2_carrier_only", "hybrid_pin", "native", "carrier_only",
                               "router", None, plan="b"),
    "B3_source_carrier": ArmSpec("B3_source_carrier", "hybrid_pin", "native",
                                 "source_plus_carrier", "router", None, plan="b"),
    "B4_mutated_source": ArmSpec("B4_mutated_source", "hybrid_pin", "native",
                                 "source_plus_carrier", "router", "source_value", plan="b"),
}

ARM_NAMES = tuple(ARMS)
DEFAULT_ARMS = ARM_NAMES


# ---------------------------------------------------------------- serving plans


@dataclass(frozen=True)
class _Slot:
    """One served entry: where it sits, what it costs, and whether it is source or carrier."""

    index: int
    event_id: str
    kind: str
    role: str            # "source" | "carrier"
    repr_kind: str
    text: str
    token_cost: int


@dataclass(frozen=True)
class _Plan:
    key: str
    packet: MemoryPacket
    slots: tuple
    seeds: tuple
    closure: EvidenceClosure


def _resolve_router(router, router_cfg: RouterConfig):
    if router is None:
        raise ProfilerError("arm needs a router but none was configured")
    if isinstance(router, str):
        return make_router(router, router_cfg)
    if hasattr(router, "retrieve") and callable(router.retrieve):
        return router
    raise ProfilerError("router must be a name or expose .retrieve(query, graph, k), got %r"
                        % (type(router).__name__,))


def _seeds_for(arm: ArmSpec, item: Mapping, graph, query: str, router_cfg: RouterConfig):
    if arm.seed_source == "none":
        return []
    if arm.seed_source == "router":
        return list(_resolve_router(arm.router, router_cfg).retrieve(query, graph, router_cfg.k))
    if arm.seed_source == "gold_seed":
        gid = item.get("gold_seed")
        if not gid:
            raise ProfilerError("arm %r needs item['gold_seed']" % arm.name)
        if not graph.has(gid):
            raise ProfilerError("gold_seed %r is not in the graph" % (gid,))
        return [Seed(event_id=gid, score=1.0, source="oracle", rank=0)]
    ids = list(item.get("required_sources") or ())          # gold_minimal
    if not ids:
        raise ProfilerError("arm %r needs a non-empty item['required_sources']" % arm.name)
    out = []
    for i, eid in enumerate(ids):
        if not graph.has(eid):
            raise ProfilerError("required_sources names %r which is not in the graph" % (eid,))
        out.append(Seed(event_id=eid, score=1.0, source="oracle", rank=i))
    return out


def _entry_text(event, repr_kind: str) -> str:
    """Reproduce the exact text the assembler wrote for one entry (mirrors ``_variants``)."""
    src = event.representation("raw_text")
    src_text = src.text if (src is not None and src.text is not None) else event.text
    car = event.representation("materialized_text")
    car_text = car.text if (car is not None and car.text is not None) else None
    if repr_kind == "raw_text":
        return src_text
    if repr_kind == "materialized_text":
        return car_text if car_text is not None else src_text
    if repr_kind == "raw_text+materialized_text":
        if car_text is None:                       # pragma: no cover - assembler never does this
            return src_text
        return src_text + SEPARATOR + car_text
    raise ProfilerError("unknown repr_kind in manifest: %r" % (repr_kind,))


def _slots_from_packet(packet: MemoryPacket, graph, carriers) -> tuple:
    """Decompose the packet back into per-entry slots, and *verify* the decomposition.

    The whole B axis rests on being able to rewrite one slot without disturbing the others.  If
    the reconstruction does not reproduce ``packet.context`` byte for byte we would be serving
    text the manifest does not describe, so this raises instead of guessing.
    """
    slots = []
    for i, e in enumerate(packet.manifest.entries):
        ev = graph.event(e.event_id)
        role = "carrier" if e.event_id in carriers else "source"
        slots.append(_Slot(index=i, event_id=e.event_id, kind=ev.kind, role=role,
                           repr_kind=e.repr_kind, text=_entry_text(ev, e.repr_kind),
                           token_cost=e.token_cost))
    rebuilt = SEPARATOR.join(s.text for s in slots)
    if rebuilt != packet.context:
        raise ProfilerError(
            "slot reconstruction does not reproduce the served context (%d vs %d chars); the "
            "arm would serve text the manifest does not describe"
            % (len(rebuilt), len(packet.context)))
    return tuple(slots)


def _assemble(query: str, closure: EvidenceClosure, graph, budget: int, seeds, mode: str,
              repr_policy: str) -> MemoryPacket:
    asm = BudgetAssembler(AssemblerConfig(repr_policy=repr_policy, header="",
                                          separator=SEPARATOR, order="chronological"))
    return asm.assemble(query, closure, graph, budget, seeds=tuple(seeds), query_mode=mode)


def _empty_plan(query: str, graph, budget: int, mode: str) -> _Plan:
    closure = EvidenceClosure(seeds=(), required=(), optional=(), steps=(), query_mode=mode)
    packet = _assemble(query, closure, graph, budget, (), mode, "source_only")
    return _Plan(key="none", packet=packet, slots=(), seeds=(), closure=closure)


def _own_plan(arm: ArmSpec, item: Mapping, graph, query: str, budget: int, mode: str,
              router_cfg: RouterConfig, carriers) -> _Plan:
    seeds = _seeds_for(arm, item, graph, query, router_cfg)
    kw = {}
    if arm.closure_mode == "oracle":
        kw["oracle_required"] = {query: list(item.get("required_sources") or ())}
    closure = TypedClosure(ClosureConfig(mode=arm.closure_mode)).close(
        query, seeds, graph, query_mode=mode, **kw)
    packet = _assemble(query, closure, graph, budget, seeds, mode, arm.repr_policy)
    return _Plan(key="own:%s" % arm.name, packet=packet,
                 slots=_slots_from_packet(packet, graph, carriers),
                 seeds=tuple(seeds), closure=closure)


def _b_plan(item: Mapping, graph, query: str, budget: int, mode: str,
            router_cfg: RouterConfig, carriers, router) -> _Plan:
    """The ONE plan every B-arm renders (§5.3 rule 1).

    Built from the default path (router seeds + native closure) and then *forced* to contain the
    item's carriers, so ``B2_carrier_only`` has something to be about even when the closure would
    not have selected the carrier.  Forcing them in as *required* is deliberate: they are the
    object of study, and letting the budget drop them would silently turn a carrier result into a
    budget result.  Everything else -- ids, order, positions, per-entry cost, budget -- is then
    identical for B1..B4 by construction.
    """
    seeds = list(_resolve_router(router, router_cfg).retrieve(query, graph, router_cfg.k))
    base = TypedClosure(ClosureConfig(mode="native")).close(query, seeds, graph, query_mode=mode)
    forced = [cid for cid in (item.get("carrier_ids") or ()) if graph.has(cid)]
    forced += [cid for cid in carriers if cid in base.optional]
    required = graph.chronological(list(base.required) + forced)
    req_set = set(required)
    optional = tuple(o for o in base.optional if o not in req_set)
    closure = EvidenceClosure(seeds=base.seeds, required=tuple(required), optional=optional,
                              steps=base.steps, query_mode=mode, relaxations=base.relaxations)
    # repr_policy is pinned to source_only here: the B axis varies *content*, and letting the
    # assembler also switch representation would un-freeze the per-entry token costs.
    packet = _assemble(query, closure, graph, budget, seeds, mode, "source_only")
    return _Plan(key="b", packet=packet, slots=_slots_from_packet(packet, graph, carriers),
                 seeds=tuple(seeds), closure=closure)


# ---------------------------------------------------------------- content rendering


def _donor_pool(graph, plan: _Plan, forbidden) -> dict:
    """Same-kind texts from events this item does not serve (the §5.2 "shuffled" arm).

    Built from the *current* graph only, excluding the served ids -- i.e. other items' events of
    the same session.  Deliberately not accumulated across items: a pool that grew with run order
    would make the arm depend on where a checkpoint resumed.
    """
    served = {s.event_id for s in plan.slots}
    pool = {}
    for ev in graph.events:
        if ev.event_id in served or not ev.text:
            continue
        if any(contains_value(ev.text, v) for v in forbidden):
            continue
        pool.setdefault(ev.kind, []).append(ev.text)
    return pool


def _donor_text(slot: _Slot, pool: Mapping, rng: random.Random):
    """A same-kind, closest-length donor forced to ``len(slot.text)`` characters.

    Returns ``(text, used_donor)``.  Equal length is not negotiable (§5.3 rule 2), so the donor
    is padded or cut; "similar token cost" is achieved by preferring the nearest-length
    candidates, which is the closest thing to a token count this module is allowed to compute.
    """
    cands = list(pool.get(slot.kind) or ())
    if not cands:
        return equal_length_filler(slot.text, "SHUFFLED-NO-DONOR"), False
    target = len(slot.text)
    cands.sort(key=lambda t: (abs(len(t) - target), t))
    near = cands[:8]
    pick = near[rng.randrange(len(near))]
    if len(pick) >= target:
        return pick[:target], True
    return pick + _FILLER_CHAR * (target - len(pick)), True


def _render(plan: _Plan, arm: ArmSpec, item: Mapping, rng: random.Random, donors,
            nonce_override=None):
    """Apply the arm's content intervention to the frozen slots; report every §5.3 flag.

    ``nonce_override`` is a fault-injection seam, used only by :func:`_selfcheck`.  Every nonce
    :func:`make_nonce` mints is length-preserving by construction, so the equal-length guard
    below is defense in depth against a future mutation that is not; the seam is how we prove
    the guard actually fires instead of assuming it.
    """
    gold = item.get("gold")
    keep = arm.keep
    notes = {
        "plan": plan.key,
        "keep": keep,
        "n_slots": len(plan.slots),
        "n_source_slots": sum(1 for s in plan.slots if s.role == "source"),
        "n_carrier_slots": sum(1 for s in plan.slots if s.role == "carrier"),
        "replay": "text_only",
    }

    # ---- surface leak (§5.3 rule 5): read off the *plan*, before any intervention ----------
    leaks = [s.event_id for s in plan.slots
             if s.role == "carrier" and gold and contains_value(s.text, gold)]
    notes["surface_leak"] = bool(leaks)
    notes["leaking_carriers"] = leaks

    nonce = None
    if arm.mutate == "source_value":
        if not gold:
            raise ProfilerError("arm %r needs item['gold'] to mutate" % arm.name)
        forbidden = [plan.packet.context] + [str(d) for d in (item.get("distractors") or ())]
        nonce = (nonce_override if nonce_override is not None
                 else make_nonce(gold, item.get("type", ""), rng, forbidden=forbidden))

    used_donor = False
    replaced = 0
    out = []
    for slot in plan.slots:
        text = slot.text
        if arm.mutate == "shuffle":
            text, used = _donor_text(slot, donors, rng)
            used_donor = used_donor or used
        elif keep == "source" and slot.role == "carrier":
            text = equal_length_filler(text, "CARRIER-REMOVED")
        elif keep == "carrier" and slot.role == "source":
            text = equal_length_filler(text, "SOURCE-REMOVED")
        if arm.mutate == "source_value" and slot.role == "source":
            text, n = _wb(gold).subn(nonce, text)
            replaced += n
        if len(text) != len(slot.text):
            raise ProfilerError(
                "arm %r changed slot %d (%s) from %d to %d characters; equal-length replacement "
                "is the §5.3 rule that stops an arm from buying itself budget"
                % (arm.name, slot.index, slot.event_id, len(slot.text), len(text)))
        out.append(text)

    context = SEPARATOR.join(out)
    notes["equal_length_ok"] = len(context) == len(plan.packet.context)
    if not notes["equal_length_ok"]:            # pragma: no cover - the per-slot guard covers it
        raise ProfilerError("arm %r produced a different context length" % arm.name)

    if arm.mutate == "shuffle":
        notes["donor_used"] = used_donor
        notes["gold_in_context"] = bool(gold and contains_value(context, gold))
    if arm.mutate == "source_value":
        stale = [s.event_id for s in plan.slots
                 if s.role == "carrier" and contains_value(s.text, gold)]
        notes.update({
            "nonce": nonce,
            "mutated_from": gold,
            "mutation_replacements": replaced,
            # A replay that changed nothing proves nothing; say so instead of reporting a null.
            "mutation_vacuous": replaced == 0,
            "stale_carriers": stale,
            "stale_value": gold,
            "item_overrides": {"gold": nonce, "mutated_from": gold, "stale_value": gold,
                               "mutation": True},
        })
    return context, notes


# ---------------------------------------------------------------- public API


def build_arm_context(arm, item: Mapping, graph, budget: int, *, rng_seed: int = 0,
                      router_cfg: RouterConfig = RouterConfig(), donors=None,
                      plan_cache=None) -> ArmResult:
    """Build the served context for one ``(arm, item)`` pair.

    ``arm`` is a name from :data:`ARM_NAMES` or an :class:`ArmSpec`.  ``item`` is a dataset row
    (``item_id``, ``gold``, ``gold_seed``, ``required_sources``, ``carrier_ids``, ``query_mode``,
    ``type``, ``distractors``; ``query`` optional -- see :func:`default_query`).

    ``plan_cache`` is how the freeze is made *exact*: pass the same dict for every arm of one
    item and ``A0_shuffled`` renders the very plan ``A2_seed_closure`` served, and ``B1..B4``
    share one plan.  Without it each arm recomputes its plan (still identical for identical
    inputs, but you lose the cheap proof).
    """
    if isinstance(arm, str):
        if arm not in ARMS:
            raise ProfilerError("unknown arm %r (known: %s)" % (arm, ", ".join(ARM_NAMES)))
        spec = ARMS[arm]
    elif isinstance(arm, ArmSpec):
        spec = arm
    else:
        raise ProfilerError("arm must be a name or an ArmSpec, got %r" % (type(arm).__name__,))
    if not isinstance(item, Mapping):
        raise ProfilerError("item must be a mapping, got %r" % (type(item).__name__,))
    if graph is None:
        raise ProfilerError("build_arm_context needs a graph")
    if isinstance(budget, bool) or not isinstance(budget, int):
        raise ProfilerError("budget must be an int, got %r" % (type(budget).__name__,))
    if not isinstance(router_cfg, RouterConfig):
        raise ProfilerError("router_cfg must be a RouterConfig")

    item_id = _item_id(item)
    query = default_query(item)
    mode = _query_mode(item)
    carriers = carrier_event_ids(graph, item)
    cache = plan_cache if plan_cache is not None else {}

    if spec.plan == "none":
        plan = _empty_plan(query, graph, budget, mode)
    elif spec.plan == "b":
        if "b" not in cache:
            cache["b"] = _b_plan(item, graph, query, budget, mode, router_cfg, carriers,
                                 spec.router or ARMS["A2_seed_closure"].router)
        plan = cache["b"]
    elif spec.plan == "a2":
        if "a2" not in cache:
            cache["a2"] = _own_plan(ARMS["A2_seed_closure"], item, graph, query, budget, mode,
                                    router_cfg, carriers)
        plan = cache["a2"]
    else:
        key = "own:%s" % spec.name
        if key not in cache:
            cache[key] = _own_plan(spec, item, graph, query, budget, mode, router_cfg, carriers)
        plan = cache[key]

    rng = random.Random(rng_seed ^ _stable_salt(item_id, spec.name))
    pool = donors
    if pool is None and spec.mutate == "shuffle":
        forbid = [str(item.get("gold") or "")] + [str(d) for d in (item.get("distractors") or ())]
        pool = _donor_pool(graph, plan, [f for f in forbid if f])
    context, notes = _render(plan, spec, item, rng, pool or {})

    notes["budget"] = budget
    notes["query"] = query
    notes["query_mode"] = mode
    # The manifest is shared with the plan on purpose: it IS the frozen serving plan, and
    # equal-length replacement keeps its per-entry cost accounting true for every arm.
    return ArmResult(arm=spec.name, context=context,
                     packet=MemoryPacket(context=context, manifest=plan.packet.manifest),
                     manifest=plan.packet.manifest, notes=notes)


def assert_frozen_routing(results, *, where: str = "B arms") -> str:
    """Raise unless every result served the *same plan* (§5.3 rule 1); return the shared digest.

    Same event ids, same order, same per-entry token cost, same budget -- the manifest digest
    covers exactly those, and the equal-length rule makes the character counts match too.
    Contexts are expected to *differ*; that is the intervention.
    """
    results = list(results)
    if not results:
        raise ProfilerError("assert_frozen_routing needs at least one result")
    # keyed by position, not by arm name: two results for the *same* arm at different budgets
    # must still be compared, and a name-keyed dict would silently collapse them into one.
    keys = [(i, r.arm, r.frozen_key) for i, r in enumerate(results)]
    uniq = {k for _, _, k in keys}
    if len(uniq) != 1:
        raise ProfilerError("routing not frozen across %s: %s"
                            % (where, [(i, a, k[:12]) for i, a, k in keys]))
    lengths = [(i, r.arm, len(r.context)) for i, r in enumerate(results)]
    if len({n for _, _, n in lengths} ) != 1:
        raise ProfilerError("equal-length replacement violated across %s: %s" % (where, lengths))
    return uniq.pop()


# ---------------------------------------------------------------- self check


def _expect(exc, fn, *args, **kwargs):
    try:
        fn(*args, **kwargs)
    except exc:
        return
    except Exception as other:                       # pragma: no cover - diagnostic
        raise AssertionError("expected %s, got %r" % (exc.__name__, other))
    raise AssertionError("expected %s from %s" % (exc.__name__, getattr(fn, "__name__", fn)))


class _FixedRouter:
    """Deterministic stub router: the selfcheck must not depend on BM25 tuning."""

    def __init__(self, ids):
        self.ids = tuple(ids)

    def retrieve(self, query, graph, k):
        return [Seed(event_id=e, score=1.0 - i * 0.1, source="lexical", rank=i)
                for i, e in enumerate(self.ids[:k])]


def _fixture(leaky_carrier: bool = False):
    """A trace whose right answer we know.

    ``tr1`` (tool_result) is the ONLY event holding the gold value 4711003; ``a1`` (assistant)
    depends on it but never repeats it; ``sum1`` is a compaction carrier over ``tr1``.  So:
    seeding on ``a1`` alone cannot answer, closure over DEPENDS_ON can, and the carrier can only
    answer when it leaks the value.
    """
    from tracepack.core.graph import TraceGraph
    from tracepack.core.schema import RepresentationRef, TraceEdge, TraceEvent

    def ev(eid, kind, ts, text, **kw):
        cost = max(1, len(text) // 4)
        return TraceEvent(event_id=eid, kind=kind, text=text, timestamp=ts, token_cost=cost,
                          representations=(RepresentationRef(kind="raw_text", token_cost=cost,
                                                             text=text),), **kw)

    gold = "4711003"
    carrier_text = ("CARRIER: the size probe finished and the row count was recorded as %s"
                    % (gold if leaky_carrier else "a large value"))
    events = [
        ev("u1", "user", 1, "please check how many rows the export produced"),
        ev("tc1", "tool_call", 2, "TOOL_CALL wc -l export.tsv", step_id="s1",
           tool_call_id="t1", atomic_group="g1"),
        ev("tr1", "tool_result", 3, "TOOL_RESULT rows=%s in export.tsv done" % gold,
           step_id="s1", tool_call_id="t1", atomic_group="g1"),
        ev("a1", "assistant", 4, "The export finished and the row count looks plausible."),
        ev("sum1", "summary", 5, carrier_text),
        ev("n1", "tool_result", 6, "TOOL_RESULT unrelated listing of eight other files here"),
        ev("n2", "assistant", 7, "An unrelated remark about the unrelated listing above."),
        # filler so the shuffled arm has same-kind donors that were NOT served (§5.2 A0)
        ev("n3", "tool_result", 8, "TOOL_RESULT grep found nothing in the staging directory"),
        ev("n4", "tool_result", 9, "TOOL_RESULT head -3 config.yaml printed three short lines"),
        ev("n5", "assistant", 10, "A second unrelated remark, roughly this long, about config."),
        ev("n6", "tool_call", 11, "TOOL_CALL grep -n staging ."),
        ev("n7", "tool_call", 12, "TOOL_CALL head -3 config.yaml"),
        ev("n8", "user", 13, "also, unrelated: what did the config file look like again?"),
    ]
    edges = [
        TraceEdge("tr1", "tc1", "RESULT_OF"),
        TraceEdge("a1", "tr1", "DEPENDS_ON"),
        TraceEdge("sum1", "tr1", "MATERIALIZES", predicate="row_count"),
        TraceEdge("tc1", "u1", "CONTROL"),
    ]
    graph = TraceGraph(events, edges)
    item = {
        "item_id": "sc#decision_source#0", "slice": "decision_source", "query_mode": "why",
        "type": "number", "gold": gold, "distractors": ["4711010", "4710996"],
        "gold_seed": "tr1", "required_sources": ["tc1", "tr1"], "carrier_ids": ["sum1"],
        "query": "How many rows did the export produce, exactly?",
    }
    return graph, item, gold


def _selfcheck() -> None:
    from tracepack.core.schema import BudgetError

    graph, item, gold = _fixture()
    budget = 2048
    cache = {}

    def build(name, **kw):
        return build_arm_context(name, item, graph, budget, rng_seed=7, plan_cache=cache, **kw)

    # --- A ladder: each arm's answerability is known in advance -----------------------------
    a0 = build("A0_null")
    assert a0.context == "" and a0.manifest.total_tokens == 0, "A0_null must serve no evidence"
    assert not contains_value(a0.context, gold)

    # A1 seeds on the *conclusion*; closure off -> gold unreachable.  A2 adds DEPENDS_ON -> gold.
    a1_spec = ArmSpec("A1_seed_only", _FixedRouter(["a1"]), "off", "source_only", "router", None)
    a2_spec = ArmSpec("A2_seed_closure", _FixedRouter(["a1"]), "native", "source_only",
                      "router", None)
    a1 = build_arm_context(a1_spec, item, graph, budget, rng_seed=7)
    a2 = build_arm_context(a2_spec, item, graph, budget, rng_seed=7)
    assert not contains_value(a1.context, gold), "seed-only must not reach the source"
    assert contains_value(a2.context, gold), "native closure must pull the DEPENDS_ON source"
    assert "tr1" in a2.packet.event_ids and "tc1" in a2.packet.event_ids, "atomic pair travels"

    a3 = build("A3_oracle_seed")
    assert contains_value(a3.context, gold), "gold seed + closure must contain the answer"
    a4 = build("A4_gold_minimal")
    assert set(a4.packet.event_ids) == {"tc1", "tr1"}, "A4 serves the annotated minimal set"
    assert contains_value(a4.context, gold) and not a4.manifest.incomplete

    # A0_shuffled: frozen against A2, same length, no gold, different text -------------------
    sh_cache = {}
    a2_default = build_arm_context("A2_seed_closure", item, graph, budget, rng_seed=7,
                                   plan_cache=sh_cache)
    shuf = build_arm_context("A0_shuffled", item, graph, budget, rng_seed=7, plan_cache=sh_cache)
    assert shuf.frozen_key == a2_default.frozen_key, "A0_shuffled must freeze against A2"
    assert len(shuf.context) == len(a2_default.context), "equal-length replacement"
    assert shuf.context != a2_default.context and not contains_value(shuf.context, gold)
    assert shuf.notes["donor_used"] is True, "the fixture has same-kind donors available"
    assert shuf.notes["equal_length_ok"] is True
    again = build_arm_context("A0_shuffled", item, graph, budget, rng_seed=7, plan_cache=sh_cache)
    assert again.context == shuf.context, "shuffle must be deterministic given rng_seed"
    other_seed = build_arm_context("A0_shuffled", item, graph, budget, rng_seed=99,
                                   plan_cache=sh_cache)
    assert other_seed.frozen_key == shuf.frozen_key, "a different seed must not move the plan"

    # --- B axis: one frozen plan, four contents ---------------------------------------------
    bcache = {}
    bs = [build_arm_context(n, item, graph, budget, rng_seed=7, plan_cache=bcache)
          for n in ("B1_source_only", "B2_carrier_only", "B3_source_carrier",
                    "B4_mutated_source")]
    b1, b2, b3, b4 = bs
    key = assert_frozen_routing(bs)
    assert all(r.frozen_key == key for r in bs)
    assert len({r.context for r in bs}) == 4, "four interventions, four contexts"
    assert "sum1" in b1.packet.event_ids, "the carrier is forced into the frozen plan"
    assert b1.notes["n_carrier_slots"] == 1 and b1.notes["n_source_slots"] >= 2

    assert contains_value(b1.context, gold), "B1 keeps the source"
    assert "[CARRIER-REMOVED]" in b1.context, "B1 blanks the carrier slot in place"
    assert "[SOURCE-REMOVED]" in b2.context, "B2 blanks the source slots in place"
    assert not contains_value(b2.context, gold), "a clean carrier must not leak the answer"
    assert b2.notes["surface_leak"] is False and b2.notes["leaking_carriers"] == []
    assert contains_value(b3.context, gold) and "REMOVED]" not in b3.context

    # B4: the source now says the nonce, the un-regenerated carrier is stale ------------------
    nonce = b4.notes["nonce"]
    assert len(nonce) == len(gold) and nonce != gold, "nonce must preserve length (rule 2)"
    assert nonce.isdigit(), "a 'number' nonce must still be a legal number"
    assert contains_value(b4.context, nonce), "the mutation must actually reach the source"
    assert not contains_value(b4.context, gold), "clean carrier -> old value fully gone"
    assert b4.notes["mutation_replacements"] >= 1 and b4.notes["mutation_vacuous"] is False
    assert b4.notes["item_overrides"]["gold"] == nonce
    assert b4.notes["item_overrides"]["stale_value"] == gold
    assert b4.notes["replay"] == "text_only", "never claim causal transmission from a replay"

    # --- surface leak is detected when the carrier really does quote the answer --------------
    lgraph, litem, lgold = _fixture(leaky_carrier=True)
    lcache = {}
    lb2 = build_arm_context("B2_carrier_only", litem, lgraph, budget, rng_seed=7,
                            plan_cache=lcache)
    assert lb2.notes["surface_leak"] is True and lb2.notes["leaking_carriers"] == ["sum1"]
    assert contains_value(lb2.context, lgold), "a leaking carrier answers B2 for free"
    lb4 = build_arm_context("B4_mutated_source", litem, lgraph, budget, rng_seed=7,
                            plan_cache=lcache)
    assert lb4.notes["stale_carriers"] == ["sum1"], "a quoting carrier goes stale after mutation"
    assert contains_value(lb4.context, lgold), "the stale carrier still shows the OLD value"

    # --- a vacuous mutation is reported, not hidden ------------------------------------------
    vitem = dict(item, gold="9999999", distractors=["8888888"])
    vb4 = build_arm_context("B4_mutated_source", vitem, graph, budget, rng_seed=7)
    assert vb4.notes["mutation_vacuous"] is True and vb4.notes["mutation_replacements"] == 0

    # --- stats are JSON-clean and carry the flags attribution needs --------------------------
    import json
    st = b2.stats()
    json.dumps(st)
    assert st["frozen_key"] == key and "item_overrides" not in st
    assert st["n_carrier_slots"] == 1 and st["incomplete"] is False

    # ---------------- fault injection: every guard must actually fire ------------------------
    _expect(ProfilerError, build_arm_context, "NOT_AN_ARM", item, graph, budget)
    _expect(ProfilerError, build_arm_context, 17, item, graph, budget)
    _expect(ProfilerError, build_arm_context, "A1_seed_only", ["not", "a", "mapping"], graph,
            budget)
    _expect(ProfilerError, build_arm_context, "A1_seed_only", item, None, budget)
    _expect(ProfilerError, build_arm_context, "A1_seed_only", item, graph, 2048.0)
    _expect(ProfilerError, build_arm_context, "A1_seed_only", item, graph, True)
    _expect(ProfilerError, build_arm_context, "A1_seed_only", item, graph, budget,
            router_cfg="not a config")
    # a negative budget must surface as the assembler's BudgetError, never be swallowed
    _expect(BudgetError, build_arm_context, "A4_gold_minimal", item, graph, -1)

    # arms whose defining annotation is missing
    _expect(ProfilerError, build_arm_context, "A3_oracle_seed",
            {k: v for k, v in item.items() if k != "gold_seed"}, graph, budget)
    _expect(ProfilerError, build_arm_context, "A4_gold_minimal", dict(item, required_sources=[]),
            graph, budget)
    _expect(ProfilerError, build_arm_context, "A3_oracle_seed", dict(item, gold_seed="ghost"),
            graph, budget)
    _expect(ProfilerError, build_arm_context, "A4_gold_minimal",
            dict(item, required_sources=["ghost"]), graph, budget)
    _expect(ProfilerError, build_arm_context, "B4_mutated_source", dict(item, gold=""), graph,
            budget)
    _expect(ProfilerError, build_arm_context, "A1_seed_only", dict(item, query_mode="nonsense"),
            graph, budget)
    _expect(ProfilerError, build_arm_context, "A1_seed_only", dict(item, query=42), graph, budget)
    _expect(ProfilerError, build_arm_context, "A1_seed_only",
            {k: v for k, v in item.items() if k != "item_id"}, graph, budget)

    # malformed ArmSpecs
    _expect(SchemaError, ArmSpec, "x", None, "no_such_mode", "source_only", "none", None)
    _expect(SchemaError, ArmSpec, "x", None, "off", "no_such_repr", "none", None)
    _expect(SchemaError, ArmSpec, "x", None, "off", "source_only", "no_such_seed", None)
    _expect(SchemaError, ArmSpec, "x", None, "off", "source_only", "none", "no_such_mutation")
    _expect(SchemaError, ArmSpec, "x", None, "off", "source_only", "none", None, "no_such_plan")
    _expect(SchemaError, ArmSpec, "x", None, "off", "source_only", "router", None)
    _expect(SchemaError, ArmSpec, "", None, "off", "source_only", "none", None)
    _expect(SchemaError, ArmSpec, "x", None, "off", "source_only", "none", "shuffle", "none")
    _expect(ProfilerError, build_arm_context,
            ArmSpec("bad", 12345, "off", "source_only", "router", None), item, graph, budget)

    # the freeze check must reject a plan that is NOT frozen
    small = build_arm_context("B1_source_only", item, graph, 200, rng_seed=7)
    _expect(ProfilerError, assert_frozen_routing, [b1, small])
    _expect(ProfilerError, assert_frozen_routing, [])
    assert assert_frozen_routing([b1]) == key

    # equal-length is enforced: inject a nonce that is NOT length-preserving and watch it fire
    bad_arm = ArmSpec("bad_len", None, "native", "source_plus_carrier", "none", "source_value")
    _expect(ProfilerError, _render, bcache["b"], bad_arm, item, random.Random(0), {}, "42")
    # ... and the same call with a legal nonce must succeed, so the guard is not just always-on
    ok_ctx, ok_notes = _render(bcache["b"], bad_arm, item, random.Random(0), {}, "1234567")
    assert ok_notes["equal_length_ok"] and contains_value(ok_ctx, "1234567")

    # slot reconstruction must refuse a packet whose context does not match its manifest
    tampered = MemoryPacket(context=b1.packet.context + " EXTRA", manifest=b1.manifest)
    _expect(ProfilerError, _slots_from_packet, tampered, graph, frozenset({"sum1"}))

    # nonce minting must refuse rather than return a colliding / unusable value
    _expect(ProfilerError, make_nonce, "", "number", random.Random(0))
    # every single-digit surrogate already occurs in the context -> refuse, do not collide
    _expect(ProfilerError, make_nonce, "7", "number", random.Random(0),
            forbidden=("0 1 2 3 4 5 6 7 8 9",), max_tries=40)
    hx = make_nonce("deadbeef", "hash", random.Random(1))
    assert len(hx) == 8 and all(c in _HEX for c in hx) and hx != "deadbeef"
    pth = make_nonce("a/b/report.py", "path", random.Random(1))
    assert len(pth) == len("a/b/report.py") and pth.startswith("a/b/") and pth.endswith(".py")

    # word-boundary matching must not fire on substrings (this is what makes gold checks honest)
    assert contains_value("rows=4711003 done", "4711003")
    assert not contains_value("rows=14711003 done", "4711003")
    assert not contains_value("", "4711003") and not contains_value("x", "")

    # filler is exactly equal length at every size, including degenerate ones
    for n in (0, 1, 5, 9, 10, 64):
        assert len(equal_length_filler("x" * n, "TAG")) == n

    print("interventions.py selfcheck OK: %d arms, frozen key=%s, nonce=%s (gold=%s)"
          % (len(ARMS), key[:12], nonce, gold))


if __name__ == "__main__":
    _selfcheck()
