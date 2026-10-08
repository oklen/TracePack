"""tracepack.core.schema -- the frozen IR (proposal §4.1) and the packet/manifest contract.

Everything else in the package is written against these types.  Design rules baked in here:

* **Events are immutable and hashable**; identity is `event_id` (adapter-provided, never
  re-minted -- test contract #8 requires round-trip ID preservation).
* **Edges point from the current event to what it rests on** (`decision --DEPENDS_ON--> source`).
* **Edge types are a closed set** (proposal §3.2) split into STRONG (evidence-bearing, may be
  followed by closure) and WEAK (ordering/audit only, never auto-followed).
* **Representations are alternative witnesses of one event** with their own token cost; a
  `kv_handle` is an opaque reference and is never moved across models/frameworks (§4.1).
* **Packets carry a manifest** with an inclusion reason per event, missing-dependency list and a
  deterministic hash (test contract #1, #4).

Nothing here imports numpy/torch or any framework: the IR must stay dependency-free so adapters
and the eval harness can both use it.
"""
from __future__ import annotations

import hashlib
import json
from dataclasses import dataclass, field, asdict
from typing import Iterable, Mapping, Sequence

SCHEMA_VERSION = "tracepack-ir/1.0.0"

# ---------------------------------------------------------------- edge taxonomy

#: evidence-bearing edges; closure may follow these (§3.2 whitelist)
STRONG_EDGES = ("RESULT_OF", "GROUNDED_IN", "DEPENDS_ON", "MATERIALIZES", "SUPERSEDES")
#: ordering/audit only -- never followed by closure, only used for sorting and the manifest
WEAK_EDGES = ("CONTROL", "TEMPORAL")
EDGE_TYPES = STRONG_EDGES + WEAK_EDGES

#: edges whose parents are *required* evidence for the child (closure expands these).
#: MATERIALIZES is deliberately NOT here: a carrier pointing at its source does not make the
#: source optional -- relaxation is a separate, verification-gated decision (§4.5).
REQUIRED_EDGES = ("RESULT_OF", "GROUNDED_IN", "DEPENDS_ON")

#: query modes drive closure rules (§3.3)
QUERY_MODES = ("lookup", "why", "state", "audit")

#: representation kinds (§4.1)
#: `excerpt` (phase 2, WP2): a labelled, query-driven line excerpt of a raw_text event, served
#: only when the whole event does not fit; the manifest label carries the line provenance.
REPR_KINDS = ("raw_text", "materialized_text", "kv_handle", "excerpt")


class TracePackError(Exception):
    """Base class; every guard in the package raises a subclass of this."""


class SchemaError(TracePackError):
    pass


class BudgetError(TracePackError):
    pass


# ---------------------------------------------------------------- core records


@dataclass(frozen=True)
class RepresentationRef:
    """One way to serve an event.  `token_cost` is measured with the target tokenizer."""

    kind: str
    token_cost: int
    text: str | None = None
    opaque_ref: str | None = None
    predicate: str | None = None
    model_id: str | None = None
    protocol_id: str | None = None

    def __post_init__(self):
        if self.kind not in REPR_KINDS:
            raise SchemaError("unknown representation kind: %r" % (self.kind,))
        if self.token_cost < 0:
            raise SchemaError("negative token_cost")
        if self.kind in ("raw_text", "materialized_text") and self.text is None:
            raise SchemaError("%s representation needs text" % self.kind)
        if self.kind == "kv_handle":
            if self.opaque_ref is None:
                raise SchemaError("kv_handle needs opaque_ref")
            if self.model_id is None:
                # a KV handle without a model is unusable and, worse, silently portable
                raise SchemaError("kv_handle needs model_id (KV is never cross-model portable)")


@dataclass(frozen=True)
class TraceEvent:
    event_id: str
    kind: str                      # user / assistant / tool_call / tool_result / decision / summary / state
    text: str
    timestamp: int
    step_id: str | None = None
    tool_call_id: str | None = None
    native_ref: str | None = None
    token_cost: int = 0
    representations: tuple[RepresentationRef, ...] = ()
    atomic_group: str | None = None   # events sharing a group are packed all-or-nothing (§3.4)
    meta: Mapping[str, str] = field(default_factory=dict)

    def __post_init__(self):
        if not self.event_id:
            raise SchemaError("event_id must be non-empty")
        if self.token_cost < 0:
            raise SchemaError("negative token_cost")

    def representation(self, kind: str) -> RepresentationRef | None:
        for r in self.representations:
            if r.kind == kind:
                return r
        return None

    def cost_of(self, kind: str = "raw_text") -> int:
        r = self.representation(kind)
        return r.token_cost if r is not None else self.token_cost


@dataclass(frozen=True)
class TraceEdge:
    src_id: str          # the current event ...
    dst_id: str          # ... points at what it rests on
    edge_type: str
    predicate: str | None = None
    provenance: str = "native"    # native | annotated | inferred  (inferred is never STRONG by default)
    meta: Mapping[str, str] = field(default_factory=dict, compare=False, hash=False, repr=False)
    # ^ the evidence behind an inferred edge (quote edges: the quoted fragment, run length, candidate count;
    #   value edges: the value, its type, how many earlier results carried it). Never part of identity.

    def __post_init__(self):
        if self.edge_type not in EDGE_TYPES:
            raise SchemaError("unknown edge_type: %r" % (self.edge_type,))
        if self.src_id == self.dst_id:
            raise SchemaError("self edge on %s" % self.src_id)
        if self.edge_type == "MATERIALIZES" and not self.predicate:
            raise SchemaError("MATERIALIZES edge needs a predicate (§4.5 binds verification to it)")

    @property
    def is_strong(self) -> bool:
        return self.edge_type in STRONG_EDGES

    @property
    def is_required(self) -> bool:
        return self.edge_type in REQUIRED_EDGES


@dataclass(frozen=True)
class Seed:
    event_id: str
    score: float
    source: str            # lexical | dense | rrf | pin | oracle
    rank: int = 0
    pinned: bool = False

    def __post_init__(self):
        if self.source not in ("lexical", "dense", "rrf", "pin", "oracle"):
            raise SchemaError("unknown seed source: %r" % (self.source,))


@dataclass(frozen=True)
class ClosureStep:
    """One expansion, kept so the manifest can name *why* an event is in the packet."""

    child_id: str
    parent_id: str
    edge_type: str
    rule: str


@dataclass(frozen=True)
class EvidenceClosure:
    seeds: tuple[str, ...]
    required: tuple[str, ...]          # events that must be served (seeds + required parents)
    optional: tuple[str, ...]          # nice-to-have, dropped first under budget
    steps: tuple[ClosureStep, ...]
    query_mode: str
    relaxations: tuple[str, ...] = ()  # "<carrier_id>|<predicate>" that stopped an expansion

    def __post_init__(self):
        if self.query_mode not in QUERY_MODES:
            raise SchemaError("unknown query_mode: %r" % (self.query_mode,))
        overlap = set(self.required) & set(self.optional)
        if overlap:
            raise SchemaError("event both required and optional: %s" % sorted(overlap)[:3])


@dataclass(frozen=True)
class PacketEntry:
    event_id: str
    repr_kind: str
    token_cost: int
    reason: str            # seed:lexical | seed:pin | closure:RESULT_OF | optional:rank3 ...


@dataclass(frozen=True)
class CarrierVerification:
    """Record that licensed a relaxation; §4.5 requires all five fields to match at use time."""

    carrier_id: str
    predicate: str
    model_id: str
    construction_protocol: str
    readout_protocol: str
    test_version: str
    ci_low: float
    ci_high: float

    def matches(self, *, predicate: str, model_id: str,
                construction_protocol: str, readout_protocol: str) -> bool:
        return (self.predicate == predicate and self.model_id == model_id
                and self.construction_protocol == construction_protocol
                and self.readout_protocol == readout_protocol)


@dataclass(frozen=True)
class PacketManifest:
    query: str
    query_mode: str
    budget: int
    entries: tuple[PacketEntry, ...]
    seeds: tuple[Seed, ...]
    closure_steps: tuple[ClosureStep, ...]
    omitted_optional: tuple[str, ...]
    missing_required: tuple[str, ...]
    carrier_verifications: tuple[CarrierVerification, ...]
    incomplete: bool
    total_tokens: int
    schema_version: str = SCHEMA_VERSION

    def digest(self) -> str:
        """Deterministic hash over everything that can change the served context.

        Excludes float scores (retriever noise) but includes seed ids/sources and order, so the
        determinism contract (#1) is meaningful without being brittle."""
        payload = {
            "v": self.schema_version,
            "query": self.query,
            "mode": self.query_mode,
            "budget": self.budget,
            "entries": [[e.event_id, e.repr_kind, e.token_cost, e.reason] for e in self.entries],
            "seeds": [[s.event_id, s.source, s.rank, s.pinned] for s in self.seeds],
            "steps": [[c.child_id, c.parent_id, c.edge_type, c.rule] for c in self.closure_steps],
            "omitted": list(self.omitted_optional),
            "missing": list(self.missing_required),
            "incomplete": self.incomplete,
            "total": self.total_tokens,
        }
        blob = json.dumps(payload, sort_keys=True, ensure_ascii=False, separators=(",", ":"))
        return hashlib.sha256(blob.encode("utf-8")).hexdigest()


@dataclass(frozen=True)
class MemoryPacket:
    context: str
    manifest: PacketManifest

    @property
    def incomplete(self) -> bool:
        return self.manifest.incomplete

    @property
    def event_ids(self) -> tuple[str, ...]:
        return tuple(e.event_id for e in self.manifest.entries)

    def to_json(self) -> str:
        d = asdict(self.manifest)
        d["digest"] = self.manifest.digest()
        return json.dumps({"context": self.context, "manifest": d},
                          sort_keys=True, ensure_ascii=False)


def validate_events_edges(events: Sequence[TraceEvent], edges: Iterable[TraceEdge]) -> None:
    """Fail loudly on the two mistakes that silently corrupt every downstream number:
    duplicate event ids, and edges pointing at events that do not exist."""
    seen = set()
    for e in events:
        if e.event_id in seen:
            raise SchemaError("duplicate event_id: %s" % e.event_id)
        seen.add(e.event_id)
    for ed in edges:
        if ed.src_id not in seen:
            raise SchemaError("edge src not in graph: %s" % ed.src_id)
        if ed.dst_id not in seen:
            raise SchemaError("edge dst not in graph: %s" % ed.dst_id)
