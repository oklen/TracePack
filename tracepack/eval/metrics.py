"""tracepack.eval.metrics -- every proposal §6.4 metric, each with one exact definition.

A metric whose definition is ambiguous is worse than no metric: it produces a number that two
readers interpret differently and that nobody can reproduce.  So every function here states, in
its docstring, (a) what the numerator counts, (b) what the denominator counts, (c) what it does
when the denominator is empty, and (d) which of several defensible conventions was chosen and
why.  Nothing is left to the caller's intuition.

Design decisions the harness depends on
=======================================

1. **Per-item metrics take ONE record; corpus metrics take a SEQUENCE.**  The two families are
   never overloaded onto one signature, and each family *raises* when handed the other shape
   (``seed_recall_at_budget([...])`` is an error, not a silently-averaged number).  Aggregate the
   per-item family with :func:`macro_mean` / :func:`micro_ratio`, which are explicit about the
   averaging convention -- macro (per-item mean) and micro (pooled counts) differ, and reporting
   one while calling it the other is a classic silent result-faker.

2. **Undefined is ``nan``, never 0.0 and never 1.0.**  A metric with an empty denominator for a
   given item (e.g. dependency-closure recall on a ``direct_fact`` item that has no dependencies)
   returns ``float('nan')``, which :func:`macro_mean` skips while counting how many items were
   actually defined.  Returning 1.0 "vacuously" would inflate exactly the slices the metric was
   built to expose; returning 0.0 would deflate them.  A *missing* annotation, by contrast, is a
   dataset bug and raises ``SchemaError`` -- absent field and empty field are different things.

3. **"@budget" means measured on what the reader actually saw.**  Seed/evidence recall are
   computed against ``packet.manifest.entries`` (the served set), not against the router output
   and not against the closure.  A gold event that was retrieved, closed over, and then dropped
   by the budget did not help the reader, and the §6.4 family is explicitly "@Budget".  The
   closure-stage metrics (``dependency_closure_recall``, ``closure_precision``) are the ones that
   look at the closure, and they *require* ``record.closure`` instead of guessing it back out of
   the manifest.

4. **Gold id lists are sets, not multisets.**  An annotator who lists a shared dependency once per
   seed must not change the metric.  Every gold/served list is de-duplicated before counting; the
   selfcheck asserts the naive multiset implementation's answer is *not* produced.  A field
   annotated with a single bare id string (``datasets.py`` does this for ``gold_seed``) is read as
   one id, never iterated into character "ids".

5. **Scoring is never inferred.**  ``accuracy`` reads ``ReadResult.correct``; an unscored record
   raises rather than counting as wrong (silently scoring the unscored as incorrect is how an arm
   with a broken reader looks merely "worse").  Use :func:`grade` to turn answers into ``correct``
   with an explicit, inspectable match function.

6. **Paired tests are paired.**  ``mcnemar`` and ``paired_bootstrap_ci`` match records by
   ``item_id`` and refuse to run when the two arms do not cover the same item set.  Comparing
   different subsets is the failure mode that turns a null result into a "win".

7. **The bootstrap is cluster-aware and the cluster is the session.**  Items drawn from one
   transcript are not independent, and an iid item bootstrap reports a CI that is several times
   too narrow (the selfcheck asserts the clustered interval is the wider one).  ``session_id`` is
   therefore a *required* item field: falling back to "each item is its own cluster" would
   silently restore the too-narrow interval.

8. **Determinism.**  ``paired_bootstrap_ci`` uses ``random.Random(seed)`` with ``seed=1234`` and
   ``B=4000`` by default; iteration order over sessions is sorted.  No wall clock, no global RNG,
   no set-iteration order escapes into a number.

Implements §6.4 (all five metric families), plus the operational definitions that §5.3 (carrier
counterfactual controls) and §5.4 (failure taxonomy) depend on.  Pure stdlib: no network, no
torch, no LLM -- the reader is injected upstream and reaches this module only as a
:class:`ReadResult`.
"""
from __future__ import annotations

import math
import random
import re
from dataclasses import dataclass, field, replace
from typing import Any, Callable, Mapping, Sequence

try:  # normal package import
    from tracepack.core.schema import (
        EvidenceClosure,
        MemoryPacket,
        SchemaError,
    )
except ImportError:  # pragma: no cover - direct `python3 tracepack/eval/metrics.py`
    import os as _os
    import sys as _sys

    _sys.path.insert(
        0, _os.path.dirname(_os.path.dirname(_os.path.dirname(_os.path.abspath(__file__))))
    )
    from tracepack.core.schema import (  # noqa: E402
        EvidenceClosure,
        MemoryPacket,
        SchemaError,
    )

__all__ = [
    # containers / plumbing
    "ReadResult", "EvalRecord", "ITEM_FIELDS", "make_record", "grade",
    "normalize_answer", "answer_matches", "served_ids", "item_id", "session_id", "slice_of",
    "item_field", "item_id_list", "gold_evidence", "has_dependencies", "is_carrier_only_packet",
    # aggregation
    "macro_mean", "micro_ratio", "percentile", "n_defined",
    # retrieval / evidence (§6.4)
    "seed_recall_at_budget", "evidence_recall_at_budget", "dependency_closure_recall",
    "closure_precision", "stale_leakage_rate",
    # assembly / efficiency (§6.4)
    "input_tokens", "token_inflation_over_seed_only", "budget_violation_rate",
    "incomplete_packet_rate",
    # task (§6.4)
    "accuracy", "per_slice_accuracy", "slice_counts", "mcnemar", "paired_bootstrap_ci",
    # carrier (§6.4, §5.3)
    "carrier_only_accuracy", "counterfactual_flip_rate", "carrier_decodability",
    "reader_utilization_given_probe_positive", "source_carrier_conflict_consistency",
]

NAN = float("nan")

#: frozen bootstrap settings -- two runs of the harness must produce the same interval
BOOTSTRAP_B = 4000
BOOTSTRAP_SEED = 1234
BOOTSTRAP_ALPHA = 0.05

_MISSING = object()


# ============================================================ containers


@dataclass(frozen=True)
class ReadResult:
    """What the (injected) reader returned for one packet.

    ``correct`` is the *only* source of truth for accuracy; ``None`` means "not scored yet" and
    every accuracy-family metric raises on it rather than treating it as wrong.
    ``probe_positive`` is the §5.3(7) probe readout: True when a restricted probe could decode
    the carrier's predicate, ``None`` when no probe was run for this record.
    """

    answer: str = ""
    correct: "bool | None" = None
    probe_positive: "bool | None" = None
    latency_s: "float | None" = None
    meta: Mapping[str, str] = field(default_factory=dict)

    def __post_init__(self):
        if not isinstance(self.answer, str):
            raise SchemaError("ReadResult.answer must be a str, got %r" % (type(self.answer),))
        for name in ("correct", "probe_positive"):
            v = getattr(self, name)
            if v is not None and not isinstance(v, bool):
                raise SchemaError("ReadResult.%s must be a bool or None, got %r" % (name, type(v)))
        if self.latency_s is not None:
            if isinstance(self.latency_s, bool) or not isinstance(self.latency_s, (int, float)):
                raise SchemaError("ReadResult.latency_s must be a number or None")
            if not math.isfinite(float(self.latency_s)) or float(self.latency_s) < 0:
                raise SchemaError("ReadResult.latency_s must be finite and >= 0")

    @classmethod
    def of(cls, obj: Any) -> "ReadResult":
        """Coerce a ReadResult / mapping / plain answer string into a ReadResult."""
        if isinstance(obj, ReadResult):
            return obj
        if isinstance(obj, str):
            return cls(answer=obj)
        if isinstance(obj, Mapping):
            known = ("answer", "correct", "probe_positive", "latency_s", "meta")
            return cls(**{k: obj[k] for k in known if k in obj})
        raise SchemaError("cannot read a ReadResult out of %r" % (type(obj),))


@dataclass(frozen=True)
class EvalRecord:
    """One (item, packet, read result) triple for one experiment arm.

    ``item`` is duck-typed on purpose: the dataset module is a sibling deliverable, so anything
    that answers to mapping keys *or* attributes works.  The field names read here are listed in
    :data:`ITEM_FIELDS`; every accessor names the field and the metric in its error message.
    ``closure`` is the :class:`EvidenceClosure` that produced the packet -- required only by the
    two closure-stage metrics, which refuse to guess it back out of the manifest.
    """

    item: Any
    packet: "MemoryPacket | None" = None
    read: "ReadResult | None" = None
    closure: "EvidenceClosure | None" = None
    arm: str = ""

    def __post_init__(self):
        if self.item is None:
            raise SchemaError("EvalRecord.item is required")
        if self.packet is not None and not isinstance(self.packet, MemoryPacket):
            raise SchemaError("EvalRecord.packet must be a MemoryPacket, got %r"
                              % (type(self.packet),))
        if self.read is not None and not isinstance(self.read, ReadResult):
            raise SchemaError("EvalRecord.read must be a ReadResult, got %r" % (type(self.read),))
        if self.closure is not None and not isinstance(self.closure, EvidenceClosure):
            raise SchemaError("EvalRecord.closure must be an EvidenceClosure, got %r"
                              % (type(self.closure),))
        if not isinstance(self.arm, str):
            raise SchemaError("EvalRecord.arm must be a str")


#: item fields read by this module: ``canonical -> (canonical, *aliases)``.  A None-valued field
#: is an error; an *empty* one means "this metric is undefined here" (design note 2).
ITEM_FIELDS = {
    "item_id": ("item_id", "id", "qid"),
    "session_id": ("session_id", "session", "trace_id"),
    "query": ("query", "question", "query_text", "prompt"),
    "query_mode": ("query_mode", "mode"),
    "slice": ("slice", "slice_name"),
    "gold_seed": ("gold_seed", "gold_seeds"),
    "required_sources": ("required_sources", "required_source_ids"),
    "gold_minimal_packet": ("gold_minimal_packet", "gold_minimal_evidence"),
    "stale_ids": ("stale_ids", "stale_event_ids", "stale_event", "stale_events",
                  "superseded_ids"),
    "carrier_testable": ("carrier_testable", "has_testable_carrier"),
    "gold_answer": ("gold_answer", "answer", "gold"),
    "stale_answer": ("stale_answer", "stale_value", "old_answer"),
}


def make_record(item: Any, packet: Any = None, read: Any = None,
                closure: Any = None, arm: str = "") -> EvalRecord:
    """Build an :class:`EvalRecord`, coercing ``read`` from a mapping / bare answer string."""
    return EvalRecord(item=item, packet=packet,
                      read=None if read is None else ReadResult.of(read),
                      closure=closure, arm=arm)


# ============================================================ item accessors


def _raw(item: Any, key: str, default=_MISSING):
    names = ITEM_FIELDS.get(key, (key,))
    for name in names:
        if isinstance(item, Mapping):
            if name in item:
                return item[name]
        else:
            v = getattr(item, name, _MISSING)
            if v is not _MISSING:
                return v
    if default is _MISSING:
        raise SchemaError("item is missing field %r (accepted names: %s)"
                          % (key, ", ".join(names)))
    return default


def _str_field(item: Any, key: str, *, required: bool = True, default: str = "") -> str:
    v = _raw(item, key, _MISSING if required else default)
    if v is None:
        raise SchemaError("item field %r is None; an unannotated item must not be scored" % key)
    if not isinstance(v, str):
        raise SchemaError("item field %r must be a str, got %r" % (key, type(v)))
    return v


def _as_id_list(v: Any, key: str) -> "list[str]":
    """Normalise an annotated id field to a list of ids.

    A bare non-empty string is ONE id, not a sequence of characters.  ``datasets.py`` annotates
    ``gold_seed`` with a single event id, and iterating that string into ``['a', '1', ...]`` is
    the silent corruption this function exists to prevent -- so the single-id form is accepted
    explicitly rather than rejected (which would break the pipeline) or iterated (which would
    fake a recall of 0 over character "ids").
    """
    if v is None:
        raise SchemaError("item field %r is None; use [] to mean 'empty', never None" % key)
    if isinstance(v, str):
        if not v:
            raise SchemaError("item field %r is an empty string; use [] to mean 'empty'" % key)
        return [v]
    try:
        vals = list(v)
    except TypeError:
        raise SchemaError("item field %r must be an id or a list of ids, got %r" % (key, type(v)))
    for x in vals:
        if not isinstance(x, str) or not x:
            raise SchemaError("item field %r contains a non-id element: %r" % (key, x))
    return vals


def _id_set(item: Any, key: str, *, required: bool = True) -> "frozenset[str]":
    """A gold id list as a SET (design note 4): duplicates in the annotation never move a metric."""
    return frozenset(_as_id_list(_raw(item, key, _MISSING if required else ()), key))


def _item_of(record_or_item: Any) -> Any:
    return record_or_item.item if isinstance(record_or_item, EvalRecord) else record_or_item


def item_field(record_or_item: Any, key: str, default=_MISSING):
    """Read one annotated field by canonical name, honouring :data:`ITEM_FIELDS` aliases.

    Public because ``baselines.py`` (and any future harness) must read ``query`` /
    ``query_mode`` / ``gold_seed`` through the *same* alias table the metrics use -- two
    field-name conventions in one pipeline is how an arm ends up silently unannotated.
    Raises when the field is absent and no default is given.
    """
    return _raw(_item_of(record_or_item), key, default)


def item_id_list(record_or_item: Any, key: str, *, required: bool = True) -> "tuple[str, ...]":
    """An annotated id list in **annotation order**, de-duplicated.

    The set form is the right answer for counting (order-free, duplicate-free); this is the right
    answer for *constructing* things where order is meaningful -- oracle seed ranks, an oracle
    dependency list -- because the annotator's order is the only ranking those have.
    A bare id string is accepted as a one-element list (see :func:`_as_id_list`).
    """
    vals = _as_id_list(_raw(_item_of(record_or_item), key, _MISSING if required else ()), key)
    out: "list[str]" = []
    for x in vals:
        if x not in out:
            out.append(x)
    return tuple(out)


def item_id(record_or_item: Any) -> str:
    """Stable identity used to pair arms.  Required -- an unpaired comparison is forbidden."""
    return _str_field(_item_of(record_or_item), "item_id")


def session_id(record_or_item: Any) -> str:
    """Bootstrap cluster (design note 7).  Required -- there is no safe fallback."""
    return _str_field(_item_of(record_or_item), "session_id")


def slice_of(record_or_item: Any) -> str:
    """Slice label (DESIGN_FROZEN §3).  Missing -> ``"unlabelled"``, a bucket name that makes the
    gap obvious instead of quietly folding those items into a real slice."""
    return _str_field(_item_of(record_or_item), "slice",
                      required=False, default="unlabelled") or "unlabelled"


def gold_evidence(record_or_item: Any) -> "frozenset[str]":
    """The gold minimal evidence packet (§6.1 annotation), as a set.

    Definition: ``gold_minimal_packet`` when annotated and non-empty, otherwise
    ``gold_seed | required_sources``.  The fallback is part of the definition, not a repair: a
    dataset that has not yet annotated minimal packets still yields a well-defined precision, and
    the two sets coincide whenever the annotation is complete.
    """
    it = _item_of(record_or_item)
    gmp = _id_set(it, "gold_minimal_packet", required=False)
    if gmp:
        return gmp
    return (_id_set(it, "gold_seed", required=False)
            | _id_set(it, "required_sources", required=False))


def has_dependencies(record_or_item: Any) -> bool:
    """True when the item has a required source that is not itself a gold seed.

    The filter for the "dependency-heavy slice" the DESIGN_FROZEN closure gate is stated on:
    items without dependencies make the closure metrics undefined (they return ``nan``).
    """
    it = _item_of(record_or_item)
    return bool(_id_set(it, "required_sources", required=False)
                - _id_set(it, "gold_seed", required=False))


def served_ids(record: EvalRecord) -> "frozenset[str]":
    """Events the reader actually saw: the ids in ``packet.manifest.entries`` (design note 3)."""
    return frozenset(e.event_id for e in _packet(record, "served_ids").manifest.entries)


def is_carrier_only_packet(record: EvalRecord) -> bool:
    """True when every served entry is a carrier witness (``materialized_text``).

    The assembler's ``carrier_only`` policy falls back to the *source* for events that have no
    carrier representation (``assembler._variants``), so a packet built under that policy is not
    automatically carrier-pure.  §5.3(5) forbids scoring a carrier arm whose context still holds
    the raw source, hence this explicit test.  An empty packet is not carrier-pure: it carries
    nothing to decode.
    """
    entries = _packet(record, "is_carrier_only_packet").manifest.entries
    return bool(entries) and all(e.repr_kind == "materialized_text" for e in entries)


# ============================================================ shape guards


def _one(record: Any, where: str) -> EvalRecord:
    if isinstance(record, EvalRecord):
        return record
    if isinstance(record, (list, tuple, set, frozenset)):
        raise SchemaError(
            "%s is a PER-ITEM metric and takes a single EvalRecord; aggregate a sequence with "
            "macro_mean(%s, records)" % (where, where))
    raise SchemaError("%s: expected an EvalRecord, got %r" % (where, type(record)))


def _many(records: Any, where: str, *, allow_empty: bool = False) -> "list[EvalRecord]":
    if isinstance(records, EvalRecord):
        raise SchemaError("%s is a CORPUS metric and takes a sequence of EvalRecord, got a single "
                          "record" % where)
    if records is None or isinstance(records, (str, bytes, Mapping)):
        raise SchemaError("%s: expected a sequence of EvalRecord, got %r" % (where, type(records)))
    try:
        out = list(records)
    except TypeError:
        raise SchemaError("%s: expected an iterable of EvalRecord, got %r"
                          % (where, type(records)))
    for r in out:
        if not isinstance(r, EvalRecord):
            raise SchemaError("%s: sequence contains %r, not an EvalRecord" % (where, type(r)))
    if not out and not allow_empty:
        raise SchemaError("%s: no records -- an empty arm is a harness bug, not a 0.0" % where)
    return out


def _packet(record: EvalRecord, where: str) -> MemoryPacket:
    if not isinstance(record, EvalRecord):
        raise SchemaError("%s: expected an EvalRecord, got %r" % (where, type(record)))
    if record.packet is None:
        raise SchemaError("%s: record for item %r has no packet"
                          % (where, _raw(record.item, "item_id", "<no id>")))
    return record.packet


def _closure(record: EvalRecord, where: str) -> EvidenceClosure:
    if record.closure is None:
        raise SchemaError(
            "%s measures the CLOSURE stage and needs EvalRecord.closure; it will not be "
            "reconstructed from the manifest (that would silently measure something else)"
            % where)
    return record.closure


def _read(record: EvalRecord, where: str) -> ReadResult:
    if not isinstance(record, EvalRecord):
        raise SchemaError("%s: expected an EvalRecord, got %r" % (where, type(record)))
    if record.read is None:
        raise SchemaError("%s: record for item %r has no read result"
                          % (where, _raw(record.item, "item_id", "<no id>")))
    return record.read


def _correct(record: EvalRecord, where: str) -> bool:
    r = _read(record, where)
    if r.correct is None:
        raise SchemaError(
            "%s: item %r is unscored (ReadResult.correct is None). Score it with grade() -- "
            "counting an unscored record as wrong makes a broken reader look merely worse"
            % (where, _raw(record.item, "item_id", "<no id>")))
    return bool(r.correct)


# ============================================================ answer matching


_WS_RE = re.compile(r"\s+")
_EDGE_PUNCT_RE = re.compile(r"^[\s\"'`.,;:!?()\[\]{}<>*_-]+|[\s\"'`.,;:!?()\[\]{}<>*_-]+$")


def normalize_answer(text: str) -> str:
    """Casefold, collapse internal whitespace, strip edge punctuation/quotes.  Nothing else.

    Deliberately conservative: no stemming, no number reformatting, no synonym table.  §5.3(6)
    says the values under test are nonces / numbers / paths, for which aggressive normalisation
    only manufactures false matches.
    """
    if text is None:
        return ""
    if not isinstance(text, str):
        raise SchemaError("normalize_answer expects a str, got %r" % (type(text),))
    return _EDGE_PUNCT_RE.sub("", _WS_RE.sub(" ", text)).casefold()


def answer_matches(answer: str, gold: str) -> bool:
    """Default grader: normalised ``gold`` occurs as a substring of normalised ``answer``.

    Containment rather than equality, because a reader answers in a sentence ("the file was
    src/tok.py") while the annotation is the payload ("src/tok.py").  An empty gold never
    matches -- otherwise every answer would be correct.
    """
    g = normalize_answer(gold)
    if not g:
        return False
    return g in normalize_answer(answer)


def grade(records: "Sequence[EvalRecord]",
          match_fn: Callable[[str, str], bool] = answer_matches) -> "list[EvalRecord]":
    """Copies of ``records`` with ``read.correct`` filled in from ``item.gold_answer``.

    Never overwrites an existing ``correct`` (a human or LLM judge outranks string matching), and
    raises for a record with no read result -- there is nothing to grade.
    """
    out: "list[EvalRecord]" = []
    for rec in _many(records, "grade", allow_empty=True):
        r = _read(rec, "grade")
        if r.correct is not None:
            out.append(rec)
            continue
        gold = _str_field(rec.item, "gold_answer")
        out.append(replace(rec, read=replace(r, correct=bool(match_fn(r.answer, gold)))))
    return out


# ============================================================ aggregation


def macro_mean(metric_fn: Callable[..., float],
               records: "Sequence[EvalRecord]", **kw) -> float:
    """Per-item mean of a per-item metric, skipping items where it is ``nan`` (undefined).

    Macro, not micro: every item weighs the same regardless of how many gold events it carries.
    Returns ``nan`` when no item was defined -- never 0.0.  Always report :func:`n_defined`
    beside it, or the reader cannot tell a mean of 200 items from a mean of 3.
    """
    vals = [metric_fn(r, **kw) for r in _many(records, "macro_mean", allow_empty=True)]
    kept = [v for v in vals if not math.isnan(v)]
    if not kept:
        return NAN
    return sum(kept) / len(kept)


def n_defined(metric_fn: Callable[..., float],
              records: "Sequence[EvalRecord]", **kw) -> int:
    """How many records the metric was defined on -- the denominator behind :func:`macro_mean`."""
    return sum(0 if math.isnan(metric_fn(r, **kw)) else 1
               for r in _many(records, "n_defined", allow_empty=True))


def micro_ratio(num_fn: Callable[[EvalRecord], float], den_fn: Callable[[EvalRecord], float],
                records: "Sequence[EvalRecord]") -> float:
    """Pooled ratio ``sum(num) / sum(den)`` -- items with more gold events weigh more.

    Reported next to :func:`macro_mean` whenever the two can disagree; ``nan`` on an empty pool.
    """
    recs = _many(records, "micro_ratio", allow_empty=True)
    num = sum(float(num_fn(r)) for r in recs)
    den = sum(float(den_fn(r)) for r in recs)
    if den <= 0:
        return NAN
    return num / den


def percentile(values: "Sequence[float]", q: float) -> float:
    """Nearest-rank percentile: sort ascending, take index ``ceil(q*n) - 1`` (1-based rank).

    Spelled out because "p95" has at least four common definitions; this one always returns an
    *observed* value and never interpolates, which is what a token-count report wants.
    """
    if not (0.0 < q <= 1.0):
        raise SchemaError("percentile q must be in (0, 1], got %r" % (q,))
    vals = sorted(float(v) for v in values)
    if not vals:
        return NAN
    rank = max(1, math.ceil(q * len(vals)))
    return vals[min(rank, len(vals)) - 1]


# ============================================================ §6.4 retrieval / evidence


def seed_recall_at_budget(record: EvalRecord) -> float:
    """|gold_seed ∩ served| / |gold_seed|, over the events **served in the packet**.

    Numerator: annotated gold seed events that survived routing *and* closure *and* the budget.
    Denominator: the item's gold seed set (de-duplicated).
    Undefined (``nan``) when the item annotates no gold seed.
    This is not retrieval recall: a seed the router found and the assembler then dropped counts
    as a miss, because the reader never saw it -- §6.4 says "@Budget".
    """
    rec = _one(record, "seed_recall_at_budget")
    gold = _id_set(rec.item, "gold_seed")
    if not gold:
        return NAN
    return len(gold & served_ids(rec)) / len(gold)


def evidence_recall_at_budget(record: EvalRecord) -> float:
    """|required_sources ∩ served| / |required_sources|, over the served packet.

    Numerator: annotated required source events present in ``manifest.entries``.
    Denominator: the item's ``required_sources`` **set** -- an annotator who lists a shared
    dependency once per seed must not move the number (design note 4).
    Undefined (``nan``) when the item annotates no required source.
    This is the quantity DESIGN_FROZEN's closure gate ("+10pp on the dependency-heavy slice") is
    stated on.
    """
    rec = _one(record, "evidence_recall_at_budget")
    gold = _id_set(rec.item, "required_sources")
    if not gold:
        return NAN
    return len(gold & served_ids(rec)) / len(gold)


def dependency_closure_recall(record: EvalRecord, tier: str = "required") -> float:
    """Did the closure *find* the dependencies the seeds do not already contain?

    Gold set: ``required_sources - gold_seed``, i.e. the events that can only arrive by expansion.
    Numerator: gold dependencies in ``closure.required`` (``tier="required"``, the default) or in
    ``closure.required | closure.optional`` (``tier="selected"``).
    Denominator: that gold dependency set.
    Undefined (``nan``) when the item has no dependencies (e.g. the ``direct_fact`` slice) --
    filter with :func:`has_dependencies` before reporting a slice mean.

    ``required`` is the default because ``optional`` is by construction the tier the assembler
    drops first (§3.4): a gold dependency parked in ``optional`` is a policy miss, and counting it
    would hide precisely the failure this metric exists to catch.  ``selected`` is offered, and
    named, for the "did the policy see it at all?" reading.  A CLOSURE-stage metric: it ignores
    the budget, which is what :func:`evidence_recall_at_budget` measures.
    """
    rec = _one(record, "dependency_closure_recall")
    if tier not in ("required", "selected"):
        raise SchemaError("dependency_closure_recall tier must be 'required' or 'selected', "
                          "got %r" % (tier,))
    gold = _id_set(rec.item, "required_sources") - _id_set(rec.item, "gold_seed")
    if not gold:
        return NAN
    cl = _closure(rec, "dependency_closure_recall")
    got = set(cl.required)
    if tier == "selected":
        got |= set(cl.optional)
    return len(gold & got) / len(gold)


def closure_precision(record: EvalRecord) -> float:
    """|closure.required ∩ gold_evidence| / |closure.required| -- the over-expansion metric.

    Numerator: required events the annotation agrees are evidence.
    Denominator: everything the closure declared required, i.e. its whole claim on the budget.
    Gold set: :func:`gold_evidence` (the gold minimal packet, else gold_seed | required_sources).
    Undefined (``nan``) when the closure required nothing, or the item has no gold evidence.
    Optional events are excluded from the denominator on purpose: the closure does not *claim*
    they are necessary, and charging it for offering them would make ``full_ancestor`` and
    ``native`` incomparable -- they differ mostly in what they park in ``optional``.
    """
    rec = _one(record, "closure_precision")
    req = set(_closure(rec, "closure_precision").required)
    gold = gold_evidence(rec)
    if not req or not gold:
        return NAN
    return len(req & gold) / len(req)


def stale_leakage_rate(records: "Sequence[EvalRecord]", *, exclude_audit: bool = True) -> float:
    """Fraction of *stale-bearing* items whose served packet still contains a superseded event.

    Numerator: items where ``served ∩ stale_ids`` is non-empty.
    Denominator: items annotating a non-empty ``stale_ids`` -- an item with no stale twin cannot
    leak, and including it would drag the rate toward zero as the dataset grows.
    ``exclude_audit=True`` (default) drops ``query_mode == "audit"`` items: an audit query is
    *supposed* to see the superseded value (§3.3), so counting it would penalise correct
    behaviour.
    Returns ``nan`` when no item qualifies.  Packet-level, not answer-level: the answer-level
    question is :func:`source_carrier_conflict_consistency` (§5.4's ``stale_conflict``).
    """
    den = 0
    num = 0
    for rec in _many(records, "stale_leakage_rate", allow_empty=True):
        stale = _id_set(rec.item, "stale_ids", required=False)
        if not stale:
            continue
        if exclude_audit and _str_field(rec.item, "query_mode",
                                        required=False, default="lookup") == "audit":
            continue
        den += 1
        if stale & served_ids(rec):
            num += 1
    if den == 0:
        return NAN
    return num / den


# ============================================================ §6.4 assembly / efficiency


def input_tokens(records: "Sequence[EvalRecord]") -> dict:
    """Served evidence tokens per packet: ``{"mean", "p50", "p95", "max", "n"}``.

    Source: ``manifest.total_tokens`` (header + entry costs).  Per DESIGN_FROZEN §1 the budget --
    and therefore this number -- covers the *evidence* only; the fixed reader prompt, the query
    and the output reservation sit outside it and are identical across arms, so folding them in
    would add a constant that hides the between-arm differences.
    ``p50``/``p95`` are nearest-rank percentiles (:func:`percentile`).
    """
    vals = [float(_packet(r, "input_tokens").manifest.total_tokens)
            for r in _many(records, "input_tokens")]
    return {
        "mean": sum(vals) / len(vals),
        "p50": percentile(vals, 0.50),
        "p95": percentile(vals, 0.95),
        "max": max(vals),
        "n": len(vals),
    }


def token_inflation_over_seed_only(records: "Sequence[EvalRecord]",
                                   seed_only_records: "Sequence[EvalRecord]") -> float:
    """Pooled ratio ``sum(tokens of this arm) / sum(tokens of the seed-only arm)`` on paired items.

    Both sums run over the item ids present in *both* arms.  Pooled rather than a per-item mean of
    ratios, which would be dominated by the items whose seed-only packet is a handful of tokens.
    1.0 means "closure cost nothing"; 2.3 means "this arm serves 2.3x the tokens under the same
    budget cap".
    Raises when the two arms share no item; returns ``nan`` when the baseline served 0 tokens.
    """
    a = _many(records, "token_inflation_over_seed_only")
    b = _many(seed_only_records, "token_inflation_over_seed_only")
    by_b = {item_id(r): r for r in b}
    shared = [r for r in a if item_id(r) in by_b]
    if not shared:
        raise SchemaError("token_inflation_over_seed_only: the two arms share no item_id")
    num = sum(_packet(r, "token_inflation_over_seed_only").manifest.total_tokens for r in shared)
    den = sum(_packet(by_b[item_id(r)], "token_inflation_over_seed_only").manifest.total_tokens
              for r in shared)
    if den <= 0:
        return NAN
    return num / den


def budget_violation_rate(records: "Sequence[EvalRecord]") -> float:
    """Fraction of packets with ``total_tokens > budget``.  A contract check, not a tuning knob.

    §3.4 makes the budget a hard system contract, so the only acceptable value is 0.0; anything
    else is an assembler bug and every efficiency number in the same table is void.
    """
    recs = _many(records, "budget_violation_rate")
    bad = 0
    for r in recs:
        m = _packet(r, "budget_violation_rate").manifest
        if m.total_tokens > m.budget:
            bad += 1
    return bad / len(recs)


def incomplete_packet_rate(records: "Sequence[EvalRecord]") -> float:
    """Fraction of packets with ``manifest.incomplete`` set (required evidence did not fit).

    Comparable only between arms that declare a required set.  The ``full_trace`` baseline
    declares none -- it is a truncation, not a closure -- so its rate is 0.0 by construction, and
    reading that as "full_trace never drops evidence" is a category error (see
    ``baselines.FULL_TRACE_CAVEAT``).
    """
    recs = _many(records, "incomplete_packet_rate")
    return sum(1 for r in recs
               if _packet(r, "incomplete_packet_rate").manifest.incomplete) / len(recs)


# ============================================================ §6.4 task


def accuracy(records: "Sequence[EvalRecord]") -> float:
    """Mean of ``read.correct`` over records.  Raises on an unscored record (design note 5)."""
    recs = _many(records, "accuracy")
    return sum(1 for r in recs if _correct(r, "accuracy")) / len(recs)


def per_slice_accuracy(records: "Sequence[EvalRecord]") -> dict:
    """``{slice_name: accuracy}``.  Report :func:`slice_counts` beside it: a slice of 3 items has
    a 0.33 quantum and must not be read as a trend."""
    buckets: "dict[str, list[bool]]" = {}
    for r in _many(records, "per_slice_accuracy"):
        buckets.setdefault(slice_of(r), []).append(_correct(r, "per_slice_accuracy"))
    return {k: sum(1 for v in buckets[k] if v) / len(buckets[k]) for k in sorted(buckets)}


def slice_counts(records: "Sequence[EvalRecord]") -> dict:
    """``{slice_name: n}`` -- the denominators behind :func:`per_slice_accuracy`."""
    out: "dict[str, int]" = {}
    for r in _many(records, "slice_counts", allow_empty=True):
        s = slice_of(r)
        out[s] = out.get(s, 0) + 1
    return {k: out[k] for k in sorted(out)}


def mcnemar(records_a: "Sequence[EvalRecord]",
            records_b: "Sequence[EvalRecord]") -> "tuple[int, int, float]":
    """Exact two-sided McNemar on the paired correctness of arms A and B.  Returns ``(b, c, p)``.

    ``b`` = items A got right and B got wrong; ``c`` = items A got wrong and B got right.
    Concordant pairs carry no information about the difference and are excluded, per the test.
    ``p = min(1, 2 * P(X <= min(b, c)))`` with ``X ~ Binomial(b + c, 0.5)`` -- the exact binomial
    test, not the chi-square approximation, because discordant counts on a 150-200 item benchmark
    are routinely single digits, where the approximation is anti-conservative.
    ``b + c == 0`` -> ``p = 1.0`` (no evidence of any difference).
    Records are paired by ``item_id``; a mismatch between the arms' item sets raises rather than
    silently comparing different subsets.
    """
    pairs = _pair(records_a, records_b, "mcnemar")
    b = sum(1 for ra, rb in pairs if _correct(ra, "mcnemar") and not _correct(rb, "mcnemar"))
    c = sum(1 for ra, rb in pairs if not _correct(ra, "mcnemar") and _correct(rb, "mcnemar"))
    n = b + c
    if n == 0:
        return (b, c, 1.0)
    tail = sum(math.comb(n, i) for i in range(min(b, c) + 1))
    return (b, c, min(1.0, 2.0 * tail / (2.0 ** n)))


def paired_bootstrap_ci(records_a: "Sequence[EvalRecord]",
                        records_b: "Sequence[EvalRecord]",
                        *, B: int = BOOTSTRAP_B, seed: int = BOOTSTRAP_SEED,
                        alpha: float = BOOTSTRAP_ALPHA) -> "tuple[float, float]":
    """Cluster bootstrap 95% CI for ``accuracy(A) - accuracy(B)``.  Returns ``(lo, hi)``.

    **NOT the primary ruler for this project.**  The percentile cluster bootstrap is
    anti-conservative at small G: on TracePack's own ten sessions it rejects a true null 10.0% of
    the time at a nominal 5% (measured, ``tracepack/eval/stats.py`` selfcheck; the same defect was
    measured at 11.0% on the kvmemory E2 line).  Headline contrasts use the wild cluster
    bootstrap-t in ``tracepack.eval.stats`` (measured size 0.045); this function stays for
    comparison and for the metric-family selfchecks.

    Statistic: the paired difference of *accuracy means* over the resampled items.
    Cluster: ``item.session_id``.  Each replicate draws ``n_sessions`` sessions **with
    replacement** and pools every item of every drawn session (a session drawn twice contributes
    its items twice).  Items from one transcript share a trace, a router state and a topic;
    treating them as independent reports an interval several times too narrow -- the selfcheck
    asserts on data built to show it that the clustered interval is the wider one.
    Interval: percentile method on the sorted replicate statistics, ``lo`` at index
    ``floor(alpha/2 * B)`` and ``hi`` at index ``ceil((1 - alpha/2) * B) - 1`` (for the frozen
    ``B=4000, alpha=0.05``: indices 100 and 3799).
    Deterministic: ``random.Random(seed)``, sessions iterated in sorted order.
    """
    if not isinstance(B, int) or isinstance(B, bool) or B <= 0:
        raise SchemaError("B must be a positive int, got %r" % (B,))
    if not isinstance(seed, int) or isinstance(seed, bool):
        raise SchemaError("seed must be an int (an unseeded bootstrap is not reproducible)")
    if not isinstance(alpha, float) or not (0.0 < alpha < 1.0):
        raise SchemaError("alpha must be a float in (0, 1), got %r" % (alpha,))

    by_session: "dict[str, list[float]]" = {}
    for ra, rb in _pair(records_a, records_b, "paired_bootstrap_ci"):
        d = (float(_correct(ra, "paired_bootstrap_ci"))
             - float(_correct(rb, "paired_bootstrap_ci")))
        by_session.setdefault(session_id(ra), []).append(d)
    sessions = [by_session[k] for k in sorted(by_session)]
    if not sessions:
        raise SchemaError("paired_bootstrap_ci: no sessions to resample")

    rng = random.Random(seed)
    n_s = len(sessions)
    stats: "list[float]" = []
    for _ in range(B):
        total = 0.0
        count = 0
        for _ in range(n_s):
            chunk = sessions[rng.randrange(n_s)]
            total += sum(chunk)
            count += len(chunk)
        stats.append(total / count if count else 0.0)
    stats.sort()
    lo = stats[int(math.floor((alpha / 2.0) * B))]
    hi = stats[int(math.ceil((1.0 - alpha / 2.0) * B)) - 1]
    return (lo, hi)


def _pair(records_a: Any, records_b: Any, where: str) -> "list[tuple[EvalRecord, EvalRecord]]":
    """Match two arms by ``item_id``; refuse to run on differing item sets (design note 6)."""
    map_a: "dict[str, EvalRecord]" = {}
    for r in _many(records_a, where):
        k = item_id(r)
        if k in map_a:
            raise SchemaError("%s: duplicate item_id %r in arm A" % (where, k))
        map_a[k] = r
    map_b: "dict[str, EvalRecord]" = {}
    for r in _many(records_b, where):
        k = item_id(r)
        if k in map_b:
            raise SchemaError("%s: duplicate item_id %r in arm B" % (where, k))
        map_b[k] = r
    if set(map_a) != set(map_b):
        only_a = sorted(set(map_a) - set(map_b))[:3]
        only_b = sorted(set(map_b) - set(map_a))[:3]
        raise SchemaError("%s: arms are not paired (A-only=%s, B-only=%s); comparing different "
                          "item subsets is never a valid paired test" % (where, only_a, only_b))
    return [(map_a[k], map_b[k]) for k in sorted(map_a)]


# ============================================================ §6.4 / §5.3 carrier


def carrier_only_accuracy(records: "Sequence[EvalRecord]") -> float:
    """Accuracy of the B2 (carrier-only) arm, with the source-leak guard §5.3(5) demands.

    Every record's packet must be carrier-pure (:func:`is_carrier_only_packet`).  A packet that
    fell back to raw source for a carrier-less event raises, naming the items, instead of quietly
    reporting a "carrier" number that was partly read off the source.  Filter deliberately with
    ``[r for r in records if is_carrier_only_packet(r)]`` -- and report how many you dropped.
    """
    recs = _many(records, "carrier_only_accuracy")
    leaks = [_raw(r.item, "item_id", "<no id>") for r in recs if not is_carrier_only_packet(r)]
    if leaks:
        raise SchemaError(
            "carrier_only_accuracy: %d packet(s) are not carrier-pure (e.g. %s); the assembler "
            "falls back to raw source for events with no carrier, and §5.3(5) forbids scoring "
            "that as a carrier result" % (len(leaks), leaks[:3]))
    return accuracy(recs)


def counterfactual_flip_rate(base_records: "Sequence[EvalRecord]",
                             mutated_records: "Sequence[EvalRecord]", *,
                             directed: bool = False,
                             require_base_correct: bool = True,
                             match_fn: Callable[[str, str], bool] = answer_matches) -> float:
    """§5.2 B4: did mutating the source *before* the carrier was written change the answer?

    Denominator: paired items where the base arm answered correctly (``require_base_correct``,
    default True) -- on an item the reader already got wrong, a changed answer says nothing about
    causal propagation.  Set it False for the "any change at all" reading.
    Numerator (``directed=False``, the default): the normalised answers differ between arms.
    Numerator (``directed=True``): the mutated arm's answer matches the *mutated* item's
    ``gold_answer``, i.e. the change went where the mutation pointed.  Undirected counts any
    change, including a change into a third wrong answer, so it is an upper bound on propagation.
    Returns ``nan`` when no pair qualifies.  Pairing is by ``item_id``; §5.3(1)-(2) (frozen
    routing, equal-length substitution) are the caller's job -- this function cannot see them.
    """
    den = 0
    num = 0
    for rb, rm in _pair(base_records, mutated_records, "counterfactual_flip_rate"):
        if require_base_correct and not _correct(rb, "counterfactual_flip_rate"):
            continue
        den += 1
        a_base = _read(rb, "counterfactual_flip_rate").answer
        a_mut = _read(rm, "counterfactual_flip_rate").answer
        if directed:
            if match_fn(a_mut, _str_field(rm.item, "gold_answer")):
                num += 1
        elif normalize_answer(a_base) != normalize_answer(a_mut):
            num += 1
    if den == 0:
        return NAN
    return num / den


def carrier_decodability(records: "Sequence[EvalRecord]", *,
                         require_carrier_testable: bool = True) -> float:
    """Fraction of probed items whose carrier a restricted probe could read out (§5.3(7)).

    Numerator: records with ``read.probe_positive is True``.
    Denominator: records that were actually probed (``probe_positive is not None``) and, by
    default, are annotated ``carrier_testable`` -- probing an item with no testable carrier
    measures the probe, not the carrier.
    Returns ``nan`` when nothing was probed.  Decodability is about *carrier formation*; whether
    the default reader then uses it is :func:`reader_utilization_given_probe_positive`, and the
    gap between the two is §5.4's ``reader_miss``.
    """
    den = 0
    num = 0
    for r in _many(records, "carrier_decodability", allow_empty=True):
        if require_carrier_testable and not bool(_raw(r.item, "carrier_testable", False)):
            continue
        pp = _read(r, "carrier_decodability").probe_positive
        if pp is None:
            continue
        den += 1
        if pp:
            num += 1
    if den == 0:
        return NAN
    return num / den


def reader_utilization_given_probe_positive(records: "Sequence[EvalRecord]") -> float:
    """P(default reader correct | the probe says the carrier is decodable) -- §6.4's conditional.

    Numerator: records with ``probe_positive is True`` and ``correct is True``.
    Denominator: records with ``probe_positive is True``.
    Returns ``nan`` when no record is probe-positive.  A low value alongside high
    :func:`carrier_decodability` is the operational signature of §5.4's ``reader_miss``: the
    information is in the carrier and the default reader does not use it.
    """
    where = "reader_utilization_given_probe_positive"
    pos = [r for r in _many(records, where, allow_empty=True)
           if _read(r, where).probe_positive is True]
    if not pos:
        return NAN
    return sum(1 for r in pos if _correct(r, where)) / len(pos)


def source_carrier_conflict_consistency(records: "Sequence[EvalRecord]", *,
                                        match_fn: Callable[[str, str], bool] = answer_matches
                                        ) -> dict:
    """§5.3(4): with a fresh source and a stale carrier in one packet, which does the reader follow?

    Considers items annotating BOTH ``gold_answer`` (the fresh, source-consistent value) and
    ``stale_answer`` (the value the outdated carrier implies).
    ``n_source_wins``: the answer matches the fresh value only.
    ``n_carrier_wins``: the answer matches the stale value only.
    ``n_unresolved``: the answer matches both (an ambiguous annotation -- keep the two values
    non-overlapping) or neither (the reader failed for some third reason).
    ``consistency = n_source_wins / (n_source_wins + n_carrier_wins)``.  Unresolved records are
    excluded from that ratio and reported separately, because an answer matching neither value is
    not evidence about which witness won; folding them into the denominator would make a reader
    that simply fails look like one that follows the source less often.  ``nan`` when nothing
    resolved.
    Returns ``{"consistency", "n_conflict", "n_source_wins", "n_carrier_wins", "n_unresolved"}``.
    """
    where = "source_carrier_conflict_consistency"
    src = car = unres = 0
    for r in _many(records, where, allow_empty=True):
        fresh = _str_field(r.item, "gold_answer", required=False)
        stale = _str_field(r.item, "stale_answer", required=False)
        if not fresh or not stale:
            continue
        ans = _read(r, where).answer
        hit_fresh = bool(match_fn(ans, fresh))
        hit_stale = bool(match_fn(ans, stale))
        if hit_fresh and not hit_stale:
            src += 1
        elif hit_stale and not hit_fresh:
            car += 1
        else:
            unres += 1
    resolved = src + car
    return {
        "consistency": (src / resolved) if resolved else NAN,
        "n_conflict": src + car + unres,
        "n_source_wins": src,
        "n_carrier_wins": car,
        "n_unresolved": unres,
    }


# ============================================================ selfcheck
#
# Fault injection, not a happy path: every metric is checked against a value known BY
# CONSTRUCTION, and every guard is checked by feeding it the broken input it exists to catch.


def _expect(exc, fn, *args, **kwargs) -> None:
    try:
        fn(*args, **kwargs)
    except exc:
        return
    raise AssertionError("expected %s from %r(%r, %r)" % (exc.__name__, fn, args, kwargs))


def _close(a: float, b: float, tol: float = 1e-12) -> bool:
    return abs(a - b) <= tol


def _mk_packet(ids, costs=None, *, budget=2048, repr_kind="raw_text",
               total=None, incomplete=False, missing=()):
    """A MemoryPacket built by hand, so every metric's answer is known by construction."""
    from tracepack.core.schema import PacketEntry, PacketManifest

    ids = list(ids)
    costs = list(costs) if costs is not None else [10] * len(ids)
    entries = tuple(PacketEntry(event_id=e, repr_kind=repr_kind, token_cost=c, reason="test")
                    for e, c in zip(ids, costs))
    manifest = PacketManifest(
        query="q", query_mode="lookup", budget=budget, entries=entries, seeds=(),
        closure_steps=(), omitted_optional=(), missing_required=tuple(missing),
        carrier_verifications=(), incomplete=incomplete,
        total_tokens=sum(costs) if total is None else total,
    )
    return MemoryPacket(context=" | ".join(ids), manifest=manifest)


def _mk_closure(required, optional=(), *, mode="why"):
    return EvidenceClosure(seeds=tuple(required[:1]), required=tuple(required),
                           optional=tuple(optional), steps=(), query_mode=mode)


def _item(**kw):
    base = {"item_id": "i1", "session_id": "s1", "query": "q", "query_mode": "lookup",
            "slice": "direct_fact"}
    base.update(kw)
    return base


def _selfcheck() -> None:
    # ---------------------------------------------------------------- evidence family
    # gold seeds {s1, s2}; only s1 was served -> 0.5 by construction
    rec = make_record(_item(gold_seed=["s1", "s2"]), _mk_packet(["s1", "x9"]))
    assert _close(seed_recall_at_budget(rec), 0.5)
    # a seed that was retrieved but dropped by the budget is a miss: the reader never saw it
    dropped = make_record(_item(gold_seed=["s1", "s2"]), _mk_packet(["x9"]))
    assert seed_recall_at_budget(dropped) == 0.0
    # no gold seed annotated -> undefined, NOT 0.0 and NOT 1.0
    assert math.isnan(seed_recall_at_budget(make_record(_item(gold_seed=[]), _mk_packet(["a"]))))

    # ---- THE DOUBLE-COUNT CASE ------------------------------------------------------
    # "shared" is a dependency of both seeds, so the annotator listed it twice.  The gold set has
    # TWO distinct events; a naive multiset implementation would divide by three.  Served: only
    # src_a.  Correct answer 1/2; naive answer 1/3.
    shared = make_record(_item(required_sources=["src_a", "shared", "shared"]),
                         _mk_packet(["src_a"]))
    v = evidence_recall_at_budget(shared)
    assert _close(v, 0.5), v
    assert not _close(v, 1.0 / 3.0), "shared dependency was double-counted in the denominator"
    # and the numerator must not double-count either: serving both is exactly 1.0
    both = make_record(_item(required_sources=["src_a", "shared", "shared"]),
                       _mk_packet(["src_a", "shared"]))
    assert _close(evidence_recall_at_budget(both), 1.0)

    # ---- closure-stage metrics ------------------------------------------------------
    # deps = required_sources - gold_seed = {d1, d2}; closure required {s1, d1}, optional {d2}
    cl = _mk_closure(["s1", "d1"], ["d2"])
    dep = make_record(_item(gold_seed=["s1"], required_sources=["s1", "d1", "d2"]),
                      _mk_packet(["s1", "d1"]), closure=cl)
    assert _close(dependency_closure_recall(dep), 0.5), "optional must not count as required"
    assert _close(dependency_closure_recall(dep, tier="selected"), 1.0)
    assert has_dependencies(dep)
    # an item with no dependencies is undefined, not vacuously perfect
    nodep = make_record(_item(gold_seed=["s1"], required_sources=["s1"]),
                        _mk_packet(["s1"]), closure=_mk_closure(["s1"]))
    assert math.isnan(dependency_closure_recall(nodep)) and not has_dependencies(nodep)
    # closure precision: required {a,b,c}, gold minimal {a,b} -> 2/3 (c is over-expansion)
    prec = make_record(_item(gold_minimal_packet=["a", "b", "b"]), _mk_packet(["a", "b", "c"]),
                       closure=_mk_closure(["a", "b", "c"]))
    assert _close(closure_precision(prec), 2.0 / 3.0)
    # gold_evidence falls back to gold_seed | required_sources when no minimal packet is annotated
    assert gold_evidence(_item(gold_seed=["a"], required_sources=["b", "b"])) == frozenset(
        {"a", "b"})
    # a closure metric without a closure must raise, never silently reconstruct
    _expect(SchemaError, dependency_closure_recall,
            make_record(_item(gold_seed=["s1"], required_sources=["s1", "d1"]),
                        _mk_packet(["s1"])))

    # ---- stale leakage --------------------------------------------------------------
    leaky = make_record(_item(item_id="a", stale_ids=["old1"]), _mk_packet(["old1", "new1"]))
    clean = make_record(_item(item_id="b", stale_ids=["old2"]), _mk_packet(["new2"]))
    nostale = make_record(_item(item_id="c"), _mk_packet(["z"]))
    assert _close(stale_leakage_rate([leaky, clean, nostale]), 0.5), \
        "items with no stale twin must be out of the denominator"
    audit = make_record(_item(item_id="d", stale_ids=["old3"], query_mode="audit"),
                        _mk_packet(["old3"]))
    assert _close(stale_leakage_rate([leaky, clean, audit]), 0.5), "audit mode is excluded"
    assert _close(stale_leakage_rate([leaky, clean, audit], exclude_audit=False), 2.0 / 3.0)
    assert math.isnan(stale_leakage_rate([nostale]))

    # ---------------------------------------------------------------- assembly / efficiency
    toks = [make_record(_item(item_id="t%d" % i), _mk_packet(["e"], [i])) for i in range(1, 21)]
    it = input_tokens(toks)
    assert it["n"] == 20 and _close(it["mean"], 10.5) and it["max"] == 20.0
    assert it["p50"] == 10.0, it["p50"]          # ceil(0.50*20) = 10 -> 1-based rank 10
    assert it["p95"] == 19.0, it["p95"]          # ceil(0.95*20) = 19 -> 1-based rank 19
    # n=20 makes rank 0.95*n an exact integer, where ceil and truncation agree -- so also check
    # n=10, where nearest-rank (ceil(9.5)=10) and a truncating implementation (9) disagree.
    ten = [float(i) for i in range(1, 11)]
    assert percentile(ten, 0.95) == 10.0, "p95 must round the rank UP, not truncate"
    assert percentile(ten, 0.50) == 5.0
    assert percentile([5.0], 0.95) == 5.0 and math.isnan(percentile([], 0.5))
    _expect(SchemaError, percentile, [1.0, 2.0], 0.0)

    arm = [make_record(_item(item_id="p1"), _mk_packet(["a"], [300])),
           make_record(_item(item_id="p2"), _mk_packet(["b"], [100]))]
    seed_only = [make_record(_item(item_id="p1"), _mk_packet(["a"], [100])),
                 make_record(_item(item_id="p2"), _mk_packet(["b"], [100]))]
    assert _close(token_inflation_over_seed_only(arm, seed_only), 2.0), "pooled 400/200"
    _expect(SchemaError, token_inflation_over_seed_only, arm,
            [make_record(_item(item_id="zz"), _mk_packet(["a"], [10]))])

    over = make_record(_item(item_id="v1"), _mk_packet(["a"], [10], budget=8, total=10))
    okp = make_record(_item(item_id="v2"), _mk_packet(["a"], [8], budget=8))
    assert _close(budget_violation_rate([over, okp]), 0.5)
    assert budget_violation_rate([okp]) == 0.0
    inc = make_record(_item(item_id="w1"), _mk_packet(["a"], [8], incomplete=True,
                                                      missing=("gone",)))
    assert _close(incomplete_packet_rate([inc, okp]), 0.5)

    # ---------------------------------------------------------------- task family
    def rec_ab(i, sess, a_ok, b_ok, sl="direct_fact"):
        it_ = _item(item_id="q%d" % i, session_id=sess, slice=sl)
        return (make_record(it_, _mk_packet(["e"]), ReadResult(answer="x", correct=a_ok)),
                make_record(it_, _mk_packet(["e"]), ReadResult(answer="y", correct=b_ok)))

    A, Bx = zip(*[rec_ab(i, "s%d" % (i % 4), i % 3 != 0, i % 2 == 0) for i in range(12)])
    A, Bx = list(A), list(Bx)
    assert _close(accuracy(A), sum(1 for i in range(12) if i % 3 != 0) / 12)
    # per-slice
    mixed = [rec_ab(0, "s0", True, False, "tool_chain")[0],
             rec_ab(1, "s0", False, False, "tool_chain")[0],
             rec_ab(2, "s1", True, False, "direct_fact")[0]]
    assert per_slice_accuracy(mixed) == {"direct_fact": 1.0, "tool_chain": 0.5}
    assert slice_counts(mixed) == {"direct_fact": 1, "tool_chain": 2}

    # ---- mcnemar: exact two-sided binomial, hand-computed -----------------------------
    def arms(pattern):
        """pattern: list of (a_correct, b_correct)."""
        ra, rb = [], []
        for i, (a, b) in enumerate(pattern):
            it_ = _item(item_id="m%d" % i, session_id="s%d" % (i % 3))
            ra.append(make_record(it_, _mk_packet(["e"]), ReadResult(correct=a)))
            rb.append(make_record(it_, _mk_packet(["e"]), ReadResult(correct=b)))
        return ra, rb

    ra, rb = arms([(True, False)] * 5 + [(False, True)] + [(True, True)] * 4)
    b_, c_, p_ = mcnemar(ra, rb)
    assert (b_, c_) == (5, 1)
    assert _close(p_, 2.0 * (1 + 6) / 64.0), p_            # 2*P(X<=1), X~Bin(6,.5) = 0.21875
    ra, rb = arms([(True, False)] * 5)
    assert mcnemar(ra, rb) == (5, 0, 2.0 / 32.0)           # 0.0625
    ra, rb = arms([(True, True), (False, False)])
    assert mcnemar(ra, rb) == (0, 0, 1.0)                  # no discordant pairs -> no evidence
    ra, rb = arms([(True, False)])
    assert mcnemar(ra, rb) == (1, 0, 1.0)                  # 2*0.5 capped at 1.0

    # ---- paired bootstrap: known interval, determinism, cluster-awareness -------------
    ra, rb = arms([(True, False)] * 8)
    lo, hi = paired_bootstrap_ci(ra, rb)
    assert lo == 1.0 and hi == 1.0, (lo, hi)               # every resample gives diff = 1.0
    assert paired_bootstrap_ci(ra, rb) == (lo, hi), "bootstrap must be deterministic"
    assert paired_bootstrap_ci(ra, rb, seed=7) != (0.0, 0.0)

    # 10 sessions x 10 items; 5 sessions all-diff-1, 5 sessions all-diff-0.  Clustering is the
    # whole story here: the iid item bootstrap must report a visibly narrower interval.
    clus_a, clus_b, iid_a, iid_b = [], [], [], []
    for s in range(10):
        for j in range(10):
            a_ok = s < 5
            it_c = _item(item_id="c%d_%d" % (s, j), session_id="sess%d" % s)
            it_i = _item(item_id="c%d_%d" % (s, j), session_id="sess%d_%d" % (s, j))
            clus_a.append(make_record(it_c, _mk_packet(["e"]), ReadResult(correct=a_ok)))
            clus_b.append(make_record(it_c, _mk_packet(["e"]), ReadResult(correct=False)))
            iid_a.append(make_record(it_i, _mk_packet(["e"]), ReadResult(correct=a_ok)))
            iid_b.append(make_record(it_i, _mk_packet(["e"]), ReadResult(correct=False)))
    c_lo, c_hi = paired_bootstrap_ci(clus_a, clus_b)
    i_lo, i_hi = paired_bootstrap_ci(iid_a, iid_b)
    assert (c_hi - c_lo) > 2.0 * (i_hi - i_lo), \
        "cluster bootstrap must be wider than the iid one (%.3f vs %.3f)" % (
            c_hi - c_lo, i_hi - i_lo)

    # ---------------------------------------------------------------- carrier family
    pure = make_record(_item(item_id="k1"), _mk_packet(["c1"], repr_kind="materialized_text"),
                       ReadResult(correct=True))
    leak = make_record(_item(item_id="k2"), _mk_packet(["s1"], repr_kind="raw_text"),
                       ReadResult(correct=True))
    assert is_carrier_only_packet(pure) and not is_carrier_only_packet(leak)
    assert not is_carrier_only_packet(make_record(_item(), _mk_packet([], [])))
    assert carrier_only_accuracy([pure]) == 1.0
    # a source fallback inside a "carrier-only" arm must be refused, not averaged in
    _expect(SchemaError, carrier_only_accuracy, [pure, leak])

    # counterfactual flip: 3 pairs; base wrong on one (excluded), one of the remaining flipped
    def flip_pair(i, base_ok, a_base, a_mut, gold_mut):
        it_b = _item(item_id="f%d" % i, gold_answer="ORIG")
        it_m = _item(item_id="f%d" % i, gold_answer=gold_mut)
        return (make_record(it_b, _mk_packet(["e"]), ReadResult(answer=a_base, correct=base_ok)),
                make_record(it_m, _mk_packet(["e"]), ReadResult(answer=a_mut, correct=False)))

    pairs = [flip_pair(0, True, "ORIG", "NEW7", "NEW7"),
             flip_pair(1, True, "ORIG", "ORIG", "NEW8"),
             flip_pair(2, False, "junk", "NEW9", "NEW9")]
    fb = [p[0] for p in pairs]
    fm = [p[1] for p in pairs]
    assert _close(counterfactual_flip_rate(fb, fm), 0.5), "base-wrong pair must be excluded"
    assert _close(counterfactual_flip_rate(fb, fm, require_base_correct=False), 2.0 / 3.0)
    assert _close(counterfactual_flip_rate(fb, fm, directed=True), 0.5)

    # decodability vs utilization: 4 testable+probed, 3 decodable, 1 of those answered right
    probes = [
        make_record(_item(item_id="d1", carrier_testable=True), _mk_packet(["c"]),
                    ReadResult(correct=True, probe_positive=True)),
        make_record(_item(item_id="d2", carrier_testable=True), _mk_packet(["c"]),
                    ReadResult(correct=False, probe_positive=True)),
        make_record(_item(item_id="d3", carrier_testable=True), _mk_packet(["c"]),
                    ReadResult(correct=False, probe_positive=True)),
        make_record(_item(item_id="d4", carrier_testable=True), _mk_packet(["c"]),
                    ReadResult(correct=False, probe_positive=False)),
        make_record(_item(item_id="d5", carrier_testable=False), _mk_packet(["c"]),
                    ReadResult(correct=True, probe_positive=True)),
        make_record(_item(item_id="d6", carrier_testable=True), _mk_packet(["c"]),
                    ReadResult(correct=True)),          # not probed -> out of both denominators
    ]
    assert _close(carrier_decodability(probes), 0.75), "3 of 4 probed & testable"
    assert _close(carrier_decodability(probes, require_carrier_testable=False), 0.8)
    assert _close(reader_utilization_given_probe_positive(probes), 0.5), \
        "d5 is probe-positive too, so 2 of 4"
    assert math.isnan(carrier_decodability([]))

    # source/carrier conflict: fresh=NEW, stale=OLD
    def conflict(i, ans):
        return make_record(_item(item_id="x%d" % i, gold_answer="NEW42", stale_answer="OLD17"),
                           _mk_packet(["e"]), ReadResult(answer=ans, correct=False))

    cc = source_carrier_conflict_consistency(
        [conflict(0, "the value is NEW42"), conflict(1, "OLD17"), conflict(2, "no idea")])
    assert cc["n_conflict"] == 3 and cc["n_source_wins"] == 1 and cc["n_carrier_wins"] == 1
    assert cc["n_unresolved"] == 1 and _close(cc["consistency"], 0.5), cc
    assert math.isnan(source_carrier_conflict_consistency([conflict(3, "neither")])["consistency"])

    # ---------------------------------------------------------------- aggregation helpers
    agg = [make_record(_item(item_id="g1", gold_seed=["a", "b"]), _mk_packet(["a"])),
           make_record(_item(item_id="g2", gold_seed=[]), _mk_packet(["a"]))]
    assert _close(macro_mean(seed_recall_at_budget, agg), 0.5), "nan items are skipped, not 0.0"
    assert n_defined(seed_recall_at_budget, agg) == 1
    assert math.isnan(macro_mean(seed_recall_at_budget, [agg[1]]))
    # macro != micro, and the helpers must not be interchangeable
    micro = micro_ratio(lambda r: len(_id_set(r.item, "gold_seed", required=False)
                                      & served_ids(r)),
                        lambda r: len(_id_set(r.item, "gold_seed", required=False)), agg)
    assert _close(micro, 0.5)

    # ---------------------------------------------------------------- shared accessors
    # order-preserving (oracle seed ranks depend on it) and duplicate-free
    assert item_id_list(_item(gold_seed=["b", "a", "b"]), "gold_seed") == ("b", "a")
    assert item_id_list(_item(), "gold_seed", required=False) == ()
    assert item_field(_item(query="hello"), "query") == "hello"
    assert item_field({"mode": "why"}, "query_mode") == "why", "aliases must resolve"
    _expect(SchemaError, item_field, _item(), "gold_seed")
    # a bare id string is ONE id -- datasets.py annotates gold_seed that way.  Iterating it into
    # character "ids" would report a confident recall of 0 instead of failing.
    assert item_id_list(_item(gold_seed="s1"), "gold_seed") == ("s1",)
    assert _id_set(_item(gold_seed="s1"), "gold_seed") == frozenset({"s1"})
    assert seed_recall_at_budget(make_record(_item(gold_seed="s1"), _mk_packet(["s1"]))) == 1.0
    _expect(SchemaError, item_id_list, _item(gold_seed=""), "gold_seed")
    _expect(SchemaError, item_id_list, _item(gold_seed=["a", 7]), "gold_seed")

    # an item in datasets.py's actual field vocabulary must resolve through the alias table
    ds_item = {"item_id": "sess#direct_fact#0", "session": "abc12345", "slice": "direct_fact",
               "query_mode": "state", "gold": "8080", "gold_seed": "tr1",
               "required_sources": ["tc1", "tr1"], "stale_value": "9090", "stale_event": "tr0"}
    assert session_id(ds_item) == "abc12345" and slice_of(ds_item) == "direct_fact"
    assert _str_field(ds_item, "gold_answer") == "8080"
    assert _str_field(ds_item, "stale_answer") == "9090"
    assert _id_set(ds_item, "stale_ids") == frozenset({"tr0"})
    assert gold_evidence(ds_item) == frozenset({"tr1", "tc1"})
    ds_rec = make_record(ds_item, _mk_packet(["tc1", "tr1", "tr0"]),
                         ReadResult(answer="9090", correct=False))
    assert evidence_recall_at_budget(ds_rec) == 1.0
    assert seed_recall_at_budget(ds_rec) == 1.0
    assert stale_leakage_rate([ds_rec]) == 1.0, "stale_event must reach stale_ids"
    assert source_carrier_conflict_consistency([ds_rec])["n_carrier_wins"] == 1

    # ---------------------------------------------------------------- grading
    ungraded = [make_record(_item(item_id="h1", gold_answer="42"),
                            _mk_packet(["e"]), ReadResult(answer="the total is 42.")),
                make_record(_item(item_id="h2", gold_answer="42"),
                            _mk_packet(["e"]), ReadResult(answer="seventeen"))]
    _expect(SchemaError, accuracy, ungraded)          # unscored must never count as wrong
    graded = grade(ungraded)
    assert [r.read.correct for r in graded] == [True, False]
    assert accuracy(graded) == 0.5
    # an explicit judgement is never overwritten by string matching
    judged = grade([make_record(_item(item_id="h3", gold_answer="42"), _mk_packet(["e"]),
                                ReadResult(answer="42", correct=False))])
    assert judged[0].read.correct is False
    assert answer_matches("Result: /a/b.py", "/a/b.py") and not answer_matches("nope", "")
    assert normalize_answer('  "Hello   World".  ') == "hello world"
    _expect(SchemaError, normalize_answer, 42)

    # ---------------------------------------------------------------- fault injection
    ok = make_record(_item(gold_seed=["s1"]), _mk_packet(["s1"]), ReadResult(correct=True))
    # per-item vs corpus shapes are not interchangeable
    _expect(SchemaError, seed_recall_at_budget, [ok])
    _expect(SchemaError, accuracy, ok)
    _expect(SchemaError, evidence_recall_at_budget, {"gold_seed": ["a"]})
    _expect(SchemaError, accuracy, [])                       # empty arm is a bug, not 0.0
    _expect(SchemaError, input_tokens, ["not a record"])
    # missing / malformed annotations
    _expect(SchemaError, seed_recall_at_budget, make_record({"item_id": "z"}, _mk_packet(["a"])))
    _expect(SchemaError, seed_recall_at_budget,
            make_record(_item(gold_seed=None), _mk_packet(["a"])))
    _expect(SchemaError, seed_recall_at_budget,
            make_record(_item(gold_seed=["s1", 7]), _mk_packet(["a"])))
    _expect(SchemaError, served_ids, make_record(_item(), None))
    _expect(SchemaError, item_id, {"session_id": "s"})
    # session_id is required: the fallback that would silently narrow the CI does not exist
    nosess = [make_record({"item_id": "n1", "gold_answer": "x"}, _mk_packet(["e"]),
                          ReadResult(correct=True))]
    nosess_b = [make_record({"item_id": "n1", "gold_answer": "x"}, _mk_packet(["e"]),
                            ReadResult(correct=False))]
    _expect(SchemaError, paired_bootstrap_ci, nosess, nosess_b)
    # unpaired arms
    ra, rb = arms([(True, False)] * 3)
    _expect(SchemaError, mcnemar, ra, rb[:2])
    _expect(SchemaError, mcnemar, ra + ra[:1], rb + rb[:1])   # duplicate item_id
    # bootstrap parameter guards
    _expect(SchemaError, paired_bootstrap_ci, ra, rb, B=0)
    _expect(SchemaError, paired_bootstrap_ci, ra, rb, seed="1234")
    _expect(SchemaError, paired_bootstrap_ci, ra, rb, alpha=1.5)
    # container guards
    _expect(SchemaError, ReadResult, "answer", 1)             # correct must be a bool
    _expect(SchemaError, EvalRecord, None)
    _expect(SchemaError, EvalRecord, _item(), "not a packet")
    _expect(SchemaError, ReadResult.of, 3.5)
    _expect(SchemaError, dependency_closure_recall, dep, "both")

    print("metrics.py selfcheck OK: 20 metrics, %d guards, bootstrap CI clustered=%.3f "
          "iid=%.3f (wider is correct)" % (len(__all__), c_hi - c_lo, i_hi - i_lo))


if __name__ == "__main__":
    _selfcheck()
