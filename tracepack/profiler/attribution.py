"""tracepack.profiler.attribution -- the failure taxonomy (proposal §5.4) and its gate.

Turns one item's per-arm read results into ONE operational label plus the evidence that produced
it.  §5.4 is explicit that these labels are *operational diagnoses*, not claims about the model's
internal causal mechanism, and this module is written to keep that promise: every label is a
statement about which intervention flipped the answer, and the evidence dict always carries the
full verdict table so a human can disagree with the rule and not with a hidden number.

The decision order (and why it is an order at all)
==================================================

The arms are nested -- A4 ⊂ A3 ⊂ A2 ⊂ A1 in how much help the reader gets -- so several rules can
fire on one item.  Which one you report *is* the diagnosis, so the order is part of the contract
and is implemented as one literal, readable sequence in :func:`attribute`:

    1. budget_overflow        the required closure did not fit (manifest.incomplete on A2/A3)
    2. task_or_reader_failure gold minimal evidence (A4) still fails
    3. routing_miss           A3 (oracle seed) succeeds, A2 (predicted seed) fails
    4. closure_miss           A2 succeeds, A1 (no closure) fails
    5. edge_coverage_miss     A4 succeeds, A3 fails -- native edges missed what gold requires
    6. carrier_loss           B1 (source) succeeds, B2 (carrier) fails
    7. reader_miss            a probe-positive carrier still fails under the default reader
    8. stale_conflict         the answer matches the SUPERSEDED value
    9. ok                     the default path (A2) succeeded
   10. ambiguous              the pattern matches none of the above

Rules 1-5 are ordered by *how early in the pipeline the cause sits*: a packet that could not fit
makes every later comparison meaningless, and a task the reader cannot do even with gold evidence
makes the router/closure comparisons meaningless too.  Rules 6-7 are the carrier axis, which is a
separate intervention (frozen routing, §5.3) and therefore can fire even on an item whose default
path answered correctly -- "the carrier loses this answer" is a real, reportable property of the
carrier.

Two deviations from a naive reading of that list, both forced and both deliberate:

* **budget_overflow additionally requires the incomplete arm to have *failed*.**  An incomplete
  packet that still produced the right answer did not cost us anything; labelling it a failure
  would score successes as failures and make the §5 Profiler gate meaningless.  The
  ``incomplete`` flag is still reported in the evidence of whatever label does fire.
* **carrier_loss additionally requires the carrier NOT to be probe-positive.**  `reader_miss` is
  by definition a *subset* of `carrier_loss` (the carrier holds the answer, the reader cannot use
  it), so checking carrier_loss first without that exclusion would make `reader_miss` unreachable
  -- a label that can never fire is worse than a wrong order.

A third judgement call is documented rather than hidden: when the default path answers with the
superseded value **and** oracle seeding fixes it, we report `routing_miss`, not `stale_conflict`.
The stale answer is the symptom; the router serving the old event is the actionable cause.
`stale_conflict` is reserved for the case where the *evidence itself* is in conflict -- the
mutated-source replay (B4) whose carrier still quotes the pre-mutation value (§5.3 rule 4).

Flags this module reads out of ``packet_stats`` (the contract with runner.py)
----------------------------------------------------------------------------
``incomplete`` / ``missing_required``  (A2, A3)   -> budget_overflow
``probe_positive``                     (B2)       -> reader_miss vs carrier_loss
``n_carrier_slots``                    (B2)       -> guards against a degenerate carrier arm
``stale_hit``                          (any arm)  -> stale_conflict
``mutation_vacuous``                   (B4)       -> confidence of a stale finding
``frozen_key``                         (B arms)   -> confidence of any carrier finding

Missing arms are allowed: every rule that needs an arm checks for it, so a partial run degrades
to `ambiguous` instead of to a confident wrong label.  ``confident=False`` is the module's way of
saying "this label is the best reading of a pattern that is also consistent with something else";
the §5 gate reports macro-F1 over all labels and may filter on it.

Pure stdlib, deterministic, no network / LLM / torch.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Mapping, Sequence

try:                                    # normal package import
    from ..core.schema import SchemaError
except ImportError:                     # pragma: no cover - direct execution
    import os
    import sys

    sys.path.insert(
        0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))
    from tracepack.core.schema import SchemaError  # type: ignore[no-redef]

__all__ = [
    "LABELS",
    "ARM_A0_NULL", "ARM_A0_SHUFFLED", "ARM_A1", "ARM_A2", "ARM_A3", "ARM_A4",
    "ARM_B1", "ARM_B2", "ARM_B3", "ARM_B4", "DEFAULT_ARM",
    "ReadResult",
    "Attribution",
    "attribute",
    "confusion",
]

#: the §5.4 taxonomy, plus the two outcomes §5.4 leaves implicit (`ok`, `ambiguous`).
LABELS = (
    "task_or_reader_failure",
    "routing_miss",
    "closure_miss",
    "edge_coverage_miss",
    "carrier_loss",
    "reader_miss",
    "budget_overflow",
    "stale_conflict",
    "ok",
    "ambiguous",
)

ARM_A0_NULL = "A0_null"
ARM_A0_SHUFFLED = "A0_shuffled"
ARM_A1 = "A1_seed_only"
ARM_A2 = "A2_seed_closure"
ARM_A3 = "A3_oracle_seed"
ARM_A4 = "A4_gold_minimal"
ARM_B1 = "B1_source_only"
ARM_B2 = "B2_carrier_only"
ARM_B3 = "B3_source_carrier"
ARM_B4 = "B4_mutated_source"

#: the shipping configuration -- the arm whose success means "nothing to diagnose"
DEFAULT_ARM = ARM_A2
#: fallbacks when the default arm was not run, most-default-like first
_DEFAULT_FALLBACK = (ARM_A2, ARM_A3, ARM_B3, ARM_B1, ARM_A4, ARM_A1)


@dataclass(frozen=True)
class ReadResult:
    """What the injected reader returns for one served context.

    ``correct`` is the only field the taxonomy needs; ``score`` (forced-choice log-odds in the
    eval harness) and ``raw`` are carried so the runner can detect a stale answer and so a human
    can audit a label.  Defined here rather than in runner.py because this module is the leaf:
    the taxonomy must be importable and testable without the orchestration.
    """

    correct: bool
    score: float = 0.0
    raw: str = ""

    def __post_init__(self):
        if not isinstance(self.correct, bool):
            raise SchemaError("ReadResult.correct must be a bool, got %r"
                              % (type(self.correct).__name__,))
        if isinstance(self.score, bool) or not isinstance(self.score, (int, float)):
            raise SchemaError("ReadResult.score must be a number, got %r"
                              % (type(self.score).__name__,))
        if not isinstance(self.raw, str):
            raise SchemaError("ReadResult.raw must be a str, got %r" % (type(self.raw).__name__,))


@dataclass(frozen=True)
class Attribution:
    label: str
    evidence: dict = field(default_factory=dict)
    confident: bool = False

    def __post_init__(self):
        if self.label not in LABELS:
            raise SchemaError("unknown attribution label %r (want one of %s)"
                              % (self.label, list(LABELS)))

    def as_dict(self) -> dict:
        return {"label": self.label, "confident": bool(self.confident),
                "evidence": dict(self.evidence)}


# ---------------------------------------------------------------- input coercion


def _as_read(value, arm: str):
    """Accept a ReadResult, a mapping with a ``correct`` key, or a bare bool.

    The eval harness owns the reader, so it may hand us its own result type; what it must never
    do is hand us something whose correctness we have to *guess*.  Anything else raises.
    """
    if value is None:
        return None
    if isinstance(value, ReadResult):
        return value
    if isinstance(value, bool):
        return ReadResult(correct=value)
    if isinstance(value, Mapping):
        if "correct" not in value:
            raise SchemaError("per_arm[%r] has no 'correct' key" % (arm,))
        return ReadResult(correct=bool(value["correct"]),
                          score=float(value.get("score", 0.0) or 0.0),
                          raw=str(value.get("raw", "") or ""))
    correct = getattr(value, "correct", None)
    if isinstance(correct, bool):
        return ReadResult(correct=correct,
                          score=float(getattr(value, "score", 0.0) or 0.0),
                          raw=str(getattr(value, "raw", "") or ""))
    raise SchemaError("per_arm[%r] is not a ReadResult / mapping with 'correct' / bool: %r"
                      % (arm, type(value).__name__))


class _View:
    """Read-only helper over one item's arms; every accessor tolerates a missing arm."""

    def __init__(self, per_arm: Mapping, packet_stats: Mapping):
        self.reads = {}
        for arm, value in per_arm.items():
            if not isinstance(arm, str):
                raise SchemaError("per_arm keys must be arm names, got %r" % (type(arm),))
            self.reads[arm] = _as_read(value, arm)
        self.stats = {}
        for arm, value in (packet_stats or {}).items():
            if not isinstance(arm, str):
                raise SchemaError("packet_stats keys must be arm names, got %r" % (type(arm),))
            if not isinstance(value, Mapping):
                raise SchemaError("packet_stats[%r] must be a mapping, got %r"
                                  % (arm, type(value).__name__))
            self.stats[arm] = value

    def has(self, arm: str) -> bool:
        return self.reads.get(arm) is not None

    def ok(self, arm: str):
        """True / False / None (arm not run)."""
        r = self.reads.get(arm)
        return None if r is None else bool(r.correct)

    def flag(self, arm: str, name: str, default=None):
        return self.stats.get(arm, {}).get(name, default)

    def table(self) -> dict:
        return {arm: (None if r is None else bool(r.correct))
                for arm, r in sorted(self.reads.items())}

    def default_arm(self):
        for arm in _DEFAULT_FALLBACK:
            if self.has(arm):
                return arm
        return None


# ---------------------------------------------------------------- the taxonomy


def attribute(per_arm: Mapping, packet_stats: Mapping) -> Attribution:
    """Label one item's failure (§5.4).  See the module docstring for the decision order.

    ``per_arm`` maps arm name -> :class:`ReadResult` (a mapping with ``correct`` or a bare bool is
    accepted too).  ``packet_stats`` maps arm name -> the flag dict produced by
    ``ArmResult.stats()`` plus whatever the runner measured (``probe_positive``, ``stale_hit``).
    Arms that were not run are simply absent.
    """
    if not isinstance(per_arm, Mapping):
        raise SchemaError("per_arm must be a mapping of arm -> ReadResult, got %r"
                          % (type(per_arm).__name__,))
    if packet_stats is not None and not isinstance(packet_stats, Mapping):
        raise SchemaError("packet_stats must be a mapping of arm -> dict, got %r"
                          % (type(packet_stats).__name__,))
    v = _View(per_arm, packet_stats or {})
    verdicts = v.table()
    base = {"verdicts": verdicts}
    if not v.reads:
        return Attribution("ambiguous", dict(base, reason="no arms were run"), False)

    # A0 floors contaminate every comparison below them: if the item is answerable with no
    # evidence, or with shuffled evidence, the ladder is measuring the prior, not the pipeline.
    leak_floor = bool(v.ok(ARM_A0_NULL)) or bool(v.ok(ARM_A0_SHUFFLED))
    base["leak_floor"] = leak_floor

    # ---- 1. budget_overflow -----------------------------------------------------------------
    for arm in (ARM_A2, ARM_A3):
        if v.has(arm) and v.flag(arm, "incomplete") and v.ok(arm) is False:
            missing = list(v.flag(arm, "missing_required") or ())
            return Attribution(
                "budget_overflow",
                dict(base, arm=arm, missing_required=missing,
                     total_tokens=v.flag(arm, "total_tokens"), budget=v.flag(arm, "budget")),
                # incomplete without a named missing id means a dropped header, not lost evidence
                confident=bool(missing) and not leak_floor)

    # ---- 2. task_or_reader_failure ----------------------------------------------------------
    if v.ok(ARM_A4) is False:
        return Attribution(
            "task_or_reader_failure",
            dict(base, note="gold minimal evidence was served and the reader still failed"),
            confident=not leak_floor)

    # ---- 3. routing_miss --------------------------------------------------------------------
    if v.ok(ARM_A3) is True and v.ok(ARM_A2) is False:
        stale = bool(v.flag(ARM_A2, "stale_hit"))
        return Attribution(
            "routing_miss",
            dict(base, note="oracle seed answers, predicted seed does not",
                 a2_answered_stale_value=stale),
            confident=not leak_floor)

    # ---- 4. closure_miss --------------------------------------------------------------------
    if v.ok(ARM_A2) is True and v.ok(ARM_A1) is False:
        return Attribution(
            "closure_miss",
            dict(base, note="typed closure supplied evidence the seeds alone did not"),
            confident=not leak_floor)

    # ---- 5. edge_coverage_miss --------------------------------------------------------------
    if v.ok(ARM_A4) is True and v.ok(ARM_A3) is False:
        return Attribution(
            "edge_coverage_miss",
            dict(base, note="native edges missed a dependency the gold annotation requires",
                 gold_required=list(v.flag(ARM_A4, "missing_required") or ())),
            confident=not leak_floor)

    # ---- 6/7. the carrier axis --------------------------------------------------------------
    # A carrier arm that served no carrier at all is not evidence about carriers.
    b2_slots = v.flag(ARM_B2, "n_carrier_slots", 1)
    b2_meaningful = v.has(ARM_B2) and (b2_slots is None or b2_slots > 0)
    probe = bool(v.flag(ARM_B2, "probe_positive"))
    frozen = _frozen_ok(v)
    if b2_meaningful and v.ok(ARM_B2) is False:
        if v.ok(ARM_B1) is True and not probe:
            return Attribution(
                "carrier_loss",
                dict(base, note="source answers, carrier does not, under frozen routing",
                     frozen_ok=frozen, surface_leak=bool(v.flag(ARM_B2, "surface_leak"))),
                confident=frozen and not leak_floor)
        if probe:
            return Attribution(
                "reader_miss",
                dict(base, note="the carrier is probe-positive but the default reader failed",
                     frozen_ok=frozen, b1=v.ok(ARM_B1)),
                confident=frozen and not leak_floor)

    # ---- 8. stale_conflict ------------------------------------------------------------------
    stale_arms = sorted(a for a in v.reads if v.flag(a, "stale_hit"))
    if stale_arms:
        arm = ARM_B4 if ARM_B4 in stale_arms else stale_arms[0]
        vacuous = bool(v.flag(arm, "mutation_vacuous"))
        return Attribution(
            "stale_conflict",
            dict(base, arm=arm, arms=stale_arms,
                 note="the answer matched the superseded value",
                 stale_carriers=list(v.flag(arm, "stale_carriers") or ()),
                 mutation_vacuous=vacuous),
            # a replay that replaced nothing cannot support a causal claim (§5.3 rule 4)
            confident=not vacuous and not leak_floor)

    # ---- 9. ok ------------------------------------------------------------------------------
    default_arm = v.default_arm()
    if default_arm is not None and v.ok(default_arm) is True:
        return Attribution(
            "ok", dict(base, arm=default_arm),
            # succeeding when the null/shuffled floor also succeeds is not evidence of retrieval
            confident=not leak_floor)

    # ---- 10. ambiguous ----------------------------------------------------------------------
    return Attribution(
        "ambiguous",
        dict(base, note="no rule matched: the arms disagree in a shape the taxonomy does not "
                        "name (e.g. an arm with strictly more evidence did strictly worse)",
             default_arm=default_arm),
        confident=False)


def _frozen_ok(v: _View) -> bool:
    """True when every B-arm that ran reports the same frozen serving plan (§5.3 rule 1)."""
    keys = [v.flag(a, "frozen_key") for a in (ARM_B1, ARM_B2, ARM_B3, ARM_B4) if v.has(a)]
    keys = [k for k in keys if k]
    return bool(keys) and len(set(keys)) == 1


# ---------------------------------------------------------------- the Profiler gate


def confusion(true_labels: Sequence, pred_labels: Sequence) -> dict:
    """Per-label precision / recall / F1 plus macro-F1 -- the §5 Profiler gate.

    Macro-F1 is averaged over the labels that appear in ``true`` or ``pred`` only.  Averaging
    over all ten of :data:`LABELS` would let a run look better or worse purely because the
    injected-fault set happened to omit a label, which is the opposite of what a gate is for; the
    returned ``label_set`` names exactly what was averaged.  A zero denominator scores 0.0
    (never "undefined", never silently skipped).
    """
    if isinstance(true_labels, (str, bytes)) or isinstance(pred_labels, (str, bytes)):
        raise SchemaError("confusion() takes two sequences of labels, not strings")
    true = list(true_labels)
    pred = list(pred_labels)
    if len(true) != len(pred):
        raise SchemaError("confusion(): %d true labels vs %d predicted" % (len(true), len(pred)))
    if not true:
        raise SchemaError("confusion(): empty input -- an empty gate result reads as 0.0 and "
                          "would be indistinguishable from a total failure")
    for name, seq in (("true", true), ("pred", pred)):
        for lab in seq:
            if lab not in LABELS:
                raise SchemaError("confusion(): unknown %s label %r (want one of %s)"
                                  % (name, lab, list(LABELS)))

    label_set = sorted(set(true) | set(pred))
    matrix = {t: {p: 0 for p in label_set} for t in label_set}
    for t, p in zip(true, pred):
        matrix[t][p] += 1

    per_label = {}
    f1s = []
    for lab in label_set:
        tp = sum(1 for t, p in zip(true, pred) if t == lab and p == lab)
        fp = sum(1 for t, p in zip(true, pred) if t != lab and p == lab)
        fn = sum(1 for t, p in zip(true, pred) if t == lab and p != lab)
        precision = tp / (tp + fp) if (tp + fp) else 0.0
        recall = tp / (tp + fn) if (tp + fn) else 0.0
        f1 = (2 * precision * recall / (precision + recall)) if (precision + recall) else 0.0
        per_label[lab] = {"tp": tp, "fp": fp, "fn": fn,
                          "support": tp + fn, "predicted": tp + fp,
                          "precision": precision, "recall": recall, "f1": f1}
        f1s.append(f1)

    correct = sum(1 for t, p in zip(true, pred) if t == p)
    return {
        "n": len(true),
        "label_set": label_set,
        "labels": per_label,
        "macro_f1": sum(f1s) / len(f1s),
        "accuracy": correct / len(true),
        "matrix": matrix,
    }


# ---------------------------------------------------------------- self check


def _expect(exc, fn, *args, **kwargs):
    try:
        fn(*args, **kwargs)
    except exc:
        return
    except Exception as other:                       # pragma: no cover - diagnostic
        raise AssertionError("expected %s, got %r" % (exc.__name__, other))
    raise AssertionError("expected %s from %s" % (exc.__name__, getattr(fn, "__name__", fn)))


def _R(correct: bool, raw: str = "") -> ReadResult:
    return ReadResult(correct=correct, score=1.0 if correct else -1.0, raw=raw)


def _case(label: str):
    """One synthetic (per_arm, packet_stats) whose correct label is ``label``, by construction."""
    ok, no = _R(True), _R(False)
    frozen = {"frozen_key": "K", "n_carrier_slots": 1}
    if label == "budget_overflow":
        return ({ARM_A1: no, ARM_A2: no, ARM_A3: no, ARM_A4: ok},
                {ARM_A2: {"incomplete": True, "missing_required": ["e9"],
                          "total_tokens": 2048, "budget": 2048}})
    if label == "task_or_reader_failure":
        return ({ARM_A0_NULL: no, ARM_A1: no, ARM_A2: no, ARM_A3: no, ARM_A4: no}, {})
    if label == "routing_miss":
        return ({ARM_A1: no, ARM_A2: no, ARM_A3: ok, ARM_A4: ok}, {})
    if label == "closure_miss":
        return ({ARM_A1: no, ARM_A2: ok, ARM_A3: ok, ARM_A4: ok}, {})
    if label == "edge_coverage_miss":
        return ({ARM_A1: no, ARM_A2: no, ARM_A3: no, ARM_A4: ok}, {})
    if label == "carrier_loss":
        return ({ARM_A2: ok, ARM_A1: ok, ARM_A4: ok, ARM_B1: ok, ARM_B2: no, ARM_B3: ok},
                {ARM_B1: dict(frozen), ARM_B2: dict(frozen, probe_positive=False),
                 ARM_B3: dict(frozen)})
    if label == "reader_miss":
        return ({ARM_A2: ok, ARM_A1: ok, ARM_A4: ok, ARM_B1: ok, ARM_B2: no, ARM_B3: ok},
                {ARM_B1: dict(frozen), ARM_B2: dict(frozen, probe_positive=True),
                 ARM_B3: dict(frozen)})
    if label == "stale_conflict":
        return ({ARM_A1: ok, ARM_A2: ok, ARM_A4: ok, ARM_B4: _R(False, "the old value 4711003")},
                {ARM_B4: dict(frozen, stale_hit=True, stale_carriers=["sum1"],
                              mutation_vacuous=False)})
    if label == "ok":
        return ({ARM_A0_NULL: no, ARM_A1: ok, ARM_A2: ok, ARM_A3: ok, ARM_A4: ok}, {})
    if label == "ambiguous":
        # strictly more evidence did strictly worse and nothing recovers it: not a named shape
        return ({ARM_A1: ok, ARM_A2: no, ARM_A3: no}, {})
    raise AssertionError("no synthetic case for %r" % (label,))


def _selfcheck() -> None:
    # ---- every label must be reachable, and reached for the right reason -------------------
    for label in LABELS:
        per_arm, stats = _case(label)
        got = attribute(per_arm, stats)
        assert got.label == label, "case %r produced %r (%s)" % (label, got.label, got.evidence)
        assert isinstance(got.evidence, dict) and "verdicts" in got.evidence
        import json
        json.dumps(got.as_dict())
    assert attribute(*_case("ambiguous")).confident is False, "ambiguous is never confident"
    assert attribute(*_case("routing_miss")).confident is True

    # ---- the two documented deviations from the naive order --------------------------------
    # (a) an incomplete packet that ANSWERED is not a budget failure
    per_arm = {ARM_A1: _R(False), ARM_A2: _R(True), ARM_A4: _R(True)}
    got = attribute(per_arm, {ARM_A2: {"incomplete": True, "missing_required": ["e9"]}})
    assert got.label == "closure_miss", got.label
    # ... and an incomplete packet whose only defect is a dropped header is not confident
    got = attribute({ARM_A2: _R(False), ARM_A4: _R(True)},
                    {ARM_A2: {"incomplete": True, "missing_required": []}})
    assert got.label == "budget_overflow" and got.confident is False

    # (b) reader_miss must be reachable: same verdicts, probe flag is the only difference
    cl = attribute(*_case("carrier_loss"))
    rm = attribute(*_case("reader_miss"))
    assert cl.label == "carrier_loss" and rm.label == "reader_miss"
    assert cl.evidence["verdicts"] == rm.evidence["verdicts"], \
        "the two labels must be separated by the probe flag alone, not by the verdicts"

    # ---- ordering is real: earlier rules win over later ones -------------------------------
    # A4 fails AND A3 beats A2 -> task_or_reader_failure (rule 2 before rule 3)
    got = attribute({ARM_A2: _R(False), ARM_A3: _R(True), ARM_A4: _R(False)}, {})
    assert got.label == "task_or_reader_failure", got.label
    # a stale answer that oracle seeding fixes is reported as the actionable cause (documented)
    got = attribute({ARM_A2: _R(False, "old"), ARM_A3: _R(True), ARM_A4: _R(True)},
                    {ARM_A2: {"stale_hit": True}})
    assert got.label == "routing_miss" and got.evidence["a2_answered_stale_value"] is True

    # ---- degenerate carrier arm must not be read as carrier evidence ------------------------
    got = attribute({ARM_A2: _R(True), ARM_B1: _R(True), ARM_B2: _R(False)},
                    {ARM_B2: {"n_carrier_slots": 0, "frozen_key": "K"},
                     ARM_B1: {"frozen_key": "K"}})
    assert got.label == "ok", "an empty carrier arm proves nothing about carriers: %r" % got.label

    # ---- unfrozen B arms downgrade confidence, they do not change the label ------------------
    per_arm, stats = _case("carrier_loss")
    stats = dict(stats)
    stats[ARM_B2] = dict(stats[ARM_B2], frozen_key="DIFFERENT")
    got = attribute(per_arm, stats)
    assert got.label == "carrier_loss" and got.confident is False

    # ---- the leak floor downgrades everything ------------------------------------------------
    per_arm, stats = _case("routing_miss")
    per_arm = dict(per_arm, **{ARM_A0_SHUFFLED: _R(True)})
    got = attribute(per_arm, stats)
    assert got.label == "routing_miss" and got.confident is False and got.evidence["leak_floor"]

    # ---- a vacuous mutation cannot support a confident stale claim ---------------------------
    per_arm, stats = _case("stale_conflict")
    stats = {ARM_B4: dict(stats[ARM_B4], mutation_vacuous=True)}
    got = attribute(per_arm, stats)
    assert got.label == "stale_conflict" and got.confident is False

    # ---- impossible combinations degrade to ambiguous, never to a confident wrong label ------
    impossible = [
        # closure strictly helps then strictly hurts, and nothing recovers it
        ({ARM_A1: _R(True), ARM_A2: _R(False), ARM_A3: _R(False), ARM_B1: _R(False)},
         {ARM_B1: {"frozen_key": "K"}}),
        # oracle seed fails, gold minimal not run, nothing else present
        ({ARM_A2: _R(False), ARM_A3: _R(False)}, {}),
        # only the null floor was run and it failed: nothing to attribute
        ({ARM_A0_NULL: _R(False)}, {}),
    ]
    for per_arm, stats in impossible:
        got = attribute(per_arm, stats)
        assert got.label == "ambiguous", "%r should be ambiguous, got %r" % (per_arm, got.label)
        assert got.confident is False
    assert attribute({}, {}).label == "ambiguous"

    # ---- partial runs never fabricate a label ------------------------------------------------
    assert attribute({ARM_A4: _R(True)}, {}).label == "ok"          # default falls back to A4
    assert attribute({ARM_A1: _R(True)}, {}).label == "ok"

    # ---- input coercion: mappings and bools are accepted, junk is refused ---------------------
    assert attribute({ARM_A2: {"correct": True, "score": 0.5, "raw": "x"}}, {}).label == "ok"
    assert attribute({ARM_A2: True}, {}).label == "ok"
    _expect(SchemaError, attribute, {ARM_A2: {"score": 1.0}}, {})
    _expect(SchemaError, attribute, {ARM_A2: "yes"}, {})
    _expect(SchemaError, attribute, {17: _R(True)}, {})
    _expect(SchemaError, attribute, ["not", "a", "mapping"], {})
    _expect(SchemaError, attribute, {ARM_A2: _R(True)}, ["not", "a", "mapping"])
    _expect(SchemaError, attribute, {ARM_A2: _R(True)}, {ARM_A2: "not a dict"})
    _expect(SchemaError, ReadResult, "true")
    _expect(SchemaError, ReadResult, True, "high")
    _expect(SchemaError, ReadResult, True, 1.0, 7)
    _expect(SchemaError, Attribution, "not_a_label")

    # ---- confusion(): a hand-computable case ---------------------------------------------------
    true = ["ok", "ok", "routing_miss"]
    pred = ["ok", "routing_miss", "routing_miss"]
    c = confusion(true, pred)
    assert c["n"] == 3 and c["label_set"] == ["ok", "routing_miss"]
    assert c["labels"]["ok"] == {"tp": 1, "fp": 0, "fn": 1, "support": 2, "predicted": 1,
                                 "precision": 1.0, "recall": 0.5,
                                 "f1": 2 * 1.0 * 0.5 / 1.5}
    rm_ = c["labels"]["routing_miss"]
    assert (rm_["tp"], rm_["fp"], rm_["fn"]) == (1, 1, 0)
    assert abs(rm_["precision"] - 0.5) < 1e-12 and rm_["recall"] == 1.0
    assert abs(c["macro_f1"] - 2 / 3.0) < 1e-12 and abs(c["accuracy"] - 2 / 3.0) < 1e-12
    assert c["matrix"]["ok"]["routing_miss"] == 1 and c["matrix"]["ok"]["ok"] == 1
    perfect = confusion(list(LABELS), list(LABELS))
    assert perfect["macro_f1"] == 1.0 and perfect["accuracy"] == 1.0
    # a label predicted but never true still gets scored (0 recall denominator -> 0.0, not a skip)
    c2 = confusion(["ok"], ["carrier_loss"])
    assert c2["macro_f1"] == 0.0 and c2["labels"]["carrier_loss"]["recall"] == 0.0

    # ---- confusion() fault injection -----------------------------------------------------------
    _expect(SchemaError, confusion, ["ok"], ["ok", "ok"])
    _expect(SchemaError, confusion, [], [])
    _expect(SchemaError, confusion, ["ok"], ["NOT_A_LABEL"])
    _expect(SchemaError, confusion, ["NOT_A_LABEL"], ["ok"])
    _expect(SchemaError, confusion, "ok", "ok")

    # ---- the gate, end to end: label every synthetic case and score it -------------------------
    truth = list(LABELS)
    preds = [attribute(*_case(lab)).label for lab in truth]
    gate = confusion(truth, preds)
    assert gate["macro_f1"] == 1.0, "the taxonomy must recover its own injected faults: %s" % (
        {t: p for t, p in zip(truth, preds) if t != p},)

    print("attribution.py selfcheck OK: %d labels, injected-fault macro-F1=%.3f over %d cases"
          % (len(LABELS), gate["macro_f1"], gate["n"]))


if __name__ == "__main__":
    _selfcheck()
