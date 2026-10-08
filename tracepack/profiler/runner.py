"""tracepack.profiler.runner -- run every arm over every item and label the result (§5.1, §5.2).

Orchestration only.  It owns three things the other two profiler modules deliberately do not:
the *reader* (injected, never constructed here), the *checkpoint* (long runs get interrupted),
and the two measurements that need both the served context and the reader's answer
(``stale_hit`` and ``probe_positive``).

Design decisions
================

1. **The reader is injected, and is the only impure thing in the profiler.**
   ``reader(context: str, item: dict) -> ReadResult`` -- so the eval harness can pass a real
   model, the tests pass a deterministic fake, and nothing in this package ever imports torch or
   opens a socket.  A reader that returns something whose correctness we would have to guess is a
   hard error, not a coerced ``False``.

2. **[CONTRACT] The mutated arm reads a mutated item.**
   ``B4_mutated_source`` replaces the gold value with a nonce, so the *question's answer changes*.
   The runner merges ``ArmResult.notes["item_overrides"]`` into the item before calling the
   reader, so ``item["gold"]`` is the nonce and ``item["stale_value"]`` is the pre-mutation value.
   Scoring B4 against the old gold would score "the reader ignored the mutation" as *correct*,
   which inverts the entire measurement.

3. **Freezing is verified per item, not assumed.**  After building the arms we run
   :func:`assert_frozen_routing` over the B arms and over ``(A2, A0_shuffled)``.  A violation is a
   bug in interventions.py, so it raises (``cfg.strict``) rather than being recorded as a finding.

4. **Checkpoint is append-only jsonl keyed by ``item_id``.**  Resume reads the file, keeps the
   *last* record per id (a re-run supersedes), and skips those items.  A truncated final line
   from a killed process is skipped and counted in ``checkpoint_skipped`` -- never repaired, never
   silently treated as a complete record.

5. **Sequential and deterministic.**  Items are processed in the order given; every arm of one
   item shares a ``plan_cache`` so the freeze is exact; the per-item RNG is salted with the item
   id, so a resumed run reproduces the same contexts as an uninterrupted one.  This is the
   property that makes a checkpoint safe: resuming must not change the experiment.

6. **``stale_hit`` needs the raw answer.**  It is true when the reader's text carries the
   superseded value and not the current one, and the answer is wrong.  A reader that returns an
   empty ``raw`` therefore never produces a `stale_conflict` label -- reported as
   ``stale_detectable=False`` rather than as a null result.

Implements: §5.1 (per-sample intervention + attribution), §5.2 (the arm ladder and the three core
deltas), §5.4 via :mod:`tracepack.profiler.attribution`.

Not implemented on purpose: no parallelism (the reader is the bottleneck and is the harness's to
scale), and no bootstrap CI over the deltas -- :func:`summarize` reports the point estimates the
proposal defines and leaves inference to the eval harness that owns the item weighting.
"""
from __future__ import annotations

import json
import os
from dataclasses import dataclass, replace
from typing import Mapping, Sequence

try:                                    # normal package import
    from ..core.router import RouterConfig
    from .attribution import ReadResult, attribute
    from .interventions import (
        ARM_NAMES,
        ARMS,
        DEFAULT_ARMS,
        ArmSpec,
        ProfilerError,
        assert_frozen_routing,
        build_arm_context,
        contains_value,
        default_query,
    )
except ImportError:                     # pragma: no cover - direct execution
    import sys

    sys.path.insert(
        0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))
    from tracepack.core.router import RouterConfig  # type: ignore[no-redef]
    from tracepack.profiler.attribution import (  # type: ignore[no-redef]
        ReadResult, attribute)
    from tracepack.profiler.interventions import (  # type: ignore[no-redef]
        ARM_NAMES, ARMS, DEFAULT_ARMS, ArmSpec, ProfilerError, assert_frozen_routing,
        build_arm_context, contains_value, default_query)

__all__ = ["ProfilerConfig", "ProfilerRunner", "ReadResult", "summarize"]


@dataclass
class ProfilerConfig:
    """Run configuration.  ``budget`` and ``arms`` are the pre-registered §6.5 / §5.2 knobs."""

    budget: int = 2048                       # DESIGN_FROZEN §1 primary budget
    arms: tuple = DEFAULT_ARMS
    seed: int = 0
    router: str = "hybrid_pin"               # overrides the registry's router for every A/B arm
    router_k: int = 8
    checkpoint: object = None                # path to an append-only jsonl, or None
    strict: bool = True                      # arm-build failures raise instead of being recorded
    raw_chars: int = 512                     # how much of the reader's raw answer to persist

    def __post_init__(self):
        if isinstance(self.budget, bool) or not isinstance(self.budget, int) or self.budget < 0:
            raise ProfilerError("ProfilerConfig.budget must be a non-negative int, got %r"
                                % (self.budget,))
        if isinstance(self.arms, str):
            raise ProfilerError("ProfilerConfig.arms must be a sequence of names, not a string")
        arms = tuple(self.arms)
        if not arms:
            raise ProfilerError("ProfilerConfig.arms is empty; there would be nothing to run")
        for a in arms:
            if a not in ARMS:
                raise ProfilerError("unknown arm %r (known: %s)" % (a, ", ".join(ARM_NAMES)))
        self.arms = arms
        if isinstance(self.seed, bool) or not isinstance(self.seed, int):
            raise ProfilerError("ProfilerConfig.seed must be an int, got %r" % (self.seed,))
        if not isinstance(self.router, str):
            raise ProfilerError("ProfilerConfig.router must be a router name")
        if isinstance(self.router_k, bool) or not isinstance(self.router_k, int) \
                or self.router_k <= 0:
            raise ProfilerError("ProfilerConfig.router_k must be a positive int")
        if self.checkpoint is not None and not isinstance(self.checkpoint, str):
            raise ProfilerError("ProfilerConfig.checkpoint must be a path or None")
        if not isinstance(self.strict, bool):
            raise ProfilerError("ProfilerConfig.strict must be a bool")
        if isinstance(self.raw_chars, bool) or not isinstance(self.raw_chars, int) \
                or self.raw_chars < 0:
            raise ProfilerError("ProfilerConfig.raw_chars must be a non-negative int")


def _coerce_read(value, arm: str, item_id: str) -> ReadResult:
    if isinstance(value, ReadResult):
        return value
    if isinstance(value, Mapping):
        if "correct" not in value:
            raise ProfilerError("reader returned a mapping without 'correct' for arm %r of %s"
                                % (arm, item_id))
        return ReadResult(correct=bool(value["correct"]),
                          score=float(value.get("score", 0.0) or 0.0),
                          raw=str(value.get("raw", "") or ""))
    correct = getattr(value, "correct", None)
    if isinstance(correct, bool):
        return ReadResult(correct=correct,
                          score=float(getattr(value, "score", 0.0) or 0.0),
                          raw=str(getattr(value, "raw", "") or ""))
    raise ProfilerError(
        "reader must return a ReadResult (or a mapping/object with .correct) for arm %r of %s; "
        "got %r -- guessing correctness would fabricate the experiment"
        % (arm, item_id, type(value).__name__))


class ProfilerRunner:
    """Run the §5.2 arm ladder over a dataset and attribute each item's failure.

    ``graph_provider`` resolves an item to its :class:`TraceGraph`: a callable ``item -> graph``,
    a mapping keyed by ``item["transcript"]`` / ``["session"]`` / ``["item_id"]``, or a single
    graph object when every item shares one trace.  Graphs are cached per key, because
    normalising a real transcript is the second most expensive thing in a run.

    ``probe`` (optional) is the §5.3-rule-7 restricted readout: ``probe(context, item) -> bool``.
    Without it, ``probe_positive`` falls back to the documented weak textual test "the carrier
    literally contains the gold answer" -- enough to keep `reader_miss` reachable, not enough to
    claim a KV-level probe.
    """

    def __init__(self, graph_provider, reader, cfg: ProfilerConfig = ProfilerConfig(), *,
                 probe=None, donors=None):
        if graph_provider is None:
            raise ProfilerError("ProfilerRunner needs a graph_provider")
        if not callable(reader):
            raise ProfilerError("reader must be callable: reader(context, item) -> ReadResult")
        if not isinstance(cfg, ProfilerConfig):
            raise ProfilerError("cfg must be a ProfilerConfig, got %r" % (type(cfg).__name__,))
        if probe is not None and not callable(probe):
            raise ProfilerError("probe must be callable: probe(context, item) -> bool")
        self.graph_provider = graph_provider
        self.reader = reader
        self.cfg = cfg
        self.probe = probe
        self.donors = donors
        self.router_cfg = RouterConfig(k=cfg.router_k)
        self._graphs = {}
        self.checkpoint_skipped = 0        # malformed/truncated checkpoint lines, counted

    # ------------------------------------------------------------------ graphs

    @staticmethod
    def _graph_key(item: Mapping) -> str:
        return str(item.get("transcript") or item.get("session") or item.get("item_id"))

    def _graph_for(self, item: Mapping):
        key = self._graph_key(item)
        if key in self._graphs:
            return self._graphs[key]
        provider = self.graph_provider
        if callable(provider):
            graph = provider(item)
        elif isinstance(provider, Mapping):
            graph = provider.get(key)
            if graph is None:
                for alt in ("transcript", "session", "item_id"):
                    if item.get(alt) in provider:
                        graph = provider[item[alt]]
                        break
        else:
            graph = provider                       # a single shared graph
        if graph is None or not hasattr(graph, "event"):
            raise ProfilerError("graph_provider produced no usable TraceGraph for item %r"
                                % (item.get("item_id"),))
        self._graphs[key] = graph
        return graph

    # ------------------------------------------------------------------ arms

    def _spec(self, name: str) -> ArmSpec:
        """The registry arm, with the configured router substituted for the default one."""
        spec = ARMS[name]
        if isinstance(spec.router, str) and spec.router != self.cfg.router:
            return replace(spec, router=self.cfg.router)
        return spec

    def _stale_value(self, item: Mapping, notes: Mapping):
        """The value that would mean "the answer came from superseded evidence"."""
        override = (notes.get("item_overrides") or {}).get("stale_value")
        return override or item.get("stale_value")

    # ------------------------------------------------------------------ one item

    def run_item(self, item: Mapping, arms=None) -> dict:
        """Build every arm for one item, read each, and attribute the pattern (§5.1)."""
        if not isinstance(item, Mapping):
            raise ProfilerError("item must be a mapping, got %r" % (type(item).__name__,))
        names = tuple(arms) if arms is not None else self.cfg.arms
        if isinstance(arms, str):
            raise ProfilerError("arms must be a sequence of names, not a string")
        for a in names:
            if a not in ARMS:
                raise ProfilerError("unknown arm %r (known: %s)" % (a, ", ".join(ARM_NAMES)))

        item_id = str(item.get("item_id") or item.get("id") or "")
        if not item_id:
            raise ProfilerError("item has no item_id; checkpointing and resume key on it")
        graph = self._graph_for(item)
        cfg = self.cfg
        plan_cache = {}
        built = {}
        per_arm = {}
        packet_stats = {}
        errors = {}

        for name in names:
            spec = self._spec(name)
            try:
                ar = build_arm_context(spec, item, graph, cfg.budget, rng_seed=cfg.seed,
                                       router_cfg=self.router_cfg, donors=self.donors,
                                       plan_cache=plan_cache)
            except ProfilerError as exc:
                if cfg.strict:
                    raise
                errors[name] = "%s: %s" % (type(exc).__name__, exc)
                continue
            built[name] = ar

            overrides = dict(ar.notes.get("item_overrides") or {})
            item_for_reader = dict(item)
            item_for_reader.update(overrides)
            try:
                raw_result = self.reader(ar.context, item_for_reader)
            except Exception as exc:               # a reader crash is not a finding
                raise ProfilerError("reader raised on arm %r of item %s: %r"
                                    % (name, item_id, exc)) from exc
            rr = _coerce_read(raw_result, name, item_id)
            per_arm[name] = rr

            stats = ar.stats()
            stats.update(self._measure(ar, rr, item, item_for_reader))
            packet_stats[name] = stats

        frozen_ok, frozen_key = self._check_freeze(built)
        att = attribute(per_arm, packet_stats)

        record = {
            "item_id": item_id,
            "slice": item.get("slice"),
            "query": default_query(item),
            "query_mode": item.get("query_mode", "lookup"),
            "budget": cfg.budget,
            "router": cfg.router,
            "seed": cfg.seed,
            "frozen_ok": frozen_ok,
            "frozen_key": frozen_key,
            "arms": {
                name: {
                    "correct": bool(per_arm[name].correct),
                    "score": float(per_arm[name].score),
                    "raw": per_arm[name].raw[:cfg.raw_chars],
                    "stats": packet_stats[name],
                }
                for name in per_arm
            },
            "attribution": att.as_dict(),
        }
        if errors:
            record["errors"] = errors
        return record

    def _measure(self, ar, rr: ReadResult, item: Mapping, item_for_reader: Mapping) -> dict:
        """The two flags that need both the served context and the reader's answer."""
        notes = ar.notes
        stale_value = self._stale_value(item, notes)
        current = item_for_reader.get("gold")
        stale_detectable = bool(stale_value) and bool(rr.raw)
        stale_hit = bool(
            stale_detectable
            and not rr.correct
            and contains_value(rr.raw, str(stale_value))
            and not (current and contains_value(rr.raw, str(current))))

        if self.probe is not None:
            probe_positive = bool(self.probe(ar.context, item_for_reader))
        else:
            # documented weak fallback: a carrier that literally quotes the answer is trivially
            # readable by a restricted readout.  Never claimed to be a KV-level probe (§5.3 r7).
            probe_positive = bool(notes.get("surface_leak"))
        return {
            "stale_value": stale_value,
            "stale_detectable": stale_detectable,
            "stale_hit": stale_hit,
            "probe_positive": probe_positive,
            "probe_source": "injected" if self.probe is not None else "surface_leak_fallback",
        }

    def _check_freeze(self, built: Mapping):
        """§5.3 rule 1, verified per item.  A violation is a bug here, so it raises."""
        groups = [
            ("B arms", [built[n] for n in ("B1_source_only", "B2_carrier_only",
                                           "B3_source_carrier", "B4_mutated_source")
                        if n in built]),
            ("A2/A0_shuffled", [built[n] for n in ("A2_seed_closure", "A0_shuffled")
                                if n in built]),
        ]
        frozen_key = None
        for where, results in groups:
            if len(results) < 2:
                continue
            try:
                key = assert_frozen_routing(results, where=where)
            except ProfilerError:
                if self.cfg.strict:
                    raise
                return False, frozen_key
            if where == "B arms":
                frozen_key = key
        return True, frozen_key

    # ------------------------------------------------------------------ many items

    def run(self, items: Sequence, arms=None) -> list:
        """Run every item sequentially, checkpointing as we go; resumable by ``item_id``."""
        if isinstance(items, (str, bytes, Mapping)):
            raise ProfilerError("run() takes a sequence of item mappings, got %r"
                                % (type(items).__name__,))
        items = list(items)
        done = self._load_done()
        out = []
        handle = None
        try:
            for item in items:
                if not isinstance(item, Mapping):
                    raise ProfilerError("items must be mappings, got %r" % (type(item).__name__,))
                item_id = str(item.get("item_id") or item.get("id") or "")
                if not item_id:
                    raise ProfilerError("item has no item_id; checkpointing keys on it")
                if item_id in done:
                    out.append(done[item_id])
                    continue
                record = self.run_item(item, arms)
                if self.cfg.checkpoint:
                    if handle is None:
                        parent = os.path.dirname(os.path.abspath(self.cfg.checkpoint))
                        if parent and not os.path.isdir(parent):
                            os.makedirs(parent, exist_ok=True)
                        handle = open(self.cfg.checkpoint, "a", encoding="utf-8")
                    handle.write(json.dumps(record, ensure_ascii=False, sort_keys=True) + "\n")
                    handle.flush()
                out.append(record)
        finally:
            if handle is not None:
                handle.close()
        return out

    def _load_done(self) -> dict:
        """Records already on disk, keyed by item id; the LAST record for an id wins."""
        path = self.cfg.checkpoint
        self.checkpoint_skipped = 0
        if not path or not os.path.exists(path):
            return {}
        done = {}
        with open(path, "r", encoding="utf-8") as fh:
            for line in fh:
                line = line.strip()
                if not line:
                    continue
                try:
                    rec = json.loads(line)
                except ValueError:
                    # a process killed mid-write leaves a truncated final line: skip and count,
                    # never repair -- a half record would resume as a complete one.
                    self.checkpoint_skipped += 1
                    continue
                if not isinstance(rec, dict) or not rec.get("item_id"):
                    self.checkpoint_skipped += 1
                    continue
                done[str(rec["item_id"])] = rec
        return done


# ---------------------------------------------------------------- reporting


def summarize(records: Sequence) -> dict:
    """Per-arm accuracy, the three §5.2 core deltas, and the label distribution.

    Deltas are computed on the items where **both** arms of the pair ran, so a partial run cannot
    move a delta by changing the denominator (the standard way this number gets faked):
    ``G_closure = Acc(A2) - Acc(A1)``, ``L_router = Acc(A3) - Acc(A2)``,
    ``L_closure = Acc(A4) - Acc(A3)``.
    """
    records = list(records)
    if not records:
        raise ProfilerError("summarize() needs at least one record")
    totals, hits = {}, {}
    labels = {}
    for rec in records:
        if not isinstance(rec, Mapping) or "arms" not in rec:
            raise ProfilerError("summarize() takes run_item records, got %r" % (type(rec),))
        for arm, res in (rec.get("arms") or {}).items():
            totals[arm] = totals.get(arm, 0) + 1
            hits[arm] = hits.get(arm, 0) + (1 if res.get("correct") else 0)
        lab = (rec.get("attribution") or {}).get("label", "ambiguous")
        labels[lab] = labels.get(lab, 0) + 1

    def paired(a, b):
        both = [r for r in records
                if a in (r.get("arms") or {}) and b in (r.get("arms") or {})]
        if not both:
            return None
        acc = lambda arm: sum(1 for r in both if r["arms"][arm]["correct"]) / len(both)
        return {"n": len(both), "delta": acc(b) - acc(a), "acc_%s" % a: acc(a),
                "acc_%s" % b: acc(b)}

    return {
        "n_items": len(records),
        "accuracy": {arm: hits[arm] / totals[arm] for arm in sorted(totals)},
        "n_by_arm": dict(sorted(totals.items())),
        "labels": dict(sorted(labels.items())),
        "G_closure": paired("A1_seed_only", "A2_seed_closure"),
        "L_router": paired("A2_seed_closure", "A3_oracle_seed"),
        "L_closure": paired("A3_oracle_seed", "A4_gold_minimal"),
        "frozen_ok": all(bool(r.get("frozen_ok", True)) for r in records),
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


class _CountingReader:
    """Fake reader with a known answer: correct iff the served context states the gold value.

    ``prefer`` picks which value it "reads" when several are present: ``first`` walks the context
    in order (a source-leading reader), ``last`` takes the latest occurrence (a recency/carrier
    dominated reader).  The second one is how the selfcheck manufactures a stale answer.
    """

    def __init__(self, prefer: str = "first"):
        self.prefer = prefer
        self.calls = 0
        self.seen = []

    def __call__(self, context, item):
        self.calls += 1
        self.seen.append((item.get("item_id"), len(context)))
        gold = str(item.get("gold") or "")
        stale = str(item.get("stale_value") or "")
        found = []
        for value in (gold, stale):
            if value and contains_value(context, value):
                found.append((context.rindex(value) if self.prefer == "last"
                              else context.index(value), value))
        if not found:
            return ReadResult(correct=False, score=-1.0, raw="")
        found.sort(reverse=(self.prefer == "last"))
        pick = found[0][1]
        return ReadResult(correct=(pick == gold), score=1.0 if pick == gold else -1.0,
                          raw="the answer is %s" % pick)


def _selfcheck() -> None:
    import shutil
    import tempfile

    from tracepack.profiler.interventions import _fixture

    graph, item, gold = _fixture()
    lgraph, litem, lgold = _fixture(leaky_carrier=True)
    litem = dict(litem, item_id="sc#leaky#0")
    items = [item, litem, dict(item, item_id="sc#third#0")]
    graphs = {"sc#decision_source#0": graph, "sc#leaky#0": lgraph, "sc#third#0": graph}

    reader = _CountingReader()
    cfg = ProfilerConfig(budget=2048, seed=3)
    runner = ProfilerRunner(graphs, reader, cfg)

    # ---- one item, all ten arms, with a known-correct expectation per arm -------------------
    rec = runner.run_item(item)
    arms = rec["arms"]
    assert set(arms) == set(cfg.arms), sorted(set(cfg.arms) - set(arms))
    assert arms["A0_null"]["correct"] is False, "no evidence must not answer"
    assert arms["A0_shuffled"]["correct"] is False, "shuffled evidence must not answer"
    assert arms["A3_oracle_seed"]["correct"] is True
    assert arms["A4_gold_minimal"]["correct"] is True
    assert arms["B1_source_only"]["correct"] is True, "the source states the value"
    assert arms["B2_carrier_only"]["correct"] is False, "the clean carrier does not"
    assert arms["B3_source_carrier"]["correct"] is True
    assert arms["B4_mutated_source"]["correct"] is True, "the reader follows the mutated source"
    assert rec["frozen_ok"] is True and rec["frozen_key"]
    # the B arms really did share one plan, and A0_shuffled really did freeze against A2
    keys = {arms[n]["stats"]["frozen_key"] for n in ("B1_source_only", "B2_carrier_only",
                                                     "B3_source_carrier", "B4_mutated_source")}
    assert len(keys) == 1 and keys.pop() == rec["frozen_key"]
    assert arms["A0_shuffled"]["stats"]["frozen_key"] == \
        arms["A2_seed_closure"]["stats"]["frozen_key"]
    chars = {arms[n]["stats"]["context_chars"] for n in ("B1_source_only", "B2_carrier_only",
                                                         "B3_source_carrier",
                                                         "B4_mutated_source")}
    assert len(chars) == 1, "equal-length replacement across the frozen plan"
    assert rec["attribution"]["label"] in ("ok", "carrier_loss"), rec["attribution"]

    # B4 was scored against the NONCE, not the old gold (contract 2)
    b4 = arms["B4_mutated_source"]["stats"]
    assert b4["nonce"] and b4["stale_value"] == gold and b4["mutation_vacuous"] is False
    assert b4["nonce"] in arms["B4_mutated_source"]["raw"]

    # determinism: same runner state, same record
    again = ProfilerRunner(graphs, _CountingReader(), ProfilerConfig(budget=2048, seed=3))
    rec2 = again.run_item(item)
    assert json.dumps(rec2, sort_keys=True) == json.dumps(rec, sort_keys=True), "not deterministic"

    # ---- carrier_loss is produced end to end on the clean-carrier item ----------------------
    only_b = runner.run_item(item, arms=("A2_seed_closure", "B1_source_only", "B2_carrier_only"))
    assert only_b["attribution"]["label"] == "carrier_loss", only_b["attribution"]
    assert only_b["arms"]["B2_carrier_only"]["stats"]["probe_positive"] is False

    # ---- the leaking carrier turns the same pattern into reader_miss ------------------------
    leak = runner.run_item(litem, arms=("A2_seed_closure", "B1_source_only", "B2_carrier_only"))
    assert leak["arms"]["B2_carrier_only"]["stats"]["surface_leak"] is True
    assert leak["arms"]["B2_carrier_only"]["correct"] is True, "a leaking carrier answers"

    # ---- a carrier-dominated reader on a mutated source is a stale_conflict -----------------
    stale_runner = ProfilerRunner(graphs, _CountingReader(prefer="last"),
                                  ProfilerConfig(budget=2048, seed=3))
    st = stale_runner.run_item(litem, arms=("A2_seed_closure", "B4_mutated_source"))
    b4s = st["arms"]["B4_mutated_source"]
    assert b4s["correct"] is False and b4s["stats"]["stale_hit"] is True
    assert b4s["stats"]["stale_carriers"] == ["sum1"]
    assert st["attribution"]["label"] == "stale_conflict", st["attribution"]
    assert st["attribution"]["confident"] is True

    # ---- an injected probe overrides the textual fallback -----------------------------------
    probed = ProfilerRunner(graphs, _CountingReader(), cfg, probe=lambda ctx, it: True)
    pr = probed.run_item(item, arms=("A2_seed_closure", "B1_source_only", "B2_carrier_only"))
    assert pr["arms"]["B2_carrier_only"]["stats"]["probe_source"] == "injected"
    assert pr["attribution"]["label"] == "reader_miss", pr["attribution"]

    # ---- checkpoint / resume ------------------------------------------------------------------
    tmp = tempfile.mkdtemp(prefix="tracepack_profiler_")
    try:
        path = os.path.join(tmp, "nested", "ck.jsonl")
        r1 = _CountingReader()
        run1 = ProfilerRunner(graphs, r1, ProfilerConfig(budget=2048, seed=3, checkpoint=path))
        out1 = run1.run(items[:2], arms=("A2_seed_closure", "A4_gold_minimal"))
        assert len(out1) == 2 and os.path.exists(path)
        calls_after_first = r1.calls
        assert calls_after_first == 4, calls_after_first

        r2 = _CountingReader()
        run2 = ProfilerRunner(graphs, r2, ProfilerConfig(budget=2048, seed=3, checkpoint=path))
        out2 = run2.run(items, arms=("A2_seed_closure", "A4_gold_minimal"))
        assert len(out2) == 3, "resume must still return every item"
        assert r2.calls == 2, "only the unfinished item may be read again, got %d" % r2.calls
        assert [r["item_id"] for r in out2] == [i["item_id"] for i in items]
        assert out2[0] == out1[0], "a resumed record must come back byte-identical"

        # a truncated final line (process killed mid-write) is skipped and counted, not repaired
        with open(path, "a", encoding="utf-8") as fh:
            fh.write('{"item_id": "sc#third#0", "arms": {"A2_')
        r3 = _CountingReader()
        run3 = ProfilerRunner(graphs, r3, ProfilerConfig(budget=2048, seed=3, checkpoint=path))
        out3 = run3.run(items, arms=("A2_seed_closure", "A4_gold_minimal"))
        assert run3.checkpoint_skipped == 1, run3.checkpoint_skipped
        assert r3.calls == 0 and len(out3) == 3, "the complete records must still resume"

        # summarize over a real run
        summary = summarize(out2)
        assert summary["n_items"] == 3 and summary["frozen_ok"] is True
        assert summary["accuracy"]["A4_gold_minimal"] == 1.0
        assert summary["G_closure"] is None, "A1 was not run, so the delta must be None"
        lc = summary["L_closure"]
        assert lc is None or lc["n"] == 3
    finally:
        shutil.rmtree(tmp, ignore_errors=True)

    # ---- full-ladder summary, including the three §5.2 deltas ---------------------------------
    full = ProfilerRunner(graphs, _CountingReader(), cfg).run(items)
    s = summarize(full)
    assert s["n_items"] == 3
    assert s["G_closure"]["n"] == 3 and s["L_router"]["n"] == 3 and s["L_closure"]["n"] == 3
    assert s["accuracy"]["A0_null"] == 0.0, "the prior/leak floor must be at zero here"
    assert sum(s["labels"].values()) == 3

    # ---------------- fault injection ----------------------------------------------------------
    _expect(ProfilerError, ProfilerConfig, -1)
    _expect(ProfilerError, ProfilerConfig, 2048, ())
    _expect(ProfilerError, ProfilerConfig, 2048, ("NOT_AN_ARM",))
    _expect(ProfilerError, ProfilerConfig, 2048, "A2_seed_closure")
    _expect(ProfilerError, ProfilerConfig, 2048, DEFAULT_ARMS, 0, 17)
    _expect(ProfilerError, ProfilerConfig, 2048, DEFAULT_ARMS, 0, "hybrid_pin", 0)
    _expect(ProfilerError, ProfilerConfig, 2048, DEFAULT_ARMS, 0, "hybrid_pin", 8, 17)

    _expect(ProfilerError, ProfilerRunner, None, reader)
    _expect(ProfilerError, ProfilerRunner, graphs, "not callable")
    _expect(ProfilerError, ProfilerRunner, graphs, reader, "not a config")
    _expect(ProfilerError, ProfilerRunner, graphs, reader, cfg, probe=17)

    # a graph provider that cannot resolve the item must fail loudly, not serve an empty context
    _expect(ProfilerError, ProfilerRunner({}, reader, cfg).run_item, item)
    _expect(ProfilerError, ProfilerRunner(lambda it: None, reader, cfg).run_item, item)

    # readers that return something whose correctness we would have to guess
    _expect(ProfilerError, ProfilerRunner(graphs, lambda c, i: None, cfg).run_item, item)
    _expect(ProfilerError, ProfilerRunner(graphs, lambda c, i: "yes", cfg).run_item, item)
    _expect(ProfilerError, ProfilerRunner(graphs, lambda c, i: {"score": 1.0}, cfg).run_item,
            item)
    # ... and a reader that crashes is re-raised as a ProfilerError naming the arm
    def _boom(context, it):
        raise ZeroDivisionError("model died")
    _expect(ProfilerError, ProfilerRunner(graphs, _boom, cfg).run_item, item)
    # a mapping/duck-typed result IS accepted
    ok_rec = ProfilerRunner(graphs, lambda c, i: {"correct": True, "score": 2.0, "raw": "x"},
                            cfg).run_item(item, arms=("A2_seed_closure",))
    assert ok_rec["arms"]["A2_seed_closure"]["score"] == 2.0

    # bad call shapes
    r = ProfilerRunner(graphs, reader, cfg)
    _expect(ProfilerError, r.run_item, ["not", "a", "mapping"])
    _expect(ProfilerError, r.run_item, {k: v for k, v in item.items() if k != "item_id"})
    _expect(ProfilerError, r.run_item, item, ("NOT_AN_ARM",))
    _expect(ProfilerError, r.run_item, item, "A2_seed_closure")
    _expect(ProfilerError, r.run, item)
    _expect(ProfilerError, r.run, [["not", "a", "mapping"]])
    _expect(ProfilerError, summarize, [])
    _expect(ProfilerError, summarize, [{"no": "arms"}])

    # strict=False records an arm-build failure instead of raising -- and still labels the item
    loose = ProfilerRunner(graphs, _CountingReader(),
                           ProfilerConfig(budget=2048, seed=3, strict=False))
    broken = {k: v for k, v in item.items() if k != "gold_seed"}
    br = loose.run_item(broken, arms=("A2_seed_closure", "A3_oracle_seed", "A4_gold_minimal"))
    assert "A3_oracle_seed" in br["errors"] and "A3_oracle_seed" not in br["arms"]
    assert br["attribution"]["label"] in ("ok", "closure_miss", "ambiguous")

    print("runner.py selfcheck OK: %d arms x %d items, labels=%s, G_closure=%+.2f"
          % (len(cfg.arms), s["n_items"], s["labels"], s["G_closure"]["delta"]))


if __name__ == "__main__":
    _selfcheck()
