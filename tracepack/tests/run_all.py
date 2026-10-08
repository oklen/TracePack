"""tracepack.tests.run_all -- discover and run every test, and report §8.1 contract by contract.

    python3 tracepack/tests/run_all.py            # run everything, exit non-zero on failure
    python3 tracepack/tests/run_all.py -v         # per-test output
    python3 tracepack/tests/run_all.py -k budget  # only modules whose name matches

Design decisions
----------------

* **Discovery is a glob + explicit import, not ``unittest.discover``.**  ``tracepack`` has no
  ``__init__.py`` (it is an implicit namespace package), and ``TestLoader.discover`` handles
  those differently across 3.9/3.11/3.12.  Importing ``test_*.py`` by file path is one line
  longer and behaves the same on every interpreter this project will meet.

* **The fixtures are checked before the tests are.**  ``tiny_trace._selfcheck()`` verifies the
  hand-written costs and known answers the whole suite is built on.  If a fixture constant has
  drifted, every downstream failure would be misattributed, so this runs first and aborts.

* **The report is per contract, not per module.**  §8.1 numbers the ten claims; a green "31
  tests passed" says nothing about whether contract #7 was covered at all.  Test *classes* carry
  the contract number in their name (``Contract07Supersedes``), which is how
  :func:`build_report` attributes results -- so a contract with zero tests is reported as
  ``NO COVERAGE`` rather than silently passing.

* **An expected failure is reported as a KNOWN GAP, never as a pass.**  ``expectedFailure``
  marks a contract the code does not currently satisfy; the exit code stays 0 (the suite is
  green, the gap is declared) but the line says ``PASS (1 known gap)`` and the gap is listed
  underneath with the test that documents it.  An *unexpected success* -- the gap got fixed --
  is a hard failure, because the test that documented it must then be un-marked.

Exit codes: 0 all green; 1 a test failed, errored or unexpectedly succeeded; 2 the fixtures or
the runner's own selfcheck failed.
"""
from __future__ import annotations

import glob
import importlib.util
import io
import os
import re
import sys
import time
import unittest

HERE = os.path.dirname(os.path.abspath(__file__))
REPO_ROOT = os.path.dirname(os.path.dirname(HERE))
if REPO_ROOT not in sys.path:
    sys.path.insert(0, REPO_ROOT)

from tracepack.tests.fixtures import tiny_trace  # noqa: E402

__all__ = ["discover", "build_report", "run", "CONTRACTS"]

#: proposal §8.1, verbatim.  A number with no test class is reported as NO COVERAGE.
CONTRACTS = {
    1: "same query/graph/scores/policy/tokenizer/budget => identical packet hash",
    2: "Budget Violation Rate == 0",
    3: "satisfiable => no dangling required dependency in the packet",
    4: "unsatisfiable => incomplete=True and missing_required non-empty",
    5: "a pinned event is never displaced by ordinary top-k results",
    6: "tool_call/tool_result atomic group is never split",
    7: "a SUPERSEDES test never returns the stale state by default (query_mode=state)",
    8: "adapter round-trip preserves native event IDs and strong edges",
    9: "carrier verification must match model + predicate + protocol",
    10: "an UNVERIFIED carrier never triggers source removal",
    # phase 2 (PLAN_phase2.md §5): the closure-ranking packs and the old packs' byte identity
    11: "tiered/evidence_first: weak-reached material never packs ahead of a seed; pinned leads",
    12: "an excerpt is a labelled last resort, chosen blind to the gold, never a silent cut",
    13: "chrono/bundle packets are byte-identical to the pre-phase-2 assembler (golden digests)",
    14: "value-source DEPENDS_ON edges: opt-in, only for values that entered through a tool_result, never for what the agent wrote itself",
    # WP4 (WP4_sources.md): the second-format adapters and the registry
    15: "pi / OpenHands adapters: native ids, id (or declared position) pairing, atomic groups, compaction; registry leaves Claude Code graphs byte-identical",
    # the deliverable (PLAN_condenser.md §3): the part that runs inside somebody else's agent loop
    16: "condenser: every recipe default cites a reading; the >=2-hop gate decides the closure and never whether a packet is served; the corpus is the forgotten events only; recall is deterministic, inside budget, and cannot raise into the agent loop",
    # the baselines (PLAN_baselines.md): the arms a comparison is only honest if they hold
    # the product (README): the Claude Code plugin
    18: "plugin: recall serves only records that left the context, verbatim, labelled and inside the budget; the after-compaction hook adds the latest outputs under the 8,000-char cap and fails open; the MCP server speaks the protocol and binds to the calling session; credentials are masked",
    17: "baselines: the new arm builder is byte-identical to the published one for B and C; each factor bites; identifier chasing reads only retrieved text; ContextWeaver follows Algorithm 1 (BFS ancestry bounded by W, non-ancestors keep the action and lose the observation, warmup keeps everything); a dead analyzer collapses the ancestry instead of faking a recency window; graph construction cannot see an instruction issued later",
    # 0.4.0: note-taking compaction in the plugin, and the LongMemEval harness in the repo
    19: "compaction notes: PreCompact prints the note-taking instructions (nothing when off); the restore brings back the user's own sentences verbatim, dated, picked by content, inside the budget and the char cap, and leaves room for tool records",
    20: "bench lme: every prompt and judge template is byte-identical to the study's; each arm compacts as declared; the bootstrap is the study's; the Claude Code bridge keeps text and roles; the driver reads back what a compaction left",
}

_CONTRACT_RE = re.compile(r"Contract(\d{1,2})")
_ADVERSARIAL = "adversarial / regression"


# ---------------------------------------------------------------- discovery


def discover(pattern: str = "test_*.py", keyword: str | None = None):
    """``[(module_name, module)]`` for every test file in this directory, sorted by name."""
    modules = []
    for path in sorted(glob.glob(os.path.join(HERE, pattern))):
        name = os.path.splitext(os.path.basename(path))[0]
        if keyword and keyword not in name:
            continue
        spec = importlib.util.spec_from_file_location("tracepack.tests.%s" % name, path)
        if spec is None or spec.loader is None:      # pragma: no cover - unreadable file
            raise ImportError("cannot load test module: %s" % path)
        module = importlib.util.module_from_spec(spec)
        sys.modules[spec.name] = module
        spec.loader.exec_module(module)
        modules.append((name, module))
    if not modules:
        raise SystemExit("run_all: no test modules matched %r in %s" % (pattern, HERE))
    return modules


def _bucket_of(test) -> object:
    """Which §8.1 contract a test belongs to, from its class name; else the adversarial bucket."""
    match = _CONTRACT_RE.search(type(test).__name__)
    if match:
        number = int(match.group(1))
        if number in CONTRACTS:
            return number
    return _ADVERSARIAL


def _owner(test):
    """The TestCase a result entry belongs to.

    A failure inside ``with self.subTest(...)`` is reported as a ``unittest.case._SubTest``
    whose ``id()`` carries the parameters (``...test_09b (field='model_id')``) and whose class
    name is ``_SubTest`` -- so both id lookup and class-name matching miss, and the failure
    would land in the adversarial bucket instead of its contract.  ``_SubTest.test_case`` is the
    real owner; this is why `_selfcheck` fault-injects a failing subTest.
    """
    parent = getattr(test, "test_case", None)
    return test if parent is None else parent


def _iter_tests(suite):
    for item in suite:
        if isinstance(item, unittest.TestSuite):
            for sub in _iter_tests(item):
                yield sub
        else:
            yield item


# ---------------------------------------------------------------- reporting


def build_report(tests, result, *, require_full_coverage: bool = True) -> dict:
    """Aggregate a finished run into ``{bucket: {...}}`` plus an ``ok`` verdict.

    ``require_full_coverage`` is False for a ``-k`` filtered run, where the contracts the
    filter excluded are missing on purpose and must not be reported as a failure.

    ``tests`` must be the list snapshotted *before* the run: ``TestSuite`` drops its references
    as it goes (``_removeTestAtIndex``), so iterating it afterwards yields ``None`` and every
    test would land in the adversarial bucket -- a silently wrong report is worse than no report.

    Attribution is by test *identity*, not by counting: a failure in
    ``Contract06Atomicity.test_06b`` marks contract 6 and nothing else.
    """
    tests = list(tests)
    if any(t is None for t in tests):
        raise ValueError("build_report received a consumed TestSuite; snapshot before running")
    buckets = {}
    ids = {}
    for test in tests:
        bucket = _bucket_of(test)
        b = buckets.setdefault(bucket, {"total": 0, "failed": [], "gaps": [],
                                        "unexpected": []})
        b["total"] += 1
        ids[test.id()] = bucket

    def _mark(collection, key):
        for entry in collection:
            test = entry[0] if isinstance(entry, tuple) else entry
            owner = _owner(test)
            bucket = ids.get(owner.id(), _bucket_of(owner))
            buckets.setdefault(bucket, {"total": 0, "failed": [], "gaps": [],
                                        "unexpected": []})[key].append(test.id())

    _mark(result.failures, "failed")
    _mark(result.errors, "failed")
    _mark(result.expectedFailures, "gaps")
    _mark(result.unexpectedSuccesses, "unexpected")

    ok = (not result.failures and not result.errors and not result.unexpectedSuccesses)
    missing = [n for n in CONTRACTS if n not in buckets]
    if require_full_coverage:
        ok = ok and not missing
    return {"buckets": buckets, "ok": ok, "no_coverage": missing,
            "n_tests": result.testsRun}


def print_report(report: dict, elapsed: float) -> None:
    print("\n" + "=" * 78)
    print("§8.1 contract coverage")
    print("=" * 78)
    buckets = report["buckets"]
    for number in sorted(CONTRACTS):
        entry = buckets.get(number)
        title = CONTRACTS[number]
        if entry is None:
            print("  #%-2d  NO COVERAGE   %s" % (number, title))
            continue
        if entry["failed"]:
            status = "FAIL (%d)" % len(entry["failed"])
        elif entry["unexpected"]:
            status = "STALE GAP"
        elif entry["gaps"]:
            status = "PASS*"
        else:
            status = "PASS"
        print("  #%-2d  %-11s %2d tests   %s" % (number, status, entry["total"], title))
        for gap in entry["gaps"]:
            print("        known gap: %s" % gap.split(".", 2)[-1])
        for bad in entry["failed"]:
            print("        FAILED:    %s" % bad)
        for bad in entry["unexpected"]:
            print("        gap closed -- remove the expectedFailure mark: %s" % bad)
    extra = buckets.get(_ADVERSARIAL)
    if extra:
        status = "FAIL (%d)" % len(extra["failed"]) if extra["failed"] else "PASS"
        print("  --   %-11s %2d tests   %s" % (status, extra["total"], _ADVERSARIAL))
        for bad in extra["failed"]:
            print("        FAILED:    %s" % bad)
    print("-" * 78)
    print("%d tests in %.2fs -- %s%s"
          % (report["n_tests"], elapsed,
             "ALL CONTRACTS GREEN" if report["ok"] else "FAILURES PRESENT",
             "  (* = passes with a declared known gap)"
             if any(b.get("gaps") for b in buckets.values()) else ""))


# ---------------------------------------------------------------- run


def run(argv=None) -> int:
    argv = list(sys.argv[1:] if argv is None else argv)
    verbose = "-v" in argv or "--verbose" in argv
    keyword = None
    if "-k" in argv:
        idx = argv.index("-k")
        if idx + 1 >= len(argv):
            raise SystemExit("run_all: -k needs a keyword")
        keyword = argv[idx + 1]

    # 1. the fixtures the whole suite rests on
    try:
        tiny_trace._selfcheck()
    except Exception as exc:  # noqa: BLE001
        print("FIXTURES BROKEN -- every downstream failure would be misattributed:\n  %s" % exc)
        return 2

    # 2. discover + load
    suite = unittest.TestSuite()
    names = []
    for name, module in discover(keyword=keyword):
        names.append(name)
        suite.addTests(unittest.TestLoader().loadTestsFromModule(module))
    print("run_all: %d module(s): %s" % (len(names), ", ".join(names)))

    # 3. run -- snapshot the tests first, the suite empties itself as it runs
    tests = list(_iter_tests(suite))
    start = time.time()
    result = unittest.TextTestRunner(verbosity=2 if verbose else 1,
                                     stream=sys.stdout).run(suite)
    report = build_report(tests, result, require_full_coverage=keyword is None)
    print_report(report, time.time() - start)
    if report["no_coverage"]:
        print("run_all: contracts with NO test class: %s%s"
              % (report["no_coverage"],
                 " (excluded by -k %s)" % keyword if keyword else ""))
    return 0 if report["ok"] else 1


# ---------------------------------------------------------------- selfcheck


def _selfcheck() -> None:
    """Fault-inject the runner itself: a runner that cannot go red is not a test runner.

    Three injections: a failing test must be attributed to its contract and flip ``ok``; an
    expected failure must be reported as a known gap WITHOUT flipping ``ok``; and an unexpected
    success (a gap that got fixed but is still marked) must flip ``ok``.
    """

    class Contract06Injected(unittest.TestCase):
        def test_passes(self):
            self.assertTrue(True)

        def test_fails(self):
            self.assertEqual(1, 2, "deliberate failure")

    class Contract09Injected(unittest.TestCase):
        @unittest.expectedFailure
        def test_known_gap(self):
            self.assertEqual(1, 2)

        @unittest.expectedFailure
        def test_gap_already_fixed(self):
            self.assertEqual(1, 1)

    class Contract07Injected(unittest.TestCase):
        def test_subtest_fails(self):
            for value in (1, 2):
                with self.subTest(value=value):
                    self.assertEqual(1, value)

    class AdversarialInjected(unittest.TestCase):
        def test_ok(self):
            self.assertTrue(True)

    def _run(*classes):
        suite = unittest.TestSuite()
        loader = unittest.TestLoader()
        for cls in classes:
            suite.addTests(loader.loadTestsFromTestCase(cls))
        tests = list(_iter_tests(suite))
        res = unittest.TextTestRunner(verbosity=0, stream=io.StringIO()).run(suite)
        return tests, res

    # --- injection 1: a failing test is attributed to contract 6 and turns the report red
    tests, res = _run(Contract06Injected, AdversarialInjected)
    rep = build_report(tests, res)
    assert not rep["ok"], "build_report called a run with a failing test OK"
    assert len(rep["buckets"][6]["failed"]) == 1, \
        "the failure was not attributed to contract 6: %r" % (rep["buckets"][6],)
    assert rep["buckets"][6]["total"] == 2
    assert rep["buckets"][_ADVERSARIAL]["failed"] == []
    assert sorted(rep["no_coverage"]) == sorted(n for n in CONTRACTS if n != 6)

    # --- injection 2: an expected failure is a declared gap, not a failure
    tests, res = _run(Contract09Injected)
    rep = build_report(tests, res)
    assert rep["buckets"][9]["gaps"], "an expectedFailure was not reported as a known gap"
    assert rep["buckets"][9]["unexpected"], "an unexpected success was not reported"
    assert not rep["ok"], "a gap that got fixed must force the mark to be removed"

    # --- injection 3: an all-green run is green, and every contract is attributed
    class Contract01Injected(unittest.TestCase):
        def test_ok(self):
            self.assertTrue(True)

    tests, res = _run(Contract01Injected)
    rep = build_report(tests, res)
    assert rep["buckets"][1]["total"] == 1 and not rep["buckets"][1]["failed"]
    assert not rep["ok"], "uncovered contracts must not be reported as green"
    assert _bucket_of(Contract01Injected("test_ok")) == 1
    assert _bucket_of(AdversarialInjected("test_ok")) == _ADVERSARIAL

    # --- injection 4: a failure inside subTest must reach its contract, not the fallback
    tests, res = _run(Contract07Injected)
    rep = build_report(tests, res)
    assert rep["buckets"][7]["failed"], \
        "a subTest failure was not attributed to contract 7: %r" % (rep["buckets"],)
    assert _ADVERSARIAL not in rep["buckets"], \
        "a subTest failure leaked into the adversarial bucket"
    assert not rep["ok"]

    # --- injection 5: a consumed suite must be refused, not silently mis-attributed
    consumed, res = _run(Contract01Injected)
    try:
        build_report([None] + list(consumed), res)
    except ValueError:
        pass
    else:  # pragma: no cover
        raise AssertionError("build_report accepted a consumed TestSuite")

    # --- discovery must actually find this package's four test modules
    found = [name for name, _ in discover()]
    assert set(found) >= {"test_adapter", "test_budget", "test_closure", "test_determinism"}, \
        "discovery missed a test module: %r" % (found,)

    print("run_all selfcheck OK: attribution, known-gap and stale-gap paths fault-injected; "
          "discovery found %d module(s)" % len(found))


if __name__ == "__main__":
    if "--selfcheck" in sys.argv:
        _selfcheck()
        raise SystemExit(0)
    _selfcheck()
    raise SystemExit(run([a for a in sys.argv[1:] if a != "--selfcheck"]))
