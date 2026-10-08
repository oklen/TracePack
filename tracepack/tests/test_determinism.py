"""tracepack.tests.test_determinism -- proposal §8.1 contract 1.

| §8.1 | claim                                                                       | class here |
|------|-----------------------------------------------------------------------------|------------|
| #1   | fixed query / graph / scores / policy / tokenizer / budget => same packet hash | `Contract01PacketHash` |

Contracts 2-10 live in `test_budget.py`, `test_closure.py` and `test_adapter.py`.

Design decisions in these tests
-------------------------------

* **A hash that never changes is not deterministic, it is constant.**  Half of this file is
  ``Contract01HashIsSensitive``: flipping the query, the budget, the closure mode, the
  representation policy, the render order, the seed set or the query mode must each move the
  digest.  Without those, ``digest() -> "cafe"`` would pass every stability test in the suite.

* **Determinism is checked ACROSS PROCESSES, with ``PYTHONHASHSEED`` varied.**  Same-process
  repetition cannot see the one failure mode that actually bites: a set or dict iteration order
  leaking into the output.  ``test_01d`` re-runs the whole pipeline in three subprocesses with
  ``PYTHONHASHSEED`` = 0, 1 and ``random`` and demands byte-identical digests.  (The router's
  dense arm is keyed BLAKE2b for exactly this reason -- builtin ``hash()`` is salted per
  process.)

* **Input order is a factor too.**  ``TraceGraph`` sorts events by ``(timestamp, event_id)`` and
  closure sorts every out-edge list, so handing the adapter's output in reverse must produce the
  same packet.  ``test_01c`` rebuilds each fixture reversed and compares digests.

* **Scores are excluded from the digest by design** (``PacketManifest.digest`` hashes seed ids,
  sources, ranks and pinned flags, not floats).  ``test_01g`` pins that: perturbing a score
  without changing the ranking must NOT change the hash, because retriever float noise is not a
  change to the served context.

Pure stdlib + unittest; the only subprocess is a re-run of this same package.
"""
from __future__ import annotations

import json
import os
import subprocess
import sys
import tempfile
import unittest

if __package__ in (None, ""):  # pragma: no cover - direct execution
    sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(
        os.path.abspath(__file__)))))

from tracepack.core.assembler import AssemblerConfig, BudgetAssembler
from tracepack.core.closure import ClosureConfig, TypedClosure
from tracepack.core.graph import TraceGraph
from tracepack.core.router import RouterConfig, make_router
from tracepack.core.schema import SchemaError
from tracepack.tests.fixtures import tiny_trace as F

REPO_ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

#: the canonical scenario every determinism check runs; changing it changes the expected digests
SCENARIO = dict(query="which exact value did step 3 print for the retry budget?",
                router="hybrid_pin", k=5, closure_mode="native", query_mode="why",
                repr_policy="source_plus_carrier", budget=512, n_graphs=3, n_events=18)

#: the driver used by the cross-process check -- deliberately imports nothing from this module
_SUBPROCESS_DRIVER = r"""
import json, sys
sys.path.insert(0, {root!r})
from tracepack.tests.fixtures import tiny_trace as F
from tracepack.core.assembler import AssemblerConfig, BudgetAssembler
from tracepack.core.closure import ClosureConfig, TypedClosure
from tracepack.core.router import RouterConfig, make_router
S = json.loads({scenario!r})
out = []
for _, graph in F.random_graphs(S["n_graphs"], n_events=S["n_events"]):
    router = make_router(S["router"], RouterConfig(k=S["k"]))
    seeds = router.retrieve(S["query"], graph, S["k"])
    closure = TypedClosure(ClosureConfig(mode=S["closure_mode"])).close(
        S["query"], seeds, graph, query_mode=S["query_mode"])
    packet = BudgetAssembler(AssemblerConfig(repr_policy=S["repr_policy"])).assemble(
        S["query"], closure, graph, S["budget"], seeds=seeds, query_mode=S["query_mode"])
    out.append(packet.manifest.digest())
print(" ".join(out))
"""


# ---------------------------------------------------------------- instruments


def run_pipeline(graph, *, query=None, router=None, k=None, closure_mode=None,
                 query_mode=None, repr_policy=None, budget=None, order="chronological",
                 seeds=None, header=""):
    """One full route -> close -> assemble pass; returns the packet.

    Every knob defaults to :data:`SCENARIO` so a test can vary exactly one factor and attribute
    a digest change to it.
    """
    query = SCENARIO["query"] if query is None else query
    router = SCENARIO["router"] if router is None else router
    k = SCENARIO["k"] if k is None else k
    closure_mode = SCENARIO["closure_mode"] if closure_mode is None else closure_mode
    query_mode = SCENARIO["query_mode"] if query_mode is None else query_mode
    repr_policy = SCENARIO["repr_policy"] if repr_policy is None else repr_policy
    budget = SCENARIO["budget"] if budget is None else budget
    if seeds is None:
        seeds = make_router(router, RouterConfig(k=k)).retrieve(query, graph, k)
    closure = TypedClosure(ClosureConfig(mode=closure_mode)).close(
        query, seeds, graph, query_mode=query_mode)
    return BudgetAssembler(AssemblerConfig(repr_policy=repr_policy, order=order,
                                           header=header)).assemble(
        query, closure, graph, budget, seeds=seeds, query_mode=query_mode)


def fingerprint(packet) -> tuple:
    """Everything a caller can observe about a packet, in one comparable value."""
    m = packet.manifest
    return (m.digest(), packet.context, m.total_tokens, m.incomplete,
            tuple((e.event_id, e.repr_kind, e.token_cost, e.reason) for e in m.entries),
            tuple(m.missing_required), tuple(m.omitted_optional))


def reversed_graph(graph: TraceGraph) -> TraceGraph:
    """Same trace, handed to the graph constructor back to front."""
    return TraceGraph(list(reversed(graph.events)), list(reversed(graph.edges)))


def _subprocess_digests(hash_seed: str) -> str:
    env = dict(os.environ)
    env["PYTHONHASHSEED"] = hash_seed
    env.pop("PYTHONPATH", None)
    code = _SUBPROCESS_DRIVER.format(root=REPO_ROOT, scenario=json.dumps(SCENARIO))
    # run from a directory that does NOT contain the package: `python3 -c` puts the cwd on
    # sys.path, which would let the driver import tracepack even with a broken explicit path
    proc = subprocess.run([sys.executable, "-c", code], capture_output=True, text=True,
                          env=env, cwd=tempfile.gettempdir(), timeout=180)
    if proc.returncode != 0:  # pragma: no cover - surfaced as a test failure
        raise AssertionError("determinism subprocess failed (PYTHONHASHSEED=%s):\n%s"
                             % (hash_seed, proc.stderr[-2000:]))
    return proc.stdout.strip()


# ---------------------------------------------------------------- contract #1


class Contract01PacketHash(unittest.TestCase):
    """§8.1 #1: identical inputs give a byte-identical packet."""

    def test_01a_repeating_the_pipeline_reproduces_the_packet_exactly(self):
        for rng_seed, graph in F.random_graphs(SCENARIO["n_graphs"],
                                               n_events=SCENARIO["n_events"]):
            with self.subTest(seed=rng_seed):
                prints = {fingerprint(run_pipeline(graph)) for _ in range(3)}
                self.assertEqual(1, len(prints))

    def test_01b_fresh_module_objects_do_not_change_the_answer(self):
        """Routers cache their index; a stale cache would show up as a second-call difference."""
        graph = F.random_graph(11, 20)
        router = make_router("hybrid_pin", RouterConfig(k=5))
        first = [s.event_id for s in router.retrieve(SCENARIO["query"], graph, 5)]
        second = [s.event_id for s in router.retrieve(SCENARIO["query"], graph, 5)]
        third = [s.event_id for s in
                 make_router("hybrid_pin", RouterConfig(k=5)).retrieve(
                     SCENARIO["query"], graph, 5)]
        self.assertEqual(first, second)
        self.assertEqual(first, third)

    def test_01c_input_order_of_events_and_edges_does_not_matter(self):
        for name, graph in (("tool_chain", F.tool_chain_graph()),
                            ("exact_fit", F.exact_fit_graph()),
                            ("carrier", F.carrier_source_graph()),
                            ("random", F.random_graph(5, 20))):
            with self.subTest(graph=name):
                self.assertEqual(graph.event_ids, reversed_graph(graph).event_ids)
                self.assertEqual(fingerprint(run_pipeline(graph)),
                                 fingerprint(run_pipeline(reversed_graph(graph))))

    def test_01d_the_digest_survives_a_different_process_and_hash_seed(self):
        runs = {}
        for hash_seed in ("0", "1", "random"):
            runs[hash_seed] = _subprocess_digests(hash_seed)
            self.assertTrue(runs[hash_seed], "subprocess produced no digests")
        self.assertEqual(1, len(set(runs.values())),
                         "PYTHONHASHSEED changed the packet: %s" % runs)
        local = " ".join(
            run_pipeline(g).manifest.digest()
            for _, g in F.random_graphs(SCENARIO["n_graphs"], n_events=SCENARIO["n_events"]))
        self.assertEqual(local, runs["0"], "in-process and subprocess results disagree")

    def test_01e_the_closure_itself_is_order_stable(self):
        graph = F.random_graph(4, 22)
        seeds = make_router("hybrid", RouterConfig(k=6)).retrieve(SCENARIO["query"], graph, 6)
        runs = set()
        for _ in range(3):
            c = TypedClosure().close(SCENARIO["query"], seeds, graph, query_mode="why")
            runs.add((c.required, c.optional, c.relaxations,
                      tuple((s.child_id, s.parent_id, s.edge_type, s.rule) for s in c.steps)))
        self.assertEqual(1, len(runs))

    def test_01f_to_json_is_stable_and_carries_the_digest(self):
        graph = F.tool_chain_graph()
        packet = run_pipeline(graph, query="why vendor B", budget=4096)
        blob = packet.to_json()
        self.assertEqual(blob, packet.to_json())
        payload = json.loads(blob)
        self.assertEqual(packet.manifest.digest(), payload["manifest"]["digest"])
        self.assertEqual(packet.context, payload["context"])

    def test_01g_float_score_noise_does_not_move_the_hash(self):
        """The digest hashes seed ids/sources/ranks, not scores: retriever noise is not a change
        to the served context (`PacketManifest.digest` docstring)."""
        graph = F.tool_chain_graph()
        base = [F.seed("d1", rank=0, score=0.912), F.seed("tr1", rank=1, score=0.400)]
        jittered = [F.seed("d1", rank=0, score=0.913), F.seed("tr1", rank=1, score=0.399)]
        a = run_pipeline(graph, seeds=base, budget=4096, query="why vendor B")
        b = run_pipeline(graph, seeds=jittered, budget=4096, query="why vendor B")
        self.assertEqual(a.manifest.digest(), b.manifest.digest())

    def test_01h_the_fixture_generator_is_itself_reproducible(self):
        a, b = F.random_graph(42, 20), F.random_graph(42, 20)
        self.assertEqual(a.event_ids, b.event_ids)
        self.assertEqual([e.token_cost for e in a.events], [e.token_cost for e in b.events])
        self.assertEqual([(e.src_id, e.dst_id, e.edge_type) for e in a.edges],
                         [(e.src_id, e.dst_id, e.edge_type) for e in b.edges])
        self.assertNotEqual(a.event_ids, F.random_graph(43, 20).event_ids)


class Contract01HashIsSensitive(unittest.TestCase):
    """The other half of #1: the hash must MOVE when the served context could have moved.

    A digest that is stable because it ignores its inputs would pass every test in the class
    above; each case here changes exactly one factor.

    The graph is ``carrier_source_graph`` rather than a random one **on purpose**: every factor
    below has to be *observable* on the fixture, or "the digest did not move" would mean "this
    trace happened not to expose that factor" instead of "the hash is broken".  This fixture has
    a carrier with a second representation (so ``repr_policy`` bites), a seed that is
    chronologically last (so ``order`` bites) and a two-hop required chain (so ``closure_mode``
    bites).
    """

    BASE = dict(budget=4096, query="why did we buy from vendor B",
                closure_mode="native", query_mode="why",
                repr_policy="source_plus_carrier", router="hybrid", k=4)

    def setUp(self):
        self.graph = F.carrier_source_graph()
        self.base = self._run().manifest.digest()

    def _run(self, **kw):
        args = dict(self.BASE)
        args.update(kw)
        return run_pipeline(self.graph, **args)

    def _assert_moved(self, label, **kw):
        with self.subTest(factor=label):
            self.assertNotEqual(self.base, self._run(**kw).manifest.digest(),
                                "changing %s left the digest unchanged" % label)

    def test_every_pipeline_factor_moves_the_digest(self):
        self._assert_moved("query", query="what did the assistant say about the weather")
        self._assert_moved("budget", budget=64)
        self._assert_moved("closure_mode", closure_mode="off")
        self._assert_moved("query_mode", query_mode="lookup")
        self._assert_moved("repr_policy", repr_policy="source_only")
        self._assert_moved("render_order", order="seed_first")
        self._assert_moved("router", router="last_n")
        self._assert_moved("k", k=1)
        self._assert_moved("header", header="EVIDENCE PACKET")

    def test_query_and_budget_move_the_digest_on_random_traces_too(self):
        for rng_seed, graph in F.random_graphs(4, n_events=20):
            with self.subTest(seed=rng_seed):
                base = run_pipeline(graph).manifest.digest()
                self.assertNotEqual(base, run_pipeline(graph, budget=48).manifest.digest())
                self.assertNotEqual(
                    base, run_pipeline(graph, query="entirely unrelated wording").manifest.digest())

    def test_a_changed_seed_ranking_moves_the_digest(self):
        graph = F.tool_chain_graph()
        a = run_pipeline(graph, seeds=[F.seed("d1", 0), F.seed("tr1", 1)], budget=4096)
        b = run_pipeline(graph, seeds=[F.seed("tr1", 0), F.seed("d1", 1)], budget=4096)
        self.assertNotEqual(a.manifest.digest(), b.manifest.digest())

    def test_a_changed_event_cost_moves_the_digest(self):
        graph = F.tool_chain_graph()
        cheaper = TraceGraph(
            [e if e.event_id != "tr1"
             else F.event("tr1", "tool_result", 3, e.text, 3, group=F.ATOMIC_GROUP,
                          tool_call_id="t1", step_id="00000002")
             for e in graph.events],
            list(graph.edges))
        self.assertNotEqual(run_pipeline(graph, budget=4096).manifest.digest(),
                            run_pipeline(cheaper, budget=4096).manifest.digest())


# ---------------------------------------------------------------- adversarial


class AdversarialDeterminism(unittest.TestCase):
    """The failure modes that fake a deterministic result."""

    def test_non_finite_scores_raise_instead_of_silently_reordering(self):
        """NaN compares False against everything, so ``sorted`` degrades to insertion order --
        the top-k would then depend on dict iteration."""
        graph = F.pin_vs_topk_graph()

        def nan_embed(texts):
            return [[float("nan")] * 8 for _ in texts]

        def inf_embed(texts):
            return [[float("inf")] + [0.0] * 7 for _ in texts]

        for name, fn in (("nan", nan_embed), ("inf", inf_embed)):
            with self.subTest(embed=name):
                router = make_router("dense", RouterConfig(k=3, dense_dim=8), embed_fn=fn)
                with self.assertRaises(SchemaError):
                    router.retrieve("anything", graph, 3)

    def test_tied_timestamps_are_broken_by_event_id_not_by_input_order(self):
        events = [F.event("zz", "user", 5, "same time zz", 4),
                  F.event("aa", "user", 5, "same time aa", 4)]
        forward = TraceGraph(events, [])
        backward = TraceGraph(list(reversed(events)), [])
        self.assertEqual(("aa", "zz"), forward.event_ids)
        self.assertEqual(forward.event_ids, backward.event_ids)

    def test_duplicate_edges_are_collapsed_so_the_audit_trail_is_stable(self):
        graph = F.tool_chain_graph()
        doubled = TraceGraph(list(graph.events), list(graph.edges) + list(graph.edges))
        self.assertEqual(len(graph.edges), len(doubled.edges))
        self.assertEqual(len(graph.edges), doubled.stats()["n_duplicate_edges_dropped"])
        self.assertEqual(fingerprint(run_pipeline(graph, budget=4096)),
                         fingerprint(run_pipeline(doubled, budget=4096)))

    def test_router_arm_labels_come_from_the_router_not_the_seed_source(self):
        """`Seed.source` is a closed set with no `recency` member; the arm is `router.name`."""
        graph = F.random_graph(2, 16)
        for name in ("last_n", "lexical", "dense", "hybrid", "hybrid_pin"):
            with self.subTest(router=name):
                router = make_router(name, RouterConfig(k=3))
                self.assertEqual(name, router.name)
                for s in router.retrieve("anything at all", graph, 3):
                    self.assertIn(s.source, ("lexical", "dense", "rrf", "pin", "oracle"))


# ---------------------------------------------------------------- selfcheck


def _selfcheck() -> None:
    """Run the suite, then fault-inject the comparison instruments themselves."""
    graph = F.tool_chain_graph()
    good = run_pipeline(graph, budget=4096)

    # ---- instrument: `fingerprint` must distinguish two genuinely different packets
    other = run_pipeline(graph, budget=20)
    assert fingerprint(good) != fingerprint(other), \
        "fingerprint() cannot tell two different packets apart -- it proves nothing"
    assert fingerprint(good) == fingerprint(run_pipeline(graph, budget=4096))

    # ---- instrument: `reversed_graph` must really hand the constructor a different order
    rev = reversed_graph(graph)
    assert [e.event_id for e in reversed(graph.events)] != list(graph.event_ids), \
        "the fixture is palindromic; reversing it would test nothing"
    assert rev.event_ids == graph.event_ids

    # ---- instrument: the subprocess driver must fail loudly, not return an empty string
    broken = _SUBPROCESS_DRIVER.format(root="/nonexistent/path/for/fault/injection",
                                       scenario=json.dumps(SCENARIO))
    proc = subprocess.run([sys.executable, "-c", broken], capture_output=True, text=True,
                          cwd=tempfile.gettempdir(), timeout=120)
    assert proc.returncode != 0, \
        "the determinism driver survived a broken sys.path -- it is not importing the package"

    # ---- instrument: the digest must react to its inputs at all
    assert run_pipeline(graph, budget=4096, query="different").manifest.digest() \
        != good.manifest.digest(), "the digest ignores the query"
    assert run_pipeline(graph, budget=40).manifest.digest() != good.manifest.digest(), \
        "the digest ignores the budget"

    suite = unittest.TestLoader().loadTestsFromModule(sys.modules[__name__])
    result = unittest.TextTestRunner(verbosity=1, stream=sys.stdout).run(suite)
    if not result.wasSuccessful():
        raise SystemExit("test_determinism: %d failure(s), %d error(s)"
                         % (len(result.failures), len(result.errors)))
    print("test_determinism selfcheck OK: %d tests, 3 PYTHONHASHSEED subprocesses, "
          "4 instruments fault-injected" % result.testsRun)


if __name__ == "__main__":
    _selfcheck()
