"""tracepack.eval.run_eval -- the main experiment (proposal §6.2, §6.3).

Runs `Router x Closure x Representation` over the frozen dependency slice at the three frozen
budgets, plus the profiler arms, with ONE reader and ONE prompt template.

Two things this driver is strict about, because they are how such an experiment usually goes
wrong quietly:

* **The graph is normalized once per transcript and reused** for every arm/budget, so no arm can
  win by having been built from a differently-parsed trace.
* **Everything is checkpointed by (item, arm, budget)** and resumable; a partial run is analysed
  on the intersection of completed cells, never on whatever happens to be on disk.

    python3 tracepack/eval/run_eval.py --items data/tracepack_items.jsonl \
        --out data/tracepack_eval.jsonl --reader qwen [--budgets 1024,2048,4096]
"""
from __future__ import annotations

import argparse
import json
import os
import sys
import time
from collections import Counter

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

from tracepack.adapters.claude_code import ClaudeCodeAdapter
from tracepack.adapters.registry import adapter_for
from tracepack.eval.freeze_corpus import assert_frozen
from tracepack.core.assembler import AssemblerConfig, BudgetAssembler
from tracepack.core.closure import ClosureConfig, TypedClosure
from tracepack.core.router import RouterConfig, make_router
from tracepack.core.schema import Seed
from tracepack.core.textmatch import contains_value    # the ONE matcher (red team (3) and (12))
from tracepack.eval.reader import DeterministicFakeReader, QwenReader

ROUTERS = ("last_n", "lexical", "dense", "hybrid", "hybrid_pin", "oracle")
CLOSURES = ("off", "native", "full_ancestor", "oracle")
REPRS = ("source_only", "carrier_only", "source_plus_carrier")
BUDGETS = (1024, 2048, 4096)

#: the main grid is 6x4 at the primary budget; the representation axis and the other budgets are
#: swept only on the configurations that matter, to keep the run inside one GPU session.
MAIN_ROUTERS = ROUTERS
MAIN_CLOSURES = CLOSURES
SWEEP_CELLS = (("hybrid_pin", "native"), ("hybrid_pin", "off"), ("oracle", "native"))


def build_query(item):
    """Use the item's generated query when present (gen_queries.py), else the template wording.

    The first dry run showed why this matters: with template-only wording every retrieval arm sits
    at ~0.01 while the oracle router is at 0.79 -- the routers were being measured on a query that
    contains no retrievable content.  `query_source` is carried into every record so the analysis
    can stratify on it."""
    q = item.get("query")
    if q:
        assert item["gold"] not in q, "query must not leak the gold value"
        return q
    role = {"path": "file path", "fileline": "file:line reference", "hash": "identifier",
            "number": "numeric value", "version": "version number"}.get(item["type"], "value")
    mode = item["query_mode"]
    if mode == "why":
        q = "Which %s did the work in this session actually depend on?" % role
    elif mode == "state":
        q = "What is the current %s after the latest change in this session?" % role
    else:
        q = "Which %s appeared in this session?" % role
    if item["slice"] == "explicit_ref":
        ref = (item.get("explicit_ref") or {})
        sid = ref.get("step_id")
        if sid:
            q += " (see step %s)" % sid
    assert item["gold"] not in q, "query must not leak the gold value"
    return q


def seeds_for(router_name, item, graph, k, cfg):
    if router_name == "oracle":
        return [Seed(event_id=item["gold_seed"], score=1.0, source="oracle", rank=0, pinned=True)]
    r = make_router(router_name, cfg)
    return r.retrieve(build_query(item), graph, k)


def run(args):
    items = [json.loads(l) for l in open(os.path.expanduser(args.items), encoding="utf-8")]
    # The corpus must be immutable across stages.  One of the ten traces is the session that
    # built the dataset and Claude Code appends to it live, so without this check dataset
    # construction, packet assembly and profiler replay each read a different graph -- silently,
    # because the adapter's ids are positional and appending does not renumber anything.
    assert_frozen(items)
    out_path = os.path.expanduser(args.out)
    done = set()
    if os.path.exists(out_path):
        for line in open(out_path, encoding="utf-8"):
            try:
                r = json.loads(line)
                done.add((r["item_id"], r["arm"], r["budget"]))
            except Exception:
                pass
    print("[eval] %d items, %d cells already done" % (len(items), len(done)), flush=True)

    # Assembly needs the traces (Mac side); reading needs a GPU (worker side).  --emit splits the
    # two: packets are built once here and only their TEXT travels, so the worker never needs the
    # transcripts and every arm is guaranteed to have been assembled by the same code path.
    if args.emit:
        reader, model_id = None, "emit"
    else:
        reader = QwenReader() if args.reader == "qwen" else DeterministicFakeReader()
        model_id = getattr(reader, "model_path", "fake")
    graphs = {}
    fo = open(out_path, "a", encoding="utf-8")
    cfg = RouterConfig(k=args.k)
    t0 = time.time()
    n = 0
    stats = Counter()

    cells = []
    routers = tuple(args.routers.split(",")) if args.routers else MAIN_ROUTERS
    closures = tuple(args.closures.split(",")) if args.closures else MAIN_CLOSURES
    for rt in routers:
        for cl in closures:
            cells.append((rt, cl, "source_only", args.primary))
            if args.all_budgets:
                # exact-token rerun: the reduced grid at EVERY budget, no sweep subset
                for b in args.budgets:
                    if b != args.primary:
                        cells.append((rt, cl, "source_only", b))
    if not args.all_budgets:
        for rt, cl in SWEEP_CELLS:
            for b in args.budgets:
                if b != args.primary:
                    cells.append((rt, cl, "source_only", b))
        # The representation axis is DISABLED by default (--reprs to re-enable).  Measured on the
        # dry run: source_only / carrier_only / source_plus_carrier produced byte-identical packets
        # (same acc, evrec and token count) because the adapter mints no `materialized_text`
        # representation -- a carrier in a real trace is a separate EVENT (the compaction summary),
        # not an alternative rendering of a source event.  The carrier question therefore belongs
        # to the profiler's B-arms (source removed, carrier kept, budget frozen), and reporting a
        # repr sweep here would be reporting three copies of one number as if they were three
        # conditions.
        if args.reprs:
            for rp in REPRS:
                if rp != "source_only":
                    cells.append((rt, cl, rp, args.primary))
    seen = set()
    cells = [c for c in cells if not (c in seen or seen.add(c))]
    print("[eval] %d cells x %d items = %d reads" % (len(cells), len(items), len(cells) * len(items)),
          flush=True)

    for it in items:
        tr = it["transcript"]
        if tr not in graphs:
            graphs[tr] = adapter_for(tr).normalize(tr)
            stats["graphs"] += 1
        g = graphs[tr]
        query = build_query(it)
        seed_cache = {}
        for rt, cl, rp, budget in cells:
            arm = "%s|%s|%s" % (rt, cl, rp)
            key = (it["item_id"], arm, budget)
            if key in done:
                continue
            if rt not in seed_cache:
                seed_cache[rt] = seeds_for(rt, it, g, args.k, cfg)
            seeds = seed_cache[rt]
            oracle_req = {query: it["required_sources"]} if cl == "oracle" else None
            closure = TypedClosure(ClosureConfig(mode=cl)).close(
                query, seeds, g, query_mode=it["query_mode"], oracle_required=oracle_req)
            pkt = BudgetAssembler(AssemblerConfig(repr_policy=rp)).assemble(
                query, closure, g, budget, seeds=seeds, query_mode=it["query_mode"])
            m = pkt.manifest
            if args.emit:
                fo.write(json.dumps(dict(
                    item_id=it["item_id"], arm=arm, budget=budget, context=pkt.context,
                    gold=it["gold"], distractors=it["distractors"][:1],
                    query_mode=it["query_mode"], slice=it["slice"], session=it["session"],
                    router=rt, closure=cl, repr=rp,
                    tokens=m.total_tokens, n_entries=len(m.entries),
                    incomplete=int(m.incomplete), n_missing=len(m.missing_required),
                    seed_hit=int(it["gold_seed"] in set(s.event_id for s in seeds)),
                    evidence_recall=round(len(set(e.event_id for e in m.entries)
                                              & set(it["required_sources"]))
                                          / max(1, len(it["required_sources"])), 4),
                    closure_required=len(closure.required),
                    gold_in_context=int(contains_value(pkt.context, it["gold"])),
                    stale_value=it.get("stale_value"),
                    query_source=it.get("query_source", "template"),
                    digest=m.digest()[:16]), ensure_ascii=False) + "\n")
                fo.flush()
                n += 1
                continue
            res = reader(pkt.context, it)
            served = set(e.event_id for e in m.entries)
            req = set(it["required_sources"])
            rec = dict(item_id=it["item_id"], slice=it["slice"], session=it["session"],
                       arm=arm, router=rt, closure=cl, repr=rp, budget=budget,
                       correct=int(res.correct), score=res.score, raw=res.raw,
                       tokens=m.total_tokens, n_entries=len(m.entries),
                       incomplete=int(m.incomplete), n_missing=len(m.missing_required),
                       seed_hit=int(it["gold_seed"] in set(s.event_id for s in seeds)),
                       evidence_recall=round(len(served & req) / max(1, len(req)), 4),
                       closure_required=len(closure.required),
                       gold_in_context=int(contains_value(pkt.context, it["gold"])),
                       stale_hit=int(bool(it.get("stale_value")) and res.raw != ""
                                     and not res.correct),
                       query_source=it.get("query_source", "template"),
                       digest=m.digest()[:16], model=model_id)
            fo.write(json.dumps(rec, ensure_ascii=False) + "\n")
            fo.flush()
            n += 1
            stats[arm] += 1
            if n % 100 == 0:
                print("[eval] %d reads  %.1f/min  last=%s acc_so_far=%s" % (
                    n, 60 * n / max(1e-9, time.time() - t0), arm, ""), flush=True)
    fo.close()
    print("[eval] wrote %d new reads in %.1f min (graphs normalized: %d)" % (
        n, (time.time() - t0) / 60, stats["graphs"]), flush=True)
    print("TRACEPACK_EVAL_DONE", flush=True)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--items", default=os.path.expanduser("data/tracepack_items.jsonl"))
    ap.add_argument("--out", default=os.path.expanduser("data/tracepack_eval.jsonl"))
    ap.add_argument("--reader", default="fake", choices=("fake", "qwen"))
    ap.add_argument("--budgets", default="1024,2048,4096")
    ap.add_argument("--primary", type=int, default=2048)
    ap.add_argument("--k", type=int, default=8)
    ap.add_argument("--emit", action="store_true",
                    help="write packets (context + stats) instead of reading them")
    ap.add_argument("--routers", default="", help="comma list; default = the full 6")
    ap.add_argument("--closures", default="", help="comma list; default = the full 4")
    ap.add_argument("--all-budgets", action="store_true",
                    help="run the (reduced) grid at every --budgets value instead of the sweep")
    ap.add_argument("--reprs", action="store_true",
                    help="also sweep repr_policy (no-op with the current adapter, see run())")
    args = ap.parse_args()
    args.budgets = tuple(int(b) for b in args.budgets.split(","))
    run(args)


if __name__ == "__main__":
    main()
