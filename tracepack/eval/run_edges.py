"""tracepack.eval.run_edges -- does closure help when the dependency graph is better?

Everything is held fixed except the DEPENDS_ON edge set: same router, same closure algorithm, same
budget, same items, same reader.  Four edge sets:

  none    no DEPENDS_ON at all              -- the floor for this axis
  native  >=40-char literal overlap         -- what ships today (6,066 edges)
  llm     gold-blind LLM inference          -- 87,215 edges
  random  **the control**: same count and same index-distance distribution as `llm`, wired at
          random.  Denser edges pull more events into the packet, so more of the answer lands in
          it by luck; without this arm "llm helps" and "more edges helps" are indistinguishable.

What this CAN and CANNOT show (see RESULTS §5.1 and the redteam doc):
  CAN  -- accuracy.  Whether the answer's text reaches the reader is a fact about the trace, and
          does not depend on how `required_sources` was annotated.
  CANNOT -- "llm edges beat native edges at covering the gold chain".  The dependency slices are
          BUILT from native edges (`datasets.py` iterates events that already have one), so native
          scores 1.000 there by construction.  That also means this sample is native's home
          ground: an llm win here is conservative.

    python3 tracepack/eval/run_edges.py --emit out.jsonl
"""
from __future__ import annotations

import argparse
import json
import os
import sys
import time

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

from tracepack.adapters.claude_code import ClaudeCodeAdapter
from tracepack.adapters.registry import adapter_for
from tracepack.core.assembler import AssemblerConfig, BudgetAssembler
from tracepack.core.closure import ClosureConfig, TypedClosure
from tracepack.core.router import RouterConfig
from tracepack.eval.edge_arms import load_edges, with_edges
from tracepack.eval.freeze_corpus import assert_frozen
from tracepack.eval.run_eval import build_query, seeds_for
from tracepack.core.textmatch import contains_value

EDGE_SETS = ("none", "native", "llm", "random")
ROUTERS = ("hybrid_pin", "lexical")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--items", default=os.path.expanduser(
        "data/tracepack_items6.jsonl"))
    ap.add_argument("--edges", default=os.path.expanduser(
        "data/tracepack_llm_edges.jsonl"))
    ap.add_argument("--emit", required=True)
    ap.add_argument("--budget", type=int, default=2048)
    ap.add_argument("--k", type=int, default=8)
    args = ap.parse_args()

    items = [json.loads(l) for l in open(os.path.expanduser(args.items), encoding="utf-8")]
    assert_frozen(items)
    edges_by_sess = load_edges(args.edges)
    cfg = RouterConfig(k=args.k)
    by_tr = {}
    for it in items:
        by_tr.setdefault(it["transcript"], []).append(it)

    fo = open(os.path.expanduser(args.emit), "w", encoding="utf-8")
    n, t0 = 0, time.time()
    for tr, sub in sorted(by_tr.items()):
        sess = os.path.basename(tr)[:8]
        base = adapter_for(tr).normalize(tr)
        mine = edges_by_sess.get(sess, [])
        variants = {m: with_edges(base, m, llm_edges=mine) for m in EDGE_SETS}
        for it in sub:
            q = build_query(it)
            for rt in ROUTERS:
                # seeds come from the UNMODIFIED graph: the router never sees the edges, so any
                # difference between arms is the closure's doing, not the retriever's.
                seeds = seeds_for(rt, it, base, args.k, cfg)
                for cl, es in [("off", "native")] + [("native", m) for m in EDGE_SETS]:
                    g = variants[es]
                    closure = TypedClosure(ClosureConfig(mode=cl)).close(
                        q, seeds, g, query_mode=it["query_mode"])
                    pkt = BudgetAssembler(AssemblerConfig(repr_policy="source_only")).assemble(
                        q, closure, g, args.budget, seeds=seeds, query_mode=it["query_mode"])
                    m = pkt.manifest
                    arm = "%s|%s|%s" % (rt, cl, es if cl != "off" else "-")
                    fo.write(json.dumps(dict(
                        item_id=it["item_id"], arm=arm, budget=args.budget, context=pkt.context,
                        gold=it["gold"], distractors=it["distractors"][:1],
                        query_mode=it["query_mode"], slice=it["slice"], session=it["session"],
                        router=rt, closure=cl, repr="source_only", edges=es,
                        tokens=m.total_tokens, n_entries=len(m.entries),
                        incomplete=int(m.incomplete), n_missing=len(m.missing_required),
                        seed_hit=int(it["gold_seed"] in {s.event_id for s in seeds}),
                        evidence_recall=round(len({e.event_id for e in m.entries}
                                                  & set(it["required_sources"]))
                                              / max(1, len(it["required_sources"])), 4),
                        closure_required=len(closure.required),
                        gold_in_context=int(contains_value(pkt.context, it["gold"])),
                        stale_value=it.get("stale_value"),
                        query_source=it.get("query_source", "template"),
                        digest=m.digest()[:16]), ensure_ascii=False) + "\n")
                    n += 1
            if n % 200 == 0:
                fo.flush()
                print("[edges] %d reads  %.0f/min" % (n, 60 * n / max(1e-9, time.time() - t0)),
                      flush=True)
        print("  %s done (%d reads)" % (sess, n), flush=True)
    fo.close()
    print("[edges] wrote %d packets in %.1f min -> %s"
          % (n, (time.time() - t0) / 60, args.emit))
    print("EDGE_EMIT_DONE")


if __name__ == "__main__":
    main()
