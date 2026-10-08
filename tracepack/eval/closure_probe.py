"""tracepack.eval.closure_probe -- what the closure actually requires, and what capping would do.

    python3 tracepack/eval/closure_probe.py --items items6.jsonl                 # census + probe
    python3 tracepack/eval/closure_probe.py --items items6.jsonl --out cap.jsonl # full sweep

Red team round 2, finding (8).  A compaction summary MATERIALIZES every event before it, and
``_classify_native`` makes an unverified carrier's sources *required* (source-first).  So one seed
that is, or reaches, a compaction summary makes the entire session required: measured live, 874
of 878 events.  ``closure_required`` has been in every result row from the start -- it was
reporting this the whole time and nobody read it.

The probe answers the question the census cannot: **would capping the carrier fan-out fix it?**
It re-assembles every item against a graph with the high-fan-out MATERIALIZES edges deleted and
compares ``gold_in_context`` / ``evidence_recall``.  This needs no reader, because "did the answer
reach the packet" is decided at assembly time.  The measured answer is **no** -- capping removes
the saturation completely and makes delivery slightly *worse*, because the blow-up was acting as
a crude "serve more events" policy.  The real defect is that required events are packed by
timestamp, i.e. with no relation to the query at all.
"""
from __future__ import annotations

import argparse
import json
import os
import sys
import time
from collections import Counter, defaultdict

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

from tracepack.adapters.claude_code import ClaudeCodeAdapter
from tracepack.adapters.registry import adapter_for
from tracepack.core.assembler import AssemblerConfig, BudgetAssembler
from tracepack.core.closure import ClosureConfig, TypedClosure
from tracepack.core.graph import TraceGraph
from tracepack.core.router import RouterConfig
from tracepack.core.textmatch import contains_value
from tracepack.eval.run_eval import build_query, seeds_for
from tracepack.eval.stats import contrast

#: a carrier with more MATERIALIZES out-edges than this is a compaction boundary, not a summary
#: of one thing.  32 is a round number well above any per-tool carrier and far below the ~1000
#: seen on real compaction boundaries -- the probe is not sensitive to the exact value.
CAP = 32
#: required-set size above which the closure has clearly swallowed the session
SATURATED = 100


def capped_graph(graph, cap=CAP):
    """The same graph with MATERIALIZES edges out of high-fan-out carriers deleted."""
    out = Counter(e.src_id for e in graph.edges if e.edge_type == "MATERIALIZES")
    big = {sid for sid, c in out.items() if c > cap}
    return TraceGraph(list(graph.events),
                      [e for e in graph.edges
                       if not (e.edge_type == "MATERIALIZES" and e.src_id in big)]), out


def run(args):
    items = [json.loads(l) for l in open(args.items, encoding="utf-8") if l.strip()]
    bysess = defaultdict(list)
    for it in items:
        bysess[it["session"]].append(it)
    cfg = RouterConfig(k=args.k)
    fo = open(args.out, "w", encoding="utf-8") if args.out else None
    rows, t0 = [], time.time()

    for i, sess in enumerate(sorted(bysess)):
        its = bysess[sess] if args.out else bysess[sess][:args.sample]
        g = adapter_for(bysess[sess][0]["transcript"]).normalize(bysess[sess][0]["transcript"])
        gcap, fanout = capped_graph(g, args.cap)
        nmat = sum(1 for e in g.edges if e.edge_type == "MATERIALIZES")
        print("[%2d/%d] %-10s events=%-6d MATERIALIZES=%-6d carriers over cap=%-3d "
              "max fan-out=%-5d  %.1f min"
              % (i + 1, len(bysess), sess, len(g.events), nmat,
                 sum(1 for c in fanout.values() if c > args.cap),
                 max(fanout.values()) if fanout else 0, (time.time() - t0) / 60), flush=True)
        for it in its:
            q = build_query(it)
            for rt in args.routers:
                for arm, gg, mode in (("off", g, "off"), ("native", g, "native"),
                                      ("capped", gcap, "native")):
                    s = seeds_for(rt, it, gg, args.k, cfg)
                    cl = TypedClosure(ClosureConfig(mode=mode)).close(
                        q, s, gg, query_mode=it["query_mode"])
                    pkt = BudgetAssembler(AssemblerConfig()).assemble(
                        q, cl, gg, args.budget, seeds=s, query_mode=it["query_mode"])
                    m = pkt.manifest
                    row = dict(item_id=it["item_id"], session=sess, slice=it["slice"],
                               router=rt, arm=arm, required=len(cl.required),
                               entries=len(m.entries), tokens=m.total_tokens,
                               incomplete=int(m.incomplete),
                               gold_in_context=int(contains_value(pkt.context, it["gold"])),
                               evidence_recall=round(
                                   len({e.event_id for e in m.entries}
                                       & set(it["required_sources"]))
                                   / max(1, len(it["required_sources"])), 4))
                    rows.append(row)
                    if fo:
                        fo.write(json.dumps(row, ensure_ascii=False) + "\n")
        if fo:
            fo.flush()
    if fo:
        fo.close()
    report(rows, args)
    return 0


def report(rows, args):
    cell = lambda rt, arm: {r["item_id"]: r for r in rows
                            if r["router"] == rt and r["arm"] == arm}
    print("\n" + "=" * 78)
    print("[closure probe] budget=%d k=%d cap=%d  (%d assemblies over %d items)"
          % (args.budget, args.k, args.cap, len(rows), len({r["item_id"] for r in rows})))
    print("\n  %-12s %-8s %10s %8s %8s %10s %9s %10s"
          % ("router", "arm", "required", "entries", "incompl", "saturated", "goldctx", "evrec"))
    for rt in args.routers:
        for arm in ("off", "native", "capped"):
            c = list(cell(rt, arm).values())
            if not c:
                continue
            f = lambda k: sum(float(r[k]) for r in c) / len(c)
            print("  %-12s %-8s %10.1f %8.2f %8.2f %6d/%-4d %9.4f %10.4f"
                  % (rt, arm, f("required"), f("entries"), f("incomplete"),
                     sum(1 for r in c if r["required"] > SATURATED), len(c),
                     f("gold_in_context"), f("evidence_recall")))

    print("\n  paired contrasts -- does capping deliver the answer more often?")
    for rt in args.routers:
        for A, B in (("native", "off"), ("capped", "off"), ("capped", "native")):
            a, b = cell(rt, A), cell(rt, B)
            if not a or not b:
                continue
            line = []
            for field in ("gold_in_context", "evidence_recall"):
                d = defaultdict(list)
                for i in a:
                    if i in b:
                        d[a[i]["session"]].append(float(a[i][field]) - float(b[i][field]))
                st = contrast(d, with_percentile=False)
                line.append("%s %+.4f (p=%.4f%s)"
                            % (field[:7], st["delta"], st["wcr_p"],
                               "" if st["decidable"] else ", UNDECIDABLE"))
            print("  %-12s %-8s - %-8s  %s" % (rt, A, B, "   ".join(line)))
    print("\nCLOSURE_PROBE_DONE")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--items", default=os.path.expanduser(
        "data/tracepack_items6.jsonl"))
    ap.add_argument("--out", default=None, help="write every assembly row here (full sweep)")
    ap.add_argument("--sample", type=int, default=6, help="items per session when --out is unset")
    ap.add_argument("--budget", type=int, default=2048)
    ap.add_argument("--k", type=int, default=8)
    ap.add_argument("--cap", type=int, default=CAP)
    ap.add_argument("--routers", nargs="+", default=["lexical", "hybrid_pin"])
    return run(ap.parse_args())


if __name__ == "__main__":
    raise SystemExit(main())
