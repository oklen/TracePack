"""tracepack.eval.analyze -- tables + pre-registered gate settlement for a run_eval output.

Reports, in this order (the order matters: instrument checks BEFORE headline numbers):
  0. provenance + coverage (one model? which cells are complete? partial cells are excluded)
  1. sanity floors: oracle|oracle must deliver the evidence; last_n|off must not
  2. Router x Closure accuracy at the primary budget (+ evidence recall, tokens, incomplete rate)
  3. per-slice breakdown of the closure effect (the dependency-heavy slices are the point)
  4. budget sweep and representation sweep
  5. paired tests: McNemar exact + WILD CLUSTER bootstrap-t over sessions (the percentile
     cluster bootstrap is printed beside it but is NOT the ruler -- measured size 0.100 vs
     nominal 0.05 on this project's own 10-session structure; see tracepack/eval/stats.py)
  6. the four Go/No-Go gates from DESIGN_FROZEN, each settled with its own number

    python3 tracepack/eval/analyze.py [eval.jsonl]
"""
from __future__ import annotations

import json
import math
import os
import random
import sys
from collections import Counter, defaultdict

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

from tracepack.eval.stats import contrast as wcr_contrast

DEFAULT = os.path.expanduser("data/tracepack_eval.jsonl")
PRIMARY = int(os.environ.get("TRACEPACK_PRIMARY", 2048))   # override to analyse another budget
DEP_HEAVY = ("decision_source", "tool_chain", "correction_stale")


def mcnemar(pairs):
    """exact two-sided binomial on discordant pairs; pairs = [(a_correct, b_correct), ...]"""
    b = sum(1 for x, y in pairs if x and not y)
    c = sum(1 for x, y in pairs if y and not x)
    n = b + c
    if n == 0:
        return b, c, 1.0
    k = min(b, c)
    tail = sum(math.comb(n, i) for i in range(k + 1)) / (2 ** n)
    return b, c, min(1.0, 2 * tail)


def cluster_boot(diffs_by_session, B=4000, seed=1234):
    """cluster bootstrap over sessions on the paired difference; returns (mean, lo, hi)."""
    keys = sorted(diffs_by_session)
    if not keys:
        return float("nan"), float("nan"), float("nan")
    rng = random.Random(seed)
    obs = sum(sum(diffs_by_session[k]) for k in keys) / sum(len(diffs_by_session[k]) for k in keys)
    means = []
    for _ in range(B):
        pick = [keys[rng.randrange(len(keys))] for _ in keys]
        vals = [v for k in pick for v in diffs_by_session[k]]
        if vals:
            means.append(sum(vals) / len(vals))
    means.sort()
    return obs, means[int(.025 * len(means))], means[int(.975 * len(means))]


def main():
    path = sys.argv[1] if len(sys.argv) > 1 else DEFAULT
    rows = [json.loads(l) for l in open(path, encoding="utf-8")]
    if not rows:
        print("no rows"); return 1

    models = {r["model"] for r in rows}
    print("=" * 78)
    print("[0] provenance: %d reads | model(s)=%s | items=%d | sessions=%d" % (
        len(rows), models, len({r["item_id"] for r in rows}), len({r["session"] for r in rows})))
    assert len(models) == 1, "refusing to pool reads from different readers: %s" % models

    n_items = len({r["item_id"] for r in rows})
    by_cell = defaultdict(list)
    for r in rows:
        by_cell[(r["router"], r["closure"], r["repr"], r["budget"])].append(r)
    complete = {k: v for k, v in by_cell.items() if len(v) == n_items}
    partial = {k: len(v) for k, v in by_cell.items() if len(v) != n_items}
    print("[0] cells: %d complete, %d partial (excluded: %s)" % (
        len(complete), len(partial), dict(list(partial.items())[:4]) or "none"))

    def acc(cell):
        v = complete.get(cell)
        return (sum(x["correct"] for x in v) / len(v)) if v else float("nan")

    def mean(cell, field):
        v = complete.get(cell)
        return (sum(x[field] for x in v) / len(v)) if v else float("nan")

    # ---------------- 1. instrument floors -------------------------------------------------
    print("\n" + "=" * 78)
    print("[1] instrument checks (these must hold or nothing below is readable)")
    oo = ("oracle", "oracle", "source_only", PRIMARY)
    lo = ("last_n", "off", "source_only", PRIMARY)
    print("    oracle|oracle  acc=%.3f  evidence_recall=%.3f  gold_in_ctx=%.3f  (ceiling)" % (
        acc(oo), mean(oo, "evidence_recall"), mean(oo, "gold_in_context")))
    print("    last_n|off     acc=%.3f  evidence_recall=%.3f  gold_in_ctx=%.3f  (floor)" % (
        acc(lo), mean(lo, "evidence_recall"), mean(lo, "gold_in_context")))
    if complete.get(oo) and mean(oo, "gold_in_context") < 0.9:
        print("    !! the oracle path does not even deliver the gold text -- fix before reading on")

    # ---- closure saturation (red team round 2, finding (8)) -----------------------------------
    # A compaction summary MATERIALIZES every event before it, and an unverified carrier is
    # source-first, so ONE such seed makes the whole session `required`.  Measured live: 874 of
    # 878 events required; deleting MATERIALIZES brings the same closure back to 10.  When that
    # happens the closure is no longer "the seeds' dependencies", it is "the session", and the
    # assembler fills the budget with the chronologically oldest of ~900 candidates.  Any closure
    # contrast must then be read PER STRATUM, not pooled.
    SAT = 100
    sat_rate = {}
    for k, v in complete.items():
        if k[1] in ("native", "full_ancestor") and k[2] == "source_only" and k[3] == PRIMARY:
            sat = [r for r in v if r.get("closure_required", 0) > SAT]
            sat_rate[(k[0], k[1])] = (len(sat), len(v),
                                      sorted(r["closure_required"] for r in sat))
    worst = max((v[0] / max(1, v[1]) for v in sat_rate.values()), default=0.0)
    if not sat_rate:
        # An empty line here reads as "no saturation", which is the wrong direction to fail.
        print("    Closure saturation: NOT MEASURED -- no complete closure cell at budget=%d "
              "(set TRACEPACK_PRIMARY, or this file is not a router x closure grid)" % PRIMARY)
    else:
        print("    Closure saturation: %s"
              % (", ".join("%s|%s %d/%d" % (a, b, n, tot)
                           for (a, b), (n, tot, _) in sorted(sat_rate.items()) if n)
                 or "0 cells over the threshold"))
    if worst > 0.05:
        med = sorted(x for v in sat_rate.values() for x in v[2])
        print("    !! %.0f%% of items have a closure requiring >%d events (median required among "
              "them: %d).  Pooled closure contrasts are NOT readable -- report per stratum."
              % (100 * worst, SAT, med[len(med) // 2] if med else 0))

    # ---------------- 2. router x closure --------------------------------------------------
    print("\n" + "=" * 78)
    print("[2] Router x Closure @ budget=%d, source_only   (acc | evrec | tok | incomplete%%)"
          % PRIMARY)
    routers = [r for r in ("last_n", "lexical", "dense", "hybrid", "hybrid_pin", "oracle")
               if any(k[0] == r for k in complete)]
    closures = [c for c in ("off", "native", "full_ancestor", "oracle")
                if any(k[1] == c for k in complete)]
    print("    %-11s %s" % ("router", "  ".join("%-26s" % c for c in closures)))
    for rt in routers:
        cells = []
        for cl in closures:
            k = (rt, cl, "source_only", PRIMARY)
            cells.append("%.3f|%.2f|%4.0f|%3.0f%%" % (
                acc(k), mean(k, "evidence_recall"), mean(k, "tokens"),
                100 * mean(k, "incomplete")) if k in complete else "%-26s" % "-")
        print("    %-11s %s" % (rt, "  ".join("%-26s" % c for c in cells)))

    # ---------------- 3. per-slice closure effect ------------------------------------------
    print("\n" + "=" * 78)
    print("[3] closure effect per slice (hybrid_pin: native - off) @ primary")
    a = complete.get(("hybrid_pin", "native", "source_only", PRIMARY), [])
    b = complete.get(("hybrid_pin", "off", "source_only", PRIMARY), [])
    if a and b:
        bi = {r["item_id"]: r for r in b}
        by_slice = defaultdict(list)
        for r in a:
            if r["item_id"] in bi:
                by_slice[r["slice"]].append((r, bi[r["item_id"]]))
        print("    %-18s %6s %8s %8s %8s   %s" % (
            "slice", "n", "off", "native", "delta", "evrec off->native"))
        for slc in sorted(by_slice):
            pr = by_slice[slc]
            ao = sum(x[1]["correct"] for x in pr) / len(pr)
            an = sum(x[0]["correct"] for x in pr) / len(pr)
            eo = sum(x[1]["evidence_recall"] for x in pr) / len(pr)
            en = sum(x[0]["evidence_recall"] for x in pr) / len(pr)
            print("    %-18s %6d %8.3f %8.3f %+8.3f   %.2f -> %.2f" % (
                slc, len(pr), ao, an, an - ao, eo, en))

    # ---------------- 4. sweeps ------------------------------------------------------------
    print("\n" + "=" * 78)
    print("[4] budget sweep (hybrid_pin|native, source_only)")
    for bud in sorted({k[3] for k in complete}):
        k = ("hybrid_pin", "native", "source_only", bud)
        if k in complete:
            print("    budget=%5d  acc=%.3f  evrec=%.2f  tok=%4.0f  incomplete=%3.0f%%" % (
                bud, acc(k), mean(k, "evidence_recall"), mean(k, "tokens"),
                100 * mean(k, "incomplete")))
    print("[4] representation sweep (hybrid_pin|native @ primary)")
    for rp in ("source_only", "carrier_only", "source_plus_carrier"):
        k = ("hybrid_pin", "native", rp, PRIMARY)
        if k in complete:
            print("    %-20s acc=%.3f  evrec=%.2f  tok=%4.0f" % (
                rp, acc(k), mean(k, "evidence_recall"), mean(k, "tokens")))

    # ---------------- 5. paired tests ------------------------------------------------------
    print("\n" + "=" * 78)
    print("[5] paired contrasts (McNemar exact + cluster bootstrap over sessions)")
    contrasts = [
        (("hybrid_pin", "native", "source_only", PRIMARY),
         ("hybrid_pin", "off", "source_only", PRIMARY), "closure gain (native - off)"),
        (("oracle", "native", "source_only", PRIMARY),
         ("hybrid_pin", "native", "source_only", PRIMARY), "router gap (oracle - hybrid_pin)"),
        (("oracle", "native", "source_only", PRIMARY),
         ("lexical", "full_ancestor", "source_only", PRIMARY),
         "router gap (oracle - BEST automatic)"),
        (("hybrid_pin", "oracle", "source_only", PRIMARY),
         ("hybrid_pin", "native", "source_only", PRIMARY), "edge coverage (gold - native edges)"),
    ]
    for ka, kb, name in contrasts:
        A, B = complete.get(ka), complete.get(kb)
        if not (A and B):
            print("    %-38s (cells incomplete)" % name); continue
        bi = {r["item_id"]: r for r in B}
        pairs, by_sess = [], defaultdict(list)
        for r in A:
            o = bi.get(r["item_id"])
            if o is None:
                continue
            pairs.append((r["correct"], o["correct"]))
            by_sess[r["session"]].append(r["correct"] - o["correct"])
        bb, cc, p = mcnemar(pairs)
        st = wcr_contrast(by_sess)
        print("    %-38s n=%3d  delta=%+.3f  WCR p=%.4f CI[%+.3f,%+.3f]  "
              "(pct CI[%+.3f,%+.3f])  McNemar b/c=%d/%d p=%.4f" % (
                  name, len(pairs), st["delta"], st["wcr_p"], st["wcr_lo"], st["wcr_hi"],
                  st["pct_lo"], st["pct_hi"], bb, cc, p))
        # A cluster whose paired differences net to exactly zero cannot vote under the imposed
        # null, so the attainable p-floor is 2^(1-k).  When that exceeds alpha the contrast is
        # undecidable by this ruler at ANY effect size -- say so instead of "not significant".
        print("        %-34s informative sessions %d/%d, attainable p-floor %.4f -> %s"
              % ("", st["k_informative"], st["G"], st["p_floor"],
                 "decidable" if st["decidable"] else
                 "UNDECIDABLE at alpha=.05 (report the point estimate and McNemar, not 'n.s.')"))
        dep = [(r["correct"], bi[r["item_id"]]["correct"]) for r in A
               if r["item_id"] in bi and r["slice"] in DEP_HEAVY]
        if dep and name.startswith("closure"):
            d = sum(x - y for x, y in dep) / len(dep)
            print("        dependency-heavy slices only: n=%d delta=%+.3f" % (len(dep), d))

    # ---------------- 6. gates -------------------------------------------------------------
    print("\n" + "=" * 78)
    print("[6] pre-registered gates (DESIGN_FROZEN §5)")
    A = complete.get(("hybrid_pin", "native", "source_only", PRIMARY), [])
    B = complete.get(("hybrid_pin", "off", "source_only", PRIMARY), [])
    if A and B:
        bi = {r["item_id"]: r for r in B}
        dep = [(r, bi[r["item_id"]]) for r in A if r["item_id"] in bi and r["slice"] in DEP_HEAVY]
        d_acc = sum(x[0]["correct"] - x[1]["correct"] for x in dep) / max(1, len(dep))
        d_ev = sum(x[0]["evidence_recall"] - x[1]["evidence_recall"] for x in dep) / max(1, len(dep))
        ok = (d_ev >= 0.10) or (d_acc >= 0.05)
        print("    Closure gate : dep-heavy n=%d  d_evidence_recall=%+.3f  d_accuracy=%+.3f -> %s"
              % (len(dep), d_ev, d_acc, "PASS" if ok else "FAIL"))
    cc_ = complete.get(("hybrid_pin", "native", "carrier_only", PRIMARY), [])
    ss = complete.get(("hybrid_pin", "native", "source_only", PRIMARY), [])
    if cc_ and ss:
        si = {r["item_id"]: r for r in ss}
        by_sess = defaultdict(list)
        for r in cc_:
            if r["item_id"] in si:
                by_sess[r["session"]].append(r["correct"] - si[r["item_id"]]["correct"])
        st = wcr_contrast(by_sess)
        tok = mean(("hybrid_pin", "native", "carrier_only", PRIMARY), "tokens")
        tok_s = mean(("hybrid_pin", "native", "source_only", PRIMARY), "tokens")
        ok = st["wcr_lo"] >= -0.05 and tok < tok_s
        print("    Carrier gate : delta=%+.3f WCR CI[%+.3f,%+.3f]  tokens %.0f vs %.0f -> %s" % (
            st["delta"], st["wcr_lo"], st["wcr_hi"], tok, tok_s, "PASS" if ok else "FAIL"))
    print("    Adapter gate : PASS (native trace -> IR -> packet -> context, see examples/)")
    print("    Profiler gate: see tracepack/profiler/run_profiler.py output (labels + confusion)")
    print("    NOTE: with G=10 sessions the WCR power at the gate's own +5pp threshold is 0.24 "
          "(tracepack/eval/stats.py selfcheck) -- a point estimate at the threshold is NOT "
          "separable from zero, and must not be reported as if it were.")
    print("\nTRACEPACK_ANALYZE_DONE")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
