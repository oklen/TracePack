"""tracepack.eval.audit_readout -- audit the LLM judge against an independent shadow rule.

    python3 tracepack/eval/audit_readout.py <judged.jsonl>

The judge passed a 280-case self-audit on answers *we wrote*.  That measures the judge on cases we
thought of.  This measures it on every answer it actually judged, against a rule that shares no
code with it: literal containment under ``core.textmatch.contains_value``.

Two cells matter and they are not symmetric:

* **judge=0 while the gold string IS in the answer** -- under-acceptance.  There is no defensible
  reason for this cell to be non-empty, so it is a hard gate.
* **judge=1 while the gold string is NOT in the answer** -- over-acceptance.  Some of this is
  correct (a paraphrase, a longer path containing the gold's tail).  It is not gated; it is
  *measured*, and then every headline is recomputed under ``strict = judge AND literal`` so the
  reader can see whether the leniency carries any conclusion.  Red team round 2, §7.1.
"""
from __future__ import annotations

import json
import os
import sys
from collections import Counter, defaultdict

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

from tracepack.core.textmatch import contains_value
from tracepack.eval.stats import contrast

#: contrasts recomputed under both judges -- (armA, armB, label)
HEADLINES = (
    (("oracle", "native"), ("hybrid_pin", "native"), "routing gap (vs default)"),
    (("oracle", "native"), ("lexical", "full_ancestor"), "routing gap (vs best auto)"),
    (("hybrid_pin", "native"), ("hybrid_pin", "off"), "auto closure"),
    (("hybrid_pin", "oracle"), ("hybrid_pin", "native"), "inject-answer - closure"),
)


def shadow(rows):
    """Annotate every row with the independent literal verdict."""
    for r in rows:
        a = r.get("answer") or ""
        r["_lit"] = int(contains_value(a, r["gold"]))
        r["_strict"] = int(int(r["correct"]) == 1 and r["_lit"])
        r["_unknown"] = int(not a.strip() or a.strip().upper().startswith("UNKNOWN"))
    return rows


def main():
    path = sys.argv[1] if len(sys.argv) > 1 else os.path.expanduser(
        "data/tracepack_evalgen_base.jsonl")
    rows = [json.loads(l) for l in open(path, encoding="utf-8") if l.strip()]
    rows = shadow([r for r in rows if r.get("correct") is not None and "answer" in r])
    if not rows:
        print("no judged free-form rows in %s" % path)
        return 2
    # the primary grid is the budget carrying the most cells, not the smallest one:
    # 1024/4096 only carry a 3-cell ladder and would silently drop most contrasts.
    from collections import Counter as _C
    budget = int(os.environ.get("TRACEPACK_PRIMARY") or
                 _C(int(r["budget"]) for r in rows).most_common(1)[0][0])
    print("=" * 78)
    print("[judge audit] %d judged answers from %s" % (len(rows), os.path.basename(path)))

    cm = Counter((int(r["correct"]), r["_lit"]) for r in rows)
    print("\n  judge x literal-containment")
    print("            literal=0   literal=1")
    for j in (0, 1):
        print("   judge=%d  %9d   %9d" % (j, cm[(j, 0)], cm[(j, 1)]))
    under, over = cm[(0, 1)], cm[(1, 0)]
    print("   agreement = %.4f" % ((cm[(0, 0)] + cm[(1, 1)]) / len(rows)))

    print("\n  UNKNOWN/empty answers: %d; of those judged correct: %d (must be 0)"
          % (sum(r["_unknown"] for r in rows),
             sum(int(r["correct"]) for r in rows if r["_unknown"])))
    stale = [r for r in rows if r.get("stale_value") not in (None, "None")]
    bad_stale = [r for r in stale if int(r["correct"]) == 1
                 and contains_value(r.get("answer") or "", str(r["stale_value"]))
                 and not r["_lit"]]
    print("  answers accepted while committing to the STALE value: %d / %d stale-slice rows "
          "(must be 0)" % (len(bad_stale), len(stale)))

    print("\n  over-acceptance by stratum (this is what decides whether it matters):")
    byctx = Counter(int(r["gold_in_context"]) for r in rows if int(r["correct"]) == 1
                    and not r["_lit"])
    print("     gold WAS in the served context : %d" % byctx[1])
    print("     gold was NOT in the context    : %d   <- only these can inflate the floor"
          % byctx[0])

    print("\n  every headline under both judges (budget=%d, source_only):" % budget)
    cell = lambda rt, cl: {r["item_id"]: r for r in rows if int(r["budget"]) == budget
                           and r["router"] == rt and r["closure"] == cl
                           and r["repr"] == "source_only"}
    print("     %-28s %10s %10s %9s" % ("contrast", "judge", "strict", "moved"))
    for A, B, name in HEADLINES:
        ca, cb = cell(*A), cell(*B)
        if not ca or not cb:
            continue
        out = {}
        for field in ("correct", "_strict"):
            d = defaultdict(list)
            for i in ca:
                if i in cb:
                    d[ca[i]["session"]].append(float(ca[i][field]) - float(cb[i][field]))
            out[field] = contrast(d, with_percentile=False)
        print("     %-28s %+10.4f %+10.4f %+9.4f" %
              (name, out["correct"]["delta"], out["_strict"]["delta"],
               out["_strict"]["delta"] - out["correct"]["delta"]))

    yes = [r for r in rows if int(r["gold_in_context"]) == 1]
    no = [r for r in rows if int(r["gold_in_context"]) == 0]
    for field, lab in (("correct", "judge "), ("_strict", "strict")):
        ay = sum(int(r[field]) for r in yes) / len(yes)
        an = sum(int(r[field]) for r in no) / len(no)
        print("     %s  gold present %.4f | gold absent %.4f | slope %.4f"
              % (lab, ay, an, ay - an))

    print()
    if under:
        print("JUDGE AUDIT FAILED: %d answers were rejected while literally containing the gold "
              "value -- there is no defensible reading of that cell" % under)
        return 1
    if bad_stale:
        print("JUDGE AUDIT FAILED: %d stale-value answers accepted" % len(bad_stale))
        return 1
    print("JUDGE_AUDIT_OK -- 0 under-acceptances; %d over-acceptances measured (%d of them in the "
          "gold-absent stratum), every headline recomputed under the strict judge above"
          % (over, byctx[0]))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
