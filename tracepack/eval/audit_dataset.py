"""tracepack.eval.audit_dataset -- leak/validity audit of the dependency slice.

Self-built benchmarks fail in ways their builder cannot see, so this re-derives the properties from the RAW traces and FAILS LOUDLY rather than warning:

 A1  gold is actually present in the gold packet (else the item is unanswerable by design)
 A2  no distractor appears in the gold packet (else answerable by elimination)
 A3  gold is unique among the session's tool_results (recomputed, not trusted)
 A4  decision_source: the seed event does NOT contain gold (the whole point of the slice)
 A5  correction_stale: the stale value IS in the superseded event, is a listed choice, and is NOT
     in the gold packet
 A6  tool_chain: required spans >=2 tool_results and contains an atomic call/result group
 A7  no session contributes >40% of a slice
 A8  gold packet fits the primary budget
 A9  distractor SHAPE agrees with gold (same length and per-position character class for
     number/hash/version/fileline; same directory and extension for path) -- a shape difference
     is decidable with zero evidence served.  The correction_stale slice's stale value is exempt:
     it is a real earlier value from the trace and shape-matching it would defeat the slice; A11
     scores it instead.
A10  duplicate golds across items stay under 10%
A12  the QUESTION does not carry the answer: no query may contain the gold, its discriminating
     part, or any distractor's (re-checked here, not trusted from generation time)
A11  no no-evidence arm beats 0.55: an attacker holding the items file and nothing else must not
     separate gold from distractor by surface shape or by counting reuse (tracepack.eval.
     attack_surface).  This is the check the red team's 0.763 arm would have failed.

    python3 tracepack/eval/audit_dataset.py [items.jsonl]
"""
from __future__ import annotations

import collections
import json
import os
import re
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

from tracepack.adapters.claude_code import ClaudeCodeAdapter
from tracepack.adapters.registry import adapter_for
from tracepack.eval.attack_surface import run as attack_run
from tracepack.eval.gen_queries import leaks
from tracepack.eval.freeze_corpus import assert_frozen
from tracepack.eval.datasets import PRIMARY_BUDGET, shape, wb

DEFAULT = os.path.expanduser("data/tracepack_items.jsonl")


def main():
    path = sys.argv[1] if len(sys.argv) > 1 else DEFAULT
    items = [json.loads(l) for l in open(path, encoding="utf-8")]
    print("[audit] %d items from %s" % (len(items), os.path.basename(path)))
    n_frozen = assert_frozen(items)
    print("[audit] corpus: %d traces hash-verified (0 = items not frozen yet)" % n_frozen)

    fails = collections.Counter()
    notes = []
    graphs = {}
    for it in items:
        tr = it["transcript"]
        if tr not in graphs:
            graphs[tr] = adapter_for(tr).normalize(tr)
        g = graphs[tr]
        served = "\n".join(g.event(i).text for i in it["required_sources"])
        gold, dists = it["gold"], it["distractors"]

        if not wb(gold).search(served):
            fails["A1_gold_absent"] += 1
            notes.append(("A1", it["item_id"]))
        for d in dists:
            if wb(d).search(served):
                fails["A2_distractor_in_packet"] += 1
                notes.append(("A2", it["item_id"], d))
                break
        n_hits = sum(1 for e in g.events if e.kind == "tool_result" and wb(gold).search(e.text))
        if n_hits > 1 and it["slice"] != "correction_stale":
            fails["A3_gold_not_unique"] += 1
        if it["slice"] == "decision_source":
            if wb(gold).search(g.event(it["gold_seed"]).text):
                fails["A4_seed_contains_gold"] += 1
                notes.append(("A4", it["item_id"]))
        if it["slice"] == "correction_stale":
            stale = it.get("stale_value")
            old_txt = g.event(it["stale_event"]).text if it.get("stale_event") else ""
            if not stale or not wb(stale).search(old_txt):
                fails["A5_stale_not_in_old"] += 1
            if stale and wb(stale).search(served):
                fails["A5_stale_in_gold_packet"] += 1
            if stale and stale not in dists:
                fails["A5_stale_not_a_choice"] += 1
        if it["slice"] == "tool_chain":
            req = it["required_sources"]
            kinds = [g.event(i).kind for i in req]
            if len(req) < 3 or kinds.count("tool_result") < 2:
                fails["A6_chain_too_short"] += 1
            if not {g.event(i).atomic_group for i in req if g.event(i).atomic_group}:
                fails["A6_no_atomic_group"] += 1
        if it["gold_tokens"] > PRIMARY_BUDGET:
            fails["A8_gold_over_budget"] += 1
        q = it.get("query")
        if q and leaks(q, gold, dists, it["type"]):
            fails["A12_query_leaks_answer"] += 1
            notes.append(("A12", it["item_id"], q[:80]))
        for d in dists:
            if d == it.get("stale_value"):
                # the stale value is a REAL earlier value from the trace; shape-matching it would
                # defeat the slice.  It is still scored by A11, where it belongs.
                continue
            if it["type"] == "path":
                same_dir = d.rsplit("/", 1)[:-1] == gold.rsplit("/", 1)[:-1]
                same_ext = d.rsplit(".", 1)[-1] == gold.rsplit(".", 1)[-1]
                if not (same_dir and same_ext):
                    fails["A9_path_shape_mismatch"] += 1
                    notes.append(("A9", it["item_id"], d))
                    break
            elif shape(d) != shape(gold):
                fails["A9_shape_mismatch"] += 1
                notes.append(("A9", it["item_id"], d, shape(gold), shape(d)))
                break

    by_slice = collections.defaultdict(collections.Counter)
    for it in items:
        by_slice[it["slice"]][it["session"]] += 1
    for slc, c in by_slice.items():
        top, n = c.most_common(1)[0]
        if n / sum(c.values()) > 0.40:
            fails["A7_session_dominates(%s:%s=%d/%d)" % (slc, top, n, sum(c.values()))] += 1

    golds = collections.Counter(it["gold"] for it in items)
    dup = sum(v - 1 for v in golds.values() if v > 1)
    if dup / max(1, len(items)) > 0.10:
        fails["A10_duplicate_golds=%d" % dup] += 1

    n = len(items)
    print("\n[audit] per slice: %s" % dict(collections.Counter(i["slice"] for i in items)))
    print("[audit] per type : %s" % dict(collections.Counter(i["type"] for i in items)))
    print("[audit] gold in default summary: %.1f%% | in assistant text: %.1f%%" % (
        100 * sum(i["gold_in_summary"] for i in items) / n,
        100 * sum(i["gold_in_assistant"] for i in items) / n))
    print("[audit] duplicate golds: %d (%.1f%%)" % (dup, 100 * dup / n))
    qs = collections.Counter(i.get("query_source", "none") for i in items)
    print("[audit] query source: %s" % dict(qs))
    gt = sorted(i["gold_tokens"] for i in items)
    print("[audit] gold tokens p50/p90/max: %d / %d / %d" % (gt[n // 2], gt[int(n * .9)], gt[-1]))
    print("")
    ok_attack, _ = attack_run(items)
    if not ok_attack:
        fails["A11_no_evidence_arm_beats_ceiling"] += 1

    if fails:
        print("\n[audit] FAILURES: %s" % dict(fails))
        print("[audit] examples: %s" % notes[:6])
        print("AUDIT_FAILED")
        return 1
    print("\nAUDIT_OK -- no leak or validity check failed")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
