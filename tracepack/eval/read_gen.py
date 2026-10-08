"""tracepack.eval.read_gen -- GPU-side OPEN-ENDED read over emitted packets.

Same packets as `read_packets.py`, same model, different readout: the model is asked the item's
own question and writes an answer, instead of picking between two options.  Correctness is NOT
decided here -- `correct` is left null and filled in by `judge_answers.py`.  Writing a guess in
this file would let a string-matching heuristic quietly become the judge.

    TRACEPACK_READER_MODEL=/path/to/Qwen3-8B \\
      python3 tracepack/eval/read_gen.py --packets packets.jsonl --out gen.jsonl \\
      [--shard 0 --nshard 4]
"""
from __future__ import annotations

import argparse
import json
import os
import sys
import time

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

from tracepack.eval.reader import GenerativeQwenReader

CARRY = ("item_id", "arm", "budget", "slice", "session", "router", "closure", "repr", "tokens",
         "n_entries", "incomplete", "n_missing", "seed_hit", "evidence_recall",
         "closure_required", "gold_in_context", "query_source", "digest")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--packets", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--items", default="", help="items file, to recover each item's question")
    ap.add_argument("--shard", type=int, default=0)
    ap.add_argument("--nshard", type=int, default=1)
    ap.add_argument("--max-new", type=int, default=48)
    args = ap.parse_args()

    queries = {}
    if args.items:
        for line in open(args.items, encoding="utf-8"):
            it = json.loads(line)
            queries[it["item_id"]] = it.get("query") or ""

    done = set()
    if os.path.exists(args.out):
        for line in open(args.out, encoding="utf-8"):
            try:
                r = json.loads(line)
                done.add((r["item_id"], r["arm"], r["budget"]))
            except Exception:
                pass
    rows = []
    for i, line in enumerate(open(args.packets, encoding="utf-8")):
        if i % args.nshard != args.shard:
            continue
        r = json.loads(line)
        if (r["item_id"], r["arm"], r["budget"]) not in done:
            rows.append(r)
    print("[read_gen] todo=%d (done %d) shard %d/%d" % (
        len(rows), len(done), args.shard, args.nshard), flush=True)

    reader = GenerativeQwenReader(max_new=args.max_new)
    fo = open(args.out, "a", encoding="utf-8")
    t0 = time.time()
    n_unknown = 0
    for n, r in enumerate(rows, 1):
        item = dict(item_id=r["item_id"], gold=r["gold"], distractors=r["distractors"],
                    query_mode=r["query_mode"],
                    query=queries.get(r["item_id"]) or r.get("query") or "")
        res = reader(r["context"], item)
        out = {k: r.get(k) for k in CARRY}
        out.update(gold=r["gold"], answer=res.raw, n_tokens=res.n_tokens,
                   correct=None,                      # the judge fills this in
                   stale_value=r.get("stale_value"), model=reader.model_path)
        n_unknown += int(res.raw.strip().upper().startswith("UNKNOWN"))
        fo.write(json.dumps(out, ensure_ascii=False) + "\n")
        fo.flush()
        if n % 100 == 0:
            print("[read_gen] %d/%d  %.1f/min  UNKNOWN=%.0f%%  maxtok=%d" % (
                n, len(rows), 60 * n / max(1e-9, time.time() - t0),
                100 * n_unknown / n, reader.max_seen), flush=True)
    fo.close()
    print("[read_gen] done %d in %.1f min, UNKNOWN %d (%.1f%%), longest packet %d tokens" % (
        len(rows), (time.time() - t0) / 60, n_unknown,
        100 * n_unknown / max(1, len(rows)), reader.max_seen), flush=True)
    print("TRACEPACK_GEN_DONE", flush=True)


if __name__ == "__main__":
    main()
