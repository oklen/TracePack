"""tracepack.eval.read_packets -- GPU-side reader over emitted packets (worker script).

Assembly happens where the traces are (Mac); reading happens where the GPU is.  This consumes the
`--emit` output of run_eval.py, so the worker never needs a transcript and every arm was assembled
by the same code path.  Resumable by (item_id, arm, budget).

    TRACEPACK_READER_MODEL=/path/to/Qwen3-8B python3 tracepack/eval/read_packets.py \
        --packets packets.jsonl --out reads.jsonl [--shard 0 --nshard 1]
"""
from __future__ import annotations

import argparse
import json
import os
import sys
import time

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

from tracepack.eval.reader import QwenReader


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--packets", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--shard", type=int, default=0)
    ap.add_argument("--nshard", type=int, default=1)
    args = ap.parse_args()

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
    print("[read_packets] todo=%d (done %d) shard %d/%d" % (
        len(rows), len(done), args.shard, args.nshard), flush=True)

    reader = QwenReader()
    fo = open(args.out, "a", encoding="utf-8")
    t0 = time.time()
    for n, r in enumerate(rows, 1):
        item = dict(item_id=r["item_id"], gold=r["gold"], distractors=r["distractors"],
                    query_mode=r["query_mode"])
        res = reader(r["context"], item)
        out = {k: r[k] for k in ("item_id", "arm", "budget", "slice", "session", "router",
                                 "closure", "repr", "tokens", "n_entries", "incomplete",
                                 "n_missing", "seed_hit", "evidence_recall", "closure_required",
                                 "gold_in_context", "query_source", "digest")}
        out.update(correct=int(res.correct), score=res.score, raw=res.raw,
                   stale_hit=int(bool(r.get("stale_value")) and not res.correct),
                   model=reader.model_path)
        fo.write(json.dumps(out, ensure_ascii=False) + "\n")
        fo.flush()
        if n % 200 == 0:
            print("[read_packets] %d/%d  %.1f/min" % (
                n, len(rows), 60 * n / max(1e-9, time.time() - t0)), flush=True)
    fo.close()
    print("[read_packets] done %d in %.1f min" % (len(rows), (time.time() - t0) / 60), flush=True)
    print("TRACEPACK_READ_DONE", flush=True)


if __name__ == "__main__":
    main()
