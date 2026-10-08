"""tracepack.eval.read_arms -- GPU-side forced-choice read over a generic context pack.

Consumes whatever `--emit` produced (profiler arms or eval packets): every line needs
``key``/``item_id``, ``context``, ``gold``, ``distractors``, ``query_mode``.  Writes one line per
key with the verdict.  Resumable by key.

``raw`` is written as the VALUE the reader picked, not the letter, so the profiler's
``stale_hit`` test (`the answer carries the superseded value and not the current one`) can
actually fire -- with "A"/"B" it never could, and a `stale_conflict` label would be unreachable
by construction rather than absent from the data.

    TRACEPACK_READER_MODEL=/path/to/Qwen3-8B python3 -m tracepack.eval.read_arms \\
      --packs arms.jsonl --out arms_read.jsonl
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
    ap.add_argument("--packs", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--shard", type=int, default=0)
    ap.add_argument("--nshard", type=int, default=1)
    args = ap.parse_args()

    done = set()
    if os.path.exists(args.out):
        for line in open(args.out, encoding="utf-8"):
            try:
                done.add(json.loads(line)["key"])
            except Exception:
                pass
    rows = []
    for i, line in enumerate(open(args.packs, encoding="utf-8")):
        if i % args.nshard != args.shard:
            continue
        r = json.loads(line)
        k = r.get("key") or r["item_id"]
        if k not in done:
            r["key"] = k
            rows.append(r)
    print("[read_arms] todo=%d (done %d) shard %d/%d" % (
        len(rows), len(done), args.shard, args.nshard), flush=True)

    reader = QwenReader()
    fo = open(args.out, "a", encoding="utf-8")
    t0 = time.time()
    for n, r in enumerate(rows, 1):
        item = dict(item_id=r["item_id"], gold=r["gold"], distractors=r["distractors"],
                    query_mode=r.get("query_mode", "lookup"))
        res = reader(r["context"], item)
        picked = r["gold"] if res.correct else (r["distractors"] or [""])[0]
        fo.write(json.dumps({"key": r["key"], "item_id": r["item_id"],
                             "correct": int(res.correct), "score": res.score,
                             "letter": res.raw, "raw": picked,
                             "model": reader.model_path}, ensure_ascii=False) + "\n")
        fo.flush()
        if n % 200 == 0:
            print("[read_arms] %d/%d  %.1f/min" % (
                n, len(rows), 60 * n / max(1e-9, time.time() - t0)), flush=True)
    fo.close()
    print("[read_arms] done %d in %.1f min" % (len(rows), (time.time() - t0) / 60), flush=True)
    print("TRACEPACK_READ_DONE", flush=True)


if __name__ == "__main__":
    main()
