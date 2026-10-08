"""tracepack.eval.read_fake -- CPU dry run over emitted packets with the deterministic reader.

The fake reader is correct exactly when the gold text is literally in the served context, so it
models a PERFECT reader over whatever the pipeline served.  Two uses:

  * instrument check -- run analyze.py end to end before the GPU results exist, so a schema or
    aggregation bug is found on the cheap pass rather than after an hour of A100 time;
  * an upper bound on what routing+closure can deliver, independent of the model.

Its numbers are NOT the experiment: never pool them with real reads (analyze.py refuses to mix
readers, which is what makes that safe).

    python3 tracepack/eval/read_fake.py --packets packets.jsonl --out eval_fake.jsonl
"""
from __future__ import annotations

import argparse
import json
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

from tracepack.eval.reader import DeterministicFakeReader

CARRY = ("item_id", "arm", "budget", "slice", "session", "router", "closure", "repr", "tokens",
         "n_entries", "incomplete", "n_missing", "seed_hit", "evidence_recall",
         "closure_required", "gold_in_context", "query_source", "digest")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--packets", default=os.path.expanduser(
        "data/tracepack_packets5.jsonl"))
    ap.add_argument("--out", default=os.path.expanduser(
        "data/tracepack_eval_fake.jsonl"))
    args = ap.parse_args()

    reader = DeterministicFakeReader()
    n = 0
    with open(os.path.expanduser(args.out), "w", encoding="utf-8") as fo:
        for line in open(os.path.expanduser(args.packets), encoding="utf-8"):
            try:
                r = json.loads(line)
            except ValueError:
                continue
            item = dict(item_id=r["item_id"], gold=r["gold"], distractors=r["distractors"],
                        query_mode=r["query_mode"])
            res = reader(r["context"], item)
            out = {k: r.get(k) for k in CARRY}
            out.update(correct=int(res.correct), score=res.score, raw=res.raw,
                       stale_hit=int(bool(r.get("stale_value")) and not res.correct),
                       model="fake")
            fo.write(json.dumps(out, ensure_ascii=False) + "\n")
            n += 1
    print("[read_fake] %d rows -> %s" % (n, args.out))
    print("FAKE_READ_DONE")


if __name__ == "__main__":
    main()
