"""tracepack.profiler.run_profiler -- drive the A0-A4 / B1-B4 ladder over the real dataset.

The runner takes an INJECTED reader, and the reader needs a GPU that is not where the traces are.
So this driver runs the same runner twice in a record/replay split, which keeps ``runner.py``
(104 green contract tests) untouched:

  --emit   Mac: run with a RecordingReader that answers nothing and records every context it was
           handed.  Arms are built by interventions.py, never by the reader, so recording cannot
           change what gets built.
  (worker) read those contexts with the real Qwen reader   -> answers.jsonl
  --merge  Mac: run again with a ReplayReader that returns the recorded answer.  The run is
           deterministic, so the replayed context must hash to one that was actually read --
           asserted per call, never assumed.  Attribution, summarize() and the confusion matrix
           come from that pass.

The replay key is ``(item_id, sha1(context), effective_gold)``.  It is NOT the arm name, because
the runner does not tell the reader which arm it is serving; including the effective gold is what
keeps ``B4_mutated_source`` -- whose gold is a nonce -- from colliding with the arm it mutates.

    python3 tracepack/profiler/run_profiler.py --emit arms.jsonl --items items.jsonl
    python3 tracepack/profiler/run_profiler.py --merge answers.jsonl --items items.jsonl \\
        --out data/tracepack_profiler.json
"""
from __future__ import annotations

import argparse
import hashlib
import json
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

from tracepack.adapters.claude_code import ClaudeCodeAdapter
from tracepack.eval.freeze_corpus import assert_frozen
from tracepack.profiler.attribution import ReadResult
from tracepack.profiler.runner import ProfilerConfig, ProfilerRunner, summarize


def rkey(item, context):
    h = hashlib.sha1((context or "").encode("utf-8")).hexdigest()[:16]
    return "%s|%s|%s" % (item.get("item_id"), h, item.get("gold"))


class RecordingReader:
    """Answers nothing; records every distinct context it is handed."""

    def __init__(self):
        self.rows = {}
        self.n = 0

    def __call__(self, context, item):
        self.n += 1
        k = rkey(item, context)
        if k not in self.rows:
            self.rows[k] = {
                "key": k, "item_id": item.get("item_id"), "gold": item.get("gold"),
                "distractors": list(item.get("distractors") or []),
                "query_mode": item.get("query_mode", "lookup"),
                "stale_value": item.get("stale_value"),
                "context": context,
            }
        return ReadResult(correct=False, raw="")


class ReplayReader:
    """Returns the recorded answer; a context that was never read is a hard error, not a miss."""

    def __init__(self, answers, strict=True):
        self.answers = answers
        self.strict = strict
        self.missing = 0

    def __call__(self, context, item):
        k = rkey(item, context)
        a = self.answers.get(k)
        if a is None:
            self.missing += 1
            if self.strict:
                raise SystemExit(
                    "replay has no answer for %s -- the emit and merge passes disagree, which "
                    "means the labels would be attached to a context nobody read" % k)
            return ReadResult(correct=False, raw="")
        return ReadResult(correct=bool(a.get("correct")), raw=a.get("raw") or "")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--items", default=os.path.expanduser(
        "data/tracepack_items5_q.jsonl"))
    ap.add_argument("--emit", default="")
    ap.add_argument("--merge", default="")
    ap.add_argument("--out", default=os.path.expanduser(
        "data/tracepack_profiler.json"))
    ap.add_argument("--budget", type=int, default=2048)
    ap.add_argument("--limit", type=int, default=0)
    args = ap.parse_args()

    items = [json.loads(l) for l in open(os.path.expanduser(args.items), encoding="utf-8")]
    if args.limit:
        items = items[:args.limit]
    # The corpus must be immutable across stages.  One of the ten traces is the session that
    # built the dataset and Claude Code appends to it live, so without this check dataset
    # construction, packet assembly and profiler replay each read a different graph -- silently,
    # because the adapter's ids are positional and appending does not renumber anything.
    assert_frozen(items)

    graphs = {}

    def graph_provider(item):
        tr = item["transcript"]
        if tr not in graphs:
            graphs[tr] = ClaudeCodeAdapter().normalize(tr)
        return graphs[tr]

    cfg = ProfilerConfig(budget=args.budget, router="hybrid_pin", router_k=8, strict=True)

    if args.emit:
        rd = RecordingReader()
        ProfilerRunner(graph_provider, rd, cfg).run(items)
        with open(os.path.expanduser(args.emit), "w", encoding="utf-8") as fo:
            for row in rd.rows.values():
                fo.write(json.dumps(row, ensure_ascii=False) + "\n")
        print("[profiler] %d arm reads over %d items -> %d distinct contexts -> %s"
              % (rd.n, len(items), len(rd.rows), args.emit))
        print("PROFILER_EMIT_DONE")
        return 0

    answers = {}
    if args.merge:
        for line in open(os.path.expanduser(args.merge), encoding="utf-8"):
            try:
                r = json.loads(line)
            except ValueError:
                continue
            if r.get("key"):
                answers[r["key"]] = r
        print("[profiler] loaded %d answers" % len(answers))
    runner = ProfilerRunner(graph_provider, ReplayReader(answers), cfg)
    records = runner.run(items)
    summ = summarize(records)
    summ["n_answers"] = len(answers)
    with open(os.path.expanduser(args.out), "w", encoding="utf-8") as fo:
        json.dump({"summary": summ, "records": records}, fo, ensure_ascii=False, indent=1)
    print(json.dumps(summ, ensure_ascii=False, indent=1)[:2500])
    print("[profiler] wrote %s" % args.out)
    print("PROFILER_DONE")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
