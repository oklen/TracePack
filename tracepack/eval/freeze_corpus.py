"""tracepack.eval.freeze_corpus -- copy the traces the dataset uses into an immutable corpus.

Why this exists (found 2026-09-01, while the profiler's replay guard refused to run): one of the
ten transcripts is the session that BUILT the dataset, and Claude Code appends to it continuously.
It was 310 MB and one minute old.  So `ClaudeCodeAdapter().normalize(...)` returned a different
graph every time it was called, and the three stages that each normalise independently -- dataset
construction, packet assembly, profiler replay -- were reading three different corpora.  Nothing
raised; the packets were simply built against a trace that no longer existed.

The adapter's event ids are positional (`cc:<index>:<part>`), and appending does not renumber what
came before, so the drift is invisible to every id-based check.  It shows up only as text that
changed length.

Fix: copy the traces once, point the items at the copies, and record each file's sha1 in every
item.  `assert_frozen()` re-checks the hash wherever a trace is read, so a stale or swapped corpus
is a loud failure instead of a quiet drift.

    python3 tracepack/eval/freeze_corpus.py --items items_q.jsonl --out items_frozen.jsonl \\
        --dir data/tp_traces
"""
from __future__ import annotations

import argparse
import hashlib
import json
import os
import shutil
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))


def sha1_file(path, chunk=1 << 22):
    h = hashlib.sha1()
    with open(path, "rb") as f:
        while True:
            b = f.read(chunk)
            if not b:
                break
            h.update(b)
    return h.hexdigest()


def assert_frozen(items):
    """Every item's trace must still hash to what it hashed to when the item was written."""
    seen = {}
    bad = []
    for it in items:
        want = it.get("transcript_sha1")
        if not want:
            continue
        tr = it["transcript"]
        if tr not in seen:
            seen[tr] = sha1_file(tr) if os.path.exists(tr) else "MISSING"
        if seen[tr] != want:
            bad.append((os.path.basename(tr)[:12], want[:10], seen[tr][:10]))
    if bad:
        uniq = sorted(set(bad))
        raise SystemExit(
            "corpus drift: %d trace(s) no longer match the hash recorded in the items file.\n"
            "  %s\nRe-freeze (tracepack/eval/freeze_corpus.py) and rebuild -- do NOT mix."
            % (len(uniq), "\n  ".join("%s recorded=%s now=%s" % b for b in uniq)))
    return len(seen)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--items", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--dir", default=os.path.expanduser("data/tp_traces"))
    ap.add_argument("--verify-only", action="store_true")
    args = ap.parse_args()

    items = [json.loads(l) for l in open(os.path.expanduser(args.items), encoding="utf-8")]
    if args.verify_only:
        n = assert_frozen(items)
        print("[freeze] %d traces verified against the recorded hashes" % n)
        print("FREEZE_VERIFY_OK")
        return 0

    d = os.path.expanduser(args.dir)
    os.makedirs(d, exist_ok=True)
    srcs = sorted({it["transcript"] for it in items})
    mapping = {}
    for s in srcs:
        dst = os.path.join(d, os.path.basename(s))
        if not os.path.exists(dst) or os.path.getsize(dst) != os.path.getsize(s):
            shutil.copyfile(s, dst)
        h = sha1_file(dst)
        mapping[s] = (dst, h)
        print("  %-14s %7.1f MB  sha1=%s" % (os.path.basename(s)[:12],
                                             os.path.getsize(dst) / 1e6, h[:12]))
    with open(os.path.expanduser(args.out), "w", encoding="utf-8") as fo:
        for it in items:
            dst, h = mapping[it["transcript"]]
            it["transcript_live"] = it["transcript"]
            it["transcript"] = dst
            it["transcript_sha1"] = h
            fo.write(json.dumps(it, ensure_ascii=False) + "\n")
    print("[freeze] %d items -> %s (corpus in %s)" % (len(items), args.out, d))
    print("FREEZE_DONE")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
