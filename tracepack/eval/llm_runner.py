"""tracepack.eval.llm_runner -- run a prompt pack against any OpenAI-compatible chat endpoint.

The harness never calls a model directly.  `gen_queries.py`, `infer_edges.py` and
`judge_answers.py` write a *prompt pack* (one JSON line per prompt: ``{"key": ..., "prompt": ...}``)
and read back a *result file* (one JSON line per prompt: ``{"key": ..., "text": ...}``).  This
module is the runner in between, so the prompts, the screens and the parsing stay in one place no
matter which model answers.

Configure it with environment variables; there are no defaults, so a missing one fails fast:

    TRACEPACK_LLM_BASE_URL    base URL of an OpenAI-compatible server, e.g. https://api.openai.com/v1
    TRACEPACK_LLM_API_KEY     the key for that server
    TRACEPACK_LLM_MODEL       the model name (``--model`` overrides it)
    TRACEPACK_LLM_EXTRA_BODY  optional JSON merged into every request body, e.g.
                              '{"reasoning_effort": "medium"}' on servers that accept it

The published runs used GPT-5.6-Sol for query generation and judging; any capable model can stand
in, but the numbers that depend on it (the generated questions, the judge verdicts) will differ.
The judge is audited before use either way (`judge_answers.py --audit`).

    python3 -m tracepack.eval.llm_runner --pack pack.jsonl --out res.jsonl [--conc 12]

Resumable: keys already present in ``--out`` are skipped.  A prompt that still fails after the
retries is written with ``"text": ""`` and an ``"error"`` field, and counted at the end.
"""
from __future__ import annotations

import argparse
import concurrent.futures as cf
import json
import os
import threading
import time
import urllib.error
import urllib.request


def _config(model=None):
    missing = [v for v in ("TRACEPACK_LLM_BASE_URL", "TRACEPACK_LLM_API_KEY") if not os.environ.get(v)]
    model = model or os.environ.get("TRACEPACK_LLM_MODEL", "")
    if not model:
        missing.append("TRACEPACK_LLM_MODEL")
    if missing:
        raise RuntimeError("set %s (see tracepack/eval/llm_runner.py)" % ", ".join(missing))
    extra = json.loads(os.environ.get("TRACEPACK_LLM_EXTRA_BODY") or "{}")
    return os.environ["TRACEPACK_LLM_BASE_URL"].rstrip("/"), os.environ["TRACEPACK_LLM_API_KEY"], model, extra


def call(prompt, model=None, effort=None, timeout=600, attempts=3):
    """One user message -> the assistant text.  Retries with backoff; raises after the last attempt.

    ``effort`` is accepted for signature compatibility with the published runner and is NOT sent:
    servers disagree on the parameter, so put it in TRACEPACK_LLM_EXTRA_BODY if yours takes it.
    """
    base, key, model, extra = _config(model)
    body = {"model": model, "messages": [{"role": "user", "content": prompt}]}
    body.update(extra)
    data = json.dumps(body).encode()
    last = ""
    for a in range(attempts):
        try:
            req = urllib.request.Request(base + "/chat/completions", data=data,
                                         headers={"Content-Type": "application/json",
                                                  "Authorization": "Bearer " + key})
            with urllib.request.urlopen(req, timeout=timeout) as r:
                j = json.load(r)
            choices = j.get("choices") or []
            return ((choices[0].get("message") or {}).get("content") or "") if choices else ""
        except (urllib.error.URLError, OSError, ValueError) as e:
            last = repr(e)[:200]
            if a == attempts - 1:
                raise RuntimeError(last)
            time.sleep(min(2 ** a, 8))
    raise RuntimeError(last)


def run_pack(jobs, model=None, effort=None, conc=12, label="llm"):
    """jobs = [(key, prompt)]; returns (dict key -> text, [(key, error)])."""
    _config(model)                      # fail before starting threads, not inside each one
    res, errs = {}, []
    t0 = time.time()
    lock = threading.Lock()
    done = [0]

    def work(job):
        k, p = job
        try:
            t = call(p, model=model, effort=effort)
        except Exception as e:  # noqa: BLE001 -- one bad prompt must not end the pack
            with lock:
                errs.append((k, repr(e)[:150]))
        else:
            with lock:
                res[k] = t
        with lock:
            done[0] += 1
            if done[0] % 25 == 0:
                print("[%s] %d/%d ok=%d err=%d %.1f/min" % (
                    label, done[0], len(jobs), len(res), len(errs),
                    60 * done[0] / max(1e-9, time.time() - t0)), flush=True)

    with cf.ThreadPoolExecutor(max(1, conc)) as ex:
        list(ex.map(work, jobs))
    print("[%s] FINAL ok=%d err=%d in %.1f min" % (label, len(res), len(errs), (time.time() - t0) / 60),
          flush=True)
    return res, errs


def main():
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--pack", required=True, help="prompt pack: one {key, prompt} JSON object per line")
    ap.add_argument("--out", required=True, help="result file: one {key, text} JSON object per line")
    ap.add_argument("--model", default="", help="default: TRACEPACK_LLM_MODEL")
    ap.add_argument("--conc", type=int, default=12)
    args = ap.parse_args()

    jobs = []
    for line in open(args.pack, encoding="utf-8"):
        if line.strip():
            r = json.loads(line)
            jobs.append((r["key"], r["prompt"]))
    have = set()
    if os.path.exists(args.out):
        for line in open(args.out, encoding="utf-8"):
            try:
                r = json.loads(line)
            except ValueError:
                continue
            if r.get("key") and not r.get("error"):
                have.add(r["key"])
    todo = [j for j in jobs if j[0] not in have]
    print("[llm_runner] %d prompts, %d already done, %d to run" % (len(jobs), len(have), len(todo)), flush=True)
    res, errs = run_pack(todo, model=args.model or None, conc=args.conc)
    with open(args.out, "a", encoding="utf-8") as fo:
        for k, _ in todo:
            if k in res:
                fo.write(json.dumps({"key": k, "text": res[k]}, ensure_ascii=False) + "\n")
        for k, e in errs:
            fo.write(json.dumps({"key": k, "text": "", "error": e}, ensure_ascii=False) + "\n")
    print("[llm_runner] wrote %d results, %d errors -> %s" % (len(res), len(errs), args.out))
    return 1 if errs else 0


if __name__ == "__main__":
    raise SystemExit(main())
