"""tracepack.bench.codememo -- does the right evidence come back, inside a token budget?

CodeMemo (MIT, github.com/laynepenney/codememo-benchmark) has 158 questions over 66 real Claude Code
sessions from three projects. Every question names the turns that hold its answer ("evidence").
This benchmark asks, for each question and budget, whether a method hands back those exact turns.
It is model-free, so it is deterministic and costs nothing to run.

Each project's sessions are concatenated in order into one long history, as if it were one
session compacted many times. Methods, all under the same token budget:

  tracepack   `tracepack.session.recall` over the whole history: what the plugin's recall tool does
  bm25        BM25 over messages, best first, whole messages until the budget is full
  recent      the newest messages that fit: what is still in context when nothing is recalled

Metrics per budget:
  evidence    share of questions where at least one evidence turn is delivered
  all         share of questions where every evidence turn is delivered
  value       among questions whose short answer contains an exact value (a number, version,
              path or identifier), the share where that value appears verbatim in what was delivered

    python3 -m tracepack.bench.codememo --data /path/to/codememo [--budgets 1000,2000,4000] [--json out.json]
"""
from __future__ import annotations

import argparse
import json
import os
import re
import sys
import tempfile
import time

from tracepack import session as S
from tracepack.adapters.claude_code import ClaudeCodeAdapter
from tracepack.core.router import RouterConfig, make_router

_VALUE = re.compile(r"[A-Za-z0-9_./:\-]*\d[A-Za-z0-9_./:\-]*|[A-Za-z_][A-Za-z0-9_]*_[A-Za-z0-9_]+|"
                    r"[A-Za-z0-9_\-]+\.(?:py|rs|toml|json|ya?ml|md|sql|sh|ts|js|txt|cfg|ini)\b")


def load_project(pdir: str):
    """-> (rows, line_of[(session, turn_index)] -> 1-based line in the combined history, questions)."""
    sdir = os.path.join(pdir, "sessions")
    rows, line_of = [], {}
    for name in sorted(os.listdir(sdir)):
        if not name.endswith(".jsonl"):
            continue
        sid = name[:-6]
        with open(os.path.join(sdir, name), encoding="utf-8") as fh:
            srows = [json.loads(l) for l in fh if l.strip()]
        for i, r in enumerate(srows):
            line_of[(sid, i)] = len(rows) + 1
            rows.append(r)
    with open(os.path.join(pdir, "questions.json"), encoding="utf-8") as fh:
        qs = json.load(fh)
    return rows, line_of, qs


def values_of(text: str):
    """Exact values worth checking verbatim: tokens with a digit, snake_case identifiers, file names."""
    out = []
    for m in _VALUE.finditer(text or ""):
        v = m.group(0).strip(".:-/")
        if len(v) >= 3 and not v.isalpha():
            out.append(v)
    return sorted(set(out))


def _line(eid: str) -> int:
    return int(eid.split(":")[1])


def method_tracepack(path, graph, question, budget):
    res = S.recall(path, question, budget=budget, include_recent=True, max_mb=1024)
    lines = {int(e["line"]) for e in res.get("entries", []) if e.get("line")}
    return lines, res.get("text", "")


_BM25 = {}


def method_bm25(path, graph, question, budget):
    if _BM25.get("graph") is not graph:                 # one router per history keeps its index
        _BM25.clear()
        _BM25.update(graph=graph, router=make_router("lexical", RouterConfig(k=200)))
    seeds = _BM25["router"].retrieve(question, graph, 200)
    used, lines, parts = 0, set(), []
    for s in seeds:
        ev = graph.event(s.event_id)
        if S._is_noise(ev) or used + ev.token_cost > budget:
            continue
        used += ev.token_cost
        lines.add(_line(ev.event_id))
        parts.append(ev.text or "")
    return lines, "\n\n".join(parts)


def method_recent(path, graph, question, budget):
    used, lines, parts = 0, set(), []
    for ev in reversed(graph.events):
        if S._is_noise(ev):
            continue
        if used + ev.token_cost > budget:
            break
        used += ev.token_cost
        lines.add(_line(ev.event_id))
        parts.append(ev.text or "")
    return lines, "\n\n".join(reversed(parts))


_GREP = {}


def method_grep(path, graph, question, budget):
    """What an agent does without TracePack: grep its own transcript file for the question's two most
    specific words (fewest matching lines), and read the matching JSON lines in file order, each cut to
    2,000 characters, until the budget is full."""
    if _GREP.get("path") != path:
        with open(path, encoding="utf-8", errors="replace") as fh:
            raw = fh.read().split("\n")
        _GREP.clear()
        _GREP.update(path=path, raw=raw, low=[l.lower() for l in raw])
    raw, low = _GREP["raw"], _GREP["low"]
    words = sorted(set(S._terms(question)))
    df = {w: sum(1 for l in low if w in l) for w in words}
    picks = [w for w in sorted(words, key=lambda w: (df[w] == 0, df[w])) if df[w] > 0][:2]
    if not picks:
        return set(), ""
    used, lines, parts = 0, set(), []
    for i, l in enumerate(low):
        if not any(w in l for w in picks):
            continue
        piece = raw[i][:2000]
        cost = S.estimate_text_tokens(piece) + 1
        if used + cost > budget:
            break
        used += cost
        lines.add(i + 1)
        parts.append(piece)
    return lines, "\n".join(parts)


METHODS = {"tracepack": method_tracepack, "bm25": method_bm25, "grep": method_grep, "recent": method_recent}


def _blank():
    return {"n": 0, "any": 0, "all": 0, "nv": 0, "v": 0, "ms": 0.0}


def run(data_dir: str, budgets, methods=("tracepack", "bm25", "grep", "recent"), quiet=False) -> dict:
    """Every method on every question and budget; also split by question-number parity, because TracePack's
    defaults were chosen on the odd-numbered questions (the even half is the held-out check)."""
    stats = {m: {b: _blank() for b in budgets} for m in methods}
    halves = {h: {m: {b: _blank() for b in budgets} for m in methods} for h in ("odd", "even")}
    per_cat = {}
    projects = sorted(d for d in os.listdir(data_dir) if os.path.isfile(os.path.join(data_dir, d, "questions.json")))
    tmp = tempfile.mkdtemp(prefix="tp-codememo-")
    n_q = 0
    for proj in projects:
        rows, line_of, qs = load_project(os.path.join(data_dir, proj))
        path = os.path.join(tmp, proj + ".jsonl")
        with open(path, "w", encoding="utf-8") as fh:
            for r in rows:
                fh.write(json.dumps(r) + "\n")
        graph, info = S.load_graph(path, max_mb=1024)
        if not quiet:
            print("%s: %d sessions, %d messages, %d events, %.1f MB" % (
                proj, len({k[0] for k in line_of}), len(rows), info["events"], info["bytes"] / 1048576.0), file=sys.stderr)
        for q in qs:
            ev_lines = {line_of.get((e.get("session_id"), e.get("turn_index"))) for e in q.get("evidence", [])}
            ev_lines.discard(None)
            if not ev_lines:
                continue
            n_q += 1
            half = "odd" if int(re.search(r"q(\d+)", q["id"]).group(1)) % 2 else "even"
            vals = values_of(q.get("answer_short") or "")
            for m in methods:
                for b in budgets:
                    t0 = time.time()
                    lines, text = METHODS[m](path, graph, q["question"], b)
                    hit_any = bool(lines & ev_lines)
                    for st in (stats[m][b], halves[half][m][b]):
                        st["ms"] += (time.time() - t0) * 1000
                        st["n"] += 1
                        st["any"] += hit_any
                        st["all"] += ev_lines <= lines
                        if vals:
                            st["nv"] += 1
                            st["v"] += all(v in text for v in vals)
                    key = (m, b, q.get("category"))
                    c = per_cat.setdefault(key, [0, 0])
                    c[0] += 1
                    c[1] += hit_any
    out = {"questions": n_q, "budgets": list(budgets), "methods": {}}
    for m in methods:
        out["methods"][m] = {}
        for b in budgets:
            st = stats[m][b]
            out["methods"][m][str(b)] = {
                "evidence": round(st["any"] / max(1, st["n"]), 3), "all_evidence": round(st["all"] / max(1, st["n"]), 3),
                "exact_value": round(st["v"] / max(1, st["nv"]), 3), "value_questions": st["nv"],
                "ms_per_question": round(st["ms"] / max(1, st["n"]), 1)}
    out["halves"] = {h: {m: {str(b): {"n": s["n"], "evidence": round(s["any"] / max(1, s["n"]), 3),
                                       "exact_value": round(s["v"] / max(1, s["nv"]), 3), "value_questions": s["nv"]}
                            for b, s in by_b.items()} for m, by_b in hm.items()} for h, hm in halves.items()}
    out["by_category"] = {"%s|%s|%s" % k: round(v[1] / max(1, v[0]), 3) for k, v in sorted(per_cat.items(), key=str)}
    return out


def main(argv=None) -> int:
    p = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    p.add_argument("--data", required=True, help="folder with project_*/ (questions.json + sessions/)")
    p.add_argument("--budgets", default="1000,2000,4000")
    p.add_argument("--json", default="")
    a = p.parse_args(argv)
    budgets = [int(x) for x in a.budgets.split(",") if x.strip()]
    res = run(a.data, budgets)
    print("CodeMemo: %d questions; share with the evidence turn delivered (exact values in parentheses)" % res["questions"])
    print("%-10s" % "budget" + "".join("%18s" % b for b in budgets))
    for m, by_b in res["methods"].items():
        print("%-10s" % m + "".join("%10.1f%% (%3.0f%%)" % (100 * by_b[str(b)]["evidence"], 100 * by_b[str(b)]["exact_value"])
                                   for b in budgets))
    if a.json:
        with open(a.json, "w", encoding="utf-8") as fh:
            json.dump(res, fh, indent=1)
    return 0


if __name__ == "__main__":
    sys.exit(main())
