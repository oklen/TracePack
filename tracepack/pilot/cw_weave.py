#!/usr/bin/env python3
"""Run ContextWeaver over the archived round-2 corpora and read its kept context like a packet.

Three subcommands, in the order PLAN_baselines §6-T3 requires them:

  control   the planted-chain positive control.  RUN THIS FIRST.  If the analyzer cannot put a known
            parent in the top-m, every comparison number below is an implementation artifact and is
            not quotable.  Same judgement this project passed on its own graph arm.
  run       build the kept context for each corpus, under either anchor, and score it with the same
            instruments cw_gate.py uses for the retrieval arms (E_served / answer_in_packet), so the
            two families are read with ONE ruler.
  cost      just the LLM-call and token accounting, for the §5 cost table.

Anchors
-------
  last    faithful ContextWeaver: ancestry from the newest node of the compacted history.  This is
          what the method does at the compaction boundary, BEFORE the next instruction exists.
  query   the ADAPTED variant (arm Wq).  A synthetic node carrying the phase-2 instruction is
          appended, the analyzer picks ITS parents, and the BFS runs from there.  Reported
          separately and never merged into the faithful arm's numbers.

Note what `last` can and cannot show.  ContextWeaver is an ONLINE method: in a live run the anchor
advances with every agent step, so after the agent's first phase-2 action the ancestry is recomputed
against the new goal.  This file measures the boundary only.  The online arm is the live agent run.
"""
from __future__ import annotations

import argparse
import json
import gzip
import os
import sys
import time

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(os.path.dirname(HERE))
if ROOT not in sys.path:
    sys.path.insert(0, ROOT)

from tracepack.baselines import contextweaver as CW                # noqa: E402
from tracepack.pilot import cw_arms, tp_serve                      # noqa: E402
from tracepack.pilot.cw_gate import corpora, geometry              # noqa: E402
from tracepack.pilot.gate_deps import EVIDENCE_HEADER, READER_TAIL, hit, verify_patterns  # noqa: E402

CW_HEADER = ("[contextweaver] Context kept by the dependency-structured memory (ancestors verbatim; "
             "other steps keep their action with the observation replaced):\n\n")


def read_rows(path):
    opener = gzip.open if path.endswith(".gz") else open
    with opener(path, "rt", encoding="utf-8") as fh:
        return [json.loads(l) for l in fh if l.strip()]


def model_id(base):
    import urllib.request
    with urllib.request.urlopen(base.rstrip("/").replace("/v1", "") + "/v1/models", timeout=60) as r:
        return json.load(r)["data"][0]["id"]



def build_id() -> dict:
    """Which build produced this row.

    A run's artifacts must say what code made them.  Two passes of the same configuration differed
    (12/16 vs 14/16) and it took a three-way file comparison to establish that the older one had run
    against a build predating the prefix-leak fix -- a build stamp on the row would have said so
    immediately.  Hash the engine and the binding, not just a timestamp: the mtime survives an rsync
    that changes nothing.
    """
    import hashlib
    out = {}
    for name in ("baselines/contextweaver.py", "pilot/cw_weave.py", "pilot/cw_arms.py"):
        fp = os.path.join(ROOT, "tracepack", name)
        try:
            out[name] = hashlib.sha1(open(fp, "rb").read()).hexdigest()[:12]
        except OSError:
            out[name] = "missing"
    # the leak fix is load-bearing enough to assert on, not just record
    try:
        src = open(os.path.join(ROOT, "tracepack", "baselines", "contextweaver.py"),
                   encoding="utf-8").read()
        out["leak_fix_present"] = "n.goal or goal" in src and "goal: str" in src
    except OSError:
        out["leak_fix_present"] = None
    return out


def query_node(nodes, query: str):
    """The adapted anchor: a node standing for the instruction itself (arm Wq)."""
    n = CW.Node(idx=len(nodes), thought="The user has just asked for this.",
                action="user_instruction", observation=query, validation="unknown")
    n.summary = query[:400]
    return n


def build_for(path, task, cfg: CW.CWConfig, anchor: str):
    rows = read_rows(path)
    query = task["phase2"]
    chat = CW.Chat(cfg.base_url, cfg.model, cfg.api_key, cfg.timeout)
    t0 = time.time()
    # PINNED, not inherited: this driver owns the published ContextWeaver numbers, so it names the
    # candidate rules it ran with rather than taking whatever the module default is today.  Both were
    # later found over-broad and corrected (RESULTS_swebench 3.7); arm V uses the corrected pair.
    nodes = CW.extract_nodes(rows, cfg.validation)
    if not nodes:
        return None, chat, {}
    CW.mark_superseded(nodes, cfg.supersede)
    # LEAK GUARD: the graph is built with the instruction that was in force at each step (Node.goal,
    # recovered from the user messages in the archived corpus), NEVER with the phase-2 query.  Only
    # the CW-Q anchor is allowed to see the new instruction, and only to pick an entry point.
    for n in nodes:
        cands = [c for c in nodes[:n.idx] if c.validation not in ("failed", "superseded")]
        n.parents, n.scores = CW.select_parents(chat, n, cands, n.goal or "complete the task", cfg)
    n_graph_calls = chat.calls
    if anchor == "query":
        qn = query_node(nodes, query)
        nodes.append(qn)
        cands = [c for c in nodes[:qn.idx] if c.validation not in ("failed", "superseded")]
        qn.parents, qn.scores = CW.select_parents(chat, qn, cands, query, cfg)
    CW.summarize(chat, nodes, cfg)
    t1 = time.time()
    k = len(nodes) - 1
    warm = len(nodes) <= cfg.W
    A = tuple(range(len(nodes))) if warm else CW.ancestry(nodes, k, cfg.W)
    text = CW.weave(nodes, A, k)
    res = CW.CWResult(context=text, nodes=nodes, ancestry=A, chars=len(text),
                      tokens=max(1, len(text) // 4), llm_calls=chat.calls, llm_tokens=chat.tokens,
                      graph_ms=int(1000 * (t1 - t0)), select_ms=int(1000 * (time.time() - t1)),
                      warmup=warm)
    return res, chat, {"query": query, "graph_calls": n_graph_calls}



def budget_matched(nodes, ancestry, budget: int, anchor: int) -> dict:
    """The comparison plan's §5.3 requirement: ContextWeaver's kept context at OUR token budget.

    Reported SEPARATELY and labelled a budget-adapted version, because it changes the method's own
    behaviour: the paper keeps every ancestor's observation in full, and a budget can force one out.

    The adaptation is the least destructive one available: ancestors are emitted whole, in trace
    order, nearest-to-the-anchor first (a nearer ancestor is what the anchor actually depends on),
    and an ancestor that does not fit is DROPPED rather than truncated -- never half a step, the
    same rule the shipped packer follows.  Truncating the text instead would cut the tail, which is
    exactly where a one-shot record's value tends to sit, and would make this a straw man.
    """
    order = sorted(ancestry, key=lambda i: (abs(i - anchor), i))
    kept, used, dropped = [], 0, []
    for i in order:
        n = nodes[i]
        unit = "step %d | %s\naction: %s\nobservation: %s" % (n.idx, n.validation, n.action,
                                                               n.observation or "")
        cost = max(1, len(unit) // 4)
        if used + cost > budget:
            dropped.append(i)
            continue
        kept.append(i)
        used += cost
    kept.sort()
    text = "\n\n".join(
        "step %d | %s\naction: %s\nobservation: %s" % (nodes[i].idx, nodes[i].validation,
                                                        nodes[i].action, nodes[i].observation or "")
        for i in kept)
    return {"text": text, "tokens": used, "kept": kept, "dropped": dropped,
            "n_dropped_by_budget": len(dropped)}


def record_nodes(nodes, kv: str):
    """Which nodes carry the one-shot record, i.e. have `id: <K>` in their observation."""
    return [n.idx for n in nodes if kv and ("id: %s" % kv) in (n.observation or "")]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("cmd", choices=["control", "run", "cost"])
    ap.add_argument("run_dir", nargs="?", default="")
    ap.add_argument("--tasks-dir", default=os.path.join(HERE, "tp-pilot-tasks", "tasks"))
    ap.add_argument("--vllm", required=True, help="OpenAI-compatible base URL ending in /v1")
    ap.add_argument("--model", default="")
    ap.add_argument("--anchor", default="last", choices=["last", "query"])
    ap.add_argument("--W", type=int, default=5)
    ap.add_argument("--m", type=int, default=3)
    ap.add_argument("--scorer", default="pairwise", choices=["pairwise", "batched"])
    ap.add_argument("--workers", type=int, default=16)
    ap.add_argument("--no-summaries", action="store_true")
    ap.add_argument("--trials", type=int, default=5)
    ap.add_argument("--only-src", default="C")
    ap.add_argument("--budget", type=int, default=2048,
                    help="token budget for the budget-adapted rendering (plan §5.3)")
    ap.add_argument("--only-rep", default="", help="e.g. r1 -- one replicate is enough for the packet layer")
    ap.add_argument("--out", default="")
    ap.add_argument("--dump", default="")
    ap.add_argument("--supersede", default="legacy", choices=["legacy", "write_path"],
                    help="candidate filtering; DEFAULT legacy, which is what the published numbers ran")
    ap.add_argument("--validation", default="legacy", choices=["legacy", "anchored"],
                    help="failure classifier; DEFAULT legacy, which is what the published numbers ran")
    a = ap.parse_args()

    cfg = CW.CWConfig(W=a.W, m=a.m, model=a.model or model_id(a.vllm), base_url=a.vllm.rstrip("/"),
                      scorer=a.scorer, max_workers=a.workers, summarize=not a.no_summaries,
                      supersede=a.supersede, validation=a.validation)
    BUILD = build_id()
    print("build: %s" % BUILD)
    if BUILD.get("leak_fix_present") is False:
        raise SystemExit("REFUSING TO RUN: this build predates the prefix-leak fix, so the graph "
                         "would be built with the phase-2 instruction.  Ship the current code.")

    if a.cmd == "control":
        out = CW.plant_control(cfg, trials=a.trials)
        print(json.dumps({kk: vv for kk, vv in out.items() if kk != "details"}, indent=1))
        for d in out["details"][:3]:
            print("  top=%s rank_of_true=%s scores=%s" % (d["top"], d["rank_of_true"], d["scores"]))
        print("\nPLANT CONTROL: %s  (rate %.2f, need >= 0.80)"
              % ("PASS" if out["pass"] else "FAIL -- ContextWeaver numbers are NOT quotable",
                 out["rate"]))
        sys.exit(0 if out["pass"] else 3)

    rows, dump = [], (open(a.dump, "w", encoding="utf-8") if a.dump else None)
    out = open(a.out, "w", encoding="utf-8") if a.out else None
    tp_serve.LINK_VALUES, tp_serve.EVIDENCE_HOPS = cw_arms.LINK_VALUES, cw_arms.EVIDENCE_HOPS
    for rep, src, tid, path in corpora(a.run_dir):
        if a.only_src and src != a.only_src:
            continue
        if a.only_rep and rep != a.only_rep:
            continue
        tj = os.path.join(a.tasks_dir, tid, "task.json")
        if not os.path.exists(tj):
            continue
        task = json.load(open(tj, encoding="utf-8"))
        pats = verify_patterns(os.path.join(a.tasks_dir, tid))
        facts = {f["key"]: f for f in task.get("facts", [])}
        kv = (facts.get("K") or facts.get("K2") or {}).get("value", "")
        vval = (facts.get("V") or {}).get("value", "")
        res, chat, meta = build_for(path, task, cfg, a.anchor)
        if res is None:
            print("%s/%s no nodes" % (rep, tid))
            continue
        recs = record_nodes(res.nodes, kv)
        h, n = hit(res.context, pats)
        # the in-compression-loss instrument: does the record's own node SUMMARY keep the value?
        surv = [bool(vval and vval[:40] in (res.nodes[i].summary or "")) for i in recs]
        bm = budget_matched(res.nodes, res.ancestry, a.budget, len(res.nodes) - 1)
        bm_rec = bool(set(recs) & set(bm["kept"]))
        bm_h, bm_n = hit(bm["text"], pats)
        r = {"rep": rep, "task": tid, "kind": task.get("kind"), "arm": "W" if a.anchor == "last" else "Wq",
             "anchor": a.anchor, "W": cfg.W, "m": cfg.m, "scorer": cfg.scorer,
             "n_nodes": len(res.nodes), "ancestry": list(res.ancestry), "warmup": res.warmup,
             "record_nodes": recs, "record_in_ancestry": bool(set(recs) & set(res.ancestry)),
             "value_survives_summary": any(surv), "n_record_nodes": len(recs),
             "E_served": bool(set(recs) & set(res.ancestry)),
             "answer_in_pkt": "%d/%d" % (h, n), "answer_full": h == n and n > 0,
             "chars": res.chars, "tokens": res.tokens,
             "llm_calls": res.llm_calls, "llm_tokens": res.llm_tokens,
             "graph_ms": res.graph_ms, "select_ms": res.select_ms,
             "build": BUILD,
             "graph_llm_calls_build_only": meta.get("graph_calls", 0),
             # the budget-adapted half (plan §5.3): same ancestry, our token budget
             "bm_budget": a.budget, "bm_tokens": bm["tokens"], "bm_kept": bm["kept"],
             "bm_dropped_by_budget": bm["n_dropped_by_budget"],
             "bm_record_kept": bm_rec, "bm_answer_full": bm_h == bm_n and bm_n > 0,
             # node -> event ids, so an ancestry can be replayed without re-running the analyzer
             "node_event_ids": [list(n.event_ids) for n in res.nodes]}
        rows.append(r)
        if out:
            out.write(json.dumps(r, ensure_ascii=False) + "\n")
        if dump:
            dump.write(json.dumps(
                {"key": "%s|%s|%s|2048" % (rep, tid, r["arm"]), "rep": rep, "task": tid,
                 "kind": task.get("kind"), "cond": r["arm"], "budget": 2048, "query": task["phase2"],
                 "packet": res.context, "patterns": pats,
                 "prompt": CW_HEADER + res.context + "\n\n" + task["phase2"] + READER_TAIL},
                ensure_ascii=False) + "\n")
        print("%s/%s %-5s nodes %3d anc %-16s record%s%s tok %5d llm %4d calls / %7d tok  %s"
              % (rep, tid, task.get("kind"), len(res.nodes), str(list(res.ancestry))[:16],
                 "=" + str(recs), " IN-ANC" if r["record_in_ancestry"] else " missed",
                 res.tokens, res.llm_calls, res.llm_tokens,
                 "ans %s" % r["answer_in_pkt"]), flush=True)
    if out:
        out.close()
    if dump:
        dump.close()
    if not rows:
        print("no rows")
        return
    n = len(rows)
    print("\n== ContextWeaver (anchor=%s, W=%d, m=%d, scorer=%s) over %d corpora =="
          % (a.anchor, cfg.W, cfg.m, cfg.scorer, n))
    print("record in ancestry   %d/%d  %.1f%%" % (sum(r["record_in_ancestry"] for r in rows), n,
                                                  100.0 * sum(r["record_in_ancestry"] for r in rows) / n))
    print("answer in context    %d/%d  %.1f%%" % (sum(r["answer_full"] for r in rows), n,
                                                  100.0 * sum(r["answer_full"] for r in rows) / n))
    print("value kept by the record's own node summary  %d/%d"
          % (sum(r["value_survives_summary"] for r in rows), n))
    print("warmup (history <= W, nothing dropped)       %d/%d" % (sum(r["warmup"] for r in rows), n))
    print("context tokens  mean %.0f   LLM calls/cell mean %.0f   LLM tokens/cell mean %.0f"
          % (sum(r["tokens"] for r in rows) / n, sum(r["llm_calls"] for r in rows) / n,
             sum(r["llm_tokens"] for r in rows) / n))
    print("graph build ms  mean %.0f" % (sum(r["graph_ms"] for r in rows) / n))
    if any("bm_tokens" in r for r in rows):
        print("\n-- budget-adapted (plan 5.3: the SAME ancestry rendered under a %d-token budget;\n"
              "   a version of the method, not the method) --" % rows[0].get("bm_budget", 0))
        print("record kept          %d/%d  %.1f%%" % (sum(r["bm_record_kept"] for r in rows), n,
                                                      100.0 * sum(r["bm_record_kept"] for r in rows) / n))
        print("answer in context    %d/%d  %.1f%%" % (sum(r["bm_answer_full"] for r in rows), n,
                                                      100.0 * sum(r["bm_answer_full"] for r in rows) / n))
        print("context tokens  mean %.0f  (unbudgeted: %.0f)"
              % (sum(r["bm_tokens"] for r in rows) / n, sum(r["tokens"] for r in rows) / n))
        print("ancestors the budget had to drop: %d over %d cells"
              % (sum(r["bm_dropped_by_budget"] for r in rows), n))


if __name__ == "__main__":
    main()
