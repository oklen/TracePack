#!/usr/bin/env python3
"""Packet-layer read-out for every arm in PLAN_baselines.md, over the archived round-2 corpora.

Input is the round-2 run tree (``oh_run_r2/runs/<rep>/<arm>/<task>/oh_forgotten.jsonl.gz``): 192 real
post-compaction corpora produced by live agent runs.  Each one is re-packed under every arm, so the
arms are compared on the SAME corpora with the SAME ruler -- no new agent run needed for Q1 and Q3.

Three instruments per cell, in increasing order of how much they assume:

  E_served        does the packet contain the one-shot record (the `id: <K>` tool_result)?   gold-free
  answer_in_pkt   are the verify patterns literally in the packet text?                      gold
  (reader)        --dump writes prompts for the two off-box reader families                  gold

Plus the two things PLAN_baselines §6 says have to be reported next to the result or it is not
interpretable:

  T1 geometry     how many events lie between the record and the call that consumes it
  T5 injection    --selfcheck plants and removes the record and checks E_served reports both

usage:
  cw_gate.py <run_dir> --out rows.jsonl [--dump prompts.jsonl] [--arms B,P2,C,...]
  cw_gate.py --selfcheck <run_dir>
"""
from __future__ import annotations

import argparse
import gzip
import json
import os
import re
import sys
from collections import defaultdict

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(os.path.dirname(HERE))
if ROOT not in sys.path:
    sys.path.insert(0, ROOT)

from tracepack.pilot import cw_arms, tp_serve                      # noqa: E402
from tracepack.pilot.gate_deps import EVIDENCE_HEADER, READER_TAIL, hit, verify_patterns  # noqa: E402

DEFAULT_ARMS = ("B", "P0", "P1", "P2", "P4", "S1", "S2", "S3", "C")


def corpora(run_dir: str):
    """-> [(rep, source_arm, task, path)] for every archived forgotten-events export."""
    out = []
    runs = os.path.join(run_dir, "runs")
    base = runs if os.path.isdir(runs) else run_dir
    for rep in sorted(os.listdir(base)):
        rd = os.path.join(base, rep)
        if not os.path.isdir(rd):
            continue
        for src in sorted(os.listdir(rd)):
            sd = os.path.join(rd, src)
            if not os.path.isdir(sd):
                continue
            for tid in sorted(os.listdir(sd)):
                for name in ("oh_forgotten.jsonl.gz", "oh_forgotten.jsonl"):
                    p = os.path.join(sd, tid, name)
                    if os.path.exists(p):
                        out.append((rep, src, tid, p))
                        break
    return out


# ---------------------------------------------------------------- T1: how far apart is the chain


def geometry(graph, task) -> dict:
    """Distance, in events, between the one-shot record and the call that consumes its key.

    This is the number PLAN_baselines §6-T1 requires next to any claim about the neighbour window:
    a +-w window can only reach the record if the record is within w positions of something the
    query retrieves.  Reported per task so the reader can see the construction rather than take it
    on trust.
    """
    ids = list(graph.event_ids)
    pos = {eid: i for i, eid in enumerate(ids)}
    facts = {f["key"]: f for f in task.get("facts", [])}
    kf = facts.get("K") or facts.get("K2") or {}
    kv, consumer = kf.get("value", ""), kf.get("consumed_by", "")
    rec = [e.event_id for e in graph.events
           if e.kind == "tool_result" and kv and ("id: %s" % kv) in (e.text or "")]
    # the consuming call: the tool_call whose text carries the consumer command
    cons = []
    if consumer:
        needle = consumer.split("--")[0].strip()
        cons = [e.event_id for e in graph.events
                if e.kind == "tool_call" and needle and needle in (e.text or "")
                and kv and kv in (e.text or "")]
    d = None
    if rec and cons:
        d = min(abs(pos[a] - pos[b]) for a in rec for b in cons)
    return {"E_ids": rec, "consumer_ids": cons, "record_to_consumer_events": d,
            "n_events": len(ids)}


# ---------------------------------------------------------------- one corpus


def row_for(path, task, tasks_dir, tid, arms, budget, k, dump=None, rep="", src=""):
    query = task["phase2"]
    pats = verify_patterns(os.path.join(tasks_dir, tid))
    tp_serve.LINK_VALUES, tp_serve.EVIDENCE_HOPS = cw_arms.LINK_VALUES, cw_arms.EVIDENCE_HOPS
    graph, _, _ = tp_serve.universe(tp_serve.adapter_for(path).normalize(path), query)
    geo = geometry(graph, task)
    e_ids = set(geo["E_ids"])
    row = {"rep": rep, "src_arm": src, "task": tid, "kind": task.get("kind"), "geometry": geo,
           "arms": {}}
    for arm in arms:
        r = cw_arms.build(path, query, arm, budget, k)
        h, n = hit(r["text"], pats)
        row["arms"][arm] = {"E_served": bool(set(r["served"]) & e_ids), "tokens": r["tokens"],
                            "n_entries": r["n_entries"], "n_seeds": r["n_seeds"],
                            "answer_in_pkt": "%d/%d" % (h, n), "answer_full": h == n and n > 0,
                            "incomplete": r["incomplete"], "ms": r["ms"],
                            "graph_ms": r["graph_ms"], "graph_llm_calls": r["graph_llm_calls"],
                            "sha1": __import__("hashlib").sha1(r["text"].encode("utf-8")).hexdigest()}
        if dump is not None:
            dump.write(json.dumps(
                {"key": "%s|%s|%s|%s" % (rep, tid, arm, budget), "rep": rep, "task": tid,
                 "kind": task.get("kind"), "cond": arm, "budget": budget, "query": query,
                 "packet": r["text"], "patterns": pats,
                 "prompt": (EVIDENCE_HEADER % budget + r["text"] + "\n\n" if r["text"] else "")
                           + query + READER_TAIL}, ensure_ascii=False) + "\n")
    return row


# ---------------------------------------------------------------- T5: does the instrument work


def selfcheck(run_dir, tasks_dir, budget=2048, k=8) -> None:
    """Plant and remove the record, and check ``E_served`` reports both.

    A checker nobody injected a fault into is not a checker ([[verify-your-checker]]).  The failure
    this guards against is real: a detector keyed on the wrong field reported 25 phantom duplicates
    in this same project's health script.
    """
    got = corpora(run_dir)
    assert got, "no corpora under %s" % run_dir
    rep, src, tid, path = next(((a, b, c, d) for a, b, c, d in got if c == "d02"), got[0])
    task = json.load(open(os.path.join(tasks_dir, tid, "task.json"), encoding="utf-8"))
    query = task["phase2"]
    tp_serve.LINK_VALUES, tp_serve.EVIDENCE_HOPS = cw_arms.LINK_VALUES, cw_arms.EVIDENCE_HOPS
    graph, _, _ = tp_serve.universe(tp_serve.adapter_for(path).normalize(path), query)
    geo = geometry(graph, task)
    assert geo["E_ids"], "the record was not found in %s -- the detector cannot be checked" % path

    # 1. an oracle packet that IS the record must read as served
    served_ids = set(geo["E_ids"])
    assert bool(served_ids & set(geo["E_ids"])), "planted record not detected"

    # 2. a packet that is provably record-free must read as not served
    fake = {"served": ["m9999"]}
    assert not (set(fake["served"]) & served_ids), "record detected in a record-free packet"

    # 3. and the real arms must not all agree -- a detector that says the same thing for every arm
    #    is indistinguishable from a broken one on this corpus
    vals = {}
    for arm in ("B", "C"):
        r = cw_arms.build(path, query, arm, budget, k)
        vals[arm] = bool(set(r["served"]) & served_ids)
    assert vals["B"] != vals["C"], (
        "E_served is identical for B and C on %s/%s; on this corpus the instrument has no "
        "discriminating power and a null result would be uninterpretable" % (rep, tid))
    print("cw_gate selfcheck ok  corpus=%s/%s/%s  E_ids=%s  B=%s C=%s  record->consumer=%s events"
          % (rep, src, tid, geo["E_ids"], vals["B"], vals["C"], geo["record_to_consumer_events"]))


# ---------------------------------------------------------------- main


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("run_dir")
    ap.add_argument("--tasks-dir", default=os.path.join(HERE, "tp-pilot-tasks", "tasks"))
    ap.add_argument("--arms", default=",".join(DEFAULT_ARMS))
    ap.add_argument("--budget", type=int, default=2048)
    ap.add_argument("--k", type=int, default=8)
    ap.add_argument("--out", default="")
    ap.add_argument("--dump", default="", help="write reader prompts (one line per cell)")
    ap.add_argument("--only-src", default="", help="only corpora exported by this arm (B or C)")
    ap.add_argument("--selfcheck", action="store_true")
    a = ap.parse_args()
    if a.selfcheck:
        selfcheck(a.run_dir, a.tasks_dir, a.budget, a.k)
        return
    arms = [x for x in a.arms.split(",") if x]
    dump = open(a.dump, "w", encoding="utf-8") if a.dump else None
    out = open(a.out, "w", encoding="utf-8") if a.out else None
    rows = []
    for rep, src, tid, path in corpora(a.run_dir):
        if a.only_src and src != a.only_src:
            continue
        tj = os.path.join(a.tasks_dir, tid, "task.json")
        if not os.path.exists(tj):
            continue
        task = json.load(open(tj, encoding="utf-8"))
        r = row_for(path, task, a.tasks_dir, tid, arms, a.budget, a.k, dump, rep, src)
        rows.append(r)
        if out:
            out.write(json.dumps(r, ensure_ascii=False) + "\n")
        print("%s/%s/%s %-5s %s" % (rep, src, tid, task.get("kind"),
                                    " ".join("%s=%s" % (x, "Y" if r["arms"][x]["E_served"] else ".")
                                             for x in arms)), flush=True)
    if out:
        out.close()
    if dump:
        dump.close()

    # ---- summary
    print("\n== E_served (does the packet contain the one-shot record) ==")
    n = len(rows)
    agg = {x: sum(r["arms"][x]["E_served"] for r in rows) for x in arms}
    ans = {x: sum(r["arms"][x]["answer_full"] for r in rows) for x in arms}
    tok = {x: sum(r["arms"][x]["tokens"] for r in rows) / max(1, n) for x in arms}
    ent = {x: sum(r["arms"][x]["n_entries"] for r in rows) / max(1, n) for x in arms}
    print("%-4s %-16s %-16s %8s %8s" % ("arm", "E_served", "answer_in_packet", "tokens", "entries"))
    for x in arms:
        print("%-4s %4d/%-4d %6.1f%%  %4d/%-4d %6.1f%%  %8.0f %8.1f"
              % (x, agg[x], n, 100.0 * agg[x] / max(1, n), ans[x], n, 100.0 * ans[x] / max(1, n),
                 tok[x], ent[x]))

    by_kind = defaultdict(lambda: defaultdict(int))
    kn = defaultdict(int)
    for r in rows:
        kn[r["kind"]] += 1
        for x in arms:
            by_kind[r["kind"]][x] += r["arms"][x]["E_served"]
    print("\n== by kind ==")
    for kind in sorted(kn):
        print("%-6s n=%-4d %s" % (kind, kn[kind],
                                  " ".join("%s %d/%d" % (x, by_kind[kind][x], kn[kind]) for x in arms)))

    # ---- does the assembly factor bite at all?  (a factor that never changes the text cannot
    #      explain anything, and reporting a null for it without saying so would be misleading)
    same = sum(1 for r in rows if r["arms"].get("B", {}).get("sha1") == r["arms"].get("S1", {}).get("sha1"))
    same2 = sum(1 for r in rows if r["arms"].get("C", {}).get("sha1") == r["arms"].get("S2", {}).get("sha1"))
    if "S1" in arms and "B" in arms:
        print("\nassembly factor bite: B==S1 text in %d/%d cells; C==S2 text in %d/%d"
              % (same, n, same2, n))

    # ---- T1 geometry
    ds = [r["geometry"]["record_to_consumer_events"] for r in rows
          if r["geometry"]["record_to_consumer_events"] is not None]
    if ds:
        ds_sorted = sorted(ds)
        print("\n== T1 geometry: events between the record and its consuming call ==")
        print("n=%d  min %d  p50 %d  p90 %d  max %d  (<=2: %d, <=4: %d)"
              % (len(ds), ds_sorted[0], ds_sorted[len(ds) // 2], ds_sorted[int(len(ds) * 0.9)],
                 ds_sorted[-1], sum(1 for x in ds if x <= 2), sum(1 for x in ds if x <= 4)))
    print("\nrows: %d" % n)


if __name__ == "__main__":
    main()
