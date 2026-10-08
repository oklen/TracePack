"""tracepack.pilot.tp_serve -- evidence service for the pi extension (PLAN_pilot §2, arms B / C).

POST /evidence  {"session_file": ..., "query": ..., "mode": "bm25" | "tracepack", "budget": 2048, "k": 8}
  -> {"text": ..., "tokens": ..., "n_entries": ..., "served": [...], "mode": ..., "ms": ...}

* ``bm25``     : arm B.  Same router (BM25 over event text, k seeds), no closure, no pairing; the
                 seeds' raw text packed chronologically under the same hard budget.
* ``tracepack``: arm C.  Same seeds, then the gold-blind default the WP4 results support: a seed
                 tool_call brings its own tool_result (``paired``), native closure, evidence-first
                 pack, hard budget, never a half event.

Both modes read the LIVE pi session file through ``tracepack.adapters.pi`` (the same adapter the
offline results used), so the extension never has to describe the history itself.  The graph is
rebuilt per request (sessions in the pilot are a few hundred events; ~100 ms) -- correctness over
caching, and no stale-graph bug class.

Runs on python3.9+ with no third-party dependency (stdlib http.server).
"""
from __future__ import annotations

import json
import os
import sys
import time
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer

ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, ROOT)

from tracepack.adapters.base import open_trace                    # noqa: E402
from tracepack.adapters.openhands import OpenHandsAdapter        # noqa: E402
from tracepack.adapters.pi import PiAdapter                      # noqa: E402
from tracepack.core.assembler import AssemblerConfig, BudgetAssembler   # noqa: E402
from tracepack.core.closure import ClosureConfig, TypedClosure   # noqa: E402
from tracepack.core.graph import TraceGraph                      # noqa: E402
from tracepack.core.router import RouterConfig, make_router      # noqa: E402
from tracepack.core.schema import Seed                           # noqa: E402

MODES = ("bm25", "tracepack")


def paired(seeds, graph):
    """A seed tool_call brings its own tool_result (copied verbatim from eval/emit_phase_a.py)."""
    have = {s.event_id for s in seeds}
    out = list(seeds)
    for s in seeds:
        ev = graph.event(s.event_id)
        if getattr(ev, "kind", "") != "tool_call":
            continue
        for ed in graph.children(s.event_id, "RESULT_OF"):
            rid = ed.src_id
            if rid not in have:
                have.add(rid)
                out.append(Seed(event_id=rid, score=s.score, source=s.source, rank=s.rank, pinned=s.pinned))
    return out


LINK_VALUES = False    # round 2: --link-values turns on DEPENDS_ON(value) edges in both adapters
EVIDENCE_HOPS = 1      # round 2: --hops 2 lets the evidence layer reach two-hop dependencies
TOOL_RECORDS_ONLY = False   # round 2: retrieval universe = user + tool records (agent-authored text dropped, both modes)


def adapter_for(session_file: str):
    """pi sessions and OpenHands exports are both transcripts on disk; the first line says which."""
    with open_trace(session_file) as fh:
        head = fh.readline()
    try:
        if json.loads(head or "{}").get("tracepack_format") == "openhands":
            return OpenHandsAdapter(link_values=LINK_VALUES)
    except Exception:
        pass
    return PiAdapter(link_values=LINK_VALUES)


def in_context_ids(graph, query: str) -> set:
    """Events that are already in the model's context when the call is made: the compaction summary
    itself, everything after the last compaction (kept verbatim), and the current user message (the
    query). The first full run served all three back to the model (the summary and the query were
    two of six entries in every packet), wasting budget on text the model already had."""
    ids = list(graph.event_ids)
    ex = set()
    last_sum = None
    for i, eid in enumerate(ids):
        ev = graph.event(eid)
        if ev.kind == "summary" and (ev.meta or {}).get("compaction") == "1":
            last_sum = i
    if last_sum is not None:
        ex.update(ids[last_sum:])
    q = " ".join(query.split())
    for eid in ids:
        ev = graph.event(eid)
        if ev.kind == "user" and " ".join((ev.text or "").split()) == q:
            ex.add(eid)
    return ex


def without(graph, ex: set):
    """A graph restricted to the events not in ``ex`` (edges touching them dropped)."""
    if not ex:
        return graph
    evs = [e for e in graph.events if e.event_id not in ex]
    eds = [d for d in graph.edges if d.src_id not in ex and d.dst_id not in ex]
    return TraceGraph(evs, eds)


def universe(full, query: str):
    """The event set the service actually searches: minus what is already in context, and -- round 2,
    TOOL_RECORDS_ONLY -- minus agent-authored text (assistant turns, compaction summaries, no-op tool
    observations, and OpenHands `finish` observations, which are the agent's own turn-final reply).
    Both modes search this same set; the gate and the probe compute ranks on it too."""
    ex = in_context_ids(full, query)
    graph = without(full, ex)
    dropped = set()
    if TOOL_RECORDS_ONLY:
        dropped = {ev.event_id for ev in graph.events if ev.kind in ("assistant", "summary")
                   or (ev.meta or {}).get("noop") == "1" or (ev.meta or {}).get("agent_authored") == "1"}
        graph = without(graph, dropped)
    return graph, ex, dropped


def evidence(session_file: str, query: str, mode: str, budget: int, k: int, exclude_after: str = "") -> dict:
    if mode not in MODES:
        raise ValueError("mode must be one of %s" % (MODES,))
    t0 = time.time()
    full = adapter_for(session_file).normalize(session_file)
    graph, ex, dropped = universe(full, query)
    cfg = RouterConfig(k=k)
    seeds = list(make_router("lexical", cfg).retrieve(query, graph, k))
    if mode == "tracepack":
        seeds = paired(seeds, graph)
        closure = TypedClosure(ClosureConfig(mode="native")).close(query, seeds, graph, query_mode="lookup")
        # LEGACY PACKER, on purpose: every published TracePack number was produced before the budget
        # non-monotonicity fix (RESULTS_hops §7).  Flip these two off to get the fixed packer --
        # measured effect on this corpus is in that section.
        acfg = AssemblerConfig(repr_policy="source_only", pack="evidence_first", evidence_hops=EVIDENCE_HOPS, evidence_share=0.5,
                                      cost_order=False, unit_cap_share=None, excerpt=False)
    else:
        closure = TypedClosure(ClosureConfig(mode="off")).close(query, seeds, graph, query_mode="lookup")
        acfg = AssemblerConfig(repr_policy="source_only", pack="chrono")
    packet = BudgetAssembler(acfg).assemble(query, closure, graph, budget, seeds=seeds, query_mode="lookup")
    m = packet.manifest
    return {"text": packet.context, "tokens": m.total_tokens, "n_entries": len(m.entries),
            "served": [e.event_id for e in m.entries], "incomplete": bool(m.incomplete),
            "mode": mode, "n_events": _n_events(graph), "n_seeds": len(seeds), "n_in_context_excluded": len(ex),
            "link_values": LINK_VALUES, "evidence_hops": EVIDENCE_HOPS, "tool_records_only": TOOL_RECORDS_ONLY, "n_dropped_agent_text": len(dropped), "n_edges_depends": sum(1 for d in graph.edges if d.edge_type == "DEPENDS_ON"),
            "ms": int(1000 * (time.time() - t0))}


def _n_events(graph) -> int | None:
    """``graph.events`` is a tuple on TraceGraph (the first smoke run called it -> TypeError on every request)."""
    ev = getattr(graph, "events", None)
    if ev is None:
        return None
    if callable(ev):
        ev = ev()
    try:
        return len(ev)
    except TypeError:
        return len(list(ev))


class Handler(BaseHTTPRequestHandler):
    def log_message(self, fmt, *args):  # quieter
        sys.stderr.write("[tp_serve] " + (fmt % args) + "\n")

    def do_POST(self):
        n = int(self.headers.get("Content-Length") or 0)
        body = json.loads(self.rfile.read(n) or b"{}")
        try:
            if self.path != "/evidence":
                raise ValueError("unknown path %s" % self.path)
            out = evidence(body["session_file"], body["query"], body.get("mode", "tracepack"),
                           int(body.get("budget", 2048)), int(body.get("k", 8)))
            code = 200
        except Exception as e:  # report, never hide -- the runner records every failure
            out = {"error": "%s: %s" % (type(e).__name__, e)}
            code = 500
        data = json.dumps(out, ensure_ascii=False).encode("utf-8")
        self.send_response(code)
        self.send_header("Content-Type", "application/json")
        self.send_header("Content-Length", str(len(data)))
        self.end_headers()
        self.wfile.write(data)

    def do_GET(self):
        data = b'{"ok": true}'
        self.send_response(200)
        self.send_header("Content-Type", "application/json")
        self.send_header("Content-Length", str(len(data)))
        self.end_headers()
        self.wfile.write(data)


def main():
    import argparse
    ap = argparse.ArgumentParser()
    ap.add_argument("--port", type=int, default=8765)
    ap.add_argument("--selfcheck", default="", help="a pi session .jsonl: print both modes for a query and exit")
    ap.add_argument("--query", default="what port does the service use")
    ap.add_argument("--link-values", action="store_true", help="round 2: DEPENDS_ON(value) edges on")
    ap.add_argument("--hops", type=int, default=1, help="round 2: evidence hops for the tracepack pack")
    ap.add_argument("--tool-records-only", action="store_true", help="round 2: drop assistant/summary/noop/agent-authored events from retrieval (both modes)")
    a = ap.parse_args()
    global LINK_VALUES, EVIDENCE_HOPS, TOOL_RECORDS_ONLY
    LINK_VALUES = bool(a.link_values); EVIDENCE_HOPS = int(a.hops)
    TOOL_RECORDS_ONLY = bool(a.tool_records_only)
    if a.selfcheck:
        for mode in MODES:
            out = evidence(a.selfcheck, a.query, mode, 2048, 8)
            print("==", mode, {k: v for k, v in out.items() if k != "text"})
            print(out["text"][:800])
        return
    # threading: with several runner lanes in parallel a single-threaded server makes them queue
    ThreadingHTTPServer(("127.0.0.1", a.port), Handler).serve_forever()


if __name__ == "__main__":
    main()
