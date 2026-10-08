"""End-to-end example / integration check: real Claude Code transcript -> MemoryPacket.

    python3 tracepack/examples/why_end_to_end.py [transcript.jsonl]

Walks the whole control plane on a REAL trace and prints what each stage did:
  normalize -> route (hybrid+pin) -> typed closure -> hard-budget assembly -> manifest.
Exits non-zero if any stage violates a contract we can check here (budget, dangling required,
non-determinism across two identical runs).
"""
from __future__ import annotations

import glob
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

from tracepack.adapters.claude_code import ClaudeCodeAdapter
from tracepack.core.assembler import AssemblerConfig, BudgetAssembler
from tracepack.core.closure import ClosureConfig, TypedClosure
from tracepack.core.router import RouterConfig, make_router

BUDGETS = (1024, 2048, 4096)


def pick_transcript(argv):
    if len(argv) > 1:
        return argv[1]
    cands = sorted(glob.glob(os.path.expanduser("~/.claude/projects/*/*.jsonl")),
                   key=os.path.getsize)
    mid = [c for c in cands if 200_000 < os.path.getsize(c) < 30_000_000]
    if not mid:
        raise SystemExit("no transcript found; pass one as argv[1]")
    return mid[len(mid) // 2]


def main():
    path = pick_transcript(sys.argv)
    print("transcript:", os.path.basename(path), "(%.1f MB)" % (os.path.getsize(path) / 1e6))

    adapter = ClaudeCodeAdapter()
    graph = adapter.normalize(path)
    st = graph.stats()
    print("[normalize] events=%d edges=%d | kinds=%s" % (
        len(graph.events), len(graph.edges),
        {k: v for k, v in sorted(st["kinds"].items())}))
    print("[normalize] edge types=%s" % {k: v for k, v in sorted(st["edge_types"].items()) if v})

    query = "Which file was edited to fix the tokenizer error, and what did the tool output say?"
    router = make_router("hybrid_pin", RouterConfig(k=6))
    seeds = router.retrieve(query, graph, 6)
    print("\n[route] %d seeds: %s" % (
        len(seeds), [(s.event_id[:22], s.source, round(s.score, 3)) for s in seeds[:6]]))

    closure = TypedClosure(ClosureConfig(mode="native")).close(
        query, seeds, graph, query_mode="why")
    print("[closure] required=%d optional=%d steps=%d relaxations=%d" % (
        len(closure.required), len(closure.optional), len(closure.steps), len(closure.relaxations)))
    rules = {}
    for s in closure.steps:
        rules[s.rule] = rules.get(s.rule, 0) + 1
    print("[closure] rules fired: %s" % rules)

    ok = True
    digests = {}
    for b in BUDGETS:
        asm = BudgetAssembler(AssemblerConfig(repr_policy="source_only"))
        pkt = asm.assemble(query, closure, graph, b, seeds=seeds, query_mode="why")
        m = pkt.manifest
        print("\n[assemble budget=%d] tokens=%d entries=%d incomplete=%s missing=%d omitted=%d"
              % (b, m.total_tokens, len(m.entries), m.incomplete,
                 len(m.missing_required), len(m.omitted_optional)))
        print("   digest=%s  first reasons=%s" % (
            m.digest()[:16], [e.reason for e in m.entries[:4]]))
        if m.total_tokens > b:
            print("   !! BUDGET VIOLATION"); ok = False
        if not m.incomplete and m.missing_required:
            print("   !! missing_required without incomplete flag"); ok = False
        # determinism: same inputs twice -> same digest
        pkt2 = BudgetAssembler(AssemblerConfig(repr_policy="source_only")).assemble(
            query, closure, graph, b, seeds=seeds, query_mode="why")
        if pkt2.manifest.digest() != m.digest():
            print("   !! NON-DETERMINISTIC digest"); ok = False
        digests[b] = m.digest()

    # the served context should actually contain the events the manifest claims
    asm = BudgetAssembler(AssemblerConfig(repr_policy="source_only"))
    pkt = asm.assemble(query, closure, graph, 2048, seeds=seeds, query_mode="why")
    served = pkt.context
    for e in pkt.manifest.entries[:5]:
        ev = graph.event(e.event_id)
        probe = (ev.text or "")[:40].strip()
        if probe and probe not in served:
            print("   !! manifest lists %s but its text is not in the context" % e.event_id)
            ok = False
    print("\n[context] %d chars, %d entries; head:\n%s" % (
        len(served), len(pkt.manifest.entries), served[:400].replace("\n", " | ")))

    print("\nE2E_%s" % ("OK" if ok else "FAILED"))
    return 0 if ok else 1


if __name__ == "__main__":
    raise SystemExit(main())
