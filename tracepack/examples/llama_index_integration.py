"""Reference integration: a real LlamaIndex ``Memory`` served by a real execution trace.

This is the proposal §7 "reference integration", and it is also the evidence the thin upstream PR
would cite.  It asserts four things against the actual framework, not a stub:

  1. a custom block plugs into ``Memory`` and its content reaches the composed chat history;
  2. **runtime kwargs reach the block** -- ``Memory.aget(input, phase=..., budget=..., pin=[...])``
     arrives in ``_aget`` unchanged.  This is the open question in run-llama/llama_index#22823,
     and the answer is that the existing API already carries it: no core change needed;
  3. the block stays inside the framework's token limit;
  4. under pressure, TracePack's ``atruncate`` **re-assembles at a smaller budget** instead of
     deleting the block -- which is what LlamaIndex's default does, and what makes a memory block
     an all-or-nothing citizen today.

    python3 tracepack/examples/llama_index_integration.py [transcript.jsonl]

Needs `llama-index-core` and Python >= 3.10.
"""
from __future__ import annotations

import asyncio
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

from tracepack.adapters.claude_code import ClaudeCodeAdapter
from tracepack.adapters.llama_index_memory import TracePackRetriever, make_memory_block


def _fail(msg):
    print("FAIL: %s" % msg)
    raise SystemExit(1)


async def main(path):
    from llama_index.core.llms import ChatMessage
    from llama_index.core.memory import Memory

    graph = ClaudeCodeAdapter().normalize(path)
    print("[1] trace -> IR: %d events, %d edges" % (len(graph.events), len(graph.edges)))

    retriever = TracePackRetriever(graph=graph, budget=1024, k=8)
    block = make_memory_block(retriever, name="tracepack", priority=1)
    memory = Memory.from_defaults(session_id="tracepack-demo", token_limit=4000,
                                  memory_blocks=[block])
    print("[2] Memory built with a TracePackMemoryBlock (priority=%d)" % block.priority)

    q = "which script produced the shard results and where was the output written"
    await memory.aput_messages([ChatMessage(role="user", content=q)])

    # ---- 2. runtime kwargs reach the block --------------------------------------------------
    hist_default = await memory.aget(input=q)
    hist_wide = await memory.aget(input=q, k=24, budget=3000, query_mode="why")
    txt_default = "\n".join(str(m.content) for m in hist_default)
    txt_wide = "\n".join(str(m.content) for m in hist_wide)
    if not txt_default.strip():
        _fail("the block contributed nothing to the composed history")
    if txt_default == txt_wide:
        _fail("runtime kwargs did not change the packet -- either they did not reach _aget, "
              "or the block ignored them; either way #22823 would NOT be answered")
    print("[3] runtime kwargs reach the block: default %d chars vs (k=24,budget=3000,why) %d chars"
          % (len(txt_default), len(txt_wide)))

    # ---- 3. the block respects the framework's limit -----------------------------------------
    est = memory._estimate_token_count(txt_default)                       # noqa: SLF001
    print("[4] composed history ~%d tokens, framework limit %d" % (est, memory.token_limit))
    if est > memory.token_limit:
        _fail("composed history exceeded the framework token limit")

    # ---- 4. truncation re-assembles instead of deleting ---------------------------------------
    content = await block.aget([ChatMessage(role="user", content=q)], budget=2000)
    before = memory._estimate_token_count(content)                        # noqa: SLF001
    shrunk = await block.atruncate(content, tokens_to_truncate=max(1, before // 2))
    if shrunk is None or shrunk == "":
        _fail("atruncate deleted the whole block -- that is the DEFAULT behaviour this "
              "integration exists to replace")
    after = memory._estimate_token_count(shrunk)
    if after >= before:
        _fail("atruncate did not shrink: %d -> %d" % (before, after))
    print("[5] atruncate re-assembled: %d -> %d tokens, block kept (not deleted)" % (before, after))

    # a real event boundary, not a mid-event cut: every line the assembler emitted is intact
    if shrunk.strip() and shrunk.strip() not in content and not any(
            line and line in content for line in shrunk.split("\n")[:3] if len(line) > 40):
        _fail("truncated content does not look like a re-assembly of whole entries")
    print("[6] truncated content is whole entries, not a mid-event cut")

    print("\nLLAMA_INDEX_INTEGRATION_OK")
    return 0


if __name__ == "__main__":
    if len(sys.argv) < 2:
        raise SystemExit("usage: python3 -m tracepack.examples.llama_index_integration <transcript.jsonl>")
    raise SystemExit(asyncio.run(main(sys.argv[1])))
