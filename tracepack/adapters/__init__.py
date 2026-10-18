"""Trace adapters: OpenHands, pi, Claude Code, LlamaIndex memory.

Each turns a native transcript -- a path to a .jsonl export, or an in-memory sequence of rows --
into a `TraceGraph`.  The rows form is what makes the packet builder usable live, inside the loop
that is producing the trace.
"""
