"""Contract 18: the Claude Code plugin path (session recall, the after-compaction hook, the MCP server).

Everything runs on synthetic transcripts written by `tracepack.selftest` (made-up content, no real
session) inside temporary folders; nothing touches ~/.claude or the network.
"""
from __future__ import annotations

import io
import json
import os
import sys
import tempfile
import unittest

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(os.path.dirname(HERE))
if ROOT not in sys.path:
    sys.path.insert(0, ROOT)

import tracepack  # noqa: E402
from tracepack import hooks, mcp_server, redact, session as S  # noqa: E402
from tracepack.selftest import DEMO_FACTS, demo_rows, write_demo_session  # noqa: E402


def _write(path, rows):
    with open(path, "w", encoding="utf-8") as fh:
        for r in rows:
            fh.write(json.dumps(r) + "\n")
    return path


def _precompact_rows():
    """The demo session as the SessionStart(compact) hook sees it: the summary is not written yet."""
    return [r for r in demo_rows() if r.get("subtype") != "compact_boundary" and not r.get("isCompactSummary")]


class _Env(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.dir = self.tmp.name
        self._env = dict(os.environ)
        os.environ["TRACEPACK_DATA"] = os.path.join(self.dir, "data")
        os.environ["CLAUDE_CONFIG_DIR"] = os.path.join(self.dir, "claude")
        for k in ("CLAUDE_PID", "CLAUDE_CODE_SESSION_ID", "CLAUDE_PROJECT_DIR", "TRACEPACK_REDACT",
                  "TRACEPACK_INJECT_BUDGET"):
            os.environ.pop(k, None)
        S._CACHE.clear()
        self.demo = write_demo_session(os.path.join(self.dir, "demo.jsonl"))

    def tearDown(self):
        os.environ.clear()
        os.environ.update(self._env)
        S._CACHE.clear()
        self.tmp.cleanup()


class Contract18PluginRecall(_Env):
    def test_18a_recall_returns_exact_values_with_provenance(self):
        res = S.recall(self.demo, "p95 latency before and after the fix", budget=800)
        self.assertTrue(res["found"])
        for v in (DEMO_FACTS["before_p95"], DEMO_FACTS["after_p95"]):
            self.assertIn(v, res["text"])
        self.assertIn("Bash output", res["text"])
        self.assertRegex(res["text"], r"line \d+")
        self.assertLessEqual(res["tokens"], res["budget"])

    def test_18b_only_out_of_context_records(self):
        res = S.recall(self.demo, "write the report summary benchmark", budget=2000)
        self.assertNotIn("This session is being continued", res["text"])     # the summary is in context
        ids = {e["event_id"] for e in res["entries"]}
        g, _ = S.load_graph(self.demo)
        self.assertFalse(ids & S.in_context_ids(g))

    def test_18c_no_compaction_means_nothing_to_recall(self):
        pre = _write(os.path.join(self.dir, "pre.jsonl"), _precompact_rows())
        res = S.recall(pre, "p95 latency", budget=800)
        self.assertFalse(res["found"])
        self.assertIn("not been compacted", res["text"])
        self.assertTrue(S.recall(pre, "p95 latency", budget=800, include_recent=True)["found"])

    def test_18d_deterministic(self):
        a = S.recall(self.demo, "protected migration id", budget=600)["text"]
        S._CACHE.clear()
        b = S.recall(self.demo, "protected migration id", budget=600)["text"]
        self.assertEqual(a, b)

    def test_18e_expand_pages_a_long_record(self):
        rows = demo_rows()
        long_out = "\n".join("row %04d value=%d" % (i, i * 7) for i in range(3000))
        rows[2]["message"]["content"][0]["content"] = long_out      # the first Bash output, now 3,000 lines
        path = _write(os.path.join(self.dir, "long.jsonl"), rows)
        first = S.expand(path, "3", max_tokens=300)
        self.assertTrue(first["found"])
        self.assertTrue(first["more"])
        self.assertIn("row 0000 value=0", first["text"])
        nxt = S.expand(path, "3", start_line=first["next_line"], max_tokens=300)
        self.assertNotIn("row 0000 value=0", nxt["text"])

    def test_18f_pack_reads_claude_code_transcripts(self):
        pkt = tracepack.pack(self.demo, query="p95 latency after the fix", budget=1000)   # 0.1.0 found 0 events here
        self.assertGreater(pkt["n_events"], 0)
        self.assertGreater(pkt["n_entries"], 0)


class Contract18PluginHook(_Env):
    def test_18g_after_compaction_adds_latest_output(self):
        pre = _write(os.path.join(self.dir, "pre.jsonl"), _precompact_rows())
        out = hooks.session_start({"source": "compact", "transcript_path": pre, "session_id": "s1"})
        d = json.loads(out)
        ctx = d["hookSpecificOutput"]["additionalContext"]
        self.assertEqual(d["hookSpecificOutput"]["hookEventName"], "SessionStart")
        self.assertIn(DEMO_FACTS["after_p95"], ctx)
        self.assertIn("TracePack restored", d["systemMessage"])
        self.assertLessEqual(len(ctx), S.INJECT_MAX_CHARS)
        self.assertNotIn("stays untouched", ctx)       # the final reply is kept by Claude Code itself

    def test_18h_quiet_unless_compact(self):
        for src in ("startup", "resume", "clear"):
            self.assertEqual(hooks.session_start({"source": src, "transcript_path": self.demo, "session_id": "s"}), "")

    def test_18i_huge_output_stays_under_the_char_cap(self):
        rows = _precompact_rows()
        rows[-4]["message"]["content"][0]["content"] = "x" * 200000 + "\nTAIL 530ms"
        pre = _write(os.path.join(self.dir, "huge.jsonl"), rows)
        out = hooks.session_start({"source": "compact", "transcript_path": pre, "session_id": "s2"})
        if out:
            self.assertLessEqual(len(json.loads(out)["hookSpecificOutput"]["additionalContext"]), S.INJECT_MAX_CHARS)

    def test_18j_compact_instructions_steer_the_query(self):
        pre = _write(os.path.join(self.dir, "pre.jsonl"), _precompact_rows())
        hooks.pre_compact({"session_id": "s3", "transcript_path": pre, "trigger": "manual",
                           "custom_instructions": "keep the staging DB port"})
        out = hooks.session_start({"source": "compact", "transcript_path": pre, "session_id": "s3"})
        self.assertIn(DEMO_FACTS["db_port"], json.loads(out)["hookSpecificOutput"]["additionalContext"])

    def test_18k_fails_open(self):
        stdin, stdout = sys.stdin, sys.stdout
        try:
            sys.stdin = io.TextIOWrapper(io.BytesIO(b"{not json"), encoding="utf-8")
            sys.stdout = io.StringIO()
            self.assertEqual(hooks.main(["session-start"]), 0)
            self.assertEqual(sys.stdout.getvalue(), "")
        finally:
            sys.stdin, sys.stdout = stdin, stdout

    def test_18l_budget_zero_turns_injection_off(self):
        os.environ["TRACEPACK_INJECT_BUDGET"] = "0"
        pre = _write(os.path.join(self.dir, "pre.jsonl"), _precompact_rows())
        self.assertEqual(hooks.session_start({"source": "compact", "transcript_path": pre, "session_id": "s4"}), "")


class Contract18PluginMCP(_Env):
    def _call(self, msg):
        return mcp_server.handle(msg)

    def test_18m_protocol(self):
        r = self._call({"jsonrpc": "2.0", "id": 1, "method": "initialize",
                        "params": {"protocolVersion": "2025-06-18", "capabilities": {}}})
        self.assertEqual(r["result"]["protocolVersion"], "2025-06-18")
        self.assertIn("instructions", r["result"])
        self.assertIsNone(self._call({"jsonrpc": "2.0", "method": "notifications/initialized"}))
        tools = self._call({"jsonrpc": "2.0", "id": 2, "method": "tools/list"})["result"]["tools"]
        self.assertEqual([t["name"] for t in tools], ["recall", "expand", "status"])
        self.assertTrue(all(t["annotations"]["readOnlyHint"] for t in tools))
        self.assertEqual(self._call({"jsonrpc": "2.0", "id": 3, "method": "nope"})["error"]["code"], -32601)

    def test_18n_recall_tool(self):
        r = self._call({"jsonrpc": "2.0", "id": 4, "method": "tools/call",
                        "params": {"name": "recall", "arguments": {"query": "p95 latency", "session": self.demo}}})
        self.assertFalse(r["result"]["isError"])
        self.assertIn(DEMO_FACTS["after_p95"], r["result"]["content"][0]["text"])
        bad = self._call({"jsonrpc": "2.0", "id": 5, "method": "tools/call",
                          "params": {"name": "recall", "arguments": {"query": ""}}})
        self.assertTrue(bad["result"]["isError"])

    def test_18o_serve_over_stdio(self):
        lines = "\n".join(json.dumps(m) for m in [
            {"jsonrpc": "2.0", "id": 1, "method": "initialize", "params": {"protocolVersion": "2024-11-05"}},
            {"jsonrpc": "2.0", "method": "notifications/initialized"},
            {"jsonrpc": "2.0", "id": 2, "method": "tools/list"}]) + "\n"
        out = io.StringIO()
        mcp_server.serve(io.StringIO(lines), out)
        replies = [json.loads(l) for l in out.getvalue().splitlines()]
        self.assertEqual([x["id"] for x in replies], [1, 2])

    def test_18p_binds_to_the_calling_session(self):
        proj = os.path.join(self.dir, "work")
        os.makedirs(proj)
        pdir = os.path.join(S.projects_dir(), S.project_key(proj))
        os.makedirs(pdir)
        sid = "11111111-2222-4333-8444-555555555555"
        mine = write_demo_session(os.path.join(pdir, sid + ".jsonl"))
        other = write_demo_session(os.path.join(pdir, "99999999-2222-4333-8444-555555555555.jsonl"))
        os.utime(other, None)                                             # the newest file is NOT ours
        os.environ["CLAUDE_CODE_SESSION_ID"] = sid
        self.assertEqual(S.bound_transcript(proj), mine)
        os.environ["CLAUDE_PID"] = "4242"
        S.record_session("4242", "x", other, proj)                       # after /clear: the hook's record wins
        self.assertEqual(S.bound_transcript(proj), other)


class Contract18PluginRedact(unittest.TestCase):
    def test_18q_masks_credentials(self):
        txt = ("OPENAI_API_KEY=sk-proj-abcdefghijklmnopqrstuvwx1234\nAuthorization: Bearer abcdefghijklmnopqrstuvwxyz012345\n"
               "token ghp_abcdefghijklmnopqrstuvwxyz0123456789AB\npassword: hunter2hunter2\nAKIAABCDEFGHIJKLMNOP\n"
               "p95_latency=530ms")
        out, n = redact.redact(txt)
        for secret in ("sk-proj-abcdef", "abcdefghijklmnopqrstuvwxyz012345", "ghp_abcdef", "hunter2hunter2", "AKIAABCDEFGH"):
            self.assertNotIn(secret, out)
        self.assertIn("p95_latency=530ms", out)
        self.assertGreaterEqual(n, 5)


if __name__ == "__main__":
    unittest.main()
