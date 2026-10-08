"""Contract 20: `tracepack bench lme` is the study's harness.

- every prompt is byte-identical to the one that produced the published numbers (sha256 below, taken from
  the study's files), and so are the judge's official templates;
- each arm compacts the way it declares (stub model, zero calls);
- the report's bootstrap is the study's: same resampling, same seed;
- the Claude Code bridge keeps text and roles, drops Anthropic-only fields, and serves both reply shapes;
- the Claude Code driver reads back exactly what a compaction left in context.
Needs tiktoken for the arm tests (skipped without it). No network.
"""
from __future__ import annotations

import hashlib
import http.client
import json
import os
import sys
import tempfile
import unittest

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(os.path.dirname(HERE))
if ROOT not in sys.path:
    sys.path.insert(0, ROOT)

from tracepack.bench import anthropic_bridge as B  # noqa: E402
from tracepack.bench import lme as L  # noqa: E402
from tracepack.bench import lme_claude as C  # noqa: E402
from tracepack.bench import lme_judge as J  # noqa: E402

try:
    import tiktoken  # noqa: F401
    HAVE_TIKTOKEN = True
except ImportError:                                  # pragma: no cover
    HAVE_TIKTOKEN = False

STUDY_SHA256 = {
    "SYSTEM": "44971d4075ed2b080d4773c027eac6d0cb55836ac59fc867ed2c9f049b126147",
    "PICK_REQUEST": "b247d647684ef230d05cdb75365ad398f48db16d42a32fe4a7a0653c86ca5195",
    "EXTRACT_HEAD": "0be24194de3ffc2e4994652357548cfdf6887caaa0381e5121b0fbc885edc41f",
    "S_PREFIX": "2a3dcc1c323f2aacc002b9ad9fee25703a4818b461e749e979821fca36f82b2e",
    "SM_REQUEST": "7f65563604f1a7f7dc5fcb136ed915ddac919c11f4e1661a8bb8d00880c0dee8",
    "KM_NOTE": "82cd15fe327bd01c1112215d7309d2f344382f96d1b99c4fc98cc5cb5883a3b9",
    "KM_CONDENSE": "c8415cf10ed7087ba1618f0250567691e07a3ae6e85840f6872dfd98685955c0",
    "CX_PROMPT": "bab68c28f14288fc320a002bf87e6e53c3bb88288eb047893a9e65c96ef9c35c",
    "CX_PREFIX": "e9b088e794a6bb9082ac053fcc760bd818d7e720ee4bcdc72c6e480de7b7cb0e",
}
JUDGE_SHA256 = {
    ("single-session-user", False): "66896bfd8f9aa8fd5fdda1f69185eff87a03d643d4419edce05c8a3d65a7eb7b",
    ("temporal-reasoning", False): "2c6e336f0319b7732d02b59525981e02068e67b68d4dfe7042df0d77223d2671",
    ("knowledge-update", False): "3d614586a71dc7c2abf9cf07497dd551ba547a9c48fcef8eaeb3bfdd3a33a455",
    ("single-session-preference", False): "f1cfd746eafcdac83799b83c4cbc9275d838a7c1ce4cf59d0e237415aa8f97b2",
    ("multi-session", True): "059b12ca3ff08641e8dc0a4f6f6b21432b4f106ccae09bd1fc6233f8435333b6",
}


def _sha(s):
    return hashlib.sha256(s.encode("utf-8")).hexdigest()


def _item(n_sessions=12, words=260):
    """A made-up LongMemEval-shaped question: sessions of user/assistant turns, a fact in session 2."""
    sessions, dates = [], []
    for s in range(n_sessions):
        turns = []
        for k in range(3):
            fact = " My cat Miso is 4 years old." if (s == 2 and k == 1) else ""
            turns.append({"role": "user", "content": ("Session %d turn %d. " % (s, k)) + "lorem " * words + fact})
            turns.append({"role": "assistant", "content": "Reply %d.%d " % (s, k) + "ipsum " * words})
        sessions.append(turns)
        dates.append("2023/05/%02d (Mon) 10:%02d" % (s + 1, s))
    return {"question_id": "t1", "question_type": "single-session-user", "question": "How old is my cat?",
            "answer": "4", "question_date": "2023/06/01 (Thu) 09:00", "haystack_dates": dates,
            "haystack_sessions": sessions, "haystack_session_ids": ["s%d" % i for i in range(n_sessions)],
            "answer_session_ids": ["s2"]}


class Contract20BenchLme(unittest.TestCase):
    def test_20a_prompts_are_the_studys_byte_for_byte(self):
        for name, digest in STUDY_SHA256.items():
            self.assertEqual(_sha(getattr(L, name)), digest, name)
        self.assertEqual((L.K_SUMMARY_FRAC, L.K_EXTRACT_FRAC, L.PICK_SHOW_TOK, L.EXTRACT_ITEM_TOK, L.SMB_FRAC),
                         (0.15, 0.20, 400, 800, 0.35))

    def test_20b_judge_templates_are_longmemevals(self):
        for (task, abst), digest in JUDGE_SHA256.items():
            self.assertEqual(_sha(J.get_anscheck_prompt(task, "Q?", "A.", "R.", abstention=abst)), digest)

    def test_20c_judge_reads_the_first_yes_or_no(self):
        class E:
            def __init__(self, replies):
                self.replies = list(replies)

            def chat(self, *a, **k):
                return self.replies.pop(0), None, {}
        it = {"question_id": "q", "question_type": "multi-session", "question": "Q", "answer": "A"}
        self.assertTrue(J.judge(E(["Yes."]), it, "r")["ok"])
        self.assertFalse(J.judge(E(["no, it is not"]), it, "r")["ok"])
        self.assertTrue(J.judge(E(["hmm", "...", "yes"]), it, "r")["ok"])
        self.assertTrue(J.judge(E(["?", "?", "?"]), it, "r").get("empty"))

    @unittest.skipUnless(HAVE_TIKTOKEN, "needs tiktoken")
    def test_20d_arms_compact_as_declared(self):
        it = _item()
        window = 3000
        runs = {}
        for arm in ("tracepack", "codex", "notes", "full"):
            u = L.Unit(it, arm, window, L.StubEndpoint())
            rec = u.run()
            self.assertNotIn("infra", rec)
            runs[arm] = (u, rec)
        self.assertEqual(runs["full"][1]["resets"], [])
        for arm in ("tracepack", "codex", "notes"):
            self.assertGreater(len(runs[arm][1]["resets"]), 1, arm)
        raw = [t["content"] for s in it["haystack_sessions"] for t in s if t["role"] == "user"]
        u, rec = runs["tracepack"]
        for r in rec["resets"]:      # a cut summary ends in " […]", which can add a token or two
            self.assertLessEqual(r["sb"]["tokens_final"], int(L.K_SUMMARY_FRAC * window) + 3)
            self.assertLessEqual(r["extract_tokens"], int(L.K_EXTRACT_FRAC * window))
            for x in r["extract_texts"]:
                self.assertTrue(any(x.replace(" […]", "") in m for m in raw))      # verbatim, maybe cut
        self.assertTrue(any(m["content"].startswith(L.EXTRACT_HEAD) for m in u.final_ctx))
        u, rec = runs["codex"]
        for r in rec["resets"]:
            self.assertLessEqual(r["kept_tokens_codex"], r["U"])
        self.assertTrue(u.final_ctx[-2]["content"].startswith(L.CX_PREFIX + "\n"))
        u, rec = runs["notes"]
        for r in rec["resets"]:
            self.assertLessEqual(r["sb"]["tokens_final"], int(L.SMB_FRAC * window) + 3)
        self.assertFalse(any(m["content"].startswith(L.EXTRACT_HEAD) for m in u.final_ctx))

    def test_20e_bootstrap_is_the_studys(self):
        res = {}
        qids = ["q%03d" % i for i in range(60)]
        for i, q in enumerate(qids):
            res[(q, "a")] = i % 5 != 0          # 80%
            res[(q, "b")] = i % 2 == 0          # 50%
        P = L.Paired(res, qids, n_boot=500)
        d, lo, hi, p = P.diff("a", "b")
        self.assertAlmostEqual(d, 0.30)
        self.assertLess(lo, d)
        self.assertGreater(hi, d)
        self.assertEqual((d, lo, hi, p), L.Paired(res, qids, n_boot=500).diff("a", "b"))   # deterministic

    def test_20f_bridge_converts_requests(self):
        body = {"model": "x", "max_tokens": 32000, "stream": True, "thinking": {"type": "enabled"}, "tools": [{}],
                "system": [{"type": "text", "text": "x-anthropic-billing-header: cc_version=1"},
                           {"type": "text", "text": "You are Claude Code."}],
                "messages": [{"role": "user", "content": "hi"},
                             {"role": "assistant", "content": [{"type": "text", "text": "hello"},
                                                               {"type": "tool_use", "name": "Read", "input": {"p": 1}}]},
                             {"role": "user", "content": [{"type": "tool_result", "content": "data"},
                                                          {"type": "text", "text": B.COMPACT_MARK + " ..."}]},
                             {"role": "system", "content": [{"type": "text", "text": "Today's date is 2023-06-01."}]}]}
        msgs = B.to_openai(body)
        self.assertEqual(msgs[0], {"role": "system", "content": "You are Claude Code."})
        self.assertEqual([m["role"] for m in msgs], ["system", "user", "assistant", "user", "system"])
        self.assertIn("[tool call Read", msgs[2]["content"])
        self.assertIn("[tool result]\ndata", msgs[3]["content"])
        self.assertEqual(B.purpose_of(body), "compact")
        self.assertEqual(B.purpose_of({"messages": [{"role": "user", "content": "q"}]}), "answer")

    def test_20g_bridge_serves_json_and_streams(self):
        class E:
            model = "m"

            def describe(self):
                return "m"

            def chat(self, messages, **kw):
                return "SUMMARY " + messages[-1]["content"][:5], None, {"prompt_tokens": 11, "completion_tokens": 3}
        seen = []
        br = B.Bridge(E(), on_request=lambda *a: seen.append(a[1]))
        port = br.start(0)
        try:
            for stream in (False, True):
                conn = http.client.HTTPConnection("127.0.0.1", port, timeout=30)
                conn.request("POST", "/v1/messages?beta=true", json.dumps(
                    {"stream": stream, "max_tokens": 10, "messages": [{"role": "user", "content": "hello there"}]}),
                    {"Content-Type": "application/json"})
                data = conn.getresponse().read().decode()
                conn.close()
                if stream:
                    self.assertIn("event: message_start", data)
                    self.assertIn('"text": "SUMMARY hello"', data)
                    self.assertIn("event: message_stop", data)
                else:
                    self.assertEqual(json.loads(data)["content"][0]["text"], "SUMMARY hello")
        finally:
            br.stop()
        self.assertEqual(seen, ["answer", "answer"])

    def test_20h_driver_reads_back_what_a_compaction_left(self):
        with tempfile.TemporaryDirectory() as d:
            tr = C.Transcript(os.path.join(d, "p", "s.jsonl"), "s", "/w", "2.1.294")
            leaf = tr.append_session([{"role": "user", "content": "My cat Miso is 4."},
                                      {"role": "assistant", "content": "Noted."}],
                                     C.parse_date("2023/05/02 (Tue) 10:00"), None)
            rows = tr.rows()
            self.assertEqual(rows[0]["timestamp"], "2023-05-02T10:00:00.000Z")
            self.assertEqual(rows[1]["parentUuid"], rows[0]["uuid"])
            extra = [
                {"type": "system", "subtype": "compact_boundary", "uuid": "b", "parentUuid": None,
                 "compactMetadata": {"preservedMessages": {"uuids": [leaf]}, "preTokens": 9, "postTokens": 3}},
                {"type": "user", "uuid": "s", "isCompactSummary": True, "message": {"role": "user", "content": "SUM"}},
                {"type": "attachment", "uuid": "h", "attachment": {"type": "hook_additional_context", "content": ["RESTORE"]}},
                {"type": "user", "uuid": "c", "message": {"role": "user", "content": "<local-command-stdout>ok</local-command-stdout>"}},
            ]
            with open(tr.path, "a") as fh:
                for r in extra:
                    fh.write(json.dumps(r) + "\n")
            kept = C.after_compaction(tr.rows())
            self.assertEqual(kept["summary"], "SUM")
            self.assertEqual(kept["hook_context"], ["RESTORE"])
            self.assertEqual([r["uuid"] for r in kept["preserved"]], [leaf])
            self.assertEqual(len(kept["commands"]), 1)
            self.assertEqual(C._boundaries(tr.rows()), 1)
            self.assertEqual(C.parse_date("2023/06/10 (Sat) 23:47").isoformat(), "2023-06-10T23:47:00+00:00")


if __name__ == "__main__":
    unittest.main()
