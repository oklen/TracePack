"""Contract 19: note-taking compaction and the user's own words.

The PreCompact hook's output is what Claude Code appends to its compaction prompt, so the hook must print
the note-taking instructions (and nothing when they are switched off). After a compaction the restore must
bring back the user's own sentences verbatim, dated, picked by content, inside the token budget and the
character cap, and leave room for tool records in a coding session.
"""
from __future__ import annotations

import json
import os
import re
import sys
import tempfile
import time
import unittest

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(os.path.dirname(HERE))
if ROOT not in sys.path:
    sys.path.insert(0, ROOT)

from tracepack import hooks, session as S, userwords as UW  # noqa: E402
from tracepack.selftest import demo_rows  # noqa: E402

CHAT = [  # (role, text): a made-up personal-assistant conversation spread over days
    ("user", "Hi! I just adopted a beagle named Biscuit; she is 3 years old. What food is good for beagles?"),
    ("assistant", "Congratulations on Biscuit! Look for a food made for small active breeds."),
    ("user", "Thanks, that's helpful!"),
    ("assistant", "You're welcome."),
    ("user", "Can you explain how compound interest works?"),
    ("assistant", "Interest is added to the principal, so later interest is earned on earlier interest."),
    ("user", "By the way, I moved to Lisbon on March 3rd for a job at a fintech called Payvia. I prefer "
             "trains over flights when I travel."),
    ("assistant", "Lisbon has good train links to Porto and Madrid."),
    ("user", "My sister Ana visits in July and my budget for her trip is $1,200."),
    ("assistant", "That's enough for a few nice day trips."),
    ("user", "Write a haiku about the sea."),
    ("assistant", "Grey waves fold on stone / ..."),
]


def _chat_rows(n_repeat=1, start_day=1):
    rows, parent, k = [], None, 0
    for rep in range(n_repeat):
        for role, text in CHAT:
            k += 1
            uid = "10000000-0000-4000-8000-%012d" % k
            row = {"uuid": uid, "parentUuid": parent, "isSidechain": False, "sessionId": "chat", "cwd": "/w",
                   "timestamp": "2023-05-%02dT10:%02d:00.000Z" % (start_day + rep, k % 60), "type": role}
            row["message"] = ({"role": "user", "content": text if rep == 0 else "%s (%d)" % (text, rep)}
                              if role == "user" else {"role": "assistant", "content": [{"type": "text", "text": text}]})
            rows.append(row)
            parent = uid
    return rows


def _user_section(text):
    """The user's-words part of a restore (before the records part, if any)."""
    return text.split("\n\nTracePack: ", 1)[0]


def _write(path, rows):
    with open(path, "w", encoding="utf-8") as fh:
        for r in rows:
            fh.write(json.dumps(r) + "\n")
    return path


class _Env(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.dir = self.tmp.name
        self._env = dict(os.environ)
        os.environ["TRACEPACK_DATA"] = os.path.join(self.dir, "data")
        for k in ("TRACEPACK_NOTES", "TRACEPACK_USER_WORDS", "TRACEPACK_INJECT_BUDGET", "TRACEPACK_REDACT"):
            os.environ.pop(k, None)
        os.environ["TZ"] = "UTC"                                  # labels show local time; pin it
        time.tzset()
        S._CACHE.clear()

    def tearDown(self):
        os.environ.clear()
        os.environ.update(self._env)
        time.tzset()
        S._CACHE.clear()
        self.tmp.cleanup()


class Contract19CompactionNotes(_Env):
    def test_19a_precompact_prints_the_note_taking_instructions(self):
        path = _write(os.path.join(self.dir, "t.jsonl"), _chat_rows())
        out = hooks.pre_compact({"transcript_path": path, "session_id": "s1", "trigger": "auto"})
        self.assertTrue(out.startswith("TracePack notes:"))
        self.assertIn("Copy values exactly as stated", out)
        self.assertFalse(out.lstrip().startswith("{"))          # plain text: Claude Code must not parse it as JSON

    def test_19b_notes_can_be_switched_off(self):
        os.environ["TRACEPACK_NOTES"] = "0"
        path = _write(os.path.join(self.dir, "t.jsonl"), _chat_rows())
        self.assertEqual(hooks.pre_compact({"transcript_path": path, "session_id": "s2"}), "")

    def test_19c_restore_brings_back_the_users_facts_verbatim_and_dated(self):
        path = _write(os.path.join(self.dir, "t.jsonl"), _chat_rows())
        res = S.post_compact_packet(path, budget=2000)
        self.assertTrue(res["found"])
        text = res["text"]
        self.assertTrue(text.startswith(UW.HEAD))
        for fact in ("I just adopted a beagle named Biscuit; she is 3 years old.",
                     "I moved to Lisbon on March 3rd for a job at a fintech called Payvia.",
                     "My sister Ana visits in July and my budget for her trip is $1,200."):
            self.assertIn(fact, text)
        for chit in ("Thanks, that's helpful!", "Write a haiku about the sea.", "How compound interest works"):
            self.assertNotIn(chit, _user_section(text))
        self.assertRegex(text, r"\[u\d+\] user said · 2023-05-01 \d\d:\d\d · L\d+")
        self.assertGreater(res["user_sentences"], 0)

    def test_19d_every_restored_sentence_is_a_verbatim_substring(self):
        path = _write(os.path.join(self.dir, "t.jsonl"), _chat_rows(n_repeat=30))
        res = S.post_compact_packet(path, budget=1200)
        said = [r["message"]["content"] for r in _chat_rows(n_repeat=30) if r["type"] == "user"]
        body = re.sub(r"^\[u\d+\] user said .*$", "", _user_section(res["text"]).split("\n\n", 1)[1], flags=re.M)
        for piece in re.split(r"\s…\s|\n+", body):
            piece = piece.strip(" …")
            if piece:
                self.assertTrue(any(piece in s for s in said), piece)

    def test_19e_budget_and_char_cap_hold(self):
        path = _write(os.path.join(self.dir, "t.jsonl"), _chat_rows(n_repeat=80))
        for budget, cap in ((400, 1600), (2000, 8000), (5000, 8000)):
            res = S.post_compact_packet(path, budget=budget, max_chars=cap)
            self.assertLessEqual(res["tokens"], budget)
            self.assertLessEqual(len(res["text"]), cap)

    def test_19f_coding_session_keeps_room_for_tool_records(self):
        rows = [r for r in demo_rows() if r.get("subtype") != "compact_boundary" and not r.get("isCompactSummary")]
        path = _write(os.path.join(self.dir, "t.jsonl"), rows)
        res = S.post_compact_packet(path, budget=1500)
        self.assertIn("6543", res["text"])                      # the user's port, from the user's words
        self.assertIn("2026_09_30_add_ledger_index", res["text"])
        self.assertIn("p95_latency=530ms", res["text"])         # the latest tool output, from the records
        self.assertGreater(len(res["entries"]), 0)

    def test_19g_user_words_can_be_switched_off(self):
        os.environ["TRACEPACK_USER_WORDS"] = "0"
        path = _write(os.path.join(self.dir, "t.jsonl"), _chat_rows())
        res = S.post_compact_packet(path, budget=2000)
        self.assertNotIn(UW.HEAD, res.get("text") or "")

    def test_19h_scorer_prefers_facts(self):
        self.assertEqual(UW.fact_score("Thanks, that's helpful!"), 0.0)
        self.assertLess(UW.fact_score("Can you explain how compound interest works?"), UW.MIN_SCORE)
        self.assertLess(UW.fact_score("See https://example.com/a/b/c/1234567"), UW.MIN_SCORE)
        self.assertGreater(UW.fact_score("I moved to Lisbon on March 3rd for a job at Payvia."), UW.MIN_SCORE)
        self.assertGreater(UW.fact_score("Never touch migration 2026_09_30_add_ledger_index."), UW.MIN_SCORE)

    def test_19i_session_start_announces_sentences(self):
        path = _write(os.path.join(self.dir, "t.jsonl"), _chat_rows())
        out = json.loads(hooks.session_start({"source": "compact", "transcript_path": path, "session_id": "s3"},
                                             record=False))
        self.assertIn("of your sentences", out["systemMessage"])
        self.assertIn("I just adopted a beagle", out["hookSpecificOutput"]["additionalContext"])


if __name__ == "__main__":
    unittest.main()
