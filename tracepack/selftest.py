"""tracepack.selftest -- a small synthetic Claude Code session with one compaction.

Used by `tracepack doctor` and the tests. Everything in it is invented: a benchmark that is profiled,
a one-line fix, the benchmark re-run, a protected migration id the user mentions, then `/compact`.
The compaction summary keeps the plan but not the numbers, which is exactly what TracePack is for.
"""
from __future__ import annotations

import json

DEMO_FACTS = {"before_p95": "812ms", "after_p95": "530ms", "migration": "2026_09_30_add_ledger_index",
              "db_port": "6543"}


def demo_rows(session_id: str = "demo-session", cwd: str = "/work/demo") -> list:
    rows, state = [], {"n": 0, "parent": None}

    def add(row, minute):
        state["n"] += 1
        uid = "00000000-0000-4000-8000-%012d" % state["n"]
        row.update({"uuid": uid, "parentUuid": state["parent"], "timestamp": "2026-10-08T09:%02d:00.000Z" % minute,
                    "sessionId": session_id, "cwd": cwd, "isSidechain": False})
        rows.append(row)
        state["parent"] = uid

    def user(text, m):
        add({"type": "user", "message": {"role": "user", "content": text}}, m)

    def call(tid, name, inp, m, text=None):
        content = ([{"type": "text", "text": text}] if text else []) + [
            {"type": "tool_use", "id": tid, "name": name, "input": inp}]
        add({"type": "assistant", "message": {"role": "assistant", "content": content}}, m)

    def result(tid, out, m):
        add({"type": "user", "message": {"role": "user", "content": [
            {"type": "tool_result", "tool_use_id": tid, "content": out}]}}, m)

    def say(text, m):
        add({"type": "assistant", "message": {"role": "assistant", "content": [{"type": "text", "text": text}]}}, m)

    user("The nightly benchmark is slow. Profile it and keep the numbers; the staging DB is at port 6543.", 0)
    call("t1", "Bash", {"command": "python bench.py --rows 200000"}, 1, "Running the benchmark first.")
    result("t1", "rows=200000 elapsed=41.7s p95_latency=812ms\nhot path: parse_rows 63% of time", 2)
    call("t2", "Read", {"file_path": cwd + "/bench/parse.py"}, 3)
    result("t2", "def parse_rows(buf):\n    out = []\n    for line in buf.split('\\n'):\n"
                 "        out.append(json.loads(line))\n    return out", 4)
    say("parse_rows decodes one line at a time; splitting with splitlines() should help.", 5)
    call("t3", "Edit", {"file_path": cwd + "/bench/parse.py", "old_string": "for line in buf.split('\\n'):",
                        "new_string": "for line in buf.splitlines():"}, 6)
    result("t3", "The file %s/bench/parse.py has been updated." % cwd, 7)
    call("t4", "Bash", {"command": "python bench.py --rows 200000"}, 8)
    result("t4", "rows=200000 elapsed=29.3s p95_latency=530ms\nhot path: parse_rows 41% of time", 9)
    user("Good. Also the migration id we must not touch is 2026_09_30_add_ledger_index.", 10)
    say("Noted: migration 2026_09_30_add_ledger_index stays untouched.", 11)
    add({"type": "system", "subtype": "compact_boundary", "content": "Conversation compacted",
         "compactMetadata": {"trigger": "manual", "preTokens": 51234, "preservedMessages": {"uuids": []}}}, 12)
    add({"type": "user", "isCompactSummary": True, "message": {"role": "user", "content":
        "This session is being continued from a previous conversation.\n\nSummary:\n"
        "1. Primary Request and Intent: speed up the nightly benchmark.\n"
        "2. Current Work: optimised parse_rows in bench/parse.py; the benchmark got faster.\n"
        "3. Pending Tasks: report the before/after p95 latency; do not touch the protected migration.\n"
        "4. Optional Next Step: write the report."}}, 12)
    return rows


def write_demo_session(path: str, **kw) -> str:
    with open(path, "w", encoding="utf-8") as fh:
        for r in demo_rows(**kw):
            fh.write(json.dumps(r) + "\n")
    return path
