"""tracepack.bench.lme_claude -- LongMemEval through real Claude Code, with and without TracePack.

The arms `claude-code` and `claude-code+tracepack` run the actual `claude` binary; nothing about its
compaction is re-implemented. For each question:

1. A Claude Code transcript is written session by session, in Claude Code's own row format, each row
   dated like the LongMemEval session it comes from.
2. Whenever the conversation passes the window (counted exactly as for the other arms), the harness
   runs `claude -p /compact --resume <session>`. Claude Code writes its summary; with the plugin loaded
   (`--plugin-dir`), TracePack's PreCompact hook adds its note-taking instructions to Claude Code's
   compaction prompt and its SessionStart hook restores the user's own words afterwards.
3. After the last session, `claude -p <question> --resume <session> --tools ""` answers from what
   Claude Code carried over, with no tools, like every other arm. The judge grades the answer.

Two ablation arms switch one half of the plugin off: `claude-code+notes` (TRACEPACK_INJECT_BUDGET=0) and
`claude-code+restore` (TRACEPACK_NOTES=0).

Claude Code needs an Anthropic Messages endpoint. By default the harness starts
`tracepack.bench.anthropic_bridge` on 127.0.0.1, which serves Claude Code from the same
TRACEPACK_LLM_* model as the other arms; --anthropic-base-url sends it to Anthropic (or any compatible
gateway, key from ANTHROPIC_API_KEY) instead. Each unit gets its own CLAUDE_CONFIG_DIR and HOME, so
your own settings, hooks and plugins are never loaded.

Claude Code tells the model today's date, the machine's. LongMemEval states each question's own date in
the question ("[Current date: 2023/05/30 (Tue) 23:40]"), so the model sees both; every Claude Code arm
sees the same mismatch. (Shifting Claude Code's clock back by years with an LD_PRELOAD shim made every
process spin on several CPU cores, so the harness leaves the clock alone.)
"""
from __future__ import annotations

import datetime as dt
import json
import os
import re
import shutil
import subprocess
import sys
import threading
import time
import uuid

from tracepack.bench import lme as L

PLUGIN_ARMS = {"claude-code": None, "claude-code+tracepack": {}, "claude-code+notes": {"TRACEPACK_INJECT_BUDGET": "0"},
               "claude-code+restore": {"TRACEPACK_NOTES": "0"}}
REPO = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
_DATE = re.compile(r"(\d{4})/(\d{2})/(\d{2})\s*(?:\([A-Za-z]+\))?\s*(\d{1,2}):(\d{2})")


def parse_date(s: str) -> dt.datetime:
    m = _DATE.search(s or "")
    if not m:
        raise ValueError("unparsed LongMemEval date %r" % s)
    y, mo, d, h, mi = (int(x) for x in m.groups())
    return dt.datetime(y, mo, d, h, mi, tzinfo=dt.timezone.utc)


def _iso(t: dt.datetime) -> str:
    return t.strftime("%Y-%m-%dT%H:%M:%S.000Z")


def project_key(path: str) -> str:
    return re.sub(r"[^A-Za-z0-9]", "-", path)


# ---------------------------------------------------------------------------- setup

class Registry:
    """What the bridge saw for each Claude Code session: every call, and the full request of the answer."""

    def __init__(self):
        self.lock = threading.Lock()
        self.calls = {}

    def on_request(self, session, purpose, msgs, content, usage):
        with self.lock:
            rec = self.calls.setdefault(session, [])
            entry = {"purpose": purpose, "prompt_tokens": usage.get("prompt_tokens"),
                     "completion_tokens": usage.get("completion_tokens"), "finish": usage.get("finish"),
                     "reply": content}
            if purpose == "compact":                # the compaction prompt is the last user message
                prompt = next((m.get("content") or "" for m in reversed(msgs) if m["role"] == "user"), "")
                entry["request_tail"] = prompt[-4000:]
            if purpose == "answer":                 # all but Claude Code's own system prompt (the first message)
                lead = 1 if msgs and msgs[0]["role"] == "system" else 0
                entry["system_chars"] = len(msgs[0]["content"]) if lead else 0
                entry["messages"] = msgs[lead:]
            rec.append(entry)

    def pop(self, session):
        with self.lock:
            return self.calls.pop(session, [])


def prepare(args, llm, out) -> dict:
    """Check the binary, start the bridge, return the options every unit needs."""
    claude = shutil.which(args.claude) or args.claude
    try:
        ver = subprocess.run([claude, "--version"], capture_output=True, text=True, timeout=60).stdout.split()[0]
    except (OSError, IndexError, subprocess.SubprocessError):
        raise SystemExit("Claude Code not found (%s); install it or pass --claude" % args.claude)
    plugin = os.path.abspath(args.plugin_dir or REPO)
    if not os.path.isfile(os.path.join(plugin, ".claude-plugin", "plugin.json")):
        raise SystemExit("not a TracePack plugin folder: %s" % plugin)
    reg = Registry()
    opts = {"claude": claude, "version": ver, "plugin": plugin, "registry": reg,
            "path": os.path.dirname(os.path.abspath(sys.executable)) + os.pathsep + os.environ.get("PATH", "")}
    if args.anthropic_base_url:
        key = os.environ.get("ANTHROPIC_API_KEY", "")
        if not key:
            raise SystemExit("--anthropic-base-url needs ANTHROPIC_API_KEY")
        opts.update(base_url=args.anthropic_base_url, api_key=key, model=os.environ.get("TRACEPACK_CLAUDE_MODEL", ""))
    else:
        from tracepack.bench.anthropic_bridge import Bridge
        if args.dry_run:
            raise SystemExit("the claude-code arms call a real model; run them without --dry-run")
        br = Bridge(llm, log_path=os.path.join(out, "bridge.jsonl"), on_request=reg.on_request)
        port = br.start(0)
        opts.update(base_url="http://127.0.0.1:%d" % port, api_key="tracepack-bridge", model=llm.model, stop=br.stop)
    return opts


# ---------------------------------------------------------------------------- the transcript

class Transcript:
    def __init__(self, path, sid, cwd, version):
        self.path, self.sid, self.cwd, self.version = path, sid, cwd, version
        os.makedirs(os.path.dirname(path), exist_ok=True)

    def row(self, kind, text, parent, when):
        u = str(uuid.uuid4())
        r = {"parentUuid": parent, "isSidechain": False, "userType": "external", "cwd": self.cwd, "sessionId": self.sid,
             "version": self.version, "gitBranch": "", "type": kind, "uuid": u, "timestamp": _iso(when)}
        if kind == "user":
            r["message"] = {"role": "user", "content": text}
        else:
            r["message"] = {"id": "msg_" + u.replace("-", "")[:24], "type": "message", "role": "assistant",
                            "model": "replay", "content": [{"type": "text", "text": text}], "stop_reason": "end_turn",
                            "stop_sequence": None, "usage": {"input_tokens": 0, "output_tokens": 0}}
        return r

    def append_session(self, msgs, date: dt.datetime, parent):
        """A LongMemEval session as Claude Code rows; returns the new leaf."""
        with open(self.path, "a", encoding="utf-8") as fh:
            for i, m in enumerate(msgs):
                r = self.row("user" if m["role"] == "user" else "assistant", m["content"], parent,
                             date + dt.timedelta(seconds=20 * i))
                fh.write(json.dumps(r, ensure_ascii=False) + "\n")
                parent = r["uuid"]
        return parent

    def rows(self):
        out = []
        with open(self.path, encoding="utf-8") as fh:
            for line in fh:
                try:
                    out.append(json.loads(line))
                except ValueError:
                    pass
        return out

    def leaf(self):
        last = None
        for r in self.rows():
            if r.get("uuid") and not r.get("isSidechain") and r.get("type") in ("user", "assistant", "system", "attachment"):
                last = r["uuid"]
        return last


def _text(row) -> str:
    c = (row.get("message") or {}).get("content")
    if isinstance(c, str):
        return c
    return "\n".join((b.get("text") or "") for b in (c or []) if isinstance(b, dict) and b.get("type") == "text")


def after_compaction(rows) -> dict:
    """What the last /compact left in context: summary message, preserved messages, hook context, command rows."""
    idx = max((i for i, r in enumerate(rows) if r.get("type") == "system" and r.get("subtype") == "compact_boundary"),
              default=None)
    if idx is None:
        return {}
    meta = rows[idx].get("compactMetadata") or {}
    keep = set((meta.get("preservedMessages") or {}).get("uuids") or [])
    by_uuid = {r.get("uuid"): r for r in rows if r.get("uuid")}
    out = {"summary": "", "preserved": [by_uuid[u] for u in keep if u in by_uuid], "hook_context": [], "commands": [],
           "pre_tokens": meta.get("preTokens"), "post_tokens": meta.get("postTokens")}
    for r in rows[idx + 1:]:
        if r.get("isCompactSummary"):
            out["summary"] = _text(r)
        elif r.get("type") == "attachment":
            a = r.get("attachment") or {}
            if a.get("type") == "hook_additional_context":
                out["hook_context"] += [x for x in (a.get("content") or []) if isinstance(x, str)]
        elif r.get("type") == "user":
            t = _text(r)
            if t.lstrip().startswith(("<local-command-", "<command-name>")):
                out["commands"].append(t)
    return out


# ---------------------------------------------------------------------------- one unit

def _die_with_parent():
    """Linux: a Claude Code process must not outlive the harness that started it."""
    try:
        import ctypes
        import signal
        ctypes.CDLL("libc.so.6", use_errno=True).prctl(1, signal.SIGTERM)    # PR_SET_PDEATHSIG
    except Exception:                                        # noqa: BLE001 - best effort
        pass


def _run_claude(opts, arm, udir, proj, sid, prompt, extra=()):
    env = {"PATH": opts["path"], "HOME": os.path.join(udir, "home"), "CLAUDE_CONFIG_DIR": os.path.join(udir, "config"),
           "ANTHROPIC_BASE_URL": opts["base_url"], "ANTHROPIC_API_KEY": opts["api_key"], "TZ": "UTC",
           "LANG": "C.UTF-8", "DISABLE_TELEMETRY": "1", "DISABLE_ERROR_REPORTING": "1", "DISABLE_AUTOUPDATER": "1",
           "CLAUDE_CODE_DISABLE_NONESSENTIAL_TRAFFIC": "1", "TRACEPACK_DATA": os.path.join(udir, "tracepack")}
    env.update(PLUGIN_ARMS[arm] or {})
    cmd = [opts["claude"], "-p", prompt, "--output-format", "json", "--resume", sid, "--strict-mcp-config"]
    if opts.get("model"):
        cmd += ["--model", opts["model"]]
    if PLUGIN_ARMS[arm] is not None:
        cmd += ["--plugin-dir", opts["plugin"]]
    cmd += list(extra)
    t0 = time.time()
    p = subprocess.run(cmd, cwd=proj, env=env, capture_output=True, text=True, timeout=3600,
                       preexec_fn=_die_with_parent if sys.platform.startswith("linux") else None)
    try:
        res = json.loads(p.stdout)
    except ValueError:
        res = {"is_error": True, "result": "", "raw": p.stdout[-800:]}
    res["_rc"], res["_seconds"], res["_stderr"] = p.returncode, round(time.time() - t0, 1), p.stderr[-800:]
    return res


def run_unit(it: dict, arm: str, window: int, out: str, opts: dict):
    """-> (record, final context as the model saw it)."""
    qid = it["question_id"]
    udir = os.path.join(out, "cc", arm.replace("+", "_"), qid)
    shutil.rmtree(udir, ignore_errors=True)                 # a unit always starts from scratch
    proj = os.path.join(udir, "p")
    for d in (proj, os.path.join(udir, "home"), os.path.join(udir, "config")):
        os.makedirs(d, exist_ok=True)
    proj = os.path.realpath(proj)
    sid = str(uuid.uuid4())
    tr = Transcript(os.path.join(udir, "config", "projects", project_key(proj), sid + ".jsonl"), sid, proj,
                    opts["version"])
    rec = {"qid": qid, "type": it["question_type"], "arm": arm, "window": window, "session_id": sid,
           "claude_version": opts["version"], "compactions": [], "errors": []}
    sysm = {"role": "system", "content": L.SYSTEM}
    base, seg, parent = [], [], None
    try:
        S = L.sessions_msgs(it)
        for date_s, msgs in zip(it["haystack_dates"], S):
            when = parse_date(date_s)
            parent = tr.append_session(msgs, when, parent)
            seg += msgs
            e = L.est(L.strip([sysm] + base + seg))
            if e <= window:
                continue
            res, kept = None, {}
            for attempt in range(2):
                n0 = _boundaries(tr.rows())
                res = _run_claude(opts, arm, udir, proj, sid, "/compact")
                rows = tr.rows()
                kept = after_compaction(rows) if _boundaries(rows) > n0 else {}
                if kept.get("summary"):
                    break
                rec["errors"].append("compaction attempt %d wrote no summary: %s" % (
                    attempt + 1, json.dumps({k: res.get(k) for k in ("result", "_rc", "_stderr")})[:400]))
            if not kept.get("summary"):
                raise RuntimeError("compaction failed twice after session %s" % date_s)
            base = ([{"role": "user", "content": kept["summary"]}]
                    + [{"role": "assistant" if r.get("type") == "assistant" else "user", "content": _text(r)} for r in kept["preserved"]]
                    + [{"role": "user", "content": c} for c in kept["hook_context"]]
                    + [{"role": "user", "content": c} for c in kept["commands"]])
            seg = []
            parent = tr.leaf()
            rec["compactions"].append({
                "after_session": date_s, "view_tokens_before": e, "seconds": res.get("_seconds"),
                "summary_tokens": L.tok(kept["summary"]), "restore_tokens": sum(L.tok(c) for c in kept["hook_context"]),
                "view_tokens_after": L.est(L.strip([sysm] + base)), "cc_pre_tokens": kept.get("pre_tokens"),
                "cc_post_tokens": kept.get("post_tokens"), "summary": kept["summary"],
                "restore": "\n\n".join(kept["hook_context"]),
                "precompact_output": [c for c in kept["commands"] if "PreCompact" in c]})
        q = L.question_msg(it)["content"]
        res = _run_claude(opts, arm, udir, proj, sid, q, extra=("--tools", ""))
        if res.get("is_error") or res.get("_rc"):
            raise RuntimeError("answer failed: %s" % json.dumps({k: res.get(k) for k in ("result", "_rc", "_stderr")})[:600])
        rec["answer"] = res.get("result") or ""
        rec["answer_seconds"] = res.get("_seconds")
    except Exception as e:                                   # noqa: BLE001 - recorded as an infrastructure failure
        rec["infra"] = repr(e)[:600]
    calls = opts["registry"].pop(sid)
    rec["calls"] = [{k: v for k, v in c.items() if k not in ("messages", "reply")} for c in calls]
    final = next((c for c in reversed(calls) if c["purpose"] == "answer"), None)
    ctx = final["messages"] if final else None
    if final:
        rec["answer_system_chars"] = final.get("system_chars")
    for c, comp in zip([c for c in calls if c["purpose"] == "compact"], rec["compactions"]):
        comp["notes_requested"] = "TracePack notes" in (c.get("request_tail") or "")
    return rec, ctx


def _boundaries(rows) -> int:
    return sum(1 for r in rows if r.get("type") == "system" and r.get("subtype") == "compact_boundary")
