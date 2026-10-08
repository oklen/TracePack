"""tracepack bench lme -- LongMemEval with compaction: the benchmark behind the README's Codex comparison.

Each question's chat history (LongMemEval-S: about 50 sessions and 115k tokens) is streamed into a
context window session by session, oldest first. Whenever the context passes the window (32k tokens by
default), the arm compacts it. When the whole history is in, the question is asked once, with no tools,
and a judge model grades the answer with LongMemEval's official per-type prompts.

Arms:
  tracepack   note-taking memory summary (capped at 0.15 x window) + the user's own messages, picked by
              content and copied verbatim by the harness, append-only across compactions (0.2 x window)
  codex       Codex CLI's local compaction, re-implemented from its source: its hand-off summary prompt and
              prefix, plus the most recent user messages kept verbatim up to min(20,000, 0.3 x window)
              tokens (bytes / 4, as Codex counts)
  notes       the same note-taking summary with the whole memory budget (0.35 x window) and no verbatim part
  full        no compaction: the whole history (needs a model with a ~128k-token window)
  claude-code, claude-code+tracepack
              real Claude Code (`claude` on PATH or --claude), compacting with /compact, without and with the
              TracePack plugin; see tracepack/bench/lme_claude.py

Usage:
  tracepack bench lme download                            # LongMemEval-S (cleaned) into ~/.cache/tracepack
  tracepack bench lme run --arms tracepack,codex --out runs/lme --workers 32
  tracepack bench lme report --out runs/lme
  tracepack bench lme run --dry-run --limit 3 --out /tmp/lme-dry     # offline, stub model, zero calls

The model is any OpenAI-compatible endpoint, set with TRACEPACK_LLM_BASE_URL / _API_KEY / _MODEL (judge:
TRACEPACK_JUDGE_*). Token counts use tiktoken's cl100k_base (`pip install tiktoken`). One question costs
roughly 4 compactions of a 32k context per compacting arm, about 150k input tokens.

The defaults reproduce the study in docs/RESEARCH.md: the same 480 questions, window, prompts,
temperatures, budgets and retry rules as the code that produced its numbers; tests/test_bench_lme.py
checks the prompts byte for byte.
"""
from __future__ import annotations

import argparse
import collections
import functools
import hashlib
import json
import os
import random
import re
import sys
import threading
import time
import traceback
import urllib.request
from concurrent.futures import ThreadPoolExecutor

from tracepack.bench import lme_judge
from tracepack.bench.llm import Endpoint, LLMError

DATA_URL = "https://huggingface.co/datasets/xiaowu0162/longmemeval-cleaned/resolve/main/longmemeval_s_cleaned.json"
DATA_SHA256 = "d6f21ea9d60a0d56f34a05b609c79c88a451d2ae03597821ea3d5a9678c3a442"
IDS_FILE = os.path.join(os.path.dirname(os.path.abspath(__file__)), "lme_s_ids.json")

# ---------------------------------------------------------------------------- prompts (verbatim from the study)

SYSTEM = ("You are a helpful personal assistant. You have ongoing chat sessions with the user over time; "
          "use what you learned in earlier sessions.")
PICK_REQUEST = ("The user's messages below come from the part of our conversation that is about to be removed from your context. "
                "Pick the messages that state lasting facts about the user that might matter later: personal details, numbers, counts, dates, "
                "names, events, plans, purchases, preferences, and anything they asked you to remember. Reply with a JSON list of the message "
                "numbers only, for example [3, 7, 12]. Do not rewrite or explain.\n\n")
EXTRACT_HEAD = ("Exact copies of messages the user sent in earlier sessions (verbatim, not a summary; kept because they state lasting facts). "
                "Treat them as the user's own words.\n\n")
S_PREFIX = ("You previously worked on this task in an earlier context window. This is a new context window, and the text provided here is a summary "
            "of the portion you completed before.\n\n")
SM_REQUEST = (
    "You are approaching the context window's length limit. The earlier part of your conversations with the user is about to be removed "
    "from your context and replaced by the memory notes you write now. You do not know what the user will ask later, so write notes that "
    "would let you answer any future question about the user and about what was discussed. Requirements:\n"
    "1. Record every personal fact the user shared (family, pets, work, places, possessions, health, hobbies, events, purchases), with exact "
    "numbers, counts, amounts, names and dates exactly as stated. Do not round, merge or paraphrase them.\n"
    "2. Record the user's preferences, opinions, plans and goals, and anything they asked you to remember.\n"
    "3. Record changes over time: when a fact was updated (moved, changed jobs, bought or sold something, changed a plan), keep both the old "
    "and the new value with the session date of each.\n"
    "4. Keep the date of the session in which each fact was mentioned, so that questions about order and elapsed time can be answered.\n"
    "5. Record what you (the assistant) said that the user may ask about later: specific recommendations, names, lists, numbers and answers.\n"
    "6. Keep all facts from any earlier memory notes in the conversation above, updated where needed.\n"
    "Write dense factual notes, one fact per line, grouped by topic. Do not include generic chit-chat.")
KM_NOTE = ("\n7. Keep the whole notes under %d words. If something must go, drop chit-chat and repetition before personal facts, exact values, "
           "dates, changes over time and what the assistant recommended.")
KM_CONDENSE = ("The following memory notes are too long for the new context window. Rewrite them in under %d words. Keep every personal fact, "
               "exact value, date, change over time and assistant recommendation; drop chit-chat and repetition.\n\n")
# Codex CLI's compaction (codex-rs/core: prompt.md and summary_prefix.md of the compact templates).
CX_PROMPT = ("You are performing a CONTEXT CHECKPOINT COMPACTION. Create a handoff summary for another LLM that will resume the task.\n\n"
             "Include:\n- Current progress and key decisions made\n- Important context, constraints, or user preferences\n"
             "- What remains to be done (clear next steps)\n- Any critical data, examples, or references needed to continue\n\n"
             "Be concise, structured, and focused on helping the next LLM seamlessly continue the work.")
CX_PREFIX = ("Another language model started to solve this problem and produced a summary of its thinking process. You also have access to the state "
             "of the tools that were used by that language model. Use this to build on the work that has already been done and avoid duplicating work. "
             "Here is the summary produced by the other language model, use the information in this summary to assist with your own analysis:")

K_SUMMARY_FRAC, K_EXTRACT_FRAC, PICK_SHOW_TOK, EXTRACT_ITEM_TOK = 0.15, 0.20, 400, 800
SMB_FRAC = 0.35                       # notes: the whole memory budget of `tracepack` (0.15 + 0.20)
CX_USER_TOKENS = 20000                # Codex keeps at most this many tokens of recent user messages
DEFAULT_WINDOW = 32768
SEED = 20261004

HARNESS_ARMS = ("tracepack", "codex", "notes", "full")
CLAUDE_ARMS = ("claude-code", "claude-code+tracepack", "claude-code+notes", "claude-code+restore")
ARM_TAGS = {"tracepack": "Km", "codex": "CX", "notes": "SMb", "full": "F"}   # the study's names for the same arms


# ---------------------------------------------------------------------------- tokens

@functools.lru_cache(maxsize=1)
def _enc():
    try:
        import tiktoken
    except ImportError:
        raise SystemExit("tracepack bench lme counts tokens with tiktoken: pip install tiktoken")
    return tiktoken.get_encoding("cl100k_base")


@functools.lru_cache(maxsize=400000)
def tok(s: str) -> int:
    return len(_enc().encode(s or "", disallowed_special=()))


def est(msgs) -> int:
    """API-side prompt tokens of a message list (calibrated on 21 full-history calls; largest error 409)."""
    return int(sum(tok(m["content"]) for m in msgs) + 2.82 * len(msgs) - 375)


def clip_tok(s: str, n: int) -> str:
    e = _enc().encode(s or "", disallowed_special=())
    return s if len(e) <= n else _enc().decode(e[:n]) + " […]"


def approx_codex(s: str) -> int:
    """Codex's approx_token_count: UTF-8 bytes / 4, rounded up."""
    return (len((s or "").encode("utf-8")) + 3) // 4


# ---------------------------------------------------------------------------- data

def default_data_path() -> str:
    base = os.environ.get("XDG_CACHE_HOME") or os.path.join(os.path.expanduser("~"), ".cache")
    return os.path.join(base, "tracepack", "longmemeval_s_cleaned.json")


def download(path: str = "") -> str:
    path = path or default_data_path()
    if os.path.isfile(path) and _sha256(path) == DATA_SHA256:
        return path
    os.makedirs(os.path.dirname(path), exist_ok=True)
    tmp = path + ".part"
    print("downloading LongMemEval-S (cleaned, 277 MB) from Hugging Face ...", file=sys.stderr)
    with urllib.request.urlopen(DATA_URL, timeout=600) as r, open(tmp, "wb") as fh:
        while True:
            chunk = r.read(1 << 20)
            if not chunk:
                break
            fh.write(chunk)
    got = _sha256(tmp)
    if got != DATA_SHA256:
        raise SystemExit("downloaded file has sha256 %s, expected %s" % (got, DATA_SHA256))
    os.replace(tmp, path)
    return path


def _sha256(path: str) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as fh:
        for chunk in iter(lambda: fh.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def load_items(path: str) -> dict:
    with open(path, encoding="utf-8") as fh:
        return {it["question_id"]: it for it in json.load(fh)}


def load_ids(spec: str = "") -> list:
    """`""`/"study": the study's 480 questions; "first200": its first batch; "all": every question in the data
    file (resolved later); or a JSON file with {"ids": [...]}."""
    if spec in ("", "study", "first200"):
        with open(IDS_FILE, encoding="utf-8") as fh:
            d = json.load(fh)
        return list(d["first_batch"] if spec == "first200" else d["ids"])
    if spec == "all":
        return ["*"]
    with open(spec, encoding="utf-8") as fh:
        return list(json.load(fh)["ids"])


def sessions_msgs(it: dict) -> list:
    """One list of messages per session; each session's first message carries its date, as streamed."""
    out = []
    for date, sess in zip(it["haystack_dates"], it["haystack_sessions"]):
        ms = []
        for i, t in enumerate(sess):
            c = t.get("content") or ""
            ms.append({"role": "user" if t.get("role") == "user" else "assistant",
                       "content": ("[Chat session on %s]\n%s" % (date, c)) if i == 0 else c, "_date": date, "_raw": c})
        out.append(ms)
    return out


def question_msg(it: dict) -> dict:
    return {"role": "user", "content": "[Current date: %s]\nBased on our past conversations, please answer: %s" % (
        it["question_date"], it["question"])}


def strip(msgs) -> list:
    return [{"role": m["role"], "content": m["content"]} for m in msgs]


# ---------------------------------------------------------------------------- a stub model (zero calls)

class StubEndpoint:
    """Deterministic stand-in for --dry-run and the tests: summaries and picks sized like the real ones."""
    model = "stub"

    def describe(self):
        return "stub model"

    def chat(self, messages, temperature=0.0, max_tokens=2048, model=""):
        last = messages[-1]["content"]
        if last.startswith(PICK_REQUEST[:40]):
            nums = [int(x) for x in re.findall(r"^\[u(\d+)\]", last, re.M)]
            out = json.dumps([x for x in nums if x % 4 == 1] or nums[:1])
        elif "too long for the new context window" in last:
            out = "## Summary\n" + "word " * 1500
        elif "Copy exact values verbatim" in last:
            out = "## Summary\n" + "word " * 3000
        elif "length limit" in last or "CONTEXT CHECKPOINT COMPACTION" in last:
            out = "## Summary\n" + "word " * 1650
        elif last.startswith("I will give you"):
            out = "no"
        else:
            out = "stub answer"
        return out, None, {"prompt_tokens": est(messages), "completion_tokens": 50}


# ---------------------------------------------------------------------------- one question under one arm

class Unit:
    """Streams one question's history through one compacting arm, then asks the question."""

    def __init__(self, item: dict, arm: str, window: int, llm):
        self.it, self.arm, self.thr, self.llm = item, arm, int(window), llm
        self.qid = item["question_id"]
        self.calls, self.resets, self.errors = [], [], []
        self.summary, self.extracts = None, []
        self.final_ctx = None

    # -- model calls ---------------------------------------------------------------------------
    def call(self, purpose, messages, attempt=1, **kw):
        sm = strip(messages)
        content, tcs, u = self.llm.chat(sm, **kw)
        rec = dict(purpose=purpose, attempt=attempt, est=est(sm), **u)
        self.calls.append(rec)
        return content, tcs, rec

    def accepted_text(self, purpose, messages, temps=(0.7, 0.7, 0.7, 0.7), max_tokens=8192, validate=None):
        """A summary or pick: a tool call, an empty reply or (with `validate`) an invalid one is re-asked,
        up to 4 times; None if all are rejected."""
        for k, T in enumerate(temps, 1):
            txt, tcs, rec = self.call(purpose, messages, attempt=k, temperature=T, max_tokens=max_tokens)
            if tcs:
                rec["rejected"] = "tool_calls"
                continue
            if not (txt or "").strip():
                rec["rejected"] = "empty"
                continue
            if validate is not None:
                v = validate(txt)
                if v is None:
                    rec["rejected"] = "invalid"
                    continue
                return v
            return txt
        return None

    # -- run -----------------------------------------------------------------------------------
    def run(self) -> dict:
        S = sessions_msgs(self.it)
        q = question_msg(self.it)
        rec = {"qid": self.qid, "type": self.it["question_type"], "arm": self.arm, "window": self.thr}
        try:
            if self.arm == "full":
                view = [{"role": "system", "content": SYSTEM}] + [m for s in S for m in s]
            else:
                view = self.stream(S)
            final = view + [q]
            self.final_ctx = strip(final)
            rec["est_final"] = est(self.final_ctx)
            ans, tcs, crec = self.call("answer", final, temperature=0.0, max_tokens=2048)
            rec.update(answer=ans, answer_prompt=crec.get("prompt_tokens"), answer_tool_calls=bool(tcs))
        except Exception as e:                  # noqa: BLE001 - an infrastructure failure, counted separately
            rec["infra"] = repr(e)[:400]
            self.errors.append(rec["infra"])
        rec.update(calls=self.calls, resets=self.resets, errors=self.errors)
        return rec

    def stream(self, S) -> list:
        sysm = {"role": "system", "content": SYSTEM}
        base, seg, n = [], [], 0
        compact = {"tracepack": self.c_tracepack, "codex": self.c_codex, "notes": self.c_notes}[self.arm]
        for s in S:
            seg += s
            n += len(s)
            view = [sysm] + base + seg
            e = est(strip(view))
            if e <= self.thr:
                continue
            r = compact(view, seg, sysm)
            if r is None:
                self.errors.append("summary all rejected")
                continue
            base, summ, kw = r
            seg = []
            self.record_reset(n, e, [sysm] + base, summ, **kw)
        return [sysm] + base + seg

    def record_reset(self, n_streamed, e_before, new_view, summ, **kw):
        r = {"event": "reset", "arm": self.arm, "reset_idx": len(self.resets) + 1, "n_msgs": n_streamed,
             "view_tokens_before": e_before, "fixed_tokens": tok(SYSTEM), "cap": self.thr, "summary_chars": len(summ),
             "view_tokens_after": est(strip(new_view)), "summary": summ}
        r.update(kw)
        self.resets.append(r)

    # -- shared pieces -------------------------------------------------------------------------
    def fit(self, summ, budget, condense):
        """Over budget: ask the model once to condense; still over: cut at the budget, marked."""
        sb = {"budget": budget, "tokens_first": tok(summ)}
        if tok(summ) > budget:
            s2 = self.accepted_text("condense", [{"role": "user", "content": condense % int(budget * 0.75) + summ}])
            if s2 and tok(s2) < tok(summ):
                summ = s2
            if tok(summ) > budget:
                summ = clip_tok(summ, budget)
                sb["truncated"] = True
        sb["tokens_final"] = tok(summ)
        return summ, sb

    @staticmethod
    def extract_text(extracts) -> str:
        return EXTRACT_HEAD + "\n\n".join("[Session on %s] User: %s" % (d, t) for d, t in extracts)

    def pick(self, seg):
        """The model picks, by number, the user messages that state lasting facts; the harness copies them."""
        users = [(i, m) for i, m in enumerate(seg) if m["role"] == "user"]
        if not users:
            return [], {"n_user": 0}
        num = {k + 1: m for k, (i, m) in enumerate(users)}
        lines = ["[u%d] (session %s) %s" % (k, m["_date"], clip_tok(m["_raw"], PICK_SHOW_TOK).replace("\n", " "))
                 for k, m in num.items()]

        def validate(txt):
            mm = re.search(r"\[[\s\d,]*\]", txt)
            if not mm:
                return None
            try:
                v = json.loads(mm.group(0))
            except ValueError:
                return None
            v = sorted({int(x) for x in v})
            return v if v and all(x in num for x in v) else None
        v = self.accepted_text("pick", [{"role": "system", "content": SYSTEM},
                                        {"role": "user", "content": PICK_REQUEST + "\n".join(lines)}],
                               temps=(0.0, 0.7, 0.7, 0.7), max_tokens=1024, validate=validate)
        if v is None:
            self.errors.append("pick all rejected")
            return [], {"n_user": len(num), "picked": None}
        return ([(num[k]["_date"], clip_tok(num[k]["_raw"], EXTRACT_ITEM_TOK)) for k in v],
                {"n_user": len(num), "picked": len(v)})

    # -- arms ----------------------------------------------------------------------------------
    def c_tracepack(self, view, seg, sysm):
        """Note-taking summary of [previous summary + new messages] within 0.15 x window, plus the picked user
        messages, verbatim and append-only; the oldest are dropped past 0.2 x window."""
        budget = int(K_SUMMARY_FRAC * self.thr)
        sum_in = [sysm] + ([{"role": "user", "content": S_PREFIX + self.summary}] if self.summary else []) + seg
        summ = self.accepted_text("summary", sum_in + [{"role": "user", "content": SM_REQUEST + KM_NOTE % int(budget * 0.75)}])
        if summ is None:
            return None
        summ, sb = self.fit(summ, budget, KM_CONDENSE)
        picked, pinfo = self.pick(seg)
        self.extracts += picked
        dropped = 0
        cap = int(K_EXTRACT_FRAC * self.thr)
        while self.extracts and tok(self.extract_text(self.extracts)) > cap:
            self.extracts.pop(0)
            dropped += 1
        self.summary = summ
        base = [{"role": "user", "content": S_PREFIX + summ}]
        if self.extracts:
            base.append({"role": "user", "content": self.extract_text(self.extracts)})
        return base, summ, dict(sb=sb, pick=pinfo, extract_items=len(self.extracts),
                                extract_tokens=tok(self.extract_text(self.extracts)) if self.extracts else 0,
                                extract_dropped=dropped, extract_texts=[x[1] for x in self.extracts])

    def c_codex(self, view, seg, sysm):
        """Codex: a hand-off summary of the whole context, then the newest user messages verbatim (bytes / 4
        tokens, the last one that does not fit is cut), then the prefixed summary."""
        summ = self.accepted_text("summary", view + [{"role": "user", "content": CX_PROMPT}])
        if summ is None:
            return None
        U = min(CX_USER_TOKENS, int(0.3 * self.thr))
        users = [m for m in view[1:] if m["role"] == "user" and not (m.get("content") or "").startswith(CX_PREFIX + "\n")]
        keep, rem = [], U
        for m in reversed(users):
            if rem <= 0:
                break
            t = approx_codex(m["content"])
            if t <= rem:
                keep.append(dict(m))
                rem -= t
            else:
                keep.append(dict(m, content=m["content"].encode("utf-8")[:rem * 4].decode("utf-8", "ignore") + " […]"))
                rem = 0
                break
        keep.reverse()
        self.summary = summ
        return keep + [{"role": "user", "content": CX_PREFIX + "\n" + summ}], summ, dict(
            U=U, n_user=len(users), kept_user_msgs=len(keep), kept_tokens_codex=U - rem, kept_texts=[m["content"] for m in keep])

    def c_notes(self, view, seg, sysm):
        """The note-taking summary of the whole context within 0.35 x window; nothing verbatim."""
        budget = int(SMB_FRAC * self.thr)
        summ = self.accepted_text("summary", view + [{"role": "user", "content": SM_REQUEST + KM_NOTE % int(budget * 0.75)}])
        if summ is None:
            return None
        summ, sb = self.fit(summ, budget, KM_CONDENSE)
        self.summary = summ
        return [{"role": "user", "content": S_PREFIX + summ}], summ, dict(sb=sb)


# ---------------------------------------------------------------------------- running a set of units

_lock = threading.Lock()


def _atomic_json(path, obj):
    d = os.path.dirname(path)
    os.makedirs(d, exist_ok=True)
    tmp = os.path.join(d, ".%s.%d.%d.tmp" % (os.path.basename(path), os.getpid(), threading.get_ident()))
    with open(tmp, "w", encoding="utf-8") as fh:
        json.dump(obj, fh, ensure_ascii=False)
    os.replace(tmp, path)


def unit_path(out: str, arm: str, qid: str) -> str:
    return os.path.join(out, "units", "%s__%s.json" % (arm.replace("+", "_"), qid))


def run_units(items: dict, ids: list, arms: list, window: int, out: str, llm, judge_llm, workers: int,
              save_context: bool = True, claude_opts: dict = None, log=print) -> int:
    todo = [(qid, arm) for qid in ids for arm in arms if not os.path.exists(unit_path(out, arm, qid))]
    log("%s: %d units to run (%d questions x %s), %d workers" % (time.strftime("%H:%M:%S"), len(todo), len(ids),
                                                                ",".join(arms), workers))
    done = [0, 0]
    started = [0]
    t0 = time.time()

    def go(job):
        qid, arm = job
        it = items[qid]
        with _lock:
            k = started[0]
            started[0] += 1
        if k < workers:                         # spread the first wave: no burst of process starts and requests
            time.sleep(k * 1.0)
        try:
            if arm in CLAUDE_ARMS:
                from tracepack.bench import lme_claude
                rec, ctx = lme_claude.run_unit(it, arm, window, out, claude_opts or {})
            else:
                u = Unit(it, arm, window, llm)
                rec = u.run()
                ctx = u.final_ctx
            j = None if rec.get("infra") else lme_judge.judge(judge_llm, it, rec.get("answer") or "")
            bundle = {"qid": qid, "arm": arm, "window": window, "type": it["question_type"], "unit": rec, "judge": j,
                      "final_context": ctx if save_context else None, "finished_at": time.time()}
            _atomic_json(unit_path(out, arm, qid), bundle)
            with _lock:
                done[0] += 1
                done[1] += bool(rec.get("infra"))
                if done[0] % 10 == 0 or done[0] == len(todo):
                    rate = done[0] / max(1e-9, time.time() - t0) * 60
                    log("%s: %d/%d done (%d infra failures), %.1f units/min" % (time.strftime("%H:%M:%S"), done[0],
                                                                              len(todo), done[1], rate))
        except Exception:                           # noqa: BLE001 - keep the other units going
            log("%s: unit %s/%s failed:\n%s" % (time.strftime("%H:%M:%S"), arm, qid, traceback.format_exc()[-1500:]))

    with ThreadPoolExecutor(max(1, workers)) as ex:
        list(ex.map(go, todo))
    return len(todo)


# ---------------------------------------------------------------------------- report

def load_results(out: str) -> list:
    rows = []
    d = os.path.join(out, "units")
    for f in sorted(os.listdir(d)) if os.path.isdir(d) else []:
        if f.endswith(".json"):
            with open(os.path.join(d, f), encoding="utf-8") as fh:
                rows.append(json.load(fh))
    return rows


def verdicts(rows) -> dict:
    """(qid, arm) -> True / False, or None when the unit failed or the judge gave no verdict."""
    res = {}
    for b in rows:
        j = b.get("judge") or {}
        res[(b["qid"], b["arm"])] = None if (b["unit"].get("infra") or "ok" not in j) else bool(j["ok"])
    return res


class Paired:
    """Accuracy differences with a bootstrap over questions (4,000 resamples, seed 20261004); a unit with no
    verdict counts as missing for its own arm only."""

    def __init__(self, res: dict, qids: list, n_boot: int = 4000, seed: int = SEED):
        self.res, self.qids = res, sorted(qids)
        rng = random.Random(seed)
        self.boots = [[rng.choice(self.qids) for _ in self.qids] for _ in range(n_boot)]

    def rate(self, qs, arm):
        v = [self.res.get((q, arm)) for q in qs]
        v = [x for x in v if x is not None]
        return sum(v) / len(v) if v else float("nan")

    def diff(self, x, y, qs=None):
        qs = qs or self.qids
        sub = set(qs)
        d = self.rate(qs, x) - self.rate(qs, y)
        bs = sorted(v for v in (self.rate([q for q in bq if q in sub], x) - self.rate([q for q in bq if q in sub], y)
                                for bq in self.boots) if v == v)
        lo, hi = bs[int(0.025 * len(bs))], bs[int(0.975 * len(bs)) - 1]
        p = min(1.0, 2 * min(sum(v <= 0 for v in bs), sum(v >= 0 for v in bs)) / len(bs))
        return d, lo, hi, p


def report(out: str, pairs: list = None, n_boot: int = 4000, as_json: bool = False) -> dict:
    rows = load_results(out)
    if not rows:
        raise SystemExit("no results under %s/units" % out)
    res = verdicts(rows)
    arms = [a for a in HARNESS_ARMS + CLAUDE_ARMS if any(b["arm"] == a for b in rows)]
    arms += sorted({b["arm"] for b in rows} - set(arms))
    by_q = collections.defaultdict(set)
    for b in rows:
        by_q[b["qid"]].add(b["arm"])
    qids = sorted(q for q, a in by_q.items() if set(arms) <= a) or sorted(by_q)
    types = {b["qid"]: b.get("type") for b in rows}
    P = Paired(res, qids, n_boot)
    summary = {"questions": len(qids), "arms": {}, "pairs": [], "by_type": {}}
    for arm in arms:
        v = [res.get((q, arm)) for q in qids]
        vv = [x for x in v if x is not None]
        toks = [sum((c.get("completion_tokens") or 0) for c in b["unit"].get("calls") or [])
                for b in rows if b["arm"] == arm and b["qid"] in set(qids)]
        summary["arms"][arm] = {"correct": sum(vv), "graded": len(vv), "missing": len(v) - len(vv),
                                "accuracy": sum(vv) / max(1, len(vv)),
                                "generated_tokens_per_question": sum(toks) / max(1, len(toks))}
    if pairs is None:
        pairs = [(a, b) for a, b in (("tracepack", "codex"), ("tracepack", "notes"), ("codex", "notes"),
                                     ("claude-code+tracepack", "claude-code"), ("claude-code+notes", "claude-code"),
                                     ("claude-code+restore", "claude-code"), ("tracepack", "claude-code"),
                                     ("claude-code+tracepack", "codex")) if a in arms and b in arms]
    for a, b in pairs:
        d, lo, hi, p = P.diff(a, b)
        summary["pairs"].append({"a": a, "b": b, "diff": d, "lo": lo, "hi": hi, "p": p})
    for t in sorted({types[q] for q in qids if types.get(q)}):
        qs = [q for q in qids if types.get(q) == t]
        summary["by_type"][t] = {"n": len(qs), **{arm: P.rate(qs, arm) for arm in arms}}
    if as_json:
        print(json.dumps(summary, indent=1))
        return summary
    print("LongMemEval-S, %d questions answered by every arm; judge: LongMemEval's official prompts" % len(qids))
    print()
    print("| Arm | Accuracy | Graded | Generated tokens / question |")
    print("|---|---|---|---|")
    for arm in arms:
        s = summary["arms"][arm]
        print("| %s | %.1f%% | %d/%d | %s |" % (arm, 100 * s["accuracy"], s["graded"], s["graded"] + s["missing"],
                                             "{:,.0f}".format(s["generated_tokens_per_question"])))
    if summary["pairs"]:
        print()
        print("| Pair | Difference [95% CI] | p |")
        print("|---|---|---|")
        for x in summary["pairs"]:
            print("| %s − %s | %+.1f [%+.1f, %+.1f] | %.3f |" % (x["a"], x["b"], 100 * x["diff"], 100 * x["lo"],
                                                               100 * x["hi"], x["p"]))
    print()
    print("| Question type | n | " + " | ".join(arms) + " |")
    print("|---|---|" + "---|" * len(arms))
    for t, s in summary["by_type"].items():
        print("| %s | %d | %s |" % (t, s["n"], " | ".join("%.1f%%" % (100 * s[a]) for a in arms)))
    return summary


# ---------------------------------------------------------------------------- command line

def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(prog="tracepack bench lme", description=__doc__.split("\n\n")[0],
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = p.add_subparsers(dest="cmd")
    d = sub.add_parser("download", help="fetch LongMemEval-S (cleaned) and check its sha256")
    d.add_argument("--data", default="", help="where to put it (default ~/.cache/tracepack/)")

    r = sub.add_parser("run", help="run arms over questions (resumable: finished units are skipped)")
    r.add_argument("--arms", default="tracepack,codex", help="comma-separated: " + ", ".join(HARNESS_ARMS + CLAUDE_ARMS))
    r.add_argument("--out", required=True, help="results folder")
    r.add_argument("--data", default="", help="longmemeval_s_cleaned.json (default: download)")
    r.add_argument("--ids", default="", help="'' = the study's 480 questions, 'first200', 'all', or a JSON file")
    r.add_argument("--limit", type=int, default=0, help="only the first N questions")
    r.add_argument("--window", type=int, default=DEFAULT_WINDOW, help="context window in tokens (default 32768)")
    r.add_argument("--workers", type=int, default=16)
    r.add_argument("--dry-run", action="store_true", help="stub model and judge: zero model calls")
    r.add_argument("--no-context", action="store_true", help="don't save each answer's final context")
    r.add_argument("--claude", default="claude", help="Claude Code binary for the claude-code arms")
    r.add_argument("--plugin-dir", default="", help="TracePack plugin folder (default: this checkout)")
    r.add_argument("--anthropic-base-url", default="",
                   help="send Claude Code to this Anthropic endpoint instead of the built-in bridge to TRACEPACK_LLM_*")

    rp = sub.add_parser("report", help="accuracy per arm, paired differences, by question type")
    rp.add_argument("--out", required=True)
    rp.add_argument("--pairs", default="", help="a:b,c:d (default: the standard comparisons present)")
    rp.add_argument("--boot", type=int, default=4000)
    rp.add_argument("--json", action="store_true")

    b = sub.add_parser("bridge", help="serve Claude Code (Anthropic Messages API) from TRACEPACK_LLM_*")
    b.add_argument("--port", type=int, default=21780)
    b.add_argument("--log", default="")
    return p


def main(argv=None) -> int:
    a = build_parser().parse_args(argv)
    if a.cmd == "download":
        print(download(a.data))
        return 0
    if a.cmd == "report":
        pairs = [tuple(x.split(":", 1)) for x in a.pairs.split(",") if ":" in x] or None
        report(a.out, pairs, a.boot, a.json)
        return 0
    if a.cmd == "bridge":
        from tracepack.bench import anthropic_bridge
        anthropic_bridge.serve(a.port, Endpoint.from_env("llm"), log_path=a.log)
        return 0
    if a.cmd != "run":
        build_parser().print_help()
        return 0
    arms = [x.strip() for x in a.arms.split(",") if x.strip()]
    bad = [x for x in arms if x not in HARNESS_ARMS + CLAUDE_ARMS]
    if bad:
        raise SystemExit("unknown arm(s): %s" % ", ".join(bad))
    data = a.data or (default_data_path() if a.dry_run and os.path.isfile(default_data_path()) else "") or download()
    items = load_items(data)
    ids = load_ids(a.ids)
    if ids == ["*"]:
        ids = sorted(items)
    if a.limit:
        ids = ids[:a.limit]
    missing = [q for q in ids if q not in items]
    if missing:
        raise SystemExit("%d question ids are not in %s" % (len(missing), data))
    os.makedirs(a.out, exist_ok=True)
    if a.dry_run:
        llm = judge_llm = StubEndpoint()
    else:
        try:
            llm, judge_llm = Endpoint.from_env("llm"), Endpoint.from_env("judge")
        except LLMError as e:
            raise SystemExit(str(e))
    claude_opts = {}
    if any(x in CLAUDE_ARMS for x in arms):
        from tracepack.bench import lme_claude
        claude_opts = lme_claude.prepare(a, llm, a.out)
    with open(os.path.join(a.out, "run.json"), "a", encoding="utf-8") as fh:
        fh.write(json.dumps({"at": time.strftime("%Y-%m-%d %H:%M:%S"), "arms": arms, "window": a.window,
                             "questions": len(ids), "model": getattr(llm, "model", ""),
                             "judge": getattr(judge_llm, "model", ""), "dry_run": a.dry_run,
                             "data_sha256": DATA_SHA256 if os.path.basename(data).startswith("longmemeval_s") else ""}) + "\n")
    try:
        run_units(items, ids, arms, a.window, a.out, llm, judge_llm, a.workers, not a.no_context, claude_opts,
                  log=lambda m: print(m, file=sys.stderr, flush=True))
    finally:
        if claude_opts.get("stop"):
            claude_opts["stop"]()
    report(a.out)
    return 0


if __name__ == "__main__":
    sys.exit(main())
