"""tracepack.eval.reader -- the forced-choice reader used by every arm.

Register (identical to the one validated across the earlier EC0-EC4/EC1b experiments):
the packet context is served, then a two-choice question is asked and the FIRST-TOKEN logits of
" A" / " B" are read.  No free generation, no judge: `correct` is a deterministic function of the
logits, so re-running an arm cannot drift.

Two properties this file is responsible for:

* **The letter side is fixed by a hash of the item id**, not by argument order, so a model with a
  position bias cannot help one arm more than another (the arms share the item).
* **The question text never contains the gold value or the distractor beyond the two options** --
  the caller supplies a wording template that references the value's ROLE, and we assert the gold
  string does not appear in the stem.

`ReaderResult.score` is the gold-vs-distractor log-odds, so a null arm sits at ~0 and the scale is
comparable across arms.
"""
from __future__ import annotations

import hashlib
import math
import os
import sys
from dataclasses import dataclass
from tracepack.core.textmatch import contains_value

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

STEM = {
    "lookup": "Exactly one of the two options below appeared in the session above. Which one?",
    "why": "Exactly one of the two options below is the value the work above actually depended on. Which one?",
    "state": "Exactly one of the two options below is the CURRENT value after the latest change above. Which one?",
    "audit": "Exactly one of the two options below is supported by the evidence above. Which one?",
}
STEM_SHA = hashlib.sha1("|".join(sorted(STEM.values())).encode()).hexdigest()[:10]


@dataclass(frozen=True)
class ReaderResult:
    correct: bool
    score: float          # log-odds of gold vs distractor
    raw: str              # "A"/"B" chosen
    gold_is_a: bool
    n_tokens: int = 0     # what the packet ACTUALLY tokenised to (budgets use a len//4 heuristic)


def gold_is_a(item_id: str) -> bool:
    return bool(int(hashlib.sha1(item_id.encode()).hexdigest(), 16) & 1)


def build_prompt(context: str, item: dict, distractor: str) -> tuple[str, bool]:
    ga = gold_is_a(item["item_id"])
    a, b = (item["gold"], distractor) if ga else (distractor, item["gold"])
    stem = STEM[item.get("query_mode", "lookup")]
    head = "Session evidence:\n" + (context if context else "(no evidence provided)")
    q = "%s\nOption A: %s\nOption B: %s\nAnswer with the letter only." % (stem, a, b)
    return head + "\n\n" + q, ga


def _model_path(model_path=None):
    """The reader checkpoint: the argument, else TRACEPACK_READER_MODEL.  There is no built-in default."""
    path = model_path or os.environ.get("TRACEPACK_READER_MODEL", "")
    if not path:
        raise RuntimeError("set TRACEPACK_READER_MODEL (or pass model_path) to a local Hugging Face "
                           "checkpoint; every published read-out used Qwen3-8B or Qwen3-32B")
    return path


class _HF:
    """Tokenizer + causal LM in bf16.  torch / transformers are imported only when a reader is built."""

    def __init__(self, path):
        import torch
        from transformers import AutoModelForCausalLM, AutoTokenizer
        self.tok = AutoTokenizer.from_pretrained(path, trust_remote_code=True)
        self.tok.padding_side = "left"
        if self.tok.pad_token_id is None:
            self.tok.pad_token = self.tok.eos_token
        self.model = AutoModelForCausalLM.from_pretrained(
            path, torch_dtype=torch.bfloat16, device_map="auto", trust_remote_code=True)
        self.model.eval()


def _wrap_nothink(tok):
    """(head, tail) around one user turn under the model's chat template, thinking switched off.

    For Qwen3, ``enable_thinking=False`` writes an empty think block into the assistant prefix, so
    the next token is the answer itself -- which is what the forced-choice logits read."""
    mark = "<<<SPLITPOINT>>>"
    msgs = [{"role": "user", "content": mark}]
    try:
        full = tok.apply_chat_template(msgs, tokenize=False, add_generation_prompt=True,
                                       enable_thinking=False)
    except TypeError:
        full = tok.apply_chat_template(msgs, tokenize=False, add_generation_prompt=True)
    head, tail = full.split(mark)
    return head, tail


class QwenReader:
    """Real reader: Qwen3-8B forced choice.  Import is lazy so the module loads without torch."""

    def __init__(self, model_path=None, max_ctx=32000):
        # Budgets are counted with a len//4 heuristic, so a CJK/code-heavy packet can tokenise to
        # well over its nominal budget.  Nothing downstream truncates -- the model would simply
        # extrapolate RoPE past its trained window and degrade silently, which is exactly the
        # failure mode that looks like "the evidence did not help".  So: count, and refuse.
        self.max_ctx = int(os.environ.get("TRACEPACK_MAX_CTX", max_ctx))
        self.n_over = 0
        self.max_seen = 0
        self.model_path = _model_path(model_path)
        self.llm = _HF(self.model_path)
        self.head, self.tail = _wrap_nothink(self.llm.tok)
        self._A = self._letter("A")
        self._B = self._letter("B")

    def _letter(self, ch):
        ids = []
        for s in (ch, " " + ch):
            t = self.llm.tok(s, add_special_tokens=False).input_ids
            if len(t) == 1:
                ids.append(t[0])
        if not ids:
            raise RuntimeError("tokenizer has no single-token %r" % ch)
        return ids

    def __call__(self, context: str, item: dict) -> ReaderResult:
        import torch
        distractor = item["distractors"][0]
        prompt, ga = build_prompt(context, item, distractor)
        assert item["gold"] not in STEM[item.get("query_mode", "lookup")], "gold leaked into stem"
        text = self.head + prompt + self.tail + "Answer:"
        ids = self.llm.tok(text, add_special_tokens=False).input_ids
        self.max_seen = max(self.max_seen, len(ids))
        if len(ids) > self.max_ctx:
            self.n_over += 1
            raise RuntimeError(
                "packet tokenises to %d tokens, over TRACEPACK_MAX_CTX=%d (item %s). Refusing: past "
                "the trained window the model degrades silently and the run would look like an "
                "evidence failure. Lower the budget or raise the window deliberately."
                % (len(ids), self.max_ctx, item.get("item_id")))
        with torch.inference_mode():
            t = torch.tensor([ids], device=self.llm.model.device)
            logits = self.llm.model(t).logits[0, -1].float()
        la = max(float(logits[i]) for i in self._A)
        lb = max(float(logits[i]) for i in self._B)
        d = la - lb
        score = d if ga else -d
        chose_a = d > 0
        return ReaderResult(correct=(chose_a == ga), score=round(score, 4),
                            raw="A" if chose_a else "B", gold_is_a=ga, n_tokens=len(ids))


def build_open_prompt(context: str, item: dict) -> str:
    """Free-form readout: ask the real question, take whatever the model writes.

    Why this exists alongside the two-choice reader: forced choice measures RECOGNITION, and it
    throws away half the scale by construction (a model that has read nothing still scores .500).
    It also made the distractor generator load-bearing -- three of the red team's five findings
    were about how the wrong options get built.  Asking the question openly removes that whole
    attack surface: there are no wrong options to construct.

    The question is the item's OWN generated question (leak-screened at build time), not the
    generic stem the two-choice prompt used, so this is also a more faithful task.
    """
    q = item.get("query") or STEM[item.get("query_mode", "lookup")]
    head = "Session evidence:\n" + (context if context else "(no evidence provided)")
    ask = ("%s\n\nAnswer with the value only -- a single file path, number, identifier or "
           "version string. If the evidence above does not contain it, answer exactly: UNKNOWN"
           % q)
    return head + "\n\n" + ask


class GenerativeQwenReader:
    """Same model, same served context, open-ended answer.  Correctness is decided later by a
    judge (see tracepack/eval/judge_answers.py) -- this class never guesses at it.

    Deliberately calls ``model.generate`` on the exact token ids, with no ``truncation=True,
    max_length=...``: truncation would silently cut long packets and turn a context-length problem
    into what looks like an evidence problem.  Here an over-long packet raises instead.
    """

    def __init__(self, model_path=None, max_ctx=32000, max_new=48):
        self.model_path = _model_path(model_path)
        self.llm = _HF(self.model_path)
        self.head, self.tail = _wrap_nothink(self.llm.tok)
        self.max_ctx = int(os.environ.get("TRACEPACK_MAX_CTX", max_ctx))
        self.max_new = max_new
        self.max_seen = 0

    def __call__(self, context: str, item: dict) -> ReaderResult:
        import torch
        text = self.head + build_open_prompt(context, item) + self.tail
        ids = self.llm.tok(text, add_special_tokens=False).input_ids
        self.max_seen = max(self.max_seen, len(ids))
        if len(ids) > self.max_ctx - self.max_new:
            raise RuntimeError(
                "packet tokenises to %d tokens, over TRACEPACK_MAX_CTX=%d minus %d generated "
                "(item %s). Refusing rather than truncating."
                % (len(ids), self.max_ctx, self.max_new, item.get("item_id")))
        with torch.inference_mode():
            t = torch.tensor([ids], device=self.llm.model.device)
            out = self.llm.model.generate(t, max_new_tokens=self.max_new, do_sample=False,
                                          pad_token_id=self.llm.tok.pad_token_id)
        ans = self.llm.tok.decode(out[0][len(ids):], skip_special_tokens=True).strip()
        # correct=False is a PLACEHOLDER: the judge fills it in.  Never inferred here.
        return ReaderResult(correct=False, score=0.0, raw=ans,
                            gold_is_a=gold_is_a(item["item_id"]), n_tokens=len(ids))


class DeterministicFakeReader:
    """Offline stand-in used by tests and by the profiler selfcheck.

    Answers correctly iff the gold value literally appears in the served context -- i.e. it models
    a PERFECT reader over the served text.  Any arm that beats it is leaking; any arm far below it
    is a reader/format problem rather than an evidence problem.
    """

    def __call__(self, context: str, item: dict) -> ReaderResult:
        ga = gold_is_a(item["item_id"])
        present = contains_value(context or "", item["gold"])
        return ReaderResult(correct=present, score=3.0 if present else -0.0,
                            raw="A" if (present == ga) else "B", gold_is_a=ga)


def _selfcheck():
    item = dict(item_id="x#1", gold="kvmemory/ca_gate.py", distractors=["kvmemory/ca_llm.py"],
                query_mode="lookup")
    p, ga = build_prompt("we edited kvmemory/ca_gate.py today", item, item["distractors"][0])
    assert "Option A:" in p and "Option B:" in p
    want = ("Option A: " if ga else "Option B: ") + item["gold"]
    other = ("Option B: " if ga else "Option A: ") + item["distractors"][0]
    assert want in p and other in p, "gold/distractor must sit on the sides the id hash picked"
    # side assignment is stable and depends only on the id
    assert gold_is_a("x#1") == gold_is_a("x#1")
    sides = [gold_is_a("i#%d" % i) for i in range(200)]
    assert 60 < sum(sides) < 140, "letter side must be balanced, got %d/200" % sum(sides)
    fr = DeterministicFakeReader()
    assert fr("we edited kvmemory/ca_gate.py", item).correct is True
    assert fr("nothing relevant here", item).correct is False
    # fault injection: an empty context must not be scored as correct
    assert fr("", item).correct is False
    print("reader selfcheck OK (stem=%s, side balance %d/200)" % (STEM_SHA, sum(sides)))


if __name__ == "__main__":
    _selfcheck()
