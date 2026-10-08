"""ContextWeaver (arXiv:2604.23069), implemented from the paper's description.

    Yating Wu, Yuhao Zhang, Sayan Ghosh, Sourya Basu, Anoop Deoras, Jun Huan, Gaurav Gupta.
    "ContextWeaver: Selective and Dependency-Structured Memory Construction for LLM Agents."
    UT Austin / AWS AI Labs, 2026-04-24.

**No code was released with the paper.**  Everything here is written from the text, so every number
it produces has to be reported as *faithful-as-described*, never as "ContextWeaver scores X".  The
honest guard against "your baseline is bad because you implemented it badly" is not a disclaimer, it
is a positive control: :func:`plant_control` builds a trace with a known dependency chain and checks
the analyzer puts the true parent in the top-m.  A run that fails it is an instrument failure and its
numbers are not quotable -- the same judgement this project already applied to its own graph arm.

What the paper fixes, and what it leaves to the implementer
-----------------------------------------------------------
FIXED by the paper (reproduced here):
  * node = one (Thought, Action, Observation) entry, carrying ``summary``, ``dependency_summary``,
    ``parents``, ``validation``
  * parents chosen by an LLM "Logical Dependency Analyzer", ``S_k = Top_m{N_i in C_k} LLM(N_i->N_k|Q)``,
    prompted about *information flow and causal relationships, not file similarity*
  * candidates exclude nodes marked Failed or Superseded
  * ancestry = breadth-first upward through parent edges from the current node, capped at W
  * warmup: while |H| <= W keep the whole history
  * non-ancestor entries keep thought+action and have their **observation replaced by a placeholder**
  * ``dependency_summary_k = LLMSUM({dependency_summary_i : N_i in S_k}, summary_k)``
  * W = 5 in every experiment the paper reports

LEFT OPEN (chosen here, and flagged so a reader can dispute the choice):
  * ``m`` is never given a value in the main text -> :data:`CWConfig.m` = 3, reported next to results
  * the rule for ``validation`` is not specified -> rule-based (see :func:`_validation`), not an LLM
    call, because guessing an LLM prompt for it would add a second unverifiable degree of freedom
  * the scorer's output format -> an integer 0-10; the top-m is by that score, ties broken by recency
"""
from __future__ import annotations

import json
import re
import threading
import time
import urllib.error
import urllib.request
from dataclasses import dataclass, field

PLACEHOLDER = "[observation omitted -- this step is not in the current dependency ancestry]"

#: Quoted from the paper's description of the analyzer prompt.
PARENT_PROMPT = (
    "You are the Logical Dependency Analyzer of an agent's memory.\n"
    "The agent is working towards this goal:\n{goal}\n\n"
    "CURRENT OPERATION:\n{child}\n\n"
    "PREVIOUS OPERATION:\n{parent}\n\n"
    "Does the current operation need specific information from the previous operation?\n"
    "Judge information flow and causal relationships, not file similarity: two steps touching the "
    "same file are NOT dependent unless one uses a value, identifier, or result the other produced.\n"
    "Answer with a single integer 0-10 and nothing else, where 10 means the current operation could "
    "not have been performed without the previous one's output, and 0 means no information flows "
    "between them."
)

NODE_PROMPT = (
    "Summarize this step of an agent's trajectory in at most 60 words, covering what it intended, "
    "what it did, and what came back.\n\n"
    "THOUGHT:\n{thought}\n\nACTION:\n{action}\n\nOBSERVATION:\n{observation}\n\n"
    "Reply with the summary only."
)

DEPSUM_PROMPT = (
    "Write one compact narrative, at most 70 words, of the reasoning path that leads to the current "
    "step, by extending the paths already established by its parents.\n\n"
    "PARENT PATHS:\n{parents}\n\nCURRENT STEP:\n{summary}\n\n"
    "Reply with the narrative only."
)


# ---------------------------------------------------------------- config / state


@dataclass
class CWConfig:
    """W and the prompts are the paper's; m and the scorer format are ours (see module docstring)."""

    W: int = 5
    m: int = 3
    model: str = ""
    base_url: str = ""
    api_key: str = "local"
    scorer: str = "pairwise"          # pairwise (faithful) | batched (cheaper, a DEVIATION)
    supersede: str = "write_path"     # write_path (corrected) | legacy (what published W ran)
    validation: str = "anchored"      # anchored (corrected)   | legacy (what published W ran)
    max_workers: int = 16
    summarize: bool = True            # node summaries + dependency summaries (the LLMSUM half)
    timeout: int = 180
    max_candidates: int = 0           # 0 = every non-failed, non-superseded prior node


@dataclass
class Node:
    idx: int
    thought: str = ""
    action: str = ""
    observation: str = ""
    summary: str = ""
    dependency_summary: str = ""
    parents: tuple = ()
    validation: str = "unknown"       # passed | failed | unknown | superseded
    event_ids: tuple = ()
    #: the user instruction in force when this step ran.  Parent scoring is conditioned on THIS,
    #: never on a later query: letting a node see the instruction that only arrives after the
    #: cut point would build the graph with future information (the leak constraint in the
    #: comparison plan, sec 8.3).  The live binding gets this for free -- it reads the view as it
    #: is at that moment -- so this field is what makes the OFFLINE replay match it.
    goal: str = ""
    scores: dict = field(default_factory=dict)

    def brief(self, obs_chars: int = 900) -> str:
        o = self.observation or ""
        if len(o) > obs_chars:
            o = o[:obs_chars] + " ...[truncated]"
        return "step %d\nthought: %s\naction: %s\nobservation: %s" % (
            self.idx, (self.thought or "")[:400], (self.action or "")[:400], o)


@dataclass
class CWResult:
    context: str
    nodes: list
    ancestry: tuple
    chars: int = 0
    tokens: int = 0
    llm_calls: int = 0
    llm_tokens: int = 0
    graph_ms: int = 0
    select_ms: int = 0
    warmup: bool = False

    def as_dict(self) -> dict:
        return {"chars": self.chars, "tokens": self.tokens, "n_nodes": len(self.nodes),
                "ancestry": list(self.ancestry), "llm_calls": self.llm_calls,
                "llm_tokens": self.llm_tokens, "graph_ms": self.graph_ms,
                "select_ms": self.select_ms, "warmup": self.warmup}


# ---------------------------------------------------------------- LLM


class Chat:
    """Minimal OpenAI-compatible client with a call/token counter.  No dependency on purpose."""

    def __init__(self, base_url: str, model: str, api_key: str = "local", timeout: int = 180):
        self.base_url = base_url.rstrip("/")
        self.model = model
        self.api_key = api_key
        self.timeout = timeout
        self.calls = 0
        self.tokens = 0
        # an empty reply is not an error here -- the scorer's parser reads the first integer and
        # treats no integer as zero -- but a run where EVERY reply is empty is an analyzer that
        # never spoke, which reads out as the collapse we are measuring.  Count them.
        self.empty = 0
        #: scoring calls that raised even after the retries, absorbed as a zero score by
        #: :func:`select_parents`.  Counted here so an absorbed failure is never a silent one.
        self.failed = 0
        self._lock = threading.Lock()

    def ask(self, prompt: str, max_tokens: int = 200, temperature: float = 0.0) -> str:
        body = {"model": self.model, "max_tokens": max_tokens, "temperature": temperature,
                "messages": [{"role": "user", "content": prompt}]}
        req = urllib.request.Request(
            self.base_url + "/chat/completions", data=json.dumps(body).encode("utf-8"),
            headers={"Content-Type": "application/json", "Authorization": "Bearer " + self.api_key})
        last = None
        for _ in range(3):
            try:
                with urllib.request.urlopen(req, timeout=self.timeout) as r:
                    out = json.load(r)
                break
            except (urllib.error.URLError, TimeoutError, OSError) as exc:   # transient
                last = exc
                time.sleep(2)
        else:
            raise RuntimeError("LLM call failed after 3 attempts: %s" % (last,))
        txt = (out["choices"][0]["message"].get("content") or "")
        txt = re.sub(r"<think>.*?</think>", "", txt, flags=re.S).strip()
        used = (out.get("usage") or {}).get("total_tokens") or 0
        with self._lock:
            self.calls += 1
            self.tokens += used
            if not txt:
                self.empty += 1
        return txt


def _pmap(fn, items, workers: int):
    """Thread-pool map that keeps order.  The pairwise scorer is O(k) independent calls per step;
    issuing them concurrently is what makes the faithful scorer affordable on a local vLLM."""
    if not items:
        return []
    try:
        from concurrent.futures import ThreadPoolExecutor
    except ImportError:                                            # pragma: no cover
        return [fn(x) for x in items]
    with ThreadPoolExecutor(max_workers=max(1, workers)) as ex:
        return list(ex.map(fn, items))


# ---------------------------------------------------------------- node extraction (phi)


#: The published classifier: a bare word anywhere in the observation.  Kept so the W numbers
#: reproduce, and because it is the thing the measurement below is about.
_FAIL = re.compile(r"\b(error|traceback|exception|no such file|command not found|failed|"
                   r"exit code [1-9]|permission denied)\b", re.I)

#: The corrected classifier: a failure announces itself at the START of a line, or as a process's
#: own verdict.  Text that merely mentions an error -- a file body, a grep hit, source being typed --
#: is not a failure.
_FAIL_ANCHORED = re.compile(
    r"^\s*(Traceback \(most recent call last\)"
    r"|\w*(Error|Exception)\s*:"
    r"|E\s+\w"
    r"|FAILED\b"
    r"|fatal:|error:|ERROR:"
    # a tool announcing its own failure, with the offending path usually in between:
    #   `grep: /testbed/nope.py: No such file or directory`
    r"|\S+: .*(command not found|No such file or directory|Permission denied)"
    r"|exit code [1-9])", re.M)


def _validation(observation: str, mode: str = "anchored") -> str:
    """passed | failed | unknown.  The paper names the states but not the classifier, so this is ours.

    ``mode="legacy"`` is the published behaviour: any of error/traceback/exception/failed as a bare
    word anywhere marks the step failed.  Measured over 1,577 real nodes it marked 11.9% failed, and
    **65% of those show no failure at the start of any line** -- successful edits and file reads whose
    echoed content happens to contain the word.  Since failed steps are dropped from the dependency
    analyzer's candidate list, that removed 11.6% of all needed sources (only 29 of 95 of which
    contained a traceback).  See RESULTS_swebench 3.7.
    """
    if not observation:
        return "unknown"
    if mode == "legacy":
        return "failed" if _FAIL.search(observation) else "passed"
    if mode != "anchored":
        raise ValueError("unknown validation mode %r" % (mode,))
    return "failed" if _FAIL_ANCHORED.search(observation) else "passed"


def extract_nodes(rows, validation: str = "anchored") -> list:
    """Rows in the project's adapter format -> ContextWeaver nodes, one per (action, observation).

    ``rows`` is the same shape ``tracepack.condenser.oh_rows`` emits: dicts with ``role``,
    ``content`` and, for assistant turns, ``tool_calls``.  An assistant turn's text is the Thought,
    its tool call the Action, and the following tool message the Observation.
    """
    nodes, pending, goal = [], None, ""
    for r in rows:
        if r.get("tracepack_format"):
            continue
        role = r.get("role")
        if role == "user":
            goal = r.get("content") or goal      # the instruction in force from here on
            pending = None
            continue
        if role == "assistant":
            calls = r.get("tool_calls") or []
            thought = r.get("content") or ""
            if not calls:
                pending = None
                continue
            fn = (calls[0] or {}).get("function") or {}
            pending = Node(idx=len(nodes), thought=thought,
                           action="%s %s" % (fn.get("name") or "", fn.get("arguments") or ""),
                           goal=goal, event_ids=(r.get("id") or "",))
        elif role == "tool" and pending is not None:
            pending.observation = r.get("content") or ""
            pending.validation = _validation(pending.observation, validation)
            pending.event_ids = tuple(list(pending.event_ids) + [r.get("id") or ""])
            nodes.append(pending)
            pending = None
    return nodes


#: file-editor operations that CHANGE the file.  A view does not supersede anything, and is never
#: itself superseded: its observation is usually the evidence a later write is based on.
_WRITE_OPS = ("str_replace", "create", "insert", "append", "write", "edit")


def mark_superseded(nodes, mode: str = "write_path") -> None:
    """A later node writing the same target supersedes an earlier one.

    The paper lists ``superseded`` as a validation state but gives no rule, so this one is ours and
    has to earn its keep.  ``mode="write_path"`` is the narrow rule the sentence above describes:
    a later file-editor WRITE to a path supersedes an earlier WRITE to the same path.

    ``mode="legacy"`` is what produced the published W numbers, kept only so they reproduce.  It
    matched ``"command"`` as well as ``"path"``, and its capture stops at the first escaped quote,
    so a terminal step was keyed on a truncated prefix.  Measured over 1,584 nodes from 31 real
    trajectories: it superseded 684 (43%), 456 of them terminal steps; ``grep -n`` with a truncated capture alone accounted
    for 151 and covers 42 unrelated commands; for file-editor actions the first matching field is the
    OPERATION NAME, so every ``str_replace`` superseded every earlier one whatever the path.  The
    effect on the arm under test is not cosmetic: 49% of the steps a later step demonstrably depends
    on never reached the dependency analyser (RESULTS_swebench 3.7).
    """
    if mode == "legacy":
        last = {}
        for n in nodes:
            m = re.search(r'"(?:path|file_path|command)"\s*:\s*"([^"]+)"', n.action or "")
            if not m:
                continue
            key = m.group(1)
            if key in last and n.validation == "passed":
                nodes[last[key]].validation = "superseded"
            last[key] = n.idx
        return
    if mode != "write_path":
        raise ValueError("unknown supersede mode %r" % (mode,))
    last_write = {}
    for n in nodes:
        act = n.action or ""
        if not act.startswith("file_editor"):
            continue
        op = re.search(r'"command"\s*:\s*"([^"]+)"', act)
        path = re.search(r'"(?:path|file_path)"\s*:\s*"([^"]+)"', act)
        if not path or not op or op.group(1) not in _WRITE_OPS:
            continue
        key = path.group(1)
        if key in last_write and n.validation == "passed":
            nodes[last_write[key]].validation = "superseded"
        last_write[key] = n.idx


# ---------------------------------------------------------------- the analyzer


def _score_text(txt: str) -> float:
    m = re.search(r"\d+", txt or "")
    return float(m.group(0)) if m else 0.0


def select_parents(chat, node, candidates, goal: str, cfg: CWConfig) -> tuple:
    """``S_k = Top_m{N_i in C_k} LLM(N_i -> N_k | Q)``.

    Faithful form is one call per candidate.  They are independent, so they go out concurrently --
    that is an execution detail, not a change to the estimator.
    """
    if not candidates:
        return (), {}
    if cfg.scorer == "batched":                                    # DEVIATION, must be labelled
        listing = "\n\n".join("[%d] %s" % (c.idx, c.brief(300)) for c in candidates)
        txt = chat.ask(PARENT_PROMPT.format(goal=goal, child=node.brief(500), parent=listing)
                       + "\n\nScore EVERY previous operation. Reply as `index:score` per line.",
                       max_tokens=400)
        sc = {}
        for line in (txt or "").splitlines():
            m = re.match(r"\s*\[?(\d+)\]?\s*[:=]\s*(\d+)", line)
            if m:
                sc[int(m.group(1))] = float(m.group(2))
        scores = {c.idx: sc.get(c.idx, 0.0) for c in candidates}
    else:
        def one(c):
            # A scoring call that fails is worth a zero, not a dead cell.  The analyzer may sit
            # behind a shim that is restarted mid-run (arm T), and `ask` gives up after three
            # attempts; letting that propagate cost three live cells on 09-16.  Zero is exactly what
            # an unparseable reply already scores -- the difference is that this one is counted, so
            # a run where EVERY call fails cannot masquerade as a run where the analyzer found no
            # information flow.  Summarisation does NOT absorb: an empty summary changes what the
            # agent reads, not just how a pair ranks.
            try:
                return _score_text(chat.ask(
                    PARENT_PROMPT.format(goal=goal, child=node.brief(), parent=c.brief()), max_tokens=8))
            except Exception:
                with chat._lock:
                    chat.failed += 1
                    chat.empty += 1
                return 0.0
        vals = _pmap(one, candidates, cfg.max_workers)
        scores = {c.idx: v for c, v in zip(candidates, vals)}
    ranked = sorted(candidates, key=lambda c: (-scores[c.idx], -c.idx))
    top = tuple(c.idx for c in ranked[:cfg.m] if scores[c.idx] > 0)
    return top, scores


def build_graph(chat, nodes, goal: str, cfg: CWConfig) -> None:
    """Algorithm 1 steps 1-5, incrementally over the whole history.  Mutates ``nodes`` in place.

    ``goal`` is only the fallback: a node is scored against the instruction that was in force
    when it ran (``Node.goal``), so replaying a history offline cannot leak an instruction that
    had not been issued yet.
    """
    mark_superseded(nodes, cfg.supersede)
    for n in nodes:
        cands = [c for c in nodes[:n.idx] if c.validation not in ("failed", "superseded")]
        if cfg.max_candidates:
            cands = cands[-cfg.max_candidates:]
        n.parents, n.scores = select_parents(chat, n, cands, n.goal or goal, cfg)


def summarize(chat, nodes, cfg: CWConfig) -> None:
    """The LLMSUM half: node summaries, then dependency summaries in topological (index) order."""
    if not cfg.summarize:
        return
    outs = _pmap(lambda n: chat.ask(NODE_PROMPT.format(
        thought=n.thought[:1500], action=n.action[:1500], observation=n.observation[:3000]),
        max_tokens=120), nodes, cfg.max_workers)
    for n, s in zip(nodes, outs):
        n.summary = s
    for n in nodes:
        pars = "\n".join(nodes[p].dependency_summary or nodes[p].summary for p in n.parents)
        n.dependency_summary = chat.ask(
            DEPSUM_PROMPT.format(parents=pars or "(none -- this is a root step)",
                                 summary=n.summary), max_tokens=140) if pars else n.summary


# ---------------------------------------------------------------- ancestry + weaving


def ancestry(nodes, anchor: int, W: int) -> tuple:
    """Algorithm 1 steps 7-8: BFS upward through parent edges, capped at W nodes."""
    A, queue = [], [anchor]
    seen = set()
    while queue and len(A) < W:
        cur = queue.pop(0)
        if cur in seen:
            continue
        seen.add(cur)
        A.append(cur)
        for p in nodes[cur].parents:
            if p not in seen:
                queue.append(p)
    return tuple(sorted(A))


def weave(nodes, A, anchor: int, include_depsum: bool = True) -> str:
    """Algorithm 1 step 9: ancestors in full, everything else with its observation replaced."""
    keep = set(A)
    out = []
    if include_depsum and nodes and nodes[anchor].dependency_summary:
        out.append("[dependency summary] " + nodes[anchor].dependency_summary + "\n")
    for n in nodes:
        head = "step %d | %s\naction: %s" % (n.idx, n.validation, n.action)
        if n.idx in keep:
            out.append(head + "\nobservation: " + (n.observation or ""))
        else:
            line = head + "\nobservation: " + PLACEHOLDER
            if n.summary:
                line += "\nsummary: " + n.summary
            out.append(line)
    return "\n\n".join(out)


def build(rows, goal: str, cfg: CWConfig, anchor: "int | None" = None) -> CWResult:
    """One full pass of Algorithm 1 over a history, returning the context it would keep."""
    chat = Chat(cfg.base_url, cfg.model, cfg.api_key, cfg.timeout)
    t0 = time.time()
    nodes = extract_nodes(rows, cfg.validation)
    if not nodes:
        return CWResult(context="", nodes=[], ancestry=())
    build_graph(chat, nodes, goal, cfg)
    summarize(chat, nodes, cfg)
    t1 = time.time()
    k = len(nodes) - 1 if anchor is None else anchor
    warm = len(nodes) <= cfg.W
    A = tuple(range(len(nodes))) if warm else ancestry(nodes, k, cfg.W)
    text = weave(nodes, A, k)
    return CWResult(context=text, nodes=nodes, ancestry=A, chars=len(text),
                    tokens=max(1, len(text) // 4), llm_calls=chat.calls, llm_tokens=chat.tokens,
                    graph_ms=int(1000 * (t1 - t0)), select_ms=int(1000 * (time.time() - t1)),
                    warmup=warm)


# ---------------------------------------------------------------- live-harness node extraction


def _dump(e) -> dict:
    if hasattr(e, "model_dump"):
        return e.model_dump(mode="json")
    return dict(e) if isinstance(e, dict) else {}


def flatten(x) -> str:
    """Flatten harness content (str / {text} / list / object) to text."""
    if x is None:
        return ""
    if isinstance(x, str):
        return x
    if isinstance(x, (list, tuple)):
        return "\n".join(flatten(i) for i in x)
    if isinstance(x, dict):
        for k in ("text", "content", "output", "command", "arguments", "result"):
            if k in x:
                return flatten(x[k])
        return json.dumps(x, ensure_ascii=False)[:4000]
    for k in ("text", "content", "output", "command", "result"):
        v = getattr(x, k, None)
        if v is not None:
            return flatten(v)
    return str(x)[:4000]


#: The live-harness event classes this file understands.
_EVENT_KINDS = ("ActionEvent", "ObservationEvent", "MessageEvent")


def _kind_of(ev, d) -> str:
    """Class name, falling back to a declared ``_kind``.

    Duck-typing has to be real duck-typing: keying only on ``type(ev).__name__`` meant any stand-in
    whose class was not literally named ``ActionEvent`` was invisible, which is how the first
    version of this extraction shipped looking correct and finding nothing.
    """
    k = type(ev).__name__
    if k in _EVENT_KINDS:
        return k
    raw = ev.get("_kind") if isinstance(ev, dict) else getattr(ev, "_kind", None)
    if raw is None:
        raw = d.get("_kind")
    return str(raw or k)


def pairs_from_events(events) -> list:
    """-> [(action_index, obs_index, action_dump, obs_dump)] over a live OpenHands view.

    The SDK's events are ``ActionEvent`` / ``ObservationEvent`` objects carrying ``tool_call``,
    ``tool_call_id``, ``thought`` and ``tool_name``.  There is NO ``tool_calls`` list on them, and
    reading for one produced zero nodes -- which made the condenser a silent no-op: every cell ran
    with the full history and passed, and that would have been reported as ContextWeaver winning.
    This lives here, not in the binding, so it can be tested without the SDK installed.

    Pairing is by ``tool_call_id`` with a positional fallback -- the same rule
    ``tracepack.condenser.oh_rows`` uses for the archived corpus.
    """
    out, pending = [], {}
    for i, ev in enumerate(events):
        d = _dump(ev)
        kind = _kind_of(ev, d)
        if kind == "ActionEvent":
            pending[d.get("tool_call_id") or ("#%d" % i)] = (i, d)
        elif kind == "ObservationEvent":
            key = d.get("tool_call_id")
            if key in pending:
                j, ad = pending.pop(key)
            elif pending:
                j, ad = pending.pop(list(pending)[-1])
            else:
                continue
            out.append((j, i, ad, d))
    out.sort()
    return out


def nodes_from_events(events) -> list:
    """Live view -> ContextWeaver nodes, each carrying the instruction in force when it ran."""
    goal, goals = "", {}
    for i, ev in enumerate(events):
        d = _dump(ev)
        if _kind_of(ev, d) == "MessageEvent":
            msg = d.get("llm_message") or {}
            if msg.get("role") == "user":
                t = flatten(msg.get("content"))
                if t.strip():
                    goal = t
        goals[i] = goal
    nodes = []
    for idx, (i, j, ad, od) in enumerate(pairs_from_events(events)):
        call = ad.get("tool_call") or {}
        fn = (call.get("function") or {}) if isinstance(call, dict) else {}
        args = fn.get("arguments")
        if not isinstance(args, str):
            args = json.dumps(ad.get("action") or {}, ensure_ascii=False)
        action = "%s %s" % (ad.get("tool_name") or "tool", args)
        obs = flatten(od.get("observation") or od.get("content") or od)
        nodes.append(Node(idx=idx, thought=flatten(ad.get("thought")), action=action,
                          observation=obs, validation=_validation(obs), goal=goals.get(i, ""),
                          event_ids=(str(ad.get("id") or i), str(od.get("id") or j))))
    mark_superseded(nodes)
    return nodes


def last_user_text(events) -> str:
    """The instruction in force right now, read off the view -- never a later one."""
    for ev in reversed(events):
        d = _dump(ev)
        if _kind_of(ev, d) == "MessageEvent":
            msg = d.get("llm_message") or {}
            if msg.get("role") == "user":
                t = flatten(msg.get("content"))
                if t.strip():
                    return t
    return "continue the task"


# ---------------------------------------------------------------- T3: the positive control


PLANT_ROWS = [
    {"tracepack_format": "openhands"},
    {"id": "p0", "role": "assistant", "content": "First list the repository.",
     "tool_calls": [{"function": {"name": "terminal", "arguments": '{"command": "ls -R"}'}}]},
    {"id": "p1", "role": "tool", "content": "core/api.py\nsvc/config.py\ntests/test_api.py"},
    {"id": "p2", "role": "assistant", "content": "Run the one-shot record; it will not repeat.",
     "tool_calls": [{"function": {"name": "terminal", "arguments": '{"command": "bash oneshot/os.sh"}'}}]},
    {"id": "p3", "role": "tool",
     "content": "edge certificate issued\nid: 4b7d21c9\nrenewal blocked: OCSP stapling unsupported "
                "by the edge build 2.14.0\none-time issuance; the issuer will not repeat it"},
    {"id": "p4", "role": "assistant", "content": "Check the test suite.",
     "tool_calls": [{"function": {"name": "terminal", "arguments": '{"command": "python -m pytest -q"}'}}]},
    {"id": "p5", "role": "tool", "content": "12 passed in 0.4s"},
    {"id": "p6", "role": "assistant", "content": "Print the config.",
     "tool_calls": [{"function": {"name": "terminal", "arguments": '{"command": "cat svc/config.py"}'}}]},
    {"id": "p7", "role": "tool", "content": "PORT = 8080\nRETRIES = 3\nTIMEOUT = 30"},
    {"id": "p8", "role": "assistant", "content": "Register the material with the id it printed.",
     "tool_calls": [{"function": {"name": "terminal",
                                  "arguments": '{"command": "python scripts/ops.py tls-register --id 4b7d21c9"}'}}]},
    {"id": "p9", "role": "tool", "content": "tls material 4b7d21c9 installed on the edge for svc"},
]
#: In PLANT_ROWS the true parent of the last node (index 4, the register call) is node 1 -- the
#: one-shot record, whose output supplied the id.  Nodes 2 and 3 are unrelated.
PLANT_CHILD, PLANT_TRUE_PARENT = 4, 1


def plant_control(cfg: CWConfig, trials: int = 1) -> dict:
    """Does the analyzer put the TRUE parent in the top-m on a trace where we know the answer?

    This is the gate PLAN_baselines §6-T3 requires.  A ContextWeaver run that fails it is an
    implementation failure, and its comparison numbers must not be quoted -- exactly the judgement
    this project already passed on its own graph arm when the planted-row control failed there.
    """
    chat = Chat(cfg.base_url, cfg.model, cfg.api_key, cfg.timeout)
    hits, ranks, details = 0, [], []
    for _ in range(max(1, trials)):
        nodes = extract_nodes(PLANT_ROWS)
        mark_superseded(nodes)
        child = nodes[PLANT_CHILD]
        cands = [c for c in nodes[:child.idx] if c.validation not in ("failed", "superseded")]
        top, scores = select_parents(chat, child, cands, "register the tls material and document it", cfg)
        ordered = sorted(scores, key=lambda i: (-scores[i], -i))
        rank = ordered.index(PLANT_TRUE_PARENT) + 1 if PLANT_TRUE_PARENT in ordered else 0
        hits += PLANT_TRUE_PARENT in top
        ranks.append(rank)
        details.append({"top": list(top), "scores": scores, "rank_of_true": rank})
    out = {"trials": max(1, trials), "hits": hits, "rate": hits / float(max(1, trials)),
           "ranks": ranks, "m": cfg.m, "scorer": cfg.scorer,
           "llm_calls": chat.calls, "llm_tokens": chat.tokens, "details": details}
    out["pass"] = out["rate"] >= 0.8
    return out


# ---------------------------------------------------------------- selfcheck (no LLM, no GPU)


class _StubChat:
    """A scorer with a script, so the graph machinery can be tested without a model.

    ``plan`` maps (child_idx, parent_idx) -> score; anything unlisted scores 0.
    """

    def __init__(self, plan=None, fixed=None):
        self.plan, self.fixed, self.calls, self.tokens = plan or {}, fixed, 0, 0
        self.seen = []
        self.prompts = []      # kept so a test can assert what the analyzer was NOT shown

    def ask(self, prompt, max_tokens=200, temperature=0.0):
        self.calls += 1
        self.prompts.append(prompt)
        self.tokens += len(prompt) // 4
        if self.fixed is not None:
            return str(self.fixed)
        m = re.search(r"CURRENT OPERATION:\s*\nstep (\d+)", prompt)
        p = re.search(r"PREVIOUS OPERATION:\s*\nstep (\d+)", prompt)
        if m and p:
            self.seen.append((int(m.group(1)), int(p.group(1))))
            return str(self.plan.get((int(m.group(1)), int(p.group(1))), 0))
        return "summary"


def _selfcheck() -> None:
    nodes = extract_nodes(PLANT_ROWS)
    assert len(nodes) == 5, len(nodes)
    assert "4b7d21c9" in nodes[1].observation and "OCSP" in nodes[1].observation
    assert nodes[4].action.count("tls-register") == 1
    assert [n.validation for n in nodes] == ["passed"] * 5, [n.validation for n in nodes]

    # a failing observation must be excluded from the candidate set
    bad = extract_nodes(PLANT_ROWS)
    bad[2].observation = "Traceback (most recent call last): RuntimeError"
    bad[2].validation = _validation(bad[2].observation)
    assert bad[2].validation == "failed"

    cfg = CWConfig(W=3, m=2)

    # 1. the analyzer's top-m is what the ancestry walks
    chat = _StubChat({(4, 1): 9, (4, 3): 2, (4, 0): 0, (4, 2): 0,
                      (3, 0): 7, (2, 0): 1, (1, 0): 4})
    build_graph(chat, nodes, "register the tls material", cfg)
    assert nodes[4].parents[0] == 1, nodes[4].parents
    A = ancestry(nodes, 4, cfg.W)
    assert 1 in A and 4 in A, A
    assert len(A) <= cfg.W

    # 2. weaving keeps ancestors verbatim and replaces everything else
    text = weave(nodes, A, 4, include_depsum=False)
    assert "renewal blocked: OCSP stapling unsupported by the edge build 2.14.0" in text
    assert PLACEHOLDER in text, "non-ancestor observations were not replaced"
    for i in range(5):
        if i not in A:
            assert nodes[i].observation.split("\n")[0] not in text or not nodes[i].observation, i

    # 3. warmup keeps everything, so the placeholder must NOT appear
    full = weave(nodes, tuple(range(5)), 4, include_depsum=False)
    assert PLACEHOLDER not in full and "PORT = 8080" in full

    # 4. FAULT INJECTION -- an analyzer that scores everything 0 must produce an EMPTY ancestry
    #    beyond the anchor, not a quietly-plausible one.  Without this, a dead scorer would look
    #    like "ContextWeaver kept the last few steps", which is a different method
    #    ([[verify-your-checker]]).
    dead = extract_nodes(PLANT_ROWS)
    build_graph(_StubChat(fixed=0), dead, "goal", cfg)
    assert all(not n.parents for n in dead), "a zero-scoring analyzer still produced parents"
    assert ancestry(dead, 4, cfg.W) == (4,), ancestry(dead, 4, cfg.W)
    assert "renewal blocked" not in weave(dead, ancestry(dead, 4, cfg.W), 4, include_depsum=False)

    # 5. the pairwise scorer really is one call per candidate
    c2 = _StubChat({})
    n2 = extract_nodes(PLANT_ROWS)
    select_parents(c2, n2[4], n2[:4], "goal", CWConfig(W=5, m=3, max_workers=4))
    assert c2.calls == 4, c2.calls

    print("contextweaver selfcheck ok  nodes=%d ancestry=%s parents(4)=%s"
          % (len(nodes), A, nodes[4].parents))


if __name__ == "__main__":
    _selfcheck()
