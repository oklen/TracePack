"""tracepack.userwords -- pick the user's own sentences that state facts, without a model.

After a compaction the summary is all that is left of what the user said, and summaries paraphrase:
numbers get rounded, names dropped, old values overwritten by new ones. The user's words are the one
thing in a session that cannot be re-read from anywhere else, so TracePack puts some of them back,
verbatim.

Which ones: sentences that carry facts (numbers, dates, names, identifiers, first-person statements,
preferences, plans, decisions, constraints such as "never" / "must"). Questions, thanks and chit-chat
score low. Each sentence is scored on its own, so the one fact inside a long message can be kept
without the rest; the choice is by facts per token, and the result is shown in the order it was said,
dated, with "…" where a message was shortened.

Stdlib only and deterministic.
"""
from __future__ import annotations

import re
import time

MIN_SCORE = 1.5
MAX_UNIT_CHARS = 600
LONG_MESSAGE_CHARS = 1500        # past this a message is mostly pasted material (a story, a log, a document)

_SPLIT = re.compile(r"\n+|(?<=[.!?])\s+(?=[\"'(\[]?[A-Z0-9])")
_TAG_LINE = re.compile(r"^\s*\[[^\]\n]{0,80}\]\s*$")           # a bracketed tag alone on a line
_NUMBER = re.compile(r"(?<![A-Za-z])[$€£¥]?\d[\d,.:/%x-]*")
_MONTHS = re.compile(r"\b(jan(uary)?|feb(ruary)?|mar(ch)?|apr(il)?|may|june?|july?|aug(ust)?|sept?(ember)?|oct(ober)?|"
                     r"nov(ember)?|dec(ember)?|monday|tuesday|wednesday|thursday|friday|saturday|sunday|weekend|"
                     r"yesterday|tomorrow|tonight|last (week|month|year)|next (week|month|year)|ago)\b", re.I)
_FIRST = re.compile(r"\b(i|i'm|i've|i'd|i'll|my|mine|we|we're|we've|our|ours)\b", re.I)
_CUE = re.compile(r"\b(prefer\w*|favou?rite|like[sd]?|love[sd]?|hate[sd]?|enjoy\w*|want\w*|plan\w*|going to|decided|"
                  r"switch\w*|moved|bought|purchased|got|started|finished|joined|left|sold|booked|signed up|adopted|"
                  r"visited|named|called|born|married|graduated|work(s|ed|ing)? (at|as|for)|live[sd]?|living|"
                  r"must|never|always|don't|do not|should|need to|remember|note that|make sure|important|deadline|"
                  r"instead|allergic|budget|cost|paid|spent|earn\w*|salary|price|owe\w*)\b", re.I)
_IDENT = re.compile(r"[\w.-]+/[\w./-]+|\b\w+\.(py|js|ts|tsx|go|rs|java|rb|json|ya?ml|toml|md|sql|sh|txt|csv)\b|"
                    r"\b[a-z0-9]+_[a-z0-9_]+\b|(?<!\w)--?[a-z][\w-]+|\bv?\d+\.\d+(\.\d+)?\b|`[^`]{2,60}`")
_PROPER = re.compile(r"(?<=[a-z,;:)] )[A-Z][a-zA-Z]+")
_MY = re.compile(r"\b(my|our) (?:new |old |own |current |first |last |best )?[a-z]{3,}", re.I)
_URL = re.compile(r"https?://\S+|www\.\S+")
_REQUEST = re.compile(r"^\W*(please |can you|could you|would you|will you|write|explain|describe|list|give|show|tell|"
                      r"create|generate|make|help|pretend|act as|translate|summari[sz]e|rewrite|provide|suggest|"
                      r"recommend|what|how|why|which|who|where|when|is|are|do|does)\b", re.I)


def split_units(text: str) -> list:
    """Sentences and lines of a message, minus bracketed tags on their own line."""
    out = []
    for piece in _SPLIT.split(text or ""):
        s = piece.strip()
        if not s or _TAG_LINE.match(s):
            continue
        if len(s) > MAX_UNIT_CHARS:
            s = s[:MAX_UNIT_CHARS].rstrip() + " …"
        out.append(s)
    return out


def fact_score(s: str) -> float:
    """How much a sentence looks like a lasting fact (0 for questions-only, thanks and chit-chat)."""
    s = _URL.sub(" ", s)
    letters = sum(c.isalpha() for c in s)
    if letters < 8:
        return 0.0
    first = bool(_FIRST.search(s))
    score = 1.5 * min(4, len(_NUMBER.findall(s)))
    score += 1.0 * min(2, len(_MONTHS.findall(s)))
    score += 1.0 if first else 0.0
    score += 0.5 * min(2, len(_MY.findall(s)))
    score += 1.0 * min(3, len(_CUE.findall(s)))
    score += 1.0 * min(3, len(_IDENT.findall(s)))
    score += 0.8 * min(4, len(_PROPER.findall(s)))
    if s.rstrip().endswith("?"):
        score *= 0.6
    elif not first and _REQUEST.match(s):          # an instruction to the assistant, not a fact
        score *= 0.5
    if letters < 0.5 * len(s.strip()):             # mostly symbols: pasted code or logs
        score *= 0.3
    return score


def _tokens(text: str) -> int:
    return (len(text) + 3) // 4 if text else 0


def _date(ms: int) -> str:
    return time.strftime("%Y-%m-%d %H:%M", time.localtime(ms / 1000.0)) if ms else ""


def _line(event_id: str) -> str:
    parts = event_id.split(":")
    return str(int(parts[1])) if len(parts) >= 2 and parts[1].isdigit() else ""


HEAD = ("TracePack: the user's own words from before the compaction, verbatim. These sentences were picked "
        "because they state facts (names, numbers, dates, preferences, decisions, constraints); \"…\" marks "
        "where a message was shortened. The summary above may have paraphrased them.")


def label(ev) -> str:
    bits = ["user said"]
    if ev.timestamp:
        bits.append(_date(ev.timestamp))
    ln = _line(ev.event_id)
    if ln:
        bits.append("L" + ln)
    return " · ".join(bits)


def pick(events, budget: int, max_chars: int):
    """Choose sentences from user `events` (chronological) within `budget` tokens and `max_chars` of
    rendered text. -> (text, [event ids used], n_sentences); text is "" when nothing qualifies."""
    if budget <= 0 or max_chars <= 0:
        return "", [], 0
    cands, seen, n_units = [], set(), {}
    for pos, ev in enumerate(events):
        units = split_units(ev.text or "")
        n_units[pos] = len(units)
        for j, u in enumerate(units):
            norm = " ".join(u.lower().split())
            if len(norm) < 12 or norm in seen:
                continue
            seen.add(norm)
            sc = fact_score(u)
            if len(ev.text or "") > LONG_MESSAGE_CHARS:
                sc *= 0.5
            if sc < MIN_SCORE:
                continue
            cost = _tokens(u) + 1
            cands.append((sc / cost ** 0.6, pos, j, u, cost))
    cands.sort(key=lambda c: (-c[0], -c[1], c[2]))   # facts per token, then the more recent
    head_cost = _tokens(HEAD) + 2
    used, chars = head_cost, len(HEAD) + 2
    chosen = {}
    for dens, pos, j, u, cost in cands:
        ev = events[pos]
        extra = 0 if pos in chosen else _tokens("[u99] " + label(ev)) + 2
        extra_chars = 0 if pos in chosen else len("[u99] " + label(ev)) + 3
        if used + cost + extra > budget or chars + len(u) + 3 + extra_chars > max_chars:
            continue
        chosen.setdefault(pos, []).append((j, u))
        used += cost + extra
        chars += len(u) + 3 + extra_chars
    if not chosen:
        return "", [], 0
    blocks = []
    for n, pos in enumerate(sorted(chosen), 1):
        parts = sorted(chosen[pos])
        body = parts[0][1]
        for (pj, _pu), (j, u) in zip(parts, parts[1:]):
            body += (" " if j == pj + 1 else " … ") + u
        if parts[0][0] > 0:
            body = "… " + body
        if parts[-1][0] < n_units[pos] - 1:
            body += " …"
        blocks.append("[u%d] %s\n%s" % (n, label(events[pos]), body))
    text = HEAD + "\n\n" + "\n\n".join(blocks)
    return text, [events[p].event_id for p in sorted(chosen)], sum(len(v) for v in chosen.values())
