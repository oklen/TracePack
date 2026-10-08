"""tracepack.core.excerpt -- query-driven line excerpt of a large event (PLAN_phase2.md WP2).

Why this exists.  Gold events are tool results: median 432 tokens, upper quartile 1,232, and 57 of
the 190 items' gold events are over 1,024 tokens.  Under a 2,048 budget the seeds leave ~1,000
tokens, so the assembler's "whole event or nothing" rule (§3.4) drops the one event that holds
the answer in 36/190 items (RESULTS_tracepack_edges2.md §1.6, E bucket).  A head-truncation was
tried and LOST (§1.7a): the gold literal sits in the first quarter of its event in only 84 items.

What this is NOT.  It is not truncation.  An excerpt is a *declared representation* ("excerpt")
with provenance (which lines, out of how many) written into the manifest, chosen only when the
whole event does not fit, and never for a required event that fits.  The reader is told it is an
excerpt.  §3.4's ban is on *silent* cutting; the manifest makes this one loud.

What it may look at -- the gold-blindness contract (#12).  Line selection sees three things:
the QUERY, the event TEXT, and the texts of events that DEPEND ON this one (they quote it, so
the quoted lines are evidence someone already used).  It never sees the item, the gold, the
distractors or the required-source list; the function signature has no place to pass them.
`tests/test_phase2.py::Contract12` injects a gold that lives on a line with no query overlap and
asserts it is NOT selected.

Scoring is deliberately dumb and auditable:
  +1 per distinct query term found in the line (case-insensitive, tokens >= 3 chars, stop words
     dropped, path-like tokens also contribute their basename)
  +3 if the stripped line (>= 20 chars) occurs verbatim in a quoting event's text
  line 0 is always kept (a tool result's first line names what it is)
Selected lines are expanded by `context` lines on each side, ranges merged, emitted in file
order with a one-line header and `...` between gaps.  Two tiers are offered to the packer:
context=2, then context=0 capped at `max_lines`.  Cost is measured by the caller-supplied
`cost_fn`, so this module never owns a tokenizer either (same rule as the assembler).
"""
from __future__ import annotations

import re
from typing import Callable, Sequence

__all__ = ["query_terms", "score_lines", "select_ranges", "render", "make_excerpt_fn",
           "EXCERPT_KIND"]

EXCERPT_KIND = "excerpt"

_STOP = frozenset("""
the and for are was were with this that from into which what when where who why how did does
has have had not but its also than then them they their there these those been being will would
could should may might must can about after before over under again all any each few more most
other some such only own same too very just don now use used using into out off per via
""".split())

_TOKEN = re.compile(r"[A-Za-z0-9_][A-Za-z0-9_./\-:@#+]*")
#: quoted-line match needs at least this many characters, else "ok" would match everything
_QUOTE_MIN = 20


def query_terms(query: str) -> frozenset[str]:
    """Lower-cased content tokens of the query, plus the basename of every path-like token."""
    out = set()
    for tok in _TOKEN.findall(query or ""):
        t = tok.lower().strip(".:,;/")
        if len(t) < 3 or t in _STOP:
            continue
        out.add(t)
        if "/" in t:
            base = t.rsplit("/", 1)[-1]
            if len(base) >= 3 and base not in _STOP:
                out.add(base)
    return frozenset(out)


def score_lines(lines: Sequence[str], terms: frozenset[str],
                quoting_texts: Sequence[str] = ()) -> list[int]:
    """One integer score per line; see the module docstring for the rule."""
    quotes = [q for q in quoting_texts if q]
    scores = []
    for i, line in enumerate(lines):
        low = line.lower()
        s = sum(1 for t in terms if t in low)
        stripped = line.strip()
        if len(stripped) >= _QUOTE_MIN and any(stripped in q for q in quotes):
            s += 3
        if i == 0:
            s = max(s, 1)                      # the header line is always kept
        scores.append(s)
    return scores


def select_ranges(scores: Sequence[int], context: int, max_lines: int | None = None
                  ) -> list[tuple[int, int]]:
    """Inclusive (start, end) line ranges around every positively scored line, merged.

    With ``max_lines`` the highest-scoring lines are taken first (ties by position) until the
    cap is reached, *then* expanded -- so a tight tier still keeps the best lines, not the first.
    """
    keep = [i for i, s in enumerate(scores) if s > 0]
    if max_lines is not None and len(keep) > max_lines:
        keep = sorted(sorted(keep, key=lambda i: (-scores[i], i))[:max_lines])
    n = len(scores)
    ranges: list[tuple[int, int]] = []
    for i in keep:
        a, b = max(0, i - context), min(n - 1, i + context)
        if ranges and a <= ranges[-1][1] + 1:
            ranges[-1] = (ranges[-1][0], max(ranges[-1][1], b))
        else:
            ranges.append((a, b))
    return ranges


def render(lines: Sequence[str], ranges: Sequence[tuple[int, int]]) -> tuple[str, str]:
    """``(text, label)``: the excerpt body with a header line, and the manifest label."""
    n = len(lines)
    spans = ",".join("%d-%d" % (a + 1, b + 1) if a != b else "%d" % (a + 1) for a, b in ranges)
    label = "%s[%s/%d]" % (EXCERPT_KIND, spans, n)
    parts = ["[excerpt: lines %s of %d]" % (spans, n)]
    prev_end = -1
    for a, b in ranges:
        if prev_end >= 0 and a > prev_end + 1:
            parts.append("...")
        parts.extend(lines[a:b + 1])
        prev_end = b
    return "\n".join(parts), label


def make_excerpt_fn(cost_fn: Callable[[str], int], *, min_lines: int = 4,
                    context: int = 2, max_lines: int = 12,
                    quoting_of: Callable[[object], Sequence[str]] | None = None):
    """Build the ``excerpt_fn(event, query, quoting_texts) -> [(label, text, cost), ...]`` the
    assembler consumes.  Returns ``[]`` for events too short to be worth excerpting, and never
    returns an excerpt at least as long as the whole event (then the whole event should be served).

    ``quoting_of(event) -> [text, ...]`` lets the caller add the texts of every event that
    DEPENDS ON this one according to the *graph* (the assembler only knows the closure's steps,
    and a quoting child is usually not in the closure -- edges point from child to parent).
    Both sources are unioned; both are graph structure, neither is item-specific.
    """
    if not callable(cost_fn):
        raise TypeError("cost_fn must be callable")
    if quoting_of is not None and not callable(quoting_of):
        raise TypeError("quoting_of must be callable or None")

    def excerpt_fn(event, query: str, quoting_texts: Sequence[str] = ()) -> list[tuple[str, str, int]]:
        text = event.text or ""
        lines = text.split("\n")
        if len(lines) < min_lines:
            return []
        quotes = list(quoting_texts or ())
        if quoting_of is not None:
            quotes.extend(quoting_of(event) or ())
        scores = score_lines(lines, query_terms(query), quotes)
        whole = cost_fn(text)
        out: list[tuple[str, str, int]] = []
        seen: set[str] = set()
        for ctx, cap in ((context, None), (0, max_lines)):
            ranges = select_ranges(scores, ctx, cap)
            if not ranges:
                continue
            body, label = render(lines, ranges)
            if body in seen:
                continue
            seen.add(body)
            cost = cost_fn(body)
            if cost >= whole:
                continue
            out.append((label, body, cost))
        return out

    return excerpt_fn


def _selfcheck() -> None:
    from dataclasses import dataclass

    @dataclass
    class Ev:
        text: str

    cost = lambda s: max(1, len(s) // 4)
    body = "\n".join(["$ ls /srv/app/config", "a.yaml", "b.yaml", "deploy_target: ap-southeast-1",
                      "replicas: 3", "secret: hunter2", "z.log"] + ["noise %d" % i for i in range(30)])
    fn = make_excerpt_fn(cost)
    tiers = fn(Ev(body), "what is the deploy target after the config change")
    assert tiers, "an excerpt must be offered for a 37-line event"
    label, text, c = tiers[0]
    assert label.startswith("excerpt[") and text.startswith("[excerpt: lines"), (label, text[:40])
    assert "deploy_target: ap-southeast-1" in text and "$ ls /srv/app/config" in text
    assert "noise 25" not in text and c < cost(body)
    # gold-blindness: a value on a line with no query overlap is not pulled in
    assert "hunter2" not in fn(Ev(body), "which yaml files were listed")[0][1]
    # a quoting child pulls its quoted line in even with zero query overlap
    quoted = fn(Ev(body), "unrelated question about the weather",
                ["as seen in `deploy_target: ap-southeast-1` earlier"])[0][1]
    assert "deploy_target: ap-southeast-1" in quoted
    # ... but a quoted line shorter than _QUOTE_MIN does not count (would match everything)
    assert "hunter2" not in fn(Ev(body), "unrelated question about the weather",
                                ["... secret: hunter2 ..."])[0][1]
    # tiers shrink and never reach the whole event
    assert all(t[2] < cost(body) for t in tiers)
    assert fn(Ev("one\ntwo"), "one") == []
    # the label is exact provenance: every excerpt line is a real line of the event
    for _, t, _ in tiers:
        for ln in t.split("\n")[1:]:
            assert ln == "..." or ln in body.split("\n"), ln
    print("excerpt selfcheck OK")


if __name__ == "__main__":
    _selfcheck()
