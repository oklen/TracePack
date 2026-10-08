"""tracepack.core.textmatch -- ONE definition of "this value occurs here".

Why this module exists (red-team finding #3, 2026-09-01): the dataset builder and the profiler
each carried their own word-boundary matcher and they disagreed.  ``interventions._wb`` put ``/``
in the boundary class, so ``home/b/x.py`` -- which is how the path extractor emits a SUFFIX of an
absolute path -- never matched inside ``/home/b/x.py``.  The runtime matcher was therefore blind
to 68% of the path golds while the builder considered them present: every gold-in-context check,
every redaction arm and every stale-conflict label silently read the wrong answer.

The rule below is the builder's, and it is now the only one:

  boundary = ``[\\w.-]``      word chars, dot, dash -- NOT ``/``.

``\\b`` is wrong for these values (paths, ``file.py:120`` locations, hashes) because their edges
are punctuation.  Excluding ``.`` and ``-`` keeps ``4711`` out of ``14711`` and ``4711.2``;
admitting ``/`` at the edges keeps a path suffix findable in the absolute path it came from.

Anything that asks "is this value in this text" MUST come through here.
"""
from __future__ import annotations

import re

#: characters that may NOT sit immediately left/right of a match
BOUNDARY = re.compile(r"[\w.-]")

_CACHE: dict = {}


def wb(value: str):
    """Compiled boundary-anchored matcher for a literal ``value`` (cached)."""
    r = _CACHE.get(value)
    if r is None:
        r = re.compile(r"(?<![\w.-])" + re.escape(value) + r"(?![\w.-])")
        if len(_CACHE) < 4096:
            _CACHE[value] = r
    return r


def boundary_clean(text: str, start: int, end: int) -> bool:
    """True when ``text[start:end]`` is a whole token, not a fragment of a longer one.

    The extractor must use the SAME boundary as the matcher, or it harvests values that are
    unanswerable by construction: ``029023746701846966`` out of the float
    ``0.029023746701846966``, the first 8 chars of a UUID, ``x.json`` out of ``x.jsonl``.
    """
    if start > 0 and BOUNDARY.match(text[start - 1]):
        return False
    if end < len(text) and BOUNDARY.match(text[end]):
        return False
    return True


def contains_value(text: str, value: str) -> bool:
    """True when ``value`` occurs in ``text`` as a standalone value, not as a substring."""
    if not text or not value:
        return False
    return wb(value).search(text) is not None


def _selfcheck():
    # the exact regression that motivated the module: a path suffix inside its absolute path
    assert contains_value("cd /home/b/x.py now", "home/b/x.py")
    assert contains_value("wrote a/b/x.py", "a/b/x.py")
    # fragments must stay invisible
    assert contains_value("rows=4711003 done", "4711003")
    assert not contains_value("rows=14711003 done", "4711003")
    assert not contains_value("val 4711003.2", "4711003")
    assert not contains_value("x.jsonl", "x.json")
    assert not contains_value("", "a") and not contains_value("a", "")
    # extractor/matcher agreement
    t = "0.029023746701846966"
    assert not boundary_clean(t, 2, len(t))
    assert boundary_clean("hash 029023746701 ok", 5, 17)
    print("textmatch.py selfcheck OK")


if __name__ == "__main__":
    _selfcheck()
