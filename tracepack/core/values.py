"""tracepack.core.values -- the ONE definition of "a value" (paths, file:line refs, hashes,
numbers, versions) shared by the dataset builder and the adapter's value-source edges.

Moved verbatim out of eval/datasets.py (phase 2, WP3a) so the adapter can link "this value came
from that tool output" with exactly the extractor the items were built with -- and so that this
shared origin is visible rather than a silent coincidence (PLAN_phase2.md §3.5 WP3 結局 3d: the
held-out items of WP5a are the honest readout for edges built this way).
"""
from __future__ import annotations

import re

from .textmatch import boundary_clean as _boundary_clean

__all__ = ["PATTERNS", "MIN_V", "MAX_V", "plausible_path", "extract_values"]

MIN_V, MAX_V = 6, 120

# value extractors -- deliberately narrow so gold is unambiguous and checkable by string match
PATTERNS = [
    ("path", re.compile(r"(?:[\w.\-]+/){1,}[\w.\-]+\.(?:py|md|sh|json|jsonl|txt|csv|yaml|yml|tsv|png)")),
    ("fileline", re.compile(r"[\w./\-]+\.(?:py|sh|md|json|txt):\d{1,6}")),
    ("hash", re.compile(r"\b[0-9a-f]{8,40}\b")),
    ("number", re.compile(r"(?<![\w.])\d{3,10}(?:\.\d+)?(?![\w.])")),
    ("version", re.compile(r"\b\d+\.\d+\.\d+\b")),
]

_EXT_RE = re.compile(r"\.(?:py|md|sh|json|jsonl|txt|csv|yaml|yml|tsv|png)$")


def plausible_path(v):
    """Reject slash-joined ENUMERATIONS masquerading as paths.

    Found on the second audit pass (2026-09-01): prose lists files as
    `sp50_artifact_extract.py/sp50_artifact_eval.py/build_artifact_review.py/build_review_html.py`
    -- very common in the Chinese notes in these traces -- and the path pattern happily matched the
    whole run.  The value is real and findable, so no leak check caught it, but its "directory" is
    three filenames, which makes the distractor an absurd string and gives the pair a huge shared
    prefix.  A real path has at most one extension, on the last segment.
    """
    segs = v.split("/")
    return not any(_EXT_RE.search(x) for x in segs[:-1])


def extract_values(text):
    out = []
    seen = set()
    text = text or ""
    for typ, pat in PATTERNS:
        for m in pat.finditer(text):
            v = m.group(0)
            if not (MIN_V <= len(v) <= MAX_V) or v in seen:
                continue
            if not _boundary_clean(text, m.start(), m.end()):
                continue
            if typ == "path" and not plausible_path(v):
                continue
            seen.add(v)
            out.append((typ, v))
    return out
