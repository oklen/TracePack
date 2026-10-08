"""tracepack.redact -- mask credential-shaped strings before text is handed back to a model.

Recall is verbatim by design, so anything the agent once printed (an `.env` file, a curl with a token)
could come back. Values that look like secrets are replaced with `[redacted]`; the surrounding key
name is kept so the record still makes sense. Off with `TRACEPACK_REDACT=0`.
"""
from __future__ import annotations

import os
import re

_PATTERNS = [
    re.compile(r"-----BEGIN [A-Z ]*PRIVATE KEY-----.*?-----END [A-Z ]*PRIVATE KEY-----", re.S),
    re.compile(r"\b(?:sk|rk|pk)-(?:ant-|proj-|live-|test-)?[A-Za-z0-9_\-]{20,}"),     # OpenAI / Anthropic / Stripe style
    re.compile(r"\bgh[pousr]_[A-Za-z0-9]{30,}"),                                      # GitHub
    re.compile(r"\bgithub_pat_[A-Za-z0-9_]{40,}"),
    re.compile(r"\bxox[abprs]-[A-Za-z0-9\-]{10,}"),                                   # Slack
    re.compile(r"\bAKIA[0-9A-Z]{16}\b"),                                              # AWS access key id
    re.compile(r"\bAIza[0-9A-Za-z_\-]{35}\b"),                                        # Google API key
    re.compile(r"\bhf_[A-Za-z0-9]{30,}\b"),                                           # Hugging Face
    re.compile(r"\beyJ[A-Za-z0-9_\-]{10,}\.eyJ[A-Za-z0-9_\-]{10,}\.[A-Za-z0-9_\-]{10,}"),  # JWT
]
_BEARER = re.compile(r"(?i)\b(bearer|token)\s+([A-Za-z0-9_\-\.=]{20,})")
_ASSIGN = re.compile(r"(?i)\b([A-Z0-9_]*(?:PASSWORD|PASSWD|SECRET|TOKEN|API_?KEY|ACCESS_?KEY|PRIVATE_?KEY)[A-Z0-9_]*)"
                     r"(\s*[:=]\s*[\"']?)([^\s\"']{8,})")


def enabled() -> bool:
    return os.environ.get("TRACEPACK_REDACT", "1").strip().lower() not in ("0", "false", "no", "off")


def redact(text: str) -> "tuple[str, int]":
    """`(text with secrets masked, number of masks)`."""
    if not text:
        return text, 0
    n = 0
    for pat in _PATTERNS:
        text, k = pat.subn("[redacted]", text)
        n += k
    text, k = _BEARER.subn(lambda m: "%s [redacted]" % m.group(1), text)
    n += k
    text, k = _ASSIGN.subn(lambda m: "%s%s[redacted]" % (m.group(1), m.group(2)), text)
    n += k
    return text, n
