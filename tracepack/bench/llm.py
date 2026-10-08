"""tracepack.bench.llm -- a small OpenAI-compatible chat client for the benchmarks (stdlib only).

It is configured only through environment variables; there is no default endpoint.

    TRACEPACK_LLM_BASE_URL    anything that serves POST <base>/chat/completions
    TRACEPACK_LLM_API_KEY
    TRACEPACK_LLM_MODEL

The judge reads TRACEPACK_JUDGE_BASE_URL / _API_KEY / _MODEL; each one it does not find falls back to the
TRACEPACK_LLM_ value. TRACEPACK_LLM_MAX_INFLIGHT caps the requests one process has open at a time
(default: no cap), so a run can keep many questions going without flooding a busy endpoint.

Rate limits, server errors and dropped connections are retried with exponential backoff (5 s doubling
to 60 s, 10 attempts). If the server rejects `max_completion_tokens` or `temperature`, the request is
re-sent with `max_tokens` or without a temperature.
"""
from __future__ import annotations

import http.client
import json
import os
import random
import re
import socket
import threading
import time
import urllib.error
import urllib.request

# Some gateways write this notice into the content (often several times) while a saturated request waits;
# it is never part of the answer. A reply that is nothing but notices is retried.
QUEUE_NOTICE = re.compile(r"\s*Too many current requests\. Your queue position is \d+\. Please wait for a while\.[ \t]*\n?")
RETRY_CODES = {408, 409, 425, 429, 500, 502, 503, 504, 520, 522, 524, 529}


class LLMError(RuntimeError):
    pass


def _inflight_cap():
    try:
        n = int(os.environ.get("TRACEPACK_LLM_MAX_INFLIGHT") or 0)
    except ValueError:
        n = 0
    return threading.BoundedSemaphore(n) if n > 0 else None


_CAP = _inflight_cap()


def _env(role: str, name: str) -> str:
    if role == "judge":
        v = os.environ.get("TRACEPACK_JUDGE_" + name)
        if v:
            return v
    return os.environ.get("TRACEPACK_LLM_" + name) or ""


class Endpoint:
    """One model behind an OpenAI-compatible /chat/completions URL."""

    def __init__(self, base_url: str, api_key: str, model: str, timeout: float = 1200.0, tries: int = 10,
                 label: str = "llm"):
        self.base_url, self.api_key, self.model = base_url.rstrip("/"), api_key, model
        self.timeout, self.tries, self.label = timeout, tries, label
        self.max_field = "max_completion_tokens"
        self.send_temperature = True

    @classmethod
    def from_env(cls, role: str = "llm") -> "Endpoint":
        base, key, model = _env(role, "BASE_URL"), _env(role, "API_KEY"), _env(role, "MODEL")
        missing = [n for n, v in (("BASE_URL", base), ("API_KEY", key), ("MODEL", model)) if not v]
        if missing:
            pref = "TRACEPACK_JUDGE_ (or TRACEPACK_LLM_)" if role == "judge" else "TRACEPACK_LLM_"
            raise LLMError("the %s model is not configured: set %s%s" % (role, pref, ", ".join(missing)))
        return cls(base, key, model, label=role)

    def describe(self) -> str:
        return "%s model %s" % (self.label, self.model)

    def _post(self, body: dict) -> dict:
        req = urllib.request.Request(self.base_url + "/chat/completions", data=json.dumps(body).encode("utf-8"),
                                     headers={"Content-Type": "application/json",
                                              "Authorization": "Bearer " + self.api_key})
        if _CAP is not None:
            _CAP.acquire()
        try:
            with urllib.request.urlopen(req, timeout=self.timeout) as resp:
                return json.loads(resp.read())
        finally:
            if _CAP is not None:
                _CAP.release()

    def chat(self, messages, temperature: float = 0.0, max_tokens: int = 2048, model: str = ""):
        """-> (content, tool_calls, usage). Raises LLMError once the retries are used up."""
        err = ""
        for attempt in range(self.tries):
            body = {"model": model or self.model, "messages": messages, self.max_field: int(max_tokens)}
            if self.send_temperature:
                body["temperature"] = temperature
            try:
                d = self._post(body)
                ch = (d.get("choices") or [{}])[0]
                m = ch.get("message") or {}
                content = m.get("content") or ""
                content, notice = QUEUE_NOTICE.subn("", content)
                if notice and not content.strip() and not m.get("tool_calls"):
                    raise ValueError("only queue notices (%d) in the reply" % notice)
                u = d.get("usage") or {}
                usage = {"prompt_tokens": u.get("prompt_tokens"), "completion_tokens": u.get("completion_tokens"),
                         "cached_tokens": (u.get("prompt_tokens_details") or {}).get("cached_tokens"),
                         "finish": ch.get("finish_reason")}
                if notice:
                    usage["queue_notice"] = notice
                return content, m.get("tool_calls"), usage
            except urllib.error.HTTPError as e:
                try:
                    text = e.read()[:600].decode("utf-8", "replace")
                except Exception:
                    text = ""
                err = "HTTP %s %s" % (e.code, text)
                low = text.lower()
                if e.code == 400 and self.max_field == "max_completion_tokens" and "max_completion_tokens" in low:
                    self.max_field = "max_tokens"
                    continue
                if e.code == 400 and self.send_temperature and "temperature" in low:
                    self.send_temperature = False
                    continue
                if e.code not in RETRY_CODES:
                    raise LLMError("%s: %s" % (self.describe(), err))
            except (urllib.error.URLError, http.client.HTTPException, socket.timeout, TimeoutError,
                    ConnectionError, ValueError) as e:
                err = repr(e)[:300]
            if attempt + 1 < self.tries:
                time.sleep(min(60.0, 5.0 * 2 ** attempt) * (0.8 + 0.4 * random.random()))
        raise LLMError("%s: gave up after %d attempts: %s" % (self.describe(), self.tries, err))
