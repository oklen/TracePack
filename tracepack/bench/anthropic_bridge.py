"""tracepack.bench.anthropic_bridge -- serve Claude Code's model requests from an OpenAI-compatible model.

Claude Code speaks the Anthropic Messages API. For the LongMemEval benchmark its two arms must use the
same model as the other arms, so this small server (stdlib only) sits between them:

    ANTHROPIC_BASE_URL=http://127.0.0.1:<port>  claude -p ...      ->  bridge  ->  TRACEPACK_LLM_* model

What it does to a request:
- the system blocks become one system message (Claude Code's billing-header block is dropped);
- messages keep their role, including the system-role reminders Claude Code puts into the message list
  (dates, agent listings); text blocks are kept verbatim; a tool call or tool result, if any, becomes a
  one-line text note;
- `tools`, `thinking`, `context_management` and other Anthropic-only fields are dropped, so the model
  answers in text (the benchmark runs Claude Code with --tools "");
- `max_tokens` is passed on; Claude Code sends no temperature, so the model's default applies.
The reply goes back as a normal Anthropic message, or as server-sent events when Claude Code streams,
with keep-alive pings while the model is working. `count_tokens` is answered with an estimate.

Every request is logged (one JSON line: session, purpose, tokens, seconds); `on_request` lets the
benchmark keep the full request and reply of each compaction and answer.
"""
from __future__ import annotations

import json
import threading
import time
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer

COMPACT_MARK = "Your task is to create a detailed summary of the conversation so far"


def _block_text(b) -> str:
    if isinstance(b, str):
        return b
    if not isinstance(b, dict):
        return ""
    t = b.get("type")
    if t == "text":
        return b.get("text") or ""
    if t == "tool_use":
        return "[tool call %s %s]" % (b.get("name"), json.dumps(b.get("input"), ensure_ascii=False)[:2000])
    if t == "tool_result":
        c = b.get("content")
        inner = c if isinstance(c, str) else "\n".join(_block_text(x) for x in (c or []))
        return "[tool result]\n" + inner
    if t in ("thinking", "redacted_thinking"):
        return ""
    if t == "image":
        return "[image]"
    return ""


def to_openai(body: dict) -> list:
    """Anthropic Messages request -> OpenAI chat messages (text only)."""
    out = []
    sysp = body.get("system")
    if isinstance(sysp, list):
        parts = [(b.get("text") or "") for b in sysp if isinstance(b, dict)]
        parts = [p for p in parts if p and not p.startswith("x-anthropic-billing-header")]
        sys_text = "\n\n".join(parts)
    else:
        sys_text = sysp or ""
    if sys_text:
        out.append({"role": "system", "content": sys_text})
    for m in body.get("messages") or []:
        c = m.get("content")
        if isinstance(c, str):
            text = c
        else:
            text = "\n\n".join(x for x in (_block_text(b) for b in (c or [])) if x)
        role = m.get("role") if m.get("role") in ("assistant", "system") else "user"
        out.append({"role": role, "content": text})
    return out


def purpose_of(body: dict) -> str:
    """"compact" when the last user message carries Claude Code's compaction prompt; "answer" for a request
    without tools (the benchmark's question); else "turn"."""
    for m in reversed(body.get("messages") or []):
        if m.get("role") != "user":
            continue
        c = m.get("content")
        last = c if isinstance(c, str) else "\n".join(_block_text(b) for b in (c or []))
        if COMPACT_MARK in last:
            return "compact"
        break
    return "answer" if not body.get("tools") else "turn"


class Bridge:
    def __init__(self, endpoint, log_path: str = "", on_request=None, max_tokens_cap: int = 32000):
        self.endpoint, self.log_path, self.on_request = endpoint, log_path, on_request
        self.max_tokens_cap = max_tokens_cap
        self.lock = threading.Lock()
        self.httpd = None

    def log(self, rec: dict):
        if not self.log_path:
            return
        with self.lock, open(self.log_path, "a", encoding="utf-8") as fh:
            fh.write(json.dumps(rec, ensure_ascii=False) + "\n")

    def complete(self, body: dict, session: str):
        msgs = to_openai(body)
        mt = min(int(body.get("max_tokens") or 4096), self.max_tokens_cap)
        kw = {"max_tokens": mt}
        if body.get("temperature") is not None:
            kw["temperature"] = body["temperature"]
        else:
            kw["temperature"] = 1.0
        t0 = time.time()
        purpose = purpose_of(body)
        content, _tcs, usage = self.endpoint.chat(msgs, **kw)
        rec = {"t": time.time(), "session": session, "purpose": purpose, "seconds": round(time.time() - t0, 2),
               "prompt_tokens": usage.get("prompt_tokens"), "completion_tokens": usage.get("completion_tokens"),
               "finish": usage.get("finish"), "chars_out": len(content or "")}
        self.log(rec)
        if self.on_request:
            try:
                self.on_request(session, purpose, msgs, content, usage)
            except Exception:                    # noqa: BLE001 - bookkeeping must not break a request
                pass
        return content or "", usage

    def handler(self):
        bridge = self

        class H(BaseHTTPRequestHandler):
            protocol_version = "HTTP/1.1"

            def log_message(self, *a):
                pass

            def _json(self, code, obj):
                data = json.dumps(obj).encode("utf-8")
                self.send_response(code)
                self.send_header("Content-Type", "application/json")
                self.send_header("Content-Length", str(len(data)))
                self.end_headers()
                self.wfile.write(data)

            def do_HEAD(self):
                self.send_response(200)
                self.send_header("Content-Length", "0")
                self.end_headers()

            def do_GET(self):
                self._json(200, {"data": [{"id": bridge.endpoint.model, "type": "model"}]})

            def do_POST(self):
                n = int(self.headers.get("Content-Length") or 0)
                try:
                    body = json.loads(self.rfile.read(n) or b"{}")
                except ValueError:
                    self._json(400, {"type": "error", "error": {"type": "invalid_request_error", "message": "bad JSON"}})
                    return
                path = self.path.split("?")[0]
                if path.endswith("/count_tokens"):
                    chars = sum(len(m.get("content") or "") for m in to_openai(body))
                    self._json(200, {"input_tokens": chars // 4})
                    return
                if not path.endswith("/messages"):
                    self._json(404, {"type": "error", "error": {"type": "not_found_error", "message": path}})
                    return
                session = self.headers.get("X-Claude-Code-Session-Id") or ""
                if body.get("stream"):
                    self._stream(body, session)
                else:
                    try:
                        text, usage = bridge.complete(body, session)
                    except Exception as e:      # noqa: BLE001
                        self._json(529, {"type": "error", "error": {"type": "overloaded_error", "message": repr(e)[:300]}})
                        return
                    self._json(200, _message(body, text, usage))

            def _stream(self, body, session):
                self.send_response(200)
                self.send_header("Content-Type", "text/event-stream")
                self.send_header("Cache-Control", "no-cache")
                self.send_header("Connection", "close")
                self.end_headers()
                self.close_connection = True

                def ev(name, data):
                    self.wfile.write(("event: %s\ndata: %s\n\n" % (name, json.dumps(data))).encode("utf-8"))
                    self.wfile.flush()

                start = _message(body, "", {"prompt_tokens": 0, "completion_tokens": 0})
                start.update(content=[], stop_reason=None)
                ev("message_start", {"type": "message_start", "message": start})
                box = {}

                def work():
                    try:
                        box["ok"] = bridge.complete(body, session)
                    except Exception as e:      # noqa: BLE001
                        box["err"] = repr(e)[:300]
                th = threading.Thread(target=work, daemon=True)
                th.start()
                while th.is_alive():
                    th.join(10)
                    if th.is_alive():
                        ev("ping", {"type": "ping"})
                if "err" in box:
                    ev("error", {"type": "error", "error": {"type": "overloaded_error", "message": box["err"]}})
                    return
                text, usage = box["ok"]
                ev("content_block_start", {"type": "content_block_start", "index": 0,
                                           "content_block": {"type": "text", "text": ""}})
                for i in range(0, len(text), 4000):
                    ev("content_block_delta", {"type": "content_block_delta", "index": 0,
                                               "delta": {"type": "text_delta", "text": text[i:i + 4000]}})
                ev("content_block_stop", {"type": "content_block_stop", "index": 0})
                ev("message_delta", {"type": "message_delta", "delta": {"stop_reason": "end_turn", "stop_sequence": None},
                                     "usage": {"output_tokens": usage.get("completion_tokens") or 0}})
                ev("message_stop", {"type": "message_stop"})

        return H

    def start(self, port: int, host: str = "127.0.0.1"):
        self.httpd = ThreadingHTTPServer((host, port), self.handler())
        self.httpd.daemon_threads = True
        threading.Thread(target=self.httpd.serve_forever, daemon=True).start()
        return self.httpd.server_address[1]

    def stop(self):
        if self.httpd:
            self.httpd.shutdown()
            self.httpd.server_close()


def _message(body, text, usage):
    return {"id": "msg_bridge_%d" % int(time.time() * 1000), "type": "message", "role": "assistant",
            "model": body.get("model") or "bridge", "content": [{"type": "text", "text": text}],
            "stop_reason": "end_turn", "stop_sequence": None,
            "usage": {"input_tokens": usage.get("prompt_tokens") or 0, "output_tokens": usage.get("completion_tokens") or 0}}


def serve(port: int, endpoint, log_path: str = ""):
    b = Bridge(endpoint, log_path)
    b.start(port)
    print("bridge: Anthropic Messages API on http://127.0.0.1:%d -> %s" % (port, endpoint.describe()), flush=True)
    try:
        while True:
            time.sleep(3600)
    except KeyboardInterrupt:
        b.stop()
