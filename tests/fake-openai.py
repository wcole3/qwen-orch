#!/usr/bin/env python3
"""Minimal OpenAI-compatible server for smoke.py's tests.

    fake-openai.py PORT MODE

MODE: ok | stream-error (an SSE error event, as llama-server sends when a
request fails after the headers) | empty-stream (200, no tokens) | no-tool
(answers a tool request with plain text).
"""
import json
import sys
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer

PORT, MODE = int(sys.argv[1]), sys.argv[2]


class Handler(BaseHTTPRequestHandler):
    def log_message(self, *args):
        pass

    def _json(self, obj):
        data = json.dumps(obj).encode()
        self.send_response(200)
        self.send_header("Content-Type", "application/json")
        self.send_header("Content-Length", str(len(data)))
        self.end_headers()
        self.wfile.write(data)

    def do_GET(self):
        self._json({"data": [{"id": "fake"}], "status": "ok"})

    def _event(self, obj):
        payload = obj if isinstance(obj, str) else json.dumps(obj)
        self.wfile.write(("data: %s\n\n" % payload).encode())
        self.wfile.flush()

    def do_POST(self):
        body = json.loads(self.rfile.read(int(self.headers["Content-Length"])))
        if body.get("stream"):
            self.send_response(200)
            self.send_header("Content-Type", "text/event-stream")
            self.end_headers()
            self.close_connection = True
            if MODE == "stream-error":
                self._event({"error": {"code": 500, "message": "slot unavailable"}})
                return
            if MODE != "empty-stream":
                for word in ("one", "two", "three"):
                    self._event({"choices": [{"delta": {"content": word}}]})
                self._event({"choices": [], "usage": {"completion_tokens": 3}})
            self._event("[DONE]")
        elif body.get("tools") and MODE != "no-tool":
            self._json({"choices": [{"finish_reason": "tool_calls", "message": {
                "content": None,
                "tool_calls": [{"id": "c1", "type": "function", "function": {
                    "name": "get_weather", "arguments": '{"city": "Paris"}'}}]}}]})
        else:
            self._json({"choices": [{"finish_reason": "stop",
                                     "message": {"content": "pong"}}]})


ThreadingHTTPServer(("127.0.0.1", PORT), Handler).serve_forever()
