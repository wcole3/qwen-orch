#!/usr/bin/env python3
"""smoke.py — check a running OpenAI-compatible endpoint the way a team will use it.

    python3 smoke.py --base-url http://localhost:8081/v1 --model flint -n 8

Three checks, standard library only:
  1. one chat completion comes back with text;
  2. one request with a tool returns a well-formed call to that tool;
  3. N streaming requests at once all finish, with tokens/s for each.

The API key is read from --api-key or, better, the SMOKE_API_KEY environment
variable (which keeps it out of the process list). `./deploy.sh smoke` runs
this against whatever it deployed. Exit status 0 means every check passed.
"""
import argparse
import json
import os
import statistics
import sys
import threading
import time
import urllib.error
import urllib.request

PROMPT = ("Write a Python function that merges overlapping intervals, "
          "then explain in three sentences how it works.")
WEATHER_TOOL = {
    "type": "function",
    "function": {
        "name": "get_weather",
        "description": "Get the current weather for a city.",
        "parameters": {
            "type": "object",
            "properties": {"city": {"type": "string", "description": "City name"}},
            "required": ["city"],
        },
    },
}


class Client:
    def __init__(self, base_url, model, key, timeout):
        self.base_url = base_url.rstrip("/")
        self.model = model
        self.key = key
        self.timeout = timeout
        # Straight to the server, whatever http_proxy says: this checks one
        # endpoint, usually on this very machine, and a proxy in between would
        # be testing the proxy.
        self.opener = urllib.request.build_opener(urllib.request.ProxyHandler({}))

    def _request(self, path, body=None):
        headers = {"Content-Type": "application/json"}
        if self.key:
            headers["Authorization"] = "Bearer " + self.key
        data = json.dumps(body).encode() if body is not None else None
        req = urllib.request.Request(self.base_url + path, data=data, headers=headers)
        return self.opener.open(req, timeout=self.timeout)

    def first_model(self):
        with self._request("/models") as resp:
            return json.load(resp)["data"][0]["id"]

    def chat(self, messages, **extra):
        body = {"model": self.model, "messages": messages, **extra}
        with self._request("/chat/completions", body) as resp:
            return json.load(resp)

    def stream(self, messages, max_tokens):
        """Returns (completion_tokens, seconds_to_first_token, seconds_total)."""
        body = {"model": self.model, "messages": messages, "max_tokens": max_tokens,
                "stream": True, "stream_options": {"include_usage": True}}
        start = time.monotonic()
        first = None
        chunks = 0
        usage_tokens = None
        with self._request("/chat/completions", body) as resp:
            for raw in resp:
                line = raw.decode("utf-8", "replace").strip()
                if not line.startswith("data:"):
                    continue
                payload = line[5:].strip()
                if payload == "[DONE]":
                    break
                event = json.loads(payload)
                # A request that fails after the headers are sent arrives as an
                # error event inside a 200 response, not as an HTTP error.
                if "error" in event:
                    raise RuntimeError("server error: " + json.dumps(event["error"])[:300])
                if event.get("usage"):
                    usage_tokens = event["usage"].get("completion_tokens")
                for choice in event.get("choices") or []:
                    delta = choice.get("delta") or {}
                    # Reasoning arrives under either name depending on the server.
                    if delta.get("content") or delta.get("reasoning_content") \
                            or delta.get("reasoning") or delta.get("tool_calls"):
                        chunks += 1
                        if first is None:
                            first = time.monotonic() - start
        total = time.monotonic() - start
        # A server that omits usage still gets a figure: one chunk is roughly a token.
        tokens = usage_tokens if usage_tokens is not None else chunks
        if not tokens:
            raise RuntimeError("the stream ended without producing a token")
        return tokens, first if first is not None else total, total


def describe(exc):
    if isinstance(exc, urllib.error.HTTPError):
        detail = exc.read().decode("utf-8", "replace")[:300]
        return "HTTP %s %s" % (exc.code, detail)
    return "%s: %s" % (type(exc).__name__, exc)


def check_chat(client):
    reply = client.chat([{"role": "user", "content": "Reply with the single word: pong"}],
                        max_tokens=4096)
    message = reply["choices"][0]["message"]
    text = message.get("content") or ""
    reasoning = message.get("reasoning_content") or message.get("reasoning") or ""
    if not (text.strip() or reasoning.strip()):
        raise AssertionError("empty reply (finish_reason=%s)" % reply["choices"][0].get("finish_reason"))
    return "content=%r%s" % (text.strip()[:60], ", with reasoning" if reasoning.strip() else "")


def check_tool_call(client):
    reply = client.chat([{"role": "user", "content": "What is the weather in Paris right now?"}],
                        tools=[WEATHER_TOOL], tool_choice="required", max_tokens=4096)
    message = reply["choices"][0]["message"]
    calls = message.get("tool_calls") or []
    if not calls:
        raise AssertionError("no tool call; content=%r" % (message.get("content") or "")[:200])
    function = calls[0]["function"]
    if function["name"] != "get_weather":
        raise AssertionError("called %r, expected get_weather" % function["name"])
    arguments = json.loads(function["arguments"])
    if not isinstance(arguments, dict) or "city" not in arguments:
        raise AssertionError("arguments lack 'city': %r" % function["arguments"])
    return "get_weather(%s)" % json.dumps(arguments)


def check_parallel(client, n, max_tokens):
    results = [None] * n
    errors = [None] * n

    def worker(i):
        try:
            results[i] = client.stream([{"role": "user", "content": PROMPT}], max_tokens)
        except Exception as exc:  # reported per stream below
            errors[i] = describe(exc)

    start = time.monotonic()
    threads = [threading.Thread(target=worker, args=(i,)) for i in range(n)]
    for thread in threads:
        thread.start()
    for thread in threads:
        thread.join()
    wall = time.monotonic() - start

    done = [r for r in results if r is not None]
    lines = []
    for i in range(n):
        if results[i] is None:
            lines.append("    stream %d: FAILED %s" % (i + 1, errors[i]))
        else:
            tokens, first, total = results[i]
            lines.append("    stream %d: %4d tokens, first token %5.1fs, %6.1f tok/s"
                         % (i + 1, tokens, first, tokens / total if total else 0.0))
    if done:
        rates = [t / s for t, _, s in done if s]
        summary = ("%d/%d finished in %.1fs — per stream %.1f / %.1f / %.1f tok/s "
                   "(min / median / max), aggregate %.1f tok/s"
                   % (len(done), n, wall, min(rates), statistics.median(rates), max(rates),
                      sum(t for t, _, _ in done) / wall))
    else:
        summary = "0/%d finished" % n
    return len(done) == n, summary, lines


def main():
    parser = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    parser.add_argument("--base-url", default="http://localhost:8081/v1",
                        help="OpenAI-compatible base URL (default %(default)s)")
    parser.add_argument("--model", default="",
                        help="model id to request (default: the first one /models lists)")
    parser.add_argument("--api-key", default=os.environ.get("SMOKE_API_KEY", ""),
                        help="bearer key (default: $SMOKE_API_KEY)")
    parser.add_argument("-n", type=int, default=8, help="parallel streams (default %(default)s)")
    parser.add_argument("--max-tokens", type=int, default=300,
                        help="tokens per parallel stream (default %(default)s)")
    parser.add_argument("--timeout", type=float, default=600,
                        help="seconds per request (default %(default)s)")
    args = parser.parse_args()
    if args.n < 1:
        parser.error("-n must be at least 1")

    client = Client(args.base_url, args.model, args.api_key, args.timeout)
    failed = False

    try:
        if not client.model:
            client.model = client.first_model()
    except Exception as exc:
        print("cannot reach %s: %s" % (client.base_url, describe(exc)))
        return 1
    print("endpoint %s   model %s" % (client.base_url, client.model))

    for name, check in (("chat", check_chat), ("tool call", check_tool_call)):
        try:
            print("  %-10s PASS  %s" % (name, check(client)))
        except Exception as exc:
            failed = True
            detail = str(exc) if isinstance(exc, AssertionError) else describe(exc)
            print("  %-10s FAIL  %s" % (name, detail))

    passed, summary, lines = check_parallel(client, args.n, args.max_tokens)
    failed = failed or not passed
    print("  %-10s %s  %s" % ("parallel", "PASS" if passed else "FAIL", summary))
    for line in lines:
        print(line)

    print("FAILED" if failed else "OK")
    return 1 if failed else 0


if __name__ == "__main__":
    sys.exit(main())
