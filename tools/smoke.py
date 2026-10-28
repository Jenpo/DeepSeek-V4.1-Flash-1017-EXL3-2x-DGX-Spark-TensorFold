#!/usr/bin/env python3
"""Correctness smoke for the served model: reply shape, stop handling, tool calls, streaming usage, determinism.

    python3 -I tools/smoke.py [--url URL] [--model NAME] [--json FILE]

Checks (each PASS / FAIL / INFO; exit code 1 on any FAIL): /health answers; a plain reply ends with finish_reason
"stop" and carries usage; max_tokens=5 ends with "length" and at most 5 completion tokens; a `stop` string cuts the
reply before it and ends with "stop"; no raw special tokens leak into the text; a streamed reply ends with [DONE] and
a final usage chunk; a tool call with an array argument comes back as a JSON array (finish_reason "tool_calls"); a
tool result sent back gets a text answer; two identical greedy requests (fixed seed) return the same text; thinking
ON yields reasoning text and OFF none (INFO only: the field name is the server's).

New for this recipe; the tool-call check follows tools/toolcheck.py (GLM-5.3-Flash recipe, Mia's AI Lab, Apache-2.0).
Stdlib only.
"""
import argparse
import json
import os
import re
import sys
import urllib.error
import urllib.request

OPENER = urllib.request.build_opener(urllib.request.ProxyHandler({}))


def _api_key():
    """The server's API key (start.sh with API_KEY=1): API_KEY, else the first key in API_KEY_FILE, else none (as client.py)."""
    key = os.environ.get("API_KEY", "").strip()
    if key:
        return key
    path = os.environ.get("API_KEY_FILE", os.path.expanduser("~/.local/state/dsv41-tensorfold/api-key"))
    try:
        with open(path) as f:
            line = next((x.strip() for x in f if x.strip() and not x.startswith("#")), "")
    except OSError:
        return ""
    return line.split(":", 1)[1].strip() if ":" in line else line


if _api_key():
    OPENER.addheaders = [("Authorization", "Bearer " + _api_key())]
TOOLS = [{"type": "function", "function": {
    "name": "add_tags", "description": "Attach tags to a document.",
    "parameters": {"type": "object", "required": ["doc_id", "tags"], "properties": {
        "doc_id": {"type": "integer", "description": "The document id."},
        "tags": {"type": "array", "items": {"type": "string"}, "description": "The tags to attach."}}}}}]
LEAK = re.compile(r"<\|?[^>\s]{0,24}(?:begin|end|im_|eos|bos|think|User|Assistant|tool)[^>]{0,24}\|?>|<｜[^>]*｜>", re.I)
results = []


def record(name, status, detail=""):
    results.append({"check": name, "status": status, "detail": detail})
    print(f"{status:4s} {name}" + (f": {detail}" if detail else ""), flush=True)


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--url", default=os.environ.get("API_URL", "http://127.0.0.1:" + os.environ.get("PORT", "8888")))
    ap.add_argument("--model", default=os.environ.get("MODEL", "DeepSeek-V4.1-Flash-1017"))
    ap.add_argument("--json", help="write the results here")
    a = ap.parse_args()
    url = a.url.rstrip("/")

    def post(body, stream=False):
        body = {"model": a.model, "temperature": 0, "seed": 7, "chat_template_kwargs": {"enable_thinking": False},
                **body}
        req = urllib.request.Request(url + "/v1/chat/completions", json.dumps(body).encode(),
                                     {"Content-Type": "application/json"})
        try:
            r = OPENER.open(req, timeout=600)
        except urllib.error.HTTPError as e:
            return {"_error": f"HTTP {e.code}: {e.read().decode(errors='replace')[:200]}"}
        except Exception as e:  # noqa: BLE001
            return {"_error": f"{type(e).__name__}: {e}"}
        if stream:
            return r
        return json.load(r)

    def msg(reply):
        return reply["choices"][0]["message"], reply["choices"][0].get("finish_reason")

    try:
        with OPENER.open(url + "/health", timeout=15) as r:
            record("health", "PASS", r.read(300).decode(errors="replace"))
    except Exception as e:  # noqa: BLE001
        record("health", "FAIL", f"{e}")
        return 1

    r = post({"messages": [{"role": "user", "content": "Say hello in one short sentence."}], "max_tokens": 200})
    if "_error" in r:
        record("plain reply", "FAIL", r["_error"])
    else:
        m, fin = msg(r)
        record("plain reply", "PASS" if fin == "stop" and (m.get("content") or "").strip() and r.get("usage") else "FAIL",
               f"finish={fin} usage={r.get('usage')} text={(m.get('content') or '')[:60]!r}")
        leak = LEAK.search(m.get("content") or "")
        record("no special-token leak", "FAIL" if leak else "PASS", leak.group(0) if leak else "")

    r = post({"messages": [{"role": "user", "content": "Write a long story about a dragon."}], "max_tokens": 5})
    if "_error" in r:
        record("max_tokens", "FAIL", r["_error"])
    else:
        m, fin = msg(r)
        n = r["usage"]["completion_tokens"]
        record("max_tokens", "PASS" if fin == "length" and n <= 5 else "FAIL", f"finish={fin} completion_tokens={n}")

    r = post({"messages": [{"role": "user", "content": "Count from 1 to 10 as digits separated by spaces, nothing else."}],
              "max_tokens": 64, "stop": ["5"]})
    if "_error" in r:
        record("stop string", "FAIL", r["_error"])
    else:
        m, fin = msg(r)
        t = m.get("content") or ""
        record("stop string", "PASS" if "5" not in t and fin == "stop" and "4" in t else "FAIL", f"finish={fin} text={t!r}")

    r = post({"messages": [{"role": "user", "content": "Reply with the single word: pong"}], "max_tokens": 16,
              "stream": True, "stream_options": {"include_usage": True}}, stream=True)
    if isinstance(r, dict):
        record("stream", "FAIL", r["_error"])
    else:
        done = usage = False
        text = ""
        for raw in r:
            line = raw.decode("utf-8", "replace").strip()
            if line == "data: [DONE]":
                done = True
            elif line.startswith("data:"):
                ev = json.loads(line[5:])
                usage = usage or bool(ev.get("usage"))
                for ch in ev.get("choices") or []:
                    text += (ch.get("delta") or {}).get("content") or ""
        record("stream", "PASS" if done and usage else "FAIL", f"[DONE]={done} usage_chunk={usage} text={text!r}")

    tool_msgs = [{"role": "user", "content": "Tag document 42 with 'urgent', 'finance' and 'q3' using the tool."}]
    r = post({"messages": tool_msgs, "tools": TOOLS, "max_tokens": 1024})
    calls = []
    if "_error" in r:
        record("tool call", "FAIL", r["_error"])
    else:
        m, fin = msg(r)
        calls = m.get("tool_calls") or []
        ok = False
        for c in calls:
            try:
                args = json.loads(c["function"]["arguments"])
            except ValueError:
                continue
            ok |= (c["function"]["name"] == "add_tags" and isinstance(args.get("tags"), list)
                   and len(args["tags"]) >= 3 and isinstance(args.get("doc_id"), int))
        record("tool call", "PASS" if ok else "FAIL", f"finish={fin} calls={json.dumps(calls)[:200]}")
        if ok and fin != "tool_calls":
            record("tool call finish_reason", "FAIL", f"{fin!r}, expected 'tool_calls'")
    if calls:
        follow = tool_msgs + [{"role": "assistant", "content": None, "tool_calls": calls},
                              {"role": "tool", "tool_call_id": calls[0].get("id", "call_0"), "content": "ok: 3 tags attached"}]
        r = post({"messages": follow, "tools": TOOLS, "max_tokens": 200})
        if "_error" in r:
            record("tool result round trip", "FAIL", r["_error"])
        else:
            m, fin = msg(r)
            record("tool result round trip", "PASS" if (m.get("content") or "").strip() and fin == "stop" else "FAIL",
                   f"finish={fin} text={(m.get('content') or '')[:80]!r}")

    body = {"messages": [{"role": "user", "content": "Name three primary colors, comma separated."}], "max_tokens": 60}
    r1, r2 = post(body), post(body)
    if "_error" in r1 or "_error" in r2:
        record("greedy repeat", "FAIL", (r1.get("_error") or r2.get("_error")))
    else:
        t1, t2 = msg(r1)[0].get("content"), msg(r2)[0].get("content")
        record("greedy repeat", "PASS" if t1 == t2 else "FAIL", f"{t1!r} vs {t2!r}")

    for think in (True, False):
        r = post({"messages": [{"role": "user", "content": "What is 17 * 23?"}], "max_tokens": 600,
                  "chat_template_kwargs": {"enable_thinking": think}})
        if "_error" in r:
            record(f"thinking {'on' if think else 'off'}", "FAIL", r["_error"])
            continue
        m, fin = msg(r)
        reasoning = m.get("reasoning_content") or m.get("reasoning") or ""
        record(f"thinking {'on' if think else 'off'}", "INFO",
               f"finish={fin} reasoning_chars={len(reasoning)} content={(m.get('content') or '')[:60]!r} keys={sorted(m)}")

    fails = [x for x in results if x["status"] == "FAIL"]
    print(f"smoke: {'OK' if not fails else 'FAILED: ' + ', '.join(x['check'] for x in fails)}")
    if a.json:
        json.dump(results, open(a.json, "w"), indent=1)
    return 1 if fails else 0


if __name__ == "__main__":
    sys.exit(main())
