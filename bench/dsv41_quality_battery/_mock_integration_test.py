#!/usr/bin/env python3
"""
_mock_integration_test.py - OFFLINE integration test for battery.py.

Spins up a stdlib http.server that returns canned OpenAI-shaped responses
(needle recall, tool_calls, multilingual free prose incl. one glued fragment,
park recall) and runs `battery.py all` against it. Proves the request/parse/
score/detect/persist/aggregate pipeline end-to-end WITHOUT the cluster.

Run:  python3 _mock_integration_test.py
"""
import json
import os
import re
import subprocess
import sys
import threading
import time
from http.server import BaseHTTPRequestHandler, HTTPServer

HERE = os.path.dirname(os.path.abspath(__file__))


class Handler(BaseHTTPRequestHandler):
    def log_message(self, *a):
        pass

    def do_POST(self):
        n = int(self.headers.get("Content-Length", 0))
        body = json.loads(self.rfile.read(n) or b"{}")
        msgs = body.get("messages") or []
        prompt = "\n".join((m.get("content") or "") for m in msgs)
        low = prompt.lower()
        # route on the QUESTION TAIL (last user message), not the whole prompt --
        # needles re-send the giant build prompt with the question appended.
        last = (msgs[-1].get("content") or "").lower()[-400:] if msgs else ""
        tail = last
        tools = body.get("tools")
        content = ""
        tool_calls = None
        if tools:
            if "outlook" in last and "weather right now" in last:
                tool_calls = [
                    {"id": "c1", "type": "function", "function": {
                        "name": "get_weather",
                        "arguments": json.dumps({"city": "madrid", "unit": "celsius"})}},
                    {"id": "c2", "type": "function", "function": {
                        "name": "get_forecast",
                        "arguments": json.dumps({"city": "madrid", "days": 4, "unit": "celsius"})}},
                ]
            elif "forecast" in last or re.search(r"\d+[- ]day", last):
                city = next((c for c in ["tokyo", "madrid", "sydney", "new york", "london"]
                             if c in last), "tokyo")
                m = re.search(r"(\d+)[- ]?day", last)
                args = {"city": city, "days": int(m.group(1)) if m else 5}
                if "celsius" in last or "fahrenheit" in last:
                    args["unit"] = "fahrenheit" if "fahrenheit" in last else "celsius"
                tool_calls = [{"id": "c1", "type": "function", "function": {
                    "name": "get_forecast", "arguments": json.dumps(args)}}]
            elif "thermostat" in last or "heat" in last:
                city = next((c for c in ["berlin", "rome"] if c in last), "berlin")
                mm = re.search(r"(\d+)\s*degrees", last)
                tool_calls = [{"id": "c1", "type": "function", "function": {
                    "name": "set_thermostat",
                    "arguments": json.dumps({"city": city,
                                             "target_temp": int(mm.group(1)) if mm else 21})}}]
            else:
                city = next((c for c in ["paris", "london", "sydney", "new york", "madrid"]
                             if c in last), "paris")
                tool_calls = [{"id": "c1", "type": "function", "function": {
                    "name": "get_weather",
                    "arguments": json.dumps({"city": city,
                                             "unit": "fahrenheit" if "fahrenheit" in last else "celsius"})}}]
        elif "favorite color" in tail:
            content = "Your favorite color is teal."
        elif "verification token" in tail:
            content = "The verification token is QK-7731."
        elif "is the vault code 8493" in tail:
            content = "No. The correct vault code is 8492, not 8493."
        elif "vault code" in tail or "opens the vault" in tail:
            content = "The vault code is 8492."
        elif "what month" in tail:
            content = "The project launches in November."
        elif "autumn rain" in tail or "\u79cb\u96e8" in prompt:
            if os.environ.get("MOCK_CLEAN") == "1":
                content = ("\u96e8\u58f0\u6dc5\u6ca5\u3002The rain whispers softly on the roof.")
            else:
                # deliberately inject the lm_head defect signature into one prose answer
                content = ("\u96e8\u58f0\u6dc5\u6ca5\u3002The rain whispers on the roof, "
                           "the angle\u0441\u044c of the gutter held a quiet note.")
        elif "alten Baum" in prompt or "old tree" in tail:
            content = ("Der alte Baum steht still im Wind. "
                       "The old tree stands still in the wind.")
        else:
            content = "A clean sentence about the topic, with no defects at all."
        resp = {"choices": [{"message": {"content": content, "tool_calls": tool_calls,
                                         "reasoning_content": ""},
                             "finish_reason": "stop"}],
                "usage": {"prompt_tokens": 111, "completion_tokens": 12}}
        data = json.dumps(resp).encode()
        self.send_response(200)
        self.send_header("Content-Type", "application/json")
        self.send_header("Content-Length", str(len(data)))
        self.end_headers()
        self.wfile.write(data)


def main():
    srv = HTTPServer(("127.0.0.1", 0), Handler)
    port = srv.server_address[1]
    t = threading.Thread(target=srv.serve_forever, daemon=True)
    t.start()
    api = "http://127.0.0.1:{}".format(port)
    label = "_mock"
    # clean prior
    resdir = os.path.join(HERE, "results", label)
    subprocess.run(["rm", "-rf", resdir], check=False)

    print("== mock server on", api, "==")
    cmd = [sys.executable, os.path.join(HERE, "battery.py"),
           "--api", api, "--label", label, "--depth", "800",
           "--park-tokens", "800", "--prose-tokens", "128", "--timeout", "30",
           "--force", "all"]
    r = subprocess.run(cmd, cwd=HERE, capture_output=True, text=True, timeout=120)
    print(r.stdout)
    if r.returncode != 0:
        print("STDERR:", r.stderr)
    srv.shutdown()

    # ---- assertions ----
    def load(p):
        with open(os.path.join(resdir, p)) as f:
            return json.load(f)

    ok = True
    needles = load("needles.json")
    tools = load("tools.json")
    prose = load("free_prose/index.json")
    park = load("park.json")
    summary = load("summary.json")

    def check(name, cond):
        nonlocal ok
        ok = ok and cond
        print("  [{}] {}".format("PASS" if cond else "FAIL", name))

    print("\n== assertions ==")
    check("needles {}/{} pass".format(needles["n_pass"], needles["n"]),
          needles["n_pass"] == needles["n"] == 6)
    check("tools {}/{} pass".format(tools["n_pass"], tools["n"]),
          tools["n_pass"] == tools["n"] == 10)
    check("park recall_teal", park.get("recall_teal") is True)
    dirty = [p["id"] for p in prose["probes"]
             if (p.get("detector") or {}).get("verdict") == "DIRTY"]
    check("one prose prompt DIRTY (glue injected): {}".format(dirty),
          dirty == ["zh_autumn"])
    check("summary phase_verdict DIRTY", summary["phase_verdict"] == "DIRTY")
    # raw tool_calls recorded
    with open(os.path.join(resdir, "raw", "tools_t2_forecast_tokyo.json")) as f:
        raw = json.load(f)
    check("raw tool_calls persisted", "get_forecast" in raw.get("raw_body", ""))

    # resumability: re-run must skip (no --force)
    cmd2 = [sys.executable, os.path.join(HERE, "battery.py"),
            "--api", api, "--label", label, "eval"]
    r2 = subprocess.run(cmd2, cwd=HERE, capture_output=True, text=True, timeout=60)
    skipped = all(s in r2.stdout for s in ["[needles] exists", "[tools] exists"])
    check("resumable: cached run skips needles/tools", skipped)

    print("\n" + ("INTEGRATION TEST: ALL PASS" if ok else "INTEGRATION TEST: FAILURES"))
    return 0 if ok else 1


if __name__ == "__main__":
    sys.exit(main())
