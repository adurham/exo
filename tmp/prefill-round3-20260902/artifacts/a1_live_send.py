import json, time, urllib.request, urllib.error, sys

SRC = "/Users/adam.durham/repos/exo/tmp/prefill-round3-20260902/artifacts/a1_offline_render.json"
URL = "http://192.168.86.201:52415/v1/chat/completions"
OUT = "/Users/adam.durham/repos/exo/tmp/prefill-round3-20260902/artifacts/a1_live_responses.json"

with open(SRC) as f:
    data = json.load(f)

bodies = data["request_bodies"]
# Ensure sampling/stream params exactly as spec
for key in bodies:
    bodies[key]["max_tokens"] = 1
    bodies[key]["temperature"] = 0
    bodies[key]["stream"] = False
    bodies[key]["model"] = "deepseek-ai/DeepSeek-V4-Flash-0731"

results = {}
for key in ("a_absent", "b_empty", "c_space"):
    body = bodies[key]
    payload = json.dumps(body).encode("utf-8")
    attempts = 0
    last_exc = None
    while attempts < 2:
        attempts += 1
        t0 = time.time()
        req = urllib.request.Request(URL, data=payload, headers={"Content-Type": "application/json"})
        try:
            with urllib.request.urlopen(req, timeout=600) as resp:
                raw = resp.read().decode("utf-8")
                status = resp.status
            dt = time.time() - t0
            try:
                parsed = json.loads(raw)
            except Exception:
                parsed = None
            results[key] = {
                "attempts": attempts,
                "http_status": status,
                "wall_time_seconds": round(dt, 3),
                "response_json": parsed,
                "response_raw": raw,
            }
            print(f"{key}: status={status} wall={dt:.3f}s attempts={attempts}")
            break
        except urllib.error.HTTPError as e:
            dt = time.time() - t0
            raw = e.read().decode("utf-8", "replace")
            parsed = None
            try:
                parsed = json.loads(raw)
            except Exception:
                pass
            print(f"{key}: HTTP {e.code} wall={dt:.3f}s attempts={attempts}")
            last_exc = e
            if attempts >= 2:
                results[key] = {
                    "attempts": attempts,
                    "http_status": e.code,
                    "wall_time_seconds": round(dt, 3),
                    "response_json": parsed,
                    "response_raw": raw,
                    "error": f"HTTPError {e.code}",
                }
            # else loop and retry once
        except Exception as e:
            dt = time.time() - t0
            print(f"{key}: EXC {type(e).__name__}: {e} wall={dt:.3f}s attempts={attempts}")
            last_exc = e
            if attempts >= 2:
                results[key] = {
                    "attempts": attempts,
                    "http_status": None,
                    "wall_time_seconds": round(dt, 3),
                    "response_json": None,
                    "response_raw": repr(last_exc),
                    "error": f"{type(last_exc).__name__}: {last_exc}",
                }
            # else loop and retry
    # ensure key recorded if only failed non-http path
    if key not in results:
        continue

# Preserve the ORDER a_absent, b_empty, c_space
ordered = {k: results[k] for k in ("a_absent", "b_empty", "c_space") if k in results}

with open(OUT, "w") as f:
    json.dump(ordered, f, indent=2)

print("\n--- usage blocks ---")
for k, r in ordered.items():
    if r.get("response_json") and isinstance(r["response_json"], dict):
        print(k, "usage =", json.dumps(r["response_json"].get("usage")))
    else:
        print(k, "NO USAGE:", r.get("error"), r.get("http_status"))
print("\nWrote", OUT)
