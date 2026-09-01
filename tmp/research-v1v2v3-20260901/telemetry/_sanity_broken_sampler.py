#!/usr/bin/env python3
"""Local sanity test: deliberately-broken sampler must NOT abort the run.

Runs the kit's samplers with the memory_pressure command pointed at a
nonexistent binary, and asserts the record still emits with an error field
for that key and real values elsewhere. Run with the exo venv python so mlx
is importable.
"""
import datetime
import importlib.util
import json
import socket
import sys
import time

spec = importlib.util.spec_from_file_location(
    "ct", "/Users/adam.durham/repos/exo/tmp/research-v1v2v3-20260901/telemetry/collect_telemetry.py"
)
ct = importlib.util.module_from_spec(spec)
spec.loader.exec_module(ct)

orig_run = ct._run


def broken_run(cmd, timeout=15.0):
    if cmd[0] == "memory_pressure":
        return 127, "", "command not found: /nonexistent/memory_pressure"
    return orig_run(cmd, timeout)


ct._run = broken_run
t0 = time.monotonic()
record = {
    "timestamp": datetime.datetime.now().astimezone().isoformat(),
    "hostname": socket.gethostname(),
    "label": "rep1",
    "mlx": ct.sample_mlx_memory(),
    "powermetrics": ct.sample_powermetrics(interval_ms=200, n_samples=2),
    "memory_pressure": ct.sample_memory_pressure(),
    "wired_limit": ct.sample_wired_limit(),
    "runner": ct.sample_runner(ct._DEFAULT_RUNNER_PATTERN),
    "elapsed_seconds": round(time.monotonic() - t0, 3),
}
ct._run = orig_run

print("memory_pressure (broken):", json.dumps(record["memory_pressure"]))
print("mlx (real):", json.dumps(record["mlx"]))
print("wired_limit (real):", json.dumps(record["wired_limit"]))
print("runner count (real):", record["runner"]["count"])
print("elapsed_seconds:", record["elapsed_seconds"])

assert "error" in record["memory_pressure"], "broken sampler must carry error"
assert record["memory_pressure"]["error"], "error must be non-empty"
assert record["mlx"]["error"] is None, "mlx must still be real"
assert record["wired_limit"]["error"] is None, "wired_limit must still be real"
print("\nPASS: broken sampler recorded error, rest of checkpoint still emitted real values")
