#!/usr/bin/env python3
"""Passive per-node cluster telemetry sampler for the exo research campaign.

Samples per-node state at a caller-defined checkpoint (T0 / warmup / repN)
during a FUTURE cluster launch. READ-ONLY and non-perturbing: it samples,
never changes cluster state, never restarts anything, never sets sysctls.

Every sampler degrades gracefully: a missing/failing/password-gated command
records an explicit error string in that field and the checkpoint still
emits. A single failing sampler never aborts the run.

Output: one JSON object per invocation, appended to a JSONL file (one line
per checkpoint) so a whole run is a single parseable artifact.

Usage:
    .venv/bin/python collect_telemetry.py LABEL [--out FILE] [--runner-pattern RE]

    LABEL   checkpoint label, e.g. 'T0', 'warmup', 'rep1' (required)
    --out   JSONL output file (default: telemetry_<hostname>.jsonl in cwd)
    --runner-pattern  regex for identifying the exo runner process(es)
            (default: EXO_TELEMETRY_RUNNER_PATTERN env or a built-in set)

Run with the exo venv python (`.venv/bin/python`) so `mlx` is importable.
If mlx is not importable, the mlx field records an error and everything
else still works.

No third-party dependencies beyond stdlib + mlx.
"""

from __future__ import annotations

import argparse
import datetime
import json
import os
import re
import socket
import subprocess
import sys
import time

# ---------------------------------------------------------------------------
# Sampler helpers
# ---------------------------------------------------------------------------


def _run(cmd: list[str], timeout: float = 15.0) -> tuple[int, str, str]:
    """Run a command, return (returncode, stdout, stderr). Never raises."""
    try:
        proc = subprocess.run(
            cmd,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            text=True,
            timeout=timeout,
        )
        return proc.returncode, proc.stdout, proc.stderr
    except FileNotFoundError as e:
        return 127, "", f"command not found: {e}"
    except subprocess.TimeoutExpired as e:
        return 124, "", f"timed out after {timeout}s"
    except Exception as e:  # noqa: BLE001 - degrade gracefully
        return 1, "", f"error running {cmd[0]}: {e!r}"


def _err(msg: str) -> dict:
    return {"error": msg}


# ---------------------------------------------------------------------------
# (a) MLX GPU memory: active + allocator cache + peak
# ---------------------------------------------------------------------------


def sample_mlx_memory() -> dict:
    """Sample MLX Metal active / cache / peak memory.

    The MLX memory API has moved across versions:
      - old: mx.metal.get_active_memory() / get_cache_memory() / get_peak_memory()
      - new: mx.get_active_memory() / get_cache_memory() / get_peak_memory()
    Both are detected at runtime; the current (top-level) form is preferred.
    """
    try:
        import mlx.core as mx  # type: ignore
    except Exception as e:  # noqa: BLE001
        return _err(f"mlx not importable: {e!r}")

    # Detect which API surface is present. Prefer the current top-level form.
    api = None
    if all(hasattr(mx, n) for n in ("get_active_memory", "get_cache_memory", "get_peak_memory")):
        api = "mx.* (current top-level)"
        get_active, get_cache, get_peak = (
            mx.get_active_memory,
            mx.get_cache_memory,
            mx.get_peak_memory,
        )
    elif all(hasattr(mx.metal, n) for n in ("get_active_memory", "get_cache_memory", "get_peak_memory")):
        api = "mx.metal.* (deprecated)"
        get_active, get_cache, get_peak = (
            mx.metal.get_active_memory,
            mx.metal.get_cache_memory,
            mx.metal.get_peak_memory,
        )
    else:
        return _err(
            "no known MLX memory API found; "
            "tried mx.get_active_memory/get_cache_memory/get_peak_memory and "
            "mx.metal.get_active_memory/get_cache_memory/get_peak_memory"
        )

    try:
        return {
            "api": api,
            "active_bytes": int(get_active()),
            "cache_bytes": int(get_cache()),
            "peak_bytes": int(get_peak()),
            "error": None,
        }
    except Exception as e:  # noqa: BLE001
        return _err(f"mlx memory read failed via {api}: {e!r}")


# ---------------------------------------------------------------------------
# (b) powermetrics: GPU clocks, residency, power
# ---------------------------------------------------------------------------

# Field regexes validated against real `sudo powermetrics --samplers gpu_power`
# output (same patterns as bench/section100_gpu_ground_truth.py, which was
# validated against live cluster captures on 2026-08-22).
_RE_ACTIVE_RES = re.compile(r"GPU HW active residency:\s*([\d.]+)%")
_RE_ACTIVE_FREQ = re.compile(r"GPU HW active frequency:\s*([\d.]+)\s*MHz")
_RE_IDLE_RES = re.compile(r"GPU idle residency:\s*([\d.]+)%")
_RE_POWER = re.compile(r"GPU Power:\s*([\d.]+)\s*mW")


def parse_powermetrics_gpu(text: str) -> list[dict]:
    """Parse gpu_power sampler text into a list of per-sample dicts.

    Each sample block ends with the 'GPU idle residency' line (the last line
    of a gpu_power block), so that line is the flush marker.
    """
    samples: list[dict] = []
    active_res = active_freq = idle_res = power_mw = None
    for raw_line in text.splitlines():
        line = raw_line.strip()
        m = _RE_ACTIVE_RES.search(line)
        if m:
            active_res = float(m.group(1))
        m = _RE_ACTIVE_FREQ.search(line)
        if m:
            active_freq = float(m.group(1))
        m = _RE_POWER.search(line)
        if m:
            power_mw = float(m.group(1))
        m = _RE_IDLE_RES.search(line)
        if m:
            idle_res = float(m.group(1))
            samples.append(
                {
                    "gpu_hw_active_residency_pct": active_res,
                    "gpu_hw_active_freq_mhz": active_freq,
                    "gpu_idle_residency_pct": idle_res,
                    "gpu_power_mw": power_mw,
                }
            )
            active_res = active_freq = idle_res = power_mw = None
    return samples


def sample_powermetrics(interval_ms: int = 200, n_samples: int = 2) -> dict:
    """Sample a short powermetrics gpu_power window.

    Uses `sudo -n` (NOPASSWD for powermetrics is granted in sudoers on the
    NODES). Window = interval_ms * n_samples, e.g. 200ms * 2 = ~400ms, so a
    checkpoint stays fast and does not meaningfully perturb a running bench.
    """
    cmd = [
        "sudo", "-n", "powermetrics",
        "--samplers", "gpu_power",
        "-i", str(interval_ms),
        "-n", str(n_samples),
    ]
    rc, out, err = _run(cmd, timeout=15.0)
    combined = out + err
    if rc != 0 or "GPU HW active" not in combined:
        # Distinguish password-gated from other failures for the doc.
        if "a password is required" in combined or "password" in combined.lower():
            return _err(
                f"powermetrics needs a password (sudo -n failed): rc={rc} "
                f"stderr={err.strip()!r}"
            )
        return _err(
            f"powermetrics failed or produced no GPU data: rc={rc} "
            f"stderr={err.strip()!r}"
        )
    samples = parse_powermetrics_gpu(combined)
    if not samples:
        return _err("powermetrics ran but no GPU sample blocks were parsed")
    return {
        "samples": samples,
        "n_samples": len(samples),
        "raw": combined,
        "error": None,
    }


# ---------------------------------------------------------------------------
# (c) memory_pressure: free/available memory + pressure percentage
# ---------------------------------------------------------------------------


def parse_memory_pressure(text: str) -> dict:
    """Parse `memory_pressure` output into structured fields."""
    result: dict = {}
    m = re.search(r"System-wide memory free percentage:\s*([\d.]+)%", text)
    if m:
        result["free_pct"] = float(m.group(1))
    m = re.search(r"Pages free:\s*(\d+)", text)
    if m:
        result["pages_free"] = int(m.group(1))
    m = re.search(r"Pages active:\s*(\d+)", text)
    if m:
        result["pages_active"] = int(m.group(1))
    m = re.search(r"Pages inactive:\s*(\d+)", text)
    if m:
        result["pages_inactive"] = int(m.group(1))
    m = re.search(r"Pages wired down:\s*(\d+)", text)
    if m:
        result["pages_wired"] = int(m.group(1))
    m = re.search(r"Pages purgeable:\s*(\d+)", text)
    if m:
        result["pages_purgeable"] = int(m.group(1))
    m = re.search(r"Pages used by compressor:\s*(\d+)", text)
    if m:
        result["pages_compressor"] = int(m.group(1))
    m = re.search(r"page size of (\d+)", text)
    if m:
        result["page_size_bytes"] = int(m.group(1))
    return result


def sample_memory_pressure() -> dict:
    rc, out, err = _run(["memory_pressure"], timeout=10.0)
    if rc != 0:
        return _err(f"memory_pressure failed: rc={rc} stderr={err.strip()!r}")
    parsed = parse_memory_pressure(out)
    if not parsed:
        return _err("memory_pressure ran but no fields were parsed")
    parsed["raw"] = out
    parsed["error"] = None
    return parsed


# ---------------------------------------------------------------------------
# (d) wired memory limit (iogpu.wired_limit_mb sysctl)
# ---------------------------------------------------------------------------


def parse_sysctl_wired(text: str) -> dict:
    """Parse `sysctl iogpu.wired_limit_mb` output."""
    m = re.search(r"iogpu\.wired_limit_mb:\s*(\d+)", text)
    if m:
        return {"wired_limit_mb": int(m.group(1))}
    return {}


def sample_wired_limit() -> dict:
    rc, out, err = _run(["sysctl", "iogpu.wired_limit_mb"], timeout=10.0)
    if rc != 0:
        return _err(f"sysctl iogpu.wired_limit_mb failed: rc={rc} stderr={err.strip()!r}")
    parsed = parse_sysctl_wired(out)
    if not parsed:
        return _err(f"sysctl ran but no iogpu.wired_limit_mb parsed from: {out.strip()!r}")
    parsed["raw"] = out
    parsed["error"] = None
    return parsed


# ---------------------------------------------------------------------------
# (e) runner process identity: PID + lstart (restart detection)
# ---------------------------------------------------------------------------

# Default patterns for identifying the exo runner process(es). The runner is
# launched as `.venv/bin/python -m exo -v` under a `screen -dmS exorun`
# session, but the exact name can vary (multiprocessing spawn children,
# batch_generator, etc.). We match the runner's distinctive signature and
# record ALL matches.
#
# NOTE: a bare `repos/exo/.venv/bin/python` path match is deliberately NOT
# used — it would match any shell wrapper that merely invokes that python
# (observed locally: a `bash -c` wrapper whose args contained the path
# matched, producing a false positive whose lstart changes on every new
# shell, which would corrupt restart detection). We anchor on the exo
# runtime marker (`-m exo` / `exo -v` / batch_generator / exo.worker /
# exo.main), which the actual runner and its multiprocessing spawn children
# carry and plain wrappers do not.
_DEFAULT_RUNNER_PATTERN = (
    r"(-m exo|exo -v|batch_generator|exo\.worker|exo\.main)"
)


def parse_ps_runner(text: str, pattern: str) -> list[dict]:
    """Parse `ps -axo pid,lstart,comm,args` output, keep matching rows.

    ps output columns (with = to suppress header):
        PID  LSTART  COMM  ARGS
    LSTART format: 'Tue Sep  1 13:20:44 2026' (may contain double spaces).
    """
    rx = re.compile(pattern)
    procs: list[dict] = []
    for line in text.splitlines():
        line = line.rstrip("\n")
        if not line.strip():
            continue
        # PID is the first whitespace-delimited token; the rest is
        # 'LSTART COMM ARGS' where LSTART has variable-width spaces.
        parts = line.split(None, 1)
        if len(parts) < 2:
            continue
        pid = parts[0]
        rest = parts[1]
        if not pid.isdigit():
            continue
        # LSTART is 4 tokens: 'Tue Sep  1 13:20:44 2026' -> split on 2+ spaces
        # to separate the fixed-width date from the rest.
        m = re.match(r"(\S+\s+\S+\s+\S+\s+\S+\s+\S+)\s+(.*)$", rest)
        if not m:
            continue
        lstart = m.group(1)
        comm_args = m.group(2)
        # comm is the first token of comm_args (may itself contain spaces if
        # the comm field is long, but for our purposes first token is enough).
        comm = comm_args.split(None, 1)[0] if comm_args else ""
        args = comm_args
        # Only the real runner runs under a python interpreter. The cluster is
        # launched as `screen -dmS exorun zsh -l -c "... .venv/bin/python -m exo -v ..."`,
        # so the SCREEN, login, and zsh WRAPPER processes all embed the literal
        # `-m exo` marker in their own args and would otherwise match the regex.
        # A wrapper re-exec (new shell, screen reattach) gives the wrapper a NEW
        # lstart while the real runner keeps its old one -- indistinguishable from
        # a genuine runner restart. Requiring the comm BASENAME to start with
        # `python` excludes those wrappers. The genuine runner and its
        # multiprocessing spawn children all run under a python interpreter
        # (mp.set_start_method("spawn") re-execs the interpreter), so this does
        # not drop any real runner. Do NOT re-widen this to accept an
        # `exo`-prefixed comm without re-verifying how runners are spawned.
        comm_base = comm.rsplit("/", 1)[-1]
        if comm_base.startswith("python") and rx.search(args):
            procs.append({"pid": int(pid), "lstart": lstart, "comm": comm, "args": args})
    return procs


def sample_runner(pattern: str) -> dict:
    rc, out, err = _run(
        ["ps", "-axo", "pid=,lstart=,comm=,args="], timeout=10.0
    )
    if rc != 0:
        return _err(f"ps failed: rc={rc} stderr={err.strip()!r}")
    procs = parse_ps_runner(out, pattern)
    return {
        "processes": procs,
        "count": len(procs),
        "pattern": pattern,
        "raw": out,
        "error": None,
    }


# ---------------------------------------------------------------------------
# Main: assemble one JSON object, append to JSONL
# ---------------------------------------------------------------------------


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Passive per-node telemetry sampler (read-only)."
    )
    parser.add_argument("label", help="checkpoint label, e.g. T0, warmup, rep1")
    parser.add_argument("--out", default=None, help="JSONL output file")
    parser.add_argument(
        "--runner-pattern",
        default=os.environ.get("EXO_TELEMETRY_RUNNER_PATTERN", _DEFAULT_RUNNER_PATTERN),
        help="regex for identifying the exo runner process(es)",
    )
    parser.add_argument(
        "--powermetrics-interval-ms", type=int, default=200,
        help="powermetrics sample interval in ms (default 200)",
    )
    parser.add_argument(
        "--powermetrics-samples", type=int, default=2,
        help="number of powermetrics samples (default 2)",
    )
    args = parser.parse_args()

    t0 = time.monotonic()
    record = {
        "timestamp": datetime.datetime.now().astimezone().isoformat(),
        "hostname": socket.gethostname(),
        "label": args.label,
        "mlx": sample_mlx_memory(),
        "powermetrics": sample_powermetrics(
            interval_ms=args.powermetrics_interval_ms,
            n_samples=args.powermetrics_samples,
        ),
        "memory_pressure": sample_memory_pressure(),
        "wired_limit": sample_wired_limit(),
        "runner": sample_runner(args.runner_pattern),
        "elapsed_seconds": round(time.monotonic() - t0, 3),
    }

    out_path = args.out or f"telemetry_{socket.gethostname()}.jsonl"
    with open(out_path, "a", encoding="utf-8") as f:
        f.write(json.dumps(record) + "\n")
    print(f"appended checkpoint '{args.label}' to {out_path}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
