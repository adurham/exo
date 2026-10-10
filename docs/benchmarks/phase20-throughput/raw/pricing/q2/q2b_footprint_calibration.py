#!/usr/bin/env python3
"""q2b: LOCAL (laptop) calibration of the `footprint` / `vmmap` / libproc "peak" counters against a known MLX workload.

Why this exists. The Q1E-A2 note took `footprint -p <runner>` "phys_footprint_peak: 114 GB" as the production runner's resident
peak. A read-only `proc_pid_rusage(RUSAGE_INFO_V4)` of the SAME pid gave ri_lifetime_max_phys_footprint = 122,126,271,536 B (studio1)
and ri_phys_footprint(now) = 114,345,083,928 B while `footprint -p` printed now=106 GB / peak=114 GB, i.e. ~8% apart.
RESOLUTION (this script + q2b_footprint_unit_probe.py): `footprint`'s default output uses BINARY units (it prints GiB/MiB as
"GB"/"MB"): 122,126,271,536 B = 113.74 GiB ("114 GB"), 114,345,083,928 B = 106.49 GiB ("106 GB"). `-f bytes`, libproc and `vmmap`
(also binary: "113.7G") all agree on the same quantity. There is no tool disagreement; there is a unit slip in the A2 note.

Workload (all sizes decimal GB, fp32 ones, every byte written by a kernel => resident):
  t0 baseline -> t1 hold A=4 GB -> t2 allocate+free a transient B=3 GB -> t3 release A.
At each step it records mlx active/peak/cache, `footprint -f bytes` (now, peak), vmmap ("Physical footprint" now/peak, binary G)
and proc_pid_rusage (now, lifetime_max, interval_max). Observed (q2b_footprint_calibration.out.txt): the three peak counters are
true lifetime high-waters (they keep 7.162 GB after B and A are freed) and track mlx get_peak_memory + ~0.16 GB host.

Stdlib + mlx only. Run:  /Users/adam.durham/repos/exo/.venv/bin/python q2b_footprint_calibration.py
"""

from __future__ import annotations

import ctypes
import ctypes.util
import json
import os
import re
import subprocess
import sys
import time

import mlx.core as mx

_RU_FIELDS = [
    "ri_user_time", "ri_system_time", "ri_pkg_idle_wkups", "ri_interrupt_wkups", "ri_pageins",
    "ri_wired_size", "ri_resident_size", "ri_phys_footprint", "ri_proc_start_abstime",
    "ri_proc_exit_abstime", "ri_child_user_time", "ri_child_system_time",
    "ri_child_pkg_idle_wkups", "ri_child_interrupt_wkups", "ri_child_pageins",
    "ri_child_elapsed_abstime", "ri_diskio_bytesread", "ri_diskio_byteswritten",
    "ri_cpu_time_qos_default", "ri_cpu_time_qos_maintenance", "ri_cpu_time_qos_background",
    "ri_cpu_time_qos_utility", "ri_cpu_time_qos_legacy", "ri_cpu_time_qos_user_initiated",
    "ri_cpu_time_qos_user_interactive", "ri_billed_system_time", "ri_serviced_system_time",
    "ri_logical_writes", "ri_lifetime_max_phys_footprint", "ri_instructions", "ri_cycles",
    "ri_billed_energy", "ri_serviced_energy", "ri_interval_max_phys_footprint",
    "ri_runnable_time",
]


class _RU(ctypes.Structure):
    _fields_ = [("ri_uuid", ctypes.c_uint8 * 16)] + [(n, ctypes.c_uint64) for n in _RU_FIELDS]


_LIBPROC = ctypes.CDLL(ctypes.util.find_library("proc"))


def rusage(pid: int) -> dict[str, int]:
    ru = _RU()
    rc = _LIBPROC.proc_pid_rusage(ctypes.c_int(pid), ctypes.c_int(4), ctypes.byref(ru))
    if rc != 0:
        raise RuntimeError(f"proc_pid_rusage rc={rc}")
    return {
        "now": int(ru.ri_phys_footprint),
        "lifetime_max": int(ru.ri_lifetime_max_phys_footprint),
        "interval_max": int(ru.ri_interval_max_phys_footprint),
        "resident": int(ru.ri_resident_size),
        "wired": int(ru.ri_wired_size),
    }


def footprint(pid: int) -> dict[str, object]:
    out = subprocess.run(["footprint", "-f", "bytes", "-p", str(pid)],
                         capture_output=True, text=True, timeout=180).stdout
    now = re.search(r"phys_footprint:\s*(\d+)", out)
    peak = re.search(r"phys_footprint_peak:\s*(\d+)", out)
    return {"now": int(now.group(1)) if now else None,
            "peak": int(peak.group(1)) if peak else None,
            "raw_aux": [ln.strip() for ln in out.splitlines() if "phys_footprint" in ln]}


def _to_bytes(num: str, unit: str) -> int:
    mult = {"K": 1 << 10, "M": 1 << 20, "G": 1 << 30, "T": 1 << 40}[unit.upper()]
    return int(float(num) * mult)


def vmmap(pid: int) -> dict[str, object]:
    out = subprocess.run(["vmmap", "-summary", str(pid)], capture_output=True, text=True,
                         timeout=180).stdout
    res: dict[str, object] = {}
    for key, label in (("now", r"Physical footprint:"), ("peak", r"Physical footprint \(peak\):")):
        m = re.search(label + r"\s*([\d.]+)([KMGT])", out)
        res[key] = _to_bytes(m.group(1), m.group(2)) if m else None  # vmmap uses 1024-based units
    return res


def snap(tag: str, pid: int) -> dict[str, object]:
    row = {
        "tag": tag,
        "mlx": {"active": mx.get_active_memory(), "peak": mx.get_peak_memory(),
                "cache": mx.get_cache_memory()},
        "footprint_tool": footprint(pid),
        "vmmap_tool": vmmap(pid),
        "rusage": rusage(pid),
    }
    return row


def main() -> int:
    pid = os.getpid()
    gb = 10**9
    rows = []
    mx.eval(mx.ones((8,)))
    rows.append(snap("t0_baseline", pid))
    a = mx.ones((1 * gb,), dtype=mx.float32)          # 4.0 GB, held
    mx.eval(a)
    rows.append(snap("t1_hold_A_4GB", pid))
    b = mx.ones((750_000_000,), dtype=mx.float32)      # 3.0 GB transient
    mx.eval(b)
    rows.append(snap("t2a_hold_A_plus_B_3GB", pid))
    del b
    mx.clear_cache()
    time.sleep(1.0)
    rows.append(snap("t2b_B_freed_clear_cache", pid))
    del a
    mx.clear_cache()
    time.sleep(1.0)
    rows.append(snap("t3_all_freed", pid))

    def g(x: int | None) -> str:
        return "None" if x is None else f"{x / 1e9:8.3f}"

    print(f"{'step':26s} | {'mlx act':>8s} {'mlx peak':>9s} {'mlx cache':>9s} | "
          f"{'fp now':>8s} {'fp peak':>8s} | {'vmm now':>8s} {'vmm peak':>8s} | "
          f"{'ru now':>8s} {'ru life':>8s} {'ru intv':>8s}   (GB, decimal)")
    for r in rows:
        m, f, v, u = r["mlx"], r["footprint_tool"], r["vmmap_tool"], r["rusage"]
        print(f"{r['tag']:26s} | {g(m['active'])} {g(m['peak'])} {g(m['cache'])} | "
              f"{g(f['now'])} {g(f['peak'])} | {g(v['now'])} {g(v['peak'])} | "
              f"{g(u['now'])} {g(u['lifetime_max'])} {g(u['interval_max'])}")
    print(json.dumps({"pid": pid, "mlx_version": getattr(mx, "__version__", None),
                      "rows": rows}, indent=1))
    return 0


if __name__ == "__main__":
    sys.exit(main())
