#!/usr/bin/env python3
"""q2b: fetch raw system-memory samples from VictoriaMetrics (READ-ONLY GET /api/v1/export; stdlib only).

Produces the two raw files the model reads:
  q2b_raw_vm_sys_series.json         current boot window (2026-10-09 22:30 CDT -> now): ram_used / swap_used / ram_total, both nodes
  q2b_raw_vm_ram_soak13_window.json  soak13 window (2026-10-07 03:55 -> 08:20 CDT): ram_used + swap_used, both nodes, [epoch_s, bytes]

`exo_memory_ram_used_bytes` = ram_total - ram_available (exo metrics.py:416); `ram_available` comes from macmon
(utils/info_gatherer/macmon.py:62: ram_total - ram_usage). It is a SYSTEM number (all processes), 15 s grid.

  python3 q2b_fetch_sys_series.py
"""

from __future__ import annotations

import datetime as dt
import json
import pathlib
import urllib.parse
import urllib.request

HERE = pathlib.Path(__file__).resolve().parent
VM = "http://172.16.0.42:8428"
CDT = dt.timezone(dt.timedelta(hours=-5))
HOSTS = ("macstudio-m4-1", "macstudio-m4-2")


def export(selector: str, start: float, end: float) -> list[dict]:
    url = VM + "/api/v1/export?" + urllib.parse.urlencode(
        {"match[]": selector, "start": int(start), "end": int(end)})
    with urllib.request.urlopen(url, timeout=120) as r:  # noqa: S310 - fixed internal URL
        raw = r.read().decode()
    return [json.loads(ln) for ln in raw.strip().split("\n") if ln.strip()]


def main() -> int:
    now = dt.datetime.now(CDT)
    # ---- current boot window
    a = dt.datetime(2026, 10, 9, 22, 30, tzinfo=CDT).timestamp()
    series: dict[str, list[dict]] = {}
    for host in HOSTS:
        for name in ("exo_memory_ram_used_bytes", "exo_memory_swap_used_bytes", "exo_memory_ram_total_bytes"):
            rows = export(f'{name}{{instance="{host}"}}', a, now.timestamp())
            series[f"{name}|{host}"] = [
                {"labels": {k: v for k, v in r["metric"].items() if k != "__name__"},
                 "timestamps_ms": r["timestamps"], "values": r["values"]} for r in rows]
    (HERE / "q2b_raw_vm_sys_series.json").write_text(json.dumps({
        "note": "VM /api/v1/export raw samples; window 2026-10-09 22:30 CDT -> fetch time. ram_used = ram_total - ram_available "
                "(exo metrics.py:416); swap_used likewise. Fetched " + now.isoformat(timespec="seconds"),
        "series": series}))
    # ---- soak13 window (instance 38ed8ddb, build next13 f0840af1c / mlx-lm 6cc9c1e)
    s0 = dt.datetime(2026, 10, 7, 3, 55, tzinfo=CDT).timestamp()
    s1 = dt.datetime(2026, 10, 7, 8, 20, tzinfo=CDT).timestamp()
    out: dict[str, dict[str, list[list[float]]]] = {"ram": {}, "swap": {}}
    for host in HOSTS:
        for key, name in (("ram", "exo_memory_ram_used_bytes"), ("swap", "exo_memory_swap_used_bytes")):
            rows = export(f'{name}{{instance="{host}"}}', s0, s1)
            pts = sorted((t / 1000.0, v) for r in rows for t, v in zip(r["timestamps"], r["values"]) if v is not None)
            out[key][host] = [[t, v] for t, v in pts]
    (HERE / "q2b_raw_vm_ram_soak13_window.json").write_text(json.dumps({
        "note": "VM raw samples (epoch_s, bytes) for the soak13 window 2026-10-07 03:55-08:20 CDT (next13). ram = exo_memory_ram_used_bytes, "
                "swap = exo_memory_swap_used_bytes. Fetched " + now.isoformat(timespec="seconds"),
        "series": out["ram"], "swap": out["swap"]}))
    print("wrote q2b_raw_vm_sys_series.json and q2b_raw_vm_ram_soak13_window.json:",
          {k: len(v) for k, v in out["ram"].items()}, {k: len(v) for k, v in out["swap"].items()})
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
