#!/usr/bin/env python3
"""q2b: pull the EXL3 dsv41 history of exo_peak_memory_bytes from VictoriaMetrics (READ-ONLY GETs).

Purpose: (1) corroborate the x1.073741824 unit inflation of that gauge (GiB-vs-GB in
`Memory.from_gb(mx.get_peak_memory() / 1e9)`) and (2) give a HISTORICAL (other builds!) table of
peak vs the largest prompt seen, as a coarse cross-check of the slope. Stdlib only.

  python3 q2b_vm_history.py <out.json>

VM: http://172.16.0.42:8428  (`/api/v1/export` returns the RAW stored samples, no step interpolation).
"""

from __future__ import annotations

import datetime
import json
import sys
import urllib.parse
import urllib.request

VM = "http://172.16.0.42:8428"
MODEL = "dealignai/DeepSeek-V4.1-Flash-UNCENSORED-EXL3-2.9bpw"
K = 1024**3 / 1e9  # 1.073741824: Memory.from_gb() multiplies by 1024**3 but the engine divides by 1e9


def export(selector: str, start: int, end: int) -> list[dict]:
    url = VM + "/api/v1/export?" + urllib.parse.urlencode(
        {"match[]": selector, "start": start, "end": end})
    with urllib.request.urlopen(url, timeout=120) as r:  # noqa: S310 - fixed internal URL
        raw = r.read().decode()
    return [json.loads(ln) for ln in raw.strip().split("\n") if ln.strip()]


def main(out_path: str) -> int:
    end = int(datetime.datetime.now().timestamp())
    start = end - 30 * 86400
    peak = export(f'exo_peak_memory_bytes{{model_id="{MODEL}"}}', start, end)
    ptok = export(f'exo_prompt_tokens_total{{model_id="{MODEL}"}}', start, end)
    hits = export(f'exo_prefix_cache_hits_total{{model_id="{MODEL}"}}', start, end)
    # hits[(host, iid)][hit_kind] -> {ts: cumulative}
    ht: dict[tuple[str, str], dict[str, dict[int, float]]] = {}
    for r in hits:
        m = r["metric"]
        ht.setdefault((m["instance"], m["instance_id"]), {})[m.get("hit_kind", "?")] = {
            t: v for t, v in zip(r["timestamps"], r["values"]) if v is not None}
    pt: dict[tuple[str, str], dict[int, float]] = {}
    for r in ptok:
        m = r["metric"]
        pt[(m["instance"], m["instance_id"])] = {
            t: v for t, v in zip(r["timestamps"], r["values"]) if v is not None}
    rows = []
    for r in peak:
        m = r["metric"]
        key = (m["instance"], m["instance_id"])
        ts = r["timestamps"]
        vs = r["values"]
        P = pt.get(key, {})
        steps = []  # every upward step of the (monotone-within-process) gauge
        prev = None
        for i, (t, v) in enumerate(zip(ts, vs)):
            if v is None:
                continue
            if prev is not None and v > prev:
                d = None
                if t in P and i > 0 and ts[i - 1] in P:
                    d = P[t] - P[ts[i - 1]]
                hk = {}
                for kind, series in ht.get(key, {}).items():
                    if t in series and i > 0 and ts[i - 1] in series:
                        dv = series[t] - series[ts[i - 1]]
                        if dv:
                            hk[kind] = dv
                steps.append({"t": datetime.datetime.fromtimestamp(t / 1000).isoformat(timespec="seconds"),
                              "labelled_B": v, "prompt_tokens_in_scrape_window": d,
                              "hit_kind_increments": hk})
            prev = v
        vals = [v for v in vs if v is not None]
        rows.append({
            "host": m["instance"], "instance_id": m["instance_id"],
            "first_sample": datetime.datetime.fromtimestamp(ts[0] / 1000).isoformat(timespec="seconds"),
            "last_sample": datetime.datetime.fromtimestamp(ts[-1] / 1000).isoformat(timespec="seconds"),
            "n_samples": len(vals),
            "first_labelled_GB": vals[0] / 1e9, "max_labelled_GB": max(vals) / 1e9,
            "first_true_GB": vals[0] / 1e9 / K, "max_true_GB": max(vals) / 1e9 / K,
            "n_up_steps": len(steps), "up_steps": steps,
            "max_prompt_tokens_in_any_up_step": max(
                (s["prompt_tokens_in_scrape_window"] for s in steps
                 if s["prompt_tokens_in_scrape_window"] is not None), default=None),
        })
    rows.sort(key=lambda x: x["first_sample"])
    with open(out_path, "w") as f:
        json.dump({"query": f'exo_peak_memory_bytes{{model_id="{MODEL}"}}', "unit_factor_K": K,
                   "window": [start, end], "instances": rows}, f, indent=1)
    print(f"wrote {out_path}: {len(rows)} instance series")
    return 0


if __name__ == "__main__":
    sys.exit(main(sys.argv[1] if len(sys.argv) > 1 else "q2b_raw_vm_instances_dsv41.json"))
