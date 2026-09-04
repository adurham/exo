#!/usr/bin/env python3
"""Round-9 summarizer: reads results/<LABEL>_{2k,short}_r*.json and prints
median + full range + prompt_tokens range + prefix_cache_hit audit + residual.
residual_ms = prefill_s*1000 - ((prompt_tokens-1)/prompt_tps)*1000   [DIAGNOSTIC ONLY]
"""
import json
import statistics
import sys
from pathlib import Path

RES = Path(__file__).resolve().parent / "results"


def load(label, kind, n):
    rows = []
    for i in range(1, n + 1):
        p = RES / f"{label}_{kind}_r{i}.json"
        if not p.exists():
            continue
        d = json.loads(p.read_text())
        ss = d.get("server_stats") or {}
        ptok = d.get("prompt_tokens")
        ptps = ss.get("prompt_tps")
        ttft = d["prefill_s"] * 1000.0
        resid = None
        if ptok and ptps:
            resid = ttft - ((ptok - 1) / ptps) * 1000.0
        rows.append({
            "tag": d["tag"], "ttft_ms": round(ttft, 1), "ptok": ptok,
            "prompt_tps": ptps, "resid_ms": round(resid, 1) if resid is not None else None,
            "hit": ss.get("prefix_cache_hit", "MISSING"),
            "ctok": d.get("completion_tokens"),
        })
    return rows


def summarize(label, kind, n):
    rows = load(label, kind, n)
    if not rows:
        return None
    t = [r["ttft_ms"] for r in rows]
    pt = [r["ptok"] for r in rows if r["ptok"]]
    rs = [r["resid_ms"] for r in rows if r["resid_ms"] is not None]
    tps = [r["prompt_tps"] for r in rows if r["prompt_tps"]]
    return {
        "label": label, "kind": kind, "n": len(rows),
        "ttft_median": round(statistics.median(t), 1),
        "ttft_range": [min(t), max(t)],
        "ttft_all": t,
        "ptok_range": [min(pt), max(pt)] if pt else None,
        "prompt_tps_median": round(statistics.median(tps), 2) if tps else None,
        "resid_median": round(statistics.median(rs), 1) if rs else None,
        "resid_range": [min(rs), max(rs)] if rs else None,
        "cache_hits": sorted({str(r["hit"]) for r in rows}),
        "rows": rows,
    }


if __name__ == "__main__":
    labels = sys.argv[1:] or ["A", "Z1", "B", "Z2"]
    out = {}
    for lb in labels:
        for kind, n in (("2k", 5), ("short", 10)):
            s = summarize(lb, kind, n)
            if s:
                out[f"{lb}_{kind}"] = s
    print(json.dumps(out, indent=1))
