#!/usr/bin/env python3
"""Re-verdict ONE label's stored prose bodies with the FINAL detector code.

The fp32A prose phase ran under two earlier revisions of battery.py (noisy
same_script_glue, small budgets). Rather than re-queue the cluster, recompute
every prose verdict offline from the STORED raw bodies + content with the
current run_detectors() and rewrite free_prose/index.json (+ per-prompt files).

Content that was empty at capture stays empty (recorded as REASONING_ONLY when
the stored body carries reasoning, else ERROR) — the arm difference for those
prompts is surfaced by compare.py as a note, never a false FAIL.
"""
import importlib.util
import json
import os
import sys

BATTERY = os.path.join(os.path.dirname(__file__), "..", "quality_battery", "battery.py")

def load_battery():
    spec = importlib.util.spec_from_file_location("battery_final", BATTERY)
    m = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(m)
    return m

def main(label_dir):
    b = load_battery()
    fpdir = os.path.join(label_dir, "free_prose")
    idx_path = os.path.join(fpdir, "index.json")
    index = json.load(open(idx_path))
    probes = index.get("probes") or []
    changed = []
    for rec in probes:
        pid = rec["id"]
        fpath = os.path.join(fpdir, "{}.json".format(pid))
        disk = json.load(open(fpath))
        content = disk.get("content") or ""
        rawpath = os.path.join(label_dir, "raw", "prose_{}.json".format(pid))
        reasoning = ""
        if os.path.exists(rawpath):
            try:
                body = json.loads(json.load(open(rawpath))["raw_body"])
                m = (body.get("choices") or [{}])[0].get("message") or {}
                reasoning = m.get("reasoning_content") or ""
            except Exception:
                pass
        if content:
            det = b.run_detectors(content)
        elif reasoning.strip():
            det = {"verdict": "REASONING_ONLY", "hits": [], "counts": {}}
        else:
            det = {"verdict": "ERROR", "hits": [], "counts": {}}
        old = (rec.get("detector") or {}).get("verdict")
        if old != det["verdict"]:
            changed.append((pid, old, det["verdict"]))
        rec["detector"] = det
        disk["detector"] = det
        with open(fpath, "w") as f:
            json.dump(disk, f, indent=2, ensure_ascii=False)
    dirty = [r for r in probes if r["detector"]["verdict"] == "DIRTY"]
    review = [r for r in probes if r["detector"]["verdict"] == "REVIEW"]
    r_only = [r for r in probes if r["detector"]["verdict"] == "REASONING_ONLY"]
    errors = [r for r in probes if r["detector"]["verdict"] == "ERROR"]
    index["n_dirty"], index["n_review"] = len(dirty), len(review)
    index["probes"] = probes
    with open(idx_path, "w") as f:
        json.dump(index, f, indent=2, ensure_ascii=False)
    print("re-verdict {}: {} probes | DIRTY {} REVIEW {} REASONING_ONLY {} ERROR {}".format(
        label_dir, len(probes), len(dirty), len(review), len(r_only), len(errors)))
    for pid, old, new in changed:
        print("  {}: {} -> {}".format(pid, old, new))
    print("  remaining non-CLEAN:",
          [(r["id"], r["detector"]["verdict"]) for r in probes
           if r["detector"]["verdict"] != "CLEAN"])

if __name__ == "__main__":
    main(sys.argv[1] if len(sys.argv) > 1 else
         os.path.expanduser("~/.hermes/cache/scratch/quality_battery/results/fp32A"))
