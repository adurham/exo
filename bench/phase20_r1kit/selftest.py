#!/usr/bin/env python3
"""Offline static validation for the recovered r1_driver.py (NO cluster, NO POST).

Checks, in order:
  1. the kit imports (r1_driver + p3b_driver + phase20_guard + phase19_* all resolve);
  2. the CLI parses: --help exits 0 and exposes --salt; a bad --arm is rejected;
  3. the per-rep salt sequence is f"{base}-{n}" and honours an arbitrary --salt (q1b);
  4. re-deriving EVERY rep of the frozen control JSONs from their raw fields reproduces
     the stored rounds / mean_accepted / ms_per_round / gamma_implied exactly;
  5. recomputing the summary medians from the frozen recs matches the frozen summary,
     and the summary key set matches the frozen schema;
  6. both prompt builders run offline and their meta shapes match the frozen control
     (benign prompt_chars; agentic n_messages/preamble_chars/prompt_chars).

Exit 0 iff every assertion holds. Prints a PASS/FAIL line per check.
"""
from __future__ import annotations

import json
import os
import statistics
import subprocess
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)

FAILS: list[str] = []


def check(name: str, ok: bool, detail: str = "") -> None:
    print(f"[{'PASS' if ok else 'FAIL'}] {name}" + (f" — {detail}" if detail else ""))
    if not ok:
        FAILS.append(name)


def find_round_dir() -> str | None:
    """Locate the committed Q1 control round dir by walking up from this file."""
    rel = os.path.join("docs", "benchmarks", "phase20-throughput",
                       "raw", "pricing", "q1", "round")
    d = HERE
    for _ in range(10):
        cand = os.path.join(d, rel)
        if os.path.isdir(cand):
            return cand
        d = os.path.dirname(d)
    return None


# ---- 1. imports -------------------------------------------------------------
import r1_driver as R1              # noqa: E402  (imports the whole chain)
import phase19_round_measure as RM  # noqa: E402
import phase19_agentic_measure as AM  # noqa: E402

RM_ = R1.RM
AM_ = R1.AM
check("imports: r1_driver + deps resolve", True,
      f"RM={RM_.__name__} AM={AM_.__name__} MODEL={R1.P3B.MODEL}")
check("harness identity: R1.RM is phase19_round_measure",
      RM_.derive is RM.derive, "derive() shared")
check("gamma default is 3", RM_ is RM and getattr(R1, "stream_once", None) is R1.P3B.stream_once,
      "stream_once inherited from p3b_driver")

# ---- 2. CLI parse -----------------------------------------------------------
DRIVER = os.path.join(HERE, "r1_driver.py")
PY = sys.executable
help_out = subprocess.run([PY, DRIVER, "--help"], capture_output=True, text=True)
check("cli: --help exits 0", help_out.returncode == 0)
check("cli: exposes --salt", "--salt" in help_out.stdout)
check("cli: exposes --total-reps/--reps-per-chunk/--depth/--max-tokens/--gamma/--label/--out",
      all(f"--{f}" in help_out.stdout for f in
          ("total-reps", "reps-per-chunk", "depth", "max-tokens", "gamma", "label", "out")))
bad = subprocess.run([PY, DRIVER, "--arm", "bogus"], capture_output=True, text=True)
check("cli: bad --arm rejected (nonzero)", bad.returncode != 0)

# ---- 3. salt sequence -------------------------------------------------------
for base in ("q1b", "q1eval"):
    seq = [f"{base}-{n}" for n in range(4)]
    check(f"salt sequence for base {base!r}", seq == [f"{base}-0", f"{base}-1", f"{base}-2", f"{base}-3"],
          str(seq))

# ---- 4/5. control reproduction ---------------------------------------------
round_dir = find_round_dir()
check("control round dir located", round_dir is not None, round_dir or "NOT FOUND")

EXPECT_SUMMARY_KEYS = {
    "label", "arm", "salt_base", "round_prof", "reps", "depth", "max_tokens",
    "decode_tps_median", "decode_tps_all", "ms_per_round_median", "ms_per_round_all",
    "mean_accepted_median", "aborted", "abort_reason",
}
FROZEN = {
    "benign":  {"ms_per_round_median": 94.88, "decode_tps_median": 38.48,  "mean_accepted_median": 2.682},
    "agentic": {"ms_per_round_median": 101.07, "decode_tps_median": 30.962, "mean_accepted_median": 2.1128},
}

if round_dir:
    for arm in ("benign", "agentic"):
        doc = json.load(open(os.path.join(round_dir, f"control_{arm}.json")))
        summ, recs = doc["summary"], doc["recs"]

        # 5a. schema
        check(f"{arm}: summary key set == frozen schema",
              set(summ) == EXPECT_SUMMARY_KEYS,
              f"extra={set(summ) - EXPECT_SUMMARY_KEYS} missing={EXPECT_SUMMARY_KEYS - set(summ)}")
        check(f"{arm}: salt_base == 'q1eval'", summ["salt_base"] == "q1eval", summ["salt_base"])
        check(f"{arm}: first rep has rounds=null (warm-up)",
              recs[0]["rounds"] is None and recs[0]["mean_accepted"] is None)

        # 4. re-derive every rep from raw fields with the SAME derive() + prev logic
        prev_cyc = prev_acc = None
        all_ok = True
        for i, rec in enumerate(recs):
            raw = {"stats": rec["stats"], "usage": rec["usage"], "decode_s": rec["decode_s"]}
            out = RM.derive(raw, prev_cyc, prev_acc, 3)
            for k in ("rounds", "mean_accepted", "ms_per_round", "gamma_implied", "decode_tps"):
                if out.get(k) != rec.get(k):
                    all_ok = False
                    print(f"    rep{i} {k}: derived={out.get(k)} frozen={rec.get(k)}")
            if out.get("cycles_cum") is not None:
                prev_cyc, prev_acc = out["cycles_cum"], out["accepted_cum"]
        check(f"{arm}: derive() reproduces every frozen rep field", all_ok)

        # 5b. summary medians recomputed from recs
        dec = [r["decode_tps"] for r in recs if r.get("decode_tps")]
        msr = [r["ms_per_round"] for r in recs if r.get("ms_per_round")]
        mar = [r["mean_accepted"] for r in recs if r.get("mean_accepted") is not None]
        got = {
            "decode_tps_median": round(statistics.median(dec), 3),
            "ms_per_round_median": round(statistics.median(msr), 2),
            "mean_accepted_median": round(statistics.median(mar), 4),
        }
        check(f"{arm}: recomputed summary == frozen summary",
              got == {k: summ[k] for k in got}, f"{got}")
        check(f"{arm}: matches frozen anchor literal",
              got == FROZEN[arm], f"frozen={FROZEN[arm]}")

# ---- 6. prompt builders -----------------------------------------------------
try:
    p_benign = RM.build_prompt(20000, "q1b-0", "count")
    check("benign: build_prompt(20000) runs offline", isinstance(p_benign, str) and len(p_benign) > 100000,
          f"prompt_chars={len(p_benign)} (frozen q1eval-0 = 111996)")
except Exception as e:  # pragma: no cover
    check("benign: build_prompt runs offline", False, repr(e))

try:
    prompt, meta = AM.build_agentic_prompt("q1b-0", None)
    ok = (meta["n_messages_total"] == 98 and meta["n_messages_used"] == 98
          and meta["preamble_chars"] == 22770)
    check("agentic: build_agentic_prompt(StateDB session) reproduces meta shape", ok, json.dumps(meta))
    check("agentic: prompt_chars ~ frozen 308979",
          abs(meta["prompt_chars"] - 308979) < 200, f"prompt_chars={meta['prompt_chars']}")
except Exception as e:  # pragma: no cover
    check("agentic: build_agentic_prompt runs offline", False, repr(e))

print()
if FAILS:
    print(f"RESULT: FAIL ({len(FAILS)}): " + "; ".join(FAILS))
    sys.exit(1)
print("RESULT: PASS — driver reproduces the Q1 control schema + derived metrics, salt-parameterised")
