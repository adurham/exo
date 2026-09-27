# Phase 0 Baseline — DeepSeek-V4-Flash-Vision-Exp @ 2K, 2026-09-26

Reference datapoint taken at the start of the DeepSeek-V4.1-Flash EXL3
migration (`docs/deepseek-v41-exl3-plan.md`, phase 0 gate). It exists so the
outgoing stack has ONE honest number attached to it before the kernel/model
swap — it is NOT a target and NOT a comparator for V4.1.

**V4-Flash and V4.1 are different models with a different kernel path.** The
user's own framing, verbatim: "v4 and v4.1 are two completely different
beasts / esp. with the custom work we have to do here to get it to fit on my
cluster." Numbers below are a rough sanity reference only; the comparison that
matters is the new stack's own gate.

## Configuration

- Model: `deepseek-ai/DeepSeek-V4-Flash-Vision-Exp` (the then-serving model)
- Ruler: `bench/long_decode_probe.py` — the repo's standing decode probe. It
  deliberately asks for a long answer so the decode window is hundreds of
  tokens; the repo's own rule is to never quote tok/s from sub-~400-token
  generations (startup noise dominates).
- Depth: 2000 target tokens (2331 prompt tokens realized)
- `--max-tokens 1400`, temperature 0.0, needle-in-haystack retrieval check
  (the probe seeds a fixed needle and reports `needle_hit`)
- Command: `EXO_API=http://127.0.0.1:52415 .venv/bin/python bench/long_decode_probe.py 2000 --max-tokens 1400 --tag phase0-baseline-2k --out /tmp/baseline_2k_rN.json`
- Run on macstudio-m4-1 against the two-node cluster, 10 consecutive runs,
  same config across all 10. Raw JSON + log for every run in `raw/`.

## Results (10 runs, server-side stats)

| run | prefill_s | prefill tok/s | decode_s | **server decode tok/s** | peak GB | mtp cycles | mtp accepted | needle |
|---:|---:|---:|---:|---:|---:|---:|---:|:--:|
| 1 | 7.30 | 342.32 | 66.72 | 20.960 | 99.0 | 3343 | 4099 | yes |
| 2 | 6.76 | 372.00 | 69.89 | 20.011 | 99.0 | 4126 | 4715 | yes |
| 3 | 6.70 | 362.01 | 64.09 | 21.822 | 99.0 | 4785 | 5457 | yes |
| 4 | 7.63 | 313.15 | 69.23 | 20.201 | 99.0 | 5546 | 6095 | yes |
| 5 | 7.14 | 339.38 | 67.79 | 20.633 | 99.0 | 6288 | 6753 | yes |
| 6 | 7.50 | 326.00 | 67.89 | 20.599 | 99.0 | 7037 | 7403 | yes |
| 7 | 7.52 | 330.18 | 64.15 | 21.798 | 99.0 | 7674 | 8166 | yes |
| 8 | 7.47 | 330.18 | 64.95 | 21.534 | 99.0 | 8357 | 8882 | yes |
| 9 | 7.47 | 323.66 | 67.19 | 20.816 | 99.0 | 9101 | 9538 | yes |
| 10 | 7.13 | 339.45 | 69.04 | 20.258 | 99.0 | 9874 | 10167 | yes |

**Summary**

| metric | min | median | mean | max | stdev |
|---|---:|---:|---:|---:|---:|
| server decode tok/s | 20.01 | 20.72 | 20.86 | 21.82 | 0.659 |
| prefill tok/s | 313.15 | 334.78 | 337.83 | 372.00 | — |

- Decode tok/s shown is the server-reported `generation_tps` from the streaming
  `: generation_stats` line (the trusted number), not the client-side figure.
- Needle retrieved in **10/10** runs — the quality signal held across the set.
- `mtp accepted/cycles` mean ≈ 1.094; MTP was active for every run.
- Peak memory 99.0 GB per node, well under the 115 GB wired limit.
- All runs hit `finish_reason=length` (1400 completion tokens), i.e. none ended
  early — the decode window is real, not truncated noise.

## What this is and isn't

- **Is:** one honest reference point for the outgoing V4-Flash/Vision-Exp stack
  recorded immediately before the V4.1 migration, on the same ruler every later
  phase reuses (same prompt, same depth, same probe), so later phases have a
  like-for-like "did we keep the harness honest" check.
- **Isn't:** a performance bar for V4.1. Per the plan, phase 2/5 gates are the
  new stack's own numbers; this table is not a promotion criterion for it.

## Provenance

- Recorded 2026-09-26 on macstudio-m4-1; cluster both nodes, exo at commit
  `698caaea` (the commit that added `docs/deepseek-v41-exl3-plan.md`).
- Raw per-run JSON and logs: `raw/baseline_2k_r{1..10}.json` / `.log`.
- Companion phase-0 records: model downloads (EXL3 210.7 GB + Engram 203 GB),
  HF-checksum verification (56/56 files), and the node1→node2 Thunderbolt rsync
  (per-file sizes verified identical on both nodes).
