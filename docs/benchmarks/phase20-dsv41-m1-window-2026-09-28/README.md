# Phase 20 -- merged workstreams, first two-node window (M1)

Date: 2026-09-28. mlx-lm feat/dsv41-exl3 acf9b55 (squash of workstreams
A-F,H). Production stopped cleanly and restored (fresh pids, env verified,
real completion OK).

| prompt | prefill | tok/s | decode right after | GPU timeouts |
|---:|---:|---:|---:|---:|
| 8,192 | 62.8 s | 130.4 | 1.6 tok/s | 0 |
| 16,384 | 225.4 s | 72.7 | 5.0 tok/s | 0 (was 527 = crash) |
| 32,768 | stopped after >12 min | -- | -- | 0 |

Warmup 44 s at load. Peak 105.3 GB/rank.

- The 16K crash is fixed (indexer tiling + fences).
- Prefill cost grows faster than linear: 2x context -> 3.6x time. Something
  in the per-chunk work still scales with total context (indexer score pass
  over nb, candidate selection, or the 128-token chunk size used above 8K:
  1.35 ms/tok at 512 rows vs 2.47 at 128 on the subset harness).
- Decode right after a long prompt collapses (1.6-5 tok/s vs 16.9 at short
  context). Cause not found yet; only 16 steps were timed, so a first-shape
  compile may be part of it.
- Not usable as the default yet.

Workstream G's exo engine scaffolding had hallucinated placeholder strings for
the DSML sentinel and thinking markers; corrected from the checkpoint
tokenizer ("\uff5cDSML\uff5c", "<think>", "</think>"). Engine not wired or
tested yet (exo branch ws/G-engine 926adabb).
