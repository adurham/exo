# Phase 14 -- first end-to-end DSv4.1 decode on both nodes (TP=2)

Date: 2026-09-28
Host: macstudio-m4-2 (rank 0) + macstudio-m4-1 (rank 1), JACCL RDMA, exo's
production MLX fork (0.32.3.dev20260918+603f16eb7). Production was stopped
cleanly for the run (CLEAN_EXIT both nodes) and restored with a normal
`start_cluster.sh` afterwards (fresh pids, DSPARK_NATIVE=1, MTP_C2_MAX_CTX=1,
LMHEAD_MXFP8=0, real completion OK).

## Bottom line

DeepSeek-V4.1-Flash runs end to end across both Macs and writes correct text.
**Plain greedy decode: 9.25-9.30 tok/s** (107-108 ms/step, p10-p90 106.7-109.0),
no speculative decode. Memory: **105.9 GB active per rank** (fits the 115 GB
wired limit). Build 58 s per rank. The bar is 25 tok/s; the gap is almost all
software overhead in the unfused reference port, not bandwidth.

## Setup

`mlx_lm.models.deepseek_v41` (`feat/dsv41-exl3`) built with
`exl3_build.build_model(world=2)`: routed experts sliced to half intermediate
width per rank, routed partials `all_sum`ed, everything else replicated.
Chat template from the checkpoint, `enable_thinking=False`, 23-token prompt.

## Runs

| run | settings | decode | notes |
|---|---|---:|---|
| try 1 | no Metal-timeout env | -- | GPU Timeout on first warmup forward (one command buffer too long) |
| try 2 | production Metal-timeout env, eval every 4 layers | 9.02 tok/s | coherent 200-token answer |
| try 3 | same, no per-layer eval | **9.25 tok/s** | fencing only cost time |
| try 4 | same + profile | 9.30 tok/s | profile below |

Sample output (try 2, greedy): a correct, fluent explanation of Rayleigh
scattering and why the sky is not violet (full text in `raw/`).

## Where the 107 ms goes (fenced profile, 21 steps)

Each op synced before/after, so the total (250 ms) is inflated vs 107 ms real;
use it for shares only. `Exl3Proj` wraps every dense projection, so it
overlaps attention and shared expert -- do not add it.

| component | ms/step (fenced) | rough real floor |
|---|---:|---:|
| attention (projections, compressor, indexer, sparse attn, fake-quant, rope) | 93.3 | ~15-20 |
| hc_mixes (80 per token) | 50.9 | ~1-2 |
| routed experts | 44.9 | 17.2 (kernel microbench) |
| shared expert | 31.1 | ~3 |
| hc_post | 15.9 | <1 |
| gate | 8.1 | <1 |
| all_sum (40 per token) | 7.8 | <1 |
| engram | 2.7 | -- |

## Next (from the Fable consult, same day)

Fix the body before MTP: at 107 ms/step even good acceptance lands ~20 tok/s.
Ranked: (1) fuse the hyper-connection path (hc_mixes + hc_post); (2) find why
the shared expert costs ~10x its bandwidth; (3) close the in-situ routed
expert gap (44.9 vs 17.2); (4) attention fusion (sparse SDPA, fake-quant,
rope); (5) gate fusion, fewer/async all_sums; (6) then MTP/DSpark.
Estimated body floor after 1-5: ~35-45 ms/step = 22-28 tok/s before MTP.
Correction to the consult: the Sinkhorn input is NOT weight-only -- mixes
are `rms(x) * (x @ hc_fn.T)`, so it cannot be precomputed; it must be fused.

## Artifacts

- `raw/p47-r0-try2.log` (text), `raw/p47-r0-try3.log`, `raw/p47-r0-profile.log`
- `scripts/p47_tp2_decode.py`, `scripts/p47_launch.sh`
