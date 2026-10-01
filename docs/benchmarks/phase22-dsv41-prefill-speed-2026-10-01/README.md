# Phase 22 — DSv4.1 prefill speed, Stage A (baseline + attribution + config levers)

Date: 2026-10-01. Trees under test: exo `6909d121`, mlx-lm `ad6f9d3` (the exact
trees production served before this window; every file under test md5-matched
between the deployed `~/dsv41-test` tree and the repo checkouts).
Two-node TP=2 over JACCL/RDMA, layers spread as production places them.
Production was stopped cleanly and restored afterwards — see §7.

## Bottom line

There is **no large prefill win in the config-only knobs**, and the reason is
now measured: at 16K the prefill is dominated by the EXL3 expert matmuls
(30.7%) plus the dense EXL3 projections (16.6%) and the cross-rank `all_sum`
collectives (16.9%); attention+indexer together are only ~19%. Two exceptions
worth acting on, both measured:

- **Chunk size 2048 → 4096 is worth +5.9% / +7.1% at 8K / 16K** (reproduced in
  both reps), but it pushes peak to 116.7 GB at 16K — **over the 115 GB wired
  limit**, so it is not shippable without reclaiming memory first.
- **Chunk 8192 is worse than 2048** (229.6 vs 231.8 tok/s at 16K) and peaks at
  117.4 GB. Bigger is not better past 4096.
- `MLX_MAX_OPS_PER_BUFFER` is **neutral for prefill** at 24 and at 5 (all within
  run noise of 200); it is not the lever it was for the attention-only bench.
- Byte-identical output confirmed for every arm that does not change chunk
  boundaries (max|Δlogit| = 0, top-1 identical, identical 64-token sha256).

So a ≥2× TTFT gain at 8K/16K is **not reachable from these knobs**; it has to
come from the expert/dense GEMM path or the collective, which is Stage B
territory. Nothing in this report changes a production default.

## 1. What production actually does today (read from the code, verified)

The served model's prefill is exo's `dsv41/session.py::engine_prefill`, called
with `chunk = EXO_PREFILL_STEP_SIZE = 2048`:

- uniform 2048-row chunks; the engine passes `long_threshold=10**9`, so
  `EXO_PREFILL_STEP_SIZE_HIGH_CTX=128` and `EXO_PREFILL_STEP_SIZE_CROSSOVER=500000`
  in production's env are **dead paths** for this model;
- `return_taps=True` with a `taps_out` list, so every chunk's DSpark taps are
  projected and pushed into the draft head (`head.append_ctx`) as part of the
  turn — this is real per-chunk work the older speed measurements never paid;
- `last_logit_only=True` on every chunk, `argmax=False`, ONE `mx.eval` per
  chunk, **no eval fences**, no periodic `mx.clear_cache`.

The harness calls **this function** (its source is extracted verbatim with
`ast` from the deployed `session.py` into
`raw/run/engine_prefill_extracted.py`; the file's md5 is recorded in every
run's log: `bcfb09f661315b5783661875232db690`), not a reimplementation.

## 2. Served baseline (production up, through the API)

Fixed prompts (`scripts/p22_prep.py`, sha256-pinned in
`raw/prompt-manifest.json`), `use_prefix_cache=false`, streamed so TTFT is the
first content token. Raw: `raw/api-baseline.jsonl`.

| prompt | prompt tokens | TTFT | prompt tok/s |
|---:|---:|---:|---:|
| 2K | 2052 | 9.389 s | 218.6 |
| 8K | 8196 | 36.598 s | 223.9 |
| 16K | — | **refused** | — |

**Production cannot serve a 16K prompt at all.** The request is refused with:

```
DSV4.1: prompt 16388 + max_output_tokens 8 needs more than the 16384-token
cache this instance was configured for (max_kv_tokens / card context_length)
```

The instance's `maxKvTokens` is 16384 and the chat template adds 4 tokens on top
of the 16384 raw ids, so a 16K-prompt request is rejected before any prefill
runs. Any 16K number in this report is therefore a **harness projection**, not a
served measurement — labelled as such everywhere below.

## 3. Harness baseline and the shape effects (two-node, production down)

All arms: real `engine_prefill`, fixed prompts, 64-token greedy oracle.
`raw/analysis-all-runs.txt`; per-run `results.json` + logs in `raw/{run,run2,run3,ops24,ops5}/`
and `raw/logs/`.

### 3.1 Baseline is reproducible and the oracle is stable

| arm | 8K tok/s | 16K tok/s |
|---|---:|---:|
| `run` (ops=200) | 239.1 / 245.6 | 239.6 / 238.4 |
| `run2` (ops=200, independent process + cold load) | 237.8 / 244.1 | 238.1 / 237.4 |

Run-to-run within a process and across a full cold restart agree to ~±1%. Every
one of the 12 arms in `run` and `run2` produced the **same 64-token sha256**
(`2K b1774115…`, `8K fde9cc37…`, `16K b270128b…`), and the `pf_fenced`,
`notaps` and `msl16384` variants are **bit-identical to `base`**
(`max|Δlogit| = 0.0`, mean 0.0, top-1 identical) — the oracle is byte-stable
across processes, so the brief's requirement (baseline twice + once after a cold
restart) is met.

### 3.2 Per-chunk time does NOT grow with chunk position

16K prompt, chunk 2048, per-chunk wall (ms), from the driver's own progress hook
(`run`, rep1):

```
c0(+2048) 8255  c1(+4096) 8403  c2(+6144) 8490  c3(+8192) 8427
c4(+10240) 8716 c5(+12288) 8731 c6(+14336) 8652 c7(+16384) 8782
```

The last chunk costs **+6%** more than the first, not 2-3×. Attention and the
indexer therefore do **not** dominate at 16K on this model — the earlier
phase-21 subset result (indexer growing +29 ms at 8K / +63 ms at 16K per 512-row
chunk on 2 layers) is real but is not the driver of the full model's flat
per-token cost. This is consistent with §3.3: the indexer is 3.6% of the time.

### 3.3 Attribution at 16K (eval-fenced — SHARES only)

`P22_ATTR=1`, chunk 0 of the 16K prompt (chunk 2048; the first chunk is
representative because per-chunk cost is flat), reproduced in both reps:

| bucket | ms (rep0) | calls | share |
|---|---:|---:|---:|
| `EXL3SwitchGLU` (routed experts) | 21671.7 | 320 | **30.7%** |
| `fn:all_sum` (RDMA collectives) | 11906.5 | 648 | **16.9%** |
| `EXL3Linear` (dense projections) | 11867.0 | 3688 | **16.6%** |
| `fn:sparse_attn` | 10990.9 | 320 | 15.6% |
| `LazyEngramTable` | 4234.9 | 16 | 6.0% |
| `RMSNorm` | 2888.0 | 1352 | 4.1% |
| `Indexer` | 2553.4 | 64 | 3.6% |
| `Exl3Member` | 794.3 | 1464 | 1.1% |
| `MoE` | 410.6 | 320 | 0.6% |
| `Gate` | 367.7 | 320 | 0.5% |
| remainder (Block, Exl3Proj, Exl3Experts, Embedding, Engram, stack) | ~1217 | | ~1.7% |
| **total (fenced)** | 70600.9 | | 100% |

Read it as shares, not absolutes: the instrumentation evaluates each module's
output, which forces extra host syncs (`all_sum` in particular is inflated —
`collective.all_sum` only host-syncs when `sync_collectives()` is active, but the
fenced wrapper adds an eval of its result per call). The unfenced base arm is the
absolute: 68.5 s for 16K in that same configuration, i.e. ~8380 ms/chunk.

**Tokens-per-expert** at chunk 2048 is 64.0 (2048 rows × 6 of top-k / 192
experts per rank) and at chunk 4096 it is 128.0. The expert GEMMs are therefore
squarely in the weight-read-bound regime, which is exactly the bucket the 4096
chunk improves (§4).

## 4. Chunk-size lever: 4096 helps, 8192 hurts

Fixed 2048-row chunks vs 4096 vs 8192, same prompts, same driver (`run3`):

| prompt | chunk 2048 | chunk 4096 | chunk 8192 | 4096 vs 2048 | peak @4096 |
|---:|---:|---:|---:|---:|---:|
| 8K (rep0/rep1) | 233.6 / 242.1 | **256.3 / 253.9** | 233.0 / 234.4 | **+5.9%** | 115.3 / 115.5 GB |
| 16K (rep0/rep1) | 232.1 / 231.8 | **249.4 / 248.2** | 229.9 / 229.6 | **+7.1%** | 116.7 GB |

- The win is real and reproduced, and the shape is the expected one: bigger M
  amortises the trellis weight traffic per token across the expert GEMMs.
- It is **not shippable as-is**: peak reaches 116.71 GB at 16K against a 115000 MB
  (`112.3 GiB`) `iogpu.wired_limit_mb`, and the harness's own headroom was
  consumed down to ~-4 GB of the limit. On this box that is the paging regime
  (bad numbers, not a crash).
- 8192 rows is strictly worse (229.6 vs 231.8), so the curve turns between 4096
  and 8192; it also peaks at 117.4 GB.
- Watchdog: at 4096 a chunk costs ~16.2 s and at 8192 ~35 s. Production's
  `EXO_RUNNER_HANG_TIMEOUT_SECONDS` is 45 s of silence before the liveness probe
  runs, and the probe needs ≥0.25 GB of footprint growth per 20 s to extend — a
  35 s chunk is inside the window but has no margin. **8192 would need a
  watchdog change; 4096 would not.**

### 4.1 Chunk-size changes are NOT bit-exact (expected)

Chunk boundaries change the GEMM M and therefore the accumulation order:

```
L=8192  chunk4096 vs base: max|Δlogit| = 1.4766  mean 0.2165  top-1 agree=yes
L=8192  chunk8192 vs base: max|Δlogit| = 1.6016  mean 0.2242  top-1 agree=yes
L=16384 chunk4096 vs base: max|Δlogit| = 1.8633  mean 0.1647  top-1 agree=yes
L=16384 chunk8192 vs base: max|Δlogit| = 0.9883  mean 0.1374  top-1 agree=yes
```

The greedy first token is unchanged in every case, and both reps of each arm
produce identical logits to each other. The 64-token sha256 does differ for the
new chunk sizes (the near-tie token at position 11/12 flips: `…, 4147, 2619…`
vs `…, 4147, 554…`), which is the documented chunk-shape dependence of this
model, not a defect. **Byte-identical output is not available with a chunk-size
change**; it would have to be validated on acceptance quality (NLL) instead.

## 5. `MLX_MAX_OPS_PER_BUFFER` is neutral for prefill

Each value needs a fresh process (the limit is read once, statically, at first
use). Values verified on BOTH ranks' fresh pids before any workload
(`ops_per_buffer` echoed in each run's `env` line). Same fixed prompts, chunk
2048:

| value | 8K tok/s | 16K tok/s | oracle vs 200 |
|---|---:|---:|---|
| 200 (production) | 237.8–246.2 | 237.4–239.6 | reference |
| 24 | 235.1 | 240.5 | **bit-identical** (sha + max|Δlogit| 0) |
| 5 | 236.0 / 245.4 | 240.6 / 242.6 | **bit-identical** |

Everything lands in the same ±2% band as two baseline runs, so this knob buys
nothing for prefill (contrast with the −14% it bought on the attention-only
microbench — attention is only ~19% of prefill here). Bit-exactness holds, which
is the useful negative: the sweep costs correctness nothing.

## 6. The 8-layer-subset discrepancy, explained

The brief asked me to reconcile "8-layer subset at 16K = 679 tok/s" (workstream
C's `wsC-prefill` report: 12.8 s for 8704 tokens = 678 tok/s, `[512×16, 128×4]`
chunks, single node) against the full model's 245 tok/s, where a naive
×(8/40) scaling implies ~136 tok/s.

Three mechanisms, all real, and they compound:

1. **The subset ran on ONE node; production runs TP=2.** The 8-layer subset
   holds complete layers on one Mac, i.e. all 384 experts and all 64 heads. The
   full model splits that work across two Macs. Correct naive scaling is
   therefore ×(8/40)×2 = **272 tok/s**, not 136.
2. **Chunk size.** The subset ran 512-row chunks and measured 1.35 ms/tok at 512
   rows vs 2.47 ms/tok at 128 rows; the full model at 16K runs 2048-row chunks
   which are *cheaper* per token, so this term works in the opposite direction
   from what the raw comparison suggests.
3. **Layer composition and context.** The subset includes only 2 of the 4
   kv/index source layers (2/8 of its 8 layers = 25%, vs 4/40 = 10% of the full
   stack) but was measured at 8704 tokens, where the compressed-KV history is
   far shorter than at 16K; the indexer's per-row cost grows with the
   accumulated compressed positions.

272 → 245 is a ~10% TP tax, which is the right size for one `all_sum` per layer
per chunk plus non-overlapped collective latency — and §3.3 now prices that
collective directly at **16.9% of prefill time**. So the subset number was never
scalable: it was a different workload on a different topology. **Report the 679
figure only with this frame, never scaled.**

## 7. Production restore (verified)

Recorded before stopping: `raw/production-launch-cmd-m4-{1,2}.txt` and
`raw/production-launch-env-m4-{1,2}.txt`, plus `raw/prestop-state.txt`.
Stopped cleanly with `scripts/exo_graceful_shutdown.sh` — both nodes reported
`EXO_SHUTDOWN_VERDICT=CLEAN_EXIT` (SIGTERM only, no SIGKILL, so RDMA QPs were
released by the destructors). Restored with the launcher-generated
`~/relaunch_exo.sh` on each node, which carries the identical env (verified by
diffing the fresh pids' env against the pre-stop capture) — **plus a manual
instance placement**, because `relaunch_exo.sh` restarts the process only and
does not place a model. Placement used the same parameters the launcher uses:
`sharding=Tensor`, `instance_meta=MlxJaccl`, `maxKvTokens=16384`,
`maxPrefixSessions=4`, `maxPrefixBytes=12884901888`, `kvCacheBits=0`.

`raw/restore-verification.txt` contains, in order:

- `GET /v1/models` → 200; model listed; `/ollama/api/ps` lists DSv4.1;
- both runners `RunnerReady` in `/state`;
- real chat completion at `temperature=0` → `content: 'restored'`,
  `finish_reason: stop`, 0.7 s wall;
- env on the FRESH pids, BOTH nodes: **identical to the pre-stop capture**
  (`MLX_MAX_OPS_PER_BUFFER=200`, `EXO_PREFILL_STEP_SIZE=2048`,
  `EXO_PREFILL_STEP_SIZE_HIGH_CTX=128`, `EXO_PREFILL_STEP_SIZE_CROSSOVER=500000`);
- a 2K-prompt TTFT re-measure through the same API path: **9.75 s** vs
  **9.39 s** before the window (same prompt, same output text) — within the
  expected process-to-process spread, so the restore is behaviourally equal, not
  just config-equal.

## 8. What this means for the ≥2× goal

- **Not available from config.** Chunk 4096 (+5.9%/+7.1%) is the only knob that
  moves, and it is over the wired limit at 16K. `MLX_MAX_OPS_PER_BUFFER` is flat.
- **The addressable mass is the GEMM path and the collective**: experts 30.7% +
  dense EXL3 16.6% + `all_sum` 16.9% = **64%**. A ≥2× needs work there (fused
  expert kernels / fewer or cheaper collectives), which is Stage B.
- **Two things Stage B must not break**: (a) the oracle is byte-stable today, so
  any GEMM rework must be checked against the recorded shas in
  `raw/{run,run2}/results.json`; (b) tokens-per-expert is 64 at the production
  chunk size, so an M-padding / tile-shape change has real headroom to exploit
  *without* changing chunk boundaries (i.e. it can keep byte-identical output
  where a chunk-size change cannot).
- **A cheap Stage-B candidate the attribution names**: `LazyEngramTable` is 6.0%
  of prefill but only 16 calls, i.e. ~265 ms per call — engram table reads
  inside the prefill critical path. It deserves a look before kernel work.

## 9. Artifacts

- `scripts/` — `p22_prep.py` (fixed prompts), `p22_prefill.py` (harness),
  `p22_api_baseline.py` (served baseline), `p22_analyze.py` (tables above),
  `p22_run.sh` (two-node runner), `parent_lock.sh` (lock, from the parent's
  instrument), `p22_prefill_launch.sh` (rank launcher).
- `raw/` — per-run `results.json`, per-arm `logits_*.npy`, `progress.jsonl`,
  node logs under `raw/logs/`, `analysis-all-runs.txt`, `restore-verification.txt`,
  pre-stop production cmd/env captures, `prompt-manifest.json`.
- Fixed prompts are byte-reusable: sha256 in `raw/prompt-manifest.json`
  (2K `ddd128217ea0441c…`, 8K `7e850e4af8b89fc0…`, 16K `76361b148f861fca…`).
