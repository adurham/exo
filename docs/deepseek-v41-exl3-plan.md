# DeepSeek-V4.1-Flash EXL3 2.9 bpw on the two-node exo cluster

Implementation plan. Drafted 2026-09-24 from a survey of the cluster, the forks, and the four
public reference implementations. Every number marked *est.* is an extrapolation; every number
without that mark was measured or read from a config.

## Status (updated 2026-09-28)

Evidence lives in `docs/benchmarks/phase*-2026-09-2{6,7,8}/` (benchmark phases,
numbered separately from the plan phases below).

- Kernel spike CLOSED: EXL3 MoE lands at ~1.9-2.0x mxfp4 full-width after two
  bit-identical ALU fixes; the 1.25x gate is out of reach for this kernel design
  (phase9 README incl. day-2 closure).
- Plan B (SSD-streamed mxfp4 experts) measured and NOT chosen; build path is A,
  EXL3-resident (phase10, phase11).
- At the real TP=2 serving geometry the EXL3 MoE costs 1.61-1.74x mxfp4, and the
  MTP gamma=5 projection is ~25-31 tok/s -- borderline for the 25 tok/s bar;
  only the real build settles it (phase11 Addendum).
- Plan phase 2 integration pieces DONE on `adurham/mlx-lm` branch
  `feat/dsv41-exl3`: kernels vendored bit-identically as `mlx_lm.models.exl3`
  (`9ea86f9`), TP rank-slice recipe proven exact, loader module (`6391efc`)
  gated green (phase12). exo's recorded mlx-lm pin intentionally still
  `5c5328b` until the model file needs the branch.
- MTP/DSpark draft head ported and token-identical (phase3/4/7).
- Plan phase 3 first cut DONE: `mlx_lm.models.deepseek_v41` package + EXL3
  builder (`bd1bfd1`); per-layer cos >= 0.99983, NLL 1.003 / top-1 78.3%
  matching the reference (phase13).
- NEXT: `mlx_lm.load` entry + full-model decode loop, then exo integration
  (auto_parallel world=2, model card, generator, MTP head), first real tok/s.

## 0. Goal

Serve `dealignai/DeepSeek-V4.1-Flash-UNCENSORED-EXL3-2.9bpw` (already abliterated) on
macstudio-m4-1 + macstudio-m4-2 through exo, with Engram tables on SSD, DSpark speculative
decoding, and the existing prefix cache. Target: decode at or above today's V4-Flash numbers,
prefill within 2x of today's.

Non-goals for v1: vision tower, Linux/CUDA ranks, more than two nodes, batch > 1 serving.

Success gate for the whole project: 32K-token conversation runs end to end on both nodes,
peak memory under the wired limit, greedy output token-identical between DSpark on/off.

## 1. Ground truth

| Item | Value |
|---|---|
| Nodes | 2x Mac Studio M4 Max, 40 GPU cores, 128 GB each |
| Wired GPU limit | `iogpu.wired_limit_mb = 115000` per node |
| SSD | Apple AP1024Z, ~2.2 GB/s single stream, ~6.7 GB/s at 4 parallel streams (cold) |
| Free disk | ~105 GB per node; `~/.exo/models` holds 562 GB, of which three V4-Flash copies are 465 GB |
| exo | `~/repos/exo`, fork `adurham/exo`, main at 7cf0939dc (2026-09-23), 105 uncommitted files |
| mlx-lm | vendored editable at `~/repos/exo/mlx-lm`, fork `adurham/mlx-lm`; `deepseek_v4.py` is 8,462 lines with DSpark, sparse indexer, TP work |
| mlx | fork `adurham/mlx` main, 0.32.3.dev, JACCL RDMA work |
| Today's V4-Flash | decode ~22-30 tok/s with DSpark, prefill ~225-380 tok/s, peak 88 GB/node with 72 GB weights/node |

Checkpoint facts (from `config.json`): `model_type: deepseek_v41`, `quant_method: exl3`,
version 1.4.2, `codebook: mul1`, `avg_bits: 2.9`, `head_bits: 6`, `mtp_bits: 4`. Engram
tables are not in the repo; they come from shards 47 and 48 of `deepseek-ai/DeepSeek-V4.1-Flash`
(two FP8 tensors, ~101.5 GB each).

Architecture facts: 552B backbone, 40 layers (20 causal encoder + 20 decoder), 384 routed + 1
shared expert, top-6, Engram 196B at layers 1 and 14 (2/3/4-grams x 8 heads over a 99,092-id
compressed token map), CSA2 with four compressor-owning layers and Full/Reindex/Reuse modes,
hyper-connections with split-Sinkhorn, SWA bounded replay, FP4 main KV at 890 B/token, DSpark
with three draft stages.

## 2. Reference implementations

All Apache-2.0 unless noted. Cloned to the session scratchpad for inspection; re-clone into
`~/repos/ref/` for the work.

| Repo | What we take from it |
|---|---|
| `beamivalice/PonyExl3` | EXL3 trellis decode on Metal via `mx.fast.metal_kernel`. `ponyexl3/mlx/exl3_moe.py` (`EXL3SwitchGLU`, fused gate-up/down kernels, `_decode_fused2` and `_prefill` paths), `exl3_linear.py`, `reconstruct.py`, `codebook.py`, `model.py` (`_ARCHITECTURES` map, EXL3 tensor suffixes `trellis suh su svh sv mcg mul1 bias`). Converter in `ponyexl3/convert/`. |
| `PipeNetwork/deepseek-v41-mlx` | Validated V4.1 math. `deepseek_v41_mlx/engram.py` (`EngramHasher`, `build_compressed_token_map`, `compute_hash_multipliers`), `compressor.py`, `indexer.py`, `sparse_attention.py`, `hyper_connections.py` (`split_sinkhorn`), `cache.py`, `stream.py` (layer-at-a-time runner with explicit `StreamState`, used for divergence ladders), `docs/reference/` (DeepSeek's PyTorch reference). |
| `Jackten/deepseek-v41-mlx-three-mac` | Native MXFP4/MXFP8 runtime on Macs. `dsv41_rowstore.py` (`SafetensorsRowStore`: pread row reads, prefetch, io workers, LRU by bytes), `dsv41_prefetch.py`, `partition_moe.py`, `csrc/grouped_expert.cpp`, `patches/mlx-timing7.patch`, `engram_token_map.json`, DSpark tests. |
| `ssd-moe/deepseek-v4-flash-mlx` | Pattern only: device LRU over expert records fed by parallel pread. Plan B. |

## 3. Target shape

```
exo worker (rank 0 / rank 1)          src/exo/worker/engines/mlx/
  auto_parallel.shard_model  ->  experts TP across ranks, compressor/indexer replicated (as V4 today)
  generator / pp_* / cache   ->  cross-layer shared KV cache, prefix snapshots, DSpark driver

mlx-lm fork                            ~/repos/exo/mlx-lm/mlx_lm/models/
  deepseek_v41.py            ->  derived from deepseek_v4.py; new: hyper-connections, CSA2,
                                 compressor cache, hierarchical indexer, SWA replay, Engram,
                                 20/20 layer split, attention sinks, FP4 KV
  exl3/                      ->  vendored PonyExl3 kernels: EXL3SwitchGLU for experts,
                                 EXL3Linear for dense/attention/head/MTP
  engram_rowstore.py         ->  ported Jackten SafetensorsRowStore, batched pread, hot-row LRU

Weights                                ~/.exo/models/
  dealignai EXL3 shards (197 GiB, resident, split by expert across ranks)
  base shards 47+48 (Engram FP8, ~196 GB, on SSD on BOTH nodes, never resident)
```

## 4. Phases and gates

Order matters: phases 1 and 2 are cheap and can kill the plan early. Phases 3 and 4 are
independent of each other and of 2, so they can run in parallel. Phase 5 needs 2, 3, 4.

### Phase 0. Prep (1-2 days)

Decisions taken 2026-09-24: only `deepseek-ai--DeepSeek-V4-Flash-Vision-Exp` is in use, so
`mlx-community--DeepSeek-V4-Flash` and `deepseek-ai--DeepSeek-V4-Flash-0731` were deleted on
both nodes (verified first: Vision-Exp was the loaded model, zero open files, no links). Free
space after: ~401 GiB per node. Raising the wired limit is approved if phase 5 needs it.
Downloads run on node 1 only; node 2 is populated by rsync afterwards, never a second download.

- Disk per node at the end of phase 0: full EXL3 repo (196.2 GiB = 210.7 GB; exo keeps full
  weights on every node and loads its shard), Engram shards 47+48 (203 GB), ~20 GB working.
  That is ~434 GB against ~401 GiB (~430 GB) free — a ~4 GB deficit, not the comfortable margin
  the raw GB figures suggest. Watch free space live during the download; if it's tight, drop
  Vision-Exp (166 GB) before finishing the Engram shard pull rather than after.
- Download to node 1. Script `~/hfdl-v41.sh` on macstudio-m4-1 pulls the full dealignai repo
  into `~/.exo/models/dealignai--DeepSeek-V4.1-Flash-UNCENSORED-EXL3-2.9bpw`, then base shards
  47, 48, `config.json`, tokenizer files and index into
  `~/.exo/models/deepseek-ai--DeepSeek-V4.1-Flash-engram`, sequentially. Started and then
  stopped on request 2026-09-24 00:45 CDT with 3.7 GB down and 6 partial shards kept in the
  target's `.cache/huggingface/download/`. `hf download` resumes from those, so restarting is
  `screen -dmS hfdl-v41 bash -c '~/hfdl-v41.sh > ~/hfdl-v41.log 2>&1'`.
  Note: `quantization_config.json` in the EXL3 repo is 54 MB, almost certainly the per-tensor
  bit map. Phase 1's byte table comes from it directly.
- Rsync both directories node 1 -> node 2 over the Thunderbolt link after the download finishes.
  Verify sha256 against HF LFS pointers on node 1 before the rsync.
- `git clone` the four reference repos into `~/repos/ref/`.
- Record baseline: 10 runs of V4-Flash decode and a 2048-token prefill through exo, same prompts
  reused in every later phase. Store in `docs/benchmarks/`.
- Gate: files present and hashed, baseline table committed.

### Phase 1. EXL3 format validation (1-2 days)

Answers "can PonyExl3 read this file at all" before any porting.

- Install PonyExl3 in a throwaway venv on node 1 (`uv venv`, `uv pip install -e ~/repos/ref/PonyExl3`).
- Load one routed-expert tensor group (`trellis`, `su`, `sv`, `suh`, `svh`, `mul1`) from a
  dealignai shard with `ponyexl3.mlx.weights.load_safetensors`, build an `EXL3Layer`, and run
  `reconstruct_inner_mlx` against `ponyexl3.ref` CPU reference. Bit-exact is the pass condition.
  Do the same for one attention projection, the 6-bit head, and one 4-bit MTP tensor.
- Confirm format version 1.4.2 is accepted; if PonyExl3 pins an older exllamav3 version, diff
  the tensor layouts against `turboderp-org/exllamav3` for that version bump.
- Walk every shard header and tabulate bits per tensor by component: routed experts, shared
  expert, attention, dense MLP, embeddings, head, MTP. Compute per-rank resident bytes under the
  V4 sharding rule (experts split, everything else replicated). This is the real memory budget.
- Gate: bit-exact decode on all four tensor classes, and a per-rank byte table. If per-rank weight
  bytes exceed ~98 GB, the memory mitigations in section 5 become mandatory, not optional.

### Phase 2. Trellis kernels in the existing V4 path (1-2 weeks)

De-risks the two unknowns that nobody has measured: EXL3 gather-GEMV on a 384-expert MoE, and
EXL3 decode-in-GEMM prefill at 200 GB scale, without touching V4.1.

- Vendor `ponyexl3/mlx/{exl3_moe,exl3_linear,exl3_qmv,exl3_qmm,exl3_fused,reconstruct,codebook,
  hadamard,signs,perm,stripe,metal_kernels,weights}.py` into `mlx_lm/models/exl3/` with license
  and NOTICE. Keep the import surface to `EXL3SwitchGLU` and `EXL3Linear`.
- `EXL3MoEBlock` assumes Qwen softmax routing. Do not use it. Keep the existing DeepSeek router
  in `deepseek_v4.py` (sigmoid, bias, shared expert) and swap only `switch_mlp` for
  `EXL3SwitchGLU` behind `EXO_DSV4_EXL3_EXPERTS=1`.
- Convert experts for layers 0-3 of `deepseek-ai/DeepSeek-V4-Flash-Vision-Exp` bf16-dequantized
  (the only V4-Flash copy phase 0 kept) to EXL3 4 bpw with `ponyexl3-convert` (direct mode, no
  calibration, quality irrelevant here). Leave other layers affine. `mlx-community/DeepSeek-V4-Flash`
  was deleted in phase 0; re-download it instead if Vision-Exp's tower weights get in the way of
  a clean layer-0-3 extract.
- Run under exo with TP across both ranks. `EXL3SwitchGLU` has no notion of ranks; the V4 TP
  code already slices expert tensors per rank, so the slice must happen on the EXL3 tensor
  group as a unit (all six suffixes for an expert move together). This is the main integration
  edit in `auto_parallel.py`.
- Measure per-layer decode time and 2048-token prefill time for the four EXL3 layers against
  four affine layers, using the phase-marks instrumentation already in the engine.
- Gate: EXL3 layer decode <= 1.25x affine layer decode; EXL3 layer prefill <= 1.7x affine.
  If prefill misses by more than 2x, stop and go to Plan B (section 6). If decode misses,
  profile `_decode_fused2` tile sizes on M4 Max before deciding; PonyExl3's numbers are M5.

### Phase 3. V4.1 architecture port (3-6 weeks)

New file `mlx_lm/models/deepseek_v41.py`, copied from `deepseek_v4.py`, then modified in this
order. Validate each step against PipeNetwork's `stream.py` layer runner on node 1 with the
base checkpoint in PipeNetwork's `fakequant` mode, comparing hidden states per layer. Targets:
cosine >= 0.999 per layer on a 512-token prompt, and argmax agreement on the next 64 greedy
tokens.

1. `ModelArgs` from `config.json`: layer roles (encoder/decoder), compressor owners, CSA2 mode
   per layer, Engram layer ids, hyper-connection count, SWA window, DSpark stage count.
2. Hyper-connections: port `split_sinkhorn` and the staggered `pre_mix` stream. This changes
   the residual path everywhere, so it goes first.
3. Compressor + `CompressorState`: four owner layers, consumers reference the owner's cache.
   New cache class in the model file; exo's `cache.py` wraps it in phase 5.
4. CSA2 attention: Full/Reindex/Reuse. Reuse mode consumes Top-K indices from the owner layer,
   so the indexer output must be part of the shared state. Port `sparse_attention.sparse_attn`
   with attention sinks. Reuse the existing tiled sparse-SDPA and tiled indexer-score code from
   `deepseek_v4.py` where the math is identical; the hierarchical candidate pool is new.
5. SWA bounded replay for the window ring.
6. Engram module: `EngramHasher` + `build_compressed_token_map` (cache the map to disk, it
   matches Jackten's `engram_token_map.json`), KV projections, and a row-provider interface that
   phase 4 fills. Until then use PipeNetwork's `QuantizedEngramEmbedding` on a tiny synthetic
   table for shape tests.
7. 20/20 layer split and per-layer attention sinks.
8. FP4 main KV cache (E2M1 + one E4M3 scale per 16 channels). Match PipeNetwork's
   `fakequant` path first, then a real packed cache.
9. DSpark: three stages, adapted from the existing `DeepseekV4DSparkModule` and the TP-sharded
   draft-head code. Draft tensors are 4-bit EXL3, so they load through `EXL3Linear`.
10. Vision: `deepseek_v4_vision.py` passthrough skip, drop tower weights at load.

Gate: end-to-end greedy generation on node 1 using PipeNetwork's streaming loader as the
weight source (slow, correctness only), matching PipeNetwork's `smoke_generate.py` output.

### Phase 4. Engram SSD row store (1 week, parallel with 3)

- Port `SafetensorsRowStore` from Jackten into `mlx_lm/models/engram_rowstore.py`. Keep pread,
  the io-worker pool, the byte-bounded LRU, and the prefetch hook. Rows are FP8 with per-row or
  per-block scales; dequantize on read into a small bf16 device buffer.
- Both ranks hold both Engram shards locally and read them independently. No collective. This
  wastes 196 GB of disk per node but removes a per-layer cross-rank dependency at layers 1 and
  14. Revisit only if disk becomes the constraint.
- Decode path: 48 row reads per token (2 modules x 3 orders x 8 heads), issued as one batched
  call. Budget: under 1 ms with the pool warm.
- Prefill path: gather unique row ids across the whole chunk, dedupe, sort by file offset and
  coalesce adjacent rows into single preads, then batch-pread and scatter. A 2048-token chunk is
  at most ~98K rows and typically far fewer after dedupe. Budget: under 0.3 s per chunk. A serial
  implementation here costs ~10 s per chunk and would sink prefill on its own, so this budget is
  a hard gate.
- Hot-row LRU sized at 2-4 GB per rank. Measure hit rate on real prompts.
- Gate: decode lookup < 1 ms, 2048-token chunk < 0.3 s, outputs identical to PipeNetwork's
  in-memory table on the same ids.

### Phase 5. exo integration (2-3 weeks)

- `auto_parallel.shard_model`: `deepseek_v41` branch. Experts TP by expert id as in V4, EXL3
  tensor groups sliced as units. Compressor, indexer, Engram, hyper-connections, sinks
  replicated. DSpark heads TP-sharded like V4.
- Model card: `custom_model_cards/` entry with the resident footprint computed in phase 1, not
  the on-disk size, so placement does not reject it. Point Engram shards via an env var or card
  field; do not register them as model weights.
- `cache.py`: wrap the compressor-owned shared cache so prefix snapshots (`EXO_LEAF_SNAPSHOT_RETENTION`)
  and KV eviction see one object per sequence. Reuse-mode layers must snapshot their index
  selections with the owner's cache or prefix hits will produce wrong attention.
- Generator: DSpark driver already exists for V4; the V4.1 change is three stages of the new
  module plus the FP4 KV handling in verify.
- JIT model lifecycle: loading 105 GB of trellis tensors per rank at boot. Reuse the existing
  lifecycle branch; check load-time peak via `mx.get_peak_memory()` / wired-page tracking (not
  process RSS, which doesn't reflect wired GPU memory), since PonyExl3 reported load peak well
  above resident.
- Gate: the section-0 success gate on a 32K conversation, plus the phase-0 benchmark prompts.

### Phase 6. Performance (ongoing)

Only after phase 5 passes. In priority order: per-token fixed overhead (this is ~25 ms today and
dominates), DSpark acceptance tuning per stage, Engram prefetch on the draft tokens, prefill
chunk size versus indexer cost, `_decode_fused2` tile sizes on M4 Max.

## 5. Memory budget and the fit problem

Today: 72 GB weights per rank, 88 GB peak, so ~16 GB per rank of KV, DSpark, indexer scratch and
MLX allocator overhead. The EXL3 file is ~105 GB per rank under the same sharding. 105 + 16 = 121 GB
against a 115 GB wired limit. This does not fit as-is. V4.1's 890 B/token KV returns a few GB;
it is not enough on its own. Mitigations, safest first:

1. Measure before assuming. Phase 1's per-rank byte table replaces the 105 GB estimate.
2. Drop the vision tower and unused MTP tensors at load. Free but small.
3. Shard the head and embeddings across ranks instead of replicating them. The 6-bit head on a
   129K vocab is a few GB per rank.
4. Stream the coldest expert slice. Keep the ssd-moe LRU as a bounded overflow for, say, the
   least-routed 10 percent of experts per rank. This reuses Plan B code and costs decode
   speed only on cold misses.
5. Raise the wired limit. `sudo sysctl iogpu.wired_limit_mb=122880` leaves 8 GiB nominal headroom
   for macOS (~5 GB effective once kernel-wired pages are accounted for), which is workable on a
   headless node but fragile. Test with the wedge watchdog active. Last resort: it's the only
   mitigation here that risks the node itself, rather than just costing memory budget or decode
   speed.

If mitigations 1-4 together do not get peak under the limit with 32K context, 5 is required and
the decode estimate below moves down.

## 6. Plan B: native MXFP4 experts with SSD expert streaming

Triggered by a phase 2 prefill miss or an unfixable phase 5 memory miss. Phases 3, 4, 5 carry
over unchanged. Phase 2 is replaced by: keep the base checkpoint's MXFP4 experts and MXFP8 dense
(MLX has native `mxfp4`/`mxfp8` quantized matmul), hold ~60 percent of experts per rank resident,
and stream the rest through a byte-bounded device LRU fed by parallel pread, per the ssd-moe
design. Quality is the Jackten kit's, decode becomes SSD-bound (single digits to low teens
tok/s depending on hit rate), prefill is unaffected since a chunk touches nearly every expert
once. The abliteration then has to be applied by us: rank-1 refusal projection on attention
output biases (the cebeuq V4 method), computed with our own port once it runs.

## 7. Expected performance (est.)

| Metric | Today (V4-Flash) | Estimate | Confidence |
|---|---|---|---|
| Decode, raw forward | not logged | 22-28 tok/s | +/- 40% |
| Decode with DSpark | 22-30 tok/s | 30-45 tok/s | +/- 40% |
| Prefill, short/mid context | 225-380 tok/s | 250-450 tok/s | +/- 50% |
| Prefill, > 50K context | degrades | 150-250 tok/s | low |
| Plan B decode with DSpark | | 8-15 tok/s | low |

Basis: 16B active at 2.9 bpw is 2.9 GB per rank per token; trellis GEMV at ~35-40 percent of
546 GB/s is 14-20 ms; today's fixed overhead is ~25 ms and largely unchanged. Prefill gains
from 8B active (vs 13B) and CSA2 are offset by decode-in-GEMM at ~32 tokens per expert.

## 8. Risk register

| Risk | Phase | Detection | Response |
|---|---|---|---|
| PonyExl3 rejects format 1.4.2 or mul1 layout differs | 1 | bit-exact test fails | diff against exllamav3 tag, patch `reconstruct.py` |
| EXL3 prefill at scale much slower than dense benchmarks | 2 | gate miss | Plan B |
| Per-rank weights exceed budget | 1, 5 | byte table, load peak | section 5 mitigations |
| Engram lookup serial or hash mismatch | 4 | budget miss, id mismatch vs PipeNetwork | batch pread; regenerate token map |
| CSA2 Reuse mode wrong under prefix cache | 5 | greedy divergence after prefix hit | snapshot index selections with owner cache |
| Load-time peak exceeds limit | 5 | wedge at boot | per-shard lazy load, drop tower early |
| DSpark 4-bit heads degrade acceptance | 6 | acceptance rate vs V4 | tune stage count; heads are tiny, consider re-quantizing from base |
| M4 Max slower than PonyExl3's M5 Max numbers | 2 | measured | accept or retune tiles |

## 9. Files touched

Fork `adurham/mlx-lm` (`~/repos/exo/mlx-lm`):
`mlx_lm/models/deepseek_v41.py` (new), `mlx_lm/models/exl3/` (vendored), `mlx_lm/models/engram_rowstore.py` (new),
`mlx_lm/models/deepseek_v4.py` (phase 2 flag only, revert after).

Fork `adurham/exo`:
`src/exo/worker/engines/mlx/auto_parallel.py`, `cache.py`, `generator/`, `deepseek_v4_vision.py`,
`src/exo/shared/models/model_cards.py`, `src/exo/master/placement*.py`, `~/.exo/custom_model_cards/`.

Fork `adurham/mlx`: none expected. Jackten's `mlx-timing7.patch` is an optimization to evaluate
in phase 6, not a dependency.

## 10. Open questions to settle in phase 1

- Exact tensor names and shapes for the four EXL3 tensor classes, and whether `mul1` appears as
  a per-tensor flag tensor or a config field (PonyExl3 supports both).
- Whether dealignai kept the shared expert and dense MLPs at a higher bit than routed experts.
  Affects the per-rank budget since those are replicated.
- Whether the MTP tensors in the file are all three DSpark stages or one.
- PipeNetwork and Jackten disagree on whether Engram hash multipliers depend on layer id
  (`compute_hash_multipliers(layer_ids, ...)`). Verify against `docs/reference/engram.py`.
