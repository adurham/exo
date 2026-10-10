# Q1D sub-task results — memory-fit arithmetic + original-checkpoint shard fetch

Author: Phase-20 subagent (delegation). Date: 2026-10-10 (CDT). Worktree `/private/tmp/phase20-campaign`.
Feeds GATE 1 (ROUND-Q1D-MICROBENCH.md §2b memory-fit limb + §6 D3/D4). Node: `studio1` (network/disk/CPU only, no GPU).

Scripts:
- `docs/benchmarks/phase20-throughput/raw/pricing/q1/q1d/q1d_memfit.py` (Task A)
- `docs/benchmarks/phase20-throughput/raw/pricing/q1/q1d/fetch_q1d_orig_shards.sh` (Task B)
Raw node output: `studio1:/tmp/q1d_mem/q1d_memfit.out`, fetch log `studio1:/tmp/q1d_mem/orig/fetch_q1d.log`.

---

## TASK A — per-rank memory-fit arithmetic (EXL3 experts-requant arms) vs 128 GiB

Config (EXL3 `config.json`): 40 layers, E=384 routed experts, D=hidden=5120, H=moe_intermediate=2304,
TP world=2 (intermediate-width slice, proven phase-12 geometry in `mlx_lm/models/exl3/loader.py::load_experts`
/ `_slice_intermediate`). Header-measured: per-rank routed-expert weight count **N_rank = 271,790,899,200**.

### Exact EXL3 routed experts (all 40 layers), from the 39 safetensors headers
- **full (both ranks) = 196.034 GB**;  trellis bit-width is **not uniform**:
  - k=2 trellis on layers **18–22** (5 layers): 17.072 GB full / **8.566 GB per rank**
  - k=3 trellis on the other **35 layers**: 178.962 GB full / **89.687 GB per rank**
- **per rank (TP=2 slice) = 98.253 GB (91.505 GiB)** = trellis 97.675 GB + suh/svh signs 0.578 GB
  (sign split per loader: `gu_suh`/`dn_svh` replicated full-D; `gu_svh`/`dn_suh` sliced H→H/2).
- Cross-check vs `docs/deepseek-v41-memory-budget-correction-2026-09-27.md` (measured, all headers):
  doc routed 98.02 GB, MTP 3.84, shared 0.40, replicated 6.50, total 108.76. This script: routed 98.253,
  **total per-rank weights 108.963 GB** (Δ 0.2 %, doc rounding). Measured live EXL3 anchor ≈ **105.5 GB**.

### Non-expert resident REST per rank (running-code sharding: experts/shared/MTP-ffn width-sharded, rest replicated)
| component | full GB | per-rank GB |
|---|---:|---:|
| MTP `.ffn` (3 DSpark stages) | 6.816 | 3.408 |
| replicated (attn + embed + head + router + vision + norms + HC) | 6.499 | 6.499 |
| shared expert | 0.857 | 0.429 |
| MTP non-ffn | 0.374 | 0.187 |
| **REST TOTAL** | **14.547** | **10.710** |

### KV cache per rank (attention is REPLICATED → each rank holds the whole thing)
Derived from `deepseek_v41/cache.py`: `win_kv` every layer (window=128, head_dim=512, bf16, constant);
`comp_kv` (head_dim=512) + `index_k` (128) allocated only on the 4 `kv_source` layers {2,8,14,20},
bf16-stored (lossless for the FP4/FP8-grid values), rows = ceil(ctx/ratio).
- slope = **3200 B/token** (comp_kv 2560 + index_k 640) — matches the cache.py docstring
  ("a 1M session's compressed/index caches are ~3.2 GiB in bf16").
- constant window ring = **5.24 MB**.
- **@ agentic 91,043 tok → 296.58 MB (0.297 GB)**;  @ config max 1,048,576 → **3.361 GB (3.130 GiB)**.
- (`EXO_KV_CACHE_BITS=0` in production ⇒ no extra KV quantization; the plan's "FP4 main KV at
  890 B/token" is the FP4-packed figure, 3200 B/token is the bf16-resident reality, ×3.6.)

### FIT TABLE — per-rank total (experts + rest 10.710 + KV@91K 0.297) vs 128 GiB = 137.44 GB
| arm | experts GB | total GB | total GiB | % of 128 GiB | verdict |
|---|---:|---:|---:|---:|---|
| **EXL3 2.9 bpw trellis** | 98.25 | **109.26** | 101.76 | 79.5 % | **FITS** |
| q4g32 (4.50 bpw) | 152.88 | 163.89 | 152.63 | 119.2 % | DOES-NOT-FIT |
| q4g64 (4.25 bpw) | 144.39 | 155.40 | 144.72 | 113.1 % | DOES-NOT-FIT |
| mixed gu q4 / dn q6 | 167.04 | 178.04 | 165.82 | 129.5 % | DOES-NOT-FIT |
| q5g64 (5.25 bpw) | 178.36 | 189.37 | 176.36 | 137.8 % | DOES-NOT-FIT |
| q6g64 (6.25 bpw) | 212.34 | 223.34 | 208.00 | 162.5 % | DOES-NOT-FIT |

Native-affine expert bytes = N_rank × bpw/8 with the exact group-scale overhead (q4g64=4+16/64=4.25,
q5g64=5.25, q6g64=6.25, q4g32=4+16/32=4.50); mixed = gu(q4g64)+dn(q6g64).

**VERDICT (feeds GATE 1): the memory-fit limb FALSIFIES every native arm.**
- Only the current **EXL3** representation is arithmetically deployable at 128 GiB (109.26 GB, 28 GB headroom).
- Every native arm — even the cheapest, q4g64 — is **14–63 % over** the 128 GiB node RAM, and also over the
  production wired limit `iogpu.wired_limit_mb=115000` (= 120.59 GB).
- The routed-expert budget that would fit is 137.44 − 10.71 − 0.30 = **126.4 GB ⇒ ≤ 3.72 bpw** — *below* q4.
  So a native experts arm can only become deployable with the streaming/tiering mitigation stack
  (memory-correction doc §3–4: stream ~12–15 % of expert bytes + head/embed shard), which is out of scope
  for Round 1. **As measured here, no native arm is deployable as-is.**

---

## TASK B — original (pre-EXL3) uncensored checkpoint: sampled-shard fetch

- `dealignai/DeepSeek-V4.1-Flash-UNCENSORED` → **absent** (HF API: "Repository not found").
- Discovered original: **`dealignai/DeepSeek-V4.1-Flash-UNCENSORED-FP8`** — exists, **gated: false, private: false,
  public**, 55 files, 510.3 GB (48 shards). quant_method fp8 / expert_dtype **fp4** / block [32,32] / ue8m0;
  same arch (40 layers, 384 experts, D=5120, H=2304). Experts stored FP4-packed in I8 containers
  (w1 [2304,2560], w2 [5120,1152], + E8M0 scales) ≈ 7.4 GB/layer/shard.
- safetensors index maps **1 shard per layer**: shards 3–42 = layers 0–39 (shards 47/48 = Engram l=1,14).
- Sampled layers **{0, 20, 39}** → shards **model-00003 / 00023 / 00042** (~7.39 GB each).
- **Verified** via HTTP Range reads of each shard's safetensors header (read-only, pre-download):
  each contains its layer's attn + `ffn.experts.{0..383}` keys (e.g. `layers.20.ffn.experts.0.w1.weight`
  [2304,2560] I8 + scale [2304,160] E8M0), 2334–2341 tensors per shard.
- **Fetch**: throttled (`curl --limit-rate 20M`, resumable) to `/tmp/q1d_mem/orig/` on **studio1**,
  run under `nohup` (pid logged; log `/tmp/q1d_mem/orig/fetch_q1d.log`). Status: see §status below.
- HF LFS sha256 (for post-download verify): 00003 `e1281f85d0ce4a3dfb63d41926fc4a47fa71f36ba20992e3597e702ead49d4c9`;
  00023 `680947670e4bd32cfff3d030c97b4009b9cb007cfaa56ad20a701de513e14524`;
  00042 `e1a4d5d30ae51bafda75078d49b05a585c924b0af9b3481c075b04341507c2ef`.
- Retained for Round 2 at **`studio1:/tmp/q1d_mem/orig/model-0000{3,23,42}-of-00048.safetensors`** (+ config.json,
  model.safetensors.index.json). No blocker.

### STATUS (complete)
`ALL DONE` 2026-10-10 01:13:05 CDT. 3/3 shards fetched, all `rc=0`, exact expected sizes:
7389759032 / 7400713088 / 7389761368 B = **22.18 GB** total (`du -sh` 21 G), in `/tmp/q1d_mem/orig/`.
**sha256 verified against the HF LFS pointers — bit-exact on all three.** Read-only header re-check
confirms layers 0 / 20 / 39 with experts 0..383 present. Run throttled (`curl --limit-rate 20M`,
resumable) under nohup; survived disconnect. Idle-guard clean before/after (rc=0). No writes under
`~/repos/exo`; no git run on either node.
