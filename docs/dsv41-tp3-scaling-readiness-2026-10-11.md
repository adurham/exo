# DSv4.1-Flash — would a 3rd identical node scale? (TP3 prefill-first readiness + modeled scaling band)

**Date:** 2026-10-11 · **Author:** p20 campaign (delegated, offline)
**Live tree:** exo `ade481853` (`deploy/next22-rankfix`, detached HEAD) on both nodes; docs branch based on `origin/main` `beda06114`.
**Model:** `dealignai/DeepSeek-V4.1-Flash-UNCENSORED-EXL3-2.9bpw`, 2-rank tensor-parallel (`MlxJacclInstance`, `worldSize=2`), jaccl/RDMA over one Thunderbolt-5 cable pair.
**Every number is labelled `[MEASURED]`, `[MEASURED-STALE]`, `[DERIVED]`, `[BOUND]` or `[MODELED]`. Nothing here was measured on a 3-node cluster — none exists.**

---

## VERDICT (read this first)

**A 3rd identical node is *not* worth engineering for prefill on this stack.** The naive expectation ("3 GPUs instead of 2 ⇒ +50% prefill") is **not reachable**; the modeled band is **roughly flat-to-worse** (`[MODELED]` ≈ −20% to +20%, centre ≈ **+5 to +12% on the hard path**, and **−15% to −20% on the cheap path**). Decode is projected **flat-to-slightly-worse**.

Why the naive +50% fails, in one line: the DSv4.1 tensor-parallel design **replicates** exactly the terms that dominate at depth — the O(context) indexer (3.6% of a 16K chunk, ≈31–43% at 100–340K), the engram tables, RMSNorms and the hyper-connection mixes — and the attention path **cannot be split 3 ways without new uneven head/group geometry**, so a third rank adds a link and a hop but shrinks only the MoE expert GEMMs (≈31% of a 16K chunk) plus the head-sharded attention terms.

Cost to even try: **≈ 14–30 engineer-days** (Python geometry + transport reliability + launcher + tests), **two** new TB5 cables, and the transport's reliability (ARQ) layer — currently hard-gated to `size_==2` and *throwing* for N>2 — must be generalised or every 3-rank collective runs on the raw-UC path that this project has repeatedly measured to wedge.

**Recommendation:** do not purchase a 3rd node for prefill throughput. If the reopen-condition for the native-experts requant (`origin/main beda06114`) really needs a 3rd node, buy it for **memory headroom at 1M context / quality work**, not for speed. The prefill scaling bottleneck here is *not* node count — it is the replicated O(context) indexer and the unshardable attention path.

---

## Part 0 — what "the current 2-node sync" actually is (measured live)

**Shapes `[MEASURED]`** (node `/state` + checkpoint `config.json`, read-only):

| quantity | value |
|---|---|
| hidden `dim` | **5120** |
| layers | 40 body + 3 MTP/DSpark (`num_nextn_predict_layers=3`) |
| routed experts | 384, top-6 (+1 shared); draft 128, top-3 |
| MoE intermediate | **2304** (= 3×768, clean at world 3) |
| attention | 64 heads, `head_dim`=512, 1 KV head (MQA), `o_groups`=8 |
| vocab | **129280** (`%3 = 1`) |
| live instance | `worldSize=2`, `maxKvTokens=1048576`, `kvCacheBits=0`, sharding `Tensor` |

**Per-collective payloads (per 2048-row chunk, live bf16) `[DERIVED]`** — from `mlx_lm/models/deepseek_v41/moe.py:40-63` and `attention.py:241-243`:

| collective | shape | per call | ×40 layers / chunk |
|---|---|---|---|
| MoE all_sum (`moe.all_sum`) | `[2048, 5120]` bf16 | **20.97 MB** | **838.9 MB** |
| attention all_sum (`attn.all_sum`, head-shard partial) | `[2048, 5120]` bf16 | **20.97 MB** | **838.9 MB** |
| head all_sum (`ShardedHead`, `LMHEAD_LASTROW=1`) | `[1, 129280]` fp32 | ~0.5 MB | ~0.5 MB |

(fp32 exact for the MoE tail would be 41.94 MB/call — `moe.py:42`; `DSV41_MOE_ALLSUM_BF16` is default-ON so the live payload is the halved 20.97 MB.)
**Nominal sync traffic ≈ 1.68 GB per 2048-row chunk** (two 20.97 MB all_sums per layer × 40 layers). The reduced MoE tensor is the *full* 5120-wide output on **every** rank — it is not itself 3-divisible, and does not need to be.

**Fabric + transport config `[MEASURED]`** (live `/state` + `ps eww` + `ibv_devinfo`, read-only):

- jaccl **MeshGroup** (no `MLX_JACCL_RING` in the live env → not `RingGroup`), 2 ranks, **one** RDMA cable pair: `rdma_en4` (m4-1) ↔ `rdma_en3` (m4-2), `192.168.201.x`.
- 4 HCAs present on m4-1 (`rdma_en2..en5`), each **`max_qp = 3`** (per-HCA); the two `PORT_ACTIVE` inference links run at **80 Gb/s** (`10.0 Gbps × 8X`), not 120 — `rdma_en5` (120 Gb/s) is a *different* domain and is `PORT_DOWN`.
- Transport knobs (live): `MLX_JACCL_RELIABLE_DATA=1`, `RELIABLE_INFLIGHT=8`, `RELIABLE_MAX_SZ=2` ⇒ `FRAME_SIZE=4096`, one uniform **~15.9 KB** chunk class, `data_slots = 4`; the ARQ/reliable layer is **hard-gated `size_==2`**.
- Cluster was **idle** during this audit (`gpuUsage 1.7%`, `temp 33 °C`, `sysPower 8.8 W`, last requests = `GET /node_id`, `/metrics`).

---

## Part 1 — MEASURED per-collective cost and per-chunk sync share

### 1.1 The clean transport numbers (already measured; reused, not re-run)

| # | quantity | value | provenance |
|---|---|---|---|
| M1 | isolated jaccl `all_reduce`, 2–16 KB, cluster stopped | **118–127 µs**, flat (latency-bound) | `docs/offline-collective-microbenchmark-2026-08-21.md` |
| M2 | **in-production** jaccl-internal transport, decode, 8192 B, 45 666 calls/rank | median **36.1 / 36.0 µs**, mean 66.3 / 58.9, p95 165 / 143, p99 253 / 266 | `docs/jaccl-internal-timing-allsum-transport-fast-2026-08-21.md` |
| M3 | in-model per-op shares, **prefill @16K, chunk 2048** | routed experts **30.7%**, `fn:all_sum` **16.9%** (648 calls), dense EXL3 **16.6%**, `sparse_attn` 15.6%, engram 6.0%, RMSNorm 4.1%, indexer 3.6% | `docs/benchmarks/phase22-dsv41-prefill-speed-2026-10-01/README.md` §3.3 |
| M4 | in-model per-op shares, **deep prefill @~340K** | `attn.indexer.score` **42.9 / 44.2%**, `attn.sdpa` 31.0 / 31.1%, `ffn` 19.0 / 19.1%, `attn.all_sum` **6.9 / 5.5%** | `PERFORMANCE_HISTORY.md` (2026-09-23 window) |
| M5 | **decode** collective-removal bound (89K needle, identity vs clean) | **≤ 8.4%**; decode GPU busy 117.3% (saturated) | `PERFORMANCE_HISTORY.md` (2026-09-23) |
| M6 | prefill all_sum effective rate, **old fp32 build**, NOP-A/B | ~170–178 ms/call @16.8 MB ⇒ **~94 MB/s** (1.6% of link); 1029 chunks → 258 stop-and-wait rounds → 690 µs/round | `docs/moe-all-sum-skew-vs-comms-2026-08-19.md` — **[MEASURED-STALE]** |
| M7 | fresh per-chunk bound @100K (brief) | JACCL/all_sum **1–24%** of a 2048-row chunk; ~2–4% overlapable-class | `[BOUND]` (cited from the p20 brief / MLX_GPU_TIME fold) |
| M8 | hardware facts, **this session**, read-only | 4 HCAs × `max_qp=3`; active link 80 Gb/s; GPU idle 1.7% | `[MEASURED]` |

### 1.2 Per-chunk sync total and its share of a 2048-row chunk @ ~100K

- Per-chunk sync **traffic**: ≈ **1.68 GB** `[DERIVED]` (Part 0).
- Per-chunk wall @100K: ≈ **8.2 s** (2048 rows at the measured ~250 rows/s fresh band) `[DERIVED]`.
- **Share**: bounded **1–24%** (M7). The two in-model instruments bracket it: **16.9%** at 16K (M3, *inflated* by the fenced instrumentation that host-syncs each all_sum result) decaying to **6.9%** at 340K (M4). At ~100K the collective share therefore sits in the **low-to-mid teens**, `[DERIVED]`, consistent with the 1–24% / 2–4% / ≤8.4% trio.

**Why M6 cannot be used for the current build `[DERIVED]`.** At the historical 94 MB/s, 1.68 GB/chunk would take **17.8 s** — *longer than the whole 8.2 s chunk wall*. That is self-contradictory, and it is the expected signature of the known artifact: the 2026-08-19 NOP-A/B "removes all_sum" also removed the **drain of queued GPU work behind it**, so it priced the collective *plus* its drain. The 2026-08-21/22 work later showed the true transport is tens of µs (M2). M6 is retained only as an **upper bound**.

### 1.3 Did we re-measure? — **No. Said honestly.**

A fresh jaccl collective microbench at production shapes was **not run**, and cannot be run under this task's constraints:

- Both available methods need a boot. The isolated path (`mlx/distributed/jaccl/lib/examples/allreduce_bench.cpp` via `mlx.launch --backend jaccl`, the tool used for M1) **requires stopping the live cluster** — it needs the same RDMA interfaces and the device-wide `max_qp=3` ceiling is already fully held by the running runners, so a second jaccl client here risks the production QPs (`[MEASURED]` M8; precedent: `docs/offline-collective-microbenchmark-2026-08-21.md` §"requires stopping the cluster"). The in-production method (`JACCL_TRACE_TIMING`, the tool behind M2) needs a **relaunch with env** — also a boot. Both are forbidden here (0 boots, live traffic).
- No cheap in-engine null/duplicate-collective knob exists to bound it instead: a grep of the whole `dsv41` engine + `deepseek_v41` model for NOP/skip/duplicate-collective switches finds **none** (the `_MINIMAX_NOOP_*` knobs at `auto_parallel.py:1035-1046` belong to the MiniMax strategy, not this engine).

**So the honest Part-1 result is a reused-measurement bound, not a new number.** Tightening 1–24% requires one clean stop-and-bench window (the exact procedure is on file in `docs/offline-collective-microbenchmark-2026-08-21.md`) — a ~10-minute production interruption the owner must authorise.

---

## Part 2 — MODELED 3-rank ring

### 2.1 Fabric and protocol consequences of N=3 `[MODELED]` (inputs `[MEASURED]`)

- **Topology.** A jaccl **mesh** needs a direct RDMA cable per pair (`mesh.cpp:148-164`, `jaccl.cpp:141-155`; `placement_utils.py:638-642` raises otherwise). A 3-node full mesh ⇒ **2 cables/rank** (one to each peer), vs 1 today. `MESH_MAX_PEERS = 8`, so `size=3` is accepted by the C++ mesh — the topology is N-general, the *cabling* is not (Part 3).
- **QP budget.** `initialize()` builds **3 QPs per peer** (`mesh.cpp:810-859`), all borrowing one per-peer HCA. 3 ranks ⇒ 2 peers ⇒ **6 QPs**, but `max_qp=3` is **per HCA** — it fits **only if the two peer links enumerate as two distinct HCAs** (m4-1 has 4 HCAs, so plausible, but hardware-unverified).
- **Protocol.** At N=3, `all_reduce` with payload > 32 KiB routes to `all_reduce_scatter_gather` = reduce-scatter + all_gather (`mesh.cpp:2063` → `mesh_impl.h:2163-2210`). Wire bytes/rank ≈ `2(N−1)/N · payload` = **1.33× the N=2 figure**, and the step count goes **2 → 4**. ⇒ per-call sync time **1.3× (bytes) to 2.0× (rounds)** vs today.
- **Reliability.** The ARQ layer is `size_==2`-gated and **throws** for N>2 (`mesh_impl.h:706-709, 1171-1175, 1590`). At N=3 the production `MLX_JACCL_RELIABLE_*` env is **inert** and collectives fall to raw UC — the exact regime this project repeatedly measured to wedge. This is a *correctness* prerequisite, not a tuning item.
- **Straggler / slowest-of-3.** With 3 ranks the per-collective wait is `E[max of 3]` instead of `E[max of 2]` ⇒ **+≈0.29σ** per collective. Modeled as a growth factor `g` on the collective term.

### 2.2 Prefill scaling model `[MODELED]`

Per-rank chunk wall `T(w) = Σ_class share×scale_class`. A class measured at TP2 is either **sharded** (TP2 cost = full/2 → TP3 = full/3, ratio **2/3**) or **replicated** (TP2 = full → TP3 = full, ratio **1**), and the collective grows by `g ∈ [1.3, 2.0]`. Calibrated on the phase-22 16K attribution (M3), the live shard map from `exl3_build.py` (`attn.wq_b`/`attn.wo_b` sliced; `indexer.*`, `wq_a`, `wkv`, norms, engram **replicated**):

| shardable share of the TP2 chunk wall | path | modeled TP3 speedup | 
|---|---|---|
| S ≈ 0.54 (experts + `sdpa` + dense-o, **uneven heads 22/21/21**) | hard | **1.04× – 1.19×**  (g 2.0 → 1.3) |
| S ≈ 0.31 (experts only; **attention replicated**) | easy | **0.78× – 0.86×**  (a **regression**) |

- **Naive +50% is unreachable.** A genuine 1.5× needs ~100% shardable; this stack is ≈54% at best.
- **Depth makes it worse.** The replicated indexer grows from 3.6% (16K) to 43% (340K) (M3/M4), so at the brief's 100K target the shardable fraction is **below** the 16K figure and the band shifts **down**.
- **5120/3 "padding waste" is a red herring.** The MoE all_sum payload is the full 5120-wide reduced tensor on every rank — it does not divide by 3 and need not. The expert intermediate **2304 = 3×768 is exact** (incl. the DSpark draft — see Part 3). The real 3-way blocker is *geometry*, not padding: heads `64 % 3`, groups `8 % 3`, vocab `129280 % 3`, and the `n % (8·world)` tile alignment on three dense tensors (`exl3_build.py:316-326`) — each a **hard `raise`**, not a rounding loss.
- **"seq-split-3" is not applicable to this model.** The v41 attention has **no** seq-split and **no** `all_gather`; its only cross-rank op is `attn.all_sum` (head-shard partial, `attention.py:241-243`). The live `EXO_DSV4_SEQ_SPLIT=1` knob lives in `auto_parallel.py`/`deepseek_v4.py`, which the v41 engine **never calls** — it is inert here.

### 2.3 Decode projection `[MODELED / GUESS]`

M=1 decode is weight-read + dispatch bound, and already GPU-saturated (117.3% busy, M5). The third node's extra FLOPs are largely idle per token; what *does* change is the collective — 1.3–2× per call, on a term already ≤8.4% of the step. Expert weights split 3 ways shave a little per-rank read time. **Net: flat ± a few %, with a real risk of a small regression** from the extra round-trips and the loss of the ARQ layer. Not a reason to buy.

---

## Part 3 — CODE READINESS AUDIT (offline; two independent, source-cited audits)

### 3.1 Python / sharding / placement (`scratch/tp3/part3a-python.md`, sha `00002deb…`)

| # | item | finding | evidence | effort | risk |
|---|---|---|---|---|---|
| 1 | dense `_block_bounds` @w3 | `n_tiles % (8·3)` — **`attn.wq_b` (2048t), `attn.wo_b` (512t), `head` (8080t) all RAISE**; `wo_a` raises on groups (8%3). Only shared-expert w1/w2/w3 pass. | `exl3_build.py:316-326,336,340,364,366,811` | 2–8 d | med |
| 2 | attention @w3 | **head-sharded, not seq-split**; `64%3`+`8%3` ⇒ `raise` today. Fixes: replicate (`DSV41_TP_ATTN=0`, exact but see §2.2 — a prefill regression) or **uneven 22/21/21 heads / 3-3-2 groups** | `attention.py:151,234-243`; `exl3_build.py:698-709` | 0.5–9 d | med→high |
| 3 | head / vocab | `129280 % 3 = 1`; `ShardedHead._lo = rank·(vocab//world)` floor ⇒ **row 129279 = 0 on all ranks** (greedy-argmax / top-k correctness risk); the slice also raises | `exl3_build.py:399,408-418,436-440,811` | 0.5–2 d | low→med |
| 4 | routed-expert slice | **clean**: 2304 = 3×768 (48 tiles, `%8=0`, `%128=0`). Latent: only `%world` asserted, not `%(8·world)` | `loader.py:191-208`; `exl3_moe.py:327,366,802` | 0.5 d | low |
| 5 | engine world assumptions | **one** hard gate: `world not in (1,2)` | `load.py:78-86` | 0.5 d | low |
| 6 | placement / API | `hidden_size % len(cycle)` filters out every 3-cycle (5120%3=2); `min_nodes` plumbing is N-general | `placement.py:122,170-192`; `placement_utils.py:497-527` | 1–3 d | med |
| 7 | park / coord N=3 | **generalises as written** (everything sizes off `group.size()`); only the *test fakes* hard-code world=2. Site-identity fix = commits `ea3175a6b` + `30f32e448` | `coord.py:436-521,562-608`; `park.py:186-189,565-570` | 0.5–1.5 d | low |
| 8 | tests | suite **blind** to w=3; `test_dsv41_thinking_load.py:78` actively asserts w=3 **is refused** (must be rewritten) | `test_dsv41_thinking_load.py:67-92` | 1–2 d | low |

Python-side total: **≈ 6.5–27 eng-days** (min-risk ≈5–9; full TP-3 sharding ≈16–27).

**Resolved unknown (this session, `[MEASURED]` from the checkpoint header):** the **DSpark draft expert width is 2304** (`mtp.0.ffn.experts.0.w1.trellis = [320,144,64]`, `suh=[5120]`, `svh=[2304]`) — identical to the body, so the draft MoE is **TP3-clean**. The draft *markov head* shares the vocab `//world` floor-division defect (item 3).

### 3.2 Transport / launcher (`scratch/tp3/part3b-transport.md`, sha `9a2eb80c…`)

| # | item | finding | evidence | effort | risk |
|---|---|---|---|---|---|
| 1 | topology | mesh is N-general (`MESH_MAX_PEERS=8`; 4-node example ships) but needs a **cable per pair ⇒ 2 cables/rank** | `mesh.cpp:148-164`; `jaccl.cpp:141-175`; `examples/minimal_cfg.cpp:9-13` | 1–2 d (+cables) | med |
| 2 | **QP / reliability** | 3 QPs/peer × 2 peers fits only on 2 distinct HCAs; **ARQ layer throws for N>2** (`size_==2` gate) | `mesh.cpp:225-280,810-859`; `mesh_impl.h:706-709,1171-1175,1590` | **3–6 d** | **high** |
| 3 | coord group | N-general; port election = rank 0; `has_split_` hazard avoided on the coord path | `coord_group.h:164-173,232-281`; `mesh.cpp:656-720` | 0–1 d | low |
| 4 | all_gather size | **no size rejection** in the transport (chunked ≤512 KiB); the `exl3_build.py:392` "rejects this size" comment has no matching throw here | `mesh_impl.h:20,1905-2051`; `rdma.h:75,120-129` | 0 (doc) | low-med |
| 5 | rendezvous | matrix + coordinators + `mlx.launch` all N-general; add the 3rd host in discovery | `utils_mlx.py:168-189`; `placement_utils.py:582-645,881-981` | 1–2 d | med |
| 6 | launcher | **~15 two-node hardcodes** in `start_cluster.sh` (NODES array, Tailscale/TB discovery pair-only, per-cable route repair, `min_nodes=2` in two create calls) + relay/route machinery **commented out** | `start_cluster.sh:1051-1066,1113-1114,1156-1177,1229-1278,1339-1364,3833,3891` | 3–6 d | med-high |
| 7 | placement / API | API N-general; the gate is the geometry (see 3.1 #6) | `placement.py:180-192` | 0.5 d + geometry | high |
| 8 | tests | no N=3 transport test; placement suite covers only 3-node *rejections* | `test_placement.py:526,597,637,1046` | 3–5 d | low |

Transport-side total: **≈ 11–22 eng-days**.

### 3.3 Combined engineering estimate

**≈ 14–30 engineer-days** (items 3.1 #6/#7/#8 and 3.2 #7 overlap; deduped), of which the **prefill-viable** path (uneven head/group split, 3.1 #1/#2) is the **≈16–27 d** end and the "easy" replicate-attention path (5–9 d) is a **prefill regression** per §2.2. The **critical-path prerequisite** is 3.2 #2 (transport reliability at N>2) — a >1-week task on its own.

---

## ACQUISITION CHECKLIST (only if the 3rd node is pursued)

Buy **only** if all of these hold — the whole value depends on parity and cabling:

- **Same chip generation: M4 Max.** The MLX Metal kernels and the hand-built `jaccl` transport are compiled per-arch and were tuned on this exact part; a different generation invalidates both.
- **128 GB unified memory.** The per-rank envelope today is ~108 GB under a 115 GB wired limit; TP3 lowers weight-per-rank but must still absorb KV + engram tables + prefill transients at 1M context.
- **≥ 1 TB SSD.** Engram tables + park/restore store + model cache live on SSD.
- **macOS version match.** `buffer_size_from_message()` returns `{0,4096}` on macOS < 26.3 (`rdma.h:120-129`) — a version mismatch silently changes the buffer classes and the protocol path.
- **Two TB5 cables**, one to **each** existing node (2 cables/rank for a 3-node mesh). A single cable to "the cluster" is not enough.
- **Two distinct RDMA HCAs on the new node's links** (each HCA caps at `max_qp=3`; 3 QPs/peer × 2 peers = 6 needs 2 HCAs).
- **Used-market price:** omitted — not trivially findable offline, and the verdict does not turn on it.

**Validated pre-purchase (offline, no hardware):**
- All of 3.1 #1, #3, #4, #5, #6, #7, #8 and 3.2 #1, #3, #4, #5, #6, #7 — the divisibility math, the head remainder, `tp_geometry`, the placement predicate, the launcher hardcode list, the park/coord N-generalisation and the six new CPU-only tests.

**Only validatable with the hardware:**
- Whether the new node's two peer links land on two distinct HCAs (the `max_qp` fit).
- The per-HCA HopID/QP budget at 3 ranks, and 3-node direct-link route/relay bring-up.
- The **true** MLX-level `all_gather` cap, if any (above the jaccl lib) — the in-repo "rejects this size" comment is unreproducible offline.
- End-to-end 3-rank `mx.distributed.init(backend="jaccl")` correctness and the 3-rank hang-mode (one diverging rank hangs the other two).

---

## Method, provenance, and non-disturbance

- **No boots, no config changes, no node-side file edits, no purchases, no TP3 build work.** All node contact was **read-only over ssh**: `ps eww` (live runner env), `curl /state` + `/metrics`, `uptime`/`ps` (idle check), `ibv_devinfo` (HCAs/QP/ports), and one `safetensors` header read (DSpark draft shape).
- The studio pair served live production traffic throughout; the audit was **idle-guarded** — verified idle (`gpuUsage 1.7%`, `32–33 °C`, only `/node_id` + `/metrics` requests) and touched nothing that could delay a request. **No bench was run**, precisely because the only bench paths would have disturbed the live QPs or required a boot (§1.3).
- **Own worktree** (`/private/tmp/p20-tp3-scaling`, branch `p20/tp3-scaling-readiness`); the shared checkout at `ade481853` was left untouched.
- Audit inputs: `scratch/tp3/part3a-python.md` (sha256 `00002deb44cac98392879c94141487eb895d400b2a81d4a94132989db083ef49`) and `scratch/tp3/part3b-transport.md` (sha256 `9a2eb80c2dda7599f168291a900fa1debf44993a63164a0dbb24c316516898ab`); every file:line in Part 3 was re-verified against the live tree by the author.

**Corrections to the task brief (recorded so they are not re-litigated):**
1. `DeepseekV4ShardingStrategy` (in `auto_parallel.py`) is **not** the live TP path — the DSv4.1 engine builds its own TP in `mlx_lm/models/deepseek_v41/exl3_build.py`.
2. There is **no attention seq-split / all_gather** in this model; attention uses a head-shard `all_sum`.
3. "MoE intermediate 5120/3" is a misread of `hidden=5120`; the MoE **intermediate is 2304 (=3×768, exact)**.
4. The site-identity fix commit is `ea3175a6b`, not `ae3175a6b`.
