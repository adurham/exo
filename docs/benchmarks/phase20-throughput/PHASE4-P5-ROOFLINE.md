# PHASE 4 — P5 "roofline close-out"

Author: Phase-4 P5 subagent. Opened 2026-10-08. **Bench-only, OFFLINE analysis.** No
cluster access (no ssh, no relaunch, no API POST); no GPU work; pure-arithmetic +
source-read. Committed on branch `deploy/phase20-campaign` (worktree
`/private/tmp/phase20-campaign`).

Companion artifact: `/Users/adam.durham/.hermes/cache/scratch/p4/read_bw_canary.py`
(read-bandwidth canary — design/script only, NOT run here; to be run later on an
idle cluster node).

Two deliverables per the P5 brief:
1. **Part A** — reconcile `verify_block 92.4 ms` vs the client round `101.06 ms`
   (the brief's "91.5% vs 98.5%").
2. **Part B** — a bytes-roofline for **one verify round at 91K, m=4**, vs the measured
   read bandwidth, with a near-floor-or-headroom verdict + a localizing differential.

---

## 0. Method, provenance, and the one caveat that governs everything below

- **Frozen server numbers** (`PHASE3-M3.md` §2): `verify_block 92.4`, `draft_build 0.55`,
  `tail_bookkeep 0.09`, `round_total 93.8` ms — the **next16-instr build** (`f234b0f6d`)
  with **both levers OFF** (`DSV41_SPARSE_COLSPLIT=0 DSV41_INDEXER_HIER=0`),
  `round_prof=1`, per round, both ranks (rank0==rank1 to <0.1 ms).
- **Frozen client number** (`PHASE3B-SHIP-VALIDATION.md` §6): the **same-session**
  `next17 + DSV41_INDEXER_HIER=0` arm, agentic 91K g3, **4 reps, ms/round median 101.06**
  (101.06, 101.07, 100.67).
- **Re-derived from the raw PROF JSONL** (`raw/p3/off_s{1,2}_rank{0,1}.round_prof.jsonl`,
  3068 rounds each): median `verify/round_total = 0.9845`, `draft_build 0.564`,
  `tail_bookkeep 0.090`, `round_total 94.05` — the frozen table reproduced.
- **Cross-build/cross-workload caveat (load-bearing).** The server PROF brackets come
  from the `next16-instr` build on a **benign-shaped** workload; the client 101.06 is the
  `next17` build on the **agentic** arm. The two are *not* the same boot, *not* the same
  request, and *not* the same shape. Part A reconciles the **arithmetic of the two known
  quantities**; it does **not** claim the residual is a measured per-round overhead (§1.4).
- **The 92.4 ms bracket is an upper bound on the pure verify forward.** Per
  `rounds.py:509-528` it absorbs the verify forward **+ the draft's GPU compute + the
  compiler's per-layer host round-trips + the previous round's deferred rollback/
  `append_ctx` tail**. So "verify/round" slightly *understates* the true verify share, and
  using 92.4 in Part B is a conservative (slightly high) numerator.

---

## 1. PART A — round-accounting reconciliation

### 1.1 The two ratios

| ratio | value | meaning |
|---|---:|---|
| `verify_block / round_total` = 92.4 / 93.8 | **98.51 %** | share of the **server-internal** round bracket |
| `verify_block / client round` = 92.4 / 101.06 | **91.43 %** | share of the **client-observed** per-round time |

Both are correct; they are shares of **two different denominators**. The brief's "91.5%"
is the second; the campaign's "98.5%" is the first. Neither is wrong — they answer
different questions.

### 1.2 The missing 8.66 ms, split

`client − verify_block = 101.06 − 92.4 = 8.66 ms`, split exactly and MECE:

| # | residual | ms | expression | arithmetic |
|---|---|---:|---|---|
| (a) | **in-round server residual** | **1.40** | `round_total − verify_block` | 93.8 − 92.4 |
| — | of which `draft_build` | 0.55 | PROF | (frozen) |
| — | of which `tail_bookkeep` | 0.09 | PROF | (frozen) |
| — | of which **unaccounted** (host / serialisation) | **0.76** | 1.40 − 0.55 − 0.09 | |
| (b) | **client↔server per-round boundary** | **7.26** | `client − round_total` | 101.06 − 93.8 |
| | **total** | **8.66** | (a)+(b) | 1.40 + 7.26 ✓ |

As fractions of the client round: (a) = **1.39 %**, (b) = **7.18 %**.

### 1.3 Mechanism of (b), stated honestly

The two quantities are computed differently:

- **Client `ms_per_round`** = `decode_s × 1000 / rounds`
  (`phase19_round_measure.py:135`), where
  `decode_s` = wall between the **first and last decoded content/reasoning SSE delta**
  (`p3_driver.py:73`, `:65-70`), and
  `rounds` = **`mtp_cycles_cumulative / gamma`** (`phase19_round_measure.py:125`) — a
  *server counter* read out of the `: generation_stats` SSE frame.
- **Server `round_total_ms`** = the `_one_round` bracket in the engine
  (`engine.py:1066→1079`), i.e. **one round's server-side span**, and it **excludes**
  `emit_ms` — the window between one round's `yield` and the consumer draining it
  (`engine.py:1064`), which is exactly the client/HTTP portion.

So (b) is the **client-side bookkeeping + HTTP/SSE streaming of the round's tokens +
inter-round scheduling/hand-off gap** between decoded rounds, plus any residual
statistic mismatch (see 1.4). It is **not**:

- **draft-forwards** — `draft_build_ms` (0.55) is *inside* `round_total` (in-round, §1.2a);
  the draft's *GPU* work is likewise billed to `verify_block`. There is no draft time
  hiding in (b).
- **the C1 host-sync** — that lever was removed (lever-1 code guard / `HIER=0` on this arm);
  the twin-default foot-gun (`attention.py:58-68`) is documented in `PHASE3-M3.md` §4.

### 1.4 What is measured vs bounded — and a new decomposition finding

- **(a) = 1.40 ms is measured**: both `round_total` and `verify_block` are PROF fields of
  the *same* round on the *same* build, and (a) = the two fixed brackets + a 0.76 ms
  unaccounted host/serialisation remainder.
- **(b) = 7.26 ms is bounded by arithmetic only, not directly measured.** A PROF-capable
  build (`next16-instr`) would decompose it, and **none is deployed**: the live production
  build (`deploy/next13 @ 576e9d279`) treats `round_prof` as a no-op (`PHASE3-M3.md` §2b).
  Worse, (b) is a **cross-build difference** (next16-instr vs next17) *and* a
  **cross-workload difference** (benign vs agentic), so it also contains session variance
  and any PROF-instrumentation overhead of the next16 build (sign unknown — if the PROF
  build is *slower*, the true boundary is **> 7.26 ms**).
- **Statistic parity.** The client 101.06 is a **median of 4 reps**, the server 92.4 a
  **median of 5 benign reps** — both medians, so no mean-vs-median skew *between* them;
  but since the two come from different workloads (below), the point stands that (b) is a
  *bounded* rather than *apportioned* number.
- **New finding while re-deriving the PROF: the raw file pools two workloads.** The
  `raw/p3/off_*.round_prof.jsonl` files are **bimodal** in `verify_block_ms` (runs of ~92.5
  and ~97.8, separated by 24 single-round transition rows — the driver interleaved the
  benign and agentic arms under one `--arm both` run that wrote one shared rank file):

  | component (raw PROF, by mode) | n | `verify_block` | `round_total` | `verify/round_total` |
  |---|---:|---:|---:|---:|
  | **LOW mode** (verify < 95, **benign** shape) | 2012 | 92.25 | 93.71 | **0.9845** |
  | **HIGH mode** (verify ≥ 95, **agentic** shape) | 1056 | 97.81 | 99.27 | **0.9853** |
  | all pooled | 3068 | 92.61 | 94.06 | 0.9845 |

  → the frozen `92.4 / 93.8` is the **benign component** (the agentic component of this
  OFF build is `97.8 / 99.3`). **The 98.5% ratio is arm-robust** (98.45-98.53 % on both).
  This strengthens the reconciliation: the client−verify gap on the *agentic* shape is
  `101.06 − 97.8 = 3.3 ms`, **smaller** than the benign-component 8.66 — i.e. the brief's
  "8.7 ms unattributed" is dominated by comparing *client-agentic* against
  *server-benign*, not by a large in-round overhead.

### 1.5 Extra server-side timing fields on the live build — checked, none decompose (b)

Grepped the `: generation_stats` frame and the dsv41 engine (`api/types/api.py:184-212`).
The live frame carries `generation_tps`, `prompt_tps`, `mtp_cycles_cumulative`,
`mtp_accepted_drafts_cumulative`, `peak_memory_usage`, `phase_marks_ms`, `api_phase_marks_ms`.
There is **no per-round timing** in it. `phase_marks_ms` / `api_phase_marks_ms` are
**`None`** unless `EXO_DSV4_SECTION_TIME` is on, and even then they are per-forward
section accumulators, not a round bracket. **Conclusion: no deployed field decomposes (b);
the bound stands at 7.26 ms (arithmetic).**

---

## 2. PART B — bytes-roofline for ONE verify round at 91K (m = 4, γ = 3)

### 2.1 Geometry, read from the code (per-rank, world = 2)

| symbol | value | source (file:line) |
|---|---:|---|
| `D` (hidden) | 5120 | `config.py:33`; model card `…EXL3-2.9bpw.toml:16` |
| `n_layers` | 40 | `config.py:34`; card `:14` |
| `moe_inter_dim` | 2304 | `config.py:35` (`moe_intermediate_size`) |
| `H` = MoE width **per rank** | **1152** | = 2304 / world(2); confirmed from the real loader — `PERFORMANCE_HISTORY.md:11146` "H=1152 … world=2" |
| `E` routed experts | 384 | `config.py:48`; card `:17` |
| top-k routed | 6 | `config.py:50` |
| quant | EXL3 2.9 bpw | card `:64`; `README` phase-1: 3-bit, **2-bit on layers 18-22** |
| per-expert bytes | 6.64 MB (k=3) / 4.42 MB (k=2) | = 3 · H · D · k/8; measured `PERFORMANCE_HISTORY.md:11146` "6.64 MB/expert → 2.55 GB/layer" |
| `head_dim`, `n_heads` | 512, 64 | `config.py:38-39` |
| `window` (sliding) | 128 | `config.py:44` |
| `index_topk` (K) | 512 | `config.py:84` |
| `index_head_dim` | 128 | `config.py:83` |
| index-source layers | 20 (every other non-window layer) | loader; `_shard`-consistent census (21 in the 43-layer V4 file) |
| verify microbatch `m` = γ+1 | 4 | γ=3; `PHASE3-M3.md` §5 |
| context `L` | 91 000 | `PHASE3B` §6 "agentic 91K" |

Dense-attention + shared-expert EXL3 params per layer per rank = **141.0 M** (measured,
`phase3-kernel-sourceread.md:256`; = `wq_a 5120×1280 + wq_b 1280×16384 + wkv 5120×512 +
wo_b 8192×5120 + wo_a 8×(4096×1024) + shared w1/w3 5120×2304 + shared w2 2304×5120`, halved
on the sharded axis).

### 2.2 Bytes READ in one verify round (per rank, γ=3 → m=4)

| component | arithmetic | bytes | GB |
|---|---|---:|---:|
| **routed-expert weights** | 24 slots (= 4 rows × top-6) × **0.73** dedup × 6.36 MB/expert (avg over 40 layers: 35×k3 + 5×k2) × 40 layers | 4.46e9 | **4.46** |
| **dense / shared / attention EXL3** | 141.0 M params/layer × 40 layers × 2.9 bits/8 | 2.04e9 | **2.04** |
| **KV read** | window `40·128·512·2` = 5.2 MB + top-k gather `40·(128+512)·512·2` = 26.2 MB + index-scan `20·(91000/4)·128·2` = 116.5 MB | 0.148e9 | **0.15** |
| **collectives (TP all_sum)** | MoE tail 40×(4·5120)×2 + attn tail 40×(4·5120)×2 | 3.3e6 | **0.003** |
| **TOTAL** | | **6.65e9** | **6.65** |

Notes / ranges:
- **Expert dedup factor.** Phase-17 measured that a 4-row window touches **59-87 %
  unique experts per layer** (`phase17-.../README.md:26`), i.e. R=4 reads **~3×** R=1 — this
  is the dedup of the 24 (row, expert) slots into distinct experts. Central **0.73**
  (→ 4.46 GB); the consistent range is **0.59 → 1.00** (→ 3.60 → 6.10 GB). I report 0.73 as
  central and carry the range through the verdict.
- **KV is genuinely small.** The window ring is capped at 128 entries, so window attention
  is **O(1) in L**; the top-k gather is capped at K=512, so sparse attention is **O(1) in
  L**. Only the **indexer score scan** is context-linear (`~L/4 × 128 B`, `indexer.py:506+`)
  and it is **0.12 GB** — **0.2 % of the total**. *Correction to a common claim: "the KV
  read" is **not** a material byte term at 91K.*
- **Not counted** (each ≤ 0.1 GB or uncertain): engram SSD tables (streamed, not weights),
  the 129K-vocab head (sharded; small at m=4), window/comp gather **scratch** (transient),
  indexer intermediate writes. These do not move the verdict.

### 2.3 Floor vs measured

Measured achievable read bandwidth: **~450 GB/s** (repo loop-2 anchor,
`PERFORMANCE_HISTORY.md:11145`: "15.14 TFLOPS dense bf16 matmul … ~450 GB/s triad"). State
the caveat: 450 is a **triad** (read+write) figure, and a **pure-read** kernel can sit
somewhat lower; I therefore give the range **400-550 GB/s**.

| bandwidth | bytes floor (`total / bw`) | measured 92.4 / floor |
|---|---:|---:|
| 400 GB/s | 16.6 ms | **5.6×** |
| **450 GB/s (central)** | **14.8 ms** | **6.3×** |
| 550 GB/s | 12.1 ms | 7.6× |

Achieved rate over the central model = `6.65 GB / 92.4 ms = **72 GB/s** = **16 % of 450`
(20 % over the no-dedup bound `8.29 GB`).

**Sanity check that bytes *is* the binding roof.** FLOPs/rank at m=4 = `2·m·params`:
dense `2·4·141M·40 = 45.1 GFLOP` + routed `2·24·6.36MB·40 = 12.2 GFLOP` ≈ **57.3 GFLOP**;
at the measured 15.14 TFLOPS that is a **3.8 ms** compute floor — **4× below** the 14.8 ms
bytes floor. So the round is **memory/bytes-bound**, which is what makes a bytes-roofline
the right tool. (Independently corroborated by a full-model verify measurement: `PHASE3-M3`
agentic OFF round_total 101.3 ms ⇒ ~30.2 t/s, right at the 6.3× picture.)

### 2.4 VERDICT — **NOT near-floor; ~6× headroom. Name the component.**

Near-floor for this stack would be **within ~10-15 % of 14.8 ms**, i.e. ≈ 16-17 ms. The
measured verify is **92.4 ms ≈ 6.3× the bytes floor (range 5.0-7.6×)**. **Headroom = ~77.6 ms.**

Slice decomposition of that 77.6 ms gap (per-rank, using the repo's own macro-op
measurements):

| slice | time | its floor | ×floor | excess | share |
|---|---:|---:|---:|---:|---:|
| **dense/shared/attn EXL3** | 38.5 ms¹ | 4.5 ms | **8.5×** | **34.0 ms** | ~44 % |
| routed experts | ~15-30 ms² | 9.9 ms | ~1.5-3.0× | ~5-20 ms | ~10-26 % |
| KV read | ~0.3 ms | 0.3 ms | 1× | ~0 | ~0 % |
| other (indexer compute, host, collectives, head) | remainder | — | — | ~24-38 ms | ~30-50 % |

¹ microbench-**extrapolated** (laptop M4 Max, synthetic-but-faithful trellis, cos 0.9999999):
`962.6 µs/layer × 40 = 38.5 ms` at M=4 (`phase3-kernel-microbench.md`), i.e. **8.5×** its
`2.04 GB` floor. Its own M-proportional marginal (`99 µs/layer/row × 3 × 40 = 11.9 ms`) and
phase-17's in-situ dense marginal (`+3.6 ms`/8 layers ⇒ ~`18 ms`/40 layers for the same 3
extra rows, `README.md:20`) agree to ~1.5× — the dense path is neither free nor a
hidden lever, it is *slow per byte* at the small-M shape.
² in-situ-**inferred, soft**: phase-17's R=1→R=4 expert increment `+4.6 ms` over 8 layers
(`README.md:21`) scaled ×5 ⇒ ~23 ms central, but the byte side of it depends on how many
*extra distinct* experts 3 extra rows add (the 0.59-1.00 dedup), so the implied rate ranges
~117-194 GB/s. Both the time and the bytes here carry ~1.5× uncertainty — hence a range,
not a point.

**Which component (robust across the uncertainty):** *dense* is the **largest single slice
(34 ms, 44 % of the gap) and the clear worst-per-byte (8.5× its floor)** — the EXL3 dense
small-M GEMM reads at **~53-55 GB/s (`12.2 % of 450`)** and equals a plain fp16 matmul of the
2×-wider materialised weight (`arm A ≈ arm B`, 962.6 vs 996.1 µs), i.e. it is
**decode-ALU/issue-bound at a ~55 GB/s machine plateau**, not byte-bound. The **routed-expert
path is closer to honest bandwidth** (~1.5-3× its floor; phase-17: "bandwidth, not waste";
phase-9: EXL3 MoE is 1.61-1.74× mxfp4). So the defensible statement is: **dense is the
biggest single slice, the worst per byte, and the lever; experts are the second slice and
are not obviously wasteful.**

**Localizing differential (design, not run).** Because KV is ~0.2 % of the bytes, a
KV-knob differential cannot move ms/round — *do not bother*. The decisive test is the
**top-k routing differential**: force the routed top-k down (6 → 2/3) via a debug hook and
re-measure ms/round. **If halving activated experts barely moves ms/round, the gap is
dense/other, not expert bandwidth**; if it scales ~linearly, experts dominate. Phase-17's
R-sweep is the existing partial evidence that experts scale with rows — this differential
isolates the *expert-count* axis at fixed m. A second, cheap differential: set
`DSV41_DENSE=<affineN>` (already in `exl3_build.py:461`) to compare the EXL3 dense path vs a
plain affine dense path at the same shapes — if the EXL3 path is far slower per byte, that
confirms the dense slice is the lever.

---

## 3. Read-bandwidth canary (design + script)

Path: `/Users/adam.durham/.hermes/cache/scratch/p4/read_bw_canary.py`. **Analogue of the
matmul canary** (`~/.hermes/cache/scratch/gpu_canary2.py`, which measures the 15.14 TFLOPS
compute ceiling). This one measures the **achievable READ bandwidth** on a node — the
denominator of this whole report — because 450 GB/s is a *triad* number and a pure-read GEMV
is what the verify path actually does.

Design (script is written; **not run here** — it needs an idle GPU):
- **Size**: arrays ≥ 256 MB so they dwarf L2/machine caches; total working set ≥ 2 GB to
  force DRAM/SLC-miss traffic, matching the verify path's regime.
- **Three arms**, each amortised over K inner iterations to bury the per-eval floor
  (the repo measured a **150-250 µs** `mx.eval` floor — timing one kernel per eval is
  meaningless; `phase3-kernel-microbench.md:36-46`):
  1. **pure read (reduction)** — `mx.sum(x)` over a large fp16 array: the closest analogue
     to "stream weights once, no writeback". Report GB/s = bytes / time.
  2. **read+write (triad-like)** — `y = a*x + y` over ≥2 GB: reproduces the 450 GB/s anchor.
  3. **GEMV-shaped read** — `x @ W` with a 1-row `x` and a large `W` (the verify M=1 shape):
     this is the number that should replace 450 in the Part-B denominator for a single-row
     read.
- **Warm-up** 3 iters before timing; **median of ≥5** timed reps; report per-arm GB/s and
  the arm-1:arm-2 ratio.
- **Health gate**: flag DEGRADED if pure-read < 0.6 × triad (a stuck/thermally-throttled
  node).
- **Output**: one JSON line `{"arm1_read_gbps":…, "arm2_triad_gbps":…, "arm3_gemv_gbps":…,
  "healthy":…}` so the same file can be run per node and diffed.
- Pure `mlx.core`; no network, no cluster, no writes outside stdout; safe to run on an idle
  boot between campaigns.

---

## 4. Citations (every geometry number)

**Server bracket definition & client metric**
- `mlx-lm/.../deepseek_v41/rounds.py:509-528` (`_one_round` doc: what each bracket covers;
  verify absorbs draft GPU + prior-round deferred tail), `:462-686` (bracket timer lines
  `t_draft 591`/`t_verify 622`/`t_tail 655`), `:675-680` (`enter_fields`).
- `src/exo/worker/engines/mlx/dsv41/engine.py:1018` (`_rounds`), `:1064` (`emit_ms`),
  `:1066→1079` (`round_start` → `round_total_ms`), `:1080-1082` (`flush_pending`).
  (next16-instr build `f234b0f6d` — the PROF-capable build.)
- Client metric: `phase19_round_measure.py:125,135` (`rounds = cycles/gamma`,
  `ms_per_round = decode_s×1000/rounds`), `p3_driver.py:59-73` (`decode_s` = first→last SSE
  delta), `:45-48` (`: generation_stats` frame).
- Live frame fields: `src/exo/api/types/api.py:184-212` (`GenerationStats`; no per-round
  field; `phase_marks_ms` None unless `EXO_DSV4_SECTION_TIME`).

**Part B geometry**
- `mlx-lm/mlx_lm/models/deepseek_v41/config.py:33-50,83-84` (D, n_layers, moe_inter_dim, E,
  top-k, head_dim, n_heads, window, index_topk, index_head_dim).
- `mlx-lm/mlx_lm/models/deepseek_v41/moe.py:152-158` (`SwitchGLU(dim, moe_inter_dim, E)`,
  shared expert), `moe.py:56-63` (MoE `all_sum` payload).
- `mlx_lm/models/deepseek_v41/attention.py:102-109` (attn linears), `:148-157` (q/kv
  projections), `:159-173` (window ring write), `:222-228` (`sparse_attn`).
- `mlx_lm/models/deepseek_v41/collective.py:56-63` (all_sum), `attention.py:241-243` +
  `moe.py:180-189` (attn/MoE all_sum sites).
- `cache.py:67-92` (per-layer cache: window ring + comp_kv + index_k), `:121-130`
  (window is O(1)), `:9-29` (bf16 storage note).
- `sparse_attention.py:19-64,127-151,204-243` (gather, two-source, C1 colsplit).
- Model card `resources/inference_model_cards/dealignai--DeepSeek-V4.1-Flash-UNCENSORED-EXL3-2.9bpw.toml:14-23,64`
  (n_layers 40, hidden 5120, E 384 top-6, head_dim 512, 2.9 bpw, TP built into
  `exl3_build`).
- `exl3_build.py:354-357,461,507-551,631,713` (TP sharding: experts/attn/shared/head),
  `:276-294` (`_slice_dense`).

**Measured anchors**
- `docs/PERFORMANCE_HISTORY.md:11145` (15.14 TFLOPS dense bf16; ~450 GB/s triad),
  `:11146` (world=2 D=5120 H=1152 E=384, 6.64 MB/expert → 2.55 GB/layer).
- `docs/benchmarks/phase17-dsv41-verify-cost-2026-09-28/README.md:18-31` (verify-row ablation:
  experts +4.6, dense EXL3 +3.6 [8-layer], head +1.0, sparse-attn +0.3; "4-row window touches
  59-87 % unique experts … bandwidth, not waste"; "dense … the one real lever left"), `:57`
  (verify R1..R6).
- `docs/benchmarks/phase19-latency/raw/phase3-kernel-microbench.md` (arm A/B/C table: M=4
  962.6 µs, 52.9 MB/layer, 54.9 GB/s; arm A ≈ arm B; eval floor 150-250 µs) and
  `phase3-kernel-sourceread.md:239-293` (dense roster, per-rank shapes, rooflines).
- `docs/deepseek-v41-exl3-plan.md:16-17,301` (EXL3 MoE 1.61-1.74× mxfp4 at TP=2).
- `PHASE3-M3.md:42-52` (frozen brackets), `PHASE3B-SHIP-VALIDATION.md:157-168` (client arms).
- `raw/p3/off_s{1,2}_rank{0,1}.round_prof.jsonl` (raw PROF; re-derivation + 1.4 bimodality).

---

## 5. Limitations (stated, not hidden)

1. **Part A (b) is bounded, not measured** (§1.4) — no PROF-capable build is deployed; and
   (b) is a cross-build / cross-workload difference, so it is an *upper bound envelope* on
   the true per-round boundary, not an apportioned cost.
2. **Part B bytes model is analytic**, not a byte counters read: the dedup factor (0.59-1.00)
   and the exact index-source count (I use 20) are ranges; the head, engram and gather
   scratch are omitted (each ≤0.1 GB). The verdict is robust across the range (5.0-7.6×).
3. **The dense slice time (38.5 ms) is microbench-extrapolated** from a faithful but
   synthetic laptop-M4-Max trellis, corroborated in-situ by phase-17's full-model dense
   ablation. The routed slice (~23 ms) is in-situ-**inferred** from the 8-layer R-sweep, not
   isolated at the full model. A full-model macro-op profile would settle both.
4. **The 450 GB/s figure is a triad measurement**, from the loop-2 node calibration; the
   verify path is read-dominated, hence the 400-550 GB/s range and the canary (§3) to pin
   the read-only number.
5. **92.4 ms is an upper bound** on the pure verify forward (§0); the true multiple is
   marginally below 6.3×.

*No cluster was touched; no GPU work was done; no relaunch; no API POST. Only repo files were
read (read-only).*
