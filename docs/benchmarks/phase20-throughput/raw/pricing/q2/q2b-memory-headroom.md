# q2b — memory headroom of the SHIPPED dsv41 serve at MAX context (desk-only)

Author: Phase-20 subagent (Q2-gamma-mem). Branch `p20/q2-gamma-mem`. Desk analysis only: **0 boots, 0 generation requests,
0 writes to any node.** Read-only: `GET /state`, VictoriaMetrics GET/export, `ssh` (`ps`/`footprint`/`vmmap`/`vm_stat`/`sysctl`/`cat`/`grep`/`scp` from node),
plus local laptop probes (never on a cluster node). Independent of, and not blocking, the gamma round.

## FINDING (stated first)

**BREACH — conditional.** The shipped configuration admits contexts far past the point where its own peak memory crosses the wired limit
**W = 120000 MiB = 125.829 GB** (`sysctl iogpu.wired_limit_mb` = 120000 on both nodes, re-read live).

| | result |
|---|---|
| Admitted cap (live `/state` `maxKvTokens`) | **1,048,576** tokens (Hermes pin `context_length` 1,048,576; its config `compression.threshold` 0.7 nominally compacts at ~734,003 — application not verified) |
| **Allocator domain** (`mx.get_peak_memory`, the quantity MLX compares with `gc_limit = 0.95·W`) — cold single-request prefill | crosses W at **N\* ≈ 328K** tokens (band **272K–362K** from validation error; fit-only LOO 327K–329K) |
| **Resident domain** (OS `phys_footprint`, adds ~1.8–2.6 GB host memory) — cold prefill | would cross W at **N\* ≈ 266K–286K** tokens (model band 211K–320K) — **UNKNOWN-grade**: one footprint reading plus the gc_limit-plateau assumption, never behaviour-tested on the nodes |
| Warm turns on a resident session (small delta) | **not predicted to cross inside the cap** (123.8 GB at 1,048,576, -2.0 GB vs W) — but the margin is **inside the model error band** (+2.3 GB would reach 126.2), so a breach near the cap is **not excluded**; one extra cap-sized resident session (127.2 GB) does cross it |
| Cold prefill at the cap | extrapolates to **156 GB** (> 137 GB physical RAM) — **UNVERIFIED** (no cold prefill > 350K has ever been measured on a near-shipped build) |
| Largest real prompt seen | 126,527 tokens (`state.db`, 96 Hermes calls to this model) — **below N\* in both domains** |

This is **measured, not only extrapolated — but on older builds**: three independent cold-350K requests (instances d301ea85 / 5689f0ba / 30b396d2, builds of 2026-10-06/07, i.e. *before* next13, same engine path)
peaked at 125.3, 126.3, 126.8 GB in **true units** (W = 125.8; two of three above W). The **next13** soak ladder (build `f0840af1c`/mlx-lm `6cc9c1e`; prod is `99e2966ee`/`689e4ea`, see §7) peaked at
127.7 / 129.1 / 132.7 GB after the 500K / 750K / 1.04M deltas (W+1.9 / +3.3 / +6.9). System RAM-used reached
132.3–133.7 GB on both nodes during those deltas (idle: ~120 GB) and swap rose from 0.1 to **4.1 / 3.8 GB** (m4-1 / m4-2) at the deepest delta.
**No cold prefill > 350K has ever been run on the current (next19) build.**

**What "breach" means (which limit actually fails).** `W` is the kernel's GPU-wired ceiling (`iogpu.wired_limit_mb`), not an allocator limit. In the pinned MLX allocator source (`603f16eb7`) `malloc()` never throws at W: it only trims its
buffer cache at `gc_limit = 0.95·W`, and this fork keeps MLX residency sets disabled (`resident.cpp`; `MLX_RESIDENCY_SETS` is not set on the runner), so exo's `mx.set_wired_limit(...)` call does not itself pin anything (source reading, not behaviour-tested).
Above W the kernel cannot wire more GPU pages; the observed signature is compression/swap — the swap rise above, and studio2's runner now holds **18.7 GiB of its IOAccelerator swapped out** (studio1 1.9 GiB) — though attributing
that swap to W-exceedance (rather than other pressure) is an inference. So this is a **budget breach with an observed pressure signature, not a demonstrated crash**: those loads completed (docs: "compression absorbs, but margin is zero"). Risks: swap/compressor slowdown of the next request, the hang-watchdog false-positive class (SKILL.md "A runner killed mid-long-op"),
and the 124000-MiB Metal-allocator wedge history if anyone raises W without re-validating.

**Two inputs handed to this round are mis-scaled (details below), and one cause of the steep slope is fixable:**
1. The VM gauge `exo_peak_memory_bytes` is **×1.0737 too high** → A2's "124.833 GB allocator peak at ~102K" is really **116.260 GB**.
2. `footprint` prints **binary** units → A2's "114.0 GB resident peak" is **122.126 × 10⁹ B** (113.74 GiB). The two domains swap order once corrected.
3. `Conversation.prefill` retains every chunk's DSpark taps (**30,720 B per prefilled row**, 3.1 GB at 100K rows) until the prefill returns; it is ~73% of the cold-prefill slope. Flagged only (out of scope to fix); a projection with it fixed is in §5.

## 1. Corrected inputs (verified at entry)

| quantity | as handed over (A2 / Q1E) | verified | how |
|---|---|---|---|
| W | 125.829 GB | **125.829 GB** = 120000 MiB = 117.19 GiB (`hw.memsize` 137.44 GB) | `sysctl` on both nodes; runner log `Wired limit set to 117.19 GiB` |
| weights / rank | 104.7 GB | **104.7 GB** (decimal) | `load.py:190` prints `mx.get_active_memory() / 1e9`; both ranks |
| admitted cap | – | **1,048,576** | `GET /state` → `maxKvTokens` (also `kvCacheBits` 0) |
| allocator peak at ~102K | 124.833 GB | **116.260 GB** | gauge is `Memory.from_gb(mx.get_peak_memory()/1e9)` and `from_gb(v)=round(v·1024³)` → ×1.073741824; executed on a known 4.0 GB allocation (`q2b_memory_unit_probe.out.txt`). Independent cross-check: 75 of 104 EXL3 instances' first reading clusters at 118.651 → /1.0737 = 110.50 = the runner's own `warmup: peak=110.5 GB` line |
| resident peak at ~102K | 114.0 GB | **122.126 GB** (113.74 GiB) | `footprint` default output is binary: 4,000,006,400 B allocated → `-f bytes` reports that region as 4,000,956,416 B, which the default output prints as "3816 MB" (= 3815.6 MiB; a decimal reading would be 4001 MB) (`q2b_footprint_unit_probe.out.txt`); runner pid 49330 `-f bytes` peak 122,126,271,536 B = libproc `ri_lifetime_max_phys_footprint` = `vmmap` 113.7G; studio2 runner 122.104 GB |
| non-weight transient | 20.13 alloc / 9.3 resident | **11.56 alloc / 17.43 resident** | peak − 104.7 |
| "peak at the 102,411-token prompt" | – | the peak was set by the **100,497-row cold prefill** (finished 00:40:23); the 102,411-token request was a 2,059-row delta on the resident session | exo.log `prefill controls … rows=100497` / `turn reuse: prompt=102411 prefill=2059`; VM prompt-token counter |

Caveat on the resident number: `ri_lifetime_max_phys_footprint` is a **process-lifetime** max. System RAM-used was already within ~1 GB of its 100K value during 18K–20K-row cold prefills
(126.25 GB at 18K vs 126.55 GB at 100K), so the 122.1 GB is **most likely a plateau set by the allocator's gc_limit trimming** (§2; inference from allocator source + RAM-used series, not behaviour-tested on the nodes), not a measure of the 100K request specifically. It cannot be extrapolated linearly from one point.

**Third, independent confirmation of the gauge correction (physical consistency).** Active allocator bytes must fit inside what the OS says is in use (RAM-used + swap-used). On the next13 soak's deepest rungs (concurrent
15 s samples, both nodes, raw files in this directory; non-runner system RAM ≈ 5.0 GB on studio1):

| rung | gauge as labelled | gauge corrected (÷1.0737) | RAM+swap m4-1 | RAM+swap m4-2 | (RAM+swap) − corrected | (RAM+swap) − labelled |
|---|---:|---:|---:|---:|---:|---:|
| r500 delta | 137.1 | 127.7 | 133.3 | 132.4 | +5.6 / +4.7 | -3.8 / -4.7 |
| r750 delta | 138.6 | 129.1 | 134.7 | 134.8 | +5.6 / +5.7 | -3.9 / -3.8 |
| r1m delta | 142.5 | 132.7 | 136.9 | 135.2 | +4.2 / +2.5 | -5.6 / -7.2 |

With the corrected gauge the residual is **+2.5 … +5.7 GB**: the same order as the ≈6.8 GB of non-allocator memory seen at idle
(runner host 1.8 GB + other processes ≈5.0 GB); it sits somewhat under that because 15 s samples can miss the peak. With the labelled gauge the allocator's active bytes would **exceed** everything the OS reports in use at every rung
(negative residual in the last column) — impossible. (A consistency check, not proof: macmon's `ram_usage` definition is not independently verified.)

## 2. How the peak is built (shipped code, measured on the real classes)

Allocator peak = `mx.get_peak_memory()` = high-water of ACTIVE bytes (cache pool is a separate counter; A2 micro-test). Components of a cold prefill of `N` rows:

* **weights** 104.7 GB (+ DSpark draft head, attached after the 104.7 reading; UNMEASURED, ≈3.6 GB/rank if TP-sharded — checkpoint `mtp.*` = 7.243 GB; warmup peak 110.5 GB bounds it ≤ 5.8) → intercept **A = 112.09 GB** includes these and the base transient.
* **KV (resident session)** = `comp_kv` 2,560 B + `index_k` 640 B = **3,200 B per capacity-row** (real `ModelCache`, `q2b_kv_per_token_probe.out.txt`); capacity grows `min(max(req, 2·cap), 1,048,576)` from 65,536 (`cache.py:188`) → 3.36 GB at the cap per session (`max_sessions=2`). `win_kv` 5.2 MB fixed. **RoPE tables: none** (per-call since 2026-10-04: `attention.py:127`, `mtp.py:119`).
* **retained DSpark taps** = 3 layers × 5,120 × 2 B (bf16) = **30,720 B per PREFILLED row** (stub-model probe on the real `engine_prefill`: `sum(nbytes) = rows·30,720` exactly, `q2b_taps_retention_probe.out.full.txt`). The free fit below recovers **28,489 ± 894 B/row** (fp32 taps would be 61,440).
* **unattributed O(N)** = **7,922 B/token** (fitted). Candidates, UNVERIFIED: per-chunk offset-scaling transients (indexer block-maxima `step·offset·1 B` = 2,048 B/token at the shipped chunk 2,048; compressor `kv_all` copies; sparse-attn gathers).
* **MLX allocator** (`mlx/backend/metal/allocator.cpp`, the exact submodule `603f16eb7` the nodes run): `gc_limit = 0.95·W = 119.54 GB`; on a cache miss `if active+cache+size ≥ gc_limit → release cached buffers`; **no hard throw at W**. So while active < gc_limit the process's GPU total sits pinned near gc_limit (cache fills the gap) and OS footprint ≈ gc_limit + host ≈ 122.1 GB; only when **active** passes gc_limit (cold N ≈ 177,880) does footprint track active (+ host 1.76–2.59 GB).

Model: `peak(N) = A + t·N + 3200·cap(N) + 30,720·D`, `D` = rows prefilled by the request (cold: D = N; warm turn: D ≈ 2K).
A and t are **fitted**; KV and taps are **analytic**. Fit set = every request whose finish raised the (monotone, process-lifetime) gauge, in true units, with the VM prompt-token delta checked equal to `N` and, for the current boot, the exo.log `rows=` line checked equal to `D`:

| id | N | D | capacity | measured (true GB) | model | resid | note |
|---|---:|---:|---:|---:|---:|---:|---:|
| cur_20K_a | 20,076 | 20,076 | 65,536 | 113.145 | 113.080 | +0.064 | current boot (next19), cold |
| cur_20K_b | 20,076 | 20,076 | 65,536 | 113.181 | 113.080 | +0.101 | current boot, cold (3rd 20K) |
| cur_18K | 18,387 | 18,387 | 65,536 | 113.263 | 113.015 | +0.248 | current boot, cold |
| cur_45K | 45,702 | 45,702 | 65,536 | 113.848 | 114.071 | -0.223 | current boot, cold |
| cur_100K | 100,497 | 100,497 | 131,072 | 116.260 | 116.398 | -0.138 | current boot, cold = THE request behind A2's 'peak at ~102K' |
| n13_100K | 100,012 | 100,012 | 131,072 | 116.414 | 116.379 | +0.035 | soak13 (next13), cold |
| n13_160K | 160,006 | 160,006 | 160,006 | 118.867 | 118.790 | +0.077 | soak13, cold (r160) |
| n13_r500 | 499,992 | 340,248 | 499,992 | 127.694 | 128.108 | -0.414 | soak13 delta 340,248 on 159,744 reused (r500) |
| n13_r750 | 749,994 | 250,282 | 999,984 | 129.103 | 128.925 | +0.178 | soak13 delta 250,282 on 499,712 reused (r750) |
| n13_r1m | 1,039,974 | 290,406 | 1,048,576 | 132.681 | 132.610 | +0.071 | soak13 delta 290,406 on 749,568 reused (r1m) |

Fit: **A = 112.095 GB, t = 7,922 B/token, σ = 0.212 GB (n = 10).** The 3 delta rungs (D ≠ N) make `D` and `N` separately identifiable: free fit `cD = 28,489 ± 894 B/row` (analytic 30,720), `cN = 8,559 ± 297 B/token` (excluding KV). Standard errors are optimistic (n = 10, pooled builds); builds differ ≤ 0.15 GB at the matched ~100K point (next19 116.26 vs next13 116.41).

### Unfitted validation (cold, VM hit-kind `none`) — this sets the model-error band

| request | N | measured | model | meas − model | build |
|---|---:|---:|---:|---:|---:|
| n18 b8d90d57 188K | 188,261 | 119.021 | 119.972 | -0.951 | next18 (shipped lineage), cold |
| n13 75fa21d8 160K | 160,007 | 121.126 | 118.790 | +2.336 | next13-era other instance, cold |
| n17 49984332 91K | 91,043 | 117.588 | 116.032 | +1.555 | next17 (lever-1 era), cold |
| old 247fb3c8 160K | 159,995 | 119.852 | 118.789 | +1.063 | 10-06 build, cold |
| old d301ea85 350K | 350,124 | 125.343 | 126.745 | -1.402 | 10-06 build, cold |
| old 5689f0ba 350K | 350,124 | 126.343 | 126.745 | -0.401 | 10-06 build, cold |
| old 30b396d2 350K | 350,124 | 126.788 | 126.745 | +0.044 | 10-07 build, cold |

Error band **[-1.40, +2.34] GB** (measured higher = earlier crossing). Old pre-bf16-row/pre-M2 builds (upper envelope only, not used): OLD 867ef937 cold 500K 142.9 GB; OLD 867ef937 cold 750K 152.1 GB.

## 3. N\* (tokens) — both domains

| quantity | tokens | notes |
|---|---:|---:|
| allocator, cold prefill, central (fit) | 328,242 | slope 41,842 B/token = **4.184e-05 GB/token** |
| allocator, cold, band from validation error | 272,413 – 361,749 | e = +2.34 / -1.40 GB |
| allocator, cold, leave-one-out | 326,975 – 328,968 | fit stability only |
| allocator, cold, FREE fit (cD, cN fitted separately; taps coefficient 28.5 vs analytic 30.7 KB/row) | 338,141 | sensitivity to the ~2.4σ taps-coefficient gap (something is mildly unmodeled) |
| allocator, cold, IF capacity were rounded up 2× (worst case) | 304,922 | shipped cold path uses ONE up-front ensure_capacity ⇒ cap(N)=N for N ≥ 131,072 (verified in q2b_kv_per_token_probe.out.txt); at N* the cap is ≈ N* itself |
| allocator, cold, empirical two-anchor (100K anchor → each measured cold-350K) | 363,486 / 337,395 / 327,380 | no model: straight lines through measured points |
| resident (OS footprint), cold | 266,376 – 286,130 | active + host 1.76…2.59 GB; model band 210,546–319,637 |
| active passes gc_limit (119.5 GB) — end of the footprint plateau | 177,880 | cold |
| active passes physical RAM (137.4 GB) | 605,710 | cold (needs OS swap) |
| warm turn D=2,048 / D=25,662, allocator | beyond cap / beyond cap | model crossing 1,302,189 / 1,210,618 — outside the admitted window |
| naive 'as asked': A2 numbers (mis-scaled) + KV-only slope, allocator / resident | 413,699 / 3,799,011 | right order for the wrong reasons (resident 3.8M is wrong; KV-only slope is wrong) |

Slopes: cold prefill = 30,720 (taps) + 3,200 (KV) + 7,922 (fitted O(N)) = **41,842 B/token**; warm turn = 3,200 + 7,922 = **11,122 B/token** (taps scale with the rows PREFILLED, not the context). The KV-only slope (3,200 B/token) alone would put the crossing at 3,090,913 tokens — the measured deep points rule that out.

## 4. peak(N) curves (allocator domain; true GB; "vs W" = peak − 125.83)

### 4a. Cold prefill (a single request with no resident/parked prefix) — AS SHIPPED
| context N (tokens) | allocator peak (GB) | vs W | OS footprint (GB) |
|---|---:|---:|---:|
| 20,076 | 113.08 | -12.75 | 122.1-122.1 |
| 100,497 | 116.40 | -9.43 | 122.1-122.1 |
| 131,072 | 117.58 | -8.25 | 122.1-122.1 |
| 200,000 | 120.46 | -5.37 | 122.2-123.1 |
| 262,144 | 123.06 | -2.77 | 124.8-125.7 |
| 300,000 | 124.65 | -1.18 | 126.4-127.2 |
| 328,000 | 125.82 | -0.01 | 127.6-128.4 |
| 400,000 | 128.83 | +3.00 | 130.6-131.4 |
| 500,000 | 133.02 | +7.19 | 134.8-135.6 |
| 734,003 | 142.81 | +16.98 | 144.6-145.4 |
| 900,000 | 149.75 | +23.92 | 151.5-152.3 |
| 1,048,576 | 155.97 | +30.14 | 157.7-158.6 |

### 4b. Warm turn (≈2K new rows on a resident conversation; exact capacity)
| context N (tokens) | allocator peak (GB) | vs W | OS footprint (GB) |
|---|---:|---:|---:|
| 20,076 | 112.53 | -13.30 | 122.1-122.1 |
| 100,497 | 113.37 | -12.46 | 122.1-122.1 |
| 131,072 | 113.62 | -12.21 | 122.1-122.1 |
| 200,000 | 114.38 | -11.45 | 122.1-122.1 |
| 262,144 | 115.07 | -10.76 | 122.1-122.1 |
| 300,000 | 115.49 | -10.33 | 122.1-122.1 |
| 328,000 | 115.81 | -10.02 | 122.1-122.1 |
| 400,000 | 116.61 | -9.22 | 122.1-122.1 |
| 500,000 | 117.72 | -8.11 | 122.1-122.1 |
| 734,003 | 120.32 | -5.51 | 122.1-122.9 |
| 900,000 | 122.17 | -3.66 | 123.9-124.8 |
| 1,048,576 | 123.82 | -2.01 | 125.6-126.4 |

### 4c. Warm turn + a second resident session (`max_sessions=2`) — sensitivity (analytic)
Warm turn at the 1,048,576-row cap PLUS a second resident session: second session at 400K rows → 125.10 GB (-0.73 vs W); second session also at the cap → **127.18 GB (+1.35 vs W)**. (The engine keeps `max_sessions=2`; whether Hermes ever holds two deep conversations is UNKNOWN.)

### 4d. Warm turn with capacity rounded UP to 2× the context (upper bound; a session grown turn-by-turn has capacity 65,536·2^k, i.e. ≤ 2×)
| context N (tokens) | allocator peak (GB) | vs W | OS footprint (GB) |
|---|---:|---:|---:|
| 20,076 | 112.53 | -13.30 | 122.1-122.1 |
| 100,497 | 113.60 | -12.23 | 122.1-122.1 |
| 131,072 | 114.03 | -11.79 | 122.1-122.1 |
| 200,000 | 115.02 | -10.81 | 122.1-122.1 |
| 262,144 | 115.91 | -9.92 | 122.1-122.1 |
| 300,000 | 116.45 | -9.37 | 122.1-122.1 |
| 328,000 | 116.86 | -8.97 | 122.1-122.1 |
| 400,000 | 117.89 | -7.94 | 122.1-122.1 |
| 500,000 | 119.32 | -6.51 | 122.1-122.1 |
| 734,003 | 121.33 | -4.50 | 123.1-123.9 |
| 900,000 | 122.64 | -3.19 | 124.4-125.2 |
| 1,048,576 | 123.82 | -2.01 | 125.6-126.4 |

### 4e. Largest single warm turn (new rows) that keeps the allocator peak ≤ W (model central; error band ±1–2 GB ≈ ±35–70K rows)
| context N | max new rows (exact capacity) | max new rows (2× capacity) |
|---|---:|---:|
| 200,000 | 200,000 | 200,000 |
| 328,000 | 328,000 | 294,161 |
| 400,000 | 302,261 | 260,594 |
| 500,000 | 266,057 | 213,973 |
| 734,003 | 181,337 | 148,569 |
| 900,000 | 121,239 | 105,763 |
| 1,048,576 | 67,448 | 67,448 |

## 5. Adjacent findings (flagged, NOT fixed — outside this task's scope)

1. **Retained DSpark taps (memory bug in the shipped prefill path).** `Conversation.prefill` (`dsv41/session.py:644`) passes a local list as `taps_out=` into `SessionCache.append_turn` → `engine_prefill`, which appends *every* chunk's taps although `taps_cb` (`_on_chunk_taps`) already consumed them; the list lives until `prefill()` returns. 30,720 B per prefilled row = 3.1 GB at 100K, 10.5 GB for the r500 delta, 32.2 GB for a cold 1M. Steady-state memory is unaffected (peak only). **Projection, untested:** without it the cold curve collapses onto the warm curve (cold N\* beyond the cap; 123.8 GB at the cap). Suggested minimal change (UNTESTED; needs the dsv41 session/draft-lockstep tests and a live A/B): do not pass `taps_out` when `taps_cb` is installed (the list is only the fallback for `_feed_taps`, `session.py:677`).
2. **Gauge unit bug.** `exo_peak_memory_bytes` for dsv41 is ×1.0737 high (`engine.py:284`, `rounds.py:227`; the batched generator uses `/1024**3` correctly). Also it is a **process-lifetime** high-water (no `reset_peak_memory` in the dsv41 path), not "peak of the most recent request". Any alert or doc comparing it to W is wrong by 7.4%.
3. **`footprint` units.** Q1E-A1 states `footprint` "reports decimal GB (verified)". It is binary; the "verification" matched IOAccelerator 105 *GiB* (= 112.6 × 10⁹ B, which must also hold the draft head, resident KV and the MLX cache pool) against weights 104.7 × 10⁹ B. Prior docs quoting "113 GB peaks" (PERFORMANCE_HISTORY soak entries) are, **if** they came from `footprint` default output, GiB (113 GiB = 121.3 × 10⁹ B) — their source tool is not identified (`bench/soak_next13.sh` has no footprint sampling), so see the UNRECONCILED item in §8.
4. **studio2 anomaly (cause UNKNOWN).** 00:57 CDT, 16 min after the last request, studio2 RAM-used jumped 120.4 → 131.3–132.7 GB and swap 0.08 → 19.5 GB (then settled at 9.65 GB used; no request in flight). The studio2 runner now shows **18.7 G of IOAccelerator swapped out** (studio1: 1.9 G) — the next studio2 request pays page-in latency. Raw: `q2b_raw_node_studio*_vmmap_footprint.txt`, `q2b_raw_vm_sys_series.json`.

## 6. Q1E consequence (CLOSE verdict unchanged, but the margins were mis-stated)

Rule `B_arm + rest(6.51) + transient + 2 ≤ W` for q3g128 (B_arm 110.42 → headroom 15.41 GB):

| reading | non-weight | rest+non-weight | headroom | over by | over by (+2 GB margin) |
|---|---:|---:|---:|---:|---:|
| allocator corrected | 11.56 | 18.07 | 15.41 | +2.66 | +4.66 |
| resident corrected | 17.43 | 23.94 | 15.41 | +8.53 | +10.53 |
| allocator A2 reported | 20.13 | 26.64 | 15.41 | +11.23 | +13.23 |
| resident A2 reported | 9.3 | 15.81 | 15.41 | +0.40 | +2.40 |

Both corrected readings still FAIL, so CLOSE stands. The "razor-thin 0.4 GB on the resident reading" was an artifact of mixing a binary `footprint` number with a decimal weight; the favourable reading is now the *allocator* one (+2.66). If retained taps were fixed the allocator reading would be -0.43 GB before the pre-registered 2 GB margin and +1.57 GB with it (still FAILS) — a new hypothesis for a separate pre-registered round (Q1E reopen condition (b)), not a change to this verdict.

## 7. Assumptions

* Gauge = rank-0 runner's `mx.get_peak_memory()` (only rank 0 emits stats; `runner.py:875`); footprint read on rank 1; both ranks' lifetime max agree to 0.02 GB.
* Builds pooled: the fit mixes next19 (current boot) and next13 (soak13). `git diff 6cc9c1e 689e4ea` (mlx-lm) touches indexer small-n fp32 row (n ≤ 16), sparse-attention gates, `exl3_build.py` dense policy; exo `dsv41/{session,rounds,park}.py` identical, `engine.py` +56 lines. Matched-context check: ≤ 0.15 GB.
* The gauge is a process-lifetime high-water that never resets in the dsv41 path (no `reset_peak_memory`). Checked on the raw series: 0 downward steps in the two fit instances (7aa2dbd3, 38ed8ddb); 8 of 47,161 consecutive pairs fleet-wide step down — 3 are 0.00 GB float noise, 4 are on 10-01…10-03/10-09 instances outside the fit, and the 44.7 GB drop (instance 867ef937, 2026-10-05 05:34, 163.3 → 118.6 labelled = back to the warmup level) is consistent with a runner restart under the same instance id after the r1m watchdog kill (inference). So an up-step identifies the request that set the process maximum.
* `ensure_capacity` is called once up front with `offset + delta` (`session_cache.py:617`); the r750 rung therefore holds 2× capacity (999,984).
* Taps are bf16 (`EXO_COMPUTE_DTYPE=bf16`; the free fit gives 28,489 ± 894 B/row vs 30,720 analytic — fp32 would be 61,440, excluded; the ~2.4σ shortfall is unmodeled, hence the FREE-fit row in §3).
* Only ONE long conversation resident (other session small, 65,536-row capacity ≈ 0.21 GB — inside the residual).
* Extrapolating the cold model from D ≤ 340K to D = 1,048,576 assumes linearity; the deltas bracket the cold-N\* region (500K-rung predicted from 160K-cold within 0.4 GB), but nothing deeper-cold than 350K exists on a current build.
* "Breach" = peak allocator active memory (or OS footprint) above W. The allocator does not throw there; consequences are OS compression/swap, GPU stalls, watchdog false positives.

## 8. UNKNOWN / not verified

* Behaviour of a **cold prefill > 350K on the shipped build** (never run); the 156 GB at the cap is an extrapolation. A live test is the PM's call (needs boots/cluster time — not done here).
* Draft-head weight size per rank (inside A); mechanism of the fitted 7,922 B/token O(N) term (candidates only).
* The allocator-plateau model was **not behaviour-tested on the nodes** (read-only); it rests on the allocator source, the system RAM-used series (plateau 125.8–126.6 GB at every cold prefill 18K–160K; 133.1–133.7 GB on the deep deltas) and one footprint reading. Resident N\* is the weakest number here.
* Resident-domain N\* (and the ~122.1 GB plateau) is **UNKNOWN-grade**: it chains one `footprint` reading with the allocator-trim reading of the source.
* Warm-turn margin at the cap (2.0 GB) is smaller than the unfitted-validation error (-1.4..+2.3 GB): "no crossing" is a model prediction, not a guarantee.
* The exo swap gauge peaked at 19.5 GB on studio2 at 00:57 CDT while `vm.swapusage` reports a 10,240 MiB swap file now — the gauge and the OS number are not reconciled.
* Whether memory pressure at W+ would hang/kill a runner on the *current* build (soaks completed on older builds; docs record a watchdog false-positive kill and a 124000-MiB wedge).
* Cause of the studio2 swap/RAM event at 00:57 CDT.
* `state.db` covers only 96 calls through 2026-10-07; real traffic since then is unknown.
* **UNRECONCILED:** PERFORMANCE_HISTORY's soak13 entry says "Memory stayed 113 GB both nodes through the deep delta (well under the wired limit)" while the VM gauge (true units) says the allocator peaked at 132.7 GB and system RAM-used+swap says ~137 GB on those rungs. The doc's source tool/sampling is not identified; if it was a `footprint` snapshot it is a steady-state reading, not a peak. This report relies on the gauge and the system series, which agree with each other (§1 table), not on that sentence.

## 9. Raw readings and reproduction

All numbers above come from files in this directory; `python3 q2b_headroom_model.py` regenerates this document and `q2b-memory-headroom.json` from them (stdlib, no network).

| file | content |
|---|---|
| `q2b_raw_vm_instances_dsv41.json` | VM `exo_peak_memory_bytes` for 104 EXL3 instances: every up-step + prompt-token & hit-kind deltas (from `q2b_vm_history.py`) |
| `q2b_raw_vm_current_peak.json`, `q2b_raw_vm_sys_series.json`, `q2b_raw_vm_ram_soak13_window.json` | raw 15 s samples: current-instance gauge; system RAM-used/swap (both nodes) |
| `q2b_raw_node_studio{1,2}_vmmap_footprint.txt` | `sysctl`, `vmmap -summary`, `footprint -f bytes`, `vm_stat` on both runners (2026-10-10 04:27 CDT) |
| `q2b_raw_studio1_exo_log_excerpts.txt` | studio1 `exo.log` (current boot): load, warmup, prefill/turn-reuse/park lines |
| `q2b_raw_state_capacity.json`, `q2b_raw_checkpoint_config.json` | live `/state` capacity fields; checkpoint `config.json` |
| `q2b_memory_unit_probe.py/.out.txt`, `q2b_footprint_unit_probe.py/.out.txt`, `q2b_footprint_calibration.py/.out.txt` | unit proofs (laptop, known allocations) |
| `q2b_kv_per_token_probe.py/.out.txt`, `q2b_taps_retention_probe.py/.out.full.txt` | KV bytes/token + capacity rule; taps retention (real `ModelCache`/`engine_prefill`, stub model) |
| `q2b_raw_node_wired_swapped.txt`, `q2b_capture_node_readonly.sh` | `footprint --wired --swapped`/`--sysFootprint`/`vmmap`/`vm_stat` on both runners (06:57 CDT) and the read-only script that makes them |
| `q2b_vm_history.py`, `q2b_fetch_sys_series.py`, `q2b_headroom_model.py/.out.txt` | VM extraction; the model |
