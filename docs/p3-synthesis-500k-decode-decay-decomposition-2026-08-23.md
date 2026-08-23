# P3 synthesis: decomposing the 500K-context decode-throughput decay — 2026-08-23

## Question

Historical anchors showed decode falling 31.1 → 21.51 tok/s (short-ctx →
"500K", ~31%) post-async-fence-fix (T1, 2026-08-22). T5 ruled out
"GPU sits more idle at depth" (occupancy *rises* with depth). P3's job:
decompose the depth cost with real measurements — KV-read volume,
attention/MLA kernel wall time, all_sum depth-dependence — and say how
much of the drop is explained and what residual remains.

Six delegated investigations (workers A, B1, C, C2, C3, D) and two
independent verification passes (R1, R2) ran on 2026-08-23. Every claim
below survived review except where marked; per-worker docs:

- `docs/p3-worker-a-kv-read-inventory-2026-08-23.md` (code-derived byte model, R1-verified)
- `docs/p3-worker-b1-live-depth-anchors-2026-08-23.md` (fresh live anchors, R1-verified)
- `docs/p3-worker-c-attn-kernel-walltime-2026-08-23.md` (kernel wall-time microbench, R2-checked)
- `docs/p3-worker-c2-depth-busy-idle-capture-2026-08-23.md` (busy/idle capture, partial)
- `docs/p3-worker-c3-donation-failure-insitu-2026-08-23.md` (donation test → concat discovery, R2-checked)
- `docs/p3-worker-d-metal-timeout-crash-forensics-2026-08-23.md` (crash forensics)
- `docs/p3-reviewer-r1-verification-2026-08-23.md`, `docs/p3-reviewer-r2-verification-2026-08-23.md`

## 0. First finding: the historical -31% is partly a measurement artifact

B1 found `bench/decode_probe.py` posts `{"bench": true}` to
`/v1/chat/completions`, whose pydantic request model has no `bench`
field — silently dropped, EOS never banned, so historical decode
windows (incl. T1's anchors and T5's flagged ~9s capture) were short.
The EOS ban only engages via `/bench/chat/completions`
(`batch_generate.py:2658`). Fresh clean anchors (2000-token decode
windows, `finish_reason=length`, real `usage.prompt_tokens`, n=1/depth):

| Depth (real tokens) | tok/s | ms/token | decode window |
|---|---|---|---|
| 520 | 29.63 | 33.75 | 67.5 s |
| 100,026 | 27.94 | 35.79 | 71.6 s |
| 352,599 ("500K" nominal) | 23.48 | 42.59 | 85.1 s |

Clean short→deep decay: **-20.8%** (+8.84 ms/tok), not -31%. B1's deep
point is +9.2% above T1's 21.51; likeliest causes are T1's
short-window/unbanned-EOS path plus prompt-content differences (not
separated experimentally). The decay is real; the headline -31%
overstated it. The per-token latency distribution shows a **uniform
rightward shift** with depth (p10 and p50 move together, dispersion
falls, no tail fattening) — whatever the mechanism, it taxes every
token, corroborating T5's "real compute growth, not idle time" on a 9×
longer window.

## 1. KV-read volume at depth (measured from code, R1-verified)

Worker A derived the per-decode-step attention-path read inventory from
the fork's actual code + live config (bf16 unquantized KV,
`EXO_KV_CACHE_BITS=0`; attention **replicated** per rank — exo's TP
shards MoE only, `auto_parallel.py:1032-1034`, so per-rank = per-node):

**bytes_per_rank(L) = 5,297,553,408 + 1930.25·L**

| L | total/step | depth-dependent part |
|---|---|---|
| 100,000 | 5.49 GB | 0.193 GB |
| 352,599 | 5.98 GB | 0.681 GB |
| 500,000 | 6.26 GB | 0.965 GB |

- Core sparse attention reads a **top-k=512 subset**, not all L — O(1)
  in L (confirmed empirically flat by worker C).
- The linear term is 69.6% the **indexer full-pool scan**: L/4 pooled
  entries × 128 dims × bf16 × 21 sparse layers = 1344 B/ctx-token.
- KV-resident sanity check passes (~6.9 KB/token → 2.43 GB at 352.6K).
- Pure byte-flow at measured achieved BW predicts only **+1.19–1.64
  ms/tok** for 100K→352.6K.

## 2. Attention/MLA kernel wall time at depth (measured, production silicon)

Worker C ran the fork's real kernels at exact production shapes/env on
the (idle) rank-0 studio, per-step `mx.async_eval` fencing (production's
pattern):

| L | attention-path ms/step (43 layers) |
|---|---|
| 520 | 12.88 |
| 100,026 | 16.57 |
| 352,599 | 19.13 |
| 500,000 (synthetic) | 21.52 |

- **Scaling: LINEAR above 100K** (fit ms ≈ 15.22 + 1.21 per 100K tokens,
  residuals ±2%; two independent reruns agree). Not superlinear.
- Delta 100K→352.6K: **+2.56 ms/tok** (range +2.56 to +3.34 across
  fencing disciplines).
- Kernels cost ~1.8× the pure byte model — but achieved GB/s **rises**
  with depth (indexer-score 477→558 GB/s, above the 405 GB/s measured
  streaming ceiling via L2 reuse): the depth-growing kernel is already
  at/above bandwidth ceiling — **no optimization headroom there**, and
  "kernels degrade at depth" is refuted.
- Caveats: single-layer-per-class instantiation scaled by census (biases
  the estimate *down* — no inter-layer pipelining), and on the 520→100K
  span the kernel delta (+3.69) **overshoots** the live delta (+2.04) —
  flagged by R2, unaddressed in C's doc — so absolute deltas carry
  real uncertainty; the 100K→352.6K span is the designed, cleanest
  comparison.

## 3. New mechanism found: BatchPoolingCache per-flush concat (harness-measured)

Worker C3 set out to test worker C's donation-failure candidate for the
residual and **refuted it**: the real production decode loop
(`generate.py` — `mx.eval` at :1639, `eager_detach_caches` at :1651, no
cross-step pool reference holder) does *not* defeat buffer donation;
production-faithful config matches donation-maximized config within
0.011 ms/tok at both depths. The "+6.35 ms/tok" upper bound is **not
realized in production**.

But the test found the real adjacent mechanism (R2-confirmed):
production decode never uses `PoolingCache` — `_merge_caches`
(`generate.py:1261`) converts to **`BatchPoolingCache`**, which grows
its pool via `mx.concatenate` to *exactly* max_pool — an unconditional
O(P·D) realloc+copy on **every** pooled flush (every 4th token) —
whereas `PoolingCache` (what C benched) grows in 256-entry chunks
(`cache.py:1899-1903` vs `:1517-1528`; BatchPoolingCache structurally
cannot hold slack — its length *is* its capacity).

Measured in-situ (real production loop, real caches, MoE stubbed,
production silicon): concat cost +0.538 ms/tok @100K → +2.449 @352.6K =
**+1.91 ms/tok depth delta**. Evidence: clean mod-4 periodicity in the
raw per-step series (vanishes when concat is suppressed), per-step
allocator transient 107.1 MB → 10.1 MB (matches the 90.3+22.6 MB pool
sizes exactly), and a donation-defeated+concat-suppressed control that
is **flat in depth** (the depth-scaling pool cost is 100% concat, 0%
donation).

**Additivity verified** (the double-count check): C's 256-step bench
amortized pool growth 1-in-64 vs production's every-flush; had C paid
the production concat its r=4 layers would have measured ~+2.5 ms/tok
higher (R2-corrected arithmetic; C3's doc's "~+10 ms over 43 layers"
was 4× off — disjointness holds either way), which C did not observe.
Kernel and concat costs are disjoint; their sum stays under the live
delta.

## 4. moe.all_sum at depth: payload flat (proven), wait-growth open

- **Payload/shape L-independence: PROVEN in code** (R1-verified). The
  single surviving decode-time collective is
  `deepseek_v4.py:3007` — (1,1,4096) bf16, 43 calls/token; every other
  candidate site is dead in production env (`EXO_DSV4_ATTN_ALLSUM=0`,
  seq-split gates need L_q≥16). No dimension flows from cache length.
- **Wait time at 100K: measured flat and symmetric.** C2's 50s dual-rank
  xctrace during real decode: occupancy 82.98%/83.06% (ranks agree to
  0.08pp) → idle ≈ **6.06–6.09 ms/tok both ranks** — no measurable
  arrival skew at 100K; bounds all_sum wait ≤0.142 ms/call there. Also
  closes T5's 9s-window methodology gap at 100K (occupancy at depth
  confirmed *higher* than short-ctx: ~83% vs 78.6–78.9%).
- **Wait growth at 352.6K: NOT measured.** The deep capture was lost to
  the rank1 crash (below). So: payload-flat is settled; wait-flat at
  depth is plausible (no skew at 100K; idle non-monotone across
  historical windows) but **empirically open**.

## 5. The decomposition

Live depth cost 100K→352.6K = **+6.80 ms/token** (35.79 → 42.59):

| Component | ms/tok | share | evidence class |
|---|---|---|---|
| Attention/indexer kernel wall time | +2.56 (…+3.34) | 38–49% | measured microbench, production silicon |
| BatchPoolingCache per-flush concat | +1.91 | ~28% | in-situ harness w/ real production loop; needs live A/B |
| **Explained** | **+4.47 (…+5.25)** | **~66% (up to ~77%)** | |
| **Residual (unexplained)** | **+1.55 to +2.33** | **~23–34%** | open |

Residual candidates (none tested): all_sum arrival skew at depth,
inter-layer pipelining loss (C's harness structurally biases attention
down — some residual may actually be attention), MoE-at-depth
interplay, ~90 GB-resident allocator regime. The mild end-to-end
superlinearity B1 saw (+2.05 → +2.69 ms/100K, n=1) is *not* reproduced
in the kernels (linear ±2%) and remains unattributed.

**Scaling verdict** (task question 3): KV-read volume and attention
wall time both scale **linearly** with L above 100K; core MLA/sparse
attention itself is **O(1)** (top-k=512); the linear growth lives in
the indexer's L/4 pooled scan + pool maintenance. Neither superlinear
nor flat.

**On the original "31%"**: measured cleanly, short→352.6K is -20.8%.
The remainder of the historical -31% is attributable to the probe bug
(short windows) and prompt differences, not model compute.

## 6. Collateral findings

1. **Probe bug** (§0) — `decode_probe.py`'s EOS ban never worked via
   /v1. Fix via git (repoint at /bench or add the field); re-examine
   prior results that used it (T1's deep anchors, T5's window).
2. **Rank1 Metal GPU timeout** (worker D, full forensics): at 13:51:43
   CDT, ~6.5s after xctrace detach on the ~100K decode, rank1 (m4-1,
   PID 46718) hit `[METAL] GPU Timeout Error` in `mx.async_eval`;
   kernel logged 2 GPURestarts; peer was SIGKILLed by the hang-watchdog.
   Best-supported cause: **tracer finalize memory pressure** (~10 GB
   RunningBoard-exempt buffer on a ~90 GB-resident node → jetsam
   cascade, 26 swapfiles in 18s, compressor 23.6 GB). Thermal ruled out;
   zero similar events in 7 days of logs → primarily a
   **tracing-procedure risk**, with a real production caveat: the node
   runs 90.3 GB resident / 115.3 GB peak of 137 GB at 100K — memory
   headroom is itself a depth-scaling risk worth measuring directly.
   New capture protocol: ≤15s windows, deep point first. **Cluster left
   down** (instances:[]; memory released; supervisors healthy) —
   restore = fresh instance placement, requires operator go-ahead.
3. **Rank-label correction** (D): this run m4-2 = rank0, m4-1 = rank1;
   C2/C/C3 docs carry the swapped label (values unaffected — occupancy
   symmetric to 0.08pp).
4. **Stale doc**: `docs/dsv4-attention-kernel-efficiency-2026-08-18.md`
   :38,52-55 is wrong (assumes head-halving exo never invokes) — R1.

## 7. Next actions (root-cause path)

1. **Live A/B of chunked BatchPoolingCache growth** (C3's spec:
   env-gated `EXO_DSV4_POOL_GROW_STEP`, arm B = 256; expected deep-point
   ~42.59 → ~40.1 ms/tok, +6.1% tok/s, asymmetric-by-depth signature;
   falsification stated). Two R2 corrections MUST be carried: (a) the
   correctness invariant is the **length mask** `pool_idx <
   pool_lengths` (`cache.py:2177-2181`), *not* `_visible_width` (no-op
   on trailing pad — C3's §8.1 rationale is wrong, and
   PERFORMANCE_HISTORY's C3 entry inherits it); (b) padding flips
   `make_mask` None→valid, switching the indexer mask path
   (`deepseek_v4.py:3840/3858/3883`) — arm B changes two things, so
   add a mask-path control or normalize both arms. Requires relaunch
   authorization; goes through git per deploy rules.
2. **Deep busy/idle capture** with the safe protocol to close the
   all_sum wait-at-depth question (the one missing measurement).
3. Probe-bug fix via git; re-baseline historical anchors.
4. Residual hunt (~1.5–2.3 ms/tok) only after (1) and (2) move the
   explained fraction.
