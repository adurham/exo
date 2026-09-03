# PERFORMANCE CAMPAIGN 2 — IDEAS (brainstorm, 2026-09-03)

Premise correction: campaign 1 closed every SPECIFIC hypothesis it tested. It never established
a ceiling. The campaign's own byte accounting (P02D/P11) left decode at **1.75-2.16x below the
memory-bandwidth ceiling**, and that gap was noted, never attacked. Prefill is compute-bound at
~94% GPU busy but achieved-TFLOPS vs peak was never measured. This document enumerates every
lever across exo / mlx-lm / mlx for prefill and decode, with evidence, expected value, cost, and
PM instructions.

## Measurement constraint that shapes EVERY experiment here

Between-boot decode variance is ~6 tok/s (P13-P15: identical config, 36.08 vs 30.06). A single-
boot A/B cannot resolve deltas under ~6 tok/s. P16 (characterizing it) is PARKED by the user.
Therefore, in priority order:
1. Prefer levers measurable WITHOUT relaunch: microbenchmarks, within-boot ratios (share of
   cycle), runtime-toggleable knobs, code reads.
2. For relaunch-required A/Bs: ABAB alternation, >=2 boots per arm, report RANGES not means,
   and use prefill t/s as the boot-stability control (prefill replicates to 0.02%).
3. Within-boot phase RATIOS from the [MTP-PROF] profiler are meaningful even when absolute cycle
   time drifts — use them for attribution.
4. Any bracket-timing MUST force mx.eval() at bracket close (campaign-1 lesson: lazy-eval
   spillover fabricated a 5% "gap").

---

## TRANCHE 1 — measure the big unknowns (mostly zero/low cluster cost)

### I1. TP all_sum latency INSIDE the verify forward (decode) — LARGEST SUSPECTED LEVER

**Evidence.** start_cluster.sh fence comments: "43 layers x 2 all_sums per layer" = 86 cross-rank
collectives per forward. These sit INSIDE the verify bracket (56ms = 81% of cycle) so every
campaign-1 attribution measured AROUND them. The coord collective (1024 x int32) measured
~370us/call. Decode activations at M=4 are ~57KB per all_sum -> latency-bound, not bandwidth-
bound. 86 x 300us = ~26ms = ~46% of verify. Corroborating structural evidence: PP-decode (1 hop
per token) measured 27-33 t/s vs TP-decode (86 collectives) ~34 t/s. TP should ~halve per-token
time vs PP if collectives were free; TP barely wins, so TP's overhead is eating nearly its whole
parallelism advantage.

**Hypothesis.** Per-layer all_sum latency is 30-50% of verify time.

**PM instructions.**
1. Read `mlx-lm` deepseek_v4.py + the TP sharding layer to find EXACTLY where per-layer
   collectives fire (all_sum after attention, after MoE; any all_gather). Count them per forward
   at decode. Confirm the "2 per layer" claim from code, not comments.
2. Instrument with mx.eval-correct brackets (env-gated, EXO_DSV4_MTP_PROFILE convention, force
   `mx.eval` before end timestamp): per-forward sum of collective wall, count, mean/min/max
   per call. Also bracket the coord collectives separately (already known ~0.65ms).
3. Standalone jaccl microbench: all_sum latency vs payload (1KB, 57KB, 1MB) between the two
   nodes, N=1000 each. Compare to Thunderbolt 5 RDMA theoretical small-message latency
   (~10-20us). Report the multiple.
4. One profiled cluster session at 89K (S4-style), n>=3 reps.
5. **Pre-registered bands (share of verify):** >=30% -> FUND collective reduction (I1b);
   10-30% -> fund jaccl latency work (I1c); <10% -> CLOSED, the byte-accounting gap lives
   elsewhere (go to I3).

**Follow-ons if funded.**
- I1b (structural): DSv4 has 1 KV head + compressed sparse attention — attention is tiny at
  decode. If attention is currently head-sharded, REPLICATE attention on both ranks and shard
  only the MoE FFN -> eliminates the post-attention all_sum (43 fewer collectives). Check
  whether the NOT-FUNDED "further head-sharding beyond FFN" item means attention is ALREADY
  replicated; if so, I1b is moot and the count is 43 not 86 — measure, don't assume.
- I1c (jaccl): if per-call latency is >>20us, the jaccl implementation (polling, completion
  handling, fence granularity, UC FIFO) is the lever. User owns the mlx fork.
- I1d: overlap — issue the all_sum async and overlap with the residual/norm of the same layer
  or the router computation of the MoE (which depends on the summed activation — check the
  dependency graph; maybe only partial overlap is possible).

**Cost:** 1 relaunch session (~1h) + microbench (zero). **EV:** if 30-50% of verify, this is a
potential 1.3-1.8x decode lever.

### I2. The "c=2 tax" — settings that cost c=1 throughput and exist ONLY for concurrency the user has DROPPED

**Evidence (all from start_cluster.sh comments, each with a documented c=1 cost):**
- `EXO_DSV4_FENCE_EVERY_N_LAYERS=4`: "SELECTED for c=2 ... Cost: ~0.7 t/s on c=1 ... Set to 8
  to recover c=1 ceiling at the cost of c=2 bistability."
- `MLX_STEEL_BATCH_INVARIANT=1`: "costs ~5% c=1 decode ... required before re-enabling spec at
  c>=2". Interaction: "Set =0 only with EXO_DSV4_VERIFY_ROWSEQ_VEC=0" — read the code to see if
  that constraint still holds.
- `EXO_DSV4_FUSED_MOE=0 / COMPILE_FFN=0 / COMPILE_LAYER=0`: "DISABLED 2026-06-18 ... All three
  batch-mis-specialize at batch_size>1 ... Combined perf they buy is only ~3-4%" — disabled
  SOLELY for BS>1 correctness. At c=1 they are free.
- `EXO_BATCHED_PREFILL_RENDEZVOUS_MS=200`: "Adds the same delay to c=1 first-token." 200ms of
  pure TTFT on every request, buying nothing at c=1.
- `MLX_GEMV_BATCH_INVARIANT=1`, `EXO_DSV4_SPEC_CACHE_ROLLBACK_C2`, `EXO_DSV4_MTP_C2_MAX_CTX`,
  `EXO_MAX_ACTIVE_TASKS=5` — audit each for a c=1 cost.
User decision on record (2026-09-01): V4 c=2 concurrency DROPPED, "workload shape doesn't fit
c=2+". The production config still pays for it.

**Hypothesis.** Reverting the c=2 tax is worth ~+8-10% decode and -200ms TTFT with zero
algorithmic risk at c=1.

**PM instructions.**
1. Enumerate EVERY knob in start_cluster.sh whose comment justifies a c=1 cost for c>=2
   stability/correctness. Table: knob, current, c=1-optimal, documented c=1 cost, interaction
   constraints, whether it is read at process start (relaunch needed) or runtime.
2. Rendezvous 200ms is DETERMINISTIC (no boot-variance issue): measure TTFT delta directly.
3. For decode knobs: the boot-variance problem is real. Design: ABAB with >=2 boots per arm,
   OR (better) use within-boot cycle-time RATIOS from [MTP-PROF] where the knob changes a
   specific phase (fence changes the collective/sync pattern; compile/fused change kernel
   count). Report ranges.
4. QUALITY GATE is mandatory for FUSED_MOE/COMPILE_*: the original disable was for BS>1
   corruption; at c=1 you must still prove correctness: `bench/ab_probe_tier1.py` fixed-prompt
   battery (7/7), needle-in-haystack, and a temp=0 byte-identity check vs the current config
   on 3 prompts. Any divergence -> that knob stays off (record why).
5. Also set `EXO_MAX_CONCURRENT_REQUESTS=1`-style guards if the config assumes c=1 — but do
   NOT remove concurrency SUPPORT from code; this is config only, reversible.

**Cost:** ~2-3 relaunch sessions for ABAB. **EV:** +8-10% decode, -200ms TTFT, near-zero risk.

### I3. Where does the 1.75-2.16x go? Achieved-bandwidth microbench of the actual kernels

**Evidence.** Decode is bandwidth-bound. Verify reads the activated expert weights (13B params
at ~6-bit ≈ 10GB/token-set, ~5GB/rank at TP=2) once per verify regardless of M. At M4 Max
~546 GB/s that is ~9ms of pure weight streaming; measured verify is 56ms. Either the kernels
achieve a fraction of peak bandwidth, or the time is not in weight streaming at all (-> I1).

**PM instructions.**
1. Read model config: expert count, top-k, hidden/intermediate dims, quantization bits +
   group_size, shared_experts shape, attention projection shapes, lm_head shape.
2. Microbench `mx.quantized_matmul` (and the exact kernel path the MoE uses — `gather_qmm` /
   switch_mlp) at those shapes: M=1, M=4 (verify), bits as deployed, N=200 iters, on ONE Studio
   while the cluster is idle (note contention; or run on the other node's GPU — the cluster
   process holds memory but an idle GPU can run a microbench; verify with gpu_usage_ratio).
   Compute achieved GB/s = bytes_read / time. Report % of 546 GB/s.
3. Same for prefill: M=2048 chunk, report achieved TFLOPS vs M4 Max peak (~17-19 TFLOPS fp16;
   quantized kernels dequant on the fly, so state the assumption).
4. Sum the per-layer kernel times at M=4 x 43 layers and compare to the measured 56ms verify.
   The DIFFERENCE is dispatch/sync/collective overhead — cross-check against I1.
5. **Pre-registered:** achieved-bandwidth >=80% of peak -> kernels are fine, gap is overhead
   (I1 owns it). <60% -> mlx kernel work is funded (user owns mlx): 6-bit qmv tile/threadgroup
   tuning for M4 Max, or a fused dequant-gemv path. 60-80% -> report, decide with the I1 result.

**Cost:** zero relaunches. **EV:** decides where the 2x lives.

### I4. Fix B (decode-KV retention) was killed on an INVALID test — re-open

**Evidence.** The round-4 serialization audit used **381-token prompts**. The prefix cache is
CHUNK-GRANULAR: start_cluster.sh says smaller chunks "produce more chunk-boundary cache
snapshots, which is what the prefix-cache uses to serve mid-prompt partial-prefix hits", and
PREFILL_STEP_SIZE=2048. A 381-token prompt has ZERO chunk boundaries, so a divergence at token
378 structurally cannot partial-hit — cached=0 is the expected result at that size, not
evidence of "near-exact-match keying". The real session showed 37/40 PARTIAL hits at ~145K.
Round 3's live proof (353 tokens, cached 351) went through the exact-match path, which needs no
boundaries. The audit conflated the two paths. Fix B's value: 21.9% of uncached tokens in the
real session were the model's own completions being re-prefilled (~22% TTFT reduction on long
sessions). The serialization contract shipped in campaign 1 makes the client deterministic, so
the "any delta zeros the cache" concern is moot for the client we actually run.

**PM instructions.**
1. Read the prefix-cache code (cache.py trie, snapshot points, partial-hit logic) and CONFIRM
   the chunk-boundary requirement for partial hits, with line cites.
2. Re-run the divergence audit at a prompt >= 3 chunks (~7K tokens): baseline, then variants
   diverging at token ~6500 (past 3 boundaries). Expect cached_tokens ≈ 6144 (3 chunks), NOT 0.
   ~6 short requests. If confirmed, the round-4 blocker is INVALID and Fix B re-opens.
3. Fix B design (mlx-lm/exo): at end of decode, do NOT discard the completion's KV — extend the
   trie leaf (or add a snapshot) so the next turn's prompt (= prev prompt + prev completion +
   new msg) matches through the completion. Constraint: bitwise-lossless (the KV IS the decode
   KV; no recompute). Risk: the DSv4 compressed/pooled cache state at a non-chunk-aligned
   position — read how snapshots handle mid-chunk state; the completion end is not chunk-
   aligned. This is the real engineering question.
4. C1 (proxy validation) is moot if the server just reports cached_tokens — use cached_tokens
   as the oracle on a real 3-turn scripted session: turn N+1's cached_tokens must cover turn
   N's completion.
5. **Pre-registered:** partial hits confirmed at >=3 chunks -> Fix B funded. Design lands with
   a 3-turn cached_tokens proof + TTFT delta.

**Cost:** ~6 requests to re-open; implementation is a real mlx-lm change. **EV:** ~22% TTFT
on long sessions (the user's actual workload).

### I5. Speculative acceptance re-tune against the TRUE number

**Evidence.** True acceptance is 1.411/cycle at γ=3 (wall-attribution round, direct dump
deltas). Prior spec tuning (γ sweeps, Eagle K, draft-head selection) used 1.73. Decode =
(1+a)/cycle: 2.41/68.85ms = 35 t/s. At a=2.2, 46 t/s on the same cycle. Verify at M=γ+1 is
bandwidth-bound (weights read once for all rows) so larger γ is nearly free on verify; the
cost is the draft phase (9.19ms for 3 drafts = 3ms/draft — itself suspicious, see I8).

**PM instructions.**
1. Zero-cluster: from the existing profiler dumps + [MTP] counters (untimestamped runner
   stderr — anchor by the method V2 established), compute the acceptance HISTOGRAM (0..γ) at
   γ=3, not just the mean. The histogram tells you whether γ=4/5 would pay: if P(accept all 3)
   is high, extend; if acceptance falls off a cliff after 1, shorten.
2. Cluster: γ ∈ {2,3,4,5} at 89K, n=3 each, measuring true acceptance (counter deltas) and
   cycle time (profiler). Within-boot ratio method: run all four γ in ONE boot if γ is
   runtime-settable (check — EXO_SPECULATIVE_GAMMA is read at start; if so it needs
   relaunches -> ABAB).
3. Also check the draft: which head (DSpark native vs classic MTP), whether Eagle K=8
   "dormant" mixture would raise acceptance at γ>=4.
4. **Pre-registered:** any γ with (1+a)/cycle >= 1.10x the γ=3 value, with quality battery
   clean -> ship. Otherwise closed with the histogram as the record.

**Cost:** 1-2 sessions. **EV:** up to +30% decode if acceptance efficiency rises from 47%.

### I6. Expert weight reads at M=4: once per verify or once per ROW? (zero cost, potentially huge)

**Evidence.** "switch_mlp (the expensive expert-gather)" is batched; "FULLBLOCK_MOE=1 forces
per-row" was rejected for capping throughput. Current default MOE_PARTS_ROWSEQ=shared (only
shared_experts per-row). But the ROUTED experts at M=4: each row routes to top-k experts; the
union across 4 rows can be up to 4k experts. Does gather_qmm read each selected expert's weight
ONCE for the batch, or once per (row, expert) pair? If the latter, verify reads up to 4x the
expert bytes.

**PM instructions.** Read mlx `gather_qmm` / the MoE forward in deepseek_v4.py at M=4. Trace
bytes read. If per-row duplication exists, that is the 2x — design the dedup (group rows by
expert, read once). Report with line cites. Zero cluster cost.

---

## TRANCHE 2 — smaller or user-decision levers

### I7. lm_head vocab-parallel sharding
lm_head is REPLICATED per rank (1.059GB bf16/rank per P05 notes; ~0.53GB at mxfp8). Each rank
reads the whole thing per verify: ~1ms. Vocab-parallel shard halves it; greedy argmax needs only
a local top-1 + one scalar exchange (already have the coord collective). ~1.5%. mlx-lm change.

### I8. Draft phase: 9.19ms for γ=3 drafts is 3ms/draft
The DSpark head is depth-1 and small. 3ms per draft token suggests it is paying a full
attention pass over the KV per draft, or dispatch overhead. Profile the draft sub-phases.
If drafts can be produced in one batched call or the head's attention scoped, draft could drop
to ~3ms total (-6ms/cycle = -9%). Also relevant to I5 (cheaper drafts make higher γ free).

### I9. GPU P-state during bandwidth-bound decode (cheap measurement)
Bandwidth-bound work has low compute utilization; macOS power management may hold the GPU at a
lower P-state, and on Apple Silicon the memory controller/fabric clock is coupled. Measure
`powermetrics --samplers gpu_power` GPU frequency during decode vs prefill vs idle. If decode
runs at a lower clock than prefill, that is a systems lever (sustained-performance mode,
`pmset`, or keeping a compute-side kernel resident). Zero relaunch.

### I10. Fix A — prefix-trie persistence across relaunch
49% of the real session's uncached tokens were one cold start (92.6K uncached, 282s TTFT).
Trie is in-memory only (builder.py:156). Persisting KV to disk (multi-GB per node) and reloading
on launch. Value depends on relaunch frequency in normal use — LOW in steady state, high while
the loop is relaunching. Design only; user decides.

### I11. Weight precision: 5-bit or 4-bit routed experts (USER DECISION — quality tradeoff)
Bandwidth-bound decode scales ~linearly with expert bytes: 6->5 bit ≈ -17% bytes, 6->4 ≈ -33%.
Declined in campaign 1 on quality grounds, not performance. Prepare: (a) quantize a 5-bit
variant offline, (b) run the FULL quality battery (Tier-1 7/7, needle, DSML tool-call
correctness, the P06-era logit-divergence check), (c) one decode measurement. Do NOT ship
without the user reading the quality delta. Also: mixed precision (attention + shared_experts
stay 6-bit, routed experts 4/5-bit).

### I12. Prefill: serial vs batched code-path parity at c=1
Ask A found production prefill runs the SERIAL driver (generate.py:866), never `prefill_batched`
(needs queue_len>=2). Audit that every prefill optimization (tiled SDPA, exact top-k, chunk
handling, clear_cache cadence) applies on the SERIAL path. If the batched path got the
optimizations and serial did not, c=1 prefill is running the unoptimized branch.

### I13. Idle-gap warmup penalty (perceived speed)
P15 warmup rep: 12 t/s vs 30+ steady. Chat sessions have idle gaps; if every turn after a pause
pays a warmup, that is perceived throughput. Measure decode t/s vs idle gap before the request
(0s, 30s, 5min). If real: identify what goes cold (Metal residency, allocator, page-out) — root
cause, not a keep-alive ping.

### I14. Within-boot decode drift (P12: +1.47 t/s recovered by relaunch after deep-context work)
Not P16 (no N-boot study). Within ONE boot: measure decode at 89K, then run 300K of prefill
work, then re-measure at 89K. If decode degrades, correlate with allocator pool size
(mx.metal.get_cache_memory / active memory) and test whether `mx.metal.clear_cache()` between
requests restores it — as a DIAGNOSTIC of the mechanism, not a shipped mitigation.

### I15. Dispatch/sync count per decode step
Count mx.eval / synchronize points and kernel launches per decode cycle on the production path
(not profiler-added ones). 43 layers x (attention + router + experts + norms) — if there are
>500 launches per step, launch overhead (~10-20us each on Metal) is 5-10ms. The disabled
COMPILE_LAYER path (I2) is the existing answer; this measurement quantifies its ceiling.

---

## Explicitly NOT in scope (campaign-1 verdicts stand unless their trigger fires)
Sinkhorn truncation (numerically refuted), shared_experts pad-to-M=8 (qmv batch limit 12),
tree drafting, context caps, the batched-path SDPA 4.06x anomaly (moot at c=1), P16 N-boot
characterization (user-parked), c>=2 concurrency (user-dropped).
