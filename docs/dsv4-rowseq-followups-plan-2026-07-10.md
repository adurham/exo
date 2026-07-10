# DSv4 serving follow-ups plan — 2026-07-10

Continuation plan for the four items deliberately deferred at the end of the
verify-losslessness campaign (see `docs/` handoff lineage and the commit
chain mlx-lm `9a95c84`/`4aefe36`, exo `90b4a7eeb`..`d7d0e1088`).

## Context: what is already true

- **Root fix shipped:** `EXO_DSV4_VERIFY_ROWSEQ` makes the L>1 MTP verify
  attention bitwise-identical to sequential decode (per-row attention with
  per-row cache updates; FFN/MoE batched). Proven bitwise-zero on the ldiff
  harness (`~/scratch/ldiff_seq_vs_batched.py`, quantized 4-layer random
  model) at 4K/32K/131K.
- **Prod config:** classic batched verify below `EXO_DSV4_VERIFY_ROWSEQ_MIN_CTX=32768`
  (clean in months of batteries, 1.6x cheaper), row-seq at and above it.
  Decode @122K: row-seq 24.6 t/s vs 19.1 sequential vs 26.2 uncorrected.
- **Regression gates available:** ldiff harness (bitwise, model level),
  `~/scratch/ldiff_b2.py` (B=2), `bench/dsv4_dsml_battery.py` (serving-level
  tool-call fidelity ladder, passed 4K/65K/122K), quality harness (needle),
  perf probes (4K gen + 122K cached-prefix decode timing).
- **Instrument caveat:** trim(1)+refeed diagnostics (REFCHECK, tie-reverify)
  are unsound at pool-flush cycles — do not use them as judges. Use the
  harnesses above; for serving-vs-serving questions use one-shot
  identical-prefix payloads to avoid trajectory-divergence false positives.

## Recommended sequence: 1 → 2 → 3 → 4

Items 1 and 2 are self-contained and independently landable. Item 3 is best
validated after item 2 lands (byte-equality becomes an acceptance gate).
Item 4 is attribution-first, fix-only-if-cheap.

---

## 1. Placement/reclaim tolerance after kills (smallest, first)

**STATUS 2026-07-10 (later): VALIDATED ON CLUSTER — item 1 CLOSED.**
- Gate run (5 kill/relaunch cycles, incl. kill with 85 GB resident): ZERO
  placement 503s; kill→served 54 s (boot ~15 s + JIT load 38 s); the
  graceful SIGTERM path releases the FULL model footprint in ~1 s
  (vm_stat curve: 85 GB → 2 GB in one sample) — the ~60 s reclaim lag is
  gone at the source, so EXO_JIT_PLACEMENT_WAIT_SECONDS rarely engages and
  stays as the guard for the pathological mode. Default flipped to 120 in
  start_cluster.sh. Reclaim-curve check ran green on every bring-up.
- Note: vm_stat 'Pages wired down' does include the MLX model footprint
  (~86 GB when DSv4 is resident on a node), so the wired+compressor
  residual metric is sound.

**(earlier same day) IMPLEMENTED (laptop):**
- 1a placement wait: `EXO_JIT_PLACEMENT_WAIT_SECONDS` (note the `_SECONDS`
  suffix for consistency with the other JIT knobs). Memory blockers are now
  typed (`InsufficientMemoryError`) end-to-end: `filter_cycles_by_memory`
  empty with candidates present, the JIT reserve refusal, and per-node
  pipeline layer-fit all classify as memory-blocked; the API polls
  `state.node_memory` every 2 s inside the single-flight lock and only for
  memory blockers. Default **0 = off** per standing discipline;
  start_cluster.sh plumbs it and documents the flip to ~120 after the gate.
- 1a graceful shutdown: runner SIGTERM path now drops the model graph and
  returns the MLX pool to the OS pre-exit (`_release_gpu_memory_before_exit`
  in bootstrap.py: unbind builder/runner → gc.collect → mx.synchronize →
  mx.clear_cache). Known limitation: with `EXO_STALL_SAMPLER_SECONDS` set,
  the sampler thread's frame pins the runner ref and the release degrades to
  the old exit-time reclaim (diagnostic-only env, acceptable).
- 1b reclaim curve: `wait_for_memory_reclaim` in start_cluster.sh (kill
  timestamps per node, wired+compressor residual via vm_stat, warn-only
  explicit STUCK MEMORY alert naming the reboot escape hatch; knobs
  `EXO_RECLAIM_CHECK/RESIDUAL_MAX_GB/DEADLINE_SECONDS`). The generated
  `~/relaunch_exo.sh` now does the graceful SIGTERM kill + reclaim wait
  itself (previously it only started exo, leaving quick node-side restarts
  to hand-run `screen -X quit`/pkill −9).
- **Remaining to close item 1:** run the validation gate on the cluster
  (kill/relaunch cycling with `EXO_JIT_PLACEMENT_WAIT_SECONDS=120`; expect
  zero placement 503s, bounded TTFT), then flip the default in
  start_cluster.sh. Coordinate before touching the cluster (hermes).

**Symptom:** after killing exo, ~60GB/node stays unreclaimed for ~1 min;
JIT placement instantly 503s ("no admissible placement") on quick relaunch.
Once (2026-07-09) m4-2 kept 61GB stuck in the compressor with no owning
process and needed a reboot.

Two distinct problems:

### 1a. Normal reclaim lag (~60 s) — operational fix
- exo API/placement layer: when the ONLY placement blocker is
  `ramAvailable` (both nodes present, topology fine), poll node memory with
  backoff for up to a gated window before failing the request. Proposed env:
  `EXO_JIT_PLACEMENT_WAIT` (default ~120 s, 0 = current hard-fail).
- Runner graceful-shutdown path: SIGTERM handler that drops MLX buffer
  pools / clears caches before exit, SIGKILL only as fallback, so
  reclamation starts immediately and cleanly. (The relaunch scripts
  currently `screen -X quit` + `pkill`, which SIGKILLs.)
- **Validation:** kill/relaunch cycling in a loop; zero placement 503s;
  time-to-first-token after relaunch bounded.

### 1b. Pathological stuck memory (rare) — detect + runbook
- Likely AGX/Metal wired pages orphaned by SIGKILL mid-GPU-op; not fixable
  from userspace. Mitigation is the same graceful-termination ordering
  (cancel GPU work, drain command buffers, then kill — the sibling-load
  watchdog fix `5d5f3494a` already gives the supervisor the right hook
  point/discipline).
- Add a reclaim-curve check to relaunch tooling: if free pages have not
  recovered ~3 min after kill, alert explicitly (instead of mysterious
  503s). Reboot documented as the escape hatch.

**Scope:** contained patch (placement layer + small signal handler +
script check). No kernel work.

---

## 2. Generator-step vs model() ulp parity (moderate; unlocks the gold-standard gate)

**STATUS 2026-07-10 (later): attribution RUN + accept-rule fix LANDED;
byte-equality NOT yet achieved — residual serving-only ulp drift localized
to the first MTP cycles. Facts:**

- Harness results (m4-1, quantized random model): E1 — EVERY construction
  factor bitwise EQUAL (stream ctx, input dtype/laziness, pre_norm hook,
  full genstep replica). E3 — first trajectory split at step 30 with
  logits bitwise EQUAL → pure DECISION-RULE divergence. E2 — 4/64 steps
  flip between argmax(raw) and argmax(bf16-normalized logprobs).
- Fix landed: `EXO_DSV4_MTP_ACCEPT_LOGPROBS=1` (exo `6d48cd69f`, default
  off) aligns the three greedy accept/bonus sites to the generator's rule
  (argmax over logsumexp-normalized bf16 logprobs). Supersedes the
  tie-break fix (already off in prod).
- Serving byte-equality gate (MIN_CTX=0 + ACCEPT_LOGPROBS=1 + TIEBREAK=0,
  temp=0, 3 prompts x 600 tok): each config reproduces ITSELF 3/3
  byte-identical (incl. through the prefix cache), but MTP-on still forks
  from MTP-off at ~token 78. Logprob forensics: sampled-token logprobs
  bitwise-equal to tok 60, but top-5 TAIL values differ from tok[2]
  (abs pos 43) by 1-2 raw-logit ulps → the residual drift starts at the
  FIRST cycles and accumulates silently. Abs pos 44 = the first ratio-4
  pool-flush boundary after the 41-token prompt.
- Model level is clean EVERYWHERE tested single-node: mixed-quant ldiff
  with the REAL recipe (experts mxfp4 g32, rest affine8 g64) bitwise
  CLEAN at 4K and at tiny ctx (16/64/256); logsumexp + argmax are
  batch-shape-invariant. So the residual lives in serving-only cycle
  machinery (reject-trim / pool-flush repair / commit-forward) or TP.
- **Next bisect:** relaunch with `EXO_SPECULATIVE_GAMMA=1` (and check
  whether γ=0 = bonus-only is valid): no/l fewer rejections → no trims.
  If drift vanishes → reject-trim/flush-repair path; if it persists →
  base cycle (L=2 verify + commit) under TP → build the cycle-level
  model harness (drive verify+trim+commit against sequential).

**STATUS 2026-07-10 (evening): residual ROOT-CAUSED to the batch cache
classes; three gated fixes landed; byte-equality still blocked on two
mlx-lm defects, precisely bounded with a ready harness. Chain of evidence:**

1. Cycle stats (`EXO_DSV4_MTP_CYCLE_STATS`, exo `a012be9f6`): serving
   p1 run = 78 cycles, 58 rejections (from token 0), **regime_b = 0** →
   the pool-contamination repair NEVER runs in serving.
2. Cause: mlx-lm's batched generator converts pools to
   **BatchPoolingCache at insert, which does NOT subclass PoolingCache**
   → `_collect_pooling_caches` returns [] at every concurrency. Fixed
   gated (`EXO_DSV4_POOL_SNAPSHOT_BATCH`, exo `da6bd3437`) with
   class-aware snapshot predicate + flush detection (also fixes the batch
   path's pending-bump false positive).
3. Regime-b ordering bug (exo `e954dff99`,
   `EXO_DSV4_POOL_RESTORE_AFTER_TRIM`): restore_meta'd pools were
   re-trimmed by the blanket CacheList.trim(γ+1) — double rollback.
   Proven bitwise-fixed on plain caches by `~/scratch/ldiff_cycles.py`
   (accept/reject0/reject1 all CLEAN with trim-then-restore).
4. **Remaining defects (mlx-lm, harness-reproducible single-node):** with
   the REAL serving classes (`LDIFF_BATCH_CACHE=all`), even full-accept
   L=3 rowseq chains DRIFT from sequential (first at the first ratio-4
   flush boundary; ~0.08 logits). Bisect: ring-only CLEAN, pool-only
   accept3 CLEAN, ring+pool DRIFT → interaction (batch-ring ARRAY offsets
   into the batch pool). Pool-state comparer: first divergence is
   **buf_kv CONTENT with all structural counters equal** → the pooled/
   buffered VALUES written by an L=3 forward differ from three L=1
   forwards (accumulate_windows / update_and_fetch_deferred under array
   offsets). Reject cycles drift additionally (rollback fidelity of the
   batch pool beyond the ordering fix).
   **Implication: the campaign's rowseq bitwise proof used PLAIN caches —
   serving (batch classes) rowseq at ≥32K is likely still ulp-drifting.**
5. Also landed en route: byte-equality probe scripts, `ldiff_mixedq.py`
   (real mixed-quant recipe: BITWISE CLEAN at 4K and tiny ctx), M-dispatch
   kernel probes (quantized matmuls M-invariant at all probed shapes;
   bf16 dense M-dependent at many shapes but the only decode-path bf16
   matmul — the router gate (4096,256) — is M-invariant).

**STATUS 2026-07-10 (night): model-level cycle machinery PROVEN
bitwise-faithful (mlx-lm `c0ddfbd`, exo `8a948c3b5`+fence-drain commit);
serving still forks — TP=2 is the last suspect standing.**

- Three more root causes found + fixed via the batch-cache harness:
  (a) `EXO_DSV4_ROWSEQ_ROWMASK` — rowseq rows hardcoded mask=None, wrong
  for batch rings (make_mask returns an ARRAY at N=1 → different SDPA
  specialization; proven by forcing the sequential reference to None:
  accept-chains went CLEAN); (b) BatchPoolingCache restore left the
  pooled tensor at its deferred-write PADDED width (visible_width 13 vs
  12 → different SDPA K-length); (c) ring `trim()` is NOT rollback-safe
  once rotated (draft writes destroy the oldest window rows, decrement
  left_padding, wrap _idx) → new `save_spec_state`/`restore_spec_state`
  + `EXO_DSV4_SPEC_STATE_RESTORE` unified rollback (wholesale ring+pool
  restore + commit-replay; supersedes regime a/b at B=1). CRITICAL MLX
  FACT: `__setitem__` AND `+=` on mx arrays mutate IN PLACE (aliased) —
  reference "snapshots" do NOT preserve pre-write state; materialize
  with `mx.array()` (a held offset reference was a 0.8-2.7-logit bug).
- Harness gate: accept3/reject0/reject1 × P=48/41/4096 ALL BITWISE CLEAN
  on the real batch classes (previously drift at every reject shape and
  at all wrapped-ring shapes).
- Serving (v4/v5 gate runs, all five fixes + MIN_CTX=0): self-repeat
  DETERMINISTIC (after adding the async-fence drain around the unified
  rollback — the B=1 restore rebinds buffers and raced the verify's
  async side-chain); CYCLE-STATS shows the unified path fires on every
  rejection (158/158); FENCE_ASYNC=0 A/B is byte-identical (fence is
  value-neutral). BUT MTP-on still forks from MTP-off at a near-tie
  (~token 39), with 1-2-ulp top-5 tail drift from token 2 — the same
  signature as before all fixes. Single-node model level is exhaustively
  clean ⇒ the drift source is TP-specific (2-rank jaccl: sharded-weight
  kernels at verify-batched M vs M=1, or collective-adjacent eval
  segmentation).
- Perf note: unified rollback + MIN_CTX=0 ≈ 15-16 t/s @600tok vs ~26 t/s
  pre-fix MTP-on — do NOT flip defaults yet; needs the per-row ring
  history optimization (snapshot only the γ+1 overwritten rows +
  counters) and the rejection-rate-aware commit before prod.

**RESOLVED (same day, follow-up leg): the −41% was the commit-forward
replay, and it's gone — EXO_DSV4_SPEC_CACHE_ROLLBACK.**

- Attribution A/B (c=1, 600-tok): prod 27.3 t/s; +six fixes with rowseq
  ≥32K only 16.0 (−41%); +MIN_CTX=0 15.8 (−1% more). The ENTIRE
  regression was the per-rejection commit-forward (rejects fire on
  60-85% of cycles); **rowseq at all contexts is ~free ⇒ plan item 3 is
  effectively closed** (drop the 32K threshold whenever the fix stack is
  on).
- Fix (mlx-lm 387e567, exo 95ae7a787): cache-level exact undo. Rings
  arm_spec_stash() → rollback_spec_write(snap, keep) = restore snapshot +
  re-push committed rows through _update_in_place (sequential decode's
  own write path). Pools spec_rollback(snap, keep) by flush attribution:
  flush in committed prefix (or none) → trim() is already exact; flush
  caused by a rejected token → restore_meta + re-accumulate committed
  prefix from the stash (cannot re-flush at γ+1 ≤ ratio).
  spec_can_rollback refuses B>1 / multi-flush → commit-forward fallback.
  Deferred-bump split (applied vs pending) after a kept flush at the last
  committed row is commit_pending-equivalent; harness comparer normalized
  to totals.
- Harness: accept3/reject0/reject1 × P=48/41/4096/4222 (ratio-128 flush
  straddling the verify boundary) × batch+plain classes = **15/15 BITWISE
  CLEAN**, cache-level path firing on every reject cycle (harness raises
  on fallback; none fired).
- Serving (v6 gate runs, six fixes + CACHE_ROLLBACK + MIN_CTX=0):
  **26.2 t/s (−4% vs prod 27.3)**; CYCLE-STATS cache_rb=263/263 rejects
  (zero fallbacks); self-repeat 3/3 deterministic; **forks vs MTP-off
  moved LATER — v4 (commit-forward) forked at chars 131/149/291, v6
  forks at 473/280/361.** Every replay forward was itself a TP-drift
  injection point; removing them narrows the residual to the verify
  forward under TP. Concrete task-#8 hypothesis: jaccl/TP all_reduce (or
  sharded-kernel) numerics at verify M=γ+1 vs decode M=1.
- Defaults still OFF pending the TP residual (byte gate 0/3, forks just
  later); flip candidates after task #8: six fixes + CACHE_ROLLBACK +
  MIN_CTX=0 at −4%.

**Next unit of work:** 2-rank TP harness — run ldiff_cycles under
mx.distributed (2 local ring ranks or the real jaccl pair) with the
sharded model; bitwise-compare verify chains vs sequential per rank.
First probe: all_reduce/matmul M-dependence per rank at M=1 vs M=γ+1
(the v6 fork-later datapoint says per-forward TP numerics, not cache
state). Then fix, re-run the serving byte-equality gate
(ACCEPT_LOGPROBS+SNAPSHOT_BATCH+RESTORE_AFTER_TRIM+ROWSEQ_ROWMASK+
SPEC_STATE_RESTORE+CACHE_ROLLBACK+MIN_CTX=0), DSML battery, and only
then flip defaults.

**(earlier same day) step 1 archaeology, key findings that REVISE the
suspect list below:**

- **Mask is NOT a suspect at c=1.** Neither path passes `mask=`; the model
  builds it internally from the same cache object, and
  `RotatingKVCache.make_mask(N=1, window_size=sliding_window)` returns
  `None` whenever `max_size == window_size` (ours) — identical for both
  paths. (Batch-cache classes at B>1 DO build arrays; different story.)
- **LMHEAD_LASTROW is inert on both paths** — the slice requires
  `L > 32` (deepseek_v4.py:4443-4462); decode L=1 and verify L=γ+1≤9 never
  trigger it. Rule out (harness asserts it for the record).
- **NEW top suspect — sampling-boundary decision rules, not kernels:** the
  generator picks `argmax(logits - logsumexp(logits))` in native bf16
  (generate.py:1462, batched path has NO fp32 cast, unlike single-stream
  generate_step); the MTP accept path argmaxes RAW `verify_logits`
  (dsv4_mtp.py:3220); and the **tie-break fix (default ON,
  `EXO_DSV4_MTP_TIEBREAK_EPS=0.5`) picks the LOWEST id within 0.5 logits
  of max** for the bonus token (dsv4_mtp.py:3248-3262). Any top-2 gap
  < 0.5 therefore diverges MTP-on from MTP-off **even with bitwise-equal
  logits**. Byte-equality (step 3's gate) structurally requires retiring
  the tie-break fix — which its own comment says exists only to mask the
  batched-vs-sequential ulp gap. Sequence: prove bitwise parity → flip
  `EXO_DSV4_MTP_TIEBREAK_FIX=0` → land the gate.
- **Remaining numeric suspects (harness E1 toggles each):** thread-local
  generation-stream context (generate.py:2037) vs default stream; input
  construction (generator: materialized uint32 sampler output; commit/
  refeed: materialized int32; verify: LAZY uint32 concat of argmax rows);
  `pre_norm` capture hook (MTP generators wrap the final norm —
  mtp_batch_generator.py:58-84 — retaining an extra graph output that can
  change fusion; stock MTP-off generator does not); extra
  async_eval/eval barrier placement.
- **Interlock with item 3:** the byte-equality gate also needs rowseq at
  ALL ctx (below 32K prod still runs classic batched verify, not bitwise),
  so the gate lands at full strength only after item 3 drops the threshold.
- Harness layout: E1 single-state factor attribution (bitwise logits diff
  per factor, fresh deterministic prefill per variant — ldiff trick);
  E2 decision-rule divergence on identical logits (X vs Y vs Y+tiebreak,
  reports min top-2 gap); E3 X-replica vs Y-replica trajectory loop
  (first divergence classified DECISION-RULE vs NUMERIC via bitwise check).

**Gap:** the generator's L=1 decode step and a raw `model()` L=1 call differ
by ulps, so MTP-on vs MTP-off produce different-but-equally-valid
trajectories at temp=0. Byte-equality across serving modes is therefore not
usable as a regression assertion today (measured: X_seq reproduces itself
exactly; Y/rowseq differs from X at early near-ties while being bitwise
clean at the model level).

### Plan
1. **Attribute:** harness that freezes one cache state and computes
   next-token logits via (a) the generator step path and (b) a raw
   `model()` L=1 call; diff bitwise. Suspects in likely order:
   - mask argument differences (explicit array vs `None` → different SDPA
     kernel specialization),
   - `EXO_DSV4_LMHEAD_LASTROW` slicing differences,
   - input array construction/dtype differences,
   - capture hooks (`pre_norm`) perturbing lazy-graph fusion decisions.
2. **Align:** make the MTP verify/commit/refeed input+mask construction
   exactly match the generator step (or vice versa — whichever diff is
   smaller). Acceptance: the harness reads bitwise-equal over N random
   states.
3. **Land the gate:** one-command assertion
   `MTP-on output == MTP-off output, byte-for-byte, temp=0` (short + long
   ctx rungs). This becomes the cheapest strongest gate for every future
   serving change, including item 3.

**Scope:** plumbing archaeology in `batch_generate.py` + `dsv4_mtp.py`;
no kernel work.

---

## 3. Row-seq short-ctx perf refinement → drop the 32K threshold (largest)

**Problem:** row-seq currently loops the ENTIRE attention module per row:
3x projection dispatches and — worse under TP — 3x o_proj all_reduces per
layer per cycle. That overhead is the whole 1.6x short-ctx cost
(4K gen: 35.9 s row-seq vs 22.9 s classic).

### Design: hoist the invariant parts out of the per-row loop
- **Batched (hoisted):** Q/K/V projections at M=L (quantized qmm —
  bitwise batch-invariant, proven), and the output projection at M=L with
  its SINGLE TP all_reduce per layer (down from L).
- **Per-row (kept):** exactly the state-bearing parts — rotating-cache
  mutation, pool updates (incl. deferred bumps), indexer scoring (M=1
  gemv is REQUIRED for bitwise anyway), top-k, gather, SDPA.
- Restructure inside each of the three attention classes
  (`LocalAttention`, `CompressedAttention`, `SparseCompressedAttention`)
  rather than at the `DeepseekV4Block` level. Keep the block-level gate as
  the fallback path; new behavior behind e.g.
  `EXO_DSV4_VERIFY_ROWSEQ_HOISTED=1` until validated, then default.

### Non-negotiable gates
1. ldiff harness bitwise ZERO at 4K/32K/131K (and `ldiff_b2.py` at B=2).
2. With item 2 landed: end-to-end byte-equality vs MTP-off at temp=0.
3. DSML battery ladder clean.
4. Perf: 4K decode within ~10% of classic MTP → set
   `EXO_DSV4_VERIFY_ROWSEQ_MIN_CTX=0` and retire the threshold concept.
   If the residual is larger, pick the break-even ctx from measurement
   (probe ladder 4K/8K/16K/32K) instead of the current 32K guess.

**Scope:** real surgery across three attention implementations in a
4.6K-line file dense with env gates — the largest item; no new Metal
kernels. Method reminder from the campaign: isolated microbenches
understate in-graph costs; always A/B in-model.

---

## 4. MoE expert-batching M-dependence (attribution only; lowest priority)

**Observation (ldiff_b2, 2026-07-10):** at B=2, verify-vs-stepping residual
~0.03 logits, argmax-stable, unchanged by row-seq. Hypothesis:
tokens-per-expert differs between an M=6 batched pass and M=2 stepping,
crossing the gather-kernel B/E∈[2,8] selection gate → different kernel /
accumulation order per expert.

### Plan
1. Extend `qmm_invariance_sweep` to the MoE gather path: fixed router
   assignments, same token processed inside an M=2 vs M=6 batch, bitwise
   diff per expert. Pin the component: gather_qmv selection gate,
   sorted-run accumulation order, or dense fallback.
2. **Decide, don't assume:**
   - kernel-selection boundary → canonicalize selection (key on
     per-expert row count consistently); likely cheap;
   - accumulation order deep in gather_qmv → fixing costs perf for an
     effect 50x below the corruption threshold; document as the known
     noise floor for c≥2 and stop.

**Scope:** attribution is a small harness extension reusing existing
tooling. No fix work until the attribution says it is cheap.

---

## Standing discipline for all four
- Same commit/push/bundle-sync flow as the campaign (m4-1 WAN git broken →
  git bundles over scp; venv site-packages `deepseek_v4.py` must match the
  mlx-lm commit).
- Any relaunch kills live sessions — coordinate before touching the
  cluster while hermes is in use.
- Every change lands behind an env gate, default-off until its gates pass,
  then defaults flipped in `start_cluster.sh` + node relaunch scripts.
