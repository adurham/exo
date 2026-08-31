# P08 pre-registration — resolving P07's two open items (2026-08-30)

**Gates and criteria below are registered BEFORE any measurement is taken.**
Nothing in this document may be edited after the first capture runs; results go
in a separate results doc.

Scope: exactly the two items P07 left open. Nothing else is in scope.

- **Item 1** — is `attn.sdpa.compressed`'s ceiling denominator inflated by
  counting causally-masked positions? (P07 §8: UNDETERMINED, ~48% vs 79.1%.)
- **Item 2** — can a custom/tuned top-k beat the current `attn.indexer`
  argpartition path at production k=512 / P=55000? (P07 §7: OPEN, ceiling
  ~1.03x prefill, prior fused-indexer attempt measured 0.54x = slower.)

Item 1 is prioritized: it may reveal a LARGER headroom opportunity than item 2's
already-small ceiling.

## 0. Cluster state at pre-registration

Verified live before writing this document, from real signals (not assumed):

- m4-1 PID **59909**, m4-2 PID **60392**, both `etime` 01:48:14 → continuous
  since ~19:34 today. Zero relaunches across P06/P07/P08-start.
- Both nodes `up 4 days, 23:46`. Host uptime independently confirmed.
- Production verbon3 flag set verified via `ps eww <live pid>` on BOTH nodes —
  identical env, including `EXO_DSV4_PREFILL_ARGPARTITION=1`,
  `EXO_DSV4_INDEX_TOPK=512`, `EXO_DSV4_SEQ_SPLIT=1`, `EXO_COMPUTE_DTYPE=bf16`,
  `EXO_DSV4_HC_EXPAND_KERNEL=1`, `EXO_DSV4_HC_COLLAPSE_KERNEL=1`,
  `EXO_DSV4_LMHEAD_MXFP8=1`. **Zero leftover test env vars.**

Target for this phase: zero cluster relaunches, same as P07. Captures run as
standalone processes beside the live runner (p01/P03/P07-proven).

**Node assignment (to keep two concurrent captures from perturbing each
other):** Item 1 captures on **m4-1**; Item 2 captures on **m4-2**. Each node
therefore carries the same one-extra-GPU-process load profile P07 validated.
All A/B ratios are computed *within* a single process on a single node, so a
ratio is robust to any residual background-load difference between nodes.

## 1. Inherited, non-negotiable method constraints

Carried forward verbatim from P07 §5 / the campaign's standing rules:

- **Per-kernel GPU capture against a real eval barrier.** Never a span timer.
  Under MLX's lazy executor a timer that does not end at `mx.eval`/
  `mx.synchronize` measures graph CONSTRUCTION, not GPU work — the exact bug
  that hid the indexer's real cost for 8 days (span read 5.81 µs for a
  7.7–15.4 ms op, a ~1300–2600x under-read).
- `MLX_GPU_TIME=1` + `MLX_DISPATCH_COUNT=1`; report GPU-busy time and dispatch
  count for every timed configuration.
- `.eval()` every constructed `nn.Module` before timing (`training` defaults
  True; cost a false lead once at 8.8x error).
- Never force evaluation with cancelling arithmetic (`*0`) — MLX constant-folds
  the whole graph and reports a fictional ~500,000 GB/s.
- Rotate input banks past the L2 boundary before quoting any GB/s, and report a
  small-bank-vs-large-bank sanity check.
- Median of ≥5 timed reps after ≥3 warmup reps, per configuration.
- **An env var's code-level `os.environ.get(X, "0")` default is NOT the
  production default.** Any claim of the form "flag X is off/on" must be
  verified against the LIVE process env (`ps eww <pid>`) and
  `start_cluster.sh`, never against the source fallback. (P07 §4: a review pass
  wrongly called a real finding an artifact by making exactly this mistake.)
- Verify production flags and runner PIDs before AND after every capture;
  real post-capture smoke test before any capture is treated as valid.
- Every headline number re-derived against the raw `results*.json` before commit.

## 2. Item 1 — `attn.sdpa.compressed` ceiling denominator

### 2.1 What is actually being asked

P07 established the arithmetic but not the runtime fact. The bench
(`bench/attn_production_class_bench.py`) counted **dense** `L_band × CATTN_KV`
work using a synthetic 95%-dense mask (261.3 GFLOP), while production passes a
real **causal** mask to `mx.fast.scaled_dot_product_attention` (real causal work
158.3 GFLOP). Whether the counted denominator is honest depends entirely on a
runtime fact nobody has measured: **does MLX's fused SDPA kernel skip
fully-masked blocks, or does it compute them and mask after?**

### 2.2 Pre-registered decisive test

At production compressed-attention shape on m4-1, GPU-timed against a real eval
barrier, time `mx.fast.scaled_dot_product_attention` under two mask conditions
that differ ONLY in mask content (identical dtype, shape, layout, call path):

- **(a) causal** — the mask production actually passes.
- **(b) dense** — all positions visible, same tensor shape/dtype.

Let `R = t_causal / t_dense`.

Also required, same shape, same session:
- **(c) no-mask** (`mask=None`) as a third control, to separate "kernel skips
  masked work" from "kernel pays a fixed cost to read the mask tensor at all."

### 2.3 Pre-registered interpretation rule (assigned mechanically)

Let `f = causal_FLOP / dense_FLOP` at the measured shape, computed from the
actual shape, not reused from P07's 0.6058.

| Measured | Verdict | Real efficiency denominator |
|---|---|---|
| `R ≤ 0.75` | **KERNEL EXPLOITS MASK** | real work = causal FLOPs → efficiency is the LOW figure (~48%-class) → **more headroom than T7 believed** |
| `R ≥ 0.92` | **KERNEL DOES FULL WORK** | real work = dense FLOPs → the 79.1%-class figure STANDS, denominator was fine |
| `0.75 < R < 0.92` | **PARTIAL** | report both bounds explicitly; do not pick one silently. Tiebreak by §2.4. |

### 2.4 Mandatory second, denominator-free test (the one that actually matters)

A denominator argument only bounds a theoretical ceiling. The question that
decides whether a P09 exists is **"is there real, reachable headroom?"** So,
independently of §2.3, measure the **direct floor**:

Attention at this shape fundamentally requires two matmuls (QK^T and PV) plus a
softmax pass. Build the floor from the SAME node's own measured performance:

1. Time `mx.matmul` at the exact QK^T shape and the exact PV shape on m4-1
   (this is the honest "best MLX can do on this hardware at these shapes",
   which is the only baseline this repo has ever respected — see the
   "Slower-than-Steel trap").
2. Floor = `max(matmul_QK + matmul_PV, softmax_bandwidth_time)` under the
   roofline `max()` convention (never additive — additive overestimates fusion
   wins, a documented trap).
3. `direct_headroom = t_causal_production / floor`.

**Pre-registered significance gate for Item 1:**

`attn.sdpa.compressed` is declared a **REAL LEVER (P09 candidate)** iff BOTH:
- `direct_headroom ≥ 1.40x` (i.e. ≥29% of its own time is recoverable), AND
- `span_share × (1 − 1/direct_headroom) ≥ 1.0%` of prefill wall — the same
  ≥1.0% e2e bar every other span in this campaign was held to, using the
  hc_expand-proven triage product.

`span_share` must be derived from a cited artifact, not estimated.

Otherwise Item 1 closes as **DENOMINATOR CORRECTED, NO ACTIONABLE LEVER** — an
honest outcome, and the correct one if the arithmetic moves but the reachable
work does not.

### 2.5 Ceiling-peak hygiene (closing P07's own caveat)

P07 §9 flagged that the 11.66 TFLOPS peak in
`bench/attn_production_class_bench.py:136-145` was measured on the **laptop**
M4 Max (32-core), not the 40-core Studio nodes, and that the 14.34 TF
replacement is *theoretical*, not on-node measured. P07 named the rigorous fix
and did not do it.

**This phase does it:** run `measure_peak_gemm()` (or equivalent dense bf16 GEMM
sweep) on a Studio node and report the measured on-node peak. Every Item 1
efficiency figure is computed against the MEASURED on-node peak, with the
theoretical 14.34 TF quoted alongside for comparison. No Item 1 percentage may
be produced by scaling a stale percentage — all are re-derived from fresh
on-node timings.

## 3. Item 2 — `attn.indexer` top-k spike

### 3.1 Scope box (registered up front, deliberately tight)

The e2e ceiling is **~1.03x prefill** even if a perfect kernel is found. Per the
phase brief, a clean "not worth it" closure is an acceptable and valuable
outcome (P05 Phase C precedent). Therefore Item 2 is boxed to:

- **Phase A** (always runs): floor measurement + existing-op composition sweep.
  No custom Metal.
- **Phase B** (runs ONLY if Phase A clears the gate in §3.3): one disposable
  Metal kernel spike.

If Phase A is ambiguous or fails the gate, Item 2 closes **KILL** without a
Phase B. No third phase under any outcome.

### 3.2 The measurement P07 did not make: what is top-k's real floor?

P07 reported top-k at "~4-7% of the 424 GB/s achievable streaming bandwidth."
That comparison is structurally generous to the complaint: a radix **sort** is
inherently multi-pass, so scoring it against a *single* streaming pass
overstates the gap. The honest ceiling is the cost of the minimum number of
passes a correct top-k requires over the score tensor.

Phase A must therefore measure, at the exact production score-tensor shape and
dtype (derived from code, with file:line provenance, not assumed):

1. **Single-pass streaming floor** — a genuinely single-pass reduction over the
   identical tensor (e.g. `mx.max` along the reduce axis), GPU-timed. This is
   the absolute lower bound for any top-k.
2. **Current path** — the production argpartition top-k, same tensor, same
   harness, same barrier discipline.
3. `pass_ratio = t_current / t_single_pass` — how many single-pass-equivalents
   the current sort costs.

### 3.3 Pre-registered gate for proceeding to Phase B (Metal spike)

Proceed to Phase B iff **EITHER**:

- **(a) Composition win**: some composition of EXISTING MLX ops (chunked top-k
  + merge, `mx.partition`, threshold/quantile two-pass select, bucketed
  radix-select, etc.) is **≥1.5x faster** than the current path at P=55000,
  with **exact top-k index-set equality** against the current path (including a
  forced-tie input), **OR**
- **(b) Structural gap**: `pass_ratio ≥ 4.0` AND a concrete algorithmic
  mechanism is named that reduces real WORK (not dispatch bookkeeping) — e.g.
  "a two-pass threshold select touches the data 2x where the radix sort touches
  it N x."

Rationale for 1.5x, stated before seeing data: the e2e ceiling is 1.03x. A
marginal op-level win (say 1.15x) yields <0.5% e2e — below the campaign's
standing ≥1.0% ship gate — and would not justify production risk on a
hand-rolled kernel in a repo whose prior fused-indexer attempt measured **0.54x
(slower)**.

### 3.4 Phase B gates (if reached)

The full inherited ship gate applies, all four required:

1. Predicted e2e win **≥1.0%** of prefill wall via `span_share ×
   per-op-reduction`.
2. The fix reduces real WORK or dispatch COUNT — not just re-implements an MLX
   primitive. (Every fusion in this repo that merely re-implemented an MLX
   primitive lost once pipelined: indexer fused kernel 0.54x, wq_a+wkv −0.48%.)
3. Validated with a **PIPELINED** microbench in situ, never per-call in
   isolation — MLX's async executor already overlaps dispatch across the op
   chain, so isolated per-call savings are systematically inflated.
4. Correctness: **exact top-k index-set equality** with the production path at
   all tested P, including forced ties. Top-k is a selection op, so exactness is
   the right gate — stricter than, and replacing, the 0.2% relative-error gate
   used for numeric kernels.

Failing any one → **KILL**, recorded with the failing number.

## 4. Phase-level conclusion gate

Registered now so it cannot be softened later:

- If Item 1 closes with no actionable lever AND Item 2 closes KILL, the honest
  phase verdict is **"the easy wins in this area are now exhausted"** — recorded
  plainly, with no manufactured P09. Four consecutive phases (P05 lm_head, P06
  shared_experts, P07 prefill remainder, P08) is enough to say so.
- If either item clears its gate, it is characterized fully as a named P09
  candidate with a concrete next step, even if not implemented this phase.
- Neither item may close as "needs more investigation" without naming the
  specific measurement that would resolve it and why it was not made.

## 5. Delegation integrity

- Re-verification of any worker's result is delegated to a SEPARATE read-only
  reviewer, never done by the same worker that produced it.
- Review-only subagents are explicitly forbidden from touching git state — no
  `git stash`, `git checkout`, `git reset`, no commits, no branch changes. (A
  reviewer in an earlier phase used `git stash` despite being told not to;
  recovered cleanly, must not recur.)
- Any "the flag defaults off/on" claim must cite `ps eww <live pid>` output or
  `start_cluster.sh:<line>`, never a source-level `os.environ.get` fallback.
