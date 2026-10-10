# ROUND-Q2-FU1 — the γ batch-shape divergence diagnostic (why the verify shape moves confident argmaxes)

Author: Phase-20 PM (delegation). Date: 2026-10-10 (CDT). Campaign worktree
`/private/tmp/phase20-campaign` (branch `deploy/phase20-campaign`). Capture/eval branch
`deploy/q2-fu1-capture`.

**Diagnostic + documentation only** (no engine fix ships this round). **Branch-only** (the capture
surface is on the eval branch; production default unchanged).

**STATUS: PHASE A (offline instrumentation) COMPLETE. LIVE CAPTURE — FROZEN / PENDING GO.**
The owner requested no live action pending a personal production-state test; the pre-approved
≤1 eval boot + 1 restore is re-gated on a NEW orchestrator GO. Every live-results section below is
PENDING. Production is currently LIVE at `deploy/next19-dense @ 99e2966ee` (restored after
ROUND-Q2-GAMMA, 12:07 CDT) and is NOT to be touched until GO.

---

## 0. DECLARED BUDGET (fixed BEFORE spending)

- **Boots: 0 used so far.** Pre-approved when GO arrives: **≤1 eval boot + 1 restore** (2 boots
  total). Now gated on an explicit orchestrator GO (owner hold).
- **Process relaunches:** only via the driver's arm switches (sed `~/relaunch_exo.sh` + graceful
  node-process relaunch, NOT reboots). Arms g3,g4 ⇒ ~2 switches × 2 nodes ≈ 4 relaunches (~2–3 min
  each + a JIT warmup POST).
- **GPU budget: ~1–2 GPU-h.** **Encode spend: 0.** Capture dumps are MB-scale; memory-lean on-node
  work only (a 27 GiB fp16 load once GPU-timed-out beside the live runner — never attempted here).
- **Measurement plan:** one fixed-prefix prefill + a handful of verify shapes/rounds; faithfulness
  gate first, read-out second.

---

## 1. THE FINDING (from ROUND-Q2-GAMMA) + the diagnostic spec

ROUND-Q2-GAMMA (executed 09:04–12:07 CDT 2026-10-10) re-priced the speculative draft depth `gamma`
(verify rows **M = gamma+1**). Its identity gate (`raw/pricing/q2/round/round_eval.json`) found the
two **gamma=3 arms byte-identical** (divergence rate 0.0 — the stream is deterministic) while
**every other γ arm diverged from gamma3a at CONFIDENT positions**, margins far above ε = 0.05:

| arm | M | divergence rate | first divergence (rep0 / rep1) |
|---|---:|---:|---|
| γ2 | 3 | 0.654 | idx 159 margin 0.50 · idx 12 margin 0.69 |
| γ4 | 5 | 0.914 | idx 14 margin 0.28 · idx 12 margin 0.69 |
| γ5 | 6 | 0.967 | idx 6 margin 1.03 · idx 6 margin 0.94 |
| γ3b | 4 | 0.000 | (byte-identical to γ3a) |

Reference-model read: **flips at margins up to ~1 nat are TOO LARGE for plain bf16
reduction-order noise.** Hence this round.

**HYPOTHESES.**
- **H1 — MoE routing amplification (benign).** A tiny upstream numeric delta flips a router top-k
  selection at a near-tie; a different expert fires; final logits move ~1 nat. Read-out: **small
  |Δlogit| UNDER IDENTICAL ROUTING.**
- **H2 — kernel-path/precision discontinuity (BUG-class).** e.g. the quantized matmul switching
  gemv↔gemm at different row counts M (different accumulation), or a mask/offset/KV-rollback
  inconsistency in the verify path. Read-out: **large |Δlogit| under identical routing**, or a step
  change at a specific op/layer.
- **H3 — semantic attention/KV/indexer non-equivalence (added by our reference model).** Rotating-KV
  in-place writes truncating earlier rows' windows, pool flushes becoming visible to all rows at
  once, indexer score-GEMM M=L reduction order flipping near-cutoff top-k membership. Large deltas
  with identical *MoE* routing that would otherwise be misfiled as H2. This is the best-documented
  failure class in this repo's sibling engine (`start_cluster.sh`: *"an L>1 batched verify forward
  is NOT equivalent to L sequential steps … cumulative drift ~1.7 logits @115K"*).

**DIAGNOSTIC SPEC (reference-model-designed).** Fixed prefix; single forward passes at **M = 1,3,4,5,6**
verify rows through the **REAL engine code path**; dump final-position logits **+ per-layer MoE router
top-k**; report **(a) the FIRST LAYER where routing differs between shapes** and **(b) max |Δlogit|
WHEN ROUTING IS IDENTICAL**. Plus an **M=1 target-only forward** = the OFFLINE analog of the
γ3-vs-no-speculation baseline. Read-out: small |Δlogit| under identical routing ⇒ H1 (close; keep the
quality-gate rule); large ⇒ H2 (deliver fix sketch + escalate). **Reference = M=1 (sequential)**, not M=4.

**PRODUCTION IMPLICATION (to state, not assume).** If the verify shape moves confident argmaxes,
then γ3's decoded output does **not** equal target-only greedy either — *"speculation is lossless"*
is **not strictly true** on this stack. Canaries pass; not urgent; but it must be stated.

---

## 2. METHOD (offline-instrumented; live execution pending GO)

**Mechanism (validated by a reference model).** A faithful capture REQUIRES running new code in the
engine process; there is **no faithful 0-boot path** (offline single-node repro is impossible — TP=2
≈100 GB/rank is the object under test; live-process injection on a user-serving cluster is excluded).
The engine reads its capture env **once at construction**, so the mechanism is one process restart
running the instrumented build = the pre-approved eval boot.

**Instrumentation (branch `deploy/q2-fu1-capture` @ `898cf36a`, off `deploy/q2-gamma` @ `2cf078519`).**
- `_maybe_install_q2fu1_capture()` in `engine.py` — an exact mirror of the existing
  `_maybe_install_next18_capture()` bootstrap, gated by `DSV41_Q2FU1_CAPTURE=<dir>`; called from
  `Dsv41Engine.__post_init__`. Env unset ⇒ immediate return: **no import, no monkeypatch, zero cost** —
  production path byte-for-byte unchanged.
- `start_cluster.sh` forwards `DSV41_Q2FU1_CAPTURE{,_MS,_ROUNDS,_DMAX,_KSWEEP}` and derives
  `PYTHONPATH` (mirrors the next18 wiring).
- The operator-staged in-process hook `bench/phase20_q2_fu1/q2fu1_capture.py` monkeypatches
  Model/Block/Gate/ShardedHead + the engine request path at RUNTIME (mlx-lm source untouched);
  activated by a request whose `correlation_id` starts `q2fu1`. A `sweep=1` request runs, at the
  post-prefill anchor, an M=1 sequential greedy chain (the per-position reference), then SINGLE
  forwards of `rows[:M]` for M ∈ {1,3,4,5,6} (+ an M=4 repeat) **from exactly restored full-cache
  snapshot bytes** (window rings, compressor carries + chunk stashes, index/comp rows, engram slice,
  offset, draft windows). Also: bf16-vs-fp32 MoE all-reduce arm (runtime flip of
  `moe._MOE_ALLSUM_BF16`); fence on/off arm; a **rollback A/B arm** (production `spec.rollback` at
  accepted=1 vs + exact restore of only the rejected rows' ring slots); a `ms=N`/`rollback=1` passive
  capture of real verify rounds; a `nospec=1` pure-greedy reference. The cache is left exactly as
  production would leave it (verified stream-neutral in the self-test).
- **Fidelity discipline:** wrappers only HOLD python refs to tensors the production forward already
  computes — no `mx.eval`/sync/`astype` inside a forward, no op reordering, no return-value changes.
  The driver's **faithfulness gate** runs the instrumented build at γ3 (must reproduce
  `arm_gamma3a.json`) and γ4 (must reproduce `arm_gamma4.json`, div idx 14/12); a mismatch ⇒ dumps
  untrustworthy (reported LOUDLY).

**Controls included in the same boot:** M=4 vs M=4 repeat (noise floor, expect ~0); bf16-vs-fp32
all-reduce at fixed M (the opposite trap — small deltas that flip routes would be misfiled as H1);
fence on/off at M=1 and M=4 (calibration; M=3..6 all sit on the same side of the threshold 16, but
M=1 vs M>1 may not); and the rollback A/B (the H3 discriminator).

**Dump format** (one npz per forward, pinned): scalars `M,start_pos,gamma,round_idx,arm,rank,…`;
`rows_ids` int32[M]; `logits` float32[M,V] (rank slice; NaN-spliced across ranks in `.merged.npz`);
`hidden_in/out` float32[M,D]; per body layer `idx_L` int32[M,topk], `w_L` float32[M,topk], `hid_L`
float32[M,D], `margin_L` float32[M]. `hid_*` are hc-mean summaries of the hyper-connection residual
stream (documented deviation: the full 4×5120 stack is not stored).

---

## 3. OFFLINE FINDING — a rollback-component candidate for H3 (offline evidence only)

Surfaced while building the instrumentation; **offline evidence, NOT yet confirmed on the real
model** (the live capture's A/B arm is what confirms it):

`spec.rollback` restores the compressor carry but **not the sliding-window ring** (`win_kv`, slot =
pos mod 128). A verify forward writes slots for ALL M rows; rejected rows q ≥ target leave their KV
in slot q mod 128, which **aliases the live position q−128** read by the NEXT forward's window.
Whenever rejected ≥ 2, the continuation is contaminated. With accepted = 1: γ2 (M=3) is provably
clean (the single rejected slot aliases a dead position); γ3/4/5 (M=4/5/6) are contaminated.

- Offline probes (tiny real model, CPU): production rollback ⇒ next-forward max|Δlogit| = **0 for
  M=2/3** but **0.7–2.1 for M=4/5/6**; adding an exact rejected-slot ring restore makes it exactly
  **0.0**.
- This matches the γ-round pattern: the two γ3 arms are byte-identical (both carry the same
  deterministic contamination), yet each γ's stream is shifted by its own M-dependent contamination.
- **This is a candidate H3 mechanism, not a fix.** If the live dumps confirm the A/B collapse, the
  fix sketch is: in `spec.rollback`, restore the rejected rows' window-ring slots (or fence the ring
  write to committed positions). It could also re-price the γ economics (the per-round-time growth
  81→111 ms tracks M; a fix has its own pre-registered change).

---

## 4. LIVE RESULTS — PENDING (frozen; awaiting a new orchestrator GO)

Filled in after the eval boot + capture run: faithfulness verdict, the per-M first-differing-layer
and max|Δlogit|-under-identical-routing read-out, the residual-delta curves, the rollback A/B
collapse, the noise floor + control arms, the H1/H2/H3 verdict, and the raw dumps under
`raw/pricing/q2/fu1/`.

---

## 5. END STATE / BUDGET ACCOUNTING — PENDING

Production restored & live (if a boot is used); canary healthy; no strays; artifacts committed to
`deploy/phase20-campaign`; `PERFORMANCE_HISTORY.md` same-turn entry on `origin/main`.

---

## 6. ARTIFACTS (offline, committed)

- Branch `deploy/q2-fu1-capture` @ `898cf36a` (adurham fork): `engine.py` bootstrap, `start_cluster.sh`
  forwarding, `bench/phase20_q2_fu1/{q2fu1_capture.py,q2fu1_driver.py,self_test.py,README.md,RUNBOOK.md}`
  (+ the deep analyzer, same branch).
- Self-test: `SELF-TEST OK (100 checks)`; scoped regression `202 passed` (5 pre-existing mlx-lm
  `test_dsv41_indexer_smallm_hier` failures, reproduced on the pristine pinned checkout — unrelated).
