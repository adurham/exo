# phase20_q2_fu1 — Round Q2-FU1 verify-shape capture instrumentation

**Status: build-only artifact. NOT RUN live.** The cluster was touched only by
read-only ssh probes while building this. The PM executes the live run (see
`RUNBOOK.md`).

Round Q2-Gamma found that changing the speculative draft depth `gamma`
(verify rows M = gamma+1) changes decoded argmaxes at CONFIDENT positions
(top1-top2 margins 0.28–1.03 nats over 65–97% of positions) while the two
same-gamma arms were byte-identical.  This round instruments the engine to
explain WHY, across:

* **H1** MoE router top-k amplification (benign): small upstream delta flips a
  near-tie router choice ⇒ different expert ⇒ ~1 nat logit move.  Read-out:
  small |Δlogit| UNDER IDENTICAL ROUTING.
* **H2** kernel-path/precision discontinuity across M (gemv vs gemm,
  accumulation order).  Read-out: large |Δlogit| under identical routing.
* **H3** semantic attention/KV/indexer non-equivalence (in-place rotating-KV
  writes, pool flushes, indexer reduction order).  Read-out: large |Δlogit|
  under identical routing, or a rollback-component signature.

## Files

| file | role |
|---|---|
| `q2fu1_capture.py` | the operator-staged in-process hook (monkeypatches Model/Block/Gate/ShardedHead + the exo engine request path at RUNTIME; mlx-lm source untouched). Must be scp'd to each node's `/tmp/q2fu1/`; imported by the engine bootstrap only when `DSV41_Q2FU1_CAPTURE` names an output dir. |
| `q2fu1_driver.py` | the live driver: arm switch (verified gamma recipe) → capture requests → scp pull (bit-exact) → rank merge → gates + analysis report. |
| `self_test.py` | the OFFLINE self-test (tiny real CPU model through the real `Dsv41Engine` request path): bootstrap gating, monkeypatch firing, pinned dump format, faithfulness (armed stream == unarmed stream), sweep integrity, cross-shape read-out, the rollback A/B discrimination, the speculative ms/rollback round path (real DSparkHead: probes must be stream-neutral), driver merge/analyze/gate logic. 100 checks. |
| `q2fu1_analyze.py` | OFFLINE deep-analysis CLI (`--capdir <dir> --out <dir>` → `q2fu1_analysis.json` + `q2fu1_analysis.md`): the full reference-spec read-out — (a) first differing MoE-router layer + first residual-delta exceedance, (b) routing-identical vs routing-differed \|Δlogit\| buckets (the H1-vs-H2/H3 discriminator), (c) per-layer residual-delta curves (row0 + max-over-rows), (d) router margins at flips, (e) explicit H1/H2/H3 classification + the rollback A/B H3 test, (f) noise floor + allsum/fence arms. numpy only; reuses the driver's `_load`/`merge_all`. |
| `test_q2fu1_analyze.py` | OFFLINE self-test for the analyzer: synthetic merged-npz fixtures (D=8, V=16, 4 layers, 6 rows) encoding each signature; asserts the first-differing-layer, bucket split, step detection, H1/H2/H3 classification, the rollback B-collapse, NaN rank-slice masking, single-rank fallback, missing-file handling, and the CLI outputs. 10 test fns / 44 assertions; numpy only (runs under the system python3, no venv/mlx). |
| `README.md` / `RUNBOOK.md` | this file + the exact node-staging / eval-boot / restore sequence. |

## Engine wiring (the diff on this branch)

* `src/exo/worker/engines/mlx/dsv41/engine.py`: new
  `_maybe_install_q2fu1_capture()` (exact mirror of the existing next18
  bootstrap) called from `Dsv41Engine.__post_init__` alongside it.  Env unset
  ⇒ immediate return; no import, no monkeypatch, zero cost — the production
  path is byte-for-byte unchanged.
* `start_cluster.sh`: forwards `DSV41_Q2FU1_CAPTURE` (+`_MS`/`_ROUNDS`/`_DMAX`/
  `_KSWEEP`) exactly like the `DSV41_NEXT18_CAPTURE*` block, deriving
  `PYTHONPATH` from the capture dir's PARENT when `EXO_PYTHONPATH_DIAG` is
  unset (one-var operation: stage the hook at `/tmp/q2fu1/q2fu1_capture.py`,
  use `/tmp/q2fu1/out` as the capture dir).

## The capture program (request-driven: `correlation_id` starts with `q2fu1`)

No reboot between arms: the hook is installed once at engine construction; a
request whose `correlation_id` carries the spec activates it.  See the hook's
module docstring for the full grammar and the exact arm list.  Summary:

* `sweep=1` (one request): at the post-prefill anchor, an M=1 sequential greedy
  chain K=6 (each step a single-row forward = the M=1 reference at that
  absolute position) is derived, then SINGLE forwards of `rows[:M]` for
  M ∈ {1,3,4,5,6} (plus an M=4 repeat) run from EXACTLY restored cache bytes
  (full snapshot: window rings, compressor carries + chunk stashes, comp/index
  rows in group range, engram slice, offset, draft windows).  Also: bf16-vs-fp32
  MoE allreduce arm (runtime flip of `moe._MOE_ALLSUM_BF16`), fence on/off arm
  (both `_FENCE_MIN_ROWS` copies patched to 0), rollback A/B arm (verify rows →
  production `spec.rollback` at accepted=1 → A: production; B: + exact restore
  of ONLY the rejected rows' ring slots), and a final integrity M=1 + cache
  digest check.  The cache is left exactly as production would leave it, so the
  request decodes normally afterwards.
* `ms=N` / `rollback=1`: passive capture of the first N real verify rounds and
  A/B probes inside the first `rounds` rounds of a REAL agentic request.
* `nospec=1`: pure-greedy reference stream (engine.speculative False for that
  request only).

## Pinned dump format

One npz per captured forward: `<arm>_M<M>_<tag>.npz` with keys
`M, start_pos, gamma, round_idx, arm, rank, vocab_lo, vocab_hi,
n_captured_layers` (scalars); `rows_ids` int32[M]; `logits` float32[M, V]
(this rank's vocab slice filled from `ShardedHead.local` pre-all_sum; other
columns NaN); `hidden_in`/`hidden_out` float32[M, D]; per body layer L:
`idx_L` int32[M, topk], `w_L` float32[M, topk], `hid_L` float32[M, D],
`margin_L` float32[M] (k-th minus (k+1)-th biased router score, recomputed at
dump time post-eval).  Additive keys: `out_ids` int32[M] (the argmax stream of
that forward), `rows_txt`, `flags`, `dtype_logits`, `note`.  The driver merges
the two ranks' logits halves and asserts every other key is BITWISE equal
across ranks.

`hidden_*`/`hid_L` are hc-mean summaries of the [b, n, hc_mult, dim]
hyper-connection residual stream (the quantity the DSpark taps carry) — the
full 4×5120 stack is deliberately not stored (memory).  Documented deviation
from the letter of the spec.

## Faithfulness discipline

* Wrappers only HOLD PYTHON REFERENCES to tensors the production forward already
  computes; no mx.eval/sync/.item/astype inside a forward, no op reordering, no
  return-value changes, no extra command-buffer boundaries; numpy conversion
  happens after the engine's own eval; router margins are a post-eval side
  computation.
* The driver's faithfulness gate: the `gamma3` multi-step request must
  reproduce `arm_gamma3a.json` exactly (first 256 tokens + top-2 logprobs) and
  `gamma4` must reproduce `arm_gamma4.json` (divergence at idx 14/12).  A
  mismatch is reported LOUDLY: the dumps are then untrustworthy.

## Offline self-test

```bash
cd <worktree>
PYTHONPATH=bench/phase20_q2_fu1 EXO_DASHBOARD_DIR=nonexistent \\
  /Users/adam.durham/repos/exo/.venv/bin/python bench/phase20_q2_fu1/self_test.py
# -> SELF-TEST OK (100 checks)
```

Then the analyzer's own OFFLINE self-test (numpy only, no venv/mlx needed):

```bash
cd <worktree>
PYTHONPATH=bench/phase20_q2_fu1 /usr/bin/python3 \
  bench/phase20_q2_fu1/test_q2fu1_analyze.py
# -> SELF-TEST OK (10/10 test fns, 44 assertions)   [or: pytest ... -q -> 10 passed]
```

And the deep-analysis read-out on a pulled capture dir:

```bash
PYTHONPATH=bench/phase20_q2_fu1 /usr/bin/python3 \
  bench/phase20_q2_fu1/q2fu1_analyze.py --capdir <captures> --out <analysis>
# -> q2fu1_analysis.json + q2fu1_analysis.md (per-M read-out + H1/H2/H3 verdict)
```

Also run (no regression): 

```bash
PYTHONPATH=src:mlx-lm EXO_DASHBOARD_DIR=nonexistent \\
  /Users/adam.durham/repos/exo/.venv/bin/python -m pytest \\
  src/exo/worker/engines/mlx/dsv41/tests/ -q -p no:cacheprovider   # 202 passed
```

(The five failures in `mlx-lm/tests/test_dsv41_indexer_smallm_hier.py` are
PRE-EXISTING on the pristine checkout at the pinned mlx-lm commit — reproduced
identically without this change; that file is order/env-sensitive.)

## Offline finding (motivating the H3 arm)

`spec.rollback` restores the compressor carry but not the sliding-window ring
(`win_kv`, slot = pos mod 128).  A verify forward writes slots for ALL M rows;
rejected rows q ≥ target leave their KV in slot q mod 128, which aliases the
live position q-128 — read by the NEXT forward's window.  Whenever
rejected ≥ 2 (M-1-accepted ≥ 2) and the aliased position is in range, the
continuation is contaminated.  With accepted=1: gamma=2 (M=3) is clean
(the single rejected slot aliases a dead position); gamma=3/4/5 (M=4/5/6) are
contaminated — matching the Q2-Gamma pattern (γ3 arms identical to each other
yet each γ's stream shifted by its own M-dependent contamination).

Offline probes (tiny real model, CPU): production rollback leaves next-forward
max|Δlogit| = 0 for M=2/3 but 0.7–2.1 for M=4/5/6; adding an exact
rejected-slot ring restore makes it exactly 0.0.  The capture's `g<g>_A_m1` vs
`g<g>_B_m1` dumps A/B this on the real model with real drafts.

## Uncertainties / open questions (stated, not hidden)

* **H2 vs H1 attribution on Metal**: the offline toy cannot speak to Metal
  kernel selection; the M-sweep dumps + router margins are the real-model
  discriminator.  Add sub-block taps if H2 survives the first read-out.
* **Sweep rows are the greedy chain** (self-consistent drafts), not the real
  drafts; MoE batch composition therefore differs from a production verify.
  The multi-step arm (`ms`, real drafts) covers that case.
* **M=1-vs-M>1 fence leg**: with the default threshold 16 all of M=1..6 sit on
  the same side; the fence arm is a calibration control (0 vs 16), included as
  requested, and is not expected to matter at M ≤ 6.
* The gamma override remains whole-request (the engine never calls
  `GammaPolicy.update`), so a "gamma arm" IS a fixed M throughout a request.
* Ring contamination also affects the *first* forward after each round in
  production; the hook's A/B arm measures it but does NOT fix production — a
  fix must be gated/justified separately (the B arm is the candidate fix's
  evidence, not the fix).
