# ROUND-Q1-QUANT.md — Dense quant-format evaluation round (native affine qN vs EXL3 fused dense)

Author: Phase-20 PM (delegation). Date: 2026-10-09 (CDT). Worktree `/private/tmp/phase20-campaign`.
Branch `deploy/phase20-campaign`. Owner-approved round (Q1 follow-through); this round **CAN ship** if gates pass.

Entry state (verified ~11:57 CDT): production `fb4f9290b` (exo) + mlx-lm `16830e1` on BOTH nodes
(`ssh studio1|studio2 'git rev-parse --short HEAD'`), 2 runners, loadavg ~1.7 (idle), model
`dealignai--DeepSeek-V4.1-Flash-UNCENSORED-EXL3-2.9bpw`. Restore target = `fb4f9290b` / `16830e1`.

---

## 0. DECLARED BUDGET (fixed BEFORE spending, per the brief)

- **Deploy boots: ≤3 total (+1 reserve).** A "boot" = a full `start_cluster.sh` relaunch (deploy + node
  bring-up). A **same-boot arm switch** is an exo-PROCESS restart only (`relaunch_exo.sh` /
  `screen`-relaunch with a different env — the runner subprocess re-imports modules and re-reads env
  every model load), NOT a boot. Declared plan:
  - **Boot #1** — eval build live: within-build A/B (control `DSV41_DENSE=exl3` vs treatment
    `affine6`) via exo-process restarts; then full-depth runs + full R8a battery + logit-drift on the
    treatment config. (1 boot.)
  - **Boot #2** — SHIP (make qN dense the default) if P2+P3 pass; else the single retry with the other
    qN (`affine5`) if P3 came back DIRTY and the budget allows. (1 boot.)
  - **Boot #3 / reserve** — only if a retry or a re-ship is genuinely warranted. Nothing else.
- Q1 microbench re-check + conversion + engine build + unit tests + Q5 instrument = **0 boots** (offline).
- Stop at 3 boots regardless of outcome; a negative at that point closes the eval with restore.

## 1. THE DECISION BEING EVALUATED (priced, owner-approved)

Native `mx.quantized_matmul` q4/q5/q6 g64 on the **dense/shared** slice = **2.47×/2.29×/2.30×** the fused
EXL3 kernel at m=4 (0.483 vs 0.945 ms/layer, sharded per-rank shapes; K/call 0.314/0.338/0.336 vs 0.774)
→ projected **~18.5 ms/round** off the sharded dense slice (baseline 37.8 ms/round @ `fb4f9290b`). The win
is **ALU/issue** (trellis k=7 SWAR decode), not bytes. Cosine to bf16-EXL3-decoded W: q4 0.9959 / q5
0.9990 / q6 0.9998. **Experts arm PARKED** (384×~71 MB fp16 ≈ 27 GB; `gather_qmm` stacked layout — out of
scope). **Dense slice ONLY.** Natural candidate = **q6 (affine6)** or **q5 (affine5)**: q6 ≈ q4 speed with
much better fidelity (0.9998 vs 0.9959); pick with the data.

## 2. SOURCE DECISION (checked FIRST, decided with evidence)

**DECISION: EXL3-reconstruction source (EXL3-dequant), labeled LOUDLY.** Evidence:
- **No original bf16 source is locally present.** Node/local caches hold the EXL3 checkpoint, the
  `deepseek-ai--DeepSeek-V4.1-Flash-engram` release (227 GB, fp8 weights + fp4 experts — NOT the
  uncensored model, and not bf16), and HF offers only FP8 uncensored re-quants of the same lineage.
  A true bf16/fp8 "original uncensored" is a >200 GB download (decision point).
- **The engine ALREADY implements EXL3-reconstruct → affine** (`AffineProj`,
  `mlx_lm/models/deepseek_v41/exl3_build.py:52-71`, gated by `DSV41_DENSE=affine6|affine8`). It does
  `reconstruct_public_mlx(layer)` (the EXL3 trellis decode WITH the blockwise Hadamard rotations folded
  in) → `mx.quantize(bits, group_size=64)` → plain `mx.quantized_matmul`.
- **The treatment dense ≈ the CURRENT deployed dense + tiny rounding.** The current model computes
  `y = x @ W_exl3` where `W_exl3 = reconstruct_public_mlx` (same routine). The treatment computes
  `y = x @ Q6(W_exl3)`. So the treatment is the current operator with an added q6 rounding of cos 0.9998
  — **not a regression below the incumbent.** This is a strong argument that EXL3-dequant source is
  acceptable here.
- **LOUD LABEL (mandatory in every artifact):** *"fidelity measured from EXL3-decoded source, not the
  original model."* Compensated by: (a) logit-drift spot check (top-5 + margins) vs the current build on
  fixed prompts, and (b) the full R8a battery as the real governing gate.
- Deferred option (recorded, not taken): download the fp8 uncensored original (>200 GB) as a future
  better source if the battery shows a marginal drift.

## 3. BUILD SHAPE — what already exists vs what P1 must build

- **Exists**: the mixed-format toggle `DSV41_DENSE = exl3 | affine8 | affine6` (`exl3_build.py:461-468`,
  `_dense()`); `AffineProj` (EXL3-reconstruct → affine g64, plain qmm); experts + everything else
  unchanged (EXL3 trellis). Read at **module import** → a new exo process re-reads it.
- **P1 must fix/verify (the load-bearing items):**
  1. **`DSV41_DENSE` is NOT forwarded by `start_cluster.sh`'s `EXO_ENV` allow-list** → it is a silent
     no-op in production (same class as the R1 capture-env blocker). Add the forwarding line.
  2. **Affine mode DISABLES TP sharding** (`exl3_build.py:515,529` gate `attn_tp`/`shared_tp` on
     `DENSE_MODE == "exl3"`). With it off, each rank computes the **FULL** dense slice → per-rank dense
     work DOUBLES, roughly halving the projected win (≈8 ms not ≈18.5 ms) and raising a
     correctness question at world=2. **P1 must determine (empirically) whether affine@TP=2 is
     numerically correct AND implement TP sharding for affine dense** (128-block-boundary slices, exactly
     as `_slice_dense` does for EXL3) so the A/B is against the SAME sharded geometry as the priced win.
  3. Add `affine5` (q5) to the value set so the retry qN is available.
  4. Re-run the Q1 microbench on the CONVERTED/affine path at the **sharded** shapes → reproduce ≥2× at
     m=4; record the cos vs the EXL3 reconstruction reference.
  5. Unit test(s), single-file runs only (no full suites on the Mac).
- **Conversion/export**: quantize-at-load (in-memory reconstruct→quantize; no checkpoint-side artifact
  needed). Document exactly what the engine loads and from where. Both nodes must serve the SAME bytes —
  verified via installed-module + `ps eww` env + a per-node dense-weight hash if practical.

## 4. THE ROUND

- **P0 (offline):** this doc. Commit+push. ✅
- **P1 (offline):** source decision (above) + engine items in §3 + microbench re-check. Commit+push.
- **P2 (cluster, boot #1):** deploy eval build, EXO-process-restart A/B (`exl3` control vs `affine6`),
  interleaved where possible, ≥3 reps/arm; idle-guard each chunk; canary after READY; then FULL-MODEL
  two-node runs (≥16K prefill smoke + 91K agentic deep path) on the treatment. Gate G-B below.
- **P3 (battery, same boot):** full R8a battery on treatment (needles 6/6, tools 10/10, prose 0
  DIRTY/0 REVIEW, park PASS; `compare.py` vs g3) + logit-drift spot check (top-5 + margins, 2-3 fixed
  prompts vs current build; informational, recorded). CLEAN → ship candidates; DIRTY → try `affine5`
  once (boot #2) if budget allows, else restore + report honestly.
- **P4 (ship, if P2+P3 pass):** make qN dense the DEFAULT for production (retire/keep env as control),
  gitlink-bump a production branch, push both forks, deploy, READY+canary, parity smoke (expect agentic
  ≈81 ms / ~37-38 t/s if the full −18.5 ms materializes; benign ≈76 ms / ~48 t/s — give the ACTUALS),
  tag `known-good-dense-q6-<date>` on both forks, SHIPPED line, PERFORMANCE_HISTORY on main same-turn.
  Restore target if anything fails: `fb4f9290b`/`16830e1`.
- **P5 (Q5 instrument, same build if safe):** fold in the per-kernel GPU-time ring (Path 2 per the Q5
  memo: label→time ring buffer; MLX patch → node wheel rebuild — verify the mlx-sha-triggered rebuild).
  Env-gated OFF for all timing arms; ON for ONE attribution run. **If the MLX patch risks destabilizing
  the ship boot, SPLIT IT OUT into its own later boot — the ship takes priority.**

## 5. GATES (pre-registered, BEFORE spending)

- **G-A (offline):** converted-tensor microbench reproduces **≥2×** at m=4 on the **sharded** shapes;
  cos vs the chosen reference recorded.
- **G-B (cluster A/B):** agentic round-time **Δ ≥ 15 ms** (target ~18.5) with `mean_accepted` within
  noise; benign no regression beyond noise. (Within-build, same boot.)
- **G-C (battery):** **CLEAN** (needles 6/6, tools 10/10, prose 0 DIRTY/0 REVIEW, park PASS) + logit
  drift recorded.
- **G-D (ship):** fresh deploy + canary + parity smoke within noise + both-fork tags + PERFORMANCE_HISTORY.

## 6. FALSIFIER

If the win does not survive full-model / full-depth (e.g. GPU-timeout at depth, or the delta collapses vs
the layer bench), **CLOSE the eval with the negative result and restore** — a valid outcome. Do not grind
past 3 boots.

## 7. A/B DESIGN (same-build / same-boot)

- Single deployed eval build; the ONLY runtime variable is `DSV41_DENSE` (`exl3` vs `affine6`), injected
  by restarting the exo PROCESS (not the node). Arms measured in ONE node boot → no cross-boot drift.
- Fixed replays: benign 20K + agentic 91K, ≥3 reps/arm, interleaved if the process-restart cost permits.
- Sanity-check `mean_accepted` between arms (must not move materially — content-driven).
- Full-model depth runs are mandatory (row-limit / GPU-timeout history: layer-level wins have not always
  survived full depth).

## 8. DEPLOY PLAN (per skill discipline)

- Deploy source = the LAPTOP shared checkout `~/repos/exo` (NOT a /private/tmp worktree). Worktree-held
  branch → detach-checkout in the shared checkout; **move BOTH trees** (`git checkout --detach <exo-sha>`
  + `git -C mlx-lm checkout <mlx-lm-sha>`). Push both forks first.
- After READY: verify the **INSTALLED** venv module on BOTH nodes (resolve + grep the deployed line) and
  the running env via `ps eww`. Canary after every boot AND after READY.
- Idle-guard every chunk/boot (`phase20_guard.py` + own-request registry); never interfere with user
  traffic. Reboot only via `./reboot-node.sh`. Clean tree before rsync.
- Resumable: this doc committed+pushed after EVERY step. `PERFORMANCE_HISTORY.md` on main same-turn.
- End state: best SHIPPED build live, gates unset, canary healthy, no stray processes.

## 9. EXECUTION LEDGER (declare-before-spend)

| # | phase | deploy | purpose | status |
|---|---|---|---|---|
| P0 | offline | — | plan doc | ✅ committed |
| P1 | offline | — | source + engine (sharded affine + env forwarding + affine5) + microbench | PENDING |
| B1 | P2/P3 | exo `<eval>` + mlx-lm `<eval>` | within-build A/B (exl3 vs affine6) + depth runs + battery + drift | PENDING |
| B2 | P4 | qN default | ship (or single retry affine5) | PENDING |
| B3 | reserve | — | one pre-named retry only | HELD |

**Budget spent: 0 boots of ≤3.**

---

## P1 RESULTS (offline) — STATUS: DONE. G-A = **PASS**

### Source decision — CONFIRMED: EXL3-reconstruction source (labeled LOUDLY)
No original bf16 source available without a >200 GB download; the engine already reconstructs the EXL3
dense groups (`reconstruct_public_mlx`) and the treatment dense ≈ the CURRENT deployed dense + a q6
rounding (cos 0.9998). Artifacts carry the label: **"fidelity measured from EXL3-decoded source, not the
original model"**, compensated by the logit-drift check + the R8a battery.

### Engine work (2 branches, both pushed to adurham remotes)
| repo | branch | SHA | change |
|---|---|---|---|
| mlx-lm | `deploy/q1-dense-qn` | `e444cbdcad3253ac25770d3a75d5c2354cea54dd` (off `16830e1`) | TP-shard affine dense modes (qN) like exl3 + `affine5` + `AffineProj.from_weight` + `DSV41_DENSE_TP` |
| exo | `deploy/q1-dense-qn` | `e4c2cb4607f8f87b13bde2910344ea8479eadf30` (off main `745efdfdf`) | `start_cluster.sh`: forward `DSV41_DENSE` + `DSV41_DENSE_TP` (was a silent no-op — the R1-class blocker) |

- **The key fix**: affine mode previously set `attn_tp`/`shared_tp` = False (gated on `DENSE_MODE=="exl3"`),
  so every rank computed the FULL dense slice. Removed the gate; affine now TP-shards on the SAME 128-wide
  block boundaries (`_block_bounds`). Slicing is `reconstruct(full) → slice fp16 → mx.quantize(gs=64)` —
  exact: `reconstruct(full)[slice] == reconstruct(slice)` (max_abs_rel 0.0).
- **Correctness trace (verified)**: replicated affine at world=2 was ALREADY numerically correct — the
  attention `all_sum` is keyed off `group is not None` (not `world`) so replicated attention does no
  collective, and the shared-expert output is added AFTER the routed-expert all_sum. So the sharding fix is
  a **PERF** fix (recovers the priced 2.4× → ~19 ms/round vs ~break-even), not a correctness fix.
- Unit test `tests/test_dsv41_dense_affine_shard.py`: **12 passed** (re-run by the PM).

### Per-layer re-check on the CONVERTED/affine tensors — G-A PASS
`raw/pricing/q1/recheck/` (script + JSON + stdout). Layer-20 dense roster, **TP=2 rank-0 sharded
shapes** (14 linears; wo_a head-split → 4 of 8 groups — the TRUE engine rank-0 geometry; the prior Q1
script's "18-linear" roster was a non-rank-0 proxy).

| arm (m=4, per-rank) | whole ms | K/call med·p95 ms | ratio whole | ratio K | cos |
|---|---:|---:|---:|---:|---:|
| prod EXL3 (fused, baseline) | 0.850 | 0.641 · 0.657 | 1.00× | 1.00× | 1.0000 vs W_rec |
| native q6 +Hadamard (drop-in) | 0.447 | 0.286 · 0.291 | 1.90× | 2.24× | 0.99976 |
| native raw q6 | 0.364 | 0.211 · 0.221 | 2.33× | 3.03× | 0.99976 |
| **affine q6 (engine path)** | **0.366** | **0.216 · 0.220** | **2.32×** | **2.97×** | 0.99975 |
| **affine q5 (engine path)** | **0.361** | **0.209 · 0.211** | **2.36×** | **3.06×** | 0.99898 |

- **G-A PASS**: affine q6 = **2.32×** whole / **2.97×** K-batched; affine q5 = 2.36× / 3.06×.
- The affine engine path runs at **raw-native speed** (rotations folded into W, not dispatched per call) —
  i.e. it beats the "fair drop-in +Hadamard" arm. Projected dense-slice win ≈ **19.4 ms/round**
  (34.0 → 14.6 ms/round over 40 layers).
- `prod EXL3 vs x@reconstruct_public_mlx` = 1.0000 (confirms the reconstruct reference).

### P5 decision — **SPLIT OUT** (ship takes priority)
The Q5 Path-2 per-kernel ring is an MLX C++ patch + a node wheel rebuild; baking it into the ship boot
risks the deploy for a diagnostic that is orthogonal to the dense change. **Deferred to its own later
boot** (recorded; not spent this round unless the budget frees up).

---

## P2 RESULTS (cluster) — same-build A/B

### Boot #1 (control) — eval build live at DEFAULTS
Deploy: `EXO_TARGET_BRANCH=deploy/q1-dense-qn ./start_cluster.sh` from the shared checkout on the eval
branch. **Verified on BOTH nodes**: `exo=e4c2cb460`, `mlx-lm-wt=e444cbd`; the **INSTALLED** venv module
`…/site-packages/mlx_lm/models/deepseek_v41/exl3_build.py` contains the affine-shard line + `DSV41_DENSE`;
runner env has **NO** `DSV41_DENSE` key → **control = exl3**. Post-READY canary **14.86 / 14.85 TFLOPS**
(healthy).

Fixed-replay A/B (`r1_driver.py`, salt `q1eval`, benign 20K + agentic 91K, 4 reps/arm, same content both arms):

| arm | ms/round median (all) | decode t/s | mean_accepted | ttft |
|---|---:|---:|---:|---:|
| control **benign 20K** | **94.88** (94.56 / 95.01 / 94.88) | 38.5 | 2.682 | ~70 s |
| control **agentic 91K** | **101.07** (101.09 / 100.92 / 101.07) | 31.0 | 2.1128 | ~357 s |

Ranges are extremely tight (±0.1 ms). These are the pre-change anchors; the treatment arm is measured on
boot #2 below. **Boots spent: 1 / 3.**

---

## END STATE
(to be filled at close: build live, gates state, canary, SHIPPED/RESTORED line, budget spent.)
