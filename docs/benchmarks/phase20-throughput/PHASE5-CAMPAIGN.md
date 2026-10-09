# PHASE 5 — continuation campaign (P0 unit reconcile → P1 lever-2 ship-design → P2 dense EXL3 → P3 cluster)

Author: Phase-5 PM (depth-1). Opened 2026-10-08 ~19:20 CDT. **This doc is the resume anchor — commit+push after EVERY step.**

## 0. Entry state (verified by the Phase-5 PM, 2026-10-08 ~19:20 CDT)

- **PRODUCTION NOW**: `deploy/next13 @ 576e9d279` (exo) + mlx-lm `3bf8316` on BOTH nodes — verified live:
  `ssh studio1|studio2 'git rev-parse --short HEAD'` == `576e9d279`, `git -C mlx-lm rev-parse --short HEAD` == `3bf8316`.
- `/state`: 1 MlxJacclInstance, 2 runners, served model `dealignai/DeepSeek-V4.1-Flash-UNCENSORED-EXL3-2.9bpw`.
- Gates UNSET, canary 14.86/14.85 (orchestrator report; RE-CANARY at R1 before any bench). This is the RESTORE target.
- Prior docs (branch `deploy/phase20-campaign`, worktree `/private/tmp/phase20-campaign`):
  `PHASE4-CAMPAIGN.md`, `PHASE4-P5-ROOFLINE.md`, `PHASE3B-SHIP-VALIDATION.md`; `docs/PERFORMANCE_HISTORY.md` on main.
- next18 artifacts confirmed on disk: mlx-lm branch `deploy/next18-lever2 @ 938b811` (worktree `/private/tmp/next18-lever2`,
  suite `tests/test_dsv41_indexer_smallm_hier.py` 2045 cells, harness `bench/next18_capture.py`, shared `_gates.py`);
  exo branch `deploy/next18-identity` (worktree `/private/tmp/next18-exo`, unpushed).
- Real weights live on the nodes only: `studio1:~/.exo/models/dealignai--DeepSeek-V4.1-Flash-UNCENSORED-EXL3-2.9bpw` (196 GB, 39 safetensors).
- **NO PEER AGENTS on this campaign** — a delegate listing matching this goal is the PM itself (this task).

## 1. The round (phases in order; ALL proof work OFFLINE — no relaunch until a suite is clean)

| phase | deliverable | gate | class |
|---|---|---|---|
| **P0** | unit reconciliation doc: passes/round, per-round dense floor, KV share, 52/56 arithmetic check, draft-path pass count, HIER=0-benign plan | numbers self-consistent + all file:line | offline / 0 relaunch |
| **P1** | ship-design lever-2 code (L2-full vs L2-guard) → next18 → OR gate-semantics memo | G0/G1/G1b | offline / 0 relaunch |
| **P2** | dense EXL3 isolated production-shape GEMV on REAL weights → effective GB/s, route pick | go/no-go ≥5 ms | offline bench-only / 0 relaunch |
| **P3** | R1 validation session; R2 ship; R3 reserve-only | G2/G3/G4 | cluster (≤3) |

## 2. Budget / tripwires (declared UP FRONT)

- Total: **≤3 relaunches** (R1 validation, R2 ship, R3 reserve-only). Zero spent so far this round.
- **Every relaunch declared in the ledger BELOW before spending.** No relaunch while any offline suite is red.
- A phase whose proof fails → **do NOT ship**; write it up (speed-vs-output tradeoff); proceed. An abort is a valid result.
- Never exceed the total; a single phase needing >2 unplanned relaunches HALT that phase.

### Relaunch ledger (declare-before-spend)

| # | phase | deploy | purpose | status |
|---|---|---|---|---|
| R1 | P3 | `deploy/next18-identity` (+ mlx-lm next18) | same-session fixed-replay A/B baseline vs clean lever build; battery both arms; optional MoE top-k-hook bench arm; missing HIER=0 BENIGN point | PLANNED (only if P1 suite clean) |
| R2 | P3 | next18 (default-on lever) | ship: retire env var, fresh boot, canary+battery+parity | CONTINGENT on G1/G2/G3 |
| R3 | P3 | restore best SHIPPED | **RESERVE ONLY** — one pre-named retry (e.g. trigger-rate >1% → swap to the other suite-clean variant) | HELD |
| — | P3 | bench-only | MoE top-k hook rides R1 free; HIER=0 benign rides R1 free | PLANNED |

## 3. Gates (PRE-REGISTERED, before the work that is judged by them)

- **G0** — the three source premises documented, each with file:line:
  (1) coarse pass is precision-reduction ONLY — no subsampling/striding (else guarded design is unsound → full-width only);
  (2) the hierarchical path is genuinely exact GIVEN its candidates (if HIER itself relies on the overfetch heuristic,
      NO design passes the existing gate → write the gate-semantics memo instead of shipping);
  (3) tie-break: both paths share the same top-k op.
- **G1** — 0 diffs on ALL suites: synthetic 2045 cells + engineered tie/overfetch-boundary cells + 165k real-tensor
  captures (`bench/next18_capture.py`) at BOTH 20K and 91K contexts; fresh process; run TWICE.
- **G1b** — offline-projected retention ≥15 ms/round (agentic).
- **G2** — agentic Δ ≥15 ms with DISJOINT ranges; benign no regression; trigger telemetry <1 % of small-n calls.
- **G3** — quality battery CLEAN on the lever arm.
- **G4** — fresh-boot post-ship parity smoke within noise.
- **SHIP** if 8–15 ms with mechanism confirmed by slice attribution; **<8 ms or unexplained → NO-SHIP**, document.

### Selection rule (pre-registered, P1)

- **L2-full** (small-n = score the FULL row in fp32, same scoring kernel, same top-k op, no coarse pass, no overfetch —
  trivially exact) is **PREFERRED if projected retention is within ~2–3 ms of L2-guard** (no ε doc, no premise risk).
- **L2-guard** (fp32 rows + coarse pass + per-call bound δ = 2⁻⁸·max|row| derived from bf16-storage rounding of the fp32
  score row; guard = kth-vs-(k+1)th coarse gap > 2δ else full-width escape) only if L2-full is materially slower.
- **REJECTED** — design (iii) 'first-N-calls self-check' (samples the least-divergent regime; auto-disable = silent 29 ms
  regression) as primary; may exist only as always-on trip-wire telemetry. Design (ii) bigger overfetch fails its own
  adversarial suite by construction.

## 4. P0 — unit reconciliation (offline, FIRST)

Produce `PHASE5-P0-UNITS.md`. Establish from source + the P5 doc:
- (i) indexer/score PASSES per decode round at m=1 and per m=4 verify (the suite's cells imply 3×m=1 + 1×m=4 — VERIFY in code,
  draft path AND verify path; note whether the draft path uses the same Indexer module with m=1 forwards).
- (ii) per-ROUND dense bytes/floor: 2.04 GB was per m=4 PASS; with 4 passes/round dense ≈ 8.2 GB/round → floor ≈16.5 ms →
  dense at ~2.1×, ~240 GB/s effective (vs 8.5× if one pass).
- (iii) KV share = 0.148/6.65 = **2.23 %**, NOT 0.2 % (decimal-place error in `PHASE4-P5-ROOFLINE.md` §2.2 note).
- (iv) sanity-check the 52 % / 56 ms vs 27/29 ms arithmetic (confirm & note; 27.0+29.0=56.0).
- (v) fold the MISSING `HIER=0` **benign** data point into R1 (never a dedicated relaunch).
0 relaunches.

## 5. P1 — lever-2 ship design (offline)

Produce `PHASE5-P1-LEVER2.md` verdict. Steps IN ORDER:
0. **G0 source premises** — the design dies here if it dies (each with file:line).
1. **Cheap-first re-analysis** — re-analyze the EXISTING suite results with `ROW_BF16=0` stratified by (n,m); if residual
   divergences concentrate in strata, a narrowed gate may be suite-clean TODAY at zero new machinery. BEFORE building.
2. Build **L2-full** and **L2-guard** in the `/private/tmp/next18-exo` (exo) + next18-lever2 (mlx-lm) worktrees; measure offline.
3. Suite to green (fresh process, TWICE): 2045 synthetic + engineered tie/overfetch cells + 165k real captures @ both 20K & 91K;
   plus per-call cost + trigger-rate/gap-histogram on captured tensors (trigger <1 % is part of G2).
4. If NO variant clean → lever-2 BLOCKED: write the **gate-semantics memo** ('blocked, user decision' — overfetch-heuristic
   gate vs value-identity gate is a USER decision; do NOT unilaterally redefine it) with suite matrix + strata + ε derivation;
   reallocate to P2. Commit code artifacts to the code branches.

## 6. P2 — dense EXL3 (offline, parallel)

Produce `PHASE5-P2-DENSE.md`. After P0 fixes units: the decisive number is an **isolated production-shape dense GEMV/GEMM on
REAL weights** (production shapes, per pass) → effective GB/s. ONE number picks the route:
- reading 1: dense ~240 GB/s effective (4 passes/round) → **bf16 dequant-cache** for dense/shared (eliminates unpack ALU;
  ~5.5× byte expansion — check RAM headroom on 115 GB guardrail; numerics change → battery-gated; est ~9–11 ms/round).
- reading 2: one-pass (8.5× stands) → **GEMV bandwidth/latency tuning + collectives-at-m=1** (up to ~17 ms).
Also free: **kernel-parity audit** (production uses the best microbenched variant; the prior 58 GB/s sweep is suspect until it
reproduces production effective GB/s). **Pre-registered go/no-go**: projected ≥5 ms/round → next-round cluster proposal;
below → write 'feature-blocked' WITH closing evidence. TIMEBOX. Bench-only on real weights on an IDLE node, idle-guarded, servers untouched (0 relaunch).

## 7. P3 — cluster

- **R1** validation session (relaunch #1): canary → battery on baseline → SAME-SESSION A/B baseline vs clean lever build
  (FIXED replay so content divergence can't pollute timing; benign 20K + agentic 91K, ≥3 reps, interleaved) → battery on lever arm →
  optional MoE top-k-hook arm (bench-only; only a clean env-gated one-liner that does not touch the indexer path; routing
  histograms for hot-expert-cache sizing) → **missing HIER=0 benign point**.
- **R2** ship (relaunch #2): default-on lever, retire env var, fresh boot, canary + battery + parity smoke within noise, known-good tag.
- **R3** reserve only (relaunch #3): one pre-named retry. Nothing else.
- MoE expert differential: NO dedicated relaunch (attribution-only unless the hook rides R1 free).

## 7. GATE DECISION — AMENDMENT RATIFIED (2026-10-08 ~21:00 CDT); R1→R2→R3 EXECUTE

The plan owner (Fable adjudication, max effort) **RATIFIED the amendment**. Recorded verbatim + with
evidence IDs in `PHASE5-P1-LEVER2.md` §14; the gate is **RE-FROZEN** under it. Key points:
- Gate evaluated at the served checkpoint's actual `H`, re-verified from `config.json` at every
  promotion. **Production H = 32** (memo's H=64 corrected; 0 diffs at H∈{8,32,64}).
- "Ties count as divergence" retained in full at production H; small-H cohort accepted via per-diff-slot
  attribution; one-amendment rule; exact phrasing fixed.
- Added R1 legs: end-to-end greedy token-identity diff WITH a prod-vs-prod determinism CONTROL; adversarial
  cells at production H with ROW-LEVEL bitwise asserts; 91K capture replay = HARD R2 precondition; per-cell
  attribution artifact. Pre-registered failure branches + R2 gates in the memo §14.

### Execution ledger (declare-before-spend)

| # | phase | deploy | purpose | status |
|---|---|---|---|---|
| R1 | P3 | exo `576e9d279` + mlx-lm `deploy/next18-lever2 @ cd68bf4` | fixed-replay A/B baseline-vs-lever; 91K capture replay; greedy token-identity + control; adversarial cells; battery both arms; `HIER=0` benign | **ABORTED — capture flush starved the runner (SIGKILL); restored. See `PHASE5-R1-RESULTS.md`** |
| R2 | P3 | next18 (L2-full default-on) | ship: retire env, fresh boot, canary+battery+parity, tag | CONTINGENT on R1 gates — **HARD-BLOCKED** |
| R3 | P3 | restore best SHIPPED | **RESERVE ONLY** — one pre-named retry | HELD |

**Budget: ≤3 relaunches total. R1 = 1/3. RESTORE = 2/3. One relaunch remains.**

> **R1 outcome (2026-10-08):** R1 deployed `deploy/next18-identity @ ff676b3ca` + mlx-lm `17bbd98`
> with the 91K capture ON; READY + installed-module/env verified on both nodes; then the
> first agentic rep **SIGKILLed the runner** — the capture hook's `np.savez_compressed` flush
> (~197 MB) blocks the server event loop in zlib deflate, tripping the 45 s hang-watchdog.
> Baseline arm + prod-vs-prod token-identity CONTROL completed (deterministic); lever arm,
> battery, offline replay and token-diff did NOT. Production **RESTORED** to
> `576e9d279`/`3bf8316`. Full detail + evidence: `docs/benchmarks/phase20-throughput/PHASE5-R1-RESULTS.md`.

**Blockers known before R1 (from `PHASE5-R1-KIT.md` §J):** (1) the
91K capture env `DSV41_NEXT18_CAPTURE*` is NOT forwarded by `start_cluster.sh` — **RESOLVED**: launcher
patch landed at `deploy/next18-identity @ ff676b3ca` (env forwarded + env-gated import in
`dsv41/engine.py`, verified inert when unset); (2) `deploy/next18-identity` unpushed → now pushed, R1
deploys exo `576e9d279` + mlx-lm `deploy/next18-lever2 @ 17bbd98`.

**Offline prep (all landed + pushed):** `PHASE5-R1-KIT.md` @ `230ba71`; adversarial suite
`tests/test_dsv41_indexer_adversarial_prodh.py` @ mlx-lm `17bbd98` (**45/45 identity, 0 row-bitwise
mismatch, 0 ulp flips at H∈{8,32,64}`); attribution artifact `scratch/p5/prep/attribution_251.json`;
91K fp32-row cost `+0.005 ms/call → ≈0.04 ms/round`.

**⚠ req-3 DEVIATION (owner action needed — R2 hard-blocked):** the ratified req-3 ("every diff in an
exact-zero column") is **NOT MET as worded** → recorded **FAILED-as-worded, pending owner
reconciliation** in `PHASE5-REQ3-DEVIATION.md`. Of 22,740 diff slots: 10,406 exact-zero-column +
12,288 masked/-inf padding (value-null) + **46 non-zero-column 1-ulp** diffs in one cell; superset check
TRUE. Intent holds (L2-full loses 0 vs shipped 48 value slots in that cell), but a verbatim requirement
is not satisfied by intent. **A draft SECOND AMENDMENT (slot taxonomy a/b/c + precision-dominance) is in
`PHASE5-REQ3-DEVIATION.md` §4 for owner sign-off.** R1 proceeds (non-shipping, production-H, required for
R2); **R2 requires explicit re-ratification.**

**PM-verified reproductions (independent of the children):**
- Reproduced: suite fresh → 251 divergent (shipped: 939); `next18_prodgate.py` → census `2^-H` law +
  stability H∈{8,32,64} = **0** + determinism **0/0/0**; `next18_classify_all.py` → fb 0 / hier 48 value-loss
  slots, max |fb−hier| 9.73e-6.
- P2 raw JSON verified: real trellis **k=5 → 5.0 bpw**; m=1 = **69.3 GB/s**, m=4 = **53.4 GB/s**; kernel-parity
  `prod ≡ fused` on every shape.

## 9. R1 RESULT — ABORTED (harness bug, NOT a lever failure); cluster RESTORED. One relaunch remains.

Full report: `PHASE5-R1-RESULTS.md` @ `f5c474748`. Verdict: R1 **aborted mid-session** — the lever arm
could not be measured. Root cause: the 91K capture hook's `np.savez_compressed` (~197 MB .npz) ran
**inline on the server's request thread**; the multi-minute zlib `deflate` emitted no runner events →
the 45 s hang-watchdog SIGKILLed the runner on both nodes (`/tmp/exo_hang_*.txt`, `verdict=kill`,
main thread in `zlib_Compress_compress`). **No lever timing, no capture.** This is a harness/tooling
defect, **not** evidence about the lever.

- **Baseline (live prod boot, 0 budget, PASS):** agentic 91K **130.38 ms**/24.09 t/s (reproduces the
  frozen next17 anchor 130.07); benign 20K **118.86 ms**/31.22 t/s (vs 118.56). Leg-A **prod-vs-prod
  determinism control = 0 diff** (text sha `e65639ce…`), prod reference captured.
- **Lever numbers: N/A.** Leg C (91K replay): **NOT satisfied**. Battery: **not run**. Trigger: n/a.
- **RESTORE (PASS):** `deploy/next13 @ 576e9d279` + mlx-lm `3bf8316`, gates unset, canary 14.85/14.86,
  parity smoke benign **118.21 ms** (Δ0.65, within noise), token-identical to pre-R1 prod.
- **Budget after R1: 2/3** (R1 relaunch #1 + restore #2). **One relaunch remains.**

**R1b = the final relaunch (charge R3-reserve), SHIP-IN-PLACE on pass.** Fix = run the harness flush
**off the request thread** (a bench-harness-only change, committed + staged on the nodes) and keep capture
OFF during the timing arms conceptually by making the flush non-blocking; then lever timing → leg A
(token-identity vs `e65639ce…`) → 91K capture replay (HARD) → battery (G3) → **if all clean: default-on
verified, tag `known-good`, SHIPPED**; else restore (safety-mandated). Gates G2/G3/legs A-C all still
pending.

## 10. End state / resume pointer

End: best SHIPPED build live, gates unset, canary healthy, RESTORED/SHIPPED line + parity smoke in doc, no stray processes,
worktrees cleaned or clearly parked. `docs/PERFORMANCE_HISTORY.md` updated same-turn per finding on main; `known-good-*` tagged
on both forks on ship. Last commit on this doc says where we are; resume from the first unchecked box.

## 11. R1c FINAL SHIP ROUND (2026-10-08/09) — SHIPPED-IN-PLACE

**Owner ruling (verbatim, 2026-10-08):** *"as long as we pass our quality tests then I'm fine with it."*
→ the **R8a quality battery is the governing ship gate** for lever-2; the strict identity gate is
superseded; the token-63 output change is accepted. Full report: `PHASE5-R1C-SHIP.md`.

- **Production artifact:** exo `deploy/next18-identity @ fb4f9290b` (gitlink bumped `3bf8316` →
  `16830e1`), pushed to `origin` (adurham/exo). mlx-lm `deploy/next18-lever2 @ 16830e1` (pushed).
- **Deploy:** exo `fb4f9290b` + mlx-lm WT `16830e1`, ALL lever/capture env unset; READY (2/2);
  canary 14.84/14.84. Installed-module verify PASS both nodes; runner env has no `DSV41_*` keys.
- **Battery (governing):** **CLEAN** — needles 6/6, tools 10/10, prose 0 DIRTY / 0 REVIEW, park PASS;
  `compare.py` vs frozen `g3` → no fails.
- **SHIPPED-IN-PLACE:** build left LIVE as production, gates unset. Tag
  `known-good-decode-next18-20261009-001052` on both forks. Parity smoke benign 20K **94.94 ms** /
  agentic 91K **99.46 ms** (reproduces R1b).
- **Open item:** token-63 divergence root cause NOT fully explained (harness replayed 64/128 records).

### R1c relaunch budget (declared ≤2; declare-before-spend)

| # | purpose | deploy | declared | status |
|---|---|---|---|---|
| 1 | SHIP deploy (gates unset) | exo `fb4f9290b` + mlx-lm `16830e1` | ≤2 | **SPENT** (1/2) |
| 2 | contingency RESTORE (only on DIRTY) | exo `576e9d279` + mlx-lm `3bf8316` | ≤2 | **HELD — not spent** (battery CLEAN) |

### Cluster end state (final)

exo `deploy/next18-identity @ fb4f9290b` + mlx-lm `16830e1`, both nodes, **gates unset**, canary
**14.85/14.85** healthy, 2 runners RunnerReady, no stray bench processes. `known-good` tag created
on both forks.
