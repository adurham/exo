# ROUND-Q2-GAMMA — the γ (speculative draft depth) re-price on the post-dense build

Author: Phase-20 PM (delegation). Date: 2026-10-10 (CDT). Worktree `/private/tmp/phase20-campaign`
(branch `deploy/phase20-campaign`). Eval branch `deploy/q2-gamma`.

**Output-preserving class:** speculation is lossless in exact arithmetic; the identity gate (§1a)
handles numerics near-ties. **Branch-only:** the γ code surface lands on the eval branch; the
shipped default is NOT changed this round.

**STATUS: PHASE 0 COMPLETE · PHASE 1 HELD — awaiting explicit per-launch orchestrator GO.**
No eval boot and no cluster relaunch has been executed.

---

## 0. DECLARED BUDGET (fixed BEFORE spending)

- **Boots: ≤ 2** (eval deploy + production restore) **+ 1 reserve**. **Process restarts: ~10–12**
  (≈5 arm switches × 2 nodes, graceful node-process relaunches — NOT reboots). Declared up front.
- **Measurement wall cap: 135 min** (driver default). At that cap the driver pre-declares
  **agentic 91K × 2 reps** and **DROPS the disjoint-IQR criterion** (3 agentic reps do not fit:
  agentic alone 385 s × 3 × 5 = 96.3 min; total 151.4 min > 135). Raising `--budget-wall-min` to
  200 re-selects 3 reps + IQR (self-test-covered).
- Reboot only via `./reboot-node.sh` if ever needed; git-only; idle-guard every chunk; canary after
  the boot/READY; no destructive git; halt+report on any abort trigger.

**Entry state (verified ~01:50–02:00 CDT 2026-10-10).** Production LIVE: exo `deploy/next19-dense @
99e2966ee` + mlx-lm `689e4ea`; runner env `DSV41_DENSE=affine6` + `DSV41_DENSE_POLICY=layers.*.ffn.shared_experts.*=q8g64`.
Cluster idle; owner traffic last seen 00:40 CDT. Runner boot: studio1 pid (screen `exorun`) started
2026-10-09 22:43; `[DSV41] engine built: 40/40 layers, rank 1/2, speculative=True (gamma=3)`;
`Wired limit set to 117.19 GiB`. **Confirmed: the live nodes were launched WITHOUT `DSV41_SPEC_GAMMA`**
(`grep -c DSV41_SPEC_GAMMA ~/relaunch_exo.sh == 0` on both), so the eval deploy must export it.

---

## 0a. THE γ CODE SURFACE — `deploy/q2-gamma` (branch-only) — DONE

Branch `deploy/q2-gamma` @ **`2cf078519`** (off production `99e2966ee`), pushed to `origin`
(adurham/exo). Files: `src/exo/worker/engines/mlx/dsv41/engine.py`, `.../rounds.py`,
`.../errors.py` (new `Dsv41ConfigError`), `src/exo/api/types/api.py`, `start_cluster.sh`,
and a new scoped test `src/exo/worker/engines/mlx/dsv41/tests/test_dsv41_spec_gamma_env.py`.

- **Env var `DSV41_SPEC_GAMMA`**, read ONCE at engine construction (`Dsv41Engine.__post_init__`),
  integer-validated against the **supported set `{2,3,4,5}`**, **HARD ERROR** (`Dsv41ConfigError`)
  on anything else — never a silent fallback. UNSET/empty ⇒ dataclass default 3, **byte-identical**
  to production, no new log line. Set ⇒ a per-rank `[DSV41] spec gamma override: DSV41_SPEC_GAMMA=N
  -> effective gamma=N (supported set (2, 3, 4, 5)), rank R` line. **NO request-level surface.**
- **Supported set {2,3,4,5} justification:** `rounds._one_round` drafts `width=gamma` and verifies in
  one `gamma+1`-row forward (shape-generic); mlx-lm `spec.py::VERIFY_MS` is keyed 1..6 ⇒ `gamma+1 ≤ 6`
  ⇒ `gamma ≤ 5`; the DSpark head's native block is 5 (`args.dspark_block_size`, no clamp). **γ5 IS
  structurally supported.**
- **Acceptance histogram** (rule f) was ABSENT on production — **ported additively** from
  `deploy/next14-gamma`: engine field `_spec_accept_hist` (len 7, index k = rounds accepting exactly
  k drafts), exposed as `GenerationStats.mtp_accepted_histogram_cumulative` (default `None`). Deltas
  across requests give per-round p1..pk.

**VERIFIED MECHANISM (PM-verified, correcting an earlier subagent caveat).** `engine._rounds` builds
`policy = _spec_policy(self.gamma)` ONCE per request and passes it to `rounds._one_round`, which calls
`policy.next()` each round — but **nothing on the serving path ever calls `GammaPolicy.update()`**
(`grep -nE 'policy\.update\(' dsv41/{engine,rounds}.py` ⇒ ZERO). `GammaPolicy.next()` returns `start`
while `rounds < warmup(4)`, and `rounds` only advances inside `update()`. Therefore **the effective
per-round γ is CONSTANT for the whole request and equals the override** — a clean, full-request γ
lever (NOT warmup-only; the "adapts over {1,2,3,4} after 4 rounds" behavior is inert here). The
misleading caveat comment was corrected on `deploy/q2-gamma` (commit `2cf078519`).

**Gates:** scoped test `19 passed`; sabotage proved RED (neuter validation ⇒ 5 failed); basedpyright
171→171 (zero NEW); ruff clean. **NOT verified:** no live run (cluster read-only during build).

**Arm-switch recipe (source-verified; PM to execute at GO).** Precondition: the eval deploy launched
with `export DSV41_SPEC_GAMMA=3` so `~/relaunch_exo.sh` carries the token.
```
ssh studioN "sed -i '' -E 's/DSV41_SPEC_GAMMA=[0-9]+/DSV41_SPEC_GAMMA=N/' ~/relaunch_exo.sh"   # N in {2,3,4,5}
ssh studioN "grep -o 'DSV41_SPEC_GAMMA=[0-9]*' ~/relaunch_exo.sh"                              # confirm 1 token
ssh studioN '~/relaunch_exo.sh'                                                                # graceful relaunch ~2-3 min
```
Confirm per-rank via the `[DSV41] spec gamma override: ... effective gamma=N ... rank R` log line.

---

## 0b. DESK — INCUMBENT MEMORY HEADROOM AT MAX CONTEXT — DONE (**PROMINENT FINDING**)

Artifacts: `raw/pricing/q2/q2b-memory-headroom.md` (+`.json`, ~20 raw files/scripts). Desk-only, 0 boots.

**HEADLINE: CONDITIONAL BREACH of the wired limit W = 125.829 GB.** The live cap
`maxKvTokens = 1,048,576` admits a **cold single-request prefill that crosses W at ≈328K tokens in
the allocator domain** (band 272K–362K); the OS-footprint domain crosses at **≈266K–286K** (UNKNOWN-grade
model). Measured cold-350K peaks on older builds (125.3 / 126.3 / 126.8 GB true units) already sit AT
or ABOVE W; a cold prefill at the 1M cap extrapolates to ≈156 GB (above the 137.4 GB physical RAM —
extrapolation, unverified). **Warm turns are NOT predicted to cross inside the cap** (123.8 GB at 1M,
2.0 GB under W), but a second cap-sized resident session would. **The largest real traffic prompt seen
is 126,527 tokens — below N\* in both domains.** This is a **budget breach with observed pressure, not
a demonstrated crash** — independent of the γ round; surfaced for the owner.

**PM-verified unit corrections to the Round-2 A2 inputs (both confirmed from source/measurement):**
1. The VM gauge `exo_peak_memory_bytes` **over-reports true bytes ×1.073741824** — the dsv41 engine sets
   it as `Memory.from_gb(mx.get_peak_memory() / 1e9)` (`engine.py:284`, `rounds.py:227`), and
   `Memory.from_gb(v)` stores `v·2^30`. So the recorded **124.833 GB allocator peak is really ≈116.26 GB**.
2. `footprint`'s formatted "GB" is **binary (GiB)** — measured `footprint -f bytes` = `106 GB` formatted
   ⇒ `114,345,083,928 B` (= 106.5 GiB). Q1E's "footprint reports decimal GB" is **wrong**.
   Corrected non-weight transient: **11.56 GB allocator / 17.43 GB resident** (A2 had 20.13 / 9.3).
   The Round-2 Q1E CLOSE verdict is UNCHANGED (both corrected readings still fail its rule).
Flagged-not-fixed: `Conversation.prefill` retains every chunk's DSpark taps until prefill returns
(≈30.7 KB/row) — a candidate to collapse the cold curve onto the warm curve (untested).

---

## 0c. DESK — HOT-SET JACCARD vs WINDOW LENGTH — DONE

Artifacts: `raw/pricing/q2/q2c-jaccard-window.md` (+`.json`, script). Desk-only CPU/numpy, from the
real EXL3 clamp trace (`p30_exl3_trace_clamp.json`, 531 tokens × 40 layers, topk6;
sha256 `b4f1b142…`). Unit: **batch = one R=4 spec-verify group** (γ=3 ⇒ verify R=γ+1=4).

Adjacent-window top-24 Jaccard median (per-layer pooled): **W1 0.371 · W2 0.263 · W4 0.297 · W8 0.231
· W16 0.171 · W32 0.171** (random floor 0.032; parity control 0.714/0.678, reproducing Q1E). Turnover
(fraction of the previous hot-set replaced) rises **0.453 → 0.711** with W. Curve is non-monotone
(small-W estimation noise vs token-space drift at large W); k∈{8,24,48} sensitivity reported
(k=48 is filler-inflated at W1–2 where the window holds <48 distinct experts). Feeds the PARKED
dynamic-hot-set path (reopen condition: paired with a tiering/mixed-bit mechanism) — **does NOT
reopen it.**

---

## 1. PHASE 1 — PRE-REGISTRATION (frozen; the run that judges these must not move them)

Harness: `bench/phase20_q2_gamma/{q2_gamma_driver.py,q2_gamma_eval.py,selftest.py,README.md}` on
branch `p20/q2-gamma-driver` @ `7905947768` (merged into `deploy/phase20-campaign`), self-test **42
checks PASS** (reproduces the frozen r1kit anchors benign 38.48/94.88/2.682, agentic 30.962/101.07/2.1128).

**Arm matrix (same boot, process-restart switches):** bracketed **γ3a → γ2 → γ4 → γ5 → γ3b**.
**Fixed replays:** benign 20K × 3 reps + agentic 91K × 2 reps per arm (fixed salt base `q2gamma`,
per-rep `q2gamma-<rep>` ⇒ identical bytes across arms). Decode-time-per-output-token is the metric
(prefill excluded).

**PRE-REGISTERED GATES (Fable's rules, verbatim intent):**
- **(a) IDENTITY GATE** (replaces the naive identical-stream rule; different γ verifies at different
  batch widths so near-ties can flip): γ3a/γ3b = determinism control. Per arm: record first-divergence
  index vs γ3a + top-1/top-2 logit margin at that position. **PASS** if identical OR every divergence
  is at **margin < ε (pre-registered ε = 0.05)** AND divergence rate ≤ ~2× the γ3a-vs-γ3b rate.
  **ABORT** on any confident-position (margin ≥ ε) divergence. Metric = decode time per output token.
- **(b) MEMORY GATE:** record peak allocator (VM gauge, corrected ÷1.073741824) + peak resident
  (`footprint -f bytes`) per arm; **any arm within 0.5 GB of W is ineligible to win even if faster**
  (γ2 lowers memory — noted as a secondary benefit).
- **(c) RANK CONSISTENCY:** each rank logs its effective γ at init; **harness asserts both match before
  the first request; abort otherwise.**
- **(e) DRIFT CONTROL:** bracketed sequence; if γ3a vs γ3b differ **> 1.5%**, flag drift-contaminated and
  **RE-RUN** rather than decide.
- **(f) ACCEPTANCE INSTRUMENTATION:** per arm, accepted-length histogram + mean accepted tokens/verify
  step (explains results; prices γ6+ on paper).
- **(g) BARS:** agentic winner must beat the **MEDIAN of γ3a+γ3b by ≥ 3% decode tok/s** (or disjoint
  IQRs — DROPPED this round at 2 reps). Benign: not worse than 1.5% on median; IQR not wholly below
  γ3's. **Split** (agentic win + benign loss beyond noise) ⇒ close as *"γ3 confirmed; per-workload γ
  is a future item"* — **do NOT ship a split.** No winner ⇒ close as *"γ3 confirmed on the post-dense
  build"*.

---

## 2. PHASE 1 RUNBOOK (execute ONLY on orchestrator GO) + END STATE

**Eval deploy (boot #1):**
```
cd ~/repos/exo
git fetch origin && git checkout --detach origin/deploy/q2-gamma      # == 2cf078519
export DSV41_SPEC_GAMMA=3                                             # so relaunch_exo.sh carries the token
EXO_TARGET_BRANCH=deploy/q2-gamma ./start_cluster.sh
```
Then: canary both nodes; verify deployed SHA (`git rev-parse HEAD` on both nodes == 2cf078519) + env
(`ps eww` shows `DSV41_SPEC_GAMMA=3`, `DSV41_DENSE=affine6`, policy); READY 2/2. (The launcher's
pre-deploy check uses `merge-base --is-ancestor`, so a detached checkout at the branch tip passes.)

**Run the arm matrix:** `bench/phase20_q2_gamma/q2_gamma_driver.py` (idle-guarded chunks ≤15 min,
per-arm switch + READY + rank-consistency, fixed replays, identity/drift/bars/memory evaluation).

**Production restore (boot #2 — the FINAL live state):**
```
cd ~/repos/exo
git checkout --detach 99e2966eec26891b59e1af5467d8f0ddaad38577          # production tip
unset DSV41_SPEC_GAMMA                                                  # shipped γ3 default, byte-identical
EXO_TARGET_BRANCH=deploy/next19-dense ./start_cluster.sh
```
Verify `RESTORED 99e2966ee READY 2/2 canary parity benign g3 decode=… prefill=…`.

**END STATE:** production restored as the final live state; the γ surface does **not** ship this round
(a winner becomes a SEPARATE pre-registered change with its own quality/battery gates). Docs =
this file + `raw/pricing/q2/` committed+pushed on `deploy/phase20-campaign`; `PERFORMANCE_HISTORY.md`
carries a same-turn entry for the findings. **Awarding of the bars / the close-out verdict is filled
in after the live run.**
