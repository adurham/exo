# PHASE 20 — DENSE / SHARED EXL3 ROUND (mechanism fork → tune-existing lever OR honest close)

Author: Phase-20 PM (depth-1 subagent). Opened **2026-10-09 ~03:20 CDT**.
**This doc is the resume anchor — commit + push after EVERY step.** A PM died mid-round earlier
in this project; docs saved it.

Branch: `deploy/phase20-campaign` (worktree `/private/tmp/phase20-campaign`).
All proof work in Phases 1–3 is **OFFLINE** (bench-only on ONE idle-guarded node, no relaunch,
no server restart, no API POST). Cluster promotions are spent only in Phase 4+.

---

## 0. Entry state (PM-verified live, 2026-10-09 ~03:10 CDT)

- **PRODUCTION (RESTORE target) = the shipped next18 build, verified on BOTH nodes:**
  - exo `deploy/next18-identity @ fb4f9290b` (`ssh studio1|studio2 'git -C ~/repos/exo rev-parse --short HEAD'` == `fb4f9290b`)
  - mlx-lm `deploy/next18-lever2 @ 16830e1` (submodule working tree on both nodes == `16830e1`)
  - Tag `known-good-decode-next18-20261009-001052` on both adurham forks.
  - Gates UNSET (no `EXL3_*`, no `DSV41_*` in runner env — verified in the live `ps` command line).
- **Cluster is LIVE**: 1 `MlxJacclInstance`, served model `dealignai/DeepSeek-V4.1-Flash-UNCENSORED-EXL3-2.9bpw`,
  2 runner processes per node, `/state` reachable at `http://192.168.86.48:52415`.
- **Baseline perf (fixed-replay parity smoke, shipped build):** agentic 91K = **99.46 ms/round / 31.0 t/s**;
  benign 20K = **94.94 ms/round / 39.1 t/s**. This is the number to improve.
- **Previous production (secondary fallback):** `deploy/next13 @ 576e9d279` + mlx-lm `3bf8316`.
- **Real weights live only on the nodes**: `~/.exo/models/dealignai--DeepSeek-V4.1-Flash-UNCENSORED-EXL3-2.9bpw`.
  `studio2:~/.p5-dense-ws/` contains the surviving P2 harness (`p2_dense_real.py`, `p2_cached_w.py`,
  `p2_dense_census.py`) + their JSON.

---

## 1. THE TARGET — dense/shared EXL3 slice (measured on real weights, Phase-5 P2)

| quantity | value | source |
|---|---|---|
| effective dense rate, m=4 (verify, γ=3) | **53.4 GB/s** | `raw/p5/p2_dense_real_layer20.json` |
| effective dense rate, m=1 (draft/decode) | **69.3 GB/s** | same; flat across layers 0/10/20 |
| real bytes / rank / PASS | **2.734 GB** (packed=80 → 5.0 bpw, measured from headers) | `raw/p5/dense_census.json` |
| one m=4 verify pass | **≈ 51.2 ms** | 2.734 / 53.4 |
| dense share of round | **≈ 54 %** of a ~95 ms round (largest, worst-per-byte slice) | P2 §4 |
| measured pure-read floor | **497 GB/s** (canary; the old 450 was a triad number) | P2/Phase-4 |
| gap to floor | **9.3×** | |
| passes/round | **1** (the m=4 verify; MTP draft is a separate DSparkHead with no Indexer) | P2 §4 |

- **Kernel-parity audit (P2):** production dispatch ≡ the best fused-inner variant (C) on every shape;
  full-W (A) 2.8–3.6× slower, striped (B) 4.1–5.3× slower at m=4. `EXL3_FUSE_POST=1` neutral.
- **Measured-DEAD routes:** (1) bf16 dequant-cache — decoded-W streams 143.8–175.0 GB/s **native** but
  is 3.2–4.0× the bytes → 10–21 % **slower** wall-clock (m=4: 1.424 vs 1.179 ms whole-slice); confirmed
  **refuted at the mechanism level**, not merely projected away. (2) `EXL3_FUSE_POST=1` neutral.
- **Prior mega-eval (2026-10-07)** declared the EXL3 dense small-M GEMM space **EXHAUSTED** (no variant
  ≥ 2 ms/round); P2 **reproduced** the same plateau (53–55 m=4 / 69–77 m=1). The plateau is real, not suspect.

### THE OPEN TENSION (what this round decides)

Exhausted verdict says "no ≥2 ms variant lever", yet the floor gap is **9.3×**. Which is it?
- **(a) ALU/issue-bound in the trellis dequant itself** — near the M=1/M=4 structural ceiling for this
  quant → **close the track with calibrated evidence**.
- **(b) latency/occupancy/scheduling-bound at m=4** (few independent accumulators, serial K loop,
  no load/decode pipelining) → **split-K / geometry / async-prefetch can harvest**.

**Phase 1 Experiment 1 decides.** Everything downstream is contingent on it.

---

## 2. BUDGET (declared UP FRONT — before spending)

- **≤ 3 promotions + 1 reserve** (4 cluster relaunches total). A **promotion** = a cluster
  relaunch/deploy (reboot via `./reboot-node.sh` and/or `./start_cluster.sh`). **Bench-only sessions
  do NOT count** — running a microbench on a node against the real checkpoint with no relaunch,
  no restart, no API POST, consumes no promotion.
- **Every promotion is declared in the ledger BELOW before it is spent.** No promotion while an
  offline gate is red.
- A phase whose proof fails → do NOT ship; write it up; proceed. **An abort / honest close is a valid,
  successful outcome.**

### Promotion ledger (declare-before-spend)

| # | phase | deploy | purpose | status |
|---|---|---|---|---|
| P1 | 4 | treatment build (env kill-switch for same-build A/B) | live paired fixed-replay A/B vs shipped next18 | PLANNED |
| P2 | 4/6 | `fb4f9290b` + `16830e1` (restore) OR ship build | contingency restore if P1 gate fails / ship deploy | HELD |
| P3 | 6 | ship build (if distinct from P1 boot) | fresh-boot canary + parity smoke + tag | HELD |
| R | — | any | **RESERVE — one pre-named retry** | HELD |

Spent so far: **0 / 3 (+1 reserve)**.

---

## 3. GATES (PRE-REGISTERED — before the work judged by them)

- **G0** — this pre-registration exists and is committed **before** any Phase-1 measurement:
  budget, gates, falsifier, experiment list, owner-ruling citation. *(This doc.)*
- **G1 (offline microbench gate)** — the chosen change shows **p95 m=4 dense-pass improvement ≥ 5 ms**
  over the shipped build, with **m=1 regression ≤ 1 %**. Fail → revert the change, move to the next
  candidate mechanism. **Bench-only; consumes no promotion.**
- **G2 (live A/B)** — on the treatment build vs shipped next18, same-boot paired fixed-replay runs
  (benign 20K + agentic 91K, ≥ 3 reps/arm, interleaved, idle-guarded): **p95 round-time improvement
  ≥ 5 ms agentic**, with **disjoint ranges**, and benign **no regression**. If bitwise-identical →
  exact output-identity check; if numerics-changing → proceed to G3.
- **G3 (quality — owner ruling)** — the **R8a quality battery is the GOVERNING ship gate** for
  output-affecting / numerics-changing work: **CLEAN** on the treatment build (needles 6/6, tools 10/10,
  prose 0 DIRTY / 0 REVIEW, park PASS, `compare.py` vs the frozen `g3`). For bitwise-identical work the
  **identity gate** applies (canary prompts, exact match).
- **G4 (ship)** — gitlink bump to a production branch; verify the **INSTALLED venv module on BOTH nodes**
  (not the gitlink); canary healthy after boot; parity smoke **≤ 94.9 benign / ≤ 99.5 agentic ms**
  and improved-or-equal; tag `known-good-dense-<date>` on BOTH forks; SHIPPED line with numbers in this doc;
  `docs/PERFORMANCE_HISTORY.md` on main in the same turn.

### Owner ruling (verbatim, in force)

> **"as long as we pass our quality tests then I'm fine with it"**

→ the R8a battery is the governing ship gate for output-affecting/numerics-changing work; the strict
identity gate applies to bitwise-identical work. This is how the last ship (next18, R1c) was sanctioned;
applied here unchanged.

---

## 4. THE FALSIFIER (honest close — a VALID outcome)

**CLOSE the dense track** (do not keep grinding) if EITHER:

1. **Phase-1 Experiment 1 (raw no-dequant isolation) shows ≥ 350 GB/s** → the trellis dequant is
   **ALU/issue-bound** (gap is near-structural for this quant; R2/R3 unlikely to harvest); OR
2. after the eligible mechanisms (≤ 3 promotions) **no offline p95 m=4 pass reduction ≥ 5 ms** AND
   **no live p95 ≥ 5 ms** materializes.

Close-out write-up must attach the evidence (m-sweep, raw no-dequant rate, split-K/prefetch sweeps,
dequant-cache data) and one of these honest conclusions:
- *"trellis dequant ALU/issue-bound near the M=4 structural ceiling"*, or
- *"latency/occupancy-bound but requires a NEW custom kernel, which is scope-closed (NOT-FUNDED)"*.

---

## 5. SCOPE WALL (hard)

**FORBIDDEN:** new custom kernels (NOT-FUNDED — this is the wall), head-sharding, tree drafting,
context caps, Sinkhorn truncation, pad-to-M=8, Fix B, SDPA anomaly, `EXO_KV_CACHE_BITS≠0`,
`EXO_DSV4_INDEX_TOPK<512`, `repetition_penalty≠1.0`. Never push upstream; adurham forks only.
No destructive git.

**Consequence for this round:** a mechanism that needs a **new shader entrypoint** is CLOSED here.
A mechanism that is a **tune of an existing launch parameter / in-place patch of an existing shader**
is in scope. Phase 1 Experiment 3 (split-K) is therefore **conditional on in-place patchability**.

---

## 6. EXPERIMENT LIST (Phase 1 — offline, no cluster spend)

Real harness = the repo's own scripts, **not** an invented CLI:
`bench/p2_dense_real.py` (production-shape, REAL weights, calls the `EXL3Linear` module itself),
`bench/exl3_dense_smallm_probe.py` / `_megaeval.py` (synthetic), `bench/p2_cached_w.py` (route test).
Extend/mirror THOSE and the mega-eval methodology
(`docs/benchmarks/phase19-latency/raw/phase3-kernel-megaeval.json` + `.stdout`).

| # | experiment | shape | decision |
|---|---|---|---|
| **1** | **RAW no-dequant isolation (THE FORK)** | production shapes, m=1 + m=4 | **(a) vs (b)** — see §4 |
| 2 | m-sweep + occupancy report (R3) | m ∈ {1,2,4,8,16}, layers 0/10/20, p95 | is m=4 specifically degraded vs m=1? |
| 3 | split-K sweep (R1) — **ONLY if patchable in-place** | m=4 (+m=1) | does a better `n_splits` harvest? |
| 4 | prefetch / double-buffer (R2) — only if the load path supports it as a patch/parameter | m=4 | bitwise-identical if FMA order unchanged |

**Experiment 1 method (must not merely re-read the P2 number):** the cached-W figure (143.8–175 GB/s
native) is a *decode-ablated matmul* but was taken with the W cache-resident (polluted the fused arm)
and is a matmul, not a pure read. Exp 1 must isolate cleanly:
- (i) **pure-read control** at the dense-slice byte size/shape (streaming sum / copy of a bf16 array
  of the same total bytes) → the achievable read rate on this node *for this shape class*;
- (ii) **decode-ablated GEMV/GEMM**: `decode_full_mlx` → resident bf16 W **once**, then time only
  `x @ W` at m=1 and m=4, with the W made **non-resident** (or L2-flushed) so each rep is a real DRAM read;
- normalize per byte and report GB/s for (i) and (ii). ≥350 → (a); <100 → (b); 100–350 → **ambiguous,
  run everything, weigh.**

Timing hygiene (ALL microbenches): warmup ≥ 3 passes; flush the Metal queue before timing; **discard
the first timed rep**; report **p95** (not min); **no GPU capture inside the timed loop**; interleave
arms; record a clock canary alongside; unique SALT per feed if a feed is involved.

**Phase-1 close:** commit `MECHANISM.md` with the measured table + chosen mechanism + why. If the
chosen mechanism needs a **new custom kernel → CLOSE the dense track here** with the Phase-1 evidence.

---

## 7. PHASE PLAN (in order; adapt to the REAL harness)

- **Phase 0 (this doc, offline):** pre-register. Commit + push.
- **Phase 1 (offline, no cluster spend):** run exps 1–4 in order; `MECHANISM.md`.
- **Phase 2 (offline):** implement ONLY if a tune-existing mechanism won Phase 1. Keep it behind an
  **env kill-switch** so A/B is same-build. Unit-test locally (quick single-file runs only; NEVER full
  suites on the user's Mac).
- **Phase 3 (offline, gate G1):** microbench the chosen change. Fail → revert, next candidate.
- **Phase 4 (live A/B, G2; promotion P1):** deploy the treatment build (gitlink-bump pattern + env
  kill-switch); idle-guarded fixed-replay paired runs; canary after boot. Fail → restore (promotion P2).
- **Phase 5 (G3):** battery (numerics-changing) OR identity gate (bitwise-identical).
- **Phase 6 (G4):** ship.
- **End state:** best build live, gates unset, canary healthy, SHIPPED/RESTORED line + parity smoke in
  this doc, no stray processes, `PERFORMANCE_HISTORY.md` on main. Restore path always:
  `fb4f9290b` + `16830e1`.

---

## 8. RELAUNCH / SAFETY DISCIPLINE (from `exo-cluster-operations`)

- **Idle-guard before every chunk/relaunch** (`bench/phase20_guard.py idle` / `watch -- <cmd>` +
  persisted own-request registry `raw/own_requests.jsonl`); never interfere with user traffic; wait
  windows out. **Canary after every boot/READY before measuring.**
- Reboot ONLY via `cd ~/repos/exo && ./reboot-node.sh studio1 studio2`. NEVER raw
  `sudo shutdown -r now`.
- Deploy: move BOTH trees (`git checkout --detach <exo-sha>` + `git -C mlx-lm checkout <mlx-lm-sha>`);
  `start_cluster.sh` installs mlx-lm from the SUBMODULE WORKING TREE → **verify the INSTALLED venv
  module on BOTH nodes**, not the gitlink. If the target branch is held by a worktree, use `--detach`.

---

## 9. STEP LOG (append after every step; commit + push each time)

- **[0]** 2026-10-09 03:20 CDT — Phase 0 pre-registration written. Budget 0/3(+1 reserve) spent.
  2nd-opinion note: consult is rate-limited until 05:20; use `model='deepseek-v4-pro'`
  `provider='ollama-cloud'` if needed. Resume from §6 Experiment 1.
