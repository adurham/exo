# PHASE 5 — R1b VALIDATION SESSION: RESULTS (relaunch #3 of ≤3 — FINAL)

Session: R1b (Phase-5 P3, final relaunch), 2026-10-08 CDT. Operator: R1b subagent.
Worktree: `/private/tmp/phase20-campaign` (branch `deploy/phase20-campaign`).
Status: **NO-SHIP — the lever is a real perf win but is a genuine OUTPUT change at
production H → pre-registered ABORT. Cluster RESTORED to production.**

## 0. TL;DR / verdict

**R1b completed the measurement R1 could not, and the verdict is NO-SHIP.** The capture
harness was fixed (flush off the request thread) and the fix worked live — the runner
survived a real 91K capture with a ~800 MB `.npz` deflate, which is exactly what killed R1.
The lever arms then produced a **clean, decisive perf win** (agentic 91K **−31.08 ms/round**,
disjoint). But the pre-registered **Leg A (greedy token-identity vs production) FAILED**: on a
fixed greedy prompt the lever boot diverges from production at **token 63/300**. Per the
ratified §14 failure branch ("any token diff vs production on fixed replay → **ABORT, no
re-amendment**"), this **blocks the ship**. A bounded post-mortem capture **proved the root
cause**: the offline "0-diff at production H" evidence compared L2-full-vs-HIER (both fp32),
whereas **production's real small-n decode runs the bf16 row**, which the lever changes.

- **G2**: **PASS** — agentic Δ **31.08 ms**, fully DISJOINT ranges; benign **improves** 23.93 ms.
- **Leg A**: **FAIL** — token diff at 63/300 vs prod `e65639ce…` (clean prod-vs-prod control).
- **Leg C (91K replay)**: **PASS** — `L2full_vs_hier_total_ndiff: 0` on BOTH nodes.
- **Battery (G3)**: **NOT RUN** — the ship is already aborted by Leg A; battery is a ship gate.
- **Ship decision**: **NO-SHIP → RESTORE** (safety-mandated).

## 1. Ledger / budget

| # | phase | deploy | status |
|---|---|---|---|
| R1 | P3 | exo `ff676b3ca` + mlx-lm `17bbd98`, capture ON | SPENT (#1); aborted (harness bug) |
| — | P3 | restore `576e9d279` + `3bf8316` | SPENT (#2) |
| **R1b** | P3 | exo `deploy/next18-identity @ ff676b3ca` + mlx-lm `deploy/next18-lever2 @ 16830e1` | **SPENT (#3, FINAL)** |
| — | P3 | restore `576e9d279` + `3bf8316` | SPENT (safety-mandated; not a lever) |

**Budget: 3/3 spent. Round CLOSED.** No `known-good` tag (no ship).

## 2. THE HARNESS FIX (bench-harness only) — proven live

File: `bench/next18_capture.py` on mlx-lm `deploy/next18-lever2`, commit **`16830e1`**
(pushed to `origin` = adurham/mlx-lm). Also staged on both nodes at `/tmp/next18_capture.py`
(sha256 `73d49b56b32a804cb276ceb01bd6aa8c3342e9ffaf91e3f6c3542edf59226680`).

- R1 root cause: `np.savez_compressed` (~197 MB `.npz`) ran INLINE on the server request
  thread → multi-minute zlib deflate → no runner events → 45 s hang-watchdog SIGKILL.
- Fix: (1) the expensive `savez_compressed` + `os.replace` runs in a **daemon writer thread**
  (single-slot coalescing; the request thread only snapshots the ring and returns); (2) a
  **runtime GATE sentinel** (`<capture_path>.gate`) makes the hook an immediate no-op so a
  capture hook never runs during a timing arm; (3) daemon interval flush (30 s) + blocking
  atexit final flush.
- **Local verification** (`verify_harness.py`): gate inert (0 saved); with `_write_ring`
  slowed to 1.5 s the indexer `__call__` returned in 2–7 ms (npz absent immediately, landed
  later on the writer); schema keeps `index_k`+`out`.
- **Live proof:** the 91K capture ran with the hook ACTIVE; the runner **survived** (PIDs
  stable, no `HANG_STACK`/`SIGKILL`); a **821,164,613 B** `.npz` was written off-thread on
  both nodes. **The fix eliminated the R1 failure mode.**

**STORE_K decision:** kept **`STORE_K=1`** (full `index_k`). `bench/next18_replay_capture.py`
reads `d[p+"index_k"]` and re-scores real tensors, so `STORE_K=0` (sha+shape only) would have
made the HARD leg-C replay un-runnable.

## 3. A — PRE-FLIGHT (PASS, 2026-10-08 22:04 CDT)

- idle guard `ok=true`; both nodes `HEAD=576e9d279 MLX=3bf8316`; canary 14.85 / 14.85 healthy.

## 4. B — DEPLOY LEVER (relaunch #3/3 SPENT)

- exo `deploy/next18-identity @ ff676b3ca` (576e9d279 + capture launcher patch) +
  mlx-lm WT `16830e1`; `EXO_TARGET_BRANCH=deploy/next18-identity`; DEFAULTS; capture env ON
  + gate sentinel.
- `Nodes synchronized on commit ff676b3ca`; HEALTHY; **READY (2/2)**; post-boot canary exit 0.

## 5. B3 — INSTALLED-MODULE + RUNNER-ENV VERIFY (PASS, BOTH nodes)

| check | studio1 | studio2 |
|---|---|---|
| exo / mlx-lm-wt | `ff676b3ca` / `16830e1` | `ff676b3ca` / `16830e1` |
| installed indexer lever-2 guard (`if _HIER and n > _FENCE_MIN_ROWS`) | 1 | 1 |
| installed `_L2_FULL` occurrences | 3 | 3 |
| `_FENCE_MIN_ROWS` default | 16 | 16 |
| installed sparse_attention lever-1 guard (`and m > _FENCE_MIN_ROWS`) | 1 | 1 |
| fixed harness in node WT (`off_thread_flush=True`) | sha `73d49b56…` | sha `73d49b56…` |

Runner env (both nodes): `DSV41_NEXT18_CAPTURE=/tmp/next18_91k.npz`, `DSV41_NEXT18_CAPTURE_NS=1,4`,
and **no** `DSV41_INDEXER_L2_FULL`/`_HIER`/`_SMALLN_ROW_BF16`/`SPARSE_FENCE_MIN_ROWS`
(DEFAULTS confirmed). Capture hook log: `[DSV41] next18 capture hook ACTIVE` + stderr
`installed: … gate=/tmp/next18_91k.npz.gate interval=30.0 off_thread_flush=True`.

## 6. C — LEVER TIMING ARMS (fixedsalt `r1fix-a`, capture INERT via gate)

| arm | reps ms/round | median | decode t/s | mean_accepted |
|---|---|---|---|---|
| agentic 91K (lever) | 99.08 / 99.23 / **99.30** / 99.40 / 99.51 | **99.30** | 31.18 | 2.1089 |
| benign 20K (lever) | 94.67 / **94.93** / 94.98 | **94.93** | 40.32 | 2.9314 |

(vs R1 baseline: agentic **130.38**, benign **118.86**.)

### G2 — PASS
- **agentic Δ = 31.08 ms** (bar ≥15) with **DISJOINT** ranges: lever [99.08, 99.51] vs
  baseline [130.23, 130.63].
- **benign no regression**: lever 94.93 vs baseline 118.86 → **improves 23.93 ms**.
- **The missing `HIER=0` benign point**: the lever benign **94.93 ms** is the *fallback-path*
  benign measurement the missing `next17+HIER=0` benign cell wanted (predicted ≈95 ms). Labelled
  `r1_lever_benign_hier0proxy` (proxy, not the literal next17 cell).
- **Content caveat:** `mean_accepted` moved (agentic 2.1581→2.1089; benign 2.7736→2.9314) and
  `content_chars` differ on most reps. That is **not** an artifact — it is the lever changing
  output (see §8). The 31 ms Δ is far too large to be a content-confound artifact.

## 7. D — LEG A: greedy token-identity (FAIL)

Same fixed greedy prompt as R1 (`RM.build_prompt(20000,'r1fix-tokenid','count')`, 20,073 tok,
sha `deec4f8d1a3c71fa`, 300 gen tokens).

| arm | sha_think | 
|---|---|
| production reference (`legA_base`, R1) | `e65639ce0dd6ef41` |
| prod-vs-prod CONTROL (R1) | `e65639ce0dd6ef41` (identical, deterministic) |
| **lever boot (R1b)** | **`681b28c3fe24a5e3`** |
| restored prod (R1b) | `e65639ce0dd6ef41` (== reference) |

**First id divergence: token 63 / 300** (char 227) — early, not a tail tie-flip.
prod: `…Need verify. I'll write sequentially.`  lever: `…Need avoid mistakes. I'll write
sequentially.` Deterministic across 2 lever runs. → **§14 ABORT trigger.**

## 8. E — 91K CAPTURE REPLAY (PASS) + ROOT CAUSE

Post-mortem capture on the lever boot (gate removed; 1 agentic 91K rep; hook LIVE; runner
survived). Offline replay (`bench/next18_replay_capture.py`, mlx-lm `17bbd98`):

| node | records | replayable | `L2full_vs_hier_total_ndiff` | `bf16row_vs_hier_total_ndiff` | slots |
|---|---|---|---|---|---|
| m4-1 | 128 | 64 | **0** | 39,917 | 131,072 |
| m4-2 | 128 | 64 | **0** | 42,802 | 131,072 |

- **Leg C (HARD R2 precondition): PASS** — `L2-full` == `HIER` on real production-H tensors.
- **ROOT CAUSE of the Leg-A diff (proven):** the offline evidence compared
  **L2-full vs HIER** — both fp32 at small n → 0 diff (leg C confirms). But **production's
  real small-n decode runs the bf16 row path** (production runs `_HIER` at all n and the n=1
  fallback stores a bf16 row, `_ROW_DTYPE` default). The lever makes that small-n row fp32,
  which flips decode top-k near ties on real (bf16-quantized, tie-dense) activations
  (bf16-vs-HIER gap ≈ 40K/131K slots). That changes committed tokens → the greedy trajectory
  → the token-63 divergence. **The lever is a genuine output change at production H**, exactly
  what the pre-registered Leg A is designed to catch.
- Note: the capture's `meta.jsonl` sidecar was ~8 GB (array values serialized to JSON) —
  wasteful but off-thread; cleaned on restore. Flagged for a future harness pass.

## 9. F/G — battery + ship decision (NOT RUN / NO-SHIP)

Battery (G3) was **not run**: it is a *ship* gate, and the ship was already blocked by the
pre-registered Leg A abort; running a long quality suite on a boot that will be discarded is
not warranted. **Ship decision: NO-SHIP → RESTORE.** No `known-good` tag.

## 10. I — RESTORE (PASS)

`git checkout --detach 576e9d279` + `git -C mlx-lm checkout --detach 3bf8316`, all
gate/capture env unset, `EXO_TARGET_BRANCH=deploy/next13`, `./start_cluster.sh`.

- `Nodes synchronized on commit 576e9d279`; HEALTHY; **READY (2/2)**; canary **14.86 / 14.85**.
- Both nodes: `HEAD=576e9d279 MLX=3bf8316`; installed `_L2_FULL`=0 (absent, expected);
  runner env **no** `DSV41_*` keys; capture temp files + `/tmp/next18_capture.py` removed.
- **Restored-prod token identity:** `sha_think=e65639ce0dd6ef41` == pre-R1 production reference.
- Parity smoke (benign 20K ×3, same fixed salt): median **117.66 ms** [118.30, 117.01] vs
  baseline 118.86 → **Δ ≈1.2 ms, within noise**; `content_chars` reproduce the baseline
  exactly on reps 1/2 (1666/1546) → the restore reproduces production output byte-for-byte.

## 11. Final cluster state

- **LIVE (restored):** exo `deploy/next13 @ 576e9d279` + mlx-lm `3bf8316`, gates UNSET,
  canary 14.86/14.85 healthy, 2 runners Ready, no stray bench processes, `known-good` tag
  **NOT** created (no ship).
- **Relaunch:** R1b = #3/3 SPENT. Round closed.
- **Code pushed:** the harness fix only — mlx-lm `origin/deploy/next18-lever2 @ 16830e1`.
  No exo change. No ship branches promoted.

## 12. Evidence IDs

| ID | artifact |
|---|---|
| E-R1B-PREFLIGHT | `r1b/LEDGER.md` §A; idle ok, rev 576e9d279/3bf8316, canary 14.85 |
| E-R1B-DEPLOY | `r1b/deploy_r1b.log` — READY (2/2), synced ff676b3ca |
| E-R1B-INSTALL | `r1b/verify_installed.sh` output — installed-module + env both nodes |
| E-R1B-HARNESS | mlx-lm `16830e1` + `r1b/verify_harness.py` (PASS) |
| E-R1B-AGENTIC | `r1b/r1b_lever_agentic.json` — median 99.30 ms [99.08…99.51] |
| E-R1B-BENIGN | `r1b/r1b_lever_benign.json` — median 94.93 ms [94.67…94.98] |
| E-R1B-BASE | `r1/baseline_agentic.json` 130.38 / `r1/baseline_benign.json` 118.86 |
| E-R1B-LEGA | `r1b/legA_lever.json` sha `681b28c3fe24a5e3` vs ref `e65639ce0dd6ef41` (div tok 63) |
| E-R1B-REPLAY | `r1b/replay_r1b.log` — ndiff 0 / 0 on both nodes; bf16row 39,917 / 42,802 |
| E-R1B-RESTORE | `r1b/restore_r1b.log` §10 assertions + token identity + smoke 117.66 ms |
| E-R1B-CAPTURE | `r1b/next18_91k.m4-{1,2}.npz` (821,164,613 B each) |
