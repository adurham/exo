# dsv41 spec-rollback ring-alias fix — `deploy/next27-ringfix` construct — 2026-10-14

## What this is

`deploy/next27-ringfix` = `deploy/next26-limiter` @ `a5d33b46b` (the LIVE build:
banked small-n wide-tile route fix + degeneration loop-guard + phase-2
reasoning-budget limiter, all armed) **plus the M1 spec-rollback ring-alias fix**:
the mlx-lm fork fix (`spec.rollback` now restores the window ring + the compressed
rows), the exo call-site update that the new required argument needs, and the swept
callers. Carried as the exo fix branch's own commits, with the mlx-lm submodule pin
advanced to the fix tip.

Why: `spec.rollback` reset `cache.offset` and rebuilt the compressor carry, but
touched no window ring. The ring is position-addressed (`p % window`), so a verify
forward that writes `gamma + 1` rows overwrites the slots of the already-committed
`p - window` whenever `gamma - accepted >= 2`. The **next** forward then attends over
rejected-draft KV; on a stack with a layer above the first, the corrupted forward's
hidden states also write that layer's compressed/index caches at committed positions
that are never recomputed, so the perturbation is permanent per conversation (the
strongest form of warm != fresh). Full analysis: the audit `M1`
(`docs/dsv41-state-machinery-audit-2026-10-14.md`) and the fix buildnote
`docs/dsv41-spec-rollback-ring-fix-2026-10-14.md` (on this branch, carried by
`3e0b58a` / `4761fb3` / `e01eaea`).

- **Base**: `deploy/next26-limiter` @ `a5d33b46b` (live since 2026-10-14; guard armed,
  limiter live).

- **Carried exo fix commits** (`fix/dsv41-spec-rollback-ring-2026-10-14`, base
  `a5d33b46b` — the construct branch starts at this branch's tip, so the SHAs are
  preserved, i.e. a fast-forward merge rather than a re-authored cherry-pick):

  | sha | summary |
  |---|---|
  | `3e0b58a05` | `rounds._one_round` -> `spec.snap(cache, position, 1 + gamma)` + the M1 buildnote |
  | `4761fb386` | buildnote: record the mlx-lm 15-file bench call-site sweep + tip SHAs (docs only) |
  | `e01eaeae8` | buildnote: sweep-completion notes + update the 5 archived exo spec probes (docs + archived scripts only) |

- **mlx-lm pin bump**: submodule gitlink **and** `uv.lock:1501` ->
  `894815ae54d7ac4ff1845d36903bc6224d9f826d` (branch
  `fix/dsv41-spec-rollback-ring-2026-10-14`, pushed to `adurham/mlx-lm`). Chain on
  base `a0573e3` (the previous live pin): fix `c4d1982`, bench call-site sweep
  `08a4f45`, sweep completion `894815a` (p67 replay oracle + pW21/pW24 + the archived
  exo probes).

- **Construct-only hunks:**
  - `mlx-lm` gitlink + `uv.lock:1501` — advanced to the fix tip together. Both are
    required: `src/exo/shared/tests/test_submodule_lock_pin_alignment.py` asserts
    the committed gitlink equals the `uv.lock` pin (and, when the submodule is
    initialized, that its HEAD equals the gitlink).
  - `docs/dsv41-next27-ringfix-construct-2026-10-14.md` — this doc.
  - **No launcher change. The fix is code-only — there is NO new env knob to
    forward.** `rounds._one_round` and the mlx-lm engine changed; the span is derived
    internally (`1 + gamma`, the same expression `_ensure_capacity_for_round`
    already used two lines above). A grep of `start_cluster.sh` for
    `SNAP|SPEC|verify_rows|ROLLBACK` is empty (the only hits are the unrelated
    `EXO_SPECULATIVE*` / `EXO_DSV4_POOL_SNAPSHOT_BATCH` lines).
  - **No fingerprint entry needed** (checked against the next25/next26 precedent):
    `dsv41/fingerprint.py`'s `KV_ALLOWLIST` already carries
    `exo.worker.engines.mlx.dsv41.rounds` (the exo-side call site) and every touched
    mlx-lm module (`spec`, `cache`, `session_cache`, `prefill`; `sampling` sits in
    `KV_NEUTRAL`), and the fix adds **no new module** — so the CI guard
    `allowlist_problems()` stays empty. Verified on this tree:
    `KV_ALLOWLIST size: 34`, `problems: []`. The pin bump does move `kv_code_fp`
    (it hashes the allowlisted sources), which is the intended behaviour: parks
    written under the old pin are refused rather than served.

## Verification

All pre-flight on this construct tree unless noted. Shared venv
`/Users/adam.durham/repos/exo/.venv/bin/python`; `PYTHONPATH=<wt>/src:<wt>/mlx-lm`
first (the repo `conftest.py` landmine guard). `EXO_TESTS=1` and
`EXO_DASHBOARD_DIR=/Users/adam.durham/repos/exo/dashboard/build` for the exo suites.

| suite | command | result |
|---|---|---|
| 17 ring tests | `pytest -q mlx-lm/tests/test_dsv41_spec_rollback_ring.py` (from the construct root) | **17 passed** |
| mlx-lm scoped | `pytest -q tests/test_dsv41_sampling.py test_dsv41_session_cache.py test_dsv41_snapshot_prune.py test_dsv41_cache_growth.py test_dsv41_prefill_spans.py` | **67 passed** |
| exo engine + budget | `pytest -q src/exo/worker/engines/mlx/dsv41/tests/test_dsv41_engine.py test_dsv41_reasoning_budget.py` | **76 passed** |
| exo session + degeneration | the same two files plus `test_dsv41_session.py test_dsv41_degeneration.py` | **155 passed** (= 76 + **79** for session+degeneration; the buildnote's "79" is the two-file subtotal, recorded there as a running total) |
| broader `-k dsv41` sweep | `pytest -q -k dsv41 src/exo` | **456 passed, 0 failed** (base tree: **456 passed, 0 failed**) — zero new failures vs base |
| gitlink/uv.lock alignment | `pytest -q src/exo/shared/tests/test_submodule_lock_pin_alignment.py` | green (gitlink == uv.lock pin == initialized submodule HEAD == `894815a`) |
| fingerprint CI guard | `allowlist_problems()` on this tree | `problems: []` |
| launcher syntax | `zsh -n start_cluster.sh` | clean |

Notes on the two suite mechanics a reviewer will hit:

* **The mlx-lm scoped suite is run in the standalone mlx-lm worktree**
  (`~/.hermes/cache/scratch/mlx-lm-ringfix`), which is at the same commit
  `894815a` — `mlx_lm/` and `tests/` are byte-identical to the construct's
  `mlx-lm/` (`git rev-parse` equal; `diff -r` clean). It cannot be run from inside
  the exo construct tree: the exo root `conftest.py` plus the exo root `tests/`
  package shadow mlx-lm's own `tests/` package (`from tests.test_dsv41_session_cache
  import ...` then resolves to the exo `tests/`). This is a pre-existing structural
  gotcha, not a property of this change — the fix buildnote ran it the same way.
  The 17-ring file (which imports no `tests.*`) does run cleanly from the construct
  root and is reported above from there.

* **One transient failure in the first sweep**: `test_dsv41_park_g1.py::test_t9_
  decode_step_time_unchanged_after_checkpoint` failed once under full-sweep load and
  passed both in isolation (x2) and on a full re-run (456/456). It is a timing test
  by construction — "the 2% bound is asserted on the min-of-blocks medians (robust to
  scheduler noise); set `EXO_DSV41_T9_STRICT=0` to report-only on a noisy box" — so
  the single failure is scheduler noise, not a regression. The base tree passed its
  sweep clean, and the construct passed the re-run clean.

## Rollout

No staged rollout required. The fix acts only on the speculative path inside
`spec.snap` / `spec.rollback`: an extra bounded gather of
`min(verify_rows, window)` ring rows plus `ceil(verify_rows/ratio)+1` compressed rows
per layer per speculative round (analytic ~0.2-0.4 MB/round at 40 layers, no new
host sync — `snap` does not call `mx.eval`, preserving the one-sync-per-round
overlap), and a selective restore of only the rejected slots. Non-speculative decode
and prefill are untouched. Both TP ranks execute the same gather/restore from
byte-identical state (no collective, no new hang class).

Launch (established pattern; background + log), both nodes reset by the launcher:

```
cd ~/repos/exo && EXO_TARGET_BRANCH=deploy/next27-ringfix ./start_cluster.sh
```

## Rollback

- Relaunch `deploy/next26-limiter` @ `a5d33b46b` (the current live build; no ring
  fix, no pin bump):

  ```
  cd ~/repos/exo && EXO_TARGET_BRANCH=deploy/next26-limiter ./start_cluster.sh
  ```

- No knob off-switch exists or is needed: the fix has no env gate. Rolling back the
  branch is the rollback.

## Residual / not verified here

- **No live-cluster validation.** No relaunch, no soak, no real-model A/B, nothing
  touched the live instance. Everything below is single-machine / offline.
- **Magnitude on the real model.** All evidence is a tiny random-weight fixture
  (dim 64, head_dim 32, 2 layers). Whether the pre-fix divergence flips a greedy
  argmax at head_dim 512 over 40+ layers — and whether it is the mechanism behind the
  2026-10-13/14 reasoning loops — is **not measured** and not claimed.
- **Real TP=2 two-rank behaviour.** The tests are single-rank; whether both ranks
  hold byte-identical ring contents and execute the same restore is unverified.
- **Real-weight magnitude / acceptance-rate impact.** The added gather cost was
  measured offline on the speculative path only (held-gather ~+1.8%); its effect on
  wall-clock decode tok/s on the live cluster is not measured here.
- **`engram_ids` and the DSpark draft window** are argued safe by their addressing,
  not asserted by this fix's tests (see the fix buildnote's scope note).
- **`verify_rows > window` shapes** are reasoned about (over-estimating is harmless;
  the ring restore is bounded) but the sweep shapes exercised are `1 + gamma` with
  `gamma <= window`.
- **The 15 swept bench probes and the 5 archived exo probes were not executed**
  (they need the model package dirs; several also need the cluster hostfile). They
  are compile-checked and reviewed-by-anchor. A wrong span would raise in
  `rollback` rather than corrupt.
- **`tests/p67_serve.py` (the GPU replay oracle) was not executed** — it needs the
  EXL3 model package dirs and, for `P67_HEAD=real`, ~7.3 GB of DSpark-head weights,
  so it cannot run on this Mac. Its edit is a one-argument addition (`len(rows)`) at
  the exact call site.
- **The 15 swept pU/pV probes remain unexecuted** (carried from the fix buildnote).

## Addendum (same day, later): gate-fold — mlx-lm pin bump + 3 default-OFF launcher lines

The final external review approved the direction but required the deploy gates be
folded in before asking the owner for relaunch approval. Gate 1 (a TP=2
rank-consistency check on the rollback path) needs a tiny debug hook in
`spec.rollback`, so the mlx-lm fix tip advanced and the construct gained a second
commit:

- **mlx-lm fix tip:** `894815a` -> **`2071ba4532264e4e377408a5dd408776b1ed5cfa`**
  (`feat(dsv41): env-gated TP=2 rank-consistency hash on the spec rollback path`,
  pushed to `adurham/mlx-lm`, branch `fix/dsv41-spec-rollback-ring-2026-10-14`).
  Adds `spec.rollback_hash_enabled()` (gated on `EXO_DSV41_SPEC_ROLLBACK_HASH`,
  default OFF, zero cost when off) and one `[DSV41-RB-HASH] rank=… start=… target=…
  rows=… ring=… latents=… engram=… combined=…` line per **rejected** round, plus 12
  focused tests (the ring suite is now **29 passed**; the mlx-lm scoped suite is
  still **67 passed** at the new tip).
- **Construct commit 2:** `mlx-lm` gitlink + `uv.lock:1501` -> `2071ba4`, **plus a
  3-line default-OFF launcher allow-list addition** to `start_cluster.sh`
  (`EXO_DSV41_SPECULATIVE`, `EXO_DSV41_SPEC_ROLLBACK_HASH`,
  `EXO_DSV41_SPEC_ROLLBACK_TURN`).

  **Why the launcher lines (a departure from the first commit's "no launcher
  change"):** the runner env is an allow-list, so a knob not added to `EXO_ENV`
  never reaches the runner. **`EXO_DSV41_SPECULATIVE` is the DSv4.1 engine's OWN
  spec master switch (`builder._SPEC_ENV`) and was NOT forwarded** — the generic
  `EXO_SPECULATIVE` (which *is* forwarded, line ~2103) is **ignored by this
  engine**, so the Gate-2 spec-OFF arm could not be launched without this line.
  All three additions are `[ -n … ]`-guarded, so a launch with none of them set is
  byte-identical to `next26`. `zsh -n start_cluster.sh` clean.

- **Construct tip:** `827e30212` -> the second commit (this addendum's commit).
  Rollback target is unchanged (`deploy/next26-limiter` @ `a5d33b46b`).

## LIVE — deployed 2026-10-15 11:53 CDT (primary, spec-ON; no gate knobs)

`deploy/next27-ringfix` @ `c758bcce5` (mlx-lm pin `2071ba4`) is the **LIVE PRIMARY** as of
**2026-10-15 11:53 CDT** (16:53Z), serving DSv4.1 with speculation ON (γ=3) and **no gate
knobs / no escape hatches** set — the plain production launch:

```
cd ~/repos/exo
unset EXO_SPECULATIVE EXO_DSV41_SPECULATIVE EXO_DSV41_SPEC_ROLLBACK_HASH \
      EXO_DSV41_SPEC_ROLLBACK_TURN EXO_DSV41_PARK_ALLOW_STALE EXO_DSV41_PARK_ALLOW_MLX_DRIFT
EXO_TARGET_BRANCH=deploy/next27-ringfix ./start_cluster.sh
```

- **Both nodes** reset by the launcher to exo `c758bcce5` + mlx-lm `2071ba4`; **READY (2/2)**
  (`HEALTHY! Nodes: 2, Identities: 2`). Engine `40/40 layers, speculative=True (gamma=3)`,
  prefill chunk 2048; loop-guard digest **`633711af`** on both ranks; `DSV41_INDEXER_QSPLIT=0`
  on both. No gate knobs / hatches in either runner env.
- **Smoke** (1 live request): `POST /v1/chat/completions` → HTTP 200, `finish=length`,
  `usage = {prompt 10, completion 48, reasoning_tokens 36}`; `/v1/models` serves DSv4.1.

### Gate history (why this build is live without a waiver)

1. **Attempt 1 (2026-10-14) — false-fail → rollback.** The original pre-registered Gate 2 (a
   spec-ON vs spec-OFF *walk* compare) returned FAIL — 33/40 turns divergent from turn 1 — and
   triggered the pre-registered §(c) rollback to `deploy/next26-limiter`. The reading was a
   **control artifact, not a fix signal**: on 5 of 6 sampled forks a same-arm (spec-OFF) greedy
   re-fire did not reproduce the first spec-OFF stream, so the spec-off arm had low
   discriminating power (this heavily-quantized model is not greedy-bit-reproducible at
   near-ties). A **corrected Gate 2 was pre-registered** (teacher-forced warm-vs-cold, with a
   prefix-cache-off cold baseline) and attempt 2 ran against it.

2. **Attempt 2 (2026-10-15).**
   - **Gate 1 (TP=2 rank consistency on the rollback path) — PASSED live.** `2536` rejected
     speculative rounds compared rank-0 vs rank-1 (`gate1_rbhash_compare.out`): **0 cross-rank
     mismatches**, nothing missing-in-one-rank; VERDICT PASS.
   - **Corrected Gate 2 (warm-vs-cold teacher-forced, ε = 0.5 nat) — REGISTERED NOT-PASS, by
     exactly one position at exact ε.** **682 / 682** forceable positions swept (100 %
     coverage; 2 positions unforceable by construction). Exactly **one** position reaches the
     bar — **turn 6, warm token 80 (char 267)** — whose *registered* value (faithful
     chained-walk, k=2) is **0.5000000** nats, i.e. **equal to ε exactly**; **no** position
     *exceeds* ε (the other seven live realisations of that same context span
     `[0.109375, 0.5000000]`, all ≤ ε). The **rate clause passes**: `1/682 = 0.001466 ≤ 1/128`.
   - **γ=0 (spec-off) control — BRANCH B.** The pre-registered discriminator (one relaunch with
     speculation off) landed in **branch B**: the warm↔cold **residual deviation is NOT
     spec-attributable**. Matched-set median `|Δtop1↔top2|` = **0.258 nats at γ=0** vs **0.266
     with spec-ON** (slightly *larger* on tight pairs under γ=0) — the excess **survives with
     the fix's code path never executed** (γ=0 runs no `spec.snap` / `spec.rollback`), so it
     **pre-dates the fix by construction**. The previously documented **~0.0156-nat** figure
     does not describe this regime; the ~0.26-nat excess is **prefill-vs-decode /
     chunk-boundary numerics**, not the speculative verify path.
   - **No waiver** (per the pre-registered branches — branch B ⇒ deeper triage, never a waiver).
     The candidate is live **anyway**, because it **strictly dominates the previous build**:
     same regime behaviour as next26 *plus* the ring-alias corruption class **eliminated**.

**Net.** The fix stands as: **it removes the ring-alias corruption class; it leaves the
pre-existing regime deviation untouched.** The gross warm↔cold divergences the bug produced are
gone (turn 12: a 7.5469-nat displacement / rank 7-of-8 → byte-identical), and the two remaining
flips reproduce the *buggy* build's signature bit-for-bit (fix-independent).

### Known issues / residual (watch window)

- **Regime residual ≈ 0.26 nats** of warm↔cold top-1↔top-2 local deviation, present both with
  and without speculation, concentrated on **near-tie pairs**, together with the near-tie
  *density* that lets such a residual occasionally flip a greedy argmax. Not attributable to
  this fix and **un-triaged** (named next steps: in-engine KV-path probe on the gate-1 RB-HASH
  arm; chunk-plan-matched A/B).
- **Watch tripwire.** Any **production non-near-tie** warm/cold divergence, **if sampled**,
  triggers a **rollback review**. (Near-tie flips are in-regime and are not by themselves a
  trip.)

### Rollback (one-liner)

Relax to the prior live build (no ring fix, no pin bump):

```
cd ~/repos/exo && EXO_TARGET_BRANCH=deploy/next26-limiter ./start_cluster.sh
```

The fix has **no env off-switch**; rolling the branch back *is* the rollback.

### Artifacts (scratch only — `~/repos/exo` is rsync'd verbatim to both nodes)

- `~/.hermes/cache/scratch/deploy-run/attempt2/` — attempt-2 harness + results: invariant-suite
  logs (`suites_01..08_*.log`), `gate1_rbhash_*` (rank-consistency compare), `gate1_walk/`,
  `gate2_next27_PROFILE.md`, `gate2_next27_summary.json`, `GATE2_AMENDMENT.md`, `notes.md`,
  `OWNER_ASK.md`.
- `~/.hermes/cache/scratch/gate2-corrected/` — corrected Gate-2 harness (`gate2_corrected.py`,
  `teacherforce.py`, `freeze.py`, `MANIFEST.sha256`, `DESIGN.md`, `RESULT.md`, `results/`,
  `captures/`).
- `~/.hermes/cache/scratch/deploy-run/attempt2/gamma0/` — the γ=0 control: `gamma0_driver.py` +
  `gamma0_runs.jsonl`, `analysis.py`, `m1_specoff.json` / `m2_specoff*.json` / `m3_specoff.json`,
  `notes.md`, and the two launch logs (`logs/launch1_specoff_20261015T143117Z.log`,
  `logs/launch2_primary_20261015T165303Z.log`).
- Attempt-1 failure evidence: `~/.hermes/cache/scratch/deploy-run/GATE2_FAILURE_EVIDENCE.md`.

### LIVE record correction — residual comparator provenance (2026-10-15 afternoon, docs-only)

*Provenance audit: `~/.hermes/cache/scratch/regime-deviation/REPORT.md` (headline, §1, §6.1). No
measured data changes; this refines the **reading** of the branch-B paragraph above and of the
attempt-2 gate package.*

The LIVE section's branch-B note that "the previously documented **~0.0156-nat** figure does not
describe this regime" is refined as follows:

- The **~0.0156-nat figure was an internal conflation, not a residual measurement.** It mixes (a) the
  dsv41 logprob **emission lattice step** 1/64 = 0.015625 nats (`notes.md:249`, `:279`;
  `gate2-corrected/teacherforce.py:310-312`) and (b) an **n = 1 near-tie difference** (0.109375 −
  0.09375 = 0.015625, `notes.md:188-193`) that happened to equal one lattice step. It was then used in
  `GATE2_AMENDMENT.md` §5.2b/§7/§8.2 as if it were a measured residual.
- **The real documented comparator shows in-regime scale.** The campaign's own 2026-10-01 chunk-shape
  measurement (full 40-layer TP=2, taken *before* every era-diff change) is mean **0.14–0.22** / max
  **0.99–1.86** nats `max|Δlogit|`
  (`docs/benchmarks/phase22-dsv41-prefill-speed-2026-10-01/README.md:170-173`; cross-ref
  `mlx-lm/docs/benchmarks/wsC-prefill-2026-09-28/README.md:53-65`). Today's **γ=0 median 0.258** sits
  **inside** that regime → **no median regression**.
- **No 16.5× regression.** The "10–25×" / "≈6–17×" / "16.5×" framings were a pair-margin-change
  distribution divided by the lattice step (or a one-instance delta); they are superseded.
- **Tail lead remains.** The only unsettled quantity is the **tail**: γ=0 max ≈ 3.66–4.06 vs phase22
  max 1.86 — **~2× on non-identical statistics** (chunk-size `max|Δlogit|` on one prompt vs
  warm-vs-cold `|Δ(top1↔top2 margin)|` over 274 positions). Bounded, and still a legitimate lead
  (the named next steps — in-engine KV-path probe, chunk-plan-matched A/B — are unchanged).
- **Branch-A threshold was lattice-anchored / unreachable by construction** — anchored on the same
  conflated 0.0156 figure.

Net effect on the deploy decision: **none.** Branch B already fired and the candidate is live because
it strictly dominates `next26-limiter`; this correction only removes a mis-calibrated comparator from
the record so a future gate does not inherit it.
