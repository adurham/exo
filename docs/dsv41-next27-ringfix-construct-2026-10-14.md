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
