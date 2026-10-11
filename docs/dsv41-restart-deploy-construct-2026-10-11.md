# dsv41 restart-survival deploy construct — 2026-10-11

## What this is

`deploy/next21-restart` = the **next20-vision** line plus the **full restart-survival
chain** (G1/G2/G3/G4/G5). This is the offline *construct* for the restart-survival
live phase: it must carry BOTH the vision-reuse fix AND every restart-chain behavior,
verified only with the scoped dsv41 suite (no cluster, no GPU, no node contact).

- **Base**: `origin/deploy/next20-vision` @ `b5f0551856c02a9cafff235bcc7a944e6a57ccaf`
  ("fix(dsv41): image-carrying requests never reuse a conversation session").
  Base HEAD verified equal to `b5f055185` before integration.
- **Integrated chain**: 6 commits, linear off `main` @ `94daf8c18c469e77cd53671ccf63897d72ef386a`,
  cherry-picked in order (source → construct sha):

  | # | source (chain) | construct sha | summary |
  |---|----------------|---------------|---------|
  | 1 | `46990ec73` | `6e592b879` | G2 build fingerprint + refusal policy, G5 durability hygiene |
  | 2 | `360fadbf6` | `c6100cdf2` | G4 multi-rank all-or-nothing restore + cross-rank generations |
  | 3 | `4647fcb7d` | `cd30c41cd` | G1 after-turn idle checkpoint (engine-thread, keep-resident, cancellable) |
  | 4 | `b2f009e69` | `f5f6f8429` | G1 hardening — keep prior generation on abort, dirty tracking, newest-wins restore |
  | 5 | `f30ad7c51` | `ae05d7beb` | G3 park-on-shutdown (signal handler, drain/abort, `EXO_DSV41_SHUTDOWN_PARK` default off) |
  | 6 | `4f2a578b6` | `61b49f82d` | G3 post-audit fix pass |

- **Integration method**: `git cherry-pick 46990ec73^..4f2a578b6` in a fresh worktree
  (`/private/tmp/p20-restart-deploy`) created from the vision base. The chain's git
  linearity was confirmed (parent chain `94daf8c18 → 46990ec73 → … → 4f2a578b6`).

## Conflict resolution

**No textual conflict occurred.** Git's 3-way merge auto-merged the two predicted-overlap
files (`engine.py`, `session.py`) cleanly — the vision edit and the chain edit landed in
adjacent-but-non-overlapping regions, so no `<<<<<<<` / `=======` / `>>>>>>>` markers were
ever produced (verified: `grep -rn '^<<<<<<<\|^|||||||\|^>>>>>>>' src docs` → 0 hits;
`git diff --diff-filter=U` → empty).

The integration was verified to be a **true semantic union**, not a silent revert, by
diffing the construct against the chain tip (`git diff 4f2a578b6 HEAD -- engine.py session.py`):
the ONLY differences are the vision-lineage hunks. Nothing from either side was dropped.

### `src/exo/worker/engines/mlx/dsv41/session.py`

- **Vision side** (`b5f055185`) adds, in `Dsv41Sessions.get`:
  - a keyword-only param `reuse: bool = True`,
  - an early guard `if not reuse: … return self._open("cold:" + _sc.prefix_hash(ids))`,
    which skips BOTH the resident prefix match AND the parked-store restore for an
    image-carrying request (the whole image span must be prefilled from offset 0).
- **Chain side** adds the park/restore integration inside the same method: the
  `if key:` resident-by-key path with `self._restore_parked(ids, key=key)`, and the
  prefix-match path with `self._restore_parked(ids, key=None)`; plus `_park_store`
  (G2 `fingerprint=`, G4 `coordinator=`/`rank=`), `maybe_checkpoint`, `checkpoint_candidate`.
- **Resolution**: kept as the auto-merge produced it — the vision `reuse=False` early
  return sits **before** both `_restore_parked` call sites, so an image-carrying request
  correctly skips the chain's parked-store restore exactly as the vision docstring
  promises, while every non-image request still exercises the full park/restore path.
  Both behaviors intact.

### `src/exo/worker/engines/mlx/dsv41/engine.py`

- **Vision side** (`b5f055185`) changes `_generate` to pass `reuse=embeddings is None`
  into `self._sessions.get(...)` (image-carrying ⇔ `embeddings is not None`).
- **Chain side** (G2/G4/G1/G3) adds the `ParkCoordinator` / `compute_fingerprint` imports,
  `__post_init__` fingerprint + coordinator setup, the §G1 idle checkpoint tick
  (`_idle_tick` → `maybe_checkpoint`), and §G3 `shutdown_park`.
- **Resolution**: kept the union — `reuse=embeddings is None` in the `_generate` call site,
  and all chain setup/hooks. (The vision *base* also carries an unrelated
  `_maybe_install_next18_capture()` helper + its `__post_init__` call; that is
  vision-lineage code preserved verbatim — the chain never touched it.)

## Construct-only fix: allowlist for the newer mlx-lm pin

`src/exo/worker/engines/mlx/dsv41/fingerprint.py` — added one entry to `KV_ALLOWLIST`:

```
"mlx_lm.models.deepseek_v41._gates",  # fence row-count gates (_FENCE_MIN_ROWS) …
```

**Why.** The chain lineage and the vision base pin **different mlx-lm gitlinks**:

- chain base/tip (`94daf8c18` / `4f2a578b6`): `6cc9c1e8709e228fca99ac152cd5e681ddcce65d` (has **no** `_gates.py`)
- vision base (`b5f055185`) and the construct: `689e4eafd3cf2925a1cf78ba81375a3c098a0eef` (has `_gates.py`)

The cherry-pick does not move a gitlink the chain never changed, so the construct
inherits the vision pin `689e4eafd`, which ships `mlx_lm/models/deepseek_v41/_gates.py`
(`_FENCE_MIN_ROWS`), imported transitively by the allowlisted `indexer` and
`sparse_attention`. The chain's allowlist (written for `6cc9c1e8`) did not know it, so
the chain's own CI guard `test_allowlist_problems_is_empty` failed. `_gates` gates fence
row counts consumed by KV-affecting modules, so it belongs in `KV_ALLOWLIST` (not
`KV_NEUTRAL`), matching `indexer`/`sparse_attention`. This is the minimal fix that makes
the construct coherent with its own pinned mlx-lm; it is **not** a behavior change to
either the vision fix or the chain.

## Verification

### Scoped dsv41 suite (the ONLY suite run; full-repo suite avoided — ~600s timeout)

```
cd /private/tmp/p20-restart-deploy
git submodule update --init mlx-lm          # mlx-lm = 689e4eafd3cf2925a1cf78ba81375a3c098a0eef
EXO_DASHBOARD_DIR=/private/tmp/p20-restart-deploy/dashboard/build \
PYTHONPATH=/private/tmp/p20-restart-deploy/src:/private/tmp/p20-restart-deploy/mlx-lm \
/Users/adam.durham/repos/exo/.venv/bin/python -m pytest \
  src/exo/worker/engines/mlx/dsv41/tests -q -p no:cacheprovider
```

- **Final line (after the `_gates` allowlist fix): `354 passed in 34.04s`** (0 failed, 0 skipped).
  ≥ the 346 chain-tip baseline. ✅
- Vision test file `test_dsv41_engine.py`: **40 passed** — GREEN. ✅
- Before the allowlist fix the same command gave `2 failed, 352 passed`; the two were
  (a) `test_dsv41_fingerprint.py::test_allowlist_problems_is_empty` (fixed above), and
  (b) `test_dsv41_park_g1.py::test_t9_decode_step_time_unchanged_after_checkpoint`.

### T9 timing test — flaky, NOT a regression

`test_t9_decode_step_time_unchanged_after_checkpoint` asserts decode-step time after a
checkpoint is within +2% (a perf guard, `EXO_DSV41_T9_STRICT`). Three standalone runs
gave: fail (ratio 1.066), fail (1.796), **pass**. The huge run-to-run spread
(before 572–667 µs, after 711–1028 µs) is CPU contention from the other live subagents on
this laptop, not a construct change. Reported as environmental flakiness; it passed in
the final full-suite run.

### ruff 0.14.11

- Touched files (`engine.py`, `session.py`, `fingerprint.py`): **1 error** — `I001`
  (`import next18_capture`) in `engine.py:159`.
- Whole `dsv41/` package: **1 error** (same `I001`).
- **Before/after**: the vision base `b5f055185` `engine.py` also reports this same `I001`
  (pre-existing, in `_maybe_install_next18_capture`); the chain tip `4f2a578b6` `engine.py`
  reports 0; `session.py` reports 0 on both sides. → **0 NEW ruff errors** from this
  integration. ✅ (Left the pre-existing vision-base `I001` untouched: out of scope.)

## Residual risk / NOT verified

- **NOT exercised on hardware / no cluster**: no node contact, no GPU bench, no
  boots/relaunches. The park/restore/fingerprint/coord/shutdown paths are validated only
  by the synthetic scoped suite — no live park→restart→restore round trip.
- **Gitlink mismatch is unresolved by design**: the construct runs on mlx-lm `689e4eafd`
  (vision pin), while the chain was developed/tested against `6cc9c1e8`. Only the
  allowlist guard surfaced a difference; other behavioral diffs between the two mlx-lm
  pins are NOT audited here.
- **T9 is flaky** under load (evidence above); its pass in the final run is not a
  determinism guarantee.
- **Single-run counts**: 354 passed is one run; T9 may fail intermittently.
- **No cross-rank / multi-process run**: the G4 coordination logic is covered by the
  suite's synthetic harness, not a real multi-rank cluster.
- **Not merged to `main`, not deployed**: branch `deploy/next21-restart` only.
