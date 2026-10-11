# DSv4.1 restart-survival — G2 + G5 build note (2026-10-10)

**Scope.** Implements gaps **G2 (build fingerprint + refusal policy)** and
**G5 (durability hygiene)** of `docs/dsv41-restart-survival-plan-2026-10-10.md`,
under the corrections of `docs/dsv41-restart-survival-phase0-audit-2026-10-10.md`
(the audit wins where the two disagree). **OFFLINE / laptop-only**: no cluster
contact of any kind (no ssh to studio1/studio2, no boots, no model requests, no
benchmarks, no GUI automation), no deploy, production stays `b5f055185`
untouched. Synthetic CPU-only fixtures; no GPU, no checkpoint, no network beyond
the already-present `mlx-lm` submodule + shared `.venv`.

**Branch / worktree.** `p20/dsv41-restart-g2g5` off `origin/main` `94daf8c18`,
in the dedicated worktree `/private/tmp/p20-restart-g2g5` (mlx-lm submodule
`6cc9c1e`). Commit sha: see the push record (`git ls-remote` line) — no node or
shared-checkout was modified.

## What changed

| File | Change |
|---|---|
| `.../dsv41/fingerprint.py` | **NEW** (714 lines). The build fingerprint + refusal policy module. |
| `.../dsv41/tests/test_dsv41_fingerprint.py` | **NEW** (383 lines, 49 tests). Fingerprint + policy + allowlist-guard tests. |
| `.../dsv41/park.py` | **MODIFIED** (+380/−46). PARK_VERSION 1→2, fingerprint manifest field, generation dirs, fsync/atomic commit, non-consuming restore, retention + LRU headroom. |
| `.../dsv41/engine.py` | **MODIFIED** (+51). Compute the fingerprint once at `__post_init__`; forward it to the session store. |
| `.../dsv41/session.py` | **MODIFIED** (+9). Carry `fingerprint` from engine to the lazily-built `ParkedStore`. |
| `.../dsv41/tests/test_dsv41_park.py` | **MODIFIED** (+23/−8). Non-consuming-restore assertions + `.tmp` naming. |
| `.../dsv41/tests/test_dsv41_park_g2g5.py` | **NEW** (653 lines, 24 tests). T1/T2/T3/T4/T6 + generations/LRU/non-consuming. |

### G2 — build fingerprint (manifest fields, `PARK_VERSION` 1 → 2)

New module `dsv41/fingerprint.py`:

- `BuildFingerprint` (`frozen` dataclass): `model_fp`, `tokenizer_fp`,
  `kv_code_fp`, `mlx_version`, `layout_fp`, with `as_dict()` / `from_dict()`
  (`from_dict` returns `None` on a partial/malformed sub-object → "legacy").
- `compute_fingerprint(*, model_path, tokenizer, layout, model)` — computed
  **once at engine startup** (`Dsv41Engine.__post_init__`), never lazily at
  shutdown (plan R3). Never raises; a missing input degrades to a canonical hash
  of what is available.
- `kv_code_fp` = sha256 over an **explicit allowlist** of KV-affecting sources:
  `KV_ALLOWLIST` (31 modules: 7 exo `dsv41` incl. the park codec, 3 exo
  `disaggregated`, 21 `mlx_lm.models.deepseek_v41`) — attention, CSA/HCA, indexer,
  MTP/DSpark, engram, park codec, custom kernels. Every entry carries a one-line
  WHY. `KV_NEUTRAL` (16 modules) holds genuinely neutral modules, each with a WHY.
- `layout_fp` over (world size, rank, layer range, dtype, window, bsz).
- `check_manifest(man, current, *, allow_stale, allow_mlx_drift) -> (ok, reason)`
  with `reason ∈ {None, "fp_mismatch", "legacy", "mlx_drift"}`.
- **CI/unit guard** `allowlist_problems()`: after importing the forward-pass
  entry modules it walks `sys.modules` and returns every family module that is
  neither allowlisted nor declared neutral. Empty == complete. `test_dsv41_fingerprint.py`
  asserts `== []` AND that a planted offender IS detected. (It caught
  `fingerprint` itself on first run; now declared neutral — it only hashes.)

**Policy** (enforced in `ParkedStore._admit`, used by `find`/`try_restore`):
any `model_fp` / `tokenizer_fp` / `kv_code_fp` / `layout_fp` mismatch **or** a
pre-v2 / fingerprintless manifest → REFUSE, log exactly
`park.skip reason=fp_mismatch` (or `...reason=legacy`), and **do NOT
quarantine** (the dir is valid, just foreign/old — LRU reclaims it). mlx-only
drift refused unless `EXO_DSV41_PARK_ALLOW_MLX_DRIFT=1`.
`EXO_DSV41_PARK_ALLOW_STALE=1` overrides every refusal (debug escape hatch). The
44 existing legacy `key:null` dirs are refused this way (audit A5), never
quarantined.

### G5 — durability hygiene

- **Per-file fsync** (`_write_bytes_synced`): every blob + the manifest.
- **Commit protocol**: `manifest.json.tmp` → fsync → `os.replace` →
  fsync the dir; the whole generation is built under `<name>.tmp/` → rename →
  fsync the root. A crash leaves only a `.tmp`, never a half-valid generation.
- **Startup sweep** (`_sweep_tmp`, first thing in `ParkedStore.__init__`):
  deletes stale `*.tmp` / `*.tmp.*` dirs and stray `*.tmp` files.
- **Ignore dirs without manifests**; a garbage manifest is quarantined to
  `*.bad` (and `*.bad` dirs are excluded from candidates so they are never
  re-scanned/re-quarantined).
- **NON-CONSUMING restore** (audit A6): a successful restore leaves the
  generation on disk and marks it the **ACTIVE BASE** with an empty fsynced
  `.active` marker; only a *corrupt* dir is quarantined. So a parked restore now
  survives the **next** restart too, not just one (the old `rmtree`-on-restore is
  gone).
- **Generations**: `ParkedStore.park` names a dir `<prefix_hash>.gen<N>.<uuid8>`
  with `N = max existing + 1` (start 1). **Retention = newest two** generations
  per conversation, pruned **only after** the new generation has committed.
- **LRU headroom** (audit A9: ~471 MiB/session): before a new generation is
  written, evict down to `total + 2·(largest generation) ≤ max_bytes`, so the new
  gen fits before the superseded one is deleted. Eviction order: **foreign +
  legacy dirs first**, then plain LRU (oldest); the just-written generation is
  never evicted.

## Test evidence (exact)

Invocation:
`PYTHONPATH=<wt>/src:<wt>/mlx-lm EXO_DASHBOARD_DIR=<wt>/dashboard <venv>/python -m pytest src/exo/worker/engines/mlx/dsv41/tests/ -q`

- Baseline on `origin/main` tip (`94daf8c18`, untouched worktree): **`178 passed in 2.00s`** (0 skipped).
- Fingerprint module alone: **`49 passed in 0.46s`**.
- **Full suite after G2+G5: `251 passed in 22.78s`** (= 178 + 49 + 24 new).
- `ruff check src/exo/worker/engines/mlx/dsv41/`: **All checks passed!**

Test coverage (in `tests/test_dsv41_park_g2g5.py`, 24 tests):
- **T1** boundary zero-fill / poison invariant at a mid-HCA-block offset and at an
  exact block-boundary offset (byte-identity vs a never-parked twin; poisoned
  latents/engram tails asserted zero after restore).
- **T2** each fingerprint field mismatch → refused, dir untouched (no `.bad`);
  legacy (no fingerprint) and pre-v2 refused; `ALLOW_MLX_DRIFT` tolerates only
  mlx drift; `ALLOW_STALE` overrides a mismatch; `park.skip reason=...` logged.
- **T3** truncated `.bin` → quarantined; manifestless dir ignored; `.tmp` swept
  at store init; garbage manifest quarantined.
- **T4** crash atomicity — a spawned child commits gen 1 then is **SIGKILLed**
  at a random delay mid-gen-2 write, **50×**; each time a **fresh** store
  restores gen 1 byte-identically, no partial gen is a candidate, and the
  leftover `.tmp` is swept.
- **T6** cross-build gate — same all-fields → exact restore; changed
  `model_fp`/`tokenizer_fp`/`layout_fp` with the same `kv_code_fp` → refused.
- non-consuming restore (dir remains + `.active`), two-generation retention,
  foreign/legacy-evicted-first, park never evicts the just-written generation.

## Deliberately NOT done (out of scope for this build)

- **G1** after-turn idle checkpoint — the core fix; needs engine-thread work and
  the MLX buffer-donation trap (plan §G1). Not attempted.
- **G3** park-on-shutdown — needs the signal-handler path and the corrected
  budget (`EXO_SHUTDOWN_GRACE_SECONDS`=15 s; audit A8 — no launchd/ExitTimeOut).
- **G4** multi-rank all-or-nothing restore + cross-rank generation ack — mandatory
  for TP2 (audit A7) but a control-channel change; this build's generations are
  single-rank only.
- **G6** rewind-to-LCP — audit A4 shows divergence is at **token 0**, where
  rewind is useless; deprioritised per the audit.
- **G7** incremental append-only checkpoints.

## Deviations from the plan / audit (and why)

1. `park.skip reason=<reason>` is logged **without** the repo's usual `[DSV41] `
   prefix, to match the plan's literal log-line spec. (Grep-able either way;
   flag if a prefixed form is preferred.)
2. The version gate in `_admit` refuses a pre-v2 manifest as `legacy`
   **unconditionally** (not overridable by `ALLOW_STALE`), because
   `restore_conversation` itself refuses `version != PARK_VERSION`; admitting a
   v1 dir would only quarantine a valid-by-format old dir — contrary to the
   audit's "refuse, do not quarantine" for the 44 legacy dirs.
3. Beyond the letter of G5, `_candidates()` skips `*.bad` dirs (quarantine
   renames, keeps a magic-valid manifest — otherwise it would be re-quarantined
   forever; latent bug in the original code).
4. T4 uses `multiprocessing` **spawn** (not fork): forking pytest's
   multi-threaded process segfaults the child on its first MLX allocation. Kill
   timing semantics are unchanged.

## Open questions

- **Legacy log volume**: with the 44 legacy dirs, every `find()` on the cold path
  iterates all candidates and logs one `park.skip reason=legacy` each. The plan
  wants legacy visibility, but ~44 lines/request is heavy — consider
  once-per-dir or DEBUG-after-first-occurrence.
- **`.bad` reclamation**: quarantined `*.bad` dirs are neither candidates nor
  budget entries, so they are never auto-reclaimed (kept for inspection). A
  bounded-retention policy for `.bad` may be wanted.
- **dtype source for `layout_fp`**: `_dtype_of` degrades to the stable string
  `"unknown"` when no dtype is reachable; a change to the KV storage dtype is
  still caught by `kv_code_fp` (the allowlisted `cache` module fixes the grid
  dtype). Confirm the intended production dtype source.
- **`ACTIVE_BASE` semantics for G1/G4**: the `.active` marker is a single-rank
  concept here; the multi-rank design (G4) will need a cross-rank active/commit
  record — not built.
- **A9/A10** stay `NEEDS-GO`: exact park/restore throughput and peak-at-92K
  memory need an instrumented run — out of read-only scope, not addressed here.

**No-node-contact affirmation:** no ssh/network/cluster operation of any kind was
performed; all work is laptop-local in `/private/tmp/p20-restart-g2g5`.
