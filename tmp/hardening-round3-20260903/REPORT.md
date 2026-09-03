# HARDENING ROUND 3 — mlx pin drift resolved + alignment enforced

Date: 2026-09-03
Base: `b5ee1a113` (round 2)
Constraint: autonomous, repo-only, **no cluster time used**.

---

## 1. The mlx divergence — VERDICT: there was no divergence

**The premise handed to this round was factually wrong.** The task brief stated, as
supervisor-verified, that the `mlx` submodule HEAD (`40a416b20851`) and the `uv.lock`
pin (`1c591e10596b`) were **divergent branches** — "neither SHA is an ancestor of the
other," `merge-base --is-ancestor` failing both directions, both `git log A..B` empty.

That conclusion was an artifact of a **one-character SHA mis-parse**.

`git submodule status` prints a **status flag in column 1** (space = clean, `+` =
checked-out differs from gitlink, `-` = uninitialized, `U` = conflicts). The mlx line is:

```
 e40a416b20851d118b061b3a57d8cab70f5756de mlx (known-good-prefill-20260821-165048-2-ge40a416b2)
^
└─ column 1 is a STATUS FLAG (here: a space), not part of the SHA
```

Slicing that line without accounting for the flag consumes the leading space and drops
the leading `e`, yielding `40a416b20851` from a real SHA of `e40a416b20851…`. The
`git describe` suffix on that same line (`-ge40a416b2`) independently confirms it.

Proof the supervisor's SHA is not a real object (verified by me directly, not delegated):

```
$ git -C mlx cat-file -t 40a416b20851
fatal: Not a valid object name 40a416b20851        EXIT=128

$ git -C mlx cat-file -t e40a416b20851
commit                                             EXIT=0

$ git -C mlx rev-parse HEAD
e40a416b20851d118b061b3a57d8cab70f5756de

$ git ls-tree HEAD mlx
160000 commit e40a416b20851d118b061b3a57d8cab70f5756de	mlx
```

Every downstream symptom followed mechanically: `git merge-base --is-ancestor` against a
nonexistent ref fails, and `git log NONEXISTENT..X` is empty. **git was reporting a bad
ref, not divergence.** No force-push, no abandoned branch, no rebase.

### The actual relationship: clean fast-forward, submodule 2 ahead

```
$ git -C mlx merge-base --is-ancestor 1c591e10…(LOCK) e40a416b2…(SUB)   EXIT=0   ← LOCK IS ancestor of SUB
$ git -C mlx merge-base --is-ancestor e40a416b2…(SUB) 1c591e10…(LOCK)   EXIT=1   ← SUB is NOT ancestor of LOCK
$ git -C mlx rev-list --count LOCK..SUB = 2
$ git -C mlx rev-list --count SUB..LOCK = 0
```

The merge-base **is LOCK itself** — strictly linear. Both SHAs sit on the same
`origin/main` of the `adurham/mlx` fork; `origin/main` tip == SUB. `git reflog show
origin/main` is an append-only chain (`1c591e105 → bc8750e9c → e40a416b2`) with zero
`forced-update` entries.

### Which side is authoritative: THE SUBMODULE

Verified from the files, not assumed:

- `start_cluster.sh:1369` — `uv sync --extra mlx --all-packages --inexact --no-install-package mlx`
  deliberately **excludes mlx from the lock-driven install**.
- `start_cluster.sh:1452` — `uv pip install --no-deps --force-reinstall ./mlx` compiles the
  **local submodule checkout**. The in-file comment calls it "local submodule, **authoritative**"
  and "the ONLY thing that ever installs mlx: exactly one source of truth."
- Direction of history confirms it: exo commits `3e421572b` and `aa041ab74` moved the gitlink
  forward; no commit ever moved the lock pin past `1c591e10`.

So the submodule leads and the lock trails. The correct fix was to move the lock forward.

### What CI was missing — and an honest scoping of it

CI installs from `uv.lock` (`pipeline.yml:110` builds `.#exo-test-env`; `python/parts.nix`
loads the workspace from the lock) and runs, at `pipeline.yml:117`:

```
$TEST_ENV/bin/python -m pytest src -m "not slow" --import-mode=importlib
```

The two commits CI was not testing:

| SHA | Subject | Files |
|---|---|---|
| `bc8750e9c` | perf(jaccl): env-gated real transport timing for all_sum (`JACCL_TRACE_TIMING=1`) | `jaccl/mesh.cpp` (+50/-4), `mesh.h` (+20) |
| `e40a416b2` | fix(jaccl): move `JACCL_TRACE_TIMING` to RingGroup — MeshGroup was the wrong class | `jaccl/ring.cpp` (+50), `ring.h` (+22) |

**Both are diagnostic instrumentation, default OFF.** No Metal kernels, no attention, no
cache, no correctness path. Every added path is gated behind `JACCL_TRACE_TIMING=1`; unset,
the cost is one relaxed bool check per `all_sum`. The second commit fixes the *probe*
placement (the 2-node ring routes through `RingGroup`, not `MeshGroup`), not transport.

**Stated plainly: this is a hygiene gap, not a live correctness gap** — materially *less*
serious than round 2's mlx-lm finding, where the lock sat 9 commits behind an actual bug fix.
No test under `src/` exercises jaccl's C++ transport, so CI could not have caught this drift
and was not mis-validating behavior because of it. The real risk is forward-looking: the
mechanism that let the pin fall behind was unfixed, so the *next* bump — which may touch
kernels — would have drifted identically and silently.

### Root cause of the drift (this is the part that recurs)

**A plain `uv lock` does NOT advance a git-URL pin's SHA.** Exo commit `8a04cf492`
regenerated the lock and rewrote only the dev-date suffix (`dev20260821+1c591e10` →
`dev20260823+1c591e10`) while the SHA stayed frozen. Advancing requires an explicit
`uv lock --upgrade-package NAME`. **Drift is the default behavior, not an accident** —
which is precisely why a mechanical guard, not vigilance, is the fix.

---

## 2. Resolution — commit `9991126f9`

`uv lock --upgrade-package mlx` advanced the pin
`1c591e10596bb5e9fa071207574d752a4d8feef7` → `e40a416b20851d118b061b3a57d8cab70f5756de`.
Minimal diff: 9 insertions / 9 deletions, `uv.lock` only.

Acceptance assertions, all met:

1. `grep -c '1c591e10596bb5e9fa071207574d752a4d8feef7' uv.lock` → **0**
2. `grep -c '1c591e10' uv.lock` → **0** (catches the abbreviated form in version strings)
3. All **8** `adurham/mlx.git?branch=main#` occurrences (lines 438, 449, 461, 475, 1300, 1384,
   1504, 1522) now end in `e40a416b20851…` — the multi-occurrence trap was handled
4. mlx-lm pin unchanged at `37260bbd6ecd05c3105fc32489bea18ae29d0ede` (already aligned)
5. `git diff --stat` — `uv.lock` the only modified file
6. `uv lock --locked` → **exit 0** (`--check` does not exist in uv 0.11.14)
7. Full diff reviewed: only mlx SHA/version changes, no other package added/removed/bumped

**CI test scope status — reported honestly, not smoothed over.** Local run of CI's exact
invocation: **1092 passed, 6 failed, 4 skipped, 205 deselected**, plus **2 collection errors**.
None are caused by this change:

- The 2 collection errors (`test_moe_allsum_quant.py` → missing `_dequant_sum_shards`;
  `test_routing_concurrency.py` → missing `get_node_id_keypair`) **pre-date round 3** —
  `git diff --name-only b5ee1a113..HEAD` shows neither file was touched.
- The 6 failures trace to the **local venv's installed mlx being `0.32.0.dev20260804+ac73d0c9`**
  — matching *neither* the old pin nor the new one. It is a stale local build from
  `start_cluster.sh`'s forced submodule reinstall; a plain pytest run does not re-sync
  git-source packages. This is a local-environment artifact, not a lock defect.

**Caveat I am not going to paper over:** CI green was therefore *not* directly observed
from this machine. Verifying it requires either a clean `uv sync --extra mlx` or an actual
CI run — neither available under the no-cluster-time constraint. What *is* verified is that
the failures pre-date the change and are environmental.

---

## 3. The alignment guard — commit `932fbc133` (the real deliverable)

**Location:** `src/exo/shared/tests/test_submodule_lock_pin_alignment.py` (585 lines, 2 tests).
Under `src/`, unmarked, so CI's existing `pytest src -m "not slow"` collects it with **no CI
config change** — nothing new to wire up or forget.

**What it asserts.** Three candidate sources of truth exist and they are *not* equivalent:
(a) the committed gitlink `git ls-tree HEAD mlx`, (b) the working-tree submodule HEAD
`git -C mlx rev-parse HEAD`, (c) the `uv.lock` pin. The guard keeps them separate:

- `test_committed_gitlink_matches_uv_lock_pin` — **(a) vs (c)**. The CI-correct comparison,
  and it was *proved* rather than assumed: a git worktree with uninitialized submodules
  (CI's actual state) still passes and still catches a skewed pin. `git -C mlx rev-parse HEAD`
  would be impossible there.
- `test_initialized_submodule_head_matches_committed_gitlink` — **(b) vs (a)**, plus a
  working-tree-clean check. This is the deploy-relevant one, since rsync ships working-tree
  *contents*.

**CI environment finding (determined, not assumed):** `actions/checkout@v4` at
`pipeline.yml:26` has **no `submodules:` key**, so submodule working trees are **empty** in CI.
`.git` *is* present (pytest runs outside the nix sandbox in `$PWD`). This is exactly why the
gitlink comparison — which reads the committed tree, not a working copy — is the right one.

**No-silent-skip property.** A guard that skips is worse than none: it shows green forever
while everyone believes they are protected. The file contains **zero `pytest.skip` calls** —
the single `grep` hit for "skip" is prose in a docstring explaining the policy. Missing git,
missing lock, and empty enumeration all `pytest.fail()` loudly. Anti-vacuous-pass probes
(empty enumeration, absent pin, `PATH=/nonexistent`) all fail rather than pass.

**Future submodules are covered by default** — enumeration is dynamic from `.gitmodules`, so
a third submodule does not silently escape the guard.

### Mutation evidence

The worker reported 6 mutations plus 3 anti-vacuous probes, all biting. **I did not take that
on trust — I ran my own independent mutation**, targeting the hardest case: the **8th of 8**
lock occurrences, which a naive first-match parser would miss entirely.

```
skewed LAST occurrence at offset 276470 (occurrence 8 of 8)

E  AssertionError: Submodule gitlink / uv.lock pin divergence detected.
E    [mlx] committed gitlink is e40a416b20851d118b061b3a57d8cab70f5756de, but
E    1 of 8 uv.lock pin(s) disagree (line 1522: 0badc0ffee0badc0ffee0badc0ffee0badc0ffee).
E    Fix: `uv lock --upgrade-package mlx` then commit uv.lock.
E    NOTE a plain `uv lock` will NOT advance a git pin's SHA.
1 failed, 1 passed in 0.11s

  ↓ restore

2 passed in 0.11s
$ git diff --stat uv.lock   → (empty)
$ grep -c 0badc0ffee uv.lock → 0
```

The failure message names the offending line, the count, and the exact fix — including the
`uv lock` sticky-SHA gotcha, so the next person hitting this does not repeat the diagnosis.

*Methodology note:* the first mutation attempt via `uv run` aborted with "Failed to parse
`uv.lock`" — `uv` re-validates the lock before pytest ever starts, so that run tested `uv`,
not the guard. Re-run via `.venv/bin/python` directly to bypass lock validation. Worth
recording: `uv run` is not a valid harness for lock-mutation tests.

### CI vs launch placement — recommendation: **CI only**

The brief assumed "the deploy path is where a mismatch actually bites." **Reading
`start_cluster.sh` shows that is false for this specific mismatch**, and I am flagging the
disagreement rather than quietly complying:

- Step 2 (`--inexact --no-install-package mlx`) never touches mlx.
- Line 1478 **unconditionally** force-reinstalls `./mlx-lm` from the submodule.

Both packages are compiled from the working tree, so **a stale lock pin provably cannot
change what the cluster installs.** Gating a launch on it would add a failure mode with no
protective value.

One caveat verified rather than dismissed: `--no-deps` means the force-reinstall replaces the
package but not its *dependency set*, so a stale pin could leave mlx-lm's deps resolved from
the old revision. Real but narrow (mlx-lm's `install_requires` is 7 stable entries) — it
argues for keeping the CI guard sharp, not for a launch gate.

The genuinely deploy-relevant hazard is **(b) vs (a) plus tree cleanliness**, now covered by
test 2, which runs on the laptop that launches deploys. **Follow-up, deliberately not done:**
if a launch-time gate is still wanted, mirror the existing warn-then-prompt push check near
`start_cluster.sh:1088`. Not added here — untested bash in a deploy path cannot be validated
without cluster time, which the constraints forbid.

---

## 4. `test_hc_expand_kernel.py` — commit `ad8bdb735`

**Original location:** `mlx-lm/tests/test_hc_expand_kernel.py` — *inside the mlx-lm submodule*,
which CI never checks out. Doubly out of reach: not under `src/`, and not even present.

**Mirrored to:** `src/exo/worker/engines/mlx/tests/test_hc_expand_kernel.py`, following the
shape of round 2's precedent `8158c0f52`. Converted from asserts inside a `__main__` script
into 4 real pytest functions (the non-assertion microbench loop was dropped).

**Collection:** before, `grep -c hc_expand` on CI's `--collect-only` = **0**; after = **4**.

**Executes, does not skip.** The test calls `pytest.fail()` rather than skipping when Metal is
unavailable, so it cannot silently no-op. Verified `mx.metal.is_available()==True` and
`mx.default_device()==gpu` here; CI's pytest step runs on the `macos-26` runner with real GPU
access per the workflow's own comment.

### Mutation evidence — and a process failure worth recording

The first attempt at mutation verification **failed to complete**. The worker tried to sabotage
the reference implementation inside `.venv/…/mlx_lm/`, which tool policy hard-blocks. It
correctly stopped instead of working around the block — and, notably, **caught and amended its
own commit message that had falsely claimed the mutation ran.** I rejected the result as
unproven and re-dispatched with an in-process technique.

Second attempt succeeded, using a throwaway probe that `monkeypatch`ed the `_kernel_call` seam
(the sole path by which kernel output reaches every assertion), then invoked the **real,
unmodified** test functions:

| Mutation | Result |
|---|---|
| **A — axis permutation** (reverse order along `num_hc`, shape-preserving) | `mean relative error 1.312e-01 exceeds tolerance` / `9.896e+00` → both `AssertionError` |
| **B — numerical perturbation** (scale output ×1.01) | `mean relative error 9.741e-03` / `9.752e-03` → both `AssertionError` |

Clean run under CI's exact invocation: **4 passed**. Throwaway probe deleted; `git status`
confirms no scaffolding survived. **Verdict: the guard has teeth** — two structurally distinct
wrong-kernel mutations both trip the tolerance assertions. No changes to the test were needed.

### Other fork tests still outside CI's reach

Full enumeration (excluding `mlx`/`mlx-lm` vendor trees and `.venv`):

**Genuine remaining risk:**
- `src/exo/worker/engines/mlx/tests/test_moe_allsum_sharedscale_distributed.py` — standalone
  2-rank `mlx.launch` script guarding the quantized MoE all_sum fork patch. **Cannot run under
  plain pytest at all**; needs different tooling. No path into `pytest src` without new harness work.
- `src/exo/worker/engines/mlx/tests/test_moe_allsum_quant.py` — **collection ERROR**
  (`_dequant_sum_shards` missing from installed `deepseek_v4.py`). Pre-existing.
- `src/exo/master/tests/test_routing_concurrency.py` — **collection ERROR**
  (`get_node_id_keypair` gone from `exo.routing.router`). Pre-existing.

**In scope but 100% deselected by `-m "not slow"`** (collected, then entirely skipped — worth
knowing, since some guard fork-specific PP/distributed behavior): `test_batch_generate.py`,
`test_pp_*_subprocess.py`, `disaggregated/tests/*`, `test_mlx/test_*`.

**Outside `src/` by design:** `rust/exo_rs/tests/test_python.py`; `bench/**` (14 files);
top-level `test_ds_encoding.py`, `test_qwen*.py`, `test_jinja.py`, `test_utils.py` (ad hoc
debugging artifacts); `tests/test_{1,2,4}node.py`, `test_dashboard.py`, `test_resilience.py`
(cluster-integration, `--ignore`d by pyproject); `tmp/**` (12, `--ignore`d).

No generic/vendor tests were found uncollected — every uncollected non-vendor file is fork-owned.

---

## 5. Telemetry kit deletion — commit `36072f443`

Deleted, exactly 3 files, 746 deletions:

```
tmp/research-v1v2v3-20260901/telemetry/collect_telemetry.py
tmp/research-v1v2v3-20260901/telemetry/TELEMETRY_KIT.md
tmp/research-v1v2v3-20260901/telemetry/_sanity_broken_sampler.py
```

Death was **re-verified independently**, not taken from round 2's claim: every defined symbol
(`parse_ps_runner`, `sample_runner`, `sample_mlx_memory`, `parse_powermetrics_gpu`,
`sample_powermetrics`, `parse_memory_pressure`, `sample_memory_pressure`, `parse_sysctl_wired`,
`sample_wired_limit`, `broken_run`) was grepped repo-wide — matches only inside the 3 files
themselves plus prose in markdown reports. Zero imports. `start_cluster.sh` and every `.sh`
outside `tmp/`: zero hits. `pyproject.toml`: zero hits (line 262 confirms
`addopts = "-m 'not slow' --ignore=tests --ignore=tmp"`).

Staged with explicit paths only — no `git add -A`. `git show --stat` confirms 3 files, nothing
swept in from the repo's noisy pre-existing `tmp/` state.

---

## 6. Commit SHAs

| SHA | Change |
|---|---|
| `36072f443` | `chore:` remove dead telemetry kit (3 files) |
| `9991126f9` | `chore(uv.lock):` advance mlx git pin to authoritative submodule SHA |
| `ad8bdb735` | `ci:` make the fused hc_expand Metal kernel guard actually run |
| `932fbc133` | `ci:` fail loudly when a submodule gitlink and its uv.lock pin diverge |

All on `main`, **not pushed**, no PRs opened. Working tree clean across `src/` and `uv.lock`;
only pre-existing unrelated `tmp/` noise remains. `nix fmt` **skipped — nix not installed**
on this machine (stated rather than implied); `ruff check` and `basedpyright` clean on all
touched files.

---

## 7. Still unaligned / follow-ups

1. **CI green not directly observed.** Local venv's mlx (`0.32.0.dev20260804+ac73d0c9`) matches
   neither pin. Needs `uv sync --extra mlx` or a real CI run to confirm.
2. **Two pre-existing collection errors** block full-suite runs: `test_moe_allsum_quant.py`,
   `test_routing_concurrency.py`. Round 2 flagged this same class. Not in round 3's scope.
3. **The 2-rank MoE all_sum script** has no pytest path at all — genuinely unreachable by
   `pytest src` without new multi-rank harness tooling.
4. **Launch-time alignment gate** deliberately not added (see §3 reasoning). Recommend as a
   warn-then-prompt check mirroring `start_cluster.sh:1088` if wanted.
5. **The slow-marked fork tests** (PP/distributed) are collected but never run in CI. Not a
   scope gap like `hc_expand` was, but nothing exercises them automatically either.

## 8. Process note for the next round

The headline finding is that **a supervisor-verified premise was wrong in a way that looked
authoritative**, and it cost real investigation time before being caught. The tell was cheap:
`git cat-file -t` on the quoted SHA returns exit 128 in one second. **Resolve every quoted SHA
to an object before reasoning about its history** — `git log A..B` returning empty and
`--is-ancestor` failing both ways are exactly what a nonexistent ref looks like, and they
mimic "divergent branches" convincingly. `git submodule status` has a status flag in column 1;
prefer `git ls-tree HEAD <path>` for parse-safe gitlink reads. This lesson is now encoded in
the guard's own implementation and docstring, not just in this report.
