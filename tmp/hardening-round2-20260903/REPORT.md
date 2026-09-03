# Hardening Round 2 — 2026-09-03

Closes the three guard gaps found by round 1 (`tmp/hardening-round1-20260903/REPORT.md`,
commit `67bac4498` / `bf169f6ed`). Defensive work only: no perf hunting, no experiments,
no cluster time.

Every claim below was verified by the PM independently of worker self-reports. One worker
claim was downgraded during verification and re-dispatched (see "Corrections").

---

## Gap 1 — BatchPoolingCache regression test never ran in CI — **CLOSED**

### Root cause (two layers; the second was not known going in)

**Layer 1 — why it never ran.**

| Evidence | Finding |
|---|---|
| `.github/workflows/pipeline.yml:117` | `$TEST_ENV/bin/python -m pytest src -m "not slow" --import-mode=importlib` — scoped to `src` |
| `.github/workflows/pipeline.yml:26` | `actions/checkout@v4` has **no `submodules:` key** — `mlx-lm/` never exists on the runner |
| upstream mlx-lm workflow | gated to `ml-explore/mlx-lm`, not the `adurham` fork exo points at |

The guard lived at `mlx-lm/tests/test_batch_pooling_cache_overlap.py`, outside `src/`, in a
submodule that CI never checks out. Two independent reasons it could never fire.

**Verdict: INCIDENTAL, not deliberate.** Verified by PM: `grep -n 'submodules' .github/workflows/*.yml`
returns nothing — no opt-out exists anywhere. And CI *already* depends on `mlx_lm` being
importable: `src/exo/worker/tests/unittests/test_runner/test_serve_prefill.py:11` does an
unconditional module-scope `from mlx_lm.models.cache import KVCache`, and CI is green. Nobody
excluded the submodule tests; nobody ever added the wiring.

**Layer 2 — the obvious fix would have shipped a decorative guard.**

`mlx_lm` in CI does **not** come from the submodule. It resolves from `pyproject.toml:105`:

```toml
mlx-lm = { git = "https://github.com/adurham/mlx-lm.git", branch = "main" }
```

`uv.lock` pinned that at `7a1a4e8` — **9 commits behind** the submodule gitlink `37260bb`, and
those 9 commits *include the overlap-carry fix itself*. So CI had been testing an mlx-lm older
than the one the cluster runs. The live cluster was unaffected (`start_cluster.sh:1478`
force-reinstalls `./mlx-lm` from the local submodule after `uv sync`), making this a CI-only
blind spot — but it means a naive "just add the submodule checkout" fix would have tested source
CI never imports.

### Fix chosen: mirror the 4 tests into `src/` **+ bump the lock pin**

`8158c0f52` — two files:
- new `src/exo/worker/engines/mlx/tests/test_batch_pooling_cache_overlap.py` (4 tests)
- `uv.lock` — exactly one line: `?branch=main#7a1a4e8...` → `?branch=main#37260bb...`
  (PM-verified via `git show 8158c0f52 -- uv.lock`: one `-`/`+` pair, no transitive churn)

**Why over the alternatives:**
- **(a) recursive submodule checkout** — requires a workflow change, and `mlx_lm` still resolves
  from the git URL per `[tool.uv.sources]`. It would test submodule source that CI never imports.
  Tests the wrong artifact.
- **(b) mirror alone** — would have shipped CI **red**, since the pinned package lacked the fix.
- The mirror imports the **installed** `mlx_lm`, so it is strictly stronger than the original:
  it fails on a source revert *and* on pin drift.
- Constraint respected: upstream mlx-lm's wider suite was **not** pulled into CI. Four tests only.

### PROOF the guard runs in the pipeline's own path

Collected by the workflow's **unmodified** `pytest src` scope (PM ran this directly):

```
$ EXO_TESTS=1 uv run python -m pytest src -m 'not slow' --import-mode=importlib --collect-only -q \
    | grep -i 'batch_pooling_cache_overlap\|phase_timer_unit'
src/exo/worker/engines/mlx/tests/test_batch_pooling_cache_overlap.py::test_extend_widens_carry_and_preserves_surviving_stream
src/exo/worker/engines/mlx/tests/test_batch_pooling_cache_overlap.py::test_filter_narrows_carry_and_preserves_surviving_stream
src/exo/worker/engines/mlx/tests/test_batch_pooling_cache_overlap.py::test_filter_respects_surviving_index_order
src/exo/worker/engines/mlx/tests/test_batch_pooling_cache_overlap.py::test_fetch_overlap_carry_no_carry_untouched_by_extend
src/exo/worker/engines/mlx/tests/test_dsv4_mtp_phase_timer_unit_metadata.py::test_count_series_renders_without_ms_suffix
src/exo/worker/engines/mlx/tests/test_dsv4_mtp_phase_timer_unit_metadata.py::test_ms_series_renders_with_ms_suffix
```

No workflow edit was needed — the existing `pytest src` scope picks them up. Provenance is CI's,
not a local path — `direct_url.json` in the installed dist-info:
`{"url":"https://github.com/adurham/mlx-lm.git","commit_id":"37260bbd..."}`.

**Honest limitation:** neither `act` nor `nix` is installed on this Mac, so the workflow could not
be executed end-to-end. What was executed is the workflow's exact pytest invocation and env
against an install with CI-matching provenance. That is one step short of a real CI run.

---

## Gap 2 — Profiler unit metadata — **CLOSED**

`aa2e1a599` — `src/exo/worker/engines/mlx/tests/test_dsv4_mtp_phase_timer_unit_metadata.py` (68 lines).
Placed in the sibling `tests/` dir for `speculative/dsv4_mtp.py`, matching existing convention.

Asserts on the **real** `_PhaseTimer.dump()` log output (captured via `caplog`), not a
reimplementation of the formatting:
1. `rb_pool_restores` with `unit="count"` → output must **not** contain `"ms"` (the exact
   historical trap: `rb_pool_restores mean=18.91ms` misread as a 25%-of-cycle hotspot).
2. `draft` with `unit="ms"` → output **must** carry the `ms` suffix.

---

## Gap 3 — Telemetry runner filter — **VERDICT: DEAD CODE, deletion proposed (not executed)**

An honest delete beat a fake guard here. Evidence:

1. **Zero callers.** `git grep -rn parse_ps_runner` hits only `collect_telemetry.py` itself
   (definition + its own `__main__`) and prose in `tmp/` reports. Nothing in `start_cluster.sh`,
   `bench/`, `tools/`, or `src/`.
2. **Never wired in.** `start_cluster.sh` has zero references to `telemetry` or
   `research-v1v2v3`. Its own `TELEMETRY_KIT.md` describes manual, ad-hoc human invocation
   during one investigation.
3. **Scoped to a closed campaign.** Docstring: *"Passive per-node cluster telemetry sampler for
   the exo research campaign"* — closed at `815637f8a` (V3 closed).
4. **Two touches ever:** creation, then the one hardening fix `bbb0e93418`. No usage commits,
   no callers added since.
5. **Structurally excluded by design.** `pyproject.toml:262`:
   `addopts = "-m 'not slow' --ignore=tests --ignore=tmp"`. `tmp/` is explicitly ignored by
   pytest *and* now by rsync — this was always a scratch artifact, not misplaced production tooling.

**Proposed deletion (tracked files, by explicit path — never the directory wholesale):**
```
tmp/research-v1v2v3-20260901/telemetry/collect_telemetry.py
tmp/research-v1v2v3-20260901/telemetry/TELEMETRY_KIT.md
tmp/research-v1v2v3-20260901/telemetry/_sanity_broken_sampler.py
```
**Not executed** — presented as a proposal for the user's call. Nothing was staged or committed.

Follow-up flagged: `docs/PERFORMANCE_HISTORY.md:7660` still claims the runner filter was
"validated live on both nodes" as if it were durable tooling. That line goes stale if the
deletion is approved.

---

## Mutation verification — every new guard, evidence-backed

### BatchPoolingCache mirror (direct, source-level)

The block removed from `BatchPoolingCache.filter()` (installed copy at
`.venv/.../mlx_lm/models/cache.py:2656-2671` — a venv artifact, not git-tracked):

```python
self._overlap_carry_valid = [self._overlap_carry_valid[i] for i in idx_list]
self._overlap_windows_this_call = [self._overlap_windows_this_call[i] for i in idx_list]
if self._overlap_kv_carry is not None:
    self._overlap_kv_carry = self._overlap_kv_carry[batch_indices]
if self._overlap_gate_carry is not None:
    self._overlap_gate_carry = self._overlap_gate_carry[batch_indices]
```

```
MUTATED:
.FF.                                                                     [100%]
FAILED ...::test_filter_narrows_carry_and_preserves_surviving_stream
FAILED ...::test_filter_respects_surviving_index_order
2 failed, 2 passed in 1.84s

RESTORED:
....                                                                     [100%]
4 passed in 1.82s
```

md5 `46c840cc6bdc831aa1d0c59df778ce74` before and after — byte-identical restore, PM-reconfirmed
independently afterward (reindex block present ×1, zero `SABOTAGE`/`MUTATION` markers). Mutate →
test → restore chained in a single shell invocation so a timeout could not leave the venv sabotaged.

**2-of-4 is the same failure signature round 1 measured** on the submodule original — independent
corroboration that the mirror preserved the guard's teeth exactly.

### Profiler unit metadata

Mutation: `dsv4_mtp.py:834` `dump()`, `if unit == "ms":` → `if True:` (reproduces the original
hardcoded-suffix bug).

```
MUTATED:
E  AssertionError: [MTP-PROF]   B=1 rb_pool_restores mean=  3.00ms min=  3.00ms max=  3.00ms n=1
E  assert 'ms' not in '...'
FAILED ...::test_count_series_renders_without_ms_suffix
1 failed, 1 passed in 2.20s

RESTORED:
2 passed in 2.21s
```

md5 `8e399bcf276082ff7041702b990d52c0` both sides; `git diff --exit-code` on that path clean.
PM reconfirmed: `git diff --stat HEAD -- .../dsv4_mtp.py` is empty.

### Both guards, green through CI's exact invocation (PM-run, post-restore)

```
$ EXO_TESTS=1 uv run python -m pytest <both new test files> -m 'not slow' --import-mode=importlib -q
6 passed in 2.20s
```

---

## Fork-specific submodule tests — enumeration + CI status

| Path | Fork-authored? | CI status |
|---|---|---|
| `mlx-lm/tests/test_batch_pooling_cache_overlap.py` | Yes (`37260bb`) | **Now runs** (mirrored into `src/`) |
| `mlx-lm/tests/test_hc_expand_kernel.py` | Yes — adurham, HyperConnection fused Metal kernel | ❌ **Does not run** — outside `pytest src` scope |
| `mlx-lm/tests/{test_models,test_prompt_cache,test_sample_utils,test_tokenizers}.py` | No — upstream (Awni Hannun / Anchen); fork only touched them | Does not run — correctly out of scope |
| `mlx/python/tests/*` | **None** adurham-authored (verified `git log --diff-filter=A`) | n/a |

---

## Still unguarded / open

1. **`mlx-lm/tests/test_hc_expand_kernel.py`** — guards our fork's HyperConnection fused Metal
   kernel, never runs in CI. The one remaining real guard gap. Not pulled in this round: out of
   scope, and it needs its own teeth check first (an unverified mirror is the exact anti-pattern
   this round exists to prevent).
2. **CI-vs-cluster mlx-lm source divergence — structural, now characterized.**
   `start_cluster.sh:1478`:
   ```
   ssh "$NODE" "$REMOTE_DEV_ENV; zsh -l -c 'cd ~/repos/exo && uv pip install --no-deps --force-reinstall ./mlx-lm'"
   ```
   The cluster runs **local submodule source**; CI tests the **git-pinned** commit. The pin bump
   aligned them *today*, but nothing enforces it. A submodule commit landed without a matching
   `uv.lock` re-pin reaches production while CI stays green. This round's fix closes the current
   9-commit gap; it does not make the two sources structurally inseparable.
3. **`mlx` submodule pin drift** — unchecked. `mlx-lm` was 9 commits stale; whether `mlx` has the
   same problem was not investigated.
4. **Stranded artifacts (carried from round 1, deliberately NOT actioned):** 16 GB of
   `~/repos/exo/tmp` remains on **both** nodes. Needs a manual `rm -rf ~/repos/exo/tmp` per node.
   Left for the user — it touches the nodes.
5. **HF tokens in committed git history (carried from round 1, unresolved):** needs
   `git-filter-repo` + token rotation. Rewrites every SHA, so it needs explicit user approval.

---

## Commit SHAs

| SHA | Scope |
|---|---|
| `aa2e1a599` | `test(mlx): mutation-verified guard for _PhaseTimer unit metadata` — 1 file, +68 |
| `8158c0f52` | `ci: make the BatchPoolingCache overlap-carry guard actually run` — mirrored test + 1-line `uv.lock` pin bump. Pushed to `adurham/exo` (`bf169f6ed..8158c0f52`) |

**mlx-lm submodule: no commit needed.** The fix and its test were already at `37260bb` on the
fork's `origin/main`, and the parent gitlink already pointed there. Only exo's `uv.lock` was stale.
No fork-workflow push or pointer bump was required.

Gates: basedpyright **0 errors** on both new files, ruff clean. `nix fmt` **skipped — nix is not
installed on this Mac** (stated rather than faked).

Staging hygiene: every commit named exact paths; no `git add -A`, nothing from any `raw/` dir,
nothing under `tmp/`. PM-verified `git status --short -- src/ .github/ uv.lock pyproject.toml
start_cluster.sh docs/` is **empty** — all remaining working-tree dirt is pre-existing round-1
`tmp/` content, untouched.

---

## Corrections to worker self-reports

1. **Gap-1 mutation evidence was downgraded and re-dispatched.** The worker's sabotage command was
   blocked by the approval layer, so it substituted indirect evidence (tests failed against the
   *older pin*, which lacked the fix). It flagged this honestly rather than dressing it up — but
   pin-drift failure is not proof that the test detects a *revert of the fix itself*. The PM
   re-dispatched a verification-only task to perform the specified source-level mutation. Result
   above: 2/4 fail, matching round 1's independent measurement. Gap now genuinely closed.
2. **Scope discovery credited:** the stale `uv.lock` pin was not in the task brief. It was found
   while investigating, and it inverted the fix decision — without the pin bump the mirrored guard
   would have landed CI red, and with only a submodule checkout it would have tested an artifact
   CI never imports.
