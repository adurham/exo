# HARDENING ROUND 4 — REPORT

**Date:** 2026-09-03
**Repo:** `~/repos/exo` (origin = `adurham/exo`), branch `main`
**Headline:** CI green was **not** achieved — but for the first time it was **actually observed**, and the
reason it was never observed before is now known and fixed. All CI failures are **pre-existing and
proven not caused by rounds 2–4** via a control run.

---

## 1. Green-run evidence — THE ROOT CAUSE OF ROUNDS 2–3's BLIND SPOT

**Round 3 could not observe CI because the fork's GitHub Actions workflows were `disabled_manually`.**
Not "no runs for our SHAs" — the workflows were switched off entirely. Evidence at the start of this round:

```
$ gh api repos/adurham/exo/actions/workflows --jq '.workflows[] | "\(.name) | \(.state)"'
Build EXO macOS DMG        | disabled_manually
.github/workflows/debug.yml | disabled_manually
ci-pipeline                 | disabled_manually        <-- the pipeline that runs pytest

$ gh api 'repos/adurham/exo/actions/runs?per_page=1' --jq '{total,newest}'
{"total": 98, "newest": "2026-01-08T00:30:13Z"}        <-- newest run was ~8 months stale
```

Every pushed hardening commit returned `total_count: 0` runs (`25a7b0c79`, `932fbc133`, `ad8bdb735`,
`8158c0f52`, `910868756`). **No guard shipped in rounds 2–4 has ever executed in CI.**

### What was done
Re-enabled the pipeline (`gh api -X PUT .../workflows/pipeline.yml/enable`) and pushed a commit to
trigger a real run.

### Real CI result — run `33740154052` (commit `ba83460cc`), FIRST real run since 2026-01-08

| Job | Result | Failing step |
|---|---|---|
| Build and check (aarch64-linux) | failure | `Run nix flake check` |
| Build and check (x86_64-linux)  | failure | `Run nix flake check` |
| Build and check (aarch64-darwin)| failure | `Build all Nix outputs` |

**`Run pytest (macOS only)` was `skipped` on every platform** — CI dies before pytest is ever reached.
So the round 2–4 guards are still unexecuted in CI, now for a *known and different* reason.

Two distinct pre-existing blockers:
1. **`ruff-lint` derivation fails: `Found 1551 errors.`** (both Linux platforms). CI's ruff scope is
   wider than `src/` — it includes the vendored `mlx`/`mlx-lm` submodules.
2. **macOS `mlx` build fails patch application:** `mlx-0.32.1.dev20260903+e40a416b.drv` →
   `Hunk #1 FAILED at 177`, `Hunk #2 FAILED at 199`, `2 out of 2 hunks FAILED ... CMakeLists.txt.rej`.
   A repo patch no longer applies to the current mlx submodule source. This cascades into
   `exo-test-venv.drv`, which is exactly what `Run pytest` needs — hence the skip.

### Control run — proves round 4 did NOT cause this
Pushed the **pre-round-4** commit `910868756` to a scratch branch; CI run `33740619312`:

| Job | Result | Failing step |
|---|---|---|
| aarch64-linux | failure | `Run nix flake check` |
| x86_64-linux  | failure | `Run nix flake check` |
| aarch64-darwin| failure | `Build all Nix outputs` |

Same `Found 1551 errors.`, same `mlx-0.32.1.dev20260903+e40a416b.drv` failure. **Byte-identical failure
mode before and after round 4.** Scratch branch deleted after use.

**Verdict: CI is RED, was already red, and round 4 neither fixed nor broke it.** The honest status is
"CI observed for the first time; two pre-existing infrastructure blockers identified and located."

---

## 2. Environmental-vs-real failure split — ROUND 3's PREMISE WAS WRONG

Round 3 assumed ~6 local failures were environmental (local mlx `0.32.0.dev20260804` matching neither
pin). A full-suite triage at HEAD `1840c37ad` refuted this:

**`8 failed, 1098 passed, 4 skipped, 205 deselected` → 0 environmental / 8 real / 0 unclear.**

Not one traceback implicated an mlx version delta. Spot-verified independently, e.g.
`test_pp_metaframe.py` fails on `assert 4 == 3` — a hardcoded constant, zero mlx involvement.
All 8 were real test-fixture drift that would fail in CI too.

| # | Test | Root cause | Verdict |
|---|---|---|---|
| 1–2 | `test_routing_concurrency.py` | `ModelCard.backends` became required (`bc6661e6a`); test helper never updated | test stale |
| 3 | `test_pp_metaframe.py` | `METAFRAME_PROTOCOL_VERSION` 3→4 (`491d5ea35`, deliberate, with source migration) | test stale |
| 4 | `test_pp_speculation_cache_snapshot.py` | `PoolingCache.state` widened 3→5 tuple (mlx-lm `4bd3259`) | test stale |
| 5–6 | `test_concurrency_admission_gate.py` | test's fake lacked `set_prefill_cancel_probe()` that production calls | test stale |
| 7 | `test_event_ordering.py` | `MockGroup` duck-type violated `mx.distributed.all_min()` nanobind contract; `mx_any` mock hardcoded `False` caused an infinite hang | test stale (2 bugs) |
| 8 | `test_batch_generate_..._flag_off_smoke.py` | **cross-test global-state contamination** | **left failing on purpose — see §5** |

**Local suite after fixes: `1 failed, 1105 passed, 4 skipped, 205 deselected`** (independently re-run
by the PM). The single remaining failure is #8, deliberately not masked.

---

## 3. Collection errors — root cause + fix + proof

Both were module-level `ImportError`s, so these two files had **never guarded anything** and blocked
full-suite runs for everyone.

**`test_routing_concurrency.py`** — `from exo.routing.router import get_node_id_keypair`
Root cause: renamed to `get_node_zid` (now returning `NodeId` directly) in the libp2p→zenoh migration
`09f9ea313`; `git show` confirms the function body rewritten in place. Fix: updated the import and both
call sites to the real current API. **Not** a skip — the test genuinely runs now.

**`test_moe_allsum_quant.py`** — `cannot import name '_dequant_sum_shards'`
Root cause: **genuinely stale, not environmental.** `_dequant_sum_shards`/`_quantized_moe_all_sum` only
ever existed on the unmerged mlx-lm branch `feat/moe-allsum-quant-2026-08-19` (commit `ca5fc27`);
`git merge-base --is-ancestor ca5fc27 HEAD` confirms it is **not** an ancestor of the pinned submodule
`37260bbd`. `docs/moe-allsum-quant-root-cause-and-closure-2026-08-19.md` documents the approach as
abandoned ("mathematically dead" — hangs the real collective). CI's nix env uses the same pin, so the
symbols are missing everywhere. Fix: **deleted**, with that evidence. A permanently-erroring test is
worse than no test.

### Proof collection now succeeds
```
BEFORE: 1108/1313 tests collected (205 deselected), 2 errors
        !!!! Interrupted: 2 errors during collection !!!!
AFTER:  1110/1315 tests collected (205 deselected) in 2.74s     <-- 0 errors
```
Independently re-run by the PM. No tests lost (count went up).

---

## 4. MoE all_sum script disposition — manual-only, moved to `bench/`

`test_moe_allsum_sharedscale_distributed.py` sat in the test tree exposing **no test function** —
pytest collected the file and ran 0 tests, i.e. fake coverage.

Established facts: needs a real `mlx.launch -n 2 --backend ring`; imports
`_quantized_moe_all_sum_sharedscale`, which exists only on the unmerged mlx-lm branch
`feat/moe-allsum-sharedscale-2026-08-19`, so it would `ImportError` today even *with* 2 ranks; not
referenced by any live script, doc, or CI path.

**Decision: manual-only. Renamed to `bench/moe_allsum_sharedscale_repro.py`** (matching the existing
`bench/moe_allsum_quant_repro.py` convention), with an expanded docstring stating the exact
ImportError, the real invocation, and why CI cannot run it.

**Why not deleted like its sibling:** the sibling was provably dead. This one is not —
`docs/moe-allsum-sharedscale-CORRECTED-final-2026-08-19.md` closes the design only for the **prefill**
case and explicitly leaves **decode** as "a different, unexplored use case." Deletion isn't supported by
the evidence; a pytest entry point would be a permanent skip for a script that cannot import. Moving it
out of the test tree removes the false coverage signal while preserving the artifact.

---

## 5. The one deliberately-unfixed failure (root-cause discipline)

`test_exobatchgenerator_flag_off_matches_verified_pre_edit_baseline` fails ~1-in-3 **full-suite** runs
and passes **100% standalone** (PM verified: 3/3 standalone passes, plus a reproduced full-suite
failure).

It is **not** a stale golden. Each failure produces a *different* wrong token sequence, always starting
correct then diverging, under greedy decode (`temp=0`, explicit `seed=0`) — the signature of global
MLX/RNG state leaking from unrelated preceding tests.

**Re-recording the golden would have laundered a real cross-test contamination bug into a green suite.**
It was left failing and flagged. This is the one case where "still failing" is the correct outcome.

---

## 6. Formatting gate — answered

`nix fmt` invokes **treefmt** with 7 formatters (`flake.nix` lines ~107–131): `nixpkgs-fmt`,
`ruff-format` (excl. `rust/exo_rs/exo_rs.pyi`), `rustfmt`, `prettier` (`*.ts`/`*.svelte`),
`swift-format`, `shfmt`, `taplo`.

| Formatter | Runnable without nix? | Result |
|---|---|---|
| ruff-format | yes | **not clean** — 74 files under `src/` |
| rustfmt | yes | **not clean** — 6/14 `.rs` files |
| prettier | yes (dashboard-local) | clean |
| swift-format | yes (Xcode CLT, not nix's exact build) | **not clean** — 1 violation |
| taplo | **no** | unverifiable — 159 `.toml` files |
| shfmt | **no** | unverifiable — 15 shell scripts |
| nixpkgs-fmt | **no** | unverifiable — 7 `.nix` files |

**Verdict:** the tree is **not** formatting-clean; 3 of 5 locally-runnable formatters report real diffs,
and 3 formatters (181 files) cannot be checked at all without nix. This is pre-existing debt, deliberately
not mass-fixed — a blanket reformat would produce a 74-file unreviewed diff.

**One regression WAS caught and fixed:** `test_event_ordering.py` was format-clean at `910868756` and was
left unformatted by a round-4 edit. Verified against the baseline and corrected in `dcf07980d`. The other
touched files (`test_concurrency_admission_gate.py`, `test_pp_speculation_cache_snapshot.py`) were
already unformatted before round 4 — left as-is.

Note: CI's `ruff check` scope (`1551 errors`) is far wider than `src/` alone (`350 errors`) because it
includes the vendored submodules — relevant to whoever tackles blocker #1.

---

## 7. Commit SHAs (all pushed to `origin/main`)

| SHA | What |
|---|---|
| `1840c37ad` | fix: resolve 2 pytest collection errors blocking full-suite runs |
| `874134e0b` | fix: routing-concurrency ModelCard backends drift; disposition sharedscale repro |
| `e04c6b8bc` | fix: add missing backends field; expand sharedscale repro docstring |
| `71c8deedc` | docs(hardening-round4): full-suite failure triage + nix-fmt-without-nix evidence |
| `a9e56bed9` | fix(tests): update 4 stale test fixtures behind deliberate source changes |
| `dcf07980d` | style: ruff-format test_event_ordering.py (round-4 regression) |
| `ba83460cc` | ci: trigger pipeline to obtain real green-run evidence (round 4) |

Supporting evidence: `tmp/hardening-round4-20260903/evidence.md`.
CI runs: `33740154052` (round-4 HEAD), `33740619312` (round-3 control).

---

## 8. What remains unverifiable / open

1. **CI still cannot reach pytest.** Two pre-existing blockers must be fixed first:
   (a) `ruff-lint` `1551 errors` incl. submodules; (b) the macOS mlx `CMakeLists.txt` patch no longer
   applying (`2 out of 2 hunks FAILED`). Until then no guard from rounds 2–4 has *ever* executed in CI.
2. **The flaky isolation bug (§5)** — real, reproducible, unfixed by design.
3. **taplo / shfmt / nixpkgs-fmt** (181 files) — not checkable on this machine.
4. The macOS mlx build failure means **CI's pytest env cannot currently be built at all**, so the
   "does the local mlx mismatch matter in CI" question is still formally open — though the triage in §2
   makes it moot for these 8 failures, all of which were version-independent.
