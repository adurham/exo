# DSv4.1 spec-rollback ring-alias fix (M1) — 2026-10-14

**Status: fix written, tested, pushed. NOT deployed, NOT relaunched, no cluster
touched.** Reviewer: this doc is written to be falsified — see
"How to falsify this" at the end, and the NOT-verified list.

| item | value |
|---|---|
| mlx-lm branch | `fix/dsv41-spec-rollback-ring-2026-10-14` — tip **894815ae54d7ac4ff1845d36903bc6224d9f826d** (pushed to `origin` = adurham/mlx-lm); chain: fix **c4d19826118a8da823a33e0115d9ea71f933dcf9**, bench call-site sweep `08a4f45`, sweep completion (p67 replay oracle + pW21/pW24 + the archived exo probes) `894815a` |
| mlx-lm base | `a0573e3` (the current live pin; **not** an ancestor of `origin/main` @ 6cc9c1e — the pin is on the deploy lineage, which is the point) |
| exo branch | `fix/dsv41-spec-rollback-ring-2026-10-14` off `a5d33b46b` (live deploy tip `deploy/next26-limiter`) — one-line call-site update + this doc. Branch tip = the commit carrying this file. |
| bug source | `docs/dsv41-state-machinery-audit-2026-10-14.md` M1 (`audit/dsv41-state-machinery-2026-10-14` @ c1841c287) |
| worktrees used | `~/.hermes/cache/scratch/mlx-lm-ringfix` (mlx-lm), `~/.hermes/cache/scratch/exo-ringfix-wt` (exo). Both main checkouts left clean/untouched. |

## 1. The bug

`spec.rollback` (`mlx_lm/models/deepseek_v41/spec.py`) reset `cache.offset =
target` and rebuilt the compressor carry. It touched no window ring.

The ring is **position-addressed**, slot `p % window` (`cache.py:121-138`), so a
write at `p` lands on the slots of the already-committed `p - window`. A verify
forward reads the window at `attention.py:161` (`window_chrono`, before any
write) and then writes **all** `gamma + 1` rows at `:172` (`write_window`). So
whenever `gamma - accepted >= 2`, the rejected rows alias *committed* positions,
and the **next** forward's attention runs over rejected-draft KV.

`cache.py:142-150` (`ring_snapshot`) documents exactly this hazard, which is why
session checkpoints copy the ring (`session_cache.py:367-374`, restored at
`:412-413`) and why `prefill.decode_prime` saves/restores the ring slot by hand
(`prefill.py:301-333`). Only the speculative rollback was left out, and its own
docstring licensed it with a false claim: *"window ring and compressed caches are
position-addressed and are rewritten before they are read again"* — false for the
ring.

**Not bounded by the ring's self-overwrite.** On a stack with a layer above the
first, the corrupted forward's hidden states write that layer's compressed /
index caches for positions that are committed and never recomputed. So the
perturbation is *permanent per conversation*, not a one-forward blip — the
strongest form of warm ≠ fresh. A one-layer fixture cannot see that channel at
all.

Reproduced (pre-fix, unchanged pinned API):

```
PYTHONPATH=~/repos/exo/mlx-lm ~/repos/exo/.venv/bin/python -m pytest -q \
  ~/.hermes/cache/scratch/dsv41_spec_ring_alias_repro.py \
  ~/.hermes/cache/scratch/dsv41_spec_ring_alias_W128_repro.py
-> 4 failed, 4 passed   (acc=0,1 red; acc=2,3 green)
   W=128 acc=1 max|diff| = 0.007388602942228317 (row 0)
```

## 2. The fix (option (a): bounded pre-verify slot copy)

`spec.snap(cache, start_pos, verify_rows)` **now requires the row count of the
next forward** (`1 + gamma` at every call site) and additionally copies, per
layer:

* the `min(verify_rows, window)` ring rows that `write_window` is about to
  overwrite — the exact same slot set, from one shared definition
  (`LayerCache.window_write_slots`, which `write_window` now routes through);
* the `comp_kv` / `index_k` rows the chunk's closed groups will occupy
  (`[start_pos // ratio, end_pos // ratio)`).

`spec.rollback` then:

1. refuses (`RuntimeError`) if the forward wrote past the snapshot span — the
   only way to get rejected rows stranded is a caller passing the wrong
   `verify_rows`, and that must fail loudly, not corrupt silently;
2. restores the **rejected** ring slots (`[max(target, first), written_end)`),
   leaving the accepted rows' slots at the verify value;
3. restores the compressed / index rows **at or above** `target // ratio` (rows
   below were built from carried + accepted rows only and are what the next
   forward reads — they keep the verify value);
4. rebuilds the carry exactly as before (`_rebuild`), and canonicalizes the carry
   rows above the live count (`reset_carry` on a group-boundary target),
   so no rejected row survives anywhere in the buffer.

`spec.carries(cache, start_pos)` is the old carry-snapshot body extracted, so
`session_cache.snapshot` keeps the identical format without inventing an
argument (`_spec.snap(cache, pos)` -> `_spec.carries(cache, pos)`).

Ring rows are gathered (`mx.array(view)`), never aliased, and **not** evaluated
inside `snap`: an `mx.eval` there would materialise the lazily drafted tokens
before the verify forward and destroy the deliberate one-sync-per-round overlap
(`spec.generate`'s docstring; the audit's "one host sync per round"). MLX
read-before-write semantics across a gather built before a later in-place write
were probed directly (`~/.hermes/cache/scratch/_mlx_alias_probe.py`) and are
pinned end-to-end by the new tests.

Files (mlx-lm, commit c4d1982):

| file | change |
|---|---|
| `mlx_lm/models/deepseek_v41/cache.py` | new `window_write_slots`; `write_window` routes through it; `ring_snapshot` cross-references the rollback path; stated copy size corrected to ~128 KB bf16 (was ~64 KB) |
| `mlx_lm/models/deepseek_v41/spec.py` | module docstring rewritten (the false claim removed); `RollbackSnapshot`; `carries()`; `snap(cache, start_pos, verify_rows)`; `rollback` ring + compressed-row restore, carry canonicalization, span guard; `_rebuild` takes `need` (unused `ratio` dropped) |
| `mlx_lm/models/deepseek_v41/sampling.py` | `sp.snap(cache, pos, g + 1)` |
| `mlx_lm/models/deepseek_v41/session_cache.py` | `_spec.carries(...)`; docstring corrected (it repeated the false claim; sizes corrected) |
| `mlx_lm/models/deepseek_v41/prefill.py` | `decode_prime` bit-exactness note corrected (it described the same fallacy while its own code saves the ring slot) |
| `tests/test_dsv41_sampling.py` | `_StubSpec.snap(cache, start_pos, verify_rows)` |
| `tests/test_dsv41_spec_rollback_ring.py` | NEW — 17 cases (below) |

exo (separate repo, separate commit): `dsv41/rounds.py` -> `spec.snap(cache,
position, 1 + gamma)` with a comment; plus this doc. **Nothing else in exo
consumes `spec.rollback`/`spec.snap`/`spec.stashes`** (grep of
`src/exo/worker/engines/mlx/dsv41/`: only `rounds.py`; `park.py:711` reads the
*carry row* format, which is unchanged). The exo side is a call-site consistency
change, not a second fix.

### Call-site sweep (mlx-lm commit 08a4f45)

15 offline bench probes also called `SP.snap(cache, pos)` with two arguments and
would have failed the new required-argument contract: `bench/pDC_model_ab.py`,
`bench/pU0_memprobe.py`, `pU2_divergence`, `pU3_decompose`, `pU4_collectives`,
`pU5_identity`, `pU6_collectives`, `pU7_two_rank`, `pU8_census`, `pV2_census`,
`pV4_verify`, `pV6_phases`, `pV7_attrib`, `pV13_parity`, `pV18_attn`. Each now
passes the width of the chunk it actually feeds (`vin.shape[1]`, `len(chunk)`, or
`R`); `pU6`/`pU7` snap *after* their own verify forward (they measure rollback
cost, not state) so they pass `1 + GAMMA`, which is that round's row count and
what makes the span check pass. Sweep is mechanical: anchors asserted unique per
file, all 15 files compile, `grep -rn "SP.snap(cache, pos)" bench` is empty, ruff
12 errors before and 12 after (zero NEW). **They were not executed** (model
package dirs; pU4/pU6/pU7/pU8 also need the cluster hostfile).

### Sweep completion (mlx-lm commit 894815a) — the audit's one required change

The adversarial audit found 08a4f45's grep covered only `bench/`, leaving three
stale old-API call sites. All are now fixed (commit `894815a`, mlx-lm):

| file:line | old call | new call | why this width |
|---|---|---|---|
| `tests/p67_serve.py:571` | `SP.snap(c, off)` | `SP.snap(c, off, len(rows))` | the replay loop's `rows` is the recorded chunk and the forward fed two lines below is `rows[None]`, i.e. exactly `len(rows)` rows — exact span, replay semantics unchanged. **This is the audit's required change.** |
| `benchmarks/dsv41/pW21_whathelps.py:148,155` | `SP.snap(cache, pos)` | `SP.snap(cache, pos, 1)` | arm E snapshots before a single-token `step` (`model(nxt[None], ...)`, `nxt` shape `(1,)`) then rolls back to `pos`. Write width 1. |
| `benchmarks/dsv41/pW24_absorb.py:87` | `SP.snap(cache, pos)` | `SP.snap(cache, pos, 1)` | arm `warmstep`: same single-token throwaway step + rollback. Write width 1. |

Sweep method (so the "all callers" claim is auditable): an AST pass over both
repos that reports every `Call` whose callee attribute or name is
`snap`/`rollback`/`stashes`/`carries` — aliases and wrapped lines cannot hide a
stale site the way a fixed-spelling grep can. Post-sweep, **zero** `snap(` call
sites anywhere carry fewer than 3 positional args (80 call sites in mlx-lm, 28 in
exo). The lone 1-argument `rollback(` in exo is
`mlx/speculative/mtp_batch_generator.py:346` `c.rollback(n_accepted)` — a
cache-object method of the generic MTP generator, unrelated to `spec.rollback`.

### The five archived exo bench scripts — updated (path chosen: update)

The five archived exo probes that still called `SP.snap(cache, pos)`:
`docs/benchmarks/phase16-dsv41-dspark-spec-2026-09-28/scripts/p56_spec.py`,
`phase17-dsv41-verify-cost-2026-09-28/scripts/p48_bench.py`,
`phase17-dsv41-verify-cost-2026-09-28/scripts/p56_spec.py`,
`phase18-dsv41-adaptive-gamma-2026-09-28/scripts/p63_spec2.py`,
`phase21-dsv41-context-scaling-2026-09-29/scripts/p66_ctxscale.py`.

**Chosen path: update them (not leave + document).** The correct argument is
unambiguous at every site from the script's own context — each snapshot sits
directly above the forward whose width it must cover:

| file:line | new call | width source |
|---|---|---|
| phase16/17 `p56_spec.py:148`, phase18 `p63_spec2.py:148` | `SP.snap(cache, pos, vin.shape[1])` | `vin = mx.concatenate([nxt.reshape(1,1), d], axis=1)` on the line above; `= 1 + gamma` |
| phase16/17 `p56_spec.py:184`, phase18 `p63_spec2.py:184` | `SP.snap(cache, pos, R)` | `vin = mx.array([[int(nxt.item())] * R], ...)` on the line above (the `vcost` sweep) |
| phase17 `p48_bench.py:116` | `SP.snap(cache, pos, ROWS)` | `ROWS = int(os.environ.get("P48_ROWS", "1"))`; the branch is `if ROWS > 1:` and feeds `[chunk]` of length `ROWS` |
| phase21 `p66_ctxscale.py:46` | `SP.snap(cache, pos, rows)` | `t_chunk(cache, rows, ...)` feeds `mx.array([base[:rows]])` on the next line |

Rationale: these are archived records, and fail-loud on a stale API is acceptable
for an archive — but an unambiguous one-argument change keeps the archive
*executable* at no interpretive risk, which is strictly better than leaving five
known-broken scripts in-tree. They are still **not executed here** (they need the
model package dirs; the pU4/pU6/pU7/pU8-class probes also need the cluster
hostfile): compile-checked (`py_compile`) and reviewed-by-anchor only.

## 3. The invariant, stated exactly

For **every** acceptance count `m` in `0..gamma` (a full reject included), two
runs that differ **only** in the content of the rejected draft rows end
bit-identical in `cache.offset`, every layer's ring, `comp_kv`, `index_k`, and
compressor carry, and produce bit-identical rows from the forward that follows
the rollback.

Scope and the one place the claim is weaker than "identical to a greedy decode":
the verify forward's chunk shape differs from token-at-a-time greedy, so the
**accepted** positions' ring values legitimately differ from a greedy decode's
(pre-existing, documented at the top of `spec.py`). The fix does not change that
and does not claim to. What it does claim, and what the tests assert per-slot, is
that the rejected rows leave **no** trace: for each accepted slot the value is
still the verify's, for each rejected slot it is byte-identical to what was
there before the verify (which is the committed `p - window` KV a
never-speculated decode would read).

Also **not** covered by the fix or the tests: `engram_ids` (a plain linear
append-only host array, no aliasing — rejected rows there occupy positions above
the offset and the existing session-rewind design already tolerates that), and
the DSpark draft window (it is fed explicitly, `head.append_ctx(taps[:, :n+1])`,
accepted rows only — audit item 10).

## 4. Design alternatives and why (a)

* **(a) bounded pre-verify slot copy — CHOSEN.** Cost is bounded by
  `min(verify_rows, window)` ring rows + `ceil(verify_rows/ratio)+1` cache rows
  per layer, paid only on the speculative path, no unbounded allocation, and the
  copy is a plain gather of the exact slots `write_window` uses (one definition).
  The restore is selective (rejected only), so committed KV is never undone.
* **(b) route rejected rows to a scratch ring — rejected.** The verify forward's
  attention reads the chunk rows from the freshly computed `kv`, not from the
  ring (`attention.py:163`), so the ring write is *only* persistence. A scratch
  ring would therefore need per-layer scratch storage plus moving the accepted
  rows back into the live ring — more state, more code, no benefit.
* **(c) full-ring copy every round — rejected on cost.** Correct, but 40 x 128 KB
  = 5.2 MB per round and it would have to be materialised (a sync), where the
  actual write set is <= 5 rows. (This is the form `session_checkpoint` uses, and
  the reason it is affordable there is that checkpoints are rare.)
* **(d) post-hoc changed-slot mask diff — rejected.** Needs an elementwise
  compare over ~2.6 M elements per round and is subtler (a coincidentally-equal
  rejected row is a no-op, so it works, but cost + subtlety lose).
* **(e) save inside `write_window` on every write — rejected.** Pays on every
  prefill chunk (up to `window` rows/layer/chunk) and on every decode; the cost is
  not scoped to the spec path, violating "hot-path cost only when needed".

## 5. Cost

Per speculative round, per layer: `min(verify_rows, window)` ring rows
(`min(5, 128) = 5` at gamma=3 -> 5 x 512 x 2 B = 5 KB) plus
`ceil(verify_rows/ratio)+1` compressed rows for the two cache buffers
(<= 3 x 512 x 2 B = 3 KB each at ratio 2). At 40 layers that is ~0.2-0.4 MB of
gather per round, no new allocation that grows with context, no added host sync.
Restore writes are the same order. The guard is two integer comparisons.

## 6. Evidence

New suite, `mlx-lm/tests/test_dsv41_spec_rollback_ring.py` (**17 passed**), 2
attention layers over a real `ModelCache` (both layers kv+index sources, so the
upper layer owns a compressor and writes the compressed caches from
stale-derived hidden states — the channel a 1-layer fixture cannot see):

```
cd ~/.hermes/cache/scratch/mlx-lm-ringfix
PYTHONPATH=$PWD ~/repos/exo/.venv/bin/python -m pytest -q -p no:cacheprovider \
  tests/test_dsv41_spec_rollback_ring.py
```

Cases: `test_rejected_drafts_leave_no_trace[acc x W]` for acc in 0..3 (acc 0 =
full reject) and W in {8, 128}; `test_two_layer_compressed_caches_content_
independent[W]`; `test_accepted_rows_are_kept_and_rejected_slots_restored[W]`
(per-slot: rejected slots restored to pre-verify bytes, accepted slots keep the
verify value, plus a vacuity check); `test_rejected_drafts_do_not_survive_in_the
_compressor_carry[ratio,pos,acc]` (ratio 3 and 4, chosen so the rejected rows
actually land in the carry and survive the next forward);
`test_rollback_refuses_a_snapshot_that_does_not_cover_the_write`;
`test_window_write_slots_matches_the_write_it_describes`;
`test_control_without_the_ring_restore_the_invariant_does_break` (negative
control: patches `spec._restore_ring` away and asserts the runs then DIFFER in
the low layer's ring, in the following forward's output, and in the upper layer's
`comp_kv` *and* `index_k` — proving the assertions are load-bearing).

**Sabotage check** (`~/.hermes/cache/scratch/m1-ringfix-sabotage.py`, rewrites
`spec.py` in place, runs, restores; md5 verified identical after each run):

| arm | what is reverted | result |
|---|---|---|
| A | whole rollback body -> pre-fix behaviour | **11 failed / 6 passed** |
| B | only the ring restore | **10 failed / 7 passed** |
| C | only the compressed-row restore | **4 failed / 13 passed** |
| D | only the carry canonicalization | **2 failed / 15 passed** |
| — | fix in place | **17 passed** |

Every component of the fix is therefore load-bearing for at least one test.

**Pre-fix red (the pinned API, the audit's own repros): 4 failed / 4 passed**
(command in section 1). Post-fix the same scenarios are green (they are the
ported tests).

**No regression, scoped suites:**

```
# mlx-lm (from the mlx-lm worktree)
PYTHONPATH=$PWD ~/repos/exo/.venv/bin/python -m pytest -q -p no:cacheprovider \
  tests/test_dsv41_sampling.py tests/test_dsv41_session_cache.py \
  tests/test_dsv41_snapshot_prune.py tests/test_dsv41_cache_growth.py \
  tests/test_dsv41_prefill_spans.py
-> 67 passed

# exo (from the exo worktree, fixed mlx_lm first on the path)
EXO_TESTS=1 EXO_DASHBOARD_DIR=$HOME/repos/exo/dashboard/build \
PYTHONPATH=$PWD/src:$PWD/mlx-lm ~/repos/exo/.venv/bin/python -m pytest -q \
  -p no:cacheprovider \
  src/exo/worker/engines/mlx/dsv41/tests/test_dsv41_engine.py \
  src/exo/worker/engines/mlx/dsv41/tests/test_dsv41_reasoning_budget.py
-> 76 passed          # these drive rounds._one_round against the REAL spec module
EXO_... same, plus src/exo/worker/engines/mlx/dsv41/tests/test_dsv41_session.py \
  src/exo/worker/engines/mlx/dsv41/tests/test_dsv41_degeneration.py
-> 79 passed
```

Gates: `ruff check` (mlx-lm `ruff.toml` and exo `pyproject.toml`) — **4
pre-existing errors before and after** on the mlx-lm files (unchanged set), new
test file clean, exo `rounds.py` clean before and after. `ruff format --check`
drift is pre-existing and untouched: 26 of 28 files in
`mlx_lm/models/deepseek_v41/` already drift, and the pinned copies of all six
touched files already drift, so there is no format regression (and `nix fmt` is
not installed on this Mac). `basedpyright` was not run: it is the known
pre-existing repo-wide blocker (~13k errors, `bench/` alone 8k) and adding ~1
call site cannot move it; say so rather than implying a type gate passed.

## 7. How to falsify this

1. **Kill the claim at its root.** In `spec.rollback`, delete the
   `_restore_ring(...)` call: the new suite must go red (it does — arm B), and
   `test_control_...` already asserts the leak exists without the restore. If
   any of `test_rejected_drafts_leave_no_trace` stayed green there, the test
   fixture does not actually exercise the aliasing.
2. **Attack the "one definition" claim.** `write_window` and
   `window_write_slots` must agree for `n` >= window too (the self-overwriting
   chunk): `test_window_write_slots_matches_the_write_it_describes` covers
   `n = 2*window + 3`. Try to find a `(pos, n)` where the write touches a slot the
   snapshot does not cover.
3. **Attack the guard.** Pass `verify_rows < 1 + gamma`; rollback must raise
   rather than corrupt. Pass `verify_rows > 1 + gamma`; the extra slots must be
   restored to their own value (no-op) — `test_..._does_not_cover_the_write`
   covers the first; the second is worth a direct probe.
4. **Attack the accepted-row claim.** If the restore ever put back an *accepted*
   slot, committed KV would be undone: `test_accepted_rows_...` asserts those
   slots equal the post-verify bytes AND that at least one of them actually
   changed, so a "restore everything" cheat cannot pass.
5. **Attack the chunk-shape scope.** The doc explicitly does NOT claim
   byte-identity with a token-at-a-time greedy decode. If you measure accepted-row
   KV against a greedy decode and find a difference, that is the documented
   pre-existing property, not a regression — but it is worth checking that the
   difference is inside the previously measured magnitude (exo phase 3).
6. **Attack the carry reasoning.** `reset_carry` on a boundary target and the
   `[need, ratio)` zeroing are canonicalizations of rows that are never read. If
   you can show a read path that reaches them (or that canonicalizing them
   changes any output), that is a real finding: the fix assumes
   `m = start_pos % ratio` governs which carry rows are live
   (`compressor.py` `Compressor.__call__`).
7. **Attack the "unreadable row" reasoning for compressed caches.** The fix
   restores rows at/above `target // ratio` on the argument that
   `compress_len = end_pos // ratio` (`attention.py:178`) excludes them. If any
   reader reaches row `target // ratio` before the chunk that closes its group
   rewrites it, the restore-to-pre-verify value would be wrong *and* the
   addressing argument would be broken.

## 8. NOT verified

* **Magnitude on the real model.** Everything here is a tiny random-weight
  fixture (dim 64, head_dim 32, 2 layers). Whether the pre-fix divergence flips a
  greedy argmax at head_dim 512 over 40+ layers, and whether it is the mechanism
  that tipped the two 2026-10-13/14 reasoning loops, is **not measured** by this
  task and is not claimed. The audit's own M1 causality caveat stands
  (`spec.rollback`'s body last changed 2026-09-29, so the mechanism predates the
  incidents).
* **No live/collective verification.** No cluster op, no relaunch, no soak, no
  real-model A/B. Nothing here has touched the live instance.
* **Real TP=2 two-rank behaviour.** The tests are single-rank. Whether the ring
  contents are byte-identical across ranks, and whether both ranks execute the
  same restore, is unverified (the audit lists the same gap).
* **`engram_ids`** (see scope note) and the **DSpark draft window** are not
  asserted by the new tests; both are argued safe by their addressing, not
  proven here.
* **Performance.** The added gather/restore was not benchmarked on the real
  model; the analytic cost is in section 5 (no new sync, no context-proportional
  allocation). The one-sync-per-round property is preserved by construction
  (no `mx.eval` in `snap`) but was not measured live.
* **`prefill.decode_prime`'s own save/restore** was re-read and left unchanged; it
  handles only the single row its one-row probe can write (`window=128`,
  `verify_rows=1`), so it was not extended to the new multi-row path — that path
  is the speculative one, which now uses `spec.snap`/`spec.rollback`.
* **`origin/main` (6cc9c1e) does not contain this lineage** (`a0573e3` is not an
  ancestor of it), so no rebase/duplication check against main was possible; the
  branch is deliberately based on the live pin.
* **The swept bench probes (15 files) were not executed.** They need the model
  package dirs, and `pU4`/`pU6`/`pU7`/`pU8` also need the cluster hostfile. Only
  compile-checked + reviewed-by-anchor. If a probe's rollback span were wrong, the
  new span check would raise rather than corrupt — but that path is untested here.
* **The sweep-completion edits were not executed either**: `tests/p67_serve.py` is
  a GPU replay oracle (needs the EXL3 model package dirs and, for `P67_HEAD=real`,
  ~7.3 GB of DSpark-head weights) so it cannot run on this Mac; `pW21`/`pW24` and
  the five archived exo probes likewise need the model dirs. All were
  `py_compile`-checked; the p67 edit is a one-argument addition (`len(rows)`) at
  the exact call site, and the archived-probe edits are one-argument additions
  whose value is copied from the forward on the adjacent line.
* **Not a deploy.** The exo call-site change and this doc are on a branch. Landing
  it live requires a deploy-branch assembly + relaunch, which is out of scope
  here.
