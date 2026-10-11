# DSv4.1 restart survival — G4 build note (2026-10-10)

Gap **G4** of `docs/dsv41-restart-survival-plan-2026-10-10.md` (multi-rank
all-or-nothing restore + cross-rank generations), as corrected by
`docs/dsv41-restart-survival-phase0-audit-2026-10-10.md` (A7, blind spot #1 —
the audit wins where they conflict). Branch `p20/dsv41-restart-g4`, stacked on
G2+G5 (`46990ec73`), worktree `/private/tmp/p20-restart-g4`. **OFFLINE /
laptop-only**: no cluster contact of any kind (no ssh, no boots, no model
requests, no benchmarks, no GUI automation), nothing deployed, production
`b5f055185` untouched. Commit sha: see the push record (`git ls-remote`).

## What changed (line numbers are in the committed tree)

| File | What |
|---|---|
| `src/exo/worker/engines/mlx/dsv41/coord.py` (NEW, 532 lines) | Pure decision + codecs + drivers. `key_int` :134, `build_identity_digest` :147, `Candidate` :172, `RankParkState` :189, `RestoreDecision` :214, **`decide_restore` :242** (T8 unit), `agree_generation` :293, `prune_floor` :306, `encode/decode_restore_payload` :327/:346, `ParkCoordinator` :375 (`from_group` :390, `exchange` :412), `run_restore` :438, `run_park` :496. Constants `MAX_CANDIDATES=4` :82, magics :92-95. |
| `src/exo/worker/engines/mlx/utils_mlx.py` | `_sanitize_int32` :2587, `mx_all_gather_ints` :2603 — one fixed-width int32 `all_gather`, CPU stream + immediate `mx.eval` (same discipline as `mx_any` / `mx_all_gather_tasks`); `group=None` -> local-only; every value sanitized into `[0, 2**31-1]` BEFORE the array is built. |
| `src/exo/worker/engines/mlx/dsv41/park.py` (+406) | Module doc §G4 paragraph. `_GEN_DIR_RE` :163 now `<prefix>.gen<N>[.r<rank>].<uuid8>` (rank group OPTIONAL — pre-G4 names parse); `_MAX_GENERATION` :174; `_gen_from_name` :275 (refuses absurd N); `_rank_from_name` :291; `_gen_dir_name` :299. Manifest `"rank"` field :458 (PARK_VERSION stays 2). `ParkedStore.__init__` gains `coordinator=None, rank=None`. `park` :1106 / `try_restore` :1190 delegate to the coordinated path when a coordinator is set. Rank-local halves: `_rank_matches` :1225, `local_restore_candidates` :1249, `find_generation` :1308, `has_committed` :1346, `write_generation` :1362, `prune_acked` :1387. Drivers `coordinated_restore` :1409, `coordinated_park` :1447 (take `store=None` so a rank without a usable store still participates). |
| `src/exo/worker/engines/mlx/dsv41/session.py` | `Dsv41Sessions.__init__` gains `coordinator`, `rank` (stored :984); forwarded to the lazily built store. `_restore_parked` :1076 and `_evict` :1135 call the coordinated drivers **unconditionally** when a coordinator exists (even if parking is env-disabled or the store build failed on this rank); `_evict` wraps the call (:1152) so `session.close()` always runs. |
| `src/exo/worker/engines/mlx/dsv41/engine.py` | import; `Dsv41Sessions(..., coordinator=ParkCoordinator.from_group(get_coord_group(self.group), rank=self.loaded.rank), rank=self.loaded.rank)`. |
| `src/exo/worker/engines/mlx/dsv41/fingerprint.py` | `coord` added to `KV_NEUTRAL` (control-plane ints, no KV bytes) — required by the existing `allowlist_problems()` guard. |
| `src/exo/worker/engines/mlx/dsv41/tests/test_dsv41_park_g4.py` (NEW) | T8 + supporting tests (32 test functions, 39 collected cases). |

## Coordination design

**Transport.** The existing QP-less TCP coord subgroup
(`get_coord_group(self.group)`, the channel `RankAgreement` already uses) —
never the model TP group. Each exchange is ONE `mx.distributed.all_gather` of a
fixed-width int32 row (`mx_all_gather_ints`), CPU stream, immediately `mx.eval`'d.
Ints are kept < 2^31 (keys/digests are 60-bit, sent as two 30-bit halves).

**Exchanges** (each row carries a per-exchange magic word):

1. *Restore* — `RESTORE` row: up to 4 admissible candidates
   `(key, gen, pref, build_digest)`; key = `key_int(manifest.prefix_key)`
   (token-stream hash, identical across ranks), `pref` = LCP (prefix mode) or 0
   (client-key mode). Every rank runs `decide_restore` over the same gathered
   rows: any rank empty/unreadable -> COLD; else intersect `(key, gen)` across
   ranks, require EQUAL build digests, pick the highest `(min pref, gen, key)`;
   none -> COLD.
2. *Restore ack* (only when step 1 says RESTORE — itself rank-invariant) —
   each rank restores exactly that `(key, gen)` (not marking it active yet) and
   reports ok/fail. Unless EVERY rank acks the same `(key, gen)`, every rank
   discards (`conv.close()`) and goes COLD. Only after a unanimous ack is the
   `.active` marker written.
3. *Park gen* — each rank sends its local max gen for the conversation; all
   write `N = 1 + max over ranks` (conservative: never reuses a number any rank
   already has, so a lagging rank cannot produce a same-N/different-turn
   collision).
4. *Park ack* — each rank sends `N` iff it wrote gen N AND a re-scan shows it
   admissible (`has_committed`), else 0. Only on a unanimous ack does any rank
   prune, and then only gens `<= N - 2` (keep=2). Any missing ack -> nobody
   prunes this turn ("lazy prune"). Consequence: gen N and N−1 survive on every
   rank whenever any prune happens, so the prune can never remove the last
   common restorable generation.

**Build-identity digest.** `build_identity_digest` hashes only
`model_fp / tokenizer_fp / kv_code_fp / mlx_version`. `layout_fp` is excluded
because it embeds rank + layer range and legitimately differs between TP ranks;
it is still enforced per rank by the existing local `_admit` gate, and G4 adds
a second, independent guard (see FIX C below).

**Why rank-invariant.** (a) Every exchange is reached unconditionally at the
same between-turn call site on every rank — the coordinated path ignores local
state when deciding whether to participate (no store / disabled / build failure
-> contribute "nothing" but still call). (b) Row widths are constants, so there
is no count-agreement round and no rank-dependent shape. (c) Whether the
optional 2nd restore exchange runs is a function of the gathered rows, which are
identical on every rank. (d) Each collective is eval'd immediately, one at a
time, in program order — consistent with the Phase 0c finding that MLX matches
collectives by EVAL order (`same_order` PASS); nothing is `async_eval`'d. (e) No
cache array is touched by the coordination (only Python ints), so the
buffer-donation trap is not in play.

**Failure semantics.** Local errors (store scan, manifest read, restore) never
raise out of the drivers: they degrade that rank to "no candidates" (-> COLD
everywhere) / "not committed" (-> no prune). Malformed or desynchronised rows
(bad magic/width, different key/gen) are treated identically by every rank
(COLD / no prune). A transport failure of the collective itself propagates,
exactly like `RankAgreement` (a real fault). Single rank / PP mode
(`get_coord_group` -> `None` or size 1): `ParkCoordinator.from_group` returns
`None` and the store runs the unchanged pre-G4 local path.

## Hardening pass (after an independent adversarial audit)

A dedicated audit of this diff surfaced three defects; all three are fixed and
have regression tests:

- **FIX A (HIGH) — a rank-local error after the exchanges could hang the peer.**
  `coordinated_park`'s tail called `enforce_budget` unguarded; `enforce_budget`
  -> `_usage_entries` (`float(man["created"])`) and `_candidates`
  (`read_text`, OSError-only) can raise on a corrupt manifest. On ONE rank that
  propagated out of `_evict`/`get` while the peer entered the TP forward pass ->
  hang, and skipped `session.close()` -> KV leak. Fixed: the whole post-exchange
  tail of `coordinated_park` is guarded (never raises; returns the commit
  status); `_evict` also guards the call (defense-in-depth, guarantees
  `close()`); `_candidates` now tolerates `UnicodeDecodeError` and
  `_usage_entries` tolerates non-numeric `created`.
- **FIX B (MEDIUM) — int32 overflow raised on one rank BEFORE the collective.**
  `mx.array(row, dtype=mx.int32)` raises on a value >= 2^31 (confirmed
  empirically); a corrupt dir name (`gen2147483648`) reached the park payload,
  and a raise before the collective strands the peer and desyncs the shared
  coord call-id counter. Fixed: `_sanitize_int32` clamps/zeroes out-of-range
  values before the array is built (a bad value can now only force COLD/no-prune),
  and `_gen_from_name` refuses absurd generations (`_MAX_GENERATION`).
- **FIX C (MEDIUM) — the rank tag was written but never checked.** The
  cross-rank digest drops `layout_fp`, so rank isolation rested only on `_admit`,
  which `EXO_DSV41_PARK_ALLOW_STALE` bypasses. Fixed: `_rank_matches` refuses a
  dir whose rank (`r<rank>` in the name or manifest `rank`) is KNOWN and
  `!= self.rank`, wired into `local_restore_candidates`, `find_generation` and
  `has_committed`; permissive when `self.rank is None` or the tag is absent.

## Test evidence (real runs, this worktree)

```
$ PYTHONPATH=…/src:…/mlx-lm EXO_DASHBOARD_DIR=…/dashboard …/python -m pytest src/exo/worker/engines/mlx/dsv41/tests/ -q
290 passed in 23.55s   (baseline 251 + 32 G4 + 7 hardening = 290)

$ … -m pytest src/exo/worker/engines/mlx/dsv41/tests/test_dsv41_park_g4.py -q
39 passed in 1.60s     (T8 file; 32 test functions, 39 cases)

$ ruff check src/exo/worker/engines/mlx/dsv41/ src/exo/worker/engines/mlx/utils_mlx.py
All checks passed!
```

T8 coverage (`test_dsv41_park_g4.py`):
- pure `decide_restore`: {1,2}/{1,2} -> 2; **{1,2}/{1} -> 1 (missing gen N ->
  N−1)**; {2}/{1} -> COLD; {2}/{2} digest differ -> COLD; {1,2}/{1,2 mismatched}
  -> 1; one rank empty -> COLD; unreadable rank -> COLD; key mismatch -> COLD;
  single rank -> local best; **exhaustive** 2-rank sweep over all gen subsets of
  {1,2,3} x 2 digests asserting "highest common equal-digest gen, else COLD" and
  the single-decision invariant (`RestoreDecision` has no per-rank field — a
  mixed outcome is unrepresentable).
- fault injection: a rank's local candidate scan raising -> COLD, no raise;
  peer's restore failing after a RESTORE decision -> this rank discards -> COLD.
- two simulated ranks (two `ParkedStore`s on two tmp roots, each on its own
  thread, lockstep in-process gather with a timeout so a SKIPPED exchange FAILS
  rather than hangs): missing gen N on one rank -> both restore N−1 (`.active`
  on gen 1 on both); no common gen / one empty / build-digest mismatch / corrupt
  blob on one rank (quarantined, peer NOT marked active) / store error / parking
  disabled on one rank -> both COLD; prefix (LCP) mode; end-to-end through
  `Dsv41Sessions.get` on both ranks; a rank whose store failed to build still
  joins the exchange.
- park: identical agreed N on both ranks; rank-tagged dir names + manifest
  `rank`; exactly 2 exchanges per park per rank; ack-gated prune (rank 1 fails
  gen 3 -> rank 0 keeps {1,2,3}; cold-restart restore picks common gen 2 on
  both; after both commit gen 4 -> {3,4} / {4}).
- backward compat: old `<p>.genN.<uuid8>` and new `<p>.genN.r<rank>.<uuid8>`
  both parse; a pre-G4-named generation restores through the coordinated path.
- single rank (no coordinator): newest-two retention + restore unchanged.
- hardening regressions: `coordinated_park` returns (both exchanges ran) when
  `enforce_budget` raises; non-numeric `created` and non-UTF-8 manifests are
  tolerated; out-of-range payload ints never raise and are sanitized on the
  wire; a foreign-rank dir is ignored while a rank-less/pre-G4 dir is still
  considered and an unknown-rank store stays fully permissive.

Test-harness note: in this MLX build streams are thread-local (a lazy array
built on one thread cannot be eval'd on another), so each simulated rank does
all its MLX work on one persistent worker thread — like a real runner thread.

## Deliberately NOT done (out of scope for this build)

- **G1** after-turn idle checkpoint, **G3** park-on-shutdown, **G6**
  rewind-to-LCP — not attempted (G1/G3 need engine-thread + signal work; G6 is
  deprioritised by audit A4). **G7** incremental checkpoints — deferred.
- **No live/TP2 validation.** `mx_all_gather_ints` was exercised locally at
  `size==1` and `group=None` only (confirmed empirically that `all_gather` of a
  shape-`(width,)` int32 row returns a FLAT `(world*width,)` array split in rank
  order, which the helper's slicing matches). The real 2-rank jaccl TCP path is
  unexercised.

## Deviations from the plan (and why)

1. **No coordinator rank / broadcast.** The plan says "coordinator gathers …
   broadcasts"; this design is a SYMMETRIC all_gather and every rank runs the
   same PURE decision. That removes the only rank-dependent branch (rank 0 vs
   the rest) and is strictly safer against the Phase 0c EVAL-order trap.
2. **Per-key exchange, not the plan's `{key -> …}` map.** Restore is per-request
   between turns, and a whole-store map would be an unbounded coord payload
   (jaccl bounds it). Only the conversation under consideration is exchanged
   (bounded to `MAX_CANDIDATES=4`).
3. **`pref`-primary pick.** The plan says "highest gen present on ALL ranks";
   this picks the best prefix match first, then the highest gen among equal
   `pref`. For a single conversation (the normal case) all candidates share the
   key, so it reduces to highest-gen; for multiple candidates it matches the
   existing local `find()` preference (highest LCP), so cluster and single-rank
   semantics agree rather than diverge.
4. **`N = 1 + max` over ranks** (not local max+1). Guarantees all ranks write the
   same N and never reuses a number a lagging rank already has.

## OPEN-QUESTIONs (live/TP2 — cannot be answered offline)

1. **Call-site symmetry assumption.** Coordinated restore is entered from
   `Dsv41Sessions.get` only when the conversation is NOT resident; park from
   `_evict` on LRU overflow. This is rank-invariant iff both ranks hold the same
   resident set — true today because both rank process the identical request
   stream through identical deterministic LRU/LCP logic, but it is an invariant
   G4 relies on, not one it enforces. Worth a live check (and maybe a cheap
   `mx_any` assertion) on TP2.
2. **Cross-node build-identity equality.** `build_identity_digest` includes
   `tokenizer_fp`, which hashes the tokenizer's `name_or_path`; if the absolute
   paths differ between the two nodes every restore would go COLD. Must be
   verified on the cluster (or the field canonicalized to a basename).
3. **Coord-channel wait vs supervisor hang timeout.** The PARK_ACK /
   RESTORE_ACK exchange covers the peer's GB-scale park/restore I/O; the coord
   socket deadline is ~150 s while the supervisor hang timeout can be ~45 s. A
   slow peer write could trip the supervisor. Unmeasured; possibly needs a
   heartbeat. (The writer side predates G4; the WAITING rank is new.)
4. **Rank-local byte/count LRU** (`enforce_budget`, `_reserve_headroom`) can
   still evict a generation on one rank only (different disk usage per node).
   The restore protocol stays safe (falls to a lower common gen or COLD, never
   mixed), but the "common gen always survives" guarantee covers the ack-gated
   prune, not budget eviction. Cluster-agreed LRU eviction is a possible
   follow-up.
5. **Desync is absorbed, not surfaced.** `agree_generation` ignores bad-magic
   rows and does not compare keys across ranks, and a RESTORE row carries no
   request identity; a genuine resident-set divergence would silently agree on
   N / skip pruning instead of raising. LOW (such a divergence should hang the
   model collectives first, and `prune_floor` DOES check the key), but a loud
   tripwire (key/`ids` hash in the row) is a cheap future addition.
6. **Budget eviction is still per rank** (same as #4); **key-mode `pref` is 0**
   rather than the token count (the pick then falls to gen/key — correct, just
   not length-preferring); `_restore_parked` inserts without enforcing
   `max_sessions` (symmetric across ranks, so harmless — the next `_open` then
   evicts, each with 2 exchanges). All NIT/LOW.
7. **Pre-G4 / G2+G5-era gens on disk.** v1/legacy dirs are refused by G2 before
   G4 sees them. G2+G5-era v2 gens were numbered per rank independently, so the
   same N on two ranks may be different turns and the (same-build) digest will
   not catch it — those gens were never deployed; if any ever were, wipe the
   park dirs on G4 rollout.

**No-node-contact affirmation:** no ssh/network/cluster/boot/model request of
any kind was performed; all work is laptop-local in `/private/tmp/p20-restart-g4`.
