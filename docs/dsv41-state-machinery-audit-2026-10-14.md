# DSv4.1 state-machinery audit — warm-lineage drift/corruption vectors (2026-10-14)

**Question.** Which mechanisms in the restart/park + spec-rollback + session-reuse
machinery can make a long-lived **warm** conversation's engine state (KV window ring,
compressed/indexer caches, DSpark draft window, generation/session bookkeeping) hold
something other than what a **fresh rebuild of the same tokens** would hold — i.e. what
makes a rebuilt lineage behave differently from a cold one?

**Incident under audit.** Two reasoning-channel repetition loops (2026-10-13 ≈56K ctx,
2026-10-14 ≈78K ctx), only in sessions running 97–99 % prefix-cache reuse across ~40
turns; a 144-call clean baseline at 110–241K context had zero loops. The mlx-lm kernel
changes between clean and now were already measured selection-neutral in the relevant
bands and cannot explain them.

**Code revision audited (important).** `deploy/next26-limiter` @ **a5d33b46b** — the live
deploy tip that produced the incidents, read through a detached worktree
(`~/.hermes/cache/scratch/exo-live-audit`). **`origin/main` (8887b1fff) does not contain
this machinery at all**: on main `park.py` is 791 lines (vs 2045) and `coord.py`,
`shutdown.py`, `fingerprint.py` do not exist; the G1–G5 restart chain
(54962362c, c6100cdf2, cd30c41cd, f5f6f8429, ae05d7beb, 61b49f82d, ae3175a6b, 30f32e448)
and the next21–next26 constructs live only on the deploy lineage. Line numbers below are
that revision. mlx-lm pin `a0573e3f`, mlx pin `603f16eb`.

---

## Ranked mechanisms

### M1 — `spec.rollback` does not restore the window ring: post-rejection state is a function of the REJECTED drafts. **PROVEN. [major]**

| anchor | what it does |
|---|---|
| `mlx-lm/mlx_lm/models/deepseek_v41/spec.py:61-73` | `rollback(cache, sn, target, st)`: sets `cache.offset = target`, rebuilds the **compressor carry only**. Touches no layer's `win_kv`. |
| `spec.py:20-30` | `snap()` snapshots only `comp_state` (`kv_state`/`score_state` slice) — no ring. |
| `spec.py:4-10` | the docstring that licenses this: *"window ring and compressed caches are position-addressed and are **rewritten before they are read again**"*. **False for the ring.** |
| `attention.py:161` | `prev = lc.window_chrono(start_pos)` — the ring **read**. |
| `attention.py:172` | `lc.write_window(start_pos, kv)` — the ring **write**, 11 lines *after* the read. |
| `cache.py:121-138` | `window_chrono` / `write_window`: slot = `pos % window`, `window_size = 128` (`config.py:44`). |
| `rounds.py:458-516` | `_one_round`: `snap` (478) → verify forward of `1+gamma` rows (480-483) → `spec.rollback(cache, snapshot, position+len(committed))` (514-515). |

**Sequence.** A verify forward feeds `anchor + gamma` drafts at positions
`[position, position+gamma+1)` and writes **all** of them into the ring. If
`gamma - accepted >= 2`, `rollback` keeps `position+accepted+1` as the new offset but
leaves the rejected rows in the ring. Slot `p % 128` is also the slot of the
already-committed position `p-128`, so the **next** forward's read of `p-128` returns the
rejected draft's KV instead of the committed token's KV. Because a slot is only rewritten
when position `q+128` arrives — i.e. exactly when `q` leaves the window — the stale row
serves every read of `q` for the rest of `q`'s window life.

**Independently reproduced by me** on the real `Attention` / `ModelCache` / `spec` code
(shared venv, `PYTHONPATH` pinned to the fork's mlx-lm):

* `~/.hermes/cache/scratch/dsv41_spec_ring_alias_repro.py` (toy `window_size=8`,
  `gamma=3`): **2 failed, 2 passed**. `accepted=0` → `max|diff|` 0.236 (rows 0 and 1);
  `accepted=1` → 0.097 (row 0); `accepted=2,3` → bit-identical.
* `~/.hermes/cache/scratch/dsv41_spec_ring_alias_W128_repro.py` (**production
  `window_size=128`**, offset 1000, 128 rows of reads): **2 failed, 2 passed**;
  `accepted=1` → `max|diff|` 0.0074 per row across the next 128 rows.

The test's invariant is the audit's invariant: *the rows after a rollback must not depend
on the content of the rejected drafts.* Two runs differ only in the rejected rows' random
content; they must be bit-identical. They are not.

**Why the sibling caches are safe and this one is not.** `comp_kv`/`index_k` rows are
**group**-addressed (`row = pos // ratio`) and only complete groups are read
(`compress_len = end_pos // ratio`, `attention.py:178`), and `publish_keys`
(`attention.py:190`) runs *before* that read (`:195`); `comp_kv` is written at `:233`
before the `kv2` slice at `:242`. A write for group `g` cannot destroy another group's
data, and a group row is (re)published from committed rows + the carry that `_rebuild`
reconstructs exactly (`spec.py:38-58`). The ring is the **only** position-addressed buffer
where a write for a *future* position lands on the slot of an *already-committed* position
— and the codebase knows it: `cache.py:142-150` (`ring_snapshot` docstring) says *"slots
alias every `window` positions, so a discarded write at `q` silently corrupts a live read
of `q - window`"*, and that is exactly why session checkpoints copy the ring
(`session_cache.py:367-374`, restored at `:412-413`) and why `prefill.decode_prime`
saves/restores ring slots. Only the spec-rollback path was left out.

**Falsifiable prediction (session-walk).** Because the aliased reads are the *oldest*
`gamma - accepted` entries of the window, and only for positions whose owner is within
128 of the rejection point:
1. Any first divergence between the warm walk and a cold rebuild must appear **inside a
   turn, within ≤128 committed tokens after a round with `gamma - accepted >= 2`**, and
   must not appear in a lineage with no such round in the preceding 128 tokens.
2. It must **not** be gradual across turns; the ~40-turn reuse history is not cumulative
   here (the ring self-cleans within 128 positions).
3. Discriminating run: replay the loop turn from the recorded state **with the ring
   patched in `snap`/`rollback`** (save/restore slots for `[target, position+gamma]`).
   *Confirms M1* if the first divergence moves or the loop disappears; *falsifies M1*
   if the patched replay loops identically at the same token.
4. Cheap settle: the two scratch tests above; for the live question, patched-vs-unpatched
   replay of just the loop turn.

**Causality caveat — read this before blaming M1.** `spec.rollback`'s body last changed
**2026-09-29** (mlx-lm `4091b5f`, `git log -L 61,73:...spec.py`), so the mechanism was
present in the clean baseline too. It explains **warm ≠ fresh** and it is deterministic
(which is why the incident reproduced on both client retries — see below), but it does not
by itself explain a *new* regression. It becomes causal only if rejected tails got longer
or more frequent in the incident era.

### M2 — The reasoning-budget limiter shortens `committed`, lengthening the M1 tail. **[minor → at most contributing]**

`rounds.py:504-509` (`committed = limiter.feed(committed)`) truncates the round's committed
batch under the reasoning budget, so `position + len(committed)` moves **earlier** and
`gamma - accepted` grows by the number of cut tokens — more junk rows in the ring.
Lockstep is *not* broken (the rollback target and the draft append at `rounds.rs:516` both
use `len(committed)`).

Timing: the limiter landed `eb582f151` **2026-10-13 23:55** and its construct `248ec7a6b`
**2026-10-14 12:47** — i.e. it was live for the 10-14 event but *not* for the 10-13 event
(the 10-13 rank-1 log ends 22:35, before the guard even landed). Counter-evidence I
measured: the limiter was `armed` on **every** turn of the 10-14 incident window
(`budget=16384 starts_in_thinking=True`) but **no cut / forced-think_end line appears in
that window**, and the window's turns emitted 643 and 1613 tokens against a 16384 budget,
so it very likely never cut. **Prediction:** if it never cut, disabling the limiter must
change nothing; if it did cut, M1's divergence should cluster immediately after a cut.
**Not verified:** whether a cut ever fired in the incident turns (the cut may be silent).

### M3 — Restore identity is not numeric-complete. **[minor, design gap, not exercised]**

`coord.py:132` `_BUILD_IDENTITY_FIELDS = ("model_fp", "tokenizer_fp", "kv_code_fp",
"mlx_version")`, checked by `fingerprint.py:648-679` over `KV_ALLOWLIST` families. Missing:
`mlx_lm.models.exl3.*` (whose kernels are selected by called row count, so they change
numerics) and every numerics-affecting env knob (`DSV41_INDEXER_QSPLIT`,
`INDEXER_TILE_MB`/`HIER_EXACT_MB`, `SPARSE_BUDGET_MB`, `COLSPLIT`, prefill
chunk/fence/transient budgets, spec gamma). A park written under config A could be restored
under config B with identical sources and produce KV bytes a fresh rebuild under B would
not. `window`/`bsz` *are* covered (layout fp, re-checked at `park.py:825`). Not exercised in
either incident, because no restore ever contributed reused rows (below).

### M4 — Parked restore does not check the draft window against the cache offset. **[minor]**

`park.py:905-916` sets the DSpark `n_ctx` from the manifest with no comparison against
`conv.offset`; it relies on the write-time `park_safe` check (`park.py:434-438`).
Would need a manifest/torn-write fault. Cheap settle: assert `draft_ctx() == offset` after
`_restore_parked`.

### M5 — `cancel()` fallback when the turn-start checkpoint was evicted. **[minor, self-healing]**

`session.py:970-986`: if `_base` is not in `cache.boundaries` the fallback is
`cache.cancel()` → rewind to the newest boundary **below** the offset, which can be *above*
`_base` (a decode-ladder rung inside the turn). `_gen` is then truncated to `offset - _base`,
so the token history is shorter than what the cache holds; the next turn's `plan()` rewinds
at most to the common prefix and heals it. No drift found.

---

## VERIFIED CORRECT (evidence, not assertion)

1. **Park/restore is exonerated as the source of the incident state.**
   `park.restore` ×53 in the 10-13 rank-1 log decomposes to 51 × `cold reason=rank-empty`,
   1 × `cold reason=restore-failed-on-a-rank` (13:12:49) and exactly 1 ×
   `gen=1 (all ranks)` (18:45:26) — one exchange per `Dsv41Sessions.get()` that finds no
   resident match, not a repeat-restore bug. The single successful restore logged
   `parked session restored: 67069 rows from SSD` and was immediately followed by
   `prefill controls … (rows=25349, base=2048)`, i.e. the **entire** prompt was refed: it
   reused **0** restored rows. The 10-14 incident window contains **zero** park lines.
   `park.ckpt abort-pending … units=0 polls=1 written=False` (`park.py:2039`) is a
   request-arrives-first abort: a temp dir written and removed, read-only w.r.t. memory.
2. **All-or-nothing restore holds.** `coord.py:278-326` (`decide_restore`) admits a
   `(key, gen)` only if present on every rank with the same digest, and the result type
   cannot express a mixed outcome; the ACK round (`coord.py:594-607`) discards everywhere
   if any rank fails (observed live at 13:12:49 → both ranks went cold); per-exchange
   magic words (`coord.py:104-114`, `:385`) turn a mispaired row into COLD. Covered by the
   T8 tests (`test_t8_never_mixed_exhaustive`,
   `..._restore_failure_on_one_rank_discards_everywhere`,
   `..._one_rank_missing_generation_is_cold_both`).
3. **Rank continuity is admitted, not guessed.** Each rank scans only its own node-local
   disk; a dir with a `site` field is admitted only on an exact site match
   (`park.py:1567-1571`), otherwise the rank-tag rule applies (`park.py:1521-1543`); a
   negative identity verdict goes cold (`park.py:1809-1812`) and is retried each call,
   while a positive one is latched for the process lifetime (`coord.py:540-544`). Only a
   negative verdict latches off (`test_identity_negative_verdict_does_not_latch`).
4. **Generation numbers are not a lineage clock.** They are per prefix-hash
   (`park.py:1675-1678`), so every checkpoint of a growing conversation is `gen=1` (as the
   log shows); ordering is by LCP, and nothing compares a restore generation against an
   in-memory lineage because restores only happen when nothing is resident
   (`session.py:1126-1156`).
5. **Restore never appends to live state.** `restore_conversation` (`park.py:796-917`)
   builds a fresh `Conversation` + `ModelCache`, replaces every buffer and zeroes rows past
   the offset (latents `park.py:937-940`, engram rows `park.py:970-974`), committing
   bookkeeping only after all buffers validate (`park.py:857-864`).
6. **G3 park-on-shutdown cannot capture mid-turn state.** Force-park is default-off
   (`shutdown.py:95-100`) and left no runtime lines; in-flight conversations are dropped,
   never parked (`session.py:1524-1554`); forced checkpoints skip `_inflight` conversations
   (`session.py:1392`); the writer re-checks offset and in-flight state every unit
   (`park.py:640-647`); prefill runs synchronously with no yield, so shutdown cannot
   interleave.
7. **The guard's abort path is the failure path, and it does not poison the next turn.**
   `engine.py:1383-1400` (`_abort_turn`) is deliberately `session.cancel()` + maybe
   `_sessions.drop`. `cancel()` (`session.py:964-986`) rewinds to `_base`
   (`session.py:763`: `offset - tokens_prefilled`, the turn start, which is a boundary —
   previous turn end, or offset 0 which is never pruned), restores the ring
   (`session_cache.py:412-413` via the snapshot at `:367-374`) and the draft window
   (`session.py:977-978` → `_restore_draft`, `:921-934`), then prunes orphaned draft/anchor
   snaps (`:907-919`). Live confirmation: after the aborted attempt, both client retries
   logged `prompt=78375 prefill=889 reuse=77486 cache=78375` with **no** `rewind` field —
   i.e. the cache already sat exactly at the reuse boundary `77486` (attempt 1's `_base`),
   the identical state attempt 1 started from, and the guard fired again. Byte-identical
   inputs → byte-identical loop: the abort leaves a *deterministic, re-enterable* state.
8. **`UNCOMMITTED` is a logging artifact.** The field prints whenever
   `TurnOutcome.committed` is False (`session.py:496-497`), and the `turn reuse` line is
   logged at `engine.py:1551` — before any decode, while `committed` only becomes True in
   `_end_turn` (`engine.py:1379`). Every turn-reuse line is therefore `UNCOMMITTED`,
   including the retries; the *missing* `rewind` on the retries is the meaningful signal,
   and it is explained by (7). Every recorded `rewind=N` in both sessions is an ordinary
   `plan()` rewind to the newest checkpoint at or below the LCP (e.g. `rewind=77988`
   → boundary `77486`, 889 rows refed).
9. **The `EXO_DSV4_SPEC_*` flags are inert for this engine.** `dsv4_mtp.py:705/719/734`
   (`EXO_DSV4_SPEC_STATE_RESTORE`, `EXO_DSV4_SPEC_CACHE_ROLLBACK`, `..._C2`) and
   `EXO_DRAFT_KV_WINDOW` (`generate.py:2617`, `batch_generate.py:3488`,
   `pp_speculation.py:3493`; forwarded by `start_cluster.sh:233` and `:326-328`) belong to
   the generic DSv4-MTP and pipeline-parallel draft paths. **No `dsv41/` module imports
   `dsv4_mtp` or `pp_speculation`** (only a docstring mention, `output.py:11`), and every
   incident log line comes from `exo.worker.engines.mlx.dsv41.*` — the card-driven
   `engine="dsv41"` route (`dispatch.py`). The rollback that actually ran is the mlx-lm one
   in M1. `EXO_DRAFT_KV_WINDOW` likewise does not govern the DSpark `DraftWindow`
   (a 128-slot ring with an absolute `n_ctx`, `mtp.py:55-87`).
10. **Draft-window lockstep is preserved on every decode path.** Spec round: cache rolls
    back to `position+len(committed)` (`rounds.py:514-515`) and exactly `len(committed)`
    taps are appended (`:516`), so `draft_ctx == offset` is invariant-preserving; forced
    round: 1 row in, 1 tap appended (`:397-407`); greedy round: no draft. `prefill` rewinds
    the draft to `plan()`'s boundary before `append_turn` rewinds the body to the same one
    (`session.py:733-736`), per-chunk taps are fed before any ladder checkpoint
    (`:811-822`), the invariant is asserted after prefill (`:771-775`), and ladder
    checkpoints **refuse** an inconsistent (body, draft) pair (`:865-873`).
11. **Snapshots are copied, never aliased** — `ring_restore` (`cache.py:152-154`),
    `_restore_carry` (fresh arrays), `_restore_draft` (`session.py:933`); draft/anchor dicts
    are pruned to the surviving body boundaries (`session.py:907-919`).
12. **Stale rows above a rollback/rewind point are never read before being rewritten** —
    for `comp_kv`/`index_k` (see M1's "why the sibling caches are safe"); the argument
    fails *only* for the ring.

---

## NOT VERIFIED

* **M1's magnitude on the real model.** Measured divergence is 0.0074–0.236 on a tiny
  random model (`head_dim` 32, 1 layer). Whether it flips a **greedy argmax** at
  `head_dim` 512 over 40+ layers, and hence whether it can tip a marginal reasoning channel
  into a repeat loop, is unmeasured. This is the single most important open question.
* **Whether a limiter cut ever fired in the incident turns** (M2). No cut line appears in
  the window, but the cut may be silent — check `reasoning_budget.py` for its logging.
* **Production rejection-tail statistics** — how often `gamma - accepted >= 2` occurs at the
  incident context sizes. The clean baseline could simply have had fewer long tails.
* That KV/compressed/indexer state is byte-identical across the two TP ranks, whether the
  anchor logits and DSpark state are rank-local, and whether `model_fp` covers the
  draft-head weight file.
* The **rank-0** log of the 10-13 event and the full (non-window) 10-14 log; the 10-14
  window I read is a 4-minute extract ending at the guard fire.
* Where the cache offset advances inside the model forward (relevant to M5's failure mode
  and to whether a fault *inside* a forward can leave ring slots overwritten without the
  offset advancing).
* Any test execution for the park/coord suites: in the snapshot, collection fails at
  `conftest.py` because the dashboard assets are not built; the shared venv also lacks
  `mlx_lm.models.deepseek_v41` for the *cluster-venv* path. The M1 scratch tests above **did**
  run (`PYTHONPATH=~/repos/exo/mlx-lm`, `~/repos/exo/.venv/bin/python -m pytest`), 2.0 s.
  No test in either repo references the three spec flags, and no test exercises the window
  ring after a speculative rollback (the rewind/cancel tests use a stub prefill whose
  position-derived writes cannot expose M1).
* Anything about the live mlx-lm **kernel-selection** history as a *cause* — out of scope
  here (separate sibling task).

## Context that bounds what "fresh rebuild" can even mean

`spec.py:4-6` documents that chunk verify (`gamma+1` rows in one body forward) is **not
bit-identical to plain greedy** — the body result depends on chunk shape. So a fresh
rebuild is *never* a bit-exact oracle for a warm lineage, independently of M1. Any
first-divergence replay must therefore compare against the same chunking, and a warm-vs-fresh
diff is not by itself evidence of corruption.
