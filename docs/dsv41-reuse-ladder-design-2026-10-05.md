
# DSv4.1 session reuse-collapse fix — periodic checkpoint ladder (2026-10-05)

**Status:** design for review. Evidence: PERFORMANCE_HISTORY "2026-10-05 (correction)"
+ the live confirmation below. Companion to the soak-2 postmortem.

## The defect (confirmed live, build b5eb293f5/54cebb7, 2026-10-05 08:47)

A 30K-token conversation was built (cold prefill, 112.1 s — prompt-end checkpoint at
offset 30008). A follow-up probe whose rendered prompt shares a 30007-row prefix
(ONE row below that checkpoint) was served as:
  log: `session reuse: this 30033-token prompt matches a resident conversation on 30007 rows`
  log: `prefill controls: ... (rows=30033, base=2048)`  -> boundary resolved to 0
  wall: 112.5 s (FULL re-prefill), not the ~ms-scale delta.
Root chain: `SessionCache.plan()` rewinds to `_boundary_le(lcp)` = newest checkpoint
<= lcp. Checkpoints exist only at offset 0, prompt ends, and turn ends. lcp=30007,
checkpoints {0, 30008, ~30008+g}: newest <= 30007 is 0 -> rewind discards all 30007
rows -> full re-feed. A 1-row undershoot costs the whole context. Same mechanism
made every soak-2 rung a full re-prefill (r500/r750 "deltas" = full prefills) and
makes every deep battery needle probe cost ~42 min instead of ~seconds.

## Fix: periodic checkpoint ladder during the prefill

Take session checkpoints (body rings + carries + draft window) every
`CHECKPOINT_SPACING` rows DURING a turn's delta prefill, not only at turn ends.
Then a follow-up whose LCP undershoots a turn boundary rewinds to the nearest
ladder checkpoint and re-feeds at most `spacing` rows (~4-8 s at deep rates),
instead of the full context (minutes to hours).

Exactness: re-fed rows are token-identical by definition of the LCP, and the model
is deterministic, so the resulting cache state is bit-identical to today's
rewind-to-0 path — this is a pure compute-saving change, no numerics.

### Components

1. **Exo `Conversation.prefill` (src/exo/worker/engines/mlx/dsv41/session.py)**
   already receives `progress` calls per prefill chunk. Add: a ladder hook inside
   the prefill's per-chunk progress path that calls `self._checkpoint()` when
   `self.offset - last_ladder_offset >= spacing`. `_checkpoint()` already does the
   right thing (body `cache.snapshot()` + draft-window snap + trims `_draft_snaps`
   / `_anchor_at` to live boundaries).

2. **Feed the draft window PER CHUNK instead of at the end.** Today the driver
   accumulates per-chunk taps and `_feed_taps(taps)` runs once after the whole
   delta. A ladder checkpoint mid-prefill would pair a body snapshot at offset P
   with a STALE draft window (draft_ctx != P), breaking the `draft_ctx == offset`
   invariant (`_restore_draft` at rewind would also fail). Fix: feed each chunk's
   taps to the draft window as that chunk completes — then at EVERY chunk
   boundary `draft_ctx == offset` holds (same invariant as between turns), and a
   ladder `_checkpoint()` snapshots a consistent pair. Requires the prefill
   progress path to see the newest chunk's taps: extend the `engine_prefill`
   progress callback (or add a second optional callback) to pass the chunk's tap
   dict; keep the existing signature compatible.

3. **Retention / knobs.** `max_snapshots` default 8 -> 16 (each is ~2.5 MB of ring
   copies + carries; 16 = ~40-60 MB — measure). New env:
   `EXO_DSV41_CHECKPOINT_SPACING_ROWS` (default 8192). Eviction keeps newest N +
   offset 0 while dropping oldest middles. Spacing bounds worst-case re-feed;
   retention bounds memory.

4. **Park/restore interaction (mlx-lm session_cache.park):** parked sessions
   persist snapshots? If the parked codec carries `_snaps`, a restored session
   keeps its ladder (nice-to-have); if not, restored conversations keep working
   via the normal path (checkpoint at restore + future ladders). Verify the codec
   either round-trips the new snapshots or ignores them safely.

### Verified invariants to test

- A follow-up with a 1-2 row LCP undershoot on a long conversation: re-fed rows
  <= spacing + one chunk (prefill_tokens in the turn log), wall time ~seconds.
- Bit-identity: output tokens EXACTLY equal to the current rewind-to-0 path for
  the same inputs at temp 0 (greedy) — drive both paths in a unit test, compare
  final cache offset + produced tokens + (if feasible) logits hash.
- Draft invariant: after any ladder checkpoint + rewind + refeed,
  draft_ctx == offset throughout; decode rounds run with drafting enabled.
- `_draft_snaps` / `_anchor_at` / `snapshots` trim to the same live boundary set
  in lockstep (no orphan key raising in `_restore_draft`).
- Snapshot cadence does not fire on tiny turns (single-chunk turns take the
  existing single checkpoint).
- Memory: 16 retained snapshots measured <= ~100 MB at 1M context (test with
  instrumented sizes).

### NOT in scope (note for deploy N+2)

The mlx-lm `SessionStore` path (non-exo) has the same collapse; leave as-is unless
trivial — the exo engine owns the production path.
