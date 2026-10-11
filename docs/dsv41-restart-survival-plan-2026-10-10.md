# DSv4.1 restart-survival plan — disk-persisted prefix-KV (2026-10-10)

**STATUS: PLAN ONLY — NOT EXECUTED.** Authored via reference-model consult (Fable),
grounded in an audit of `dsv41/park.py` + live cluster state on 2026-10-10.
Goal: a cluster relaunch (deploy) — ideally also a crash — must not cost the cold-start
re-prefill of the user's active conversations (~92.6K tokens ≈ 282 s at ~250-330 t/s;
~49% of real-session uncached tokens are one such cold start). Acceptance: after a
relaunch, the next request on a previously-active conversation restores from disk and
prefills only the delta (TTFT ≈ seconds of restore + delta); quality intact
(needle/logit checks); no silent staleness.

## What already exists (verified in-fork)

`src/exo/worker/engines/mlx/dsv41/park.py` — "Option E of the SSD-offload study" — is a
full serializer/restorer for the composite conversation cache:

- Per-session dir: `manifest.json` (magic `DSV41-PARK` + `PARK_VERSION`, offset/capacity/
  max_seq_len, token-id key, per-buffer dtype+shape, snapshot positions; written LAST as
  the commit record), `ids.bin` (int32 history for LCP matching), `win_kv.<layer>.bin`,
  `comp_kv.<src>.bin`, `index_k.<owner>.bin`, `carry.<src>.bin` (fp32),
  `engram_ids.bin`, `draft.<stage>.bin`, `snapshot.* / draftsnap.* / anchor.<pos>.bin`.
- Invariants: only rows reachable by a live read are persisted and restore ZERO-FILLS
  beyond (restored cache is byte-identical to a freshly built one); anchor logits are
  persisted; park/restore run ONLY between turns and only when `draft n_ctx == offset`;
  failures NEVER raise — park removes its temp dir, restore quarantines bad dirs to
  `<dir>.bad` and falls back to a cold prefill.
- `ParkedStore`: root `EXO_DSV41_PARK_DIR` (default `~/.exo/dsv41_park`), enable
  `EXO_DSV41_PARK` (**default ON**), `EXO_DSV41_PARK_MAX_GB` (default 12.0 LRU budget),
  `EXO_DSV41_PARK_MIN_TOKENS` (default 4096), `EXO_DSV41_PARK_MAX_COUNT` (default 64).
  `_candidates()` scans the root at init — parked sessions already survive process
  restarts.
- Wiring (`session.py`): eviction of an idle conversation → park; request path: resident →
  restore-by-key → prefix-match restore from the parked store → cold prefill.
- LIVE state (2026-10-10): studio1 `~/.exo/dsv41_park` = 11 GB / 46 session dirs
  (Oct 4–10, spanning multiple deploys), ~15.6 K park/restore log lines; studio2 = 11 GB /
  44 dirs. Versioning today = FORMAT version only; NO engine-build fingerprint.
- Known out-of-scope by design: image-carrying conversations skip parked restore
  (vision-reuse fix).

## Verdict on the framing

The active-conversation-loss hypothesis is probably right, but the fix should NOT center
on park-on-shutdown. Make **after-turn idle checkpointing** the primary mechanism and
treat shutdown parking as a fast path on top of it. If the last between-turn state is
already on disk: graceful restarts and crashes are both covered, a mid-turn shutdown has
a clean answer (lose only the in-flight turn), and nothing depends on supervisor kill
timeouts.

Four blind spots that Phase 0 must check:

1. **Multi-rank consistency.** TP2 = both ranks hold slices; each rank parks its own slice
   (both nodes have park dirs). A restore that succeeds on one rank and fails on another
   hangs a collective or corrupts. Restore must be all-or-nothing, cluster-level.
2. **Silent staleness is probably already happening.** 46 dirs spanning several deploys
   with no fingerprint + a startup scan = cross-build restores have likely already served
   traffic. Exact-repeat anchor logits make it worse (sampling from old-build logits).
3. **Restore semantics.** If restore CONSUMES the parked dir (moves to residency + deletes),
   a post-restart conversation is RAM-only again and lost at the next restart. Verify.
4. **LCP divergence.** If clients strip reasoning from prior assistant turns or
   retokenization differs, LCP can end at the start of the last assistant turn; restore
   gives partial value unless a prompt-end snapshot allows rewind.

## Phase 0 — read-only audit (~0.5 day, run FIRST)

| # | Question | How |
|---|---|---|
| A1 | When did each relaunch happen? | exo log startup banners + process-start lines; cross-check node `git reflog` + deploy history |
| A2 | What did the first request after each relaunch do? | Classify restore-by-key / prefix-restore / cold with prefill rows + seconds; confirm the ~282 s events are cold |
| A3 | Was the lost conversation ever parked? | For each cold start, find a park dir whose ids.bin is a prefix of the conversation's later dir; near-zero/absent LCP = never parked (hypothesis confirmed); large LCP + still cold = matching bug (different fix) |
| A4 | LCP divergence in practice | Over the same pairs: LCP vs old offset; where it diverges (template, stripped reasoning); is there a snapshot ≤ LCP to rewind to |
| A5 | Cross-build restores already occurring? | Restore log line mtimes vs manifests vs most recent startup; count cross-build hits |
| A6 | What does restore do to the dir? | Code read: delete / move / leave in place |
| A7 | Topology + multi-rank park/restore coordination | World size, layer/tensor ranges, park dirs per rank, existing coordination |
| A8 | How is shutdown performed? | Deploy script + launchd: signal, ExitTimeOut, kill -9 use, and reset-before-signal order (risk R3) |
| A9 | Park/restore cost | Park duration + bytes → p50/p95 GiB/s; restore duration; sets the shutdown budget |
| A10 | Memory headroom | Peak unified memory at 92K session; does a per-layer CPU-copy park fit without swapping |
| A11 | Do active conversations carry images? | If yes, parked restore is skipped by design for them |
| A12 | Is the conversation key stable across processes? | grep for hash()/id()/uuid4/per-process salts; unstable key = restore-by-key never hits post-restart (everything rides on prefix match) |

**Go/no-go:** proceed if A3 = "never parked" for the 282 s events. If "parked but missed" →
fix matching first (A4/A12). If A5 > 0 → ship the fingerprint (G2) immediately.

## Ranked gaps

1. **G1 — after-turn idle checkpoint** (core fix; covers graceful restarts AND crashes)
2. **G2 — build fingerprint + refusal policy** (prerequisite; G1 makes cross-build restores common)
3. **G3 — park-on-shutdown** (drain + final checkpoint; covers just-completed turns)
4. **G4 — multi-rank all-or-nothing restore with generations** (mandatory if sharded — TP2 IS)
5. **G5 — durability hygiene** (fsync, atomic generation swap, tmp sweep, non-consuming restore)
6. **G6 — prompt-end snapshot / rewind-to-LCP** (only if A4 shows divergence)
7. **G7 — incremental append-only checkpoints** (optional, later; cuts SSD wear)

### G1 — after-turn idle checkpoint

Touch points:
- `session.py` / `Dsv41Sessions`: add per-session `dirty_tokens` (offset delta since last
  committed park) + `last_turn_end_ts`, set at the turn-commit point (where
  `draft n_ctx == offset` already holds).
- New `Dsv41Sessions.maybe_checkpoint()` called from the engine/worker loop when the
  request queue is empty. MUST run on the engine thread (never a separate thread with
  MLX). Checkpoint a session when: offset ≥ `PARK_MIN_TOKENS` AND
  (`dirty_tokens ≥ EXO_DSV41_CKPT_MIN_DELTA` [default 2048] or never parked) AND
  idle ≥ `EXO_DSV41_CKPT_IDLE_S` [default 20 s]. Most-recent-first ordering.
- `park.py`: reuse the write, but do not close/evict — session stays resident
  (`keep_resident=True` path, or split serialize() from eviction).
- Cancellability: write layer by layer, check a `request_pending` flag between layers;
  abort + remove temp dir + keep previous generation + serve the request. Never block
  TTFT by more than one layer write (tens of ms).

MLX trap (must respect): do NOT hold references to cache arrays across the next forward
pass from a background writer — MLX buffer donation (refcount==1) is otherwise broken,
silently copying whole preallocated buffers every step. Synchronous per-layer
copy→write→release on the engine thread while idle avoids it. Add perf regression test T9.

Consistent state: only when idle, after turn commit, asserting `draft n_ctx == offset`.

Failure modes: write error/ENOSPC → park returns False, previous generation survives, log
and continue; checkpoint storm → cap one session per idle tick; SSD wear → ~1–3 GiB per
qualifying turn at ≥2048-token deltas → tens of GB/day at single-user rates (fine; G7 if
it ever isn't).

### G2 — build fingerprint and policy

Manifest fields (bump `PARK_VERSION`): `model_fp` (hash of config.json + weight index +
tensor file names/sizes + quant config; NOT full weights); `tokenizer_fp` (tokenizer files
+ chat template); `kv_code_fp` (sha256 over an EXPLICIT allowlist of KV-affecting sources:
dsv41 attention, CSA/HCA, indexer, MTP/DSpark, engram, park codec, custom kernels);
`mlx_version`; `layout_fp` (world size, rank, layer range, dtype); `generation`.

Compute once at import/startup — never lazily at shutdown (would hash the NEW checkout
after reset; see R3).

Policy — strict on KV-byte-affecting, permissive otherwise:
- Any mismatch in model_fp / tokenizer_fp / layout_fp / format version / kv_code_fp →
  REFUSE. Skip + log `park.skip reason=fp_mismatch`; do NOT quarantine (dir is valid,
  just foreign); let LRU reclaim.
- mlx_version mismatch → refuse by default; `EXO_DSV41_PARK_ALLOW_MLX_DRIFT=1` overrides.
- Legacy dirs without fingerprints (the existing 46) → refuse (stale history; one-shot
  cleanup optional).
- `EXO_DSV41_PARK_ALLOW_STALE=1` escape hatch (debug only).

This resolves the tension: most deploys don't touch the allowlist → restores keep working;
deploys that change KV math correctly pay a cold start. Mitigate the allowlist-miss risk
with a CI/unit test that fails if any module imported by the dsv41 forward pass is neither
allowlisted nor explicitly marked KV-neutral.

### G3 — park-on-shutdown

Touch points: worker top-level runner installs SIGTERM/SIGINT handler
(`loop.add_signal_handler`) setting `shutting_down`; work marshaled onto the engine thread.
Sequence: (1) stop admitting (503/retry); (2) drain in-flight turn up to
`EXO_DSV41_SHUTDOWN_DRAIN_S` (default 10 s) — if it completes it commits normally;
(3) if not, abort the turn — do NOT park mid-turn state; drop that session from residency
(the last G1 generation remains the restore point; lose only the aborted delta);
(4) `maybe_checkpoint(force=True)` for remaining dirty sessions, most-recent-first, under
a total `EXO_DSV41_SHUTDOWN_PARK_S` deadline (default ≈ 0.7 × supervisor ExitTimeOut minus
drain); (5) exit.

Deploy script: SIGTERM → wait for exit → `git reset --hard` → start. If reset must stay
first, pre-import everything shutdown-related at startup. ExitTimeOut ≥ 30 s.

Failure modes: SIGKILL before completion → temp dir left, previous generation intact,
temp swept at next startup. Abort leaves a half-written turn → guarded by step 3.

### G4 — multi-rank coordination (TP2)

- `generation` = monotonically increasing per conversation; every rank writes gen N with
  the same N for the same turn; retain the last TWO generations; delete gen N−1 only after
  every rank has committed gen N (ack; else lazy prune of two-gen window).
- Restore: coordinator gathers {key → set(committed gens, fps)} from all ranks over the
  existing control channel; picks the highest gen present on ALL ranks with matching fps;
  broadcasts "restore gen N" or "cold"; any rank fails → everyone cold.

### G5 — durability hygiene

Per file: write → fsync; manifest.json.tmp → fsync → rename → fsync dir. Whole generation
into `<key>.gen<N>.tmp/` → rename → fsync root. Startup: delete `*.tmp` first, ignore dirs
without manifests. **Restore is non-consuming** — dir stays, marked active base, LRU/gen
pruning reclaims later. LRU must reserve headroom of largest-session × 2 (new generation
written before old deleted); evict foreign-fingerprint + legacy first, then LRU.

### G6 — rewind-to-LCP (only if A4 shows divergence)

Ensure every checkpoint includes a snapshot at the last prompt end (before generated
tokens) with CSA/HCA tail buffer + carry captured; restore picks the largest snapshot
position ≤ LCP. Without it, stripped reasoning can degrade a 92K restore to the prefix
before the last turn (still a win; quantify).

### G7 — incremental checkpoints (defer)

`comp_kv`, `index_k`, `ids`, `engram_ids` are append-only → per-generation delta segments;
rewrite only the small mutable parts; compact every K generations. Per-turn writes drop
from GiB to MB. Only if wear/latency becomes a problem.

## Test plan

Unit (CPU-small model or synthetic composite caches):
- **T1** Round-trip byte-identity incl. zero-fill poison invariant (mid-HCA-block and
  block-boundary offsets).
- **T2** Each fingerprint field mismatch → refusal, dir untouched (not `.bad`); legacy
  manifest refused; ALLOW_STALE works.
- **T3** Corruption: truncated .bin → quarantine; missing manifest → ignored; temp dir at
  init → swept; garbage manifest → quarantine.
- **T4** Crash atomicity: SIGKILL at random points during gen N+1 write, 50×; gen N always
  restores cleanly.
- **T5** Mid-turn shutdown during prefill and during decode → nothing parked with
  `n_ctx ≠ offset`; restored state equals gen N.
- **T6** Cross-build gate: build A park → build B restore; same kv_code_fp → exact restore;
  different → refusal.
- **T7** Request during checkpoint → write aborts within one layer; request correct; no
  partial generation.
- **T8** Multi-rank: one rank missing gen N → everyone gen N−1 or all-cold; never mixed.
- **T9** Perf: decode tok/s within 2% with checkpointing on (donation trap); checkpoint
  pause ≤ one layer write.

Live A/B acceptance (studio cluster, one ~90K conversation, temp 0):
- Control (current main): restart → next-turn TTFT ≈ 282 s (cold). Treatment: expect
  `restore gen=N` and TTFT ≤ restore + delta + 5 s; target < 15 s for a 2K delta.
- Crash case: wait for idle checkpoint → kill -9 → same expectation.
- Mid-turn case: restart during decode → restore of previous generation; retry re-prefills
  only the last turn.
- KV-changing deploy (touch an allowlisted file) → cold + `fp_mismatch` logged, no restore.

Quality: needles at 1K/20K/45K/70K/88K depths all recovered post-restore; logit agreement
vs never-restarted resident exact (or < 1e-5 kernel noise); vs fresh cold prefill top-1
≥ 99% over 256 greedy tokens, mean KL < 1e-3; greedy 256 tokens identical to resident
control; DSpark acceptance ±2%; exact-repeat prompt uses persisted anchor and matches.

## Rollout

| Step | Content | Gate |
|---|---|---|
| 0 | Phase-0 audit | Hypothesis confirmed; topology + shutdown mechanism known |
| 1 | G2 fingerprint + G5 hygiene (refuse legacy) | T1–T3, T6 green; observe legacy skipped, no .bad storm |
| 2 | G4 (sharded — ours is) | T8 green |
| 3 | G1 idle checkpoint behind `EXO_DSV41_CKPT=1` | T4, T7, T9 green; watch logs 1 day (count, bytes, pauses) |
| 4 | G3 shutdown path + deploy-script order fix | T5 green; live graceful + crash A/B pass |
| 5 | Quality validation pass | Needle/logit/acceptance pass → default-on |
| 6 | G6 / G7 if data warrants | — |

## Effort (one engineer)

Phase 0 ≈ 0.5 d · G2 ≈ 1 d · G5 ≈ 0.5–1 d · G1 ≈ 2–3 d · G3 ≈ 1–1.5 d · G4 ≈ 2–4 d
(0.5 d if unsharded — ours is sharded) · tests + live validation ≈ 2–3 d →
**core ≈ 8–13 days**; G6 +1–2 d; G7 +3–5 d (optional).

## Risks

- **R1** allowlist misses a KV-affecting file → silent staleness. Mitigate: allowlist test
  + T6 deploy gate.
- **R2** donation/perf regression → T9.
- **R3** deploy runs `git reset --hard` before process exit → lazy imports/fingerprints
  see new code. Fix script order; pre-import at startup.
- **R4** 12 GB park budget vs two generations of a 3 GiB session + others → consider
  `MAX_GB` ≈ 24 (NVMe has room).
- **R5** memory pressure from CPU copies on full unified memory → stream one layer at a time (A10).
- **R6** unstable conversation keys → restore-by-key never fires post-restart; acceptable
  if prefix-match carries it; cache ids headers in memory at init for cheap LCP over ≤64 dirs.
- **R7** image-carrying conversations uncovered by design (vision-reuse fix) → documented
  out of scope.
