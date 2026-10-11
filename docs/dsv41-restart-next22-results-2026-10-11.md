# DSv4.1 restart-survival — `deploy/next22-rankfix` redeploy + re-validation results (2026-10-11, round 2)

## Verdict (WARN — no ABORT)

**WARN — no ABORT.** The rank-continuity fix is **FUNCTIONALLY PROVEN**: round 1's rank-swap
failure (restore refused after a rank-swapping relaunch; round-1 round `deleg_508119fb`) is
closed in BOTH relaunch cases — the graceful relaunch and the `kill -9` crash relaunch — with
the crash restore firing across a **live rank swap** (m4-1→rank1, m4-2→rank0: the exact
round-1 permutation). Bars applied verbatim per runbook
`docs/dsv41-restart-live-ab-runbook-2026-10-11.md` §3.4. **NOT promoted**; `EXO_DSV41_CKPT=1`
left live for observation (the runbook binds rollback only on ABORT — none fired).

Two cells land in the WARN band by the letter:
- **Graceful TTFT 15.029 s** — **0.029 s over** the ≤ 15 s bar (WARN band 15–60 s by the
  letter), single sample. Context: **17.7×** faster than the 265.7 s cold control; the
  resident-control floor is **14.389 s**, so restore overhead is only **0.64 s** — the overage
  is the engine's intrinsic rewind+refeed floor, not restore cost. Not re-run to fit
  (would be p-hacking).
- **DSpark −4.91 %** (0.9231 vs 0.9722 baseline) on the **first post-boot turn** (band
  ±2–5 %). A second resident turn on the SAME boot returned exactly **0.9722** ⇒ a
  first-turn-after-load effect, not a KV effect.
- Bar 3 (top-1 over 256 greedy tokens) is **NOT EVALUABLE AS SPECIFIED** — the fixture's
  answer is 46 tokens then stop, so only 46 tokens were compared (1.000). Recorded as a
  deviation, not folded into a pass.

## Provenance / refs

- **Round 1 (`deleg_508119fb`) FAILED** on the rank-swap defect; round 2 re-runs the same
  A/B after the fix.
- **Base / previously-live:** `deploy/next21-restart` @ `3afce11d7`.
- **Fix branch:** `p20/dsv41-restart-rankfix` @ `bebeb0066` (3 commits off `3afce11d7`:
  `ea3175a6b`, `30f32e448`, `bebeb0066`). Fix buildnote:
  `docs/dsv41-restart-rankfix-buildnote-2026-10-11.md` — lives on
  `p20/dsv41-restart-rankfix`, NOT on main.
- **Construct/deploy branch:** `deploy/next22-rankfix` — created at `bebeb0066`, construct
  note committed ⇒ tip **`ade481853`** (= `bebeb0066` + construct note); pushed, no force.
  Construct tip (deployed) = `ade481853`. Construct doc:
  `docs/dsv41-next22-construct-2026-10-11.md` — lives on `deploy/next22-rankfix`, NOT on main.
- **Test suite on the construct worktree** (`/private/tmp/p20-next22`): **367 passed in
  51.63 s** (Phase A, offline).
- **Rollback ref (ABORT only):** `deploy/next21-restart` + `EXO_DSV41_CKPT=1` — an interim
  re-point, NOT the known-good `b5f055185` (`deploy/next20-vision` tip; tag
  `known-good-vision-20261010-174510`). UNSPENT — no abort occurred.
- **Runbook:** `docs/dsv41-restart-live-ab-runbook-2026-10-11.md` (§3.4 pre-registered bars;
  verbatim application confirmed by a second opinion — WARN, not over-strict).
- **Cluster:** API `http://192.168.86.48:52415`; nodes `macstudio-m4-1` (192.168.86.48) and
  `macstudio-m4-2` (192.168.86.47); model `dealignai/DeepSeek-V4.1-Flash-UNCENSORED-EXL3-2.9bpw`
  (Tensor/MlxJaccl, min_nodes 2). Harness: scratch `dsv41_ab/dsv41_live2/turn2.py`.
- **Fixture:** base_prompt 368,611 chars ⇒ 69,331 real tokens; delta_prompt = base + 8,296
  chars (~1.6K tok) tail; 5 needles at 1K/20K/45K/70K/88K; salt `8b71a8ac44452be1`; greedy.
- **Authoring note:** this doc is **OFFLINE-authored** — no cluster, API, or SSH contact
  during authoring; every number below is transcribed from the round-2 run log.

## Per-bar results (graceful + crash)

| Bar | Spec (§3.4) | Graceful | Crash | Result |
|---|---|---|---|---|
| 1. Next-turn TTFT | ≤ 15 s for the ~2K delta on the ~92K conversation, logged as a restore (not a cold prefill). (vs 282 s plan control / 379 s closest-logged.) | **15.029 s**, restore fired (rewind 69378, reuse 67584, prefill 3320) — **0.029 s OVER** ⇒ WARN band 15–60 s; 17.7× vs the 265.7 s cold; resident floor 14.389 s ⇒ **+0.64 s** restore overhead | **0.821 s** — PASS (exact-match restore across the live rank swap; prefill=0; 70,951-row delta-state restored) | **WARN (graceful)** — single sample, deviation #1; **PASS (crash)** |
| 2. Needles | Recovered at all 5 depths (1K/20K/45K/70K/88K) | **5/5** | **5/5** | **PASS** |
| 3. Top-1 agreement | ≥ 99 % over 256 greedy tokens; greedy-256 identical to resident control | **1.000** (46/46) vs `g_res_control` and `g_base` | **1.000** (46/46) vs `g_anchor_graceful` and `g_res_control` | **NOT EVALUABLE AS SPECIFIED** — 256 tokens not reachable (fixture stops at 46); deviation #2 |
| 4. DSpark acceptance | Within ±2 % of the resident baseline (0.9722) | 0.9231 = **−4.91 % rel** ⇒ WARN band 2–5 % | 0.9231 on the first turn after boot; same-boot resident probe (`k_resident_probe`, prefill=0, identical restored KV) returned **0.9722 = baseline** | **WARN** — first-turn-after-load effect, deviation #3 |
| 5. Exact-repeat anchor match | Exact-repeat prompt uses the persisted anchor and matches | `g_anchor_graceful`: **prefill=0**, reuse=70904, TTFT **1.715 s**, needles 5/5, mean_accept 0.9722, top-1 1.000 vs treatment | Not separately run; the crash restore itself was an exact KV match (prefill=0) and the crash generation is top-1 1.000 vs `g_anchor_graceful` | **PASS** (Bar5 anchor PASS, graceful arm) |
| 6. Cross-build/foreign refusal | Cross-build/foreign dirs refused (`fp_mismatch`); nothing stale served | `park.skip` = legacy ×36 + `fp_mismatch` ×2 (pre-fix dirs only); no foreign-rank skip on any fix dir | `park.skip` = legacy ×35 + foreign-rank ×1 — the ×1 is a **legacy no-site dir** via the documented fallback (park.py:1593-1604), expected cold-accept; **ZERO `foreign-site` skips** | **PASS** (bar6 zero foreign-site) |
| ABORT (any ⇒ stop, roll back) | §3.4 triggers 1–6 | — | — | **None fired** |

**ABORT checks — none fired:** (1) no TTFT > 60 s (worst case 15.029 s, crash 0.821 s);
(2) no needle miss (5/5 both cases), top-1 1.000 (not < 95 %), DSpark −4.91 % (inside ±5 %);
(3) no collective hang / rank skew / runner bootstrap re-load on any restart;
(4) no false `fp_mismatch` on a same-build dir (every fix dir restored fine) and no foreign
dir served; (5) the `kill -9` left no half-written generation that then restored;
(6) cluster HEALTHY + runners READY 2/2 at every boot.
Reference points: cold base (boot-1 first request, full base, cold prefill) TTFT **255.026 s**
(restore scan correctly refused stale dirs — legacy + `fp_mismatch` pre-fix dirs — so it went
cold); resident control `g_res_control`: TTFT **14.389 s**, prompt 70,904, needles 5/5,
mean_accept **0.9722** (reuse 67584, rewind 69378, refed 3320); `g_base` (resident re-send):
reuse 69331, needles 5/5, mean_accept 0.9722.

## Restore-path evidence

- **Graceful case (relaunch #2; ranks NOT swapped this boot):** `park.restore gen=2 rank=0
  (all ranks)` on m4-1 / `gen=2 rank=1 (all ranks)` on m4-2, then
  `[DSV41] parked session restored: 69378 rows from SSD (key=p:168effde...)`. NO foreign-rank
  skip on any fix dir; `park.skip` = legacy ×36 + `fp_mismatch` ×2 (pre-fix dirs only).
  Turn reuse: prefill=3320, reuse=67584, rewind=69378.
- **Crash case (relaunch #3; the exact round-1 defect — RANKS SWAPPED, m4-1=rank1,
  m4-2=rank0):** `kill -9` on exo on both nodes (m4-1 pids 12913/12914/12915/12926; m4-2 pids
  14397/14398/14399/14409), 0 processes left, `/state` unreachable. Pre-crash newest on disk =
  gen=3 @69378 (base-state). **RESTORE FIRED ACROSS THE SWAP at 13:23:22:** `park.restore
  gen=2 rank=1 (all ranks)` on m4-1 (its dir named `.r0`) and `gen=2 rank=0 (all ranks)` on
  m4-2 — admitted via **SITE match** (manifest site == store hostname) despite the rank
  permutation; `[DSV41] parked session restored: 70951 rows from SSD` on BOTH nodes
  (the gen=2 delta-state @70951, written 13:14:56). `park.skip` = legacy ×35 +
  foreign-rank ×1; source-confirmed (park.py:1593-1604) that the foreign-rank line is emitted
  ONLY for a dir with NO site field (legacy fallback) ⇒ the ×1 skip is a pre-fix legacy dir
  (expected cold-accept). **ZERO `foreign-site` skips.**
- **Checkpoint inventory (both ranks, all_ranks=True; the new build writes manifest
  site=hostname):** gen=1 @69339; gen=2 @69378 (base-state) 13:02:07; gen=2 @70951
  (delta-state) 13:14:56; gen=3 @69378 (base-state) 13:16:03. Manifest check:
  m4-1 site="Adams-Mac-Studio-M4-1" rank 0; m4-2 site="Adams-Mac-Studio-M4-2" rank 1;
  model_fp identical on both ranks; dir NAME still tagged `.r<rank>` (legacy naming) but
  **admission keyed on manifest site**. has_draft=True, draft_snaps=31/32, anchors=1.
- **Fix mechanism (brief):** SITE identity admission (manifest site == store hostname)
  **replaces the `.r<rank>` gate**; rank-invariant park layout; MAGIC_IDENTITY startup
  agreement; the audit fixed a negative-verdict latch bug.

## Mechanical knob gate (ps eww) — PASS at all 3 boots

| Boot | m4-1 | m4-2 |
|---|---|---|
| #1 treatment | pid 8833, CKPT=1, JIT=1, commit ade481853, rank 0/2 | pid 9814, CKPT=1, JIT=1, commit ade481853, rank 1/2 |
| #2 graceful | pid 12913, CKPT=1, JIT=1, commit ade481853, rank 0/2 | pid 14397, CKPT=1, JIT=1, commit ade481853, rank 1/2 |
| #3 crash (SWAP) | pid 16536, CKPT=1, JIT=1, commit ade481853, rank 1/2 | pid 18266, CKPT=1, JIT=1, commit ade481853, rank 0/2 |

- `EXO_DSV41_CKPT=1`, commit `ade481853`, and `JIT=1` present on BOTH nodes at every boot
  (ps eww-derived).
- `EXO_DSV41_SHUTDOWN_PARK` absent everywhere (intended default-off).
- Node ids regenerate each boot; rank = lexicographic(node_id) ⇒ the crash boot's swap is a
  property of the launcher, not a defect.

## Boot inventory

- **Total launcher runs: 3** (1 treatment + 2 test relaunches):
  **#1 treatment 12:47–13:00** (PASS mechanics: SIGTERM CLEAN_EXIT on both; sync `ade481853`;
  HEALTHY; READY 2/2; DSv4 auto-placed jit=False) · **#2 graceful 13:09–13:14** (PASS) ·
  **#3 crash 13:17–13:22** (after `kill -9` both runners; PASS).
- **Abort-recovery reserve: UNSPENT** — no abort occurred.
- **Pre-flight 12:44: PASS** — storm client no conn; 0 'outside int32'/shape-crash lines on
  both nodes; `/state` runners RunnerReady 2/2; DSv4 jit=False (pinned); both nodes at
  `3afce11d7` with CKPT=1.
- The launcher is the **ONLY relaunch path** — no reboots, no node-side file edits.

## Final live state

- `deploy/next22-rankfix` @ `ade481853` **live on BOTH nodes** (ps eww-confirmed).
- `EXO_DSV41_CKPT=1` left in runner env for observation; `EXO_DSV41_SHUTDOWN_PARK` absent
  (default-off).
- DSv4 instance `3844aa6d` jit=false (pinned); runners **RunnerReady 2/2**; nodes **HEALTHY**.
- Ranks swapped on the live crash boot (m4-1=rank1, m4-2=rank0), **serving normally**.
- Shared checkout `~/repos/exo` detached at `ade481853`.
- Quiet maintained: 0 laptop conns to :52415; 0 'outside int32' crashes on either node.
- No force-push; no branch/store deleted; no node-side file edits.

## Deviations / unmeasured

1. **Graceful TTFT 15.029 s** is 0.029 s over the ≤ 15 s line (single sample; run-to-run
   jitter likely > 0.03 s). Not re-run to fit (would be p-hacking).
2. **Top-1 measured over 46 tokens, not 256** (fixture stops at 46) — not evaluable as
   specified.
3. **DSpark −4.91 % on the first post-boot turn**; same-session resident turn returns to
   baseline (0.9722); draft state IS parked (has_draft=True) — warmup-vs-restore not fully
   discriminated (no fresh-boot cold-first-turn control).
4. **Mid-turn §3.3 case NOT run** (best-effort; boot budget spent: 3 launcher runs; reserve
   held for abort-recovery only).
5. **Rank-swap exercised only in the CRASH boot** (graceful boot had stable ranks). The
   site-admission path is nonetheless exercised in both (site dirs admitted, zero
   foreign-site).
6. **External note:** a light GET /metrics poller (~70 s) from another client was present at
   pre-flight (read-only, no model requests); the cluster was otherwise quiet.

## Recommended bar re-registration (proposal — runbook NOT edited)

**PROPOSAL ONLY — the runbook `docs/dsv41-restart-live-ab-runbook-2026-10-11.md` is NOT
edited by this document.**
- **TTFT bar → resident floor + 2 s** (or median-of-N): the ≤ 15 s line had only ~0.6 s of
  headroom over the 14.389 s resident-floor control, so a single-sample 0.029 s overage
  triggered the WARN band; floor+2 s (⇒ ≤ 16.4 s) or a median-of-N rule would keep the bar
  meaningful without p-hacking pressure.
- **DSpark:** re-test in a promotion round with a **fresh-boot cold-first-turn control** to
  discriminate warmup-vs-restore before ±2 % gating is trusted on first turns.

## Next steps

- **Observation** on the live `EXO_DSV41_CKPT=1` state (cluster is healthy and serving on the
  fix build).
- Any **promotion / default-on decision is reserved to the OWNER** — this round neither
  promotes nor edits the runbook.