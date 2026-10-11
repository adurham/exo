# DSv4.1 restart-survival — consolidated PRE-LIVE review (2026-10-11)

**Purpose.** One consolidated, cited read of the `dsv41` restart-survival chain
(G2+G5 → G4 → G1 → G3) as it stands at tip, plus a full **open-questions ledger**
with a recommended disposition per row, to drive the pre-live triage.

**Method / sources (all read from the branches, laptop-local; no node contact).**
This review restates **nothing** that is not quotable from the source buildnotes:

| Piece | Commit | Branch | Buildnote (source of truth) |
|---|---|---|---|
| G2 + G5 | `46990ec73` | `p20/dsv41-restart-g2g5` | `docs/dsv41-restart-g2g5-buildnote-2026-10-10.md` |
| G4 | `360fadbf6` | `p20/dsv41-restart-g4` | `docs/dsv41-restart-g4-buildnote-2026-10-10.md` |
| G1 (initial + hardening) | `4647fcb7d`, `b2f009e69` | `p20/dsv41-restart-g1` | `docs/dsv41-restart-g1-buildnote-2026-10-10.md` |
| G3 (initial + fix pass) | `f30ad7c51`, `4f2a578b6` | `p20/dsv41-restart-g3` | `docs/dsv41-restart-g3-buildnote-2026-10-10.md` |
| Plan / audit | `0ae8f4f78` / — | — | `docs/dsv41-restart-survival-plan-2026-10-10.md`, `docs/dsv41-restart-survival-phase0-audit-2026-10-10.md` |

**No-node-contact affirmation.** This review performed only `git fetch origin`,
`git worktree add`, `git show`, `git ls-tree`, `git rev-parse` and file reads
against `https://github.com/adurham/exo.git`. No ssh / no node / no boot / no
relaunch / no kill / no bench / no GUI; nothing deployed; no other worktree
touched. This document is doc-only and changes no code.

---

## PART 1 — What the full chain contains, per piece

### Chain shape (verified against the commit graph)

- **6 commits, linear, off `main` `94daf8c18`:**
  `46990ec73` → `360fadbf6` → `4647fcb7d` → `b2f009e69` → `f30ad7c51` → `4f2a578b6`.
  Verified: `git rev-parse 46990ec73^ == 94daf8c18c469e77cd53671ccf63897d72ef386a`,
  and `git log --oneline 94daf8c18..4f2a578b6` lists exactly those 6 commits.
  (`main` has since advanced past `94daf8c18`; the chain's base is `94daf8c18`.)
- Each buildnote records its branch/worktree and the "stacked on" parent:
  G4 "stacked on G2+G5 (`46990ec73`)" [G4 note, header]; G1 "stacked on G4
  (`360fadbf6`)" [G1 note, header]; G3 "stacked on G1 (`b2f009e69`)" [G3 note,
  header]. Consistent with the graph.
- **All five chain branches are pushed:** `origin/p20/dsv41-restart-{g2g5,g4,g1,g3}`
  resolve to `46990ec73`, `360fadbf6`, `b2f009e69`, `4f2a578b6` respectively
  (`git ls-remote origin`).
- All four pieces are **default-OFF**: G1 behind `EXO_DSV41_CKPT=1` [G1 note, line 6],
  G3 behind `EXO_DSV41_SHUTDOWN_PARK=1` [G3 note, lines 12-16]; G2/G5 and G4 are
  always-on paths but G4's coordination is a no-op when there is no coord group
  (`ParkCoordinator.from_group` → `None`, single-rank path) [G4 note, lines 81-82].
  Production `b5f055185` is untouched [G2G5 note, line 8; G4 note, lines 9-10;
  G3 note, line 9].

### Totals at tip (`4f2a578b6`): 346 tests — quoted from source

The G3 buildnote states verbatim (its §Test evidence):

> `322 passed in 31.27s     (baseline, before any change — given)`
> `346 passed in 26.19s     (after: 322 + 24 new)`

— i.e. **346 dsv41 tests at tip `4f2a578b6`**, the arithmetic being fully chained
across the four buildnotes:

| Piece | Commit(s) | Test file added | Cases added | Cumulative (buildnote quote) |
|---|---|---|---|---|
| G2 + G5 | `46990ec73` | `test_dsv41_fingerprint.py` (49), `test_dsv41_park_g2g5.py` (24) | +73 | **178 → 251** ("178 passed"; "251 passed in 22.78s = 178 + 49 + 24") |
| G4 | `360fadbf6` | `test_dsv41_park_g4.py` (39 cases; 32 test functions) | +39 (32 G4 + 7 hardening) | **251 → 290** ("290 passed … (baseline 251 + 32 G4 + 7 hardening = 290)") |
| G1 | `4647fcb7d` + `b2f009e69` | `test_dsv41_park_g1.py` (28 cases → +4 hardening = 32) | +28, then +4 | **290 → 318 → 322** ("318 passed … (290 + 28 new)"; "318 → 322 passed (+4)") |
| G3 | `f30ad7c51` + `4f2a578b6` | `test_dsv41_park_g3.py` (24 cases) | +24 | **322 → 346** ("346 passed … (after: 322 + 24 new)") |

178 + 73 + 39 + 32 + 24 = **346**. All six source figures are quotes from the
respective buildnotes' §Test evidence sections.

### G2 + G5 — build fingerprint + refusal policy; durability hygiene

Buildnote: `docs/dsv41-restart-g2g5-buildnote-2026-10-10.md`.

**Files (verified via `git show --numstat 46990ec73`):**

- `src/exo/worker/engines/mlx/dsv41/fingerprint.py` — **NEW** (+714/−0; note: "NEW (714 lines)").
- `src/exo/worker/engines/mlx/dsv41/park.py` — MOD (+380/−46).
- `src/exo/worker/engines/mlx/dsv41/engine.py` — MOD (+51/−0).
- `src/exo/worker/engines/mlx/dsv41/session.py` — MOD (+9/−0).
- `src/exo/worker/engines/mlx/dsv41/tests/test_dsv41_fingerprint.py` — **NEW** (383 lines, 49 tests).
- `src/exo/worker/engines/mlx/dsv41/tests/test_dsv41_park.py` — MOD (see corroboration note).
- `src/exo/worker/engines/mlx/dsv41/tests/test_dsv41_park_g2g5.py` — **NEW** (653 lines, 24 tests).

**Key functions (per buildnote):** `BuildFingerprint`
(`model_fp`/`tokenizer_fp`/`kv_code_fp`/`mlx_version`/`layout_fp`, `as_dict`/`from_dict`),
`compute_fingerprint` (computed once at `Dsv41Engine.__post_init__`, never lazily),
`kv_code_fp` = sha256 over the explicit `KV_ALLOWLIST`, `layout_fp`,
`check_manifest(...) -> (ok, reason)` (`fp_mismatch`/`legacy`/`mlx_drift`), and the
CI/unit guard `allowlist_problems()`. `PARK_VERSION` 1→2; policy enforced in
`ParkedStore._admit` (used by `find`/`try_restore`); G5 adds `_write_bytes_synced`,
`os.replace` commit, `_sweep_tmp` startup sweep, ignore-manifestless, quarantine
`*.bad`, **non-consuming restore** (`.active` marker), generations
`<prefix>.gen<N>.<uuid8>`, retention newest-two, and LRU headroom.

**Tests (24 cases in `test_dsv41_park_g2g5.py`):** T1 (boundary zero-fill/poison),
T2 (per-field mismatch → refuse, dir untouched; legacy/pre-v2 refused;
ALLOW_MLX_DRIFT/ALLOW_STALE), T3 (truncated `.bin` → quarantine; manifestless
ignored; `.tmp` swept; garbage manifest quarantined), T4 (crash atomicity, 50×
SIGKILL mid-gen-2), T6 (cross-build gate), plus non-consuming restore /
two-generation retention / foreign-legacy-evicted-first / park-never-evicts-just-written.

### G4 — multi-rank all-or-nothing restore + cross-rank generations

Buildnote: `docs/dsv41-restart-g4-buildnote-2026-10-10.md`.

**Files (verified via `git show --numstat 360fadbf6`):**

- `src/exo/worker/engines/mlx/dsv41/coord.py` — **NEW** (532 lines).
- `src/exo/worker/engines/mlx/utils_mlx.py` — MOD (+61/−0; `_sanitize_int32`, `mx_all_gather_ints`).
- `src/exo/worker/engines/mlx/dsv41/park.py` — MOD (+399/−7).
- `src/exo/worker/engines/mlx/dsv41/session.py` — MOD (+65/−15).
- `src/exo/worker/engines/mlx/dsv41/engine.py` — MOD (+7/−0).
- `src/exo/worker/engines/mlx/dsv41/fingerprint.py` — MOD (+1/−0; `coord` added to `KV_NEUTRAL`).
- `src/exo/worker/engines/mlx/dsv41/tests/test_dsv41_park_g4.py` — **NEW** (744 lines; 32 test functions, 39 collected cases).

**Key functions (per buildnote):** `key_int`, `build_identity_digest`,
`Candidate`, `RankParkState`, `RestoreDecision`, **`decide_restore`** (pure, the T8
unit), `agree_generation`, `prune_floor`, `encode/decode_restore_payload`,
`ParkCoordinator` (`from_group`, `exchange`), `run_restore`, `run_park`; constants
`MAX_CANDIDATES=4` and the per-exchange magics. Transport is the existing QP-less
TCP coord subgroup via **one** fixed-width int32 `mx.distributed.all_gather`
(`mx_all_gather_ints`, CPU stream, immediate `mx.eval`). Four exchanges: Restore,
Restore-ack, Park-gen (`N = 1 + max over ranks`), Park-ack (ack-gated prune,
keep=2). Hardening: FIX A (guarded post-exchange tail + guarded `_evict`), FIX B
(`_sanitize_int32`, `_gen_from_name` refuses absurd N), FIX C (`_rank_matches`).

**Tests:** T8 pure `decide_restore` (incl. an exhaustive 2-rank sweep over
`{1,2,3}×2` digests), fault injection, two simulated in-process ranks in lockstep
(a SKIPPED exchange FAILS rather than hangs), park ack-gated prune, backward compat
for pre-G4 dir names, single-rank unchanged, and the FIX A/B/C regressions.

### G1 — after-turn idle checkpoint

Buildnote: `docs/dsv41-restart-g1-buildnote-2026-10-10.md`.

**Files (verified via `git show --numstat` on the two commits):**

- Initial (`4647fcb7d`): `runner/runner.py` (+31/−1), `engines/base.py` (+17/−0),
  `dsv41/engine.py` (+71/−1), `dsv41/agreement.py` (+10/−2), `dsv41/session.py`
  (+234/−1), `dsv41/park.py` (+438/−134), `dsv41/coord.py` (+206/−0),
  `dsv41/tests/test_dsv41_park_g1.py` (**NEW**, 921 lines).
- Hardening (`b2f009e69`): `dsv41/park.py` (+127/−17), `dsv41/session.py` (+49/−5),
  `dsv41/tests/test_dsv41_park_g1.py` (+156).

**Key functions (per buildnote):** `Engine.idle_tick_seconds` / `Engine.idle_tick`
(base + dsv41 override); `Dsv41Sessions.checkpoint_candidate` (pure local gate),
`Dsv41Sessions.maybe_checkpoint` (manager entry point); `Conversation.finish(checkpoint=True)`
(turn-commit point), `Conversation.ckpt_hash`/`stream_changed` (MED-1 fix);
`agree_on_tasks` now **returns** the `mx_any` result; `_GenerationWriter`
(`begin`/`write_next`/`commit`/`abort`), `ParkedStore.open_generation`,
`checkpoint_conversation`, `_local_coordinator`; coord `IdleDecision`/`decide_idle`/
`decide_poll`/`CheckpointOutcome`/`run_checkpoint`; hardening
`ParkedStore._reserve_headroom(protect=…)`/`_protect_own`/`retain_keyed`.

**Design finding:** the idle hook is in the runner's idle **wait**, not `step()`
(`Runner.handle_generation_tasks` calls `step()` only `while self.active_tasks`;
between requests the runner blocks in `Runner.main` on `_work_queue.get()`)
[G1 note, lines 12-20].

**Tests:** T7 (7 cases: request-during-checkpoint aborts within one unit; abort
point sweep; pause ≤ one unit write; two ranks request-on-one-rank aborts both),
T9 (2 cases: donation survives commit+abort via address-identity oracle; decode-step
time within 2%), + keep-resident, split-preserves-format, gate, one-per-tick,
ENOSPC, disabled-store, clean-eviction, two-rank lockstep. Initial: "25 functions,
28 collected cases" (+4 hardening regressions). Verified at tip: 29 top-level
`def test_` (25 + 4 hardening), reconciling with buildnote.

### G3 — park-on-shutdown

Buildnote: `docs/dsv41-restart-g3-buildnote-2026-10-10.md`.

**Files (verified via `git show --numstat` on the two commits):**

- Initial (`f30ad7c51`): `dsv41/shutdown.py` (**NEW**, 300 lines),
  `dsv41/session.py` (+113/−4), `dsv41/engine.py` (+63/−0), `engines/base.py`
  (+19/−0), `runner/runner.py` (+117/−5), `runner/bootstrap.py` (+32/−7),
  `dsv41/fingerprint.py` (+1/−0; `dsv41.shutdown` → `KV_NEUTRAL`),
  `dsv41/tests/test_dsv41_park_g3.py` (**NEW**, 614 lines).
- Fix pass (`4f2a578b6`): `dsv41/shutdown.py` (+18/−1),
  `dsv41/tests/test_dsv41_park_g3.py` (+80/−6), buildnote corrections.

**Key functions (per buildnote):** `shutdown_park_enabled`, `ShutdownBudget`
(`fits`, `park_deadline`), `shutdown_budget`, `ShutdownSignal` (`on_signal`
idempotent, `install`), `ShutdownOutcome`, `run_shutdown_park` (deadline-aware
`request_pending` gate, MED-A); `Dsv41Sessions.checkpoint_candidate(force=, exclude=)`,
`abort_inflight`, `force_checkpoint_all`; `Dsv41Engine._activate_next` guard,
`begin_shutdown`, `shutdown_park`; runner `_shutdown_pending`,
`_shutdown_budget_or_default`, `_arm_shutdown_engine`, `_shutdown_drain_expired`,
`_abort_inflight_for_shutdown`, `_run_shutdown_park`, `_next_work_item` (bounded
wait only while armed).

**Sequence:** stop admitting → drain (`DRAIN_S`) → abort in-flight (drop from
residency; never park mid-turn) → force-checkpoint remaining dirty between-turns
conversations most-recent-first under `PARK_S` + a hard `signal+GRACE−MARGIN` cap via
the existing §G1/§G4 path → clean exit. Budget: `DRAIN 10 + PARK 4 + MARGIN 1 ≤ GRACE 15`.

**Tests (T5, 24 cases):** budget/gate, signal idempotency/`SystemExit` cases,
runner hooks, engine wiring, sequence over the real cache (mid-turn prefill and
decode → abort+drop, no `n_ctx != offset` generation written, last §G1 generation
restores byte-identically), deadline (`PARK_S` allows exactly one attempt; the park
deadline aborts an in-progress checkpoint after exactly ONE unit — MED-A
regression; `_inflight` gate rejects a re-entered committed conversation — LOW-D
regression), flag-off, and two-rank G4 lockstep.

### Corroboration & deltas (file/func/test counts)

- Test-file presence at tip (`git ls-tree -r --name-only 4f2a578b6 -- …/dsv41/tests`):
  `test_dsv41_park_g1.py`, `test_dsv41_park_g2g5.py`, `test_dsv41_park_g3.py`,
  `test_dsv41_park_g4.py`, `test_dsv41_fingerprint.py`, `test_dsv41_park.py` — **all present.**
- Top-level `def test_` counts at tip: g2g5 **18** (+3 `parametrize`), fingerprint
  **26** (+5 `parametrize`), g4 **32**, g1 **29**, g3 **24** — consistent with the
  buildnotes' case counts (g4 "32 test functions"; g3 "24"; g1 25+4).
- Allowlist counts at tip (verified by reading `fingerprint.py`):
  `KV_ALLOWLIST` = **31** entries (7 exo dsv41 + 3 disaggregated + 21
  `mlx_lm.models.deepseek_v41`) — matches G2G5 note; `KV_NEUTRAL` = **18** entries
  (16 original + `coord` per G4 + `dsv41.shutdown` per G3) — matches the additions
  claimed by the G4 and G3 notes.
- **Minor numeric deltas (buildnote figure UNVERIFIED exact; actual numstat given):**
  - G2G5 note says `tests/test_dsv41_park.py` "(+23/−8)"; `git show --numstat
    46990ec73` reports **+20/−3**. (Direction and intent match; exact line counts differ.)
  - G4 note says `park.py` "(+406)"; `git show --numstat 360fadbf6` reports
    **+399/−7**. (Net ≈ +392 vs claimed +406.)
  These do not affect any count that matters (files, functions, cases, totals); they
  are flagged rather than asserted.

---

## PART 2 — OPEN-QUESTIONS LEDGER

Dispositions are one of **fix-now / defer-to-deploy / live-validate / accept-monitor**.
"fix-now" is reserved for a defect that must be corrected before any live use and
that is correctable within this chain; **no row qualifies** (all remaining items are
either deploy-time changes or live-only measurements).

| ID | Open question | What's known (buildnote citation) | Disposition | Rationale |
|---|---|---|---|---|
| **a** | G3 MED-B — rank-local breaks in the forced shutdown checkpoint → deterministic TP2 collective-count mismatch (best-effort/COLD). | G3 note §Multi-rank (TP2) + OQ-1: the force LOOP's every break (`now() >= deadline`, `precheck()` False, candidate None, attempt non-committed) is a **rank-local decision with no agreeing exchange**, so the two ranks can issue a **different number of collectives**; compounded by step-1 `begin_shutdown` flipping `_shutting_down` at slightly different times, making `precheck()` itself asymmetric. Probe exercised "candidate-count 50 vs 0, standdown 50 vs 0, deadline 50 vs 0" — **deterministic, not a race**. "must be validated on a live TP2 pair … NOT rank-invariant." The G4 protocol *inside* one checkpoint **is** rank-invariant (a candidate disagreement is a no-go on both ranks). | **live-validate** | The mismatch is a code-deterministic property that can only be *observed and bounded* on a real TP2 pair; default-OFF means no live exposure yet. Worst case is COLD-safe **by bytes** (SIGKILL → `*.tmp` swept, previous generation kept), but a real desync hangs until SIGKILL, which **also skips the clean RDMA teardown** (the QP-leak / "No device connected" wedge `bootstrap.py:208-217` exists to prevent) [G3 OQ-1] — so the live run must confirm both the hang and whether the design-only **symmetric agreeing gate** is required before the G3 fast path is trusted. |
| **b** | G1 MED-3 — idle `mx_any` could mispair with a mid-turn peer's cancel-check collective; magic-tag `agree_on_tasks` before default-ON. | G1 note §Hardening MED-3 + OQ-2: a rank that returns to its idle wait while the peer is still inside a request could pair, "silently and with the same shape," with the peer's cancellation-check `mx_any`. Pre-existing invariant (already hangs without G1); with G1 "the failure mode could become silent mispairing instead of a hang." Mitigations in place: idle collective only after `CKPT_IDLE_S` since this rank's last `step()`, plus strict magic validation of the IDLE/POLL rows. *Not done:* "tagging `agree_on_tasks`' `mx_any` with a magic, which would change a hot-path collective. **Recommended before default-on**." | **defer-to-deploy** | This is a **pre-default-ON gate on a hot-path collective** (G1 note §Deviations 2: "the hot path is untouched"). G1 ships default-OFF, so nothing is live-exposed; the magic tag is a bounded change to make *before* the G1 default is flipped in production, i.e. a deploy-time gate — not a change this doc-only, offline chain should make. |
| **c** | coord-wait vs the 45 s supervisor hang watchdog. | G1 note OQ-1: "The supervisor's 45 s hang watchdog is gated on `in_progress`, so it does not fire while the runner is idle. Once a task is in progress, the tick has already aborted within one unit." G1 note OQ-4: each POLL exchange waits for the slowest rank's unit write — units are MB-scale ("tens of ms expected on NVMe"), "far below both the coord deadline and the hang window. **Unmeasured on hardware** (A9/A10 still NEEDS-GO)." G4 note OQ-3: the RESTORE_ACK/PARK_ACK exchange "covers the peer's GB-scale park/restore I/O; the coord socket deadline is ~150 s while the supervisor hang timeout can be ~45 s. A slow peer write could trip the supervisor. **Unmeasured; possibly needs a heartbeat.**" | **live-validate** | Two distinct claims: G1's per-unit POLL waits are argued small but **unmeasured on hardware**; G4's ACK covers a peer's GB-scale write and could exceed ~45 s. Only a live timed run resolves whether a waiting rank trips the watchdog, and whether a heartbeat is required. COLD-safe if it trips (SIGKILL → temp swept), so no code change is forced before the measurement is taken. |
| **d** | resident-set symmetry relied-on-not-enforced. | G4 note OQ-1: coordinated restore is entered from `Dsv41Sessions.get` only when the conversation is NOT resident, and park from `_evict` on LRU overflow; "rank-invariant **iff both ranks hold the same resident set** — true today because both rank process the identical request stream through identical deterministic LRU/LCP logic, but it is **an invariant G4 relies on, not one it enforces**. Worth a live check (and maybe a cheap `mx_any` assertion) on TP2." G1 note OQ-1 also lists "resident-set divergence (relied on, not enforced, as in G4 OQ-1)" as a possible cause of an asymmetric idle tick. | **accept-monitor** | Structurally guaranteed by the current architecture (identical request streams → identical deterministic LRU/LCP); a violation degrades to COLD, not corruption (G4 note OQ-1, OQ-4). Accept the invariant; monitor on TP2 and only add the optional `mx_any` tripwire if a divergence is ever observed. |
| **e** | cross-node `tokenizer_fp` path equality. | G4 note OQ-2: `build_identity_digest` includes `tokenizer_fp`, which hashes the tokenizer's `name_or_path`; "**if the absolute paths differ between the two nodes every restore would go COLD.** Must be verified on the cluster (or the field canonicalized to a basename)." | **live-validate** | A concrete, cheap cross-node equality check (compare the two nodes' tokenizer paths); if unequal, canonicalizing `tokenizer_fp` to a basename is the fix. Cannot be settled offline; a mismatch would silently COLD every restore (safe but total loss of the feature's value). |
| **f** | per-rank LRU eviction COLD-safety. | G4 note OQ-4: rank-local byte/count LRU (`enforce_budget`, `_reserve_headroom`) "can still evict a generation on one rank only (different disk usage per node). The restore protocol stays safe (falls to a lower common gen or COLD, never mixed), but the 'common gen always survives' guarantee covers the **ack-gated prune, not budget eviction.** Cluster-agreed LRU eviction is a possible follow-up." G1 note OQ-3 agrees (COLD-safe). | **accept-monitor** | Explicitly COLD-safe by design — a rank-local eviction can only drop the shared generation to a lower common one or to COLD, never a mixed state. Accept; monitor disk-usage divergence; cluster-agreed LRU is an optional follow-up, not a pre-live requirement. |
| **g** | A9/A10 exact park numbers NEEDS-LIVE. | G2G5 note §Open questions: "**A9/A10 stay `NEEDS-GO`**: exact park/restore throughput and peak-at-92K memory need an instrumented run." Phase-0 audit: A9 **PARTIAL** — a 92 K-row session park ≈ **471 MiB** written, "NEEDS-GO for an exact throughput number (requires instrumented run)"; A10 **PARTIAL** (peak-at-92K needs a run); "Use 471 MiB/session for park sizing." G3 note OQ-3: whether a typical dirty set parks inside the ~4 s `PARK_S` on NVMe "is unmeasured. … Needs an instrumented run." | **live-validate** | The exact throughput + peak-memory numbers gate the shutdown budget (`PARK_S`, GRACE headroom) and the `MAX_GB` sizing (R4: consider ~24 GB). Only an instrumented live run produces them; nothing offline can. Keep the 471 MiB/session figure as the working sizing input until then. |
| **h** | T9-on-Metal PENDING (to run AFTER the MoE spike's studio2 bench window). | G1 note OQ-6: "Donation is device-independent and the address oracle is exact, but end-to-end decode tok/s with checkpointing on was **not measured on studio hardware.** The plan's rollout gate ('watch logs 1 day: count, bytes, pauses') should include **one before/after tok/s sample.** The log line is `park.ckpt … secs=`." G1 note §Test evidence: the synthetic T9 (address-identity oracle; `test_t9_decode_step_time_unchanged_after_checkpoint`, observed ratios 1.0129 … 0.9984, all within 2%; leaked-reference control 2.9–3.3×) was chosen *instead of* the optional studio2 microbench. | **live-validate** | One before/after decode tok/s sample on studio hardware closes the donation-trap gate; the synthetic address oracle is a stronger correctness check but does not measure end-to-end Metal timing. **Schedule after the MoE spike's studio2 bench window — do not run benches now.** |
| **i** | G6/G7 deliberately not built. | G2G5 note §Deliberately NOT done: "**G6** rewind-to-LCP — audit A4 shows divergence is at **token 0**, where rewind is useless; deprioritised per the audit. **G7** incremental append-only checkpoints." G1 note §Deliberately NOT done: G6 deprioritised by A4; G7 "each checkpoint rewrites the full generation, roughly 1–3 GiB per qualifying turn at ≥2048-token deltas." Plan §G6 (only if A4 shows divergence) / §G7 (optional, later; cuts SSD wear). | **accept-monitor** | G6 is deprioritised by measured data (A4: divergence at token 0); G7 is purely optional (SSD-wear mitigation). Neither is required for the live acceptance. Accept; monitor SSD wear / checkpoint latency and revisit G7 only if wear becomes a real cost. |
| **j** | any GRACE/launcher changes (design-only; would ship on the production branch at deploy time). | G3 note OQ-2: "Launcher / GRACE change is **DESIGN-ONLY; deferred to deploy time.**" G3 needs no launcher change to be correct at the default budget (the park phase is bounded to ONE unit past the deadline, inside `GRACE − MARGIN`). It is only worth raising `GRACE` (more park budget for a large dirty set) or wiring `EXO_SHUTDOWN_GRACE_SECONDS` into `start_cluster.sh`'s env allowlist. "Any such edit **MUST land on the production branch at deploy time** (per A8: `origin/main`'s `start_cluster.sh` lacks `DSV41_DENSE`; the production branch carries it). Nothing was changed here." | **defer-to-deploy** | Explicitly design-only and deploy-time: no launcher/production file is (or should be) touched in this offline chain. Whether to raise GRACE / wire the env allowlist is a deploy-time decision that ships on the production branch, informed by the A9 park-cost measurement (row g). |

### Disposition summary

- **fix-now:** none.
- **defer-to-deploy:** b, j.
- **live-validate:** a, c, e, g, h.
- **accept-monitor:** d, f, i.

---

## UNVERIFIED / flagged items

- **G2G5 `tests/test_dsv41_park.py` "(+23/−8)"** — actual `git show --numstat
  46990ec73` = **+20/−3**. Buildnote line-count **UNVERIFIED exact**; intent matches.
- **G4 `park.py` "(+406)"** — actual `git show --numstat 360fadbf6` = **+399/−7**.
  Buildnote line-count **UNVERIFIED exact**; net ≈ +392.
- **All test-run numbers** (178 / 251 / 290 / 318 / 322 / 346 and per-file case
  counts) are **quotes from the buildnotes' §Test evidence**, not re-run here
  (running the suite is out of scope for this doc-only review and would require the
  `mlx-lm` submodule + the shared `.venv`). Counts that *could* be checked statically
  (file presence, `def test_` counts, allowlist sizes) **do corroborate** the notes.
- **"T9 on Metal" and A9/A10** are live-only by their own notes and are **not**
  verified by this review (rows g, h).
