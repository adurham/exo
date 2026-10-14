# DSv4.1 indexer — PRE-REGISTERED verification spec for the ONE relaunch A/B (2026-10-14)

**Status: PRE-REGISTERED, not executed.** Nothing here has run against the cluster. This
document fixes the arms, the probes, the pass criteria and the kill-switch matrix **before**
the launch, so that a favourable-looking result cannot be reinterpreted after the fact — the
failure mode this guards against is a greedy-decode A/B where "no loop in N tokens" is read
as "fixed".

Companion: `docs/dsv41-indexer-refdiv-audit-2026-10-14.md` (findings, harness, local
measurements). Harness: mlx-lm fork branch `audit/dsv41-indexer-refdiv-2026-10-14` @ `3255ba4`.

---

## 0. What the A/B is for, and what it is NOT for

The audit reduced the whole mlx-lm serving-math surface between the clean pin and today, at
`DSV41_INDEXER_QSPLIT=0`, to the small-n wide tile (`a0573e3`). The A/B therefore has two
independent questions and they must not be confounded:

* **Q1 (attribution):** does the wide tile change *any* selection on the box? Locally it
  changed none (0/97 cells) and a row-level ulp difference appears only in a one-column tail
  tile at a column the top-k never takes. On the box, with real weights and real tie
  densities, that is unmeasured — this is what the A/B measures.
* **Q2 (behaviour):** does the live degeneration regression survive with the wide tile OFF?
  If yes, the wide tile is not the mechanism and the arm is spent early — that is a *good*
  outcome (it saves a second relaunch).

**The A/B cannot prove the wide tile is harmless** (the engagement set is narrow and the
loop band may be outside it), and it cannot prove the qsplit path is correct. It only
discriminates the two arms on the stated probes.

## 1. Preconditions — do not launch without these

1. **Instrument the decision the A/B turns on.** `EXO_SPECULATIVE_GAMMA` sets the verify row
   count `n` (`batch_generate.py:889`, default 2 for DSv4 MTP; the wide-tile buildnote
   measured γ3 ⇒ n=4), and `n` decides whether the wide rule can engage at all (§2 of the
   audit). Requirements, all satisfied *before* the launch turn:
   * `DSV41_INDEXER_SMALLN_WIDE_TILE` **must appear in the per-rank startup env report**
     (`[DSV41] loop-guard env (rank r/w, digest=hex)` currently covers the loop-guard knobs
     only). `indexer.py:128` reads the flag and never reports it — unlike the qsplit flag,
     which prints at `indexer.py:185-187`. Patch (same shape as the existing print):
     `print(f"[DSV41] indexer wide tile: DSV41_INDEXER_SMALLN_WIDE_TILE={int(_SMALLN_WIDE_TILE)}
     (effective at import)", file=sys.stderr, flush=True)`.
   * **A per-shape indexer geometry line, once per distinct shape** (the module already has
     the `_GEOMETRY_LOGGED` precedent at `indexer.py:318-345` for the hier geometry):
     `[DSV41] indexer geometry: n=<n> nb=<nb> ratio=<r> tile=<w> route=<tiled|untiled>
     wide=<0|1> role=<source|consumer|plain>`. Without it, "the wide tile did not engage in
     this run" stays a derivation instead of an observation. **If the launch cannot produce
     this line, Q1 is unanswerable and the A/B should not be run for Q1.**
   * Record `EXO_SPECULATIVE_GAMMA` per rank, and hold it FIXED across every arm.
2. **Provenance.** Both nodes: `mlx_lm/models/deepseek_v41/indexer.py` md5 against the signed
   tree; runner log's mlx-lm sha; the two stderr lines above present on both ranks. A
   mismatch voids the arm.
3. **Frozen env across arms** (changing any of these changes the width *and* the
   engagement threshold together): `DSV41_INDEXER_TILE`, `DSV41_INDEXER_TILE_MB`,
   `DSV41_INDEXER_TILE_MIN_NB`, `DSV41_INDEXER_TILE_FORCE`, `DSV41_INDEXER_L2_FULL`,
   `DSV41_INDEXER_SMALLN_ROW_BF16`, `DSV41_INDEXER_ROW_BF16`, `DSV41_INDEXER_HIER`,
   `DSV41_INDEXER_HIER_*`, `DSV41_INDEXER_CONSUMER_SKIP`, `DSV41_INDEXER_COMPILE`,
   `DSV41_INDEXER_TILED_IMPL` (must stay `buffer` — see the audit's `merge` finding),
   `DSV41_SPARSE_FENCE_MIN_ROWS`, `DSV41_SPARSE_COLSPLIT`, `DSV41_SYNC_COLLECTIVES`,
   `DSV41_SYNC_WARM_CALLS`, `EXO_SPECULATIVE_GAMMA`. Only the two cells below vary.
4. **Cold, greedy, deterministic probes.** `use_prefix_cache=false`, temperature 0, one
   request at a time on an otherwise idle cluster; identical prompt bytes; identical node
   pair; the studio's recorded loop-token dumps
   (`~/.exo/exo_log/dsv41-degeneration-guard-token-20261014-1351*.json` and the event-1
   dump) as the incident reference.

## 2. Arm matrix (one relaunch per arm; env flip only, no code change between arms)

| arm | `DSV41_INDEXER_QSPLIT` | `DSV41_INDEXER_SMALLN_WIDE_TILE` | question answered | order |
|---|---|---|---|---|
| **A0 — known-good** | `0` | `0` | reproduces the pre-`a0573e3` width rule byte-for-byte; the reference for every comparison | 1st |
| **A1 — today's live pin** | `0` | unset (ON) | the incident's config; supplies the replay positive control | 2nd |
| **A2 — the real A/B** | `1` | `0` | qsplit alone (wide tile held OFF so the two changes are not confounded) | 3rd |
| **A3 — both** | `1` | unset (ON) | only if A2 passes; do not run otherwise | 4th |

`A0` is the arm the audit predicts to be *behaviourally identical* to A1 on every probe below
except the one-column-tail shapes (§3.3); if A0 ≠ A1 on a probe whose indexer geometry shows
no engagement, **that falsifies the §1 reduction of the audit** — stop, escalate, do not
proceed to A2. That is the single most informative possible result of this A/B and it is
worth the relaunch on its own.

## 3. Probes and pass criteria

### 3.1 Provenance gate (per arm; a failure voids the arm)
Both ranks: indexer md5 == signed tree; mlx-lm sha == the intended pin; the wide-tile flag
line and the qsplit flag line present with the intended values; both runners `RunnerReady`;
no `ERROR`/`CRITICAL`/`SIGKILL`/OOM in either log. Record `guard_env_digest` from both ranks
and require equality (the existing cross-rank equality check).

### 3.2 Clean-window byte-compare (the strongest criterion available)
Replay the clean baseline's captured request set (pin `689e4ea`, 144 calls, contexts
110-241K, zero degeneration), cold, greedy, in the order captured.
* **PASS-A0:** every response byte-identical to the stored clean capture.
* **PASS-A1:** every response byte-identical to the stored clean capture **as well**.
  Predicted by the audit for all of these calls: at γ≤3 (n≤4) the wide rule cannot engage
  below 254.2K context, and every clean-window call is 110-241K.
* **Interpretation.**
  * A0 and A1 both byte-clean ⇒ the reduction holds on the box and the clean era never
    exercised the rule. Continue to 3.3/3.4 for the qsplit question.
  * A1 differs on ≥1 call ⇒ read that call's indexer geometry line. If the geometry shows
    no engagement (`wide=0` or `route=untiled`), the divergence is **outside the wide tile**
    — i.e. either the reduction in the audit is wrong or the divergence is not serving-math
    at all. Escalate as a blocker with the geometry line attached; do **not** attribute it to
    the wide tile.
  * A0 (not A1) differs ⇒ the tree is not reproducing the clean baseline even with the rule
    off ⇒ the regression is not the wide tile. Stop the wide-tile arm; hand the evidence to
    the exo-side (guards/limiter/park-restore) workstreams.
* **Report per call:** sha256 of the response, first differing token (if any), the indexer
  geometry lines from that call, and the `[DSV41]` first-token/turn-end lines.

### 3.3 Index-set digests (the kernel-level criterion; needs §1.1 instrumentation)
For every indexer call behind an env gate (e.g. `DSV41_INDEXER_CIDX_DIGEST=1`), print
`sha256(cidx bytes)` per index-source layer per forward, together with the geometry line.
* **Tolerance (pre-registered, not to be relaxed after the fact):**
  1. **Exact digest equality per (layer, forward)** is required for every call whose geometry
     shows `route=untiled` or `wide=0`. A difference there is a **blocker** (the arms run the
     same code on the same inputs).
  2. For calls with `route=tiled` **and** `wide=1` **and** `nb % cap == 0` (no one-column tail;
     the audit's ulp class is excluded by construction), digest equality is **also** required.
  3. For the residual class — `route=tiled`, `wide=1`, `nb % cap == 1` (one-column tail) — a
     digest difference is **tolerable only if** all three hold: the differing rows have a
     k-th boundary gap ≤ 1e-6; the differing columns are the row's last column; and the
     number of differing rows per call is ≤ 0.1 % of the call's rows. Anything else is a
     blocker. (Locally this class produced **no** digest difference at all — 12 cells with a
     row-level ulp difference, 0 with a moved index set — so in practice criterion 3 should
     be satisfied by equality.)
* **Report:** per call, the geometry line, the digest, and for any difference: rows differing,
  mean per-row Jaccard overlap, the k-th gaps of the differing rows, and the differing column
  indices. Local reference values for the same-shape synthetic case: differing rows 0; the
  ulp class shows 1 element, |Δ| ≤ 4.5e-08, always at column `nb-1`, rank 12,856…793,180.

### 3.4 Incident replay (needs the deterministic payload reconstruction)
Two probes, per incident payload (event 1 @ ~56K on `6a5c78f` with QSPLIT=1; event 2 @ ~78K on
`a0573e3` with QSPLIT=0):
* **Positive control first, on A1** (today's config, the config that looped): replay the
  reconstructed payload and require the ON arm to reproduce the recorded loop **token for
  token** against `~/.exo/exo_log/dsv41-degeneration-guard-token-*.json`. If A1 does not
  reproduce the loop, the reconstruction is not faithful — **the replay is inconclusive and
  no conclusion may be drawn from the other arms**, whatever they do. (This is the step that
  keeps a "no loop here" observation from being read as a fix.)
* **Then, ON the same payloads, on A0:** compare byte-for-byte. Pre-registered outcomes:
  * A0 **also loops** with the same tokens ⇒ the wide tile is not the mechanism for that
    incident. Q2 answered negatively; the qsplit arm (A2) is still valid for Q1.
  * A0 loops with **different** tokens ⇒ a real, attributable behavioural difference between
    the arms; escalate with both token streams and both geometry logs. Do **not** call it a
    fix without a second, independent payload reproducing the same direction.
  * A0 does **not** loop ⇒ a difference; escalate, and require the same direction on the
    second incident payload before any claim of "the wide tile causes the loop".
* **Never** conclude from a single payload, and never from a truncated run: fix the max-token
  budget and the stop reason before the launch and apply it to every arm.

### 3.5 Hygiene / performance sanity (per arm)
Both-node logs free of `ERROR`/`CRITICAL`/`SIGKILL`/OOM; GPU locks released between arms; the
`[DSV41] obs first-token` / turn-end lines present. Optional but cheap: the idxprof6-style
round-time check (indexer `n=4`-class rounds at the same context) to confirm A1 is still the
fast arm — if A1 is *not* faster than A0 at ≥254.2K, the wide rule is not engaging and the
arms are not distinguishable, which voids 3.3/3.4 on that box.

## 4. Kill switches and rollback (all env-only, no code change)

| symptom | action |
|---|---|
| any probe violates 3.1 | relaunch the same arm; if it repeats, the arm is void |
| 3.2 PASS-A1 fails with an engaged geometry | stop; escalate (reduction falsified) |
| 3.2 PASS-A0 fails | stop the wide-tile arm; hand to the exo-side workstreams |
| 3.3 criterion 1 or 2 fails | relaunch with `DSV41_INDEXER_SMALLN_WIDE_TILE=0` immediately (A0) and escalate with digests |
| 3.4 A0 diverges | A0 is the safe arm by construction: hold A0, escalate with both streams |
| A2/A3 regress (qsplit) | relaunch with `DSV41_INDEXER_QSPLIT=0`; the wide-tile arms are unaffected |
| anything ambiguous | **A0 is the known-good configuration.** Every arm is one env flip from A0; no node needs a code change or a re-pin. |

Note both flags are read at **import**: a flip requires a relaunch, not a config reload.

## 5. What this A/B will still NOT prove

1. That the wide tile is harmless in general — only that it moved no selection/digest and no
   token on the probes actually run. Offline the same held across 97 cells; the engagement
   region is narrow by construction and the tie structure under real weights is unmeasured.
2. Anything about shapes the probes never exercise: prefill (`n > 16`, the hierarchical
   path), bf16 small-n rows (`DSV41_INDEXER_L2_FULL=0` or `SMALLN_ROW_BF16=1` — this audit's
   neutrality evidence covers fp32 rows only), `nb` beyond the probed contexts, bsz > 1.
3. The cross-rank collective itself (JACCL `all_sum`, the int32 reducer) — the local harness
   stubs it.
4. That a clean replay of the incident payload is a *representative* sample: two payloads is
   two samples. The positive control in 3.4 makes each sample trustworthy individually; it
   does not make the set sufficient for a general claim.
