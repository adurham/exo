# P2 results: c=2 validation under the promoted config — 2026-08-28/29

**Pre-registration + amendments:** `dspark-p1p4-campaign-preregister-2026-08-28.md`
(P2 section + Amendments P2-1/P2-2). Stack: exo `75d2402dd` + mlx-lm `d098642`.
Artifacts: `/tmp/ab/p1p4/run_c2*.json`, `run_bug3_*.json`; node logs
`~/exo_verbon3.log` (L0 spec-ON) and `~/exo_specoff_stripped.log` (L2 control,
env verified `ps eww`: `EXO_SPECULATIVE=0 EXO_DSV4_MTP=0` both nodes 22:14:53).

## Verdict summary

**c=2 is NOT production-ready — but the blocking bugs are NOT in the speculation
path.** Two shared-generator bugs found, both reproducing with spec fully OFF;
the spec-specific legs (deep batched B=2 verify, Bug-3 residual, determinism,
contamination) all came back clean or better than the PP-era baseline.

| Leg | Result | Verdict |
|-----|--------|---------|
| Short c=2 (system+user, rowseq path) | **DEGENERATION 3/3 repeats** — `.</think>Paris` period-3 cycle at token 61 (kill-switch `action=error`) | **FAIL — but NOT spec:** identical degen 2/2 on the spec-OFF control (22:16:23, 22:17:13) |
| Degen abort path at BS=2 | **Runner CRASH 3/3** — `ValueError: [reshape] Cannot reshape array of size 2 into shape (1,1,1,1)`, mlx-lm `cache.py:2050 fetch_overlap_carry`; kills BOTH streams, instance deleted, JIT reload | **FAIL — NOT spec:** same crash on spec-OFF control (8 occurrences). Availability bug: one bad stream nukes its neighbor |
| Deep c=2 batched B=2 (100K+100K, distinct code prompts) | Both streams deterministic across repeats (A `644065cd` ×2, B `32c7d9c6` ×2); own code returned, zero cross-stream leakage; zero degen/faults | **PASS** |
| Bug-3 adversarial final-digit (6 variants, batched B=2 @100K) | **flips 0/6** — every activation code emitted exactly (`8473921577{0,3,4,5,7,9}`) | **PASS** — PP-era ~80% flip class not reproduced under TP batched verify |
| c=2 spec-ON throughput @100K | Per-stream A 10.01–10.07, B 9.65–9.71 tok/s (W=831); aggregate 19.66–19.78; fully deterministic both streams | measured; see finding below |
| c=2 spec-OFF throughput control | **NOT RUN** (session wrap-up cut L2 short after the c2off short-prompt control) | outstanding |

## Finding 1 (NEW BUG, shared generator): c=2 system+user short-prompt degeneration

Two concurrent /v1 system+user short prompts (Tier-1 `sys_capital_france` +
`sys_count_to_five`) deterministically degenerate: the cap stream emits its
reasoning then loops `.</think>Paris` (period-3 token cycle `[16, 128822, 51119]`)
until the kill-switch errors the request at token 61; the second stream shows
unstopped repetition too (counts past five, re-answers repeatedly). c=1 on the
identical launch is byte-clean (Tier-1 7/7). **Spec-OFF control reproduces it
exactly** → root cause is in the shared c=2 batched-generator EOS/stop handling
(the `</think>`-boundary stop logic appears not to fire at BS=2), NOT in
DSpark/MTP/verify. Echoes the June-2026 "c=2 MTP degeneration" class but is now
shown to be mechanism-independent.

## Finding 2 (NEW BUG, mlx-lm): BS=2 degen-abort reshape crash

When the kill-switch errors one stream of a BS=2 batch, the cleanup path crashes
the whole runner: `[reshape] Cannot reshape array of size 2 into shape (1,1,1,1)`
at `fetch_overlap_carry` (mlx-lm `cache.py:2050`) — recursion through the batch
cache extract on a size-2 (two-stream) array being forced into a BS=1 shape.
Both streams die, the instance is torn down (5-retry exhaust → deletion) and the
next request eats the JIT reload (~5 min). Reproduces spec-OFF (8 hits in the
control log). Second-order effect observed: one post-crash relaunch window
produced a `ConnectToGroup outside of state machine` crash before recovery.

## Finding 3: deep c=2 spec-ON per-stream cost

Aggregate c=2 @100K spec-ON ≈ 19.7 tok/s vs c=1 spec-ON 37.7 — per-stream 10.0/9.7.
B=2 spec cycles run rowseq-style per-row verify (BS>1 batched-verify is the
Phase-5 TODO in `dsv4_mtp.py`), so the per-cycle cost roughly doubles while
acceptance stays per-stream. NOT a regression (c=2 spec was never promoted);
recorded as the baseline for any future BS>1 batched-verify work. The
spec-OFF c=2 control (L2) was cut by session wrap-up, so no spec-ON-vs-OFF c=2
delta is claimed.

## Production guidance (unchanged config, new operational caveat)

- c=1 serving: fully validated, no change.
- c=2 serving: **do not enable for system+user short-prompt traffic** until
  Findings 1–2 are fixed; deep single-shot c=2 pairs ran clean, but any stream
  hitting the degen kill-switch takes down its batchmate (Finding 2).
- Both bugs are in the shared path → fixes benefit spec-ON and spec-OFF equally;
  neither blocks the promoted c=1 spec config.
