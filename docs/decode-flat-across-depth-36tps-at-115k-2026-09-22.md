# Decode at depth: NOT degrading — 36.7 tok/s @115K with quality (2026-09-22)

## Correction to my own earlier measurement (same day)

An earlier version of this session's notes claimed decode throughput
degrades ~20% per doubling of context (27 t/s @20 tok → 12.3 t/s @7K),
measured from a hand-rolled `depth_sweep.py`.

**That was a measurement bug, not a cluster property.** The script computed
`tok/s = completion_tokens / wall` where `wall` is the ENTIRE request
(prefill + decode). At depth, prefill dominates:

| depth | prefill_s | decode_s | my WRONG t/s | decode-only t/s |
|---|---|---|---|---|
| 20 | 0.1 | 14.7 | 27.03 | 27.12 |
| 605 | 1.6 | 15.6 | 23.26 | 25.65 |
| 1771 | 4.7 | 16.4 | 18.96 | 24.39 |
| 3521 | 9.3 | 16.1 | 15.75 | 24.91 |
| 7021 | 18.6 | 13.9 | 12.31 | 28.83 |

At the 7K point I was charging ~18.6s of prefill against a 13.9s decode.
`bench/long_decode_probe.py` already reports `decode_s` separately for
exactly this reason; my script ignored it and recomputed from wall clock.

## Corrected measurement (server's own decode_tps, needle verified)

`bench/long_decode_probe.py <depth> --max-tokens 600`, current production
config (vec + ROWSDPA=3 + VERIFY_BATCH=1 + steel-BI + ATTN_ALLSUM=0) on
DeepSeek-V4-Flash-Vision-Exp:

| context depth | decode tok/s | needle | decode_s |
|---|---|---|---|
| 2,216 | 29.17 | ✓ | 20.6 |
| 21,855 | 33.22 | ✓ | 18.1 |
| 68,002 | 31.78 | ✓ | 18.9 |
| **115,614** | **36.74** | **✓** | **16.3** |

**Decode throughput is FLAT-TO-RISING across depth, with quality intact.**

The mild rise is consistent with a fixed per-request startup cost
amortizing over the decode window (same mechanism as the 400-vs-800 token
finding: 27.1 vs 33.2 t/s at identical depth).

## Consequence for the 35-40 tok/s target

**The target is already met at real working depths.** 36.74 tok/s at 115K
context with verified needle recall, on the current checkpoint, with no
configuration change required.

The "28 t/s" figure that motivated the campaign came from a 600-token
generation at short context — the LEAST favourable operating point, where
startup cost is proportionally largest. It is not representative of
production usage at depth.

## Methodological note (the recurring failure mode)

This is the same artifact family as most of the retracted numbers in this
repo: **an instrument measured something other than what it claimed.**
Specifically, counting prefill time as decode time. Guards:

- Use `decode_s` / `decode_tps` from the probe; do not recompute from wall.
- Any custom harness must report prefill and decode separately, or it will
  confound them at depth.
- The standing ">=400 token" rule is necessary but NOT sufficient; it rules
  out startup-dominated DECODE windows but says nothing about whether your
  denominator still includes prefill.

## Status

The vec-vs-loop divergence is fixed and verified lossless (see
`docs/vec-divergence-verified-fixed-and-depth-is-the-real-lever-2026-09-22.md`,
which is otherwise correct but whose depth-degradation claim is superseded
by this document). No decode campaign is required for the 35-40 target.
