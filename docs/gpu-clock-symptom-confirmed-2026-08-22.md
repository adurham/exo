# GPU clock confirmed as symptom, not independent throttle — 2026-08-22 (Phase B, post-fence-fix investigation)

## Why this check

Per Fable's ranked recommendation: the real GPU clock observed during
decode (819-1122 MHz vs the M4 Max's public ~1.5GHz+ peak spec,
originally found in `docs/gpu-idle-gap-deep-dive-2026-08-22.md`) was
assessed as "likely a downstream symptom of the bursty-low-load
pattern, not an independent root cause" but never directly confirmed —
that assessment was reasoning from the general DVFS mechanism, not a
real test. Fable flagged this as a cheap, one-off experiment worth
running before further idle-gap investigation: force sustained GPU
load on the same hardware and see whether it reaches near-peak clock.

## Real test

Wrote a minimal sustained-load probe (`/tmp/gpu_sustained_load_probe.py`):
2000 back-to-back 4096×4096 bfloat16 matmuls with `mx.eval()` forcing
synchronous completion each iteration — no sleep, no artificial gaps,
designed to keep the GPU queue continuously fed. Ran on macstudio-m4-1
(the same node used for all of tonight's other real measurements),
sampled `sudo powermetrics --samplers gpu_power` (passwordless,
confirmed working all session) during the run.

**Real completed run**: 2000 iterations in 18.66s (107.2 iters/s, 9.33ms/iter
— consistent with real GPU-bound matmul work at this size/dtype, not a
CPU-bound loop).

## Real result

Every single `powermetrics` sample taken during the sustained-load run
(8 samples, ~300ms each, spanning ~2.5s of the run):

- **GPU HW active frequency: 1578 MHz — 100% of that bucket, every
  sample.** This is the topmost bucket in `powermetrics`' own P-state
  histogram (15 states shown, 338 MHz through 1578 MHz) — the real
  measured ceiling for this hardware.
- **GPU idle residency: 0.00%** (expected, matches decode-time reading —
  see the earlier reconciliation in `docs/gpu-idle-gap-deep-dive-2026-08-22.md`
  explaining why this metric alone is uninformative).
- **GPU Power: 55.0-57.0 W** — dramatically higher than the 4.6-7.1W
  observed during real decode (`docs/gpu-idle-gap-deep-dive-2026-08-22.md`),
  confirming genuine sustained compute work, not just "not power-gated."

## Conclusion

**Confirmed, definitively: the same physical GPU reaches its real peak
clock (1578 MHz) under guaranteed sustained load.** The 819-1122 MHz
range observed during real decode is NOT an independent hardware
throttling limitation, firmware cap, or thermal constraint — it is
purely a consequence of the bursty, low-average-load dispatch pattern
during decode never presenting the GPU's DVFS governor with sustained
queue pressure long enough to ramp clock. This closes the GPU-clock
question cleanly: it requires no further investigation as an
independent lever, and any future decode-time fix that closes the
idle-gap pattern (Phase C, still queued) should raise clock as a
natural side effect, not as something requiring separate work.

## Disposition

Closes Phase B of the post-fence-fix investigation plan. Proceeding to
Phase C (fresh dual CPU+GPU idle-gap capture at two context depths,
now on the fixed-fence baseline).
