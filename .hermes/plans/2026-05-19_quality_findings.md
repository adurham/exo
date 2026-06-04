# Quality Findings — TOPK 160 vs 512 at 100K context

**Date:** 2026-05-19 ~12:35 CDT
**Plan:** `.hermes/plans/2026-05-19_to_35tps.md` Phase A/C
**Probe:** `bench/quality_probe_dsv4.py` (needle-in-haystack at 100K)

## Result: TOPK=160 is BROKEN at 100K. TOPK=512 is correct.

| TOPK | needle_found | response                       | t/s (3-iter)    |
|------|--------------|--------------------------------|-----------------|
| 160  | **False**    | `<|begin_of_sentence|>`        | 31.9-32.2       |
| 512  | **True**     | `FALCON-MERCURY-7749` (exact)  | 30.06 (10-iter) |

Two runs at TOPK=160 both produced just the BOS token. The model is
completely failing to recall the needle and not even producing coherent
text. The "+2 t/s" speed gain from TOPK=160 is moot — it's measuring
throughput on a model that's not doing the task.

## Implication for 35 t/s goal

- The 30.06 t/s baseline at TOPK=512 is the **quality-correct floor**.
- The May-17 31.5 t/s "champion" was at TOPK=160 — measured on broken
  output. Not a valid benchmark.
- The May-18 02:02 32.29 t/s "champion" was also at TOPK=160 (TOPK
  default change to 512 didn't land until 59df6258 later that day).
  Almost certainly also broken-quality.
- Any historical-bisect work was chasing a broken-quality number.

## What gets us to 35 t/s with valid quality

Levers in order of risk/EV:

1. **Run quality probe at intermediate TOPK values.** If TOPK=256
   or TOPK=384 passes the needle test AND gets us into the 31-32
   t/s range, that's a real win. The May-18 plan said "192 was
   validated +29% prefill at 100K" — there may be a quality-valid
   TOPK below 512.

2. **Skip Phase A bisect + Phase B ack_sync_pre redeploy.** Both
   chased the May-17/May-18 02:02 numbers which we now know were on
   broken quality.

3. **Architectural / mlx-side changes** — multi-day work, out of
   scope for tonight.

4. **Concurrency scaling** — sidesteps the per-stream goal but
   delivers aggregate throughput.

## Next experiment: TOPK quality sweep

Run quality probe at TOPK in {256, 384, 448}. For each that passes
needle test, run 3-iter bench to measure speed. The smallest TOPK that
still passes quality is the best operating point.

## Cluster state

Restored to TOPK=512 FENCE=43 (production baseline). Inference probe
passes. mlx-lm at 6dcdd40a (post mc_ping removal, ~no effect).
