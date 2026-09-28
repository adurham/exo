# Stage-1: the expert-speed factor A would need (the post-gate pivot's f* question)
2026-09-28, gateway. Measured inputs; the non-expert anchor is DERIVED (flagged).

Question: how much slower than production MXFP4 can the EXL3 experts be while the
resident (A) design still hits the 25 tok/s bar?

Measured inputs:
  p20 mxfp4 expert probe at V4.1 shapes (E=384, same as body layers): R=1 0.388,
    R=4 1.051, R=6 1.483 ms/layer  -> T_E40 = 15.52 / 42.04 / 59.32 ms
  adopted EXL3/prod ratios (day-1): R=1 1.909, R=4 1.979, R=6 2.048
    -> EXL3 MoE cost = 29.6 / 83.2 / 121.5 ms per 40L
  idealized-decode bounds (p19 cheap arm): R=1 1.774, R=4 1.852, R=6 none better (2.048)

Budgets per cycle at 25 tok/s (= 40 ms/token):
  plain R=1:                                  40.0 ms
  MTP gamma=3 (R=4, 2.15 tok/cycle):          86.0 ms
  MTP gamma=5 (R=6, 3.5  tok/cycle):         140.0 ms

Non-expert anchor (T_N, DERIVED):
  production Vision-Exp cycle: 972.1 tok / 44 samples = 22.09 tok/s;
    26,051 tok / 12,669 cycles = 2.056 tok/cycle -> 93.1 ms/cycle
  minus estimated MoE share -> non-expert ~1.33 ms/layer ~= 53 ms/40L;
    V4.1-scaled (hidden 5120 vs 4096, x1.25) ~= 1.66 ms/layer ~= 66 ms/40L;
    upper reach ~83 ms.   [DERIVED - no direct per-component measurement exists]
  port reference-grade: 5.1-5.8 ms/layer (p40, session record) -> 204-232 ms/40L
    [EXCLUDED: fp32 debug cache, unoptimized attention; not a production path]

T_N allowed at the 25 tok/s budget (T_N <= budget - EXL3 MoE):
  plain:  40.0  - 29.6  = 10.4 ms  (0.26 ms/layer)  -- no real attention path fits
  g3:     86.0  - 83.2  =  2.8 ms  (0.07 ms/layer)  -- infeasible
  g5:    140.0  - 121.5 = 18.5 ms  (0.46 ms/layer)  -- only shape with headroom
  ... and even granting the idealized bounds: g3 with 1.852 -> T_N <= 8.1 ms;
      g5 has no better bound than the current 2.048.

Margin: derived T_N (53-83 ms) is 3-4.5x over even the 18.5 ms allowance (g5).

Expected landing (current kernels):
  g5: 3.5 tok/cycle / (T_N40 + 121.5 ms):
      T_N40 = 53 -> 20.1 tok/s | 66 -> 18.7 | 80 -> 17.4 | 90 -> 16.5
  g3: 2.15 / (T_N40 + 83.2):   53 -> 15.8 | 66 -> 14.4 | 80 -> 13.2
  => mid-teens band, central ~15-19 tok/s (g5 best), before engine/comm overheads.
  This agrees with the independent port-kernel projection (16-19 tok/s).

Conclusion: the 25 tok/s bar is NOT reachable with EXL3 experts on this GPU.
Getting there needs either a fundamentally different kernel (2-4x beyond anything
measured or bounded) or a non-expert path 3-8x leaner than production's. The
actionable levers are software: a lean production-grade non-expert path for V4.1
(does not exist yet) and the kernel floor. The build remains the only true
measurement.
