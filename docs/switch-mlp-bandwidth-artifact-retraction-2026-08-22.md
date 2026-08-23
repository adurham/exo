# P0: decode switch_mlp "27.7% of peak BW" — RETRACTED as a measurement artifact; kernel is actually at 74-87% of peak — 2026-08-22 (session 5)

## Verdict

**(c) Genuine closure with evidence — no kernel gap exists to fix.** The
T3 finding (`switch-mlp-kernel-bandwidth-efficiency-2026-08-22.md`,
"27.7% of peak, confirmed via pipelined microbench, 3.6x headroom") is
**wrong** — not because the arithmetic was wrong, but because T3's
"pipelined" bench was not actually pipelined: it called `mx.eval(y)`
**inside the per-iteration loop**, forcing a host round-trip + graph
build + dispatch per call. That overhead (~172µs/call) was attributed to
the kernel. The kernel itself, measured correctly, is at **74-87% of the
546 GB/s peak spec** — i.e. essentially at the memory-bound floor for
this class of op (real streaming BW on these machines is ~424 GB/s ≈ 78%
of spec; see exo-perf-tuning "Hardware Truths").

## The decisive measurement (m4-1, standalone, 3 independent runs each)

Same weights, same exact `FusedSwitchGLU.__call__` replica, same
production decode shape (B=1, L=1, top_k=6-of-256, mxfp4 g=32, per-rank
inter=1024, 47.19 MB touched/token), rotated routing indices (64-entry
pool) to defeat weight-cache reuse:

| Variant | µs/call | GB/s | % of 546 peak |
|---|---|---|---|
| T3-style loop (`mx.eval` every iter) | 288.7 / 290.6 / 290.0 | ~163 | **29.7-29.9%** ← reproduces T3 exactly |
| Dependency-chained graph, ONE eval per 300 calls | 117.0 / 116.8 / 116.3 | ~404 | **73.8-74.3%** |
| 300 independent calls, one eval (max overlap) | 84.0 | 477 | **87.4%** |

The T3-style number reproduces to within 1µs of the original 290-312µs —
this is the same measurement. The ~172µs delta between serial-sync and
chained is pure host/dispatch overhead of the benchmark loop, not GPU
work. Real production decode builds one lazy graph per token and evals
once, so it sees the ~117µs/call regime, not 289µs.

## Ablation matrix (bench/switch_mlp_decode_ablation.py, run 2 with rotated indices)

| Tier | µs | % peak | What it bounds |
|---|---|---|---|
| A full path, no sort (production: indices.size=6<64 → sort skipped) | 84.0 | 87.4% | baseline |
| A with forced gather_sort/scatter_unsort | 97.5 | 75.4% | sort adds ~13µs — production correctly skips it at B=1 |
| C dense mxfp4 qmm ×6 sequential (no expert gather) | 104.0 | 70.6% | gather path is NOT slower than dense at this shape |
| D dense bf16 matmul ×6 (no quantization) | 320.0 (2× bytes) | 86.4% | bf16 dense hits same efficiency band |
| E affine 8-bit gather variant | 168.9 (2× bytes) | 92.1% | mxfp4-specific dequant tax ≤ ~5-15 pts |
| F batch sweep B=1→16 | — | 87→89% flat | **no B=1 access-pattern penalty** — efficiency does not improve with batch |

(B=32: 80.6%, mild droop from sort + raggedness — irrelevant to decode.)

First-run pitfall worth recording: with a single FIXED routing-index
tensor (as T3 used) and independent parallel calls, the bench measured
**120% of peak** — the 47MB working set of 6 fixed experts fits cache and
the "bandwidth" is fictitious. All numbers above use rotated indices.

## Attribution of the claimed 3.6x shortfall

- ~59% of it (289µs → 117µs): benchmark serial-sync artifact (host eval per iteration). Not present in production.
- ~26% (546 → ~424-477 GB/s): the machine's real achievable streaming bandwidth vs marketing spec — applies to every kernel, not switch_mlp-specific.
- Remainder (~74% vs ~87%): dependency-chain serialization between layers (real but structural — decode's per-layer graph has a data dependency; some tail latency per dispatch is unavoidable).
- mxfp4 dequant, gather/sort machinery, top-6-of-256 sparse access: **all measured individually negligible-to-small** (tiers A/C/E, F-flatness).

## Consequences for downstream claims

- T3's sanity check ("43 × 300µs ≈ 12.9ms ≈ 38.9% of decode wall time,
  consistent with span breakdowns") was double-counting the same
  artifact: span profiling forces per-section syncs (EXO_PROFILER=spans
  inserts ~430 `mx.eval` syncs/token), so spans over-attribute exactly
  like the serial-sync bench does. Real switch_mlp cost is ~43 × 117µs ≈
  **5.0ms/token ≈ 15% of the ~33ms short-context wall time**, not 39%.
- `docs/mtp-dspark-tp-port-decision-gate-2026-08-22.md` used the 27.7%
  figure as an input; its "1.29x if switch_mlp reaches 65% of peak"
  scenario is void — the kernel is already past 65%.
- **No switch_mlp kernel optimization should be pursued.** The lever
  does not exist; headroom at this kernel is ≤ ~15% of a 15% slice.

Raw data: `bench/results/switch_mlp_decode_ablation.json`,
`/tmp/p0_chain.py` + `/tmp/switch_mlp_microbench.py` on m4-1.
