# Decode vs context depth, full ladder to 565K: flat to ~120K, then a memory wall (2026-09-22)

Supersedes the "flat-to-rising across depth" claim in
`docs/decode-flat-across-depth-36tps-at-115k-2026-09-22.md`, which was based
on four points and is falsified by the two deeper rungs below.

## The ladder

`bench/long_decode_probe.py <depth> --max-tokens 600`, production config
(vec + ROWSDPA=3 + VERIFY_BATCH=1 + steel-BI + ATTN_ALLSUM=0),
DeepSeek-V4-Flash-Vision-Exp, needle verified at every rung:

| depth | decode tok/s | prefill tok/s | peak mem | % of wired limit | needle |
|---|---|---|---|---|---|
| 2,216 | 29.17 | 213.5 | 96.78 GB | 80.3% | ✓ |
| 21,855 | 33.22 | 397.6 | 98.15 GB | 81.4% | ✓ |
| 68,002 | 31.78 | 422.3 | 98.75 GB | 81.9% | ✓ |
| 115,614 | **36.74** | 423.3 | 101.12 GB | 83.9% | ✓ |
| 273,622 | 26.83 | 409.0 | 103.75 GB | 86.0% | ✓ |
| 564,926 | **13.36** | 371.8 | **115.66 GB** | **95.9%** | ✓ |

(`iogpu.wired_limit_mb = 115000` → 120.6 GB.)

## What this actually shows

1. **Throughput is NOT monotonic in depth.** The 115K reading (36.74) is a
   high outlier; 273K is lower than every shallow rung. Any claim that decode
   "improves with depth" is unsupported.

2. **Flat band to ~120K.** Roughly 29-37 t/s across 2K..120K. The spread
   inside that band is dominated by CONTENT, not depth: measured at a fixed
   ~10.5K depth, identical config, throughput ranged 30.5 (prose) to 45.9
   (list-reverse) t/s — a swing larger than any depth effect in this range.
   **A depth ladder whose prompts differ in content cannot separate the two.**

3. **Real decline past ~120K:** −27% by 273K, −64% by 565K.

4. **The deep decline correlates with GPU wired-memory pressure.** Peak
   memory rises monotonically 96.8 → 115.7 GB while throughput stays flat and
   then collapses. At 565K the run sits at 95.9% of the 120.6 GB wired limit.
   The mechanism is most likely KV-cache growth meeting the cap (eviction /
   reallocation under pressure) rather than a compute limit — consistent with
   the existing `dspark-352k-*` memory-regression family of investigations.
   NOT yet proven to be causal here; the correlation is the observation, the
   mechanism is the hypothesis.

5. **Quality is intact at every rung**, including 565K (needle recalled,
   finish_reason=length, coherent reasoning text).

## Practical summary

- **≤120K context: 29-37 t/s.** The 35-40 t/s target holds here.
- **~275K: ~27 t/s.**
- **~565K: ~13 t/s** and pressing the memory ceiling.

## Methodological guards (this session's recurring failure mode)

- **Never compute tok/s from wall clock.** `decode_s` / `decode_tps` are
  reported separately for this reason. My own first attempt charged prefill
  to decode and invented a fake 20%-per-doubling cliff.
- **Do not extrapolate a curve from ≤4 points**, especially not the last
  one. I reported "improves with depth" off a 4-point ladder; two more
  points inverted it.
- **Control content when laddering depth.** Same prompt template is not
  enough if the generated output differs in style; acceptance (and therefore
  tok/s) swings 1.5x on content alone.
- **Report every completed run.** The 565K result finished and was not
  reported for a full turn — the loop has to be closed on background work.
