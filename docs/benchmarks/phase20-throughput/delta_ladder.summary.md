# Phase-20 0c delta ladder - summary

records: 27 from 5 JSONL file(s)

## Table A - ctx ladder (2048-row delta) + fresh reference
| ctx | delta_rows | actual_rows(med) | n | rows/s med | min | max |
|---|---|---|---|---|---|---|
| 20k | 2048 | 3866 | 3 | 272.599 | 272.254 | 273.199 |
| 50k | 2048 | 2501 | 3 | 249.624 | 249.153 | 251.54 |
| 110k | 2048 | 2028.5 | 2 | 242.455 | 242.366 | 242.544 |
| fresh100k | None | 91276.0 | 1 | 272.179 | 272.179 | 272.179 |

## Table B - delta-size sweep at 50K
| delta_label | actual_rows(med) | n | rows/s med | min | max | vs_4096 |
|---|---|---|---|---|---|---|
| 256 | 913 | 3 | 227.295 | 225.888 | 227.397 | 0.88 |
| 1024 | 1614 | 3 | 255.105 | 254.293 | 256.123 | 0.988 |
| 4096 | 4415 | 3 | 258.272 | 257.765 | 258.532 | 1.0 |
| 8192 | 8155 | 3 | 267.49 | 265.376 | 268.504 | 1.036 |

## Decision-rule verdicts
- ctx-depth benign: 11.1% 20K->110K (<=15%)
- no fixed-overhead cliff: d256=227.3 vs d4096=258.3 rows/s (88% of d4096)
- FLAT across delta sizes 1024-8192: spread 4.8% <10% => Phase 4 skipped, slope recorded
- fresh reference OK: 272.2 rows/s >= 255

## Collapsed reps (excluded)
- ctx110k_d2048 rep0: reuse=None log_prefill=None (collapsed: no turn-reuse line)
