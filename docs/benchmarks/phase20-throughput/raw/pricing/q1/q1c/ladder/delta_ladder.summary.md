# Phase-20 0c delta ladder - summary

records: 12 from 3 JSONL file(s)

## Table A - ctx ladder (2048-row delta) + fresh reference
| ctx | delta_rows | actual_rows(med) | n | rows/s med | min | max |
|---|---|---|---|---|---|---|
| 20k | 2048 | 3868 | 3 | 269.18 | 266.025 | 270.476 |
| 50k | 2048 | 2503 | 3 | 253.725 | 253.328 | 254.01 |
| 110k | 2048 | 2024 | 3 | 251.522 | 238.393 | 251.747 |
| fresh100k | None | None | 0 | None | None | None |

## Table B - delta-size sweep at 50K
| delta_label | actual_rows(med) | n | rows/s med | min | max | vs_4096 |
|---|---|---|---|---|---|---|
| 256 | None | 0 | None | None | None | None |
| 1024 | None | 0 | None | None | None | None |
| 4096 | None | 0 | None | None | None | None |
| 8192 | None | 0 | None | None | None | None |

## Decision-rule verdicts
- ctx-depth benign: 6.6% 20K->110K (<=15%)
- fixed-overhead: INSUFFICIENT DATA (need 50k d256 + d4096)
- flat-vs-slope: INSUFFICIENT DATA (need >=2 sweep sizes in 1024-8192)
- fresh reference: NOT RUN (no fresh100k record)
