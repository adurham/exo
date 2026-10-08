# Phase-20 0c delta ladder - summary

records: 8 from 1 JSONL file(s)

## Table A - ctx ladder (2048-row delta) + fresh reference
| ctx | delta_rows | actual_rows(med) | n | rows/s med | min | max |
|---|---|---|---|---|---|---|
| 20k | 2048 | 3866 | 3 | 272.599 | 272.254 | 273.199 |
| 50k | 2048 | 2501 | 3 | 249.624 | 249.153 | 251.54 |
| 110k | 2048 | None | 0 | None | None | None |
| fresh100k | None | None | 0 | None | None | None |

## Table B - delta-size sweep at 50K
| delta_label | actual_rows(med) | n | rows/s med | min | max | vs_4096 |
|---|---|---|---|---|---|---|
| 256 | None | 0 | None | None | None | None |
| 1024 | None | 0 | None | None | None | None |
| 4096 | None | 0 | None | None | None | None |
| 8192 | None | 0 | None | None | None | None |

## Decision-rule verdicts
- ctx-depth: INSUFFICIENT DATA (need 20k + 110k at 2048)
- fixed-overhead: INSUFFICIENT DATA (need 50k d256 + d4096)
- flat-vs-slope: INSUFFICIENT DATA (need >=2 sweep sizes)
- fresh reference: NOT RUN (no fresh100k record)
