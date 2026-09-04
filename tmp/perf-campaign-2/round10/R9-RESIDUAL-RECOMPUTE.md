# R9-Only Residual Recompute (Round 10, zero cluster cost)

Recomputes the round-10 governing statistic (the RESIDUAL, PRE-REGISTRATION.md 
section 1) from round 9's 60 raw rep JSONs. This is the 'R9-only recompute, 
4 boots' sub-analysis required by PRE-REGISTRATION section 4.1. NO outlier 
exclusion at any stage.

```
residual_ms = prefill_s*1000 - ((prompt_tokens - 1) / prompt_tps) * 1000
```

## Acceptance oracle check (vs R9 REPORT.md section 2.2)

**ALL 8 published residual medians MATCHED to within ±1 ms.**

## prefix_cache_hit audit (all 60 reps)

**All 60 reps have prefix_cache_hit == none.**

## Per-boot / per-instrument summary

| Label | Kind | n | Resid median (ms) | Resid range (ms) | TTFT median (ms) | TTFT range (ms) | ptok range | prompt_tps median | cache_hits |
|---|---|---|---|---|---|---|---|---|---|
| A | short | 10 | 685.9 | [574.6, 891.8] | 1960.0 | [1690.0, 2260.0] | [222, 228] | 176.7 | none |
| A | 2k | 5 | 697.0 | [586.6, 1017.6] | 8370.0 | [7100.0, 8980.0] | [2239, 2331] | 296.94 | none |
| Z1 | short | 10 | 484.7 | [417.1, 725.0] | 1570.0 | [1460.0, 1940.0] | [220, 236] | 201.84 | none |
| Z1 | 2k | 5 | 431.3 | [338.3, 579.7] | 7460.0 | [7000.0, 8370.0] | [2215, 2377] | 325.28 | none |
| B | short | 10 | 634.3 | [558.6, 818.4] | 1835.0 | [1640.0, 2180.0] | [222, 238] | 192.04 | none |
| B | 2k | 5 | 674.8 | [560.4, 821.2] | 8180.0 | [7230.0, 8470.0] | [2239, 2331] | 316.63 | none |
| Z2 | short | 10 | 469.4 | [405.7, 566.5] | 1580.0 | [1430.0, 1740.0] | [222, 230] | 208.36 | none |
| Z2 | 2k | 5 | 400.4 | [336.8, 493.9] | 7940.0 | [7330.0, 8130.0] | [2285, 2354] | 304.91 | none |

## Per-rep residual values (auditable)

### A short (n=10)

| file | resid_ms | ttft_ms | prompt_tokens | prompt_tps | prefix_cache_hit |
|---|---|---|---|---|---|
| A_short_r1.json | 732.972 | 2060.0 | 226 | 169.55175199330006 | none |
| A_short_r2.json | 574.567 | 1690.0 | 226 | 201.71535746212433 | none |
| A_short_r3.json | 713.177 | 1990.0 | 224 | 174.65227077262904 | none |
| A_short_r4.json | 656.304 | 1850.0 | 226 | 188.4902808084956 | none |
| A_short_r5.json | 682.056 | 2060.0 | 228 | 164.7381547507581 | none |
| A_short_r6.json | 722.167 | 2070.0 | 224 | 165.45081535948157 | none |
| A_short_r7.json | 891.756 | 2260.0 | 224 | 162.98263579176393 | none |
| A_short_r8.json | 671.207 | 1930.0 | 226 | 178.74269414008478 | none |
| A_short_r9.json | 678.193 | 1850.0 | 222 | 188.59754323259645 | none |
| A_short_r10.json | 689.826 | 1850.0 | 226 | 193.93647559005313 | none |

resid_all = [732.972, 574.567, 713.177, 656.304, 682.056, 722.167, 891.756, 671.207, 678.193, 689.826]

### A 2k (n=5)

| file | resid_ms | ttft_ms | prompt_tokens | prompt_tps | prefix_cache_hit |
|---|---|---|---|---|---|
| A_2k_r1.json | 713.252 | 8560.0 | 2331 | 296.93831111458434 | none |
| A_2k_r2.json | 598.508 | 8340.0 | 2239 | 289.091564795005 | none |
| A_2k_r3.json | 697.017 | 8370.0 | 2308 | 300.6653524035641 | none |
| A_2k_r4.json | 586.563 | 7100.0 | 2285 | 350.6597454767746 | none |
| A_2k_r5.json | 1017.621 | 8980.0 | 2308 | 289.73753603567053 | none |

resid_all = [713.252, 598.508, 697.017, 586.563, 1017.621]

### Z1 short (n=10)

| file | resid_ms | ttft_ms | prompt_tokens | prompt_tps | prefix_cache_hit |
|---|---|---|---|---|---|
| Z1_short_r1.json | 428.628 | 1570.0 | 220 | 191.87427060732367 | none |
| Z1_short_r2.json | 437.416 | 1460.0 | 226 | 220.03083291205093 | none |
| Z1_short_r3.json | 460.612 | 1630.0 | 224 | 190.69808794918097 | none |
| Z1_short_r4.json | 417.06 | 1470.0 | 226 | 213.68738126206202 | none |
| Z1_short_r5.json | 512.558 | 1660.0 | 224 | 194.3453351041461 | none |
| Z1_short_r6.json | 517.187 | 1550.0 | 226 | 217.85161680915675 | none |
| Z1_short_r7.json | 514.322 | 1570.0 | 222 | 209.34419503612534 | none |
| Z1_short_r8.json | 724.951 | 1940.0 | 236 | 193.40789756180516 | none |
| Z1_short_r9.json | 508.807 | 1540.0 | 222 | 214.3148847984836 | none |
| Z1_short_r10.json | 431.386 | 1630.0 | 232 | 192.72260135505817 | none |

resid_all = [428.628, 437.416, 460.612, 417.06, 512.558, 517.187, 514.322, 724.951, 508.807, 431.386]

### Z1 2k (n=5)

| file | resid_ms | ttft_ms | prompt_tokens | prompt_tps | prefix_cache_hit |
|---|---|---|---|---|---|
| Z1_2k_r1.json | 431.288 | 8370.0 | 2377 | 299.2928686370618 | none |
| Z1_2k_r2.json | 338.256 | 7060.0 | 2215 | 329.37878569877995 | none |
| Z1_2k_r3.json | 488.607 | 7780.0 | 2308 | 316.40044096667225 | none |
| Z1_2k_r4.json | 418.476 | 7000.0 | 2285 | 347.03209216838553 | none |
| Z1_2k_r5.json | 579.716 | 7460.0 | 2239 | 325.2772882609193 | none |

resid_all = [431.288, 338.256, 488.607, 418.476, 579.716]

### B short (n=10)

| file | resid_ms | ttft_ms | prompt_tokens | prompt_tps | prefix_cache_hit |
|---|---|---|---|---|---|
| B_short_r1.json | 629.935 | 2180.0 | 238 | 152.89678935439105 | none |
| B_short_r2.json | 664.46 | 1960.0 | 224 | 172.12896876800176 | none |
| B_short_r3.json | 818.374 | 2130.0 | 228 | 173.06766049203597 | none |
| B_short_r4.json | 567.705 | 1730.0 | 230 | 197.02390658245687 | none |
| B_short_r5.json | 720.785 | 1900.0 | 230 | 194.19706931910858 | none |
| B_short_r6.json | 596.089 | 1760.0 | 222 | 189.87705435997833 | none |
| B_short_r7.json | 558.581 | 1820.0 | 224 | 176.78508591669336 | none |
| B_short_r8.json | 755.986 | 1850.0 | 226 | 205.66465974348162 | none |
| B_short_r9.json | 638.59 | 1640.0 | 222 | 220.688911381713 | none |
| B_short_r10.json | 608.478 | 1730.0 | 226 | 200.62024341954387 | none |

resid_all = [629.935, 664.46, 818.374, 567.705, 720.785, 596.089, 558.581, 755.986, 638.59, 608.478]

### B 2k (n=5)

| file | resid_ms | ttft_ms | prompt_tokens | prompt_tps | prefix_cache_hit |
|---|---|---|---|---|---|
| B_2k_r1.json | 674.762 | 7230.0 | 2308 | 351.93230878906684 | none |
| B_2k_r2.json | 680.671 | 7810.0 | 2308 | 323.59288249050365 | none |
| B_2k_r3.json | 594.792 | 8270.0 | 2285 | 297.58151664056817 | none |
| B_2k_r4.json | 560.383 | 8470.0 | 2239 | 282.9467142931067 | none |
| B_2k_r5.json | 821.162 | 8180.0 | 2331 | 316.6260670686771 | none |

resid_all = [674.762, 680.671, 594.792, 560.383, 821.162]

### Z2 short (n=10)

| file | resid_ms | ttft_ms | prompt_tokens | prompt_tps | prefix_cache_hit |
|---|---|---|---|---|---|
| Z2_short_r1.json | 471.927 | 1450.0 | 230 | 234.13390280003537 | none |
| Z2_short_r2.json | 490.872 | 1520.0 | 230 | 222.51855365102992 | none |
| Z2_short_r3.json | 466.934 | 1640.0 | 224 | 190.10016446422776 | none |
| Z2_short_r4.json | 405.731 | 1440.0 | 230 | 221.41234656478235 | none |
| Z2_short_r5.json | 524.24 | 1740.0 | 224 | 183.42428464444384 | none |
| Z2_short_r6.json | 435.505 | 1430.0 | 222 | 222.22324630572763 | none |
| Z2_short_r7.json | 522.436 | 1720.0 | 224 | 186.2112969546415 | none |
| Z2_short_r8.json | 566.458 | 1680.0 | 224 | 200.26179966041067 | none |
| Z2_short_r9.json | 444.646 | 1690.0 | 228 | 182.27753170349533 | none |
| Z2_short_r10.json | 450.51 | 1490.0 | 226 | 216.45233351029614 | none |

resid_all = [471.927, 490.872, 466.934, 405.731, 524.24, 435.505, 522.436, 566.458, 444.646, 450.51]

### Z2 2k (n=5)

| file | resid_ms | ttft_ms | prompt_tokens | prompt_tps | prefix_cache_hit |
|---|---|---|---|---|---|
| Z2_2k_r1.json | 449.168 | 7940.0 | 2285 | 304.9060555770536 | none |
| Z2_2k_r2.json | 400.396 | 8130.0 | 2354 | 304.4140555810006 | none |
| Z2_2k_r3.json | 336.758 | 8100.0 | 2354 | 303.09501974271393 | none |
| Z2_2k_r4.json | 379.983 | 7330.0 | 2285 | 328.632301496614 | none |
| Z2_2k_r5.json | 493.866 | 7620.0 | 2308 | 323.7379387873739 | none |

resid_all = [449.168, 400.396, 336.758, 379.983, 493.866]

## SHORT instrument (GOVERNING) — C1 / C2

Boot short residual medians: A=685.9, Z1=484.7, B=634.3, Z2=469.4

spread(RV200) = max - min across A,B short residual medians = **51.6 ms**

**C1**: min(RV200 medians) - max(RV0 medians) > spread(RV200)
- LHS = min(RV200 medians) - max(RV0 medians) = 149.6 ms
- RHS = spread(RV200) = 51.6 ms
- **C1 result: PASS**

**C2**: pooled short residual gap = median(all RV200 reps, n=20) - median(all RV0 reps, n=20)
- pooled RV200 median = 674.7 ms
- pooled RV0 median = 469.4 ms
- pooled gap magnitude = **205.3 ms**, sign direction = RV=0 LOWER
- in [150, 250] band: True
- **C2 result: PASS**

## 2K instrument (SECONDARY DIAGNOSTIC — non-governing) — C1 / C2

Boot 2K residual medians: A=697.0, Z1=431.3, B=674.8, Z2=400.4

spread(RV200)_2k = **22.2 ms**

**C1 (2K, non-governing)**: LHS = 243.5 ms, RHS = 22.2 ms -> **PASS**

**C2 (2K, non-governing)**: pooled gap magnitude = 252.8 ms, sign = RV=0 LOWER, in [150,250] band: False -> **FAIL**

---
*This report covers R9-only, 4 boots. It is one of three required breakdowns 
per PRE-REGISTRATION section 4.1 (the other two — fresh-pair-only 2 boots, and 
the full 6-boot set — require round-10 fresh data and are out of scope here).*
