# Margin Analysis Report

## Summary Table

| Context | n | Mean | Median | p10 | p90 | Frac < 3.62 | Flip Rate |
|---|---|---|---|---|---|---|---|
| ctx20k | 999 | 6.955 | 4.875 | 0.500 | 16.375 | 42.042% | 11.247% |
| ctx2k | 1000 | 6.343 | 4.750 | 0.500 | 14.513 | 43.600% | 11.773% |
| ctx5k | 1000 | 6.646 | 4.750 | 0.500 | 15.375 | 42.600% | 11.872% |
| trivial | 1000 | 7.075 | 5.125 | 0.625 | 15.750 | 42.400% | 11.060% |
| **Pooled** | **3999** | **6.754** | **4.875** | **0.500** | **15.500** | **42.661%** | **11.488%** |

## Verdict

- **Measured Frac < 3.62:** 42.661% (Asserted: ~58%)
- **Implied Flip Rate:** 11.488% (Asserted: ~16%)

**Verdict: REFUTED**

*Note: Flip rate is an estimate via synthetic-band kernel, not a direct measurement.*
