# Acceptance cliff at depth: indexer top-k threshold hypothesis (2026-09-23, overnight)

## The measured shape

Same probe (`bench/long_decode_probe.py`), same prompt at every depth, only
depth varies — acceptance and throughput:

| depth | decode tps | acc/cycle | tok/cycle | ms/cycle | pageins | needle |
|---|---|---|---|---|---|---|
| 27,850 | **46.30** | **2.077** | 3.077 | 66.5 | 587 | ✓ |
| 127,929 | 29.72 | 0.814 | 1.818 | 61.2 | 400 | ✓ |
| 284,831 | 28.94 | 0.878 | — | 64.9 | 797 | ✓ |
| 273,622 (earlier) | 26.83 | — | — | — | — | ✓ |
| 564,926 (earlier) | 13.36 | — | — | — | — | ✓ |

Two things stand out:

1. **The entire loss happens between ~28K and ~128K.** From 128K out to 285K
   throughput is FLAT (29.72 -> 28.94). That is a **step**, not a gradual
   curve.
2. **Cycle cost is flat throughout** (61.2-66.5 ms). The change is entirely in
   **acceptance** (2.077 -> 0.814, a 2.55x drop).

A step change in one quantity, with cost unchanged, points at a **threshold**
in some selection mechanism — not at gradual dilution.

## The hypothesis

DSv4's sparse attention runs an Indexer that selects the top `index_topk`
entries from the **compressed** KV (compress_ratio 128 on most layers).

Config (verified from the served checkpoint):
```
index_topk      = 512
compress_ratios = [0, 0, 4, 128, 4, 128, ...]   # 128 dominant on sparse layers
```

Arithmetic:
```
compressed entries available = depth / 128
top-512 covers ALL of them  <=>  depth <= 512 * 128 = 65,536
```

| depth | compressed entries | % of 512 budget | regime |
|---|---|---|---|
| 27,850 | 218.8 | 42.7% | dense — top-512 covers everything |
| 40,000 | 312.5 | 61.0% | dense |
| 55,000 | 429.7 | 83.9% | dense |
| **65,536** | **512.0** | **100.0%** | **crossover** |
| 70,000 | 546.9 | 106.8% | selective — entries start being dropped |
| 127,929 | 999.3 | 195.3% | selective |
| 284,831 | 2225.2 | 434.9% | selective |

**The predicted crossover (~65,536) sits squarely between the healthy 28K rung
and the degraded 128K rung.** The measured step shape is consistent with it.

Interpretation: below the crossover, sparse attention is *effectively dense* —
nothing is discarded. Above it, the indexer must actually select, dropping
compressed entries. Dropping context should plausibly change the target's
next-token distribution in a way the fixed 128-token-window DSpark drafter
tracks worse → acceptance falls → throughput falls.

## Falsifiable prediction

**The cliff should be spatially localised near 65K, not spread smoothly.**

Test: fine-grained rungs at 40K / 55K / 70K / 90K / 110K.
- If acceptance stays ~2.0 through 55K then steps down by 70-90K → hypothesis
  **supported**.
- If acceptance declines smoothly across those rungs → hypothesis **falsified**;
  the cause is gradual, and a different mechanism is responsible.

`/tmp/cliff_ladder.py` implements exactly this and writes `/tmp/cliff_ladder.jsonl`.

## SOURCE-VERIFIED: the threshold is real in code

`mlx-lm/mlx_lm/models/deepseek_v4.py`, `Indexer.__call__`, line ~4795:

```python
k = min(self.index_topk, pooled.shape[1])
```

That is exactly the predicted arithmetic. `pooled` is the compressed KV
(`depth / compress_ratio` entries). So:

- `pooled.shape[1] <= 512` -> `k = pooled.shape[1]` -> **top-k selects ALL
  entries** (no coverage loss, effectively dense)
- `pooled.shape[1] > 512`  -> `k = 512` -> **entries are dropped**

Crossover at `pooled.shape[1] == 512`, i.e. depth = 512 * 128 = **65,536
tokens** — matching the arithmetic and sitting between the healthy 28K rung
and the degraded 128K rung.

Also note from the same source: the indexer comment states
`EXO_DSV4_INDEX_TOPK` is "validated quality-neutral at 192 on AIME for
DSv4-Flash-6bit" — i.e. the *set* of selected entries matters less than
expected quality-wise, BUT that was measured on a different checkpoint and the
lever's effect on *acceptance* (not quality) at depth is what concerns us here.
Raising topk above 512 increases coverage and therefore the drafter's view.

## Why this matters for the user's targets

- **T2 (250K >= 30 t/s):** if the cliff is the indexer threshold, then 250K is
  already in the "selective" regime and cannot be improved by any config knob
  that leaves `index_topk` alone — the acceptance loss is a property of the
  model's sparse attention at that depth. The only levers would be
  `EXO_DSV4_INDEX_TOPK` itself (raising it recovers coverage at the cost of
  more compute per verify cycle — a measurable trade) or accepting it.
- Note `EXO_DSV4_INDEX_TOPK` is listed as **FORBIDDEN below 512** in the
  cluster notes (quality), but *raising* it is not the forbidden direction and
  is a legitimate, testable lever: e.g. 1024 would push the crossover to
  ~131,072 tokens.
- **T1 (500K collapse):** different mechanism (refault, measured 32,469
  page-ins at 42K under memory pressure; peak at 95.9% of the wired limit).
  Fixing the indexer crossover does NOT fix the refault.

## Caveat on variance (important)

Tonight's numbers show large run-to-run spread at similar depths:
21,855 -> 33.22 vs 27,850 -> 46.30 (+39%); 115,614 -> 36.74 vs 127,929 ->
29.72 (-19%).

**Any single rung is therefore weak evidence.** The cliff test must be read as
a SHAPE across many rungs, not as point values, and the variance at one fixed
depth should be quantified alongside. `/tmp/variance_test.py` does that.

## Status
- Hypothesis stated with a falsifiable prediction and a test script.
- Cliff ladder NOT yet run (500K rung of the depth ladder still in flight).
- No cluster config changed.
