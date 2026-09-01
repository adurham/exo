# V3 RESULTS — MTP per-cycle state-snapshot cost (`rb_snap`)

**Date:** 2026-09-01
**Plan:** `v3/V3_PROFILING_PLAN.md` (pre-registered gate, §1)
**Raw data:** `v3/run1/` (`exo_m4-1.log`, `exo_m4-2.log`, `bench_v3_100k.json`, `gate_analysis.json`, `bench_summary.json`)

---

## VERDICT: **CLOSE** — the V3 hypothesis is refuted

`rb_snap` costs **0.150 ms**, which is **0.218 %** of a 68.85 ms verify cycle.

The gate closes on the **absolute floor** (`< 1 ms`) — this is not a dead-zone
call requiring the tie-break rule. It also fails the share criterion by a
factor of ~14 (0.218 % vs the 3 % PROCEED bar). The measurement is
**trustworthy in absolute terms**: profiler inflation was **0.93×** (below the
plan's 1.2 × "absolutes are trustworthy" threshold), so the 0.150 ms is not a
perturbation artifact being explained away.

| Gate condition (pre-registered) | Threshold | Measured | Result |
|---|---|---|---|
| CLOSE — absolute floor | `rb_snap < 1 ms` | **0.150 ms** | **✅ CLOSE (met)** |
| PROCEED — share of cycle | `rb_snap/total ≥ 3 %` | **0.218 %** | ❌ not met (14× short) |
| DEAD ZONE | `1 ms ≤ rb_snap < 3 %` | n/a | not reached |

**Decision (plan §7):** close the V3 hypothesis as **allocator noise**. No code
change. The copy-on-write fix (which the corrected Claim-6 finding showed would
have to cover **both** rings and pools — larger than originally scoped) is **not
worth pursuing**. There is no meaningful latency to reclaim: eliminating the
snapshot entirely would return 0.15 ms of a 68.85 ms cycle (~0.2 %), well inside
run-to-run noise (decode σ = 1.79 t/s on a 33.99 t/s mean, i.e. ±5 %).

---

## 1. `rb_snap` measurements

### Method note (important — corrects the plan's analysis script)

The `[MTP-PROF]` dumps are **cumulative running means**, not per-interval
snapshots: `n` grows 50 → 100 → … → 1050 and `mean` is the running average over
all cycles so far. The plan's §5 script (`statistics.mean` over dump lines)
would **double-count early cycles** and silently bias the result. Actual
arithmetic used here reconstructs per-interval values:

```
sum_k    = mean_k × n_k
interval = (sum_k − sum_{k−1}) / (n_k − n_{k−1})
```

then weights each interval by its cycle count. Only cycles logged inside the
bench window (≥ 16:52) are counted, so pre-bench idle cycles are excluded.

### Summary (bench window, both nodes)

| Node | cycles | `rb_snap` mean | `total` mean | **share** |
|---|---|---|---|---|
| macstudio-m4-1 | 650 | 0.150 ms | 68.84 ms | **0.218 %** |
| macstudio-m4-2 | 650 | 0.150 ms | 68.86 ms | **0.218 %** |
| **combined** | **1300** | **0.150 ms** | **68.85 ms** | **0.218 %** |

`rb_snap` was extraordinarily stable: every reconstructed interval on both nodes
landed at 0.150 ms (raw per-dump min/max across the whole run: 0.10 – 0.33 ms).
The two nodes agree to 3 significant figures — expected, since both ranks run
the same verify cycle in lockstep.

### Full per-cycle phase budget (node1 / node2, bench window, n=650 each)

| Phase | node1 | node2 | share of cycle |
|---|---|---|---|
| `draft` | 9.189 ms | 9.183 ms | 13.3 % |
| `verify` | 56.087 ms | 55.957 ms | **81.4 %** |
| `accept` | 2.868 ms | 3.002 ms | 4.3 % |
| `rollback` | 0.705 ms | 0.722 ms | 1.0 % |
| **`total`** | **68.839 ms** | **68.863 ms** | 100 % |
| `rb_snap` | 0.150 ms | 0.150 ms | **0.218 %** |

Rollback sub-phases (n=278 — these fire only on cycles that actually restore,
not every cycle):

| Sub-phase | node1 | node2 |
|---|---|---|
| `rb_ring` | 0.561 ms | 0.580 ms |
| `rb_pool` | 0.301 ms | 0.320 ms |
| `rb_gate` | 0.060 ms | 0.060 ms |
| `rb_drain` | 0.030 ms | 0.030 ms |
| `rb_tail` | 0.030 ms | 0.020 ms |

The whole rollback family sums to ~1.0 ms on restore cycles, consistent with the
0.705 / 0.722 ms `rollback` parent averaged over all cycles. **The entire
snapshot + rollback machinery is ~1 % of cycle time.** The cycle is dominated by
`verify` at 81 %.

---

## 2. ANOMALY: `rb_pool_restores` is a COUNT mislabeled as `ms`

The dumps show `rb_pool_restores mean=18.91ms min=0.00ms max=62.00ms`, which
reads as 25 % of cycle time and would look like the real V3 smoking gun.
**It is not a time.** Verified in source, not inferred from the numbers:

- `src/exo/worker/engines/mlx/speculative/dsv4_mtp.py:5023` records
  `prof.record("rb_pool_restores", float(_rb_pool_restores))` — a float cast of
  an **integer counter** (incremented at lines 5011/5017), *not* a
  `perf_counter` delta. Every genuine time series in this file records
  `(t1 − t0) * 1000.0`.
- The formatter at `dsv4_mtp.py:819–822` is **unit-blind**: it appends a fixed
  `"ms"` to the mean/min/max of *every* series, with no per-phase unit metadata.
- The code's own comment (`dsv4_mtp.py:255–256`) states it "counts pools taking
  the restore+re-accumulate branch that cycle (vs trim)".

So the true reading is **18.33 of 62 pools restored per cycle** (max 62 = the
pool count, matching `len(_pool_caches)`; min 0 = a cycle where all pools
trimmed). Three independent cross-checks agree: a sub-phase cannot cost 18.91 ms
when its `rollback` parent costs 0.705 ms; the real sub-phases sum to ~1 ms; and
`max=62.00` is exactly the known pool count.

**This is a live trap for future readers of this profiler** — anyone eyeballing
the dump will read a 25 %-of-cycle hotspot that does not exist. Worth a
one-line label fix (e.g. emit counts without the `ms` suffix), tracked
separately; no change made here.

`rb_snap` itself **was** confirmed a genuine wall-clock duration
(`dsv4_mtp.py:4113–4115` / `4141–4145`: `mx.synchronize()` → `perf_counter()`
delta × 1000), so the 0.150 ms figure is a real measured time.

---

## 3. Benchmark context (sanity data)

`bench/concurrent_bench.py`, c=1, 5 scored iterations + 1 warmup, max_tokens=256,
`--prompt-words 75000` → **89,408 prompt tokens** actual.

| Metric | Value |
|---|---|
| Decode t/s (5 scored) | **33.99 mean** / 34.03 med / 32.01 min / 36.90 max / σ 1.79 |
| Prefill t/s (5 scored) | **425.7 mean** (424.1 – 427.0, σ ≈ 1.2) |
| Wall per iteration | 217.4 – 219.4 s (very tight) |
| Errors / bad rate | **0 / 0 %** (0 of 5) |
| `tail_ratio` | 1.00 every iteration |

Prefill was remarkably consistent (±0.7 %). Decode variance (±5 %) is normal
run-to-run MTP acceptance-rate variation.

### Profiler perturbation check (plan §6)

| | Value |
|---|---|
| Profiled `total` mean | 68.85 ms |
| Unprofiled baseline (plan) | 74.0 ms |
| **Inflation factor** | **0.93×** |

Inflation is **below 1.0** — the profiled cycle was *faster* than the 74 ms
reference baseline. The profiler is essentially free at this aggregation
interval (50 cycles ≈ one dump per ~3.4 s). Per plan §6 this is the
"inflation < 1.2× → both share AND absolute ms are trustworthy" case, so the
gate applies as-is with no caveat.

The sub-1.0 ratio means the 74 ms baseline is slightly stale rather than the
profiler being negative-cost — the current cycle is genuinely ~69 ms. This does
not affect the verdict in any direction: `rb_snap` fails the gate on the
absolute floor, and a *smaller* denominator would only make the share look
*larger*, yet it is still 14× short of the bar.

---

## 4. Telemetry sampler validation (plan Phase 4) — **PARTIAL FAIL**

Validated sampler **(e)** (runner PID/lstart restart detector) in
`telemetry/collect_telemetry.py` against the live runners. Full evidence:
`v3/TELEMETRY_VALIDATION.md`.

**Real runner signatures (ground truth, unchanged all run):**

- node1: PID **83029**, lstart `Tue Sep 1 16:19:35 2026`, `.venv/bin/python -m exo -v`
- node2: PID **85554**, lstart `Tue Sep 1 16:19:37 2026`, `.venv/bin/python -m exo -v`

| Criterion | Result |
|---|---|
| (a) Matches the genuine runner | ✅ PASS — both nodes, exact |
| (b) Rejects wrapper processes | ❌ **FAIL** — 3 false positives per node |
| (c) Stable across samples / detects restart | ✅ PASS — identical over ~20 s; new pid+lstart on restart |

**The bug:** `parse_ps_runner` applies `rx.search(args)` to the **full args
string** (`collect_telemetry.py:335`). The launcher wrappers — `SCREEN -dmS
exorun` (83017/85543), `login -pflq` (83018/85544), and `zsh -l -c`
(83019/85545) — embed the literal `-m exo` in their command line, so all three
match `_DEFAULT_RUNNER_PATTERN`. Result: **`count` = 4 per node, not 1.**

This is precisely the false-positive class the sampler's own docstring
(lines 289–293) claims to have designed around; the mitigation there (avoiding a
bare python-path match) doesn't help, because the wrappers match via the
embedded `-m exo` marker, not via a path.

**Verified independently** (I re-ran the sampler's own `parse_ps_runner` against
live `ps` output rather than trusting the report): 4 matches confirmed;
the proposed one-line fix — additionally require the `comm` basename to start
with `python` — collapses it to exactly the 1 genuine runner on both nodes.
Correctly rejected in both variants: `multiprocessing.resource_tracker` and
`multiprocessing.spawn` children.

**Impact:** restart detection still *works* (a real restart changes the runner's
pid+lstart and is visible), but the signature set is polluted by wrapper rows
whose lstart moves independently, so a naive "did the signature set change?"
comparison can raise a false restart alarm. **Fix before relying on this sampler
in an unattended run.** Not applied here — read-only constraint; diff is in
`TELEMETRY_VALIDATION.md`.

---

## 5. Other anomalies observed

- **No inference errors.** 18 `Exception` lines appear in each node's log during
  the window, but all 18 are `download_utils:fetch_file_list_with_cache` HF
  404s from an unrelated background download poller — not inference, not
  triggered by the bench. Zero KV evictions, zero `clear_cache` emergency
  reclaims, zero stream interrupts.
- **No runner restart during the run** — PIDs 83029 / 85554 and their lstart
  values are identical before and after the benchmark.
- **Plan/reality deviations** (recorded for reproducibility, none affect the verdict):
  1. Model ID is `deepseek-ai/DeepSeek-V4-Flash-0731`, not the plan's
     `mlx-community/DeepSeek-V4-Flash-8bit`. Used the actually-placed ID.
  2. The bf16-KV env var is `EXO_KV_CACHE_BITS=0` (confirmed live), not the
     plan's `DSV4_KV_CACHE_BITS=0`. bf16 KV was correctly in force.
  3. `bench/concurrent_bench.py` needs `PYTHONPATH=<repo>/tools/src` — `exo_tools`
     is a separate uv workspace member and is not importable from the root env.
     Without it the bench dies instantly on `ModuleNotFoundError`.
  4. The plan's §5 analysis snippet is wrong for cumulative dumps (see §1).

---

## 6. Bottom line

**The V3 fix is not worth pursuing.** `rb_snap` is 0.150 ms — 0.218 % of the
cycle — measured with a profiler that inflated nothing (0.93×). The gate closes
on the absolute floor, not on a judgment call. The apparent 18.91 "ms" hotspot
that would have argued the other way is a mislabeled pool counter, confirmed in
source.

If per-cycle decode latency is the goal, the data points at `verify`
(56 ms, **81 %** of the cycle), not at the snapshot. The documented pool ratchet
(1.95 → 2.27 GB) remains a separate memory-residency question and was not
touched here.
