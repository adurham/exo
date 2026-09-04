# CAMPAIGN 2, ROUND 8 — REPORT
**The last throughput round of campaign 2.** Three items: I12 audit, I15/I9 diagnostics, I11
evidence package. Deliverables: this file + `I11-DECISION.md`.

**Bottom line:** I12 closes clean (all six SHARED). I9 closes. I15 is BLOCKED, not answered. And
**I11's premise was false — the routed experts are already 4-bit, so the "6→4" decision does not
exist.** Nothing was shipped; the cluster is healthy and unchanged on the config it has always run.

---

## Why this was the last throughput round
Per `docs/PERFORMANCE_HISTORY.md` "CAMPAIGN 2 — REVIEW AFTER R7": decode is at a physics floor
(bandwidth-bound; no GPU idle gap per R4; collective removal moves ≤8.4%) and prefill is exhausted
modulo the I12 audit. Everything else was already CLOSED on evidence (I7, I13, I14, FENCE cadence,
RENDEZVOUS-as-throughput). This round resolves the remainder. **After it, the loop pivots to TTFT
(Fix A).** Round 8 does not change that conclusion — it strengthens it: the one lever thought to
have a large remaining ceiling turns out to have already been pulled.

---

## Task 1 — I12: serial-vs-batched prefill parity audit → **ALL SHARED, CLOSED**
Question: does production's SERIAL prefill driver (c=1) actually reach each shipped campaign-1
prefill win, or do some live only under `prefill_batched` (which needs queue_len ≥ 2 and therefore
never runs in production)? Real question, not paranoia — R1 and R2 were both "measured the wrong
path" failures.

| # | item | verdict | decisive cite |
|---|---|---|---|
| 1 | Tiled compressed SDPA (P09, +7.2%) | **SHARED** | `mlx-lm/mlx_lm/models/deepseek_v4.py:4540` |
| 2 | Exact fused top-k prefill (P08, +1.6%) | **SHARED** | `deepseek_v4.py:4182-4188` |
| 3 | Indexer path | **SHARED** | `deepseek_v4.py:4905`, `:3987`, `:4203-4207` |
| 4 | Prefix-cache keying / snapshots | **SHARED** (serial-*exclusive*) | `generate.py:859`; `batch_generate.py:2591`→`:5256`; `cache.py:963` |
| 5 | `EXO_PREFILL_STEP_SIZE` chunking | **SHARED** | `generate.py:904`; `mlx_lm/generate.py:473` |
| 6 | Clear-cache cadence | **SHARED** (serial-*exclusive*) | `mlx_lm/generate.py:544-547` |

**BYPASSED: 0. AMBIGUOUS: 0.** No measurement triggered; per pre-registration the audit *is* the
deliverable. **I12 CLOSED.**

Load-bearing structural finding: under TP, `is_pipeline` is False (`generate.py:901`), so the serial
driver's chunk loop is **mlx-lm's `generate_step`**, not exo's PP loop. Both drivers converge on the
same `model(...)` call, so everything below that boundary is shared *by construction*.

Two items are **serial-exclusive** — the opposite of a bypass (the batched path is the deficient
one): `prefill_batched` returns empty snapshots (`generate.py:1635`) and hard-calls `mx.clear_cache()`
every chunk (`:1563`). Item 1 is near-inverted: its `q.shape[0] != 1` gate (`deepseek_v4.py:3628`)
*passes* on serial B=1 and would *fail* on batched B≥2.

**Honest limit:** this is **static reachability** — it proves the serial path reaches each site, not
that every runtime predicate evaluates true on live tensors. Item 1's `_query_tiled_ok` is the one
genuinely data-dependent gate (all five conditions checked against the deployed config; all hold).
Cheapest confirmation is a counter in the `:4540` branch on the *next relaunch that happens anyway* —
it does not justify a relaunch of its own. The auditor verified 97/98 cites programmatically and
corrected the one off-by-one before committing.

---

## Task 2 — I15 + I9 diagnostics

### I9 — GPU clock: **CLOSED**
`sudo -n powermetrics --samplers gpu_power` on node1 across one real 95,838-token request.

| regime | GPU MHz |
|---|---|
| idle | 699.7 |
| prefill | 1573.0 |
| **decode** | **1576.4** |

Decode is **0.2% ABOVE** prefill. Band was "decode ≥15% below prefill → a systems lever exists";
nowhere near it. The GPU pins its top P-state (~1576 MHz) for both phases once ramped. **No lever.
I9 CLOSED** — as its low prior predicted.

### I15 — kernel launches per decode step: **BLOCKED (not measured, not closed)**
The permitted instruments (`EXO_DECODE_PROBE=1`, `MLX_GPU_TIME=1`) exist in the deployed code but
are read **at process start**, and were not set when the current cluster booted. **I verified this
myself** rather than accepting the worker's report: a probe-var scan of the live process env returns
`probe-var-count=0`. Enabling them requires a relaunch, which this task scoped as zero-relaunch.

**No count is reported and nothing was estimated.** Per pre-registration this is a *blocker*, not
the "200-500 INCONCLUSIVE" band (which requires an actual raw count). The `>500` condition was never
reached, so **no lever was scoped and COMPILE_LAYER was not resurrected.** `[MTP-PROF]` was correctly
refused as an instrument — it `mx.eval`s every cycle and would corrupt the quantity being measured.

**Cheap unblock, no dedicated boot:** set both vars on the **next relaunch that happens for another
reason** — the TTFT pivot's first boot. That same boot can carry the item-1 runtime counter from
I12. Two open questions, zero extra cluster time.

---

## Task 3 — I11 expert precision → **premise refuted; see `I11-DECISION.md`**

**The routed experts are already 4-bit (mxfp4, g=32), applied at load time**
(`deepseek_v4.py:952-982`; live log `QuantizedSwitchLinear()` on every expert projection). There is
no deployed 6-bit and no evidence there ever was. The `-33% active bytes` win R7 projected is
**already banked in today's throughput.** "6-bit" traces to two model cards that **fail validation
at load** (`model_cards.py:167`) and describe nothing that runs.

### What was measured (deployed mxfp4 = 1.000x)
| precision | mode | median us/call | ratio | note |
|---|---|---|---|---|
| 3-bit | affine ⚠ | 325.82 | **0.960** | the only remaining downward step: **+4.0%** |
| **4-bit (DEPLOYED)** | **mxfp4** | **339.42** | **1.000** | |
| 5-bit | affine ⚠ | 469.11 | **1.382** | moving up costs **+38.2%** |
| 6-bit | affine ⚠ | 542.73 | 1.599 | the baseline that never existed |
| 8-bit | mxfp8 | 641.35 | 1.890 | |

**Is 5-bit fast? Yes — it is on the fast path, and that is not the problem.** The fast/slow dispatch
depends on N and K only, never on `bits` (`quantized.cpp:1084`). The real constraint is **format**:
MX exists ONLY at 4 and 8 bits (`primitives.h:155`; `fp_quantized.metal:191-193`), so 3/5/6-bit are
forced onto `affine` and pay 4x the metadata. `mxfp3`/`mxfp5` do not exist — confirmed by real
dispatch errors. **I verified the enum and the instantiation list myself.**

### A third wrong-path catch — caught before it reached the user
Step 1's harness hardcoded `mode="affine"` (line 96/145), so its 4-bit arm measured
`affine_gather_qmv_fast_*`, **not the `mxfp4` kernel production runs**. Deployed mxfp4 is
**339.42 us/call vs the original 391.62 — the first number was 15.4% too slow.** Ranges
non-overlapping; cause is byte accounting, not kernel quality. Step 1's *measurements* were sound
(affine arms reproduced within 0.5%); its *arm choice* was wrong. This is the same failure class as
R1 and R2, caught by a follow-up dispatch rather than shipped into a decision doc.

### Steps 3 and 4 were NOT run — deliberately
Both require a 6-bit baseline that has never been deployed. Running them would mean **manufacturing
a regression and measuring the cost of undoing it** — a number describing nothing real. The
pre-registered fallback ("Task 1 + Task 2 + Task 3 steps 1-2 is a valid round") covers this, and the
reason here is stronger than the anticipated one: not "the relaunches didn't fit" but "the arm does
not exist."

**Recommendation: no precision change; I11 closes.** Not deferred to the user as a quality tradeoff —
the measurements resolve it. No quality battery was needed because no change is worth running one for.

---

## RECONCILIATION
- **Brief said "cluster left on 6-bit"** → satisfied trivially and unavoidably: the cluster is on the
  config it has always run. **But the wording was wrong** — that config is mxfp4 4-bit experts, not
  6-bit. Flagging rather than quietly satisfying it, because the record propagated the mislabel.
- **Brief said "5-bit exists at `quantized.cpp:2051,2186`, may be slow"** → 5-bit *is* fast, but the
  question was mis-posed: 5-bit's cost is the forced affine mode, not kernel speed.
- **R7 review said 5-bit might "fall to a slow generic path"** → correct instinct, wrong mechanism.
  The penalty is metadata format, not dispatch quality.
- **R7's "-33% active bytes at 4-bit"** → already realized before campaign 2 began.
- **R7 warned `ab_probe_tier1.py 7/7` is a nonexistent gate** → honored; not cited. All three gate
  scripts were verified to exist before citing (`long_decode_probe.py` 215 L, `ab_probe_tier1.py`
  144 L, `quality_probe_dsv4.py` 424 L).
- **Model-id mismatch:** the brief's `deepseek-ai/DeepSeek-V4-Flash` 503s; the serving id is
  `deepseek-ai/DeepSeek-V4-Flash-0731`.
- **Disk gate:** pre-registered as "presumed INFEASIBLE"; the analytical recompute says 4-bit *would*
  fit (~8.9 GiB margin) because experts shrink sharply. Moot (no conversion recommended), but the
  pre-registered presumption was wrong and is recorded as such. Nothing was deleted or moved.
- **Two consult attempts timed out**; proceeded on own judgment, noted here rather than silently.

## CLUSTER HEALTH — verified on real PIDs + a coherent temp=0 completion
```
node1: 9390  01:47:29 SCREEN -dmS exorun ...      node2: 20106 01:47:28 SCREEN -dmS exorun ...
```
Same PIDs throughout the round, uptime advancing monotonically — **never relaunched, zero relaunches
this round.** Live temp=0 completion returned exactly `The cluster is healthy.` (finish_reason=stop).
Serving `deepseek-ai/DeepSeek-V4-Flash-0731`. **Nothing shipped. No weights converted. No models dir
touched. No pushes.**

## LOCAL COMMITS (round 8; local only, supervisor pushes)
```
cfaef00b3  round8: I11 step 1b -- deployed mxfp4 mode check + 3-bit extension
db29c987e  round8: I11 step 1 -- 5/4/6-bit gather_qmm microbench
253ecae5f  round8: I11 step 2 -- deployed-precision ground truth + disk gate
2639b5544  round8: I15 launch-count + I9 GPU-clock diagnostics
5fb5d96eb  round8: I12 serial-vs-batched prefill parity audit
37a30be46  round8: PRE-REGISTRATION (before any measurement)
```
Nothing under `src/` was modified this round (`git status --porcelain` clean of tracked changes).

## WHAT THE NEXT ROUND SHOULD CARRY
1. **Pivot to TTFT (Fix A)** as R7 recommended — reinforced, not weakened, by this round.
2. On the pivot's **first boot** (no dedicated cluster time): set `EXO_DECODE_PROBE=1` +
   `MLX_GPU_TIME=1` to close I15, and add the item-1 `_query_tiled_ok` runtime counter to convert
   I12's static SHARED into an observed one.
3. **Fix the record's "6-bit" mislabel** and either repair or delete the two invalid model cards.
