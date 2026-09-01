# V3 PROFILING PLAN — Measure the MTP per-cycle state-snapshot cost (rb_snap)

**File:** `/Users/adam.durham/repos/exo/tmp/research-v1v2v3-20260901/v3/V3_PROFILING_PLAN.md`
**Date:** 2026-09-01
**Status:** AWAITING USER APPROVAL — **this plan requires a cluster relaunch and must NOT be executed until the user approves it.**
**Authoritative input:** `/Users/adam.durham/repos/exo/tmp/research-v1v2v3-20260901/v3/CODE_VERIFICATION.md` (PM-verified). Every code fact below cites that file and the live source it quotes.

---

## 0. WHY THIS NEEDS APPROVAL

**This run requires a full cluster relaunch** (`./start_cluster.sh`), which tears down any live sessions on both Mac Studio nodes and takes ~6–8 minutes (RDMA preflight + dashboard rebuild + node sync/launch + DSv4 placement/ready). It also runs a ~30-minute 100K-context benchmark. Do not execute any command in this document until the user has explicitly approved the relaunch.

---

## 1. OBJECTIVE + PRE-REGISTERED GATE

### Objective
Measure the real per-cycle cost of the MTP speculative-decode **state snapshot** (`rb_snap`): the unconditional `save_meta()`/`save_spec_state()` block that materializes all 62 pool caches plus every ring on every verify cycle (dsv4_mtp.py:4118–4128). The V3 hypothesis claims this is a material per-cycle latency cost plus a source of MLX allocator churn. This single profiling run decides whether a copy-on-write fix is worth designing.

### PRE-REGISTERED GATE (decide BEFORE looking at any data)

The gate is expressed as a **SHARE of cycle time** (rb_snap mean ÷ total cycle mean), **not** as an absolute millisecond value, because the profiler perturbs its own measurement (see §6). The absolute `rb_snap` ms is an **upper bound** on the true cost; the share is robust to the profiler's uniform inflation of the whole cycle and is the trustworthy signal.

| Outcome | Condition (primary = share) | Decision |
|---|---|---|
| **PROCEED** | `rb_snap_mean / total_mean ≥ 3%` (≈ ≥ 2.2 ms of a ~74 ms cycle at 100K) | Design the copy-on-write fix. **Note: the fix is now LARGER than originally scoped** — see §7. |
| **CLOSE** | `rb_snap_mean < 1 ms` (absolute floor) | Close the V3 hypothesis as **allocator noise**. No fix. |
| **DEAD ZONE** | `1 ms ≤ rb_snap_mean < 3% of total_mean` | **Default CLOSE** — do not proceed (rule below). |

**Dead-zone rule (explicit, not ambiguous):** if `rb_snap_mean` lands in the dead zone (≥ 1 ms but < 3% of cycle), **default to CLOSE** — do not proceed. Rationale: (a) `rb_snap` is an *upper bound* (the profiler forces device syncs inside the bracket, dsv4_mtp.py:4114/4142), so the true cost is ≤ the measured value; a sub-3% upper-bound cost is not material. (b) The fix is now larger than originally scoped (copy-on-write for **both** rings and pools — see §7), so a marginal 1–2 ms upper-bound cost does not justify it. **The only path out of the dead zone:** if the perturbation sanity check (§6) shows the profiler inflated total cycle time by **< 20%** (i.e. the profiler is cheap and the measurement is trustworthy), then escalate to a second opinion before deciding — a real 1–2 ms cost might warrant a targeted pool-only investigation.

**Why share is the primary criterion:** the profiler inserts `mx.eval`/`mx.synchronize()` at phase boundaries that serialize the pipelined draft/verify/accept (dsv4_mtp.py:238–241, 251–253). This inflates *every* phase's absolute ms roughly proportionally, so the ratio `rb_snap/total` is far more stable than either absolute number. If `rb_snap` is small even as a *share* of the inflated cycle, the hypothesis is genuinely weak — this is the correct conservative direction for the gate.

---

## 2. EXACT LAUNCH COMMAND

Run from the launch machine (the laptop that owns the cluster). This sets **both profiling env vars** and **all shipped baseline env vars** explicitly (each is a script default or the user's bf16-KV rule; set explicitly for reproducibility — the script reads them from the launching shell's environment, start_cluster.sh:2018/2022).

```bash
cd /Users/adam.durham/repos/exo && \
  DSV4_KV_CACHE_BITS=0 \
  EXO_DSV4_MTP=1 \
  EXO_SPECULATIVE=1 \
  EXO_SPECULATIVE_GAMMA=3 \
  EXO_DSV4_MTP_PROFILE=50 \
  EXO_DSV4_RB_PROFILE=1 \
  EXO_DSV4_LMHEAD_MXFP8=1 \
  EXO_DSV4_EXACT_TOPK_PREFILL=1 \
  EXO_DSV4_QUERY_TILED_SDPA=1 \
  EXO_DSV4_VERIFY_BATCH=1 \
  EXO_DSV4_VERIFY_BATCH_MIN_CTX=8192 \
  EXO_DSV4_SPEC_STATE_RESTORE=1 \
  EXO_DSV4_SPEC_CACHE_ROLLBACK=1 \
  EXO_DSV4_SPEC_CACHE_ROLLBACK_C2=1 \
  EXO_DSV4_ROWSEQ_ROWMASK=1 \
  ./start_cluster.sh
```

**Env-var provenance (all verified against live code):**

| Env var | Value | Why | Source |
|---|---|---|---|
| `EXO_DSV4_MTP_PROFILE` | `50` | **Profiling var.** Aggregation interval in cycles; `>0` creates the phase timer. **See justification below.** | dsv4_mtp.py:242 |
| `EXO_DSV4_RB_PROFILE` | `1` | **Profiling var.** Enables the `rb_snap`/`rb_*` sub-phase brackets. **Both profiling vars are required** to get `rb_snap` output (`_rbp = _RB_PROFILE and prof is not None`, dsv4_mtp.py:4111). | dsv4_mtp.py:258 |
| `DSV4_KV_CACHE_BITS` | `0` | User rule: bf16 KV always for DSv4 (script defaults to 4-bit). Matches the ~74 ms baseline. | skill: canonical invocation |
| `EXO_DSV4_MTP` | `1` | MTP ON (script default) — the V3 hypothesis is about the MTP verify cycle. | skill: script default |
| `EXO_SPECULATIVE` | `1` | Speculation ON (script default), paired with MTP=1. | skill: script default |
| `EXO_SPECULATIVE_GAMMA` | `3` | γ=3 → `_verify_len = γ+1 = 4`; the snapshot path is live at this config. | start_cluster.sh:176 |
| `EXO_DSV4_LMHEAD_MXFP8` | `1` | Shipped baseline. | start_cluster.sh:599 |
| `EXO_DSV4_EXACT_TOPK_PREFILL` | `1` | Shipped baseline. | start_cluster.sh:40 |
| `EXO_DSV4_QUERY_TILED_SDPA` | `1` | Shipped baseline. | start_cluster.sh:48 |
| `EXO_DSV4_VERIFY_BATCH` | `1` | Shipped baseline. | start_cluster.sh:335 |
| `EXO_DSV4_VERIFY_BATCH_MIN_CTX` | `8192` | Shipped baseline depth gate. | start_cluster.sh:341 |
| `EXO_DSV4_SPEC_STATE_RESTORE` | `1` | **Snapshot path is live** (production default ON). This is what makes the per-cycle snapshot real. | start_cluster.sh:246 |
| `EXO_DSV4_SPEC_CACHE_ROLLBACK` | `1` | Shipped default. | start_cluster.sh:247 |
| `EXO_DSV4_SPEC_CACHE_ROLLBACK_C2` | `1` | Shipped default. | start_cluster.sh:248 |
| `EXO_DSV4_ROWSEQ_ROWMASK` | `1` | Shipped default. | start_cluster.sh:245 |

**Why `EXO_DSV4_MTP_PROFILE=50`:** it is the aggregation interval in cycles (dsv4_mtp.py:242, 803 — the timer dumps every `_PROFILE_INTERVAL` cycles). At ~74 ms/cycle, 50 cycles ≈ **3.7 s per dump**. Each dump emits ~14 lines (6 known phases + 8 `rb_*` sub-phases) via `logger.warning`. Over a 256-token decode (~116 cycles/rep, see §3) that is ~2 dumps/rep; over 5 reps ≈ **10 dumps ≈ 500 `rb_snap` samples** (n=50 per dump) — enough for a stable mean/min/max and run-to-run variance — while producing only ~140 log lines total, so it does not flood `~/exo.log`. A smaller interval (e.g. 10) would give finer granularity but ~5× the log volume and more `logger.warning` overhead per cycle; a larger interval (e.g. 200) would yield too few dumps over a short decode. `50` also matches the value already used in historical profiling and in CODE_VERIFICATION.md's example output.

**Post-launch readiness check** (wait for both runners ready, up to ~6 min):

```bash
curl -s -m 5 http://192.168.86.201:52415/state | python3 -c "import json,sys; d=json.load(sys.stdin); print('instances:', len(d.get('instances',{})), 'runners:', len(d.get('runners',{})))"
```

Healthy = `instances: 1`, `runners: 2`, both `RunnerReady`. If `RunnerStarting` persists >3 min, check `~/exo.log` on both nodes for placement errors.

---

## 3. WORKLOAD / MEASUREMENT PROTOCOL

**Context length:** ~100K tokens. The cycle-time figures (~74 ms) reference 100K context, and the verify-cycle cost scales with context (attention over pooled KV). Use the established 100K-context prompt size from the bench harness convention: `--prompt-words 75000` (the skill's documented 100K-context bench uses exactly this).

**Bench harness:** `concurrent_bench.py` (fires chat completions against the already-deployed model; reports per-iteration `generation_tps`, `prompt_tps`, `wall_s`). It uses the `/bench` endpoint and requires the model already placed (it is, by `start_cluster.sh`).

**Reps / duration:** `--iterations 5 --warmup 1 --max-tokens 256`. Each iteration = one full 100K prefill (~5 min at ~340 tok/s) + a 256-token decode (~116 verify cycles at ~74 ms/cycle ≈ 8 s). 5 iterations ≈ **~30 min total**. The MTP-PROF data comes from the **decode** phase only (the verify cycle does not run during prefill), so the 256-token decode per rep is what feeds the gate.

**Warmup:** `--warmup 1` runs 1 warmup iteration that the harness **excludes from its stats** (concurrent_bench.py:195). The warmup is **NOT included in the measurement** — deliberately: the first request after a relaunch hits cold MLX compile/allocator state and would pollute the phase-share data. The warmup absorbs that cold-start so the 5 measured reps reflect steady-state.

**What to record per rep:** (a) the bench JSON's per-iteration `generation_tps`/`prompt_tps`/`wall_s` (sanity context), and (b) the `[MTP-PROF]` dumps from `~/exo.log` on the node running the MTP loop — specifically the `rb_snap` and `total` lines (see §5). Record the rep index alongside each dump so run-to-run variance is visible.

**Bench launch** (run on the laptop, backgrounded; first find the API master — either node may have won the bully election):

```bash
# 1) Find the API master (whichever node responds to /state)
API_HOST=""
for h in 192.168.86.201 192.168.86.202; do
  if curl -s -m 5 "http://$h:52415/state" | grep -q '"instances"'; then API_HOST=$h; break; fi
done
echo "API master: $API_HOST"

# 2) Launch the bench in the background (writes JSON + stderr log locally)
mkdir -p /Users/adam.durham/repos/exo/tmp/research-v1v2v3-20260901/v3/run1
cd /Users/adam.durham/repos/exo && \
nohup uv run python3 bench/concurrent_bench.py \
  --host "$API_HOST" --port 52415 \
  --model mlx-community/DeepSeek-V4-Flash-8bit \
  --concurrency 1 --iterations 5 --warmup 1 \
  --max-tokens 256 --prompt-words 75000 \
  --timeout 1800 \
  --json-out /Users/adam.durham/repos/exo/tmp/research-v1v2v3-20260901/v3/run1/bench_v3_100k.json \
  --label v3-profiling-100k \
  > /Users/adam.durham/repos/exo/tmp/research-v1v2v3-20260901/v3/run1/bench_v3_100k.log 2>&1 &
```

Poll completion (do not start a new poll cycle once the process is dead):

```bash
while pgrep -f concurrent_bench.py >/dev/null; do sleep 30; done; echo "bench done"
```

---

## 4. WHAT TO COLLECT

**Artifacts and node-side paths:**

| Artifact | Node-side path | Which node(s) | Why |
|---|---|---|---|
| Runner log (contains `[MTP-PROF]` lines) | `~/exo.log` | **both** `macstudio-m4-1` and `macstudio-m4-2` | The MTP loop runs on whichever rank executes `_speculative_next`; pull both so the `rb_snap` lines are captured regardless. Emitted via `logger.warning` → `>> ~/exo.log 2>&1` (start_cluster.sh:2771). |
| Bench JSON | `.../v3/run1/bench_v3_100k.json` (local) | laptop | Per-iteration throughput sanity context. |
| Bench stderr log | `.../v3/run1/bench_v3_100k.log` (local) | laptop | The bench's `2>&1` stream; captures any bench-side errors. |

**Read-only collection commands** (run from the laptop; `scp`/`ssh` are read-only):

```bash
mkdir -p /Users/adam.durham/repos/exo/tmp/research-v1v2v3-20260901/v3/run1
scp macstudio-m4-1:~/exo.log /Users/adam.durham/repos/exo/tmp/research-v1v2v3-20260901/v3/run1/exo_m4-1.log
scp macstudio-m4-2:~/exo.log /Users/adam.durham/repos/exo/tmp/research-v1v2v3-20260901/v3/run1/exo_m4-2.log
```

**Fallback:** if either `~/exo.log` is empty or missing, check the alternate log path `~/.exo/exo_log/exo.log` on that node (a documented alternate location) and pull that instead. The authoritative path per the launch script is `~/exo.log` (start_cluster.sh:2771).

**Suggested local destination dir:** `/Users/adam.durham/repos/exo/tmp/research-v1v2v3-20260901/v3/run1/` (created above).

---

## 5. HOW TO ANALYZE

**Fields to extract** (all in the `[MTP-PROF]` dump, units ms, format `[MTP-PROF]   B={b} {phase} mean=..ms min=..ms max=..ms n=..`):
- `rb_snap` — the snapshot/arm block wall time (the gate's numerator).
- `total` — the full verify-cycle wall time (the gate's denominator).
- (Optional context: `draft`, `verify`, `accept`, `commit`, `rollback`, and the other `rb_*` sub-phases.)

**Arithmetic:** `share = rb_snap_mean / total_mean` (as a fraction; ×100 for %). Compare against the gate in §1.

**Extract commands:**

```bash
cd /Users/adam.durham/repos/exo/tmp/research-v1v2v3-20260901/v3/run1
# All MTP-PROF lines, both nodes, into one file
grep -h '\[MTP-PROF\]' exo_m4-1.log exo_m4-2.log > mtp_prof_all.txt
# rb_snap lines (the gate numerator)
grep 'rb_snap' mtp_prof_all.txt
# total lines (the gate denominator)
grep 'total' mtp_prof_all.txt
```

**Compute the share (exact parse + arithmetic):**

```bash
cd /Users/adam.durham/repos/exo/tmp/research-v1v2v3-20260901/v3/run1 && python3 - <<'PY'
import re, statistics
rb, tot = [], []
for f in ("exo_m4-1.log", "exo_m4-2.log"):
    for line in open(f):
        if "[MTP-PROF]" not in line:
            continue
        m = re.search(r"rb_snap\s+mean=\s*([0-9.]+)ms", line)
        if m: rb.append(float(m.group(1)))
        m = re.search(r"\btotal\s+mean=\s*([0-9.]+)ms", line)
        if m: tot.append(float(m.group(1)))
rb_mean = statistics.mean(rb); tot_mean = statistics.mean(tot)
share = 100.0 * rb_mean / tot_mean
print(f"rb_snap mean = {rb_mean:.2f} ms   (n={len(rb)} dumps)")
print(f"total  mean = {tot_mean:.2f} ms   (n={len(tot)} dumps)")
print(f"SHARE = {share:.2f}%  of cycle time")
print(f"GATE: rb_snap < 1ms -> CLOSE | 1ms<=rb_snap<3% -> DEAD ZONE (default CLOSE) | >=3% -> PROCEED")
PY
```

**Apply the gate** using the printed `SHARE` and `rb_snap mean` against §1. Record the verdict verbatim in the run notes.

---

## 6. PERTURBATION HANDLING

**The profiler perturbs its own measurement — stated plainly.** Enabling `EXO_DSV4_MTP_PROFILE` and `EXO_DSV4_RB_PROFILE` inserts `mx.eval`/`mx.synchronize()` calls at phase boundaries that **serialize the pipelined draft/verify/accept** (dsv4_mtp.py:238–241, 251–253). The `rb_snap` bracket itself is wrapped in `mx.synchronize()` at both ends (dsv4_mtp.py:4114, 4142), forcing the snapshot's `mx.array()` copies to complete on device. **Therefore `rb_snap` is an UPPER BOUND on the true production snapshot cost, and every absolute ms is inflated.**

**Why the analysis stays valid anyway (shares vs absolutes):** the profiler inflates *every* phase's absolute ms roughly proportionally (it serializes the whole cycle, not just the snapshot). The **share** `rb_snap/total` is therefore robust to that uniform inflation and is the trustworthy signal. The gate is deliberately expressed as a share (§1). This is the correct conservative direction: if `rb_snap` is small even as a share of the *inflated* cycle, the hypothesis is genuinely weak.

**Sanity check — quantify the profiler's own cost:** compare the profiled `total_mean` (from §5) against the known **unprofiled baseline ~74 ms @100K**:

```bash
cd /Users/adam.durham/repos/exo/tmp/research-v1v2v3-20260901/v3/run1 && python3 - <<'PY'
import re, statistics
tot = []
for f in ("exo_m4-1.log", "exo_m4-2.log"):
    for line in open(f):
        m = re.search(r"\btotal\s+mean=\s*([0-9.]+)ms", line)
        if m: tot.append(float(m.group(1)))
profiled = statistics.mean(tot)
baseline = 74.0  # known unprofiled cycle time @100K
print(f"profiled total mean = {profiled:.2f} ms")
print(f"unprofiled baseline = {baseline:.2f} ms")
print(f"profiler inflation factor = {profiled/baseline:.2f}x")
PY
```

**Interpretation:**
- **Inflation < 1.2×** → the profiler is cheap; both the share AND the absolute ms are trustworthy. Apply the gate as-is.
- **Inflation 1.2×–2×** → the profiler is moderately expensive; absolute ms is inflated but the share is still meaningful. Apply the gate on the share only; do not trust absolute ms.
- **Inflation > 2×** → the profiler is expensive; the absolute ms is unreliable. The share is still directionally valid, but if the share lands in the **dead zone**, the measurement is too perturbed to trust — **do not decide from this run**; re-run with a coarser interval (e.g. `EXO_DSV4_MTP_PROFILE=200`) or accept the run as inconclusive and escalate.

**If the sanity check invalidates the gate:** if the profiler inflation is so large that the share itself is suspect (inflation > 2× AND dead-zone outcome), treat the run as **inconclusive** — do not proceed and do not close. Re-run with a coarser aggregation interval, or escalate to a second opinion before any decision.

---

## 7. DECISION TREE

| Gate outcome | Concrete next action |
|---|---|
| **PROCEED** (`rb_snap ≥ 3% of cycle`) | Design the copy-on-write fix. **Honest scope note:** per the corrected Claim-6 finding (see §8), there is **no O(1) ring path to mirror** — the ring path copies keys/values via `mx.array()` (cache.py:777–784, 832–837), and the comment at dsv4_mtp.py:692–694 claiming "O(1) reference snapshot" is **wrong**. Both rings **and** pools copy. So the fix must invent copy-on-write for **BOTH**, which is **larger than originally scoped**. This is a new, separate task requiring its own design + approval — do not fold it into this run. |
| **CLOSE** (`rb_snap < 1 ms`) | Close the V3 hypothesis as **allocator noise**. No code change. Optionally note the documented pool ratchet (1.95 → 2.27 GB) as a separate memory-residency concern to triage independently, but do not act on it here. |
| **DEAD ZONE** (`1 ms ≤ rb_snap < 3%`) | **Default CLOSE** — do not proceed (rule in §1). Exception: if the §6 sanity check shows profiler inflation < 20%, escalate to a second opinion before deciding (a real 1–2 ms cost might warrant a targeted pool-only investigation). |

---

## 8. RISK / BLAST RADIUS

**Cost:** one cluster relaunch (~6–8 min) + one ~30-min 100K benchmark. **Disturbance:** any live sessions on both nodes are torn down by the relaunch; the cluster is unavailable for the duration of the run.

**Measurement-only — does NOT change model output correctness.** Setting `EXO_DSV4_MTP_PROFILE` and `EXO_DSV4_RB_PROFILE` only gates the `_phase_timer` and the `_RB_PROFILE` timing brackets (dsv4_mtp.py:242, 258); they insert `mx.eval`/`mx.synchronize()` for timing but do **not** alter the compute graph, sampling, or any tensor value (CODE_VERIFICATION.md Q4). Critically, the snapshot path itself (`save_meta`/`save_spec_state`) is **already live in production** — `EXO_DSV4_SPEC_STATE_RESTORE=1` is the shipped default (start_cluster.sh:246) — so enabling the profiler does not change *whether* snapshots happen; it only times them. The run measures the existing production behavior; it does not modify it.

**Corrected Claim-6 finding (implications for this plan):** the original V3 fix premise — "mirror the O(1) ring path for pools" — is **false**. The ring path does **not** take O(1) references: `save_spec_state` copies keys/values via `mx.array()` (cache.py:777–784, 832–837), and the docstring states a bare reference would NOT preserve pre-write contents because `mx.__setitem__` mutates in place. The comment at dsv4_mtp.py:692–694 claiming an "O(1) reference snapshot" is a stale-comment-as-fact trap. **Consequence:** this profiling run measures the **TOTAL** snapshot cost (rings + pools, ~5.5 MB + ~5.5 MB ≈ 11 MB/cycle per CODE_VERIFICATION.md), and a future fix must invent copy-on-write for **both** rings and pools — a larger change than originally scoped. The pool count is **62**, not the ~41 a stale comment claims (dsv4_mtp.py:4107; actual count from deepseek_v4.py:7331–7353).

---

## 9. EXECUTION CHECKLIST (for the executor)

1. [ ] **Get explicit user approval** for the cluster relaunch. Do not proceed without it.
2. [ ] Run the §2 launch command. Wait for readiness (§2 check).
3. [ ] Run the §3 bench (find API master, launch backgrounded, poll to completion).
4. [ ] Run the §4 collection commands (pull `~/exo.log` from both nodes + bench artifacts).
5. [ ] Run the §5 analysis (extract, compute share, apply gate).
6. [ ] Run the §6 sanity check (profiler inflation vs 74 ms baseline).
7. [ ] Record the gate verdict + sanity-check result in the run notes; follow the §7 decision tree.
