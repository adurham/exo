# V1 / V2 / V3 execution report — 2026-09-01

**Scope:** execute the zero-cluster-cost parts of the V1/V2/V3 workstreams and prepare the one
profiling run that needs approval. **No cluster relaunch, no process kill, no config change, no
production write, no boot experiment, and no commit was performed.** All node access was read-only
ssh for log inspection. V4 (c=2 concurrency) was dropped by user decision and was not pursued.

**Artifacts:** all under `/Users/adam.durham/repos/exo/tmp/research-v1v2v3-20260901/`. Nothing
committed — the supervisor commits later.

---

## 0. Headline

| Workstream | Outcome |
|---|---|
| **V2** (acceptance-rate as unmodeled state) | **CLOSED — NOT FEASIBLE, proven by positive evidence.** The acceptance counters for the P11–P15 window do not exist. Not "can't tell": *proven absent* via an exact log anchor. |
| **V1** (passive boot-state telemetry + log mining) | **DONE.** Log mining complete; three of four candidate mechanisms return count=0. Telemetry kit written and locally sanity-checked. **Untested on cluster by design.** |
| **V3** (per-cycle snapshot cost) | **PLAN READY, AWAITING APPROVAL.** Plus a **material correction**: the premise of the proposed fix was wrong (see §4.2). |

**The one action needing user approval: the V3 profiled relaunch** (§4.4).

---

## 1. V2 — feasibility verdict: NOT FEASIBLE (closed on evidence)

**Verdict: the per-cycle MTP acceptance counters for the P11–P15 boots do not exist.** This is a
positive finding, not a failure to look hard enough.

### 1.1 What exists

Per-cycle acceptance lines **do** exist on both nodes — 59,582 of them — but in the runner's
`stderr.log`, not in the timestamped `exo.log`. Verbatim shape:

```
[MTP] cycles=50 mean_accept=0.640/2 hist=0:22,1:24,2:4
```

Two properties matter: they carry **no timestamp**, and `cycles=` is a **resetting** counter
(range 1 → 13,450; `cycles=1` occurs 73 times), so the file is ~73 monotonic segments.

The repo parser `bench/mtp_cycle_time.py:32-33` cannot read these lines at all — its regex
requires a timestamp prefix that the real lines do not have:

```python
LINE = re.compile(
    r"(\d\d):(\d\d):(\d\d)\.(\d+).*?\[MTP\] cycles=(\d+)\s+mean_accept=([\d.]+)"
)
```

Match count against the real lines: **0**. (Format mismatch noted, not fixed — out of scope.)

### 1.2 The anchor that settles it

Segmentation alone is not attribution. The decisive evidence is that the supervisor echoes some
runner-stderr lines into the **timestamped** rotated `exo.log` as `Runner stderr: ...`. The
`[jaccl] tcp coord group` lines carry **unique ephemeral ports** appearing in *both* files — an
exact, auditable bridge rather than an ordinal guess.

| Boot | Timestamp (from timestamped exo.log) | Port | Line # in stderr.log |
|---|---|---|---|
| P13 | 2026-08-31 20:42:57.370 | 59227 | 14,042,187 |
| P14 Arm A | 2026-08-31 21:26:46.566 | 62804 | 14,042,212 |
| P14 Arm B | 2026-08-31 22:05:35.154 | 50069 | 14,042,237 |
| P15 | 2026-08-31 22:55:00.978 | 54795 | 14,042,262 |
| P15 post-restart | 2026-08-31 23:24:35.131 | 57750 | 14,042,287 |
| Sep 1 | 2026-09-01 00:10:21.909 | 61748 | 14,042,312 |

**The last `[MTP]` line in stderr.log is line 14,041,962 — 225 lines BEFORE the P13 boot marker.**
Zero `[MTP]` lines occur at or after the P13 boot, on either node.

**PM-verified independently** (not taken from the subagent): re-derived by direct read-only ssh to
`adams-mac-studio-m4-1.local` against `~/.exo/exo_log/runner_log/stderr.log`, reproducing both the
last-MTP line number and all six anchor line numbers exactly.

**Conclusion:** all 73 acceptance segments predate the P13–P15 campaign; they fall in the
un-retained Aug 21→31 window. P11/P12 logs are not retained at all. There is no acceptance data
for the campaign boots to regress against. **No regression was run, and none should be** — an
ordinal alignment would have produced a confident-looking phantom result.

### 1.3 Consequence for the campaign

The V2 hypothesis (acceptance ≈ 1.62 vs 2.26/cycle explaining ~3-4 tok/s of decode variance) is
**neither confirmed nor refuted — it is untestable from existing data.** It cannot be revived by
more mining. To make it testable in future, one line of logging must change:

- Emit `[MTP]` lines through the **timestamped logger** instead of bare stderr (highest value,
  smallest change), and/or add a per-rep marker; and retain rotated logs long enough to cover a
  campaign window.

That is a prerequisite for any future acceptance work, not a fix in itself.

*Detail:* `v2/V2_ANALYSIS.md`, `v2/parse_mtp_segments.py`, `v2/segments.csv`, `v2/raw/`,
`logmine/FEASIBILITY.md`.

---

## 2. V1 — log-mining findings (read-only, P13/P14/P15 windows)

Counts are from the decompressed, timestamped, boot-attributable rotated `.zst` logs on both nodes.

| Candidate mechanism | Result |
|---|---|
| Metal / IOGPU warnings or errors | **count = 0** in all 8 Aug 31 logs (both nodes, all 4 boots) |
| Jetsam / memory-pressure / low-memory / OOM | **count = 0** in all 8 Aug 31 logs |
| Runner restarts | **exactly 1 event, P15 only** — confirmed on both nodes |

The only memory-related lines are wired-limit INFO lines at model load (`Wired limit set to
112.30 GiB`), which are normal startup, not pressure events.

**The one restart is the known P15 crash, now confirmed verbatim rather than assumed:**

```
[ 2026-08-31 23:23:46.632 | WARNING | exo.worker.runner.bootstrap:entrypoint:368 ]
Runner ... crashed with critical exception [reshape] Cannot reshape array of size 1
```

`ValueError: [reshape] Cannot reshape array of size 1 into shape (2,1,1,1)` at
`mlx_lm/models/cache.py:2050` in `fetch_overlap_carry`, at 23:23:46 on both nodes, with auto-restart
at 23:23:47 (`CreateRunner`) → 23:23:53 (`LoadModel`). No crashes in P13/P14A/P14B.

### 2.1 What this does to the hypotheses

This **strengthens the within-boot process-state hypothesis and weakens the driver-leak
hypothesis**, but does not settle either:

- The driver-leak hypothesis predicted accumulating GPU/memory-subsystem distress. The logs show
  **none** — no Metal/IOGPU warnings, no memory pressure, across all four boots. That is a real
  negative result, though absence in logs is weaker evidence than a positive measurement.
- The single strongest datapoint remains the **within-boot** P15 step (rep1 34.71 → crash+restart →
  30.06 / 29.83). The restart is now confirmed as a real, logged process event at a known
  timestamp — consistent with process-level state (allocator pool re-creation) rather than
  boot-level driver state.

**Honest limit:** this is still correlational and n=1 for the restart. It does not establish
causation, and it does not explain the boot-level differences among P13/P14A/P14B, which had **no**
restarts and **no** logged anomalies. The telemetry kit (§3) exists to close exactly that gap.

*Detail:* `logmine/V1_LOG_FINDINGS.md`, `logmine/raw/` (incl. `SSH_COMMANDS_APPENDIX.md`, all
commands auditable as read-only).

### 2.2 Adjacent finding, outside scope

922 `[METAL]` GPU-timeout errors appear in the **tail** (non-Aug-31, un-attributable) region of
stderr.log, alongside `[DSPARK-SHADOW]` lines. These are **not** in the P13–P15 window and are
**not** evidence for the campaign question. Flagged only so it is not lost; needs its own triage.

---

## 3. V1 — passive telemetry kit (written; UNTESTED ON CLUSTER by design)

`telemetry/collect_telemetry.py` + `telemetry/TELEMETRY_KIT.md`.

Samples per node per invocation, taking a caller-supplied checkpoint label (`T0`, `warmup`, `rep1`
…) and appending one JSON object per line to a JSONL file:

- **(a)** MLX active memory + allocator cache size + peak — **detects both MLX API forms at runtime**
  (top-level `mx.*` vs deprecated `mx.metal.*`) rather than hardcoding a guess
- **(b)** powermetrics GPU active residency / frequency / power, ~400 ms window, parsed + raw retained
- **(c)** `memory_pressure` free % and page counts
- **(d)** `iogpu.wired_limit_mb` sysctl
- **(e)** runner **PID + lstart for all matching processes** — the restart detector

Design constraints met: read-only and non-perturbing; every sampler degrades gracefully (a failing
sampler records an error string and the checkpoint still emits); no third-party deps beyond stdlib +
mlx.

**PM-verified:** `python3 -m py_compile` → exit 0; all five sampler functions present; grep for
`ssh|sysctl -w|kill|pkill|screen -S|start_cluster` → **no matches** (confirms it cannot mutate node
state).

**Verification status — honest:**

| Sampler | Locally verified | How |
|---|---|---|
| (a) MLX memory | YES | real mlx 0.32.0.dev; values 0 at idle (non-zero not observed) |
| (b) powermetrics | **PARTIAL** | parser validated against sample text only; **not run live** (sudo would prompt on the laptop). Password-gated failure path *was* exercised |
| (c) memory_pressure | YES | real local output |
| (d) wired_limit | YES | real local output (0 locally; cluster's value not observed) |
| (e) runner PID/lstart | YES | real `ps` output + synthetic runner line |

**Not verified, needs the cluster:** live powermetrics on M4 Max, non-zero MLX values, real cluster
wired limit, the actual runner process signature, end-to-end on nodes.

A false-positive was caught and fixed during development: the runner pattern initially matched a
shell wrapper whose args contained the venv path, which would have corrupted restart detection. Now
anchored on a runtime marker. **Sampler (e) is the kit's most load-bearing field and its process
signature is the one thing still unvalidated against a real runner** — worth a single dry-run
checkpoint at T0 of the next launch before trusting a whole run's timeseries.

---

## 4. V3 — profiling run: prepared, NOT executed

### 4.1 Code verification (PM-spot-checked against live source)

| Claim | Verdict |
|---|---|
| rb_snap bracket at `dsv4_mtp.py:4111-4145` | **CONFIRMED** |
| `:4118-4128` snapshots all pools + rings every cycle | **CONFIRMED** (mechanism); **"~41 pools" is WRONG — actually 62**; the "41" is a stale comment at `:4107` |
| `BatchPoolingCache.save_meta` (`cache.py:2533-2568`) copies via `mx.array()` | **CONFIRMED** |
| Flush-predicting filter at `:4137-4140` useless at production config | **CONFIRMED** (dead in production) |
| `start_cluster.sh:246` defaults `EXO_DSV4_SPEC_STATE_RESTORE=1` (ON) | **CONFIRMED** — snapshot path is live in production |
| Ring path takes O(1) references at `dsv4_mtp.py:692-694` | **WRONG — see below** |

### 4.2 Material correction: the V3 fix premise was false

The proposed fix was "extend the ring path's O(1)-reference discipline to the pools." **There is no
O(1) ring path to extend.**

`dsv4_mtp.py:692-694` is a **comment**, not code. The code it describes says the opposite, verbatim
at `mlx-lm/mlx_lm/models/cache.py:783-784`:

> reference does NOT preserve pre-write contents — so keys/values are copied here (``mx.array``)

Both ring implementations copy: `save_spec_state` at `cache.py:777-837` (local ring) and
`cache.py:3327-3344` (batch ring), each via `mx.array()`.

**PM-verified personally** by reading both sites — this is exactly the stale-comment-propagated-as-
fact failure mode that has burned this campaign before, so it was not taken on the subagent's word.

**Consequences:** (1) the profiling run measures **total** snapshot cost (rings **+** pools), not
just pools; (2) any future fix must invent copy-on-write for **both**, which is **larger than
originally scoped**; (3) the stale comment at `dsv4_mtp.py:692-694` should be corrected at source so
the next reader does not repeat the error — **not done here** (no commits permitted this session).

### 4.3 The plan

`v3/V3_PROFILING_PLAN.md` — execution-ready, no placeholders. Key contents:

- **Pre-registered gate, stated before any method detail**, as a **share** of cycle time (robust to
  profiler perturbation), with all three outcomes defined:
  - **PROCEED** — `rb_snap_mean / total_mean ≥ 3%` (≈ ≥2.2 ms of a ~74 ms cycle @100K)
  - **CLOSE** — `rb_snap_mean < 1 ms` → allocator noise
  - **DEAD ZONE** — 1 ms ≤ value < 3% → **default CLOSE**, with one explicit escalation path
- **Launch:** `EXO_DSV4_MTP_PROFILE=50 EXO_DSV4_RB_PROFILE=1 ./start_cluster.sh` plus shipped
  baseline vars. Both gates are required for rb_snap output (`dsv4_mtp.py:242`, `:258`); env vars
  propagate as `VAR=VAL` prefixes on `python -m exo` inside screen (`start_cluster.sh:2771, 2823`).
  Interval 50 justified: ≈3.7 s/dump, ~500 samples over 5 reps without flooding the log.
- **Protocol:** 100K context, 5 reps + 1 **excluded** warmup, 256-token decode per rep.
- **Collection:** `~/exo.log` on both nodes (output goes via `logger.warning`, format
  `[MTP-PROF]   B=1 rb_snap    mean=..ms min=..ms max=..ms n=..`), read-only pull commands, local
  destination dir.
- **Perturbation handled, not just mentioned:** enabling the profiler inserts `mx.eval` /
  `mx.synchronize()` that serialize the pipeline, so **rb_snap is an upper bound**. Includes a
  sanity check comparing profiled `total` against the ~74 ms unprofiled baseline to quantify the
  profiler's own cost, and states what to do if that comparison invalidates the gate.
- **Blast radius:** one relaunch (~6–8 min, tears down live sessions) + ~30 min benchmark. Setting
  these vars is **measurement-only and does not change model output correctness**.

### 4.4 THE ONE ACTION REQUIRING USER APPROVAL

> **Approve a single profiled cluster relaunch** to execute `v3/V3_PROFILING_PLAN.md`.
> It requires `./start_cluster.sh` with the two profiling env vars, which tears down live sessions
> on both nodes. **Not executed. Awaiting explicit approval.**

Worth noting before approving: the gate now guards a **larger** fix than when V3 was scoped (§4.2).
The measurement is still cheap and decisive — it either kills the hypothesis or justifies the bigger
design — but the "proceed" branch is more expensive than the original sketch implied.

---

## 5. Blockers and honest limits

1. **V2 is permanently closed for existing data.** No blocker to route around — the data was never
   written to a timestamped log. Future testability needs the logging change in §1.3.
2. **Telemetry kit is unvalidated against a real runner process** (sampler (e), §3). Recommend one
   dry-run checkpoint at T0 of whatever launch happens next.
3. **P11/P12 logs are not retained**, so even the log-mining findings in §2 cover only P13–P15.
4. **§2 findings are correlational.** Three count=0 results weaken the driver-leak hypothesis;
   they do not refute it, and they do not explain the boot-level spread among the three boots that
   had no anomalies at all.
5. **Uncommitted by design.** All artifacts are untracked under `tmp/`; the supervisor commits.
6. One doc inconsistency noted during verification (two docs disagree on local-layer count, 2 vs 3);
   does not affect the 62-pool total. Flagged, not resolved.
