# CAMPAIGN 2, ROUND 7 — the two LIVE c=2-tax knobs (I2)

**Date:** 2026-09-04 | **PM:** round-7 delegation
**Commits (local, UNPUSHED):** `dcb2ec162` (Task C comment corrections), `b9c91fbe0`
(pre-registration + read-throughs, before any measurement), `7e68ecbc6` (Amendment 1 + arm P,
before the arm-Z relaunch)
**Cluster end state:** production config, PM-verified on real runner PIDs both nodes (§7).

## OUTCOME: **NOTHING SHIPS. Both knobs HOLD — for opposite reasons.**

| task | knob | verdict | why |
|---|---|---|---|
| **A** | `EXO_BATCHED_PREFILL_RENDEZVOUS_MS` 200→0 | **HOLD** | rank-agreement is provably safe, and the sleep is real (−480 ms on the high-resolution instrument) — but the **pre-registered A1 delta was +120 ms, outside the [−250,−150] band.** The brief says outside band → report, do not ship. |
| **B** | `MLX_STEEL_BATCH_INVARIANT` 1→0 | **HOLD — correctness-load-bearing at c=1** | **byte-identity FAILED at BOTH context regimes.** Speed never measured; pre-registered stopping rule forbade it. |

**The headline finding is Task B.** This knob is not a c≥2 concern that we happen to pay for at
c=1. At the production configuration it is **load-bearing for c=1 correctness**, and the
`start_cluster.sh` comment framing it as a c≥2 prerequisite was actively misleading. It must
stay `=1` regardless of any throughput number.

---

## 1. TASK A — the four read-through questions, answered with cites

Full write-up: `taskA-readthrough.md`. Verified by the PM against source, not taken on report.

**(a) It is a wait for ADDITIONAL concurrent tasks.** After the first task is submitted
(`runner.py:566,569`), a drain loop pulls further `GenerationTask`s off `_work_queue` via
`queue.get(timeout=remaining)` until a deadline (`runner.py:580-604`), before `generator.step()`
(`runner.py:647`). Its purpose is to let a second request land in the same `step()` iteration so
the batched-prefill gate sees `len(queue) >= 2`.

**(b) At c=1 it is pure added latency.** The loop's only outcome with one task in flight is a
blocking `queue.Empty` timeout. Nothing downstream reads elapsed rendezvous time — no warmup, no
cache priming, no cross-rank sync keyed to the window.

**(c) 0 is a clean skip, NOT a sentinel.** The guard is a plain
`if EXO_DSV4_BATCHED_PREFILL and EXO_BATCHED_PREFILL_RENDEZVOUS_MS > 0` (`runner.py:580`). At 0
the entire block is skipped and control falls through to the main step loop. No branch anywhere
treats 0 specially. (`constants.py:135-140` reads it once at import; changing it needs a relaunch.)

**(d) 0 CANNOT cause rank disagreement at c=1 — the mechanism, not just the conclusion.**
`agree_on_tasks` (`batch_generator.py:159-185`, `505-535`) agrees via a **synchronous**
`mx.distributed.all_gather` of each rank's serialized task-id list followed by a **set
intersection** (`utils_mlx.py:2284-2338`). Both ranks compute the identical intersection from the
identical gathered data, so they cannot arrive at different task sets. The window never feeds into
that computation — it only affects *whether a second task is already queued* when the gather runs.
At c=1 there is one task id; the worst window=0 can do is defer that task to the next `step()`'s
agreement round.

The launcher's 50/100 ms warning (`constants.py:125-134`) describes a **throughput** failure, not a
correctness one: the ranks rendezvoused on different iterations, so `len(queue) >= 2` failed and
batching silently never fired. That failure class **requires ≥2 tasks racing the window** and
therefore cannot occur at c=1. **Question (d) is clean.**

### 1.1 Is the 200 ms actually being paid? — checked, yes
Worth confirming rather than assuming, since a dead outer gate would have made Task A
unmeasurable: `EXO_DSV4_BATCHED_PREFILL=1` (`start_cluster.sh:129`, propagated at `:1696-1697`) and
`EXO_MAX_CONCURRENT_REQUESTS=8` (`constants.py:108`, unset in the runner env so the default holds),
so both loop conditions at `runner.py:580,596` are live.

---

## 2. TASK A — TTFT measurement

### 2.1 Why a second instrument was added BEFORE the comparison arm ran
Arm P (production, 200 ms) came back with **median 7900 ms, range [7660, 8030], width 370 ms** —
against a **200 ms** effect and a band only **100 ms wide**. The noise was 1.85× the effect. That
was visible from arm P alone, so `PRE-REGISTRATION-AMENDMENT-1.md` was committed (`7e68ecbc6`)
**before the arm-Z relaunch**, adding a supplementary high-resolution instrument and — critically —
fixing in advance which instrument governs the ship decision and how disagreement resolves.

- **A1** (2K prompt, n=5, `prefill_s`): the pre-registered instrument. **Governs the ship call.**
- **A2** (~20-token prompt, n=10): diagnostic only. The rendezvous is prompt-length-independent
  (it runs before any prefill work), so a short prompt isolates the constant from prefill variance.

### 2.2 A1 — the pre-registered instrument (RANGES, never bare means)

| arm | boot | reps (ms) | **median** | **range** | width |
|---|---|---|---|---|---|
| **P** (200 ms) | round-6 boot | 7800, 7660, 8030, 7940, 7900 | **7900** | [7660, 8030] | 370 |
| **Z** (0 ms) | boot 2 | 8470, 6670, 7720, 8330, 8020 | **8020** | [6670, 8470] | **1800** |
| **P2** (200 ms) | boot 4 (restore) | 7800, 8290, 7410, 8440, 8010 | **8010** | [7410, 8440] | 1030 |

All 15 reps: `prefix_cache_hit = none`, `prompt_tokens` 2216–2377.

**A1 delta = median(Z) − median(P) = 8020 − 7900 = +120 ms.** Outside the pre-registered
`[−250, −150]` band, and the **wrong sign**.

The closing bracket P2 (8010) is the reason this is not read as "removing the sleep made it
slower": the two 200 ms arms are 7900 and 8010, and the 0 ms arm sits at 8020 — **all three within
120 ms of each other, while individual arm ranges span 370–1800 ms.** A1's between-arm differences
are entirely inside its own noise. The instrument cannot see a 200 ms effect. Note also that arm
Z's width (1800 ms) is ~5× arm P's — arm P was measured on a 4.5-hour-old boot, arms Z and P2
immediately after relaunch, so post-boot warmup inflates the fresh-boot arms.

### 2.3 A2 — the high-resolution instrument (diagnostic, matched ordering)
Both arms ran 5× 2K reps first, then 10× short reps, so process warm-state is structurally matched.

| arm | reps (ms) | **median** | **range** |
|---|---|---|---|
| **Z** (0 ms) | 1760,1460,1410,1430,1580,1710,1420,1620,1560,1450 | **1510** | [1410, 1760] |
| **P2** (200 ms) | 2140,1860,1730,1940,1780,2060,1930,2040,2150,3980 | **1990** | [1730, 3980] |

**A2 delta = 1510 − 1990 = −480 ms.** Excluding P2's one 3980 ms outlier makes no material
difference (−430 ms). The **ranges do not overlap** (Z max 1760 < P2 min 1730 is marginal at the
boundary; on medians the separation is clean and every Z rep but one sits below every P2 rep).

**Reading:** the rendezvous sleep is real and removing it does reduce TTFT — but by ~480 ms on this
instrument, **not the ~200 ms the code implies.** That overshoot is unexplained. A deterministic
200 ms sleep should produce a 200 ms delta. Candidate explanations (none verified, none claimed):
the window may interact with per-request scheduling beyond the sleep itself, or short-prompt TTFT
carries a warmup term that differs between arms. **I am not shipping on a number I cannot explain.**

### 2.4 Clean logs — the independent veto (PASSED)
20+ sequential requests on the RV=0 boot (5×2K + 10×short + 4 long + identity captures), mixed
short and ~89K. Both nodes' runner logs and the launch log: **zero errors, zero tracebacks, zero
rank-disagreement or task-set-mismatch evidence.** The only grep hits were build-artifact filenames
(`error.svelte.js`). Consistent with the (d) analysis: no correctness problem was observed.

### 2.5 Task A verdict: **HOLD, do not ship**
The pre-registered rule is explicit — *"Delta outside that band → something else is in the path;
report, do not ship."* A1 delivered **+120 ms**, outside the band. Amendment 1 pre-committed this
exact case: A1 outside band + A2 clean → **HOLD**, because this round will not ship on an
instrument it introduced to itself mid-round.

**This is not "the knob is worthless."** The evidence says the sleep is real, is safe to remove at
c=1 (§1d), and costs somewhere in the 200–480 ms range. It is a strong candidate that **failed on
measurement design, not on merit.** Recommended follow-up in §8.

---

## 3. TASK B — kernel scope and ROWSEQ_VEC liveness

Full write-up: `taskB-readthrough.md`; call sites and guards PM-verified against source.

**Kernel scope.** `steel_batch_invariant_enabled()` (`mlx/mlx/backend/metal/matmul.cpp:107`) has
four call sites: tile selection (`:144-146`), two split-K gates (`:963-964`, `:987-989`), and a
batch-into-M collapse (`:1511-1522`). The dispatch guard `min(M,N)==1 → gemv` (`:1554`) means
ordinary large-N linear GEMMs at M=1 (draft) bypass steel entirely, while M=4 (γ=3 verify) reaches
it. **But the attention-output (P·V) matmul collapses per-head batch into M regardless of decode
row count**, so it hits the split-K gates at **both** M=1 and M=4. The "~5% c=1 decode" claim is
therefore **live, not stale** — the flag does affect the c=1 decode path.

**ROWSEQ_VEC liveness at production depth: DEAD at ≥8192, LIVE below.** At ~89K with
`EXO_DSV4_VERIFY_BATCH=1`, `_vb_active` is True (ctx ≥ `_VERIFY_BATCH_MIN_CTX`, default 8192,
`deepseek_v4.py:1708,7094-7102`), which sets `_VERIFY_BATCH_CTX["active"]=True`. Every
rowseq-family branch is guarded by `not (_VERIFY_BATCH and _VERIFY_BATCH_CTX["active"])`
(`:5295`, `:5442`), so `rowseq_vec()` is **never reached at 89K**. Below 8192 it is live.

**Consequence for the test design:** the comment-only constraint *"Set =0 only with
EXO_DSV4_VERIFY_ROWSEQ_VEC=0"* is **moot at ≥8192** but **still binds below it** — which is exactly
why the identity gate had to cover a <8192 prompt. It did.

---

## 4. TASK B — the byte-identity gate: **FAILED at both regimes**

Pre-registered as a stopping rule: identity runs **before** decode, and a failure stops Task B.

Reference captured on the steel-BI=1 boot; candidate on the steel-BI=0 boot. Prompts made
reproducible via the additive `--run-id` flag, so both sides sent **byte-identical prompts**
(verified: same `run_id`, same `prompt_tokens`). Temperature 0 throughout.

| # | gate | prompt_tokens (both sides) | result |
|---|---|---|---|
| 1 | `identity_gate.py --compare` (5 fixed prompts) | — | **FAIL — 2/5 differ** (`sys_primary_colors`, `sys_count_to_five`) |
| 2 | **<8192** probe (`--run-id r7fixed4k`) | 3811 = 3811 | **FAIL — diverges at char 8** |
| 3 | **~89K** probe (`--run-id r7fixed89k`) | 74455 = 74455 | **FAIL — diverges at char 442** |

PM-verified diff (not taken from the worker's report):

```
4K  (<8192)  BI=1: "We need [to answer two tasks. First, state the se...]"
             BI=0: "We need [answer user. Need parse document. Need s...]"
89K (>8192)  BI=1: "...O(log n) [average/worst, space O(n), node branchin...]"
             BI=0: "...O(log n) [time, space O(n), disk I/O optimized, hi...]"
```

**The 89K failure is the significant one.** ROWSEQ_VEC is dead at that depth (§3), so this failure
**cannot** be attributed to the documented rowseq/steel-BI interaction. Turning off steel batch
invariance changes temp=0 output through the batched verify path itself — the M=4 verify rows stop
being bitwise equivalent to sequential decode. That is precisely the property the knob exists to
guarantee, and it is load-bearing **at c=1**, not merely at c≥2.

### 4.1 Decode was NOT measured — deliberately
Per §3.4 of the pre-registration, identity failure stops Task B: the three decode reps on the BI=0
boot were not run and the closing-bracket boot was not spent. Measuring the speed of a
configuration that already failed its correctness gate would burn ~30 minutes to decorate a
decision already made. **The "~5% c=1 decode" claim remains unverified, and is now moot** — the
knob cannot be flipped at any speed.

### 4.2 Opening bracket (recorded, incomplete by design)
The steel-BI=1 arm A did run before the gate: `server_generation_tps` = 29.729, 31.608, 30.779 →
median **30.779**, range **[29.729, 31.608]** (all reps valid: trustworthy, `prompt_tokens`
85086–90402, `finish_reason=length`, `prefix_cache_hit=none`). **A single bracket half proves
nothing on its own** and no comparison is drawn from it. Recorded only so the boot isn't wasted.

### 4.3 Scoring the pre-registered prediction
Amendment §3.5 predicted: *"byte-identity FAILS on the <8192 prompt and PASSES at ~89K."*
**HALF RIGHT — and the wrong half matters.** The <8192 failure was predicted. The **89K failure was
not**, and it is the more consequential result: it means the problem is not the known
rowseq/steel-BI interaction but something broader in the batched verify path. Recording plainly:
I expected the ≥8192 regime to be safe, and it is not.

---

## 5. TASK C — stale comment corrections (commit `dcb2ec162`)

Comment-only; **verified 0 non-comment changed lines**, `bash -n` exit 0, all knob values intact
(`RENDEZVOUS_MS:=200`, `STEEL_BATCH_INVARIANT:=1`, `FENCE_EVERY_N_LAYERS:=4`, `MTP_C2_MAX_CTX:=1`).
42 insertions, 12 deletions, 1 file.

| # | knob | correction |
|---|---|---|
| 1 | `FUSED_MOE`/`COMPILE_FFN`/`COMPILE_LAYER` | marked **no-ops** — wiring removed 2026-06-18 (`auto_parallel.py:110-119`); "set =1 to re-enable" was false. Historical perf numbers labelled historical. |
| 2 | `EXO_DSV4_MTP_C2_MAX_CTX` | gate removed 2026-06-24 (`dsv4_mtp.py:2371-2385`); env var is backward-compat only, no default-threshold effect. |
| 4 | `MLX_STEEL_BATCH_INVARIANT` | added the note that the knob is GEMM bit-exactness and potentially c=1-load-bearing, requiring a byte-identity gate. **§4 has since proven this stronger than "potentially."** |

### 5.1 Item 3 NOT written — a prior round's claim is wrong
The brief states `FENCE_EVERY_N_LAYERS` is dead ("zero readers since the OPT-7 revert"), the
"4th dead knob" from round 4. **That is false.** `mlx-lm/mlx_lm/models/deepseek_v4.py:2959` reads it
live inside `DeepseekV4MoE.__init__` to set `self._fence_every_n` — the active cross-rank fence
cadence. Per the pre-registered honesty rule, the correction was **not** written and the comment
left untouched, rather than propagating a false "this knob is dead" note into the launcher.

**Round 4's dead-knob count is 3, not 4.** The FENCE knob is live and its ~0.7 t/s cost is
presumably still being paid — an untested c=1 lever, not a closed one.

---

## 6. RECONCILIATION

**6a. Against round 1's I2 audit.** Row 6 predicted 0 "fully disables rendezvous (falls back to
per-task path)" — **confirmed** (`runner.py:580`). Row 2's steel-BI interaction constraint was
called "comment-only, not code-enforced" — **confirmed**, and §4 shows the consequence is worse
than documented: identity breaks even where the constraint is moot (89K).

**6b. Against round 4.** Its `FENCE_EVERY_N_LAYERS` finding is **refuted** (§5.1). Two of three
"live c=2-tax knobs" premises in this round's brief survived; the dead-knob tally did not.

**6c. Against round 6.** Its ruler (`stats.generation_tps`, warmup discarded, ranges never means)
was reused unmodified. Round 6's boot-variance lesson drove Amendment 1: arm P vs P2 (7900 vs 8010
on identical config) reproduces boot-to-boot movement of the same order round 6 measured for γ=3
(1.297 t/s), and is why a single-arm A1 comparison was never going to be trustworthy. Round 6's one
open item — an unverified coherent completion — is **closed here** (§7).

**6d. Deviation from the brief, declared.** (i) Amendment 1 added the A2 instrument, committed
before the comparison arm ran; A1 still governs the ship call, so the amendment could not have
manufactured a ship. (ii) `--run-id` was added to `long_decode_probe.py` (additive, default
behavior bit-identical, no timing math touched) because byte-identity at 89K is impossible against
a uuid-salted prompt. (iii) Task C item 3 not written (§5.1). (iv) The brief's
`bench/ab_probe_tier1.py 7/7` gate could not be run: that script is a **single-request probe with
no pass/fail concept and a stale hardcoded `--model`** that 503s as-is. It has no "7/7". The worker
flagged this rather than fabricating a result. Moot in the end — the tier-1 gate applied only to a
Task B ship candidate, which the identity failure eliminated. **The "7/7" reference in the brief is
itself stale and should be corrected in the record.**

---

## 7. Cluster health (end state) — PM-verified on real PIDs

Restored to production defaults (no overrides but `DSV4_KV_CACHE_BITS=0`) and verified by the PM
directly, not from a worker's report:

```
.201 pid 9402   .202 pid 20117
  BOTH: MLX_STEEL_BATCH_INVARIANT=1   EXO_BATCHED_PREFILL_RENDEZVOUS_MS=200
        EXO_SPECULATIVE_GAMMA=3       EXO_DSV4_MTP=1   EXO_DSV4_VERIFY_ROWSEQ_VEC=1
API /v1/models  http=200
coherent temp=0 completion: content = "The capital of France is Paris."
```

The known-bad `MLX_STEEL_BATCH_INVARIANT=0` boot was torn down immediately after the gate failed.
**Round 6's outstanding completion check is closed.**

---

## 8. Recommendations for round 8

1. **Close `MLX_STEEL_BATCH_INVARIANT` permanently.** It is correctness-load-bearing at c=1 at both
   context regimes. It is not a c≥2 tax. Do not re-open it for throughput. Its "~5% c=1 decode"
   cost is the price of bitwise-correct output and should be recorded as **not recoverable**.
2. **`EXO_BATCHED_PREFILL_RENDEZVOUS_MS` is unfinished business, not a dead lever.** Safety at c=1
   is proven (§1d) and logs were clean over 20+ requests. What's missing is a trustworthy delta.
   Round 8 should measure it **with a paired-boot design** (both arms on fresh boots, ≥5 min idle
   after READY before the first rep, n≥10 short-prompt reps), pre-register the band on the
   short-prompt instrument, and explain the 200-vs-480 ms discrepancy before shipping.
3. **`EXO_DSV4_FENCE_EVERY_N_LAYERS` is live and untested** (§5.1). Its documented ~0.7 t/s c=1 cost
   for c≥2 stability is a genuine, unexamined c=2-tax lever — arguably the best remaining I2
   candidate now that steel-BI is closed.
4. **Fix the record:** the `ab_probe_tier1.py 7/7` gate does not exist as specified (§6d), and
   round 4's fourth-dead-knob claim is wrong (§5.1). Both will mislead the next round otherwise —
   the exact failure mode Task C exists to prevent.

## 9. Artifacts
- `PRE-REGISTRATION.md` — `b9c91fbe0`, before any measurement
- `PRE-REGISTRATION-AMENDMENT-1.md` — `7e68ecbc6`, before the arm-Z relaunch
- `taskA-readthrough.md`, `taskB-readthrough.md` — source read-throughs with cites
- `taskA-armP.md`, `armZ-and-biA.md`, `arm-bi0.md`, `restore-and-armP2.md` — per-boot records
- `results/` — raw JSONs: `P_ttft_r{1..5}`, `Z_ttft_r{1..5}`, `Z_short_r{1..10}`,
  `P2_ttft_r{1..5}`, `P2_short_r{1..10}`, `biA_{warmup,r1,r2,r3}`, `identity_{biA,bi0}`,
  `{biA,bi0}_id4k`, `{biA,bi0}_id89k`
- `bench/long_decode_probe.py` — +21/−4, additive `--run-id` only, no timing math touched
- **No pushes.** Local commits only.
