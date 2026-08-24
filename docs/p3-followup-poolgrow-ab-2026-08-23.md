# P3 follow-up — BatchPoolingCache chunked-growth A/B (`EXO_DSV4_POOL_GROW_STEP`) — 2026-08-23

**Status: LIVE A/B COMPLETE. VERDICT — CONFIRMED.**

`EXO_DSV4_POOL_GROW_STEP=256` is a **real, reproducible decode win**:
**+9.79% at 352.6K ctx** (23.50 → 25.80 tok/s, −3.79 ms/tok) and **+3.46% at
100K ctx** (28.09 → 29.06 tok/s, −1.19 ms/tok). The pre-registered
falsification condition was **not** met. The pre-registered deep≫shallow
asymmetry fingerprint is present, and the pre-registered *depth-delta* numbers
were hit almost exactly (arm A +6.95 vs predicted +6.80 ms/tok; arm B +4.35 vs
predicted +4.89). Output quality is unchanged (R2 control passed).

Results are in §10-§14. §3 preserves the pre-registered hypothesis verbatim, so
the run is judged against what was written before the data existed. The
original §5 (why the A/B was blocked on the first attempt) is kept for the
record.

**The code default was deliberately NOT changed.** The lever remains opt-in and
unset everywhere; flipping the default is a separate reviewed step (§14).

---

## 0. What was asked vs what happened

| step | status |
|---|---|
| 1. Code change (env-gated chunked grow) | **DONE** — `mlx-lm` `643d42d` |
| 2. Git (mlx-lm commit+push, exo submodule bump, uv.lock, push) | **DONE** — `exo` `8a04cf492` |
| 2b. *(unplanned, required)* env forwarding in `start_cluster.sh` | **DONE** — see §2 |
| 3. Deploy arm A (relaunch) | **DONE** — §10 |
| 4. Arm A probes (100,026 / 352,599) | **DONE** — §11 |
| 5. Deploy arm B + verify env reached runners | **DONE** — §10, env verified via `ps eww` on BOTH nodes |
| 6. Analysis vs pre-registered signature | **DONE** — §12, verdict CONFIRMED |
| 6b. R2 make_mask control, static half | **DONE** — §4, and it is a real finding |
| 6c. R2 make_mask control, live half (output quality) | **DONE** — §13, passed |
| 7. This doc | **DONE** |
| 7b. `PERFORMANCE_HISTORY.md` entry | **DONE** — real numbers, see §7 |

---

## 1. The code change

`mlx-lm/mlx_lm/models/cache.py`, `BatchPoolingCache.update_and_fetch_deferred`,
lines 1899-1903 → 1899-1907. Exactly the spec's patch, no deviation:

```python
            if self.pooled.shape[1] < max_pool:
                _grow_step = int(os.environ.get("EXO_DSV4_POOL_GROW_STEP", "1"))
                _target = max_pool if _grow_step <= 1 else (
                    ((max_pool + _grow_step - 1) // _grow_step) * _grow_step
                )
                pad = mx.zeros(
                    (B, _target - self.pooled.shape[1], D), dtype=px.dtype
                )
                self.pooled = mx.concatenate([self.pooled, pad], axis=1)
```

`os` was already imported at module top (cache.py:4) — no new import.
Everything else in the function is byte-identical.

**Default-path bit-identity, verified by exhaustive check** of the ceil-div
identity over `max_pool ∈ {1, 2, 255, 256, 257, 88149, 88150, 352599}`: at
`_grow_step = 1`, `_target == max_pool` in every case. Arm A is therefore
today's behaviour bit-for-bit, not merely "equivalent".

**SHAs**

| repo | before | after | pushed to |
|---|---|---|---|
| `mlx-lm` | `1fea494` (`known-good-decode-fenceasync-20260822`) | **`643d42d`** | `adurham/mlx-lm` main |
| `exo` | `9f81cdc91` | **`8a04cf492`** | `adurham/exo` main |
| `mlx` | `1c591e10` | `1c591e10` — **untouched** | — |

`uv.lock` was regenerated with `uv lock --upgrade-package mlx-lm`. Note the
lockfile pin had **already drifted** before this task: it pointed at
`5e88545a` while the submodule gitlink was at `1fea494`. It now matches the new
gitlink `643d42d` exactly. The only `mlx` change in the lock diff is uv's dev
date-stamp string (`dev20260821` → `dev20260823`); the mlx SHA is unchanged.

---

## 2. A real blocker found: the env var would never have reached the runners

`start_cluster.sh` does **not** pass the ambient environment through to the
spawned runner processes. Every runner-visible variable is explicitly
allowlisted into `EXO_ENV` (e.g. `EXO_DSV4_POOL_DEFER_COPY_MAX_BYTES` at
:2069-2070). `EXO_DSV4_POOL_GROW_STEP` was not on that list.

**Consequence had this not been caught:** launching arm B with
`EXO_DSV4_POOL_GROW_STEP=256 ./start_cluster.sh` would have produced a runner
whose environment lacked the variable entirely. `os.environ.get(..., "1")` would
have returned the default, and **arm B would have silently measured arm A** — two
identical arms, a null result, and a false "mechanism refuted" conclusion.

Fix (exo `8a04cf492`), placed beside the sibling pool knob:

```sh
    [ -n "${EXO_DSV4_POOL_GROW_STEP:-}" ] && EXO_ENV="$EXO_ENV EXO_DSV4_POOL_GROW_STEP=$EXO_DSV4_POOL_GROW_STEP"
```

Deliberately **no** `: "${VAR:=default}"` line. An unset variable is not
forwarded at all, so the arm-A launch command line stays byte-identical to
production. This preserves the A/B parity the spec asks for.

This is also why the spec's step-5 instruction to verify the variable on the
live runner with `ps eww` matters: it is exactly the check that would have caught
this. It should still be performed when the A/B is eventually run.

---

## 3. Pre-registered hypothesis (NOT a result)

From C3 §8.2, restated so the eventual run is judged against it rather than
against a post-hoc story:

| depth | arm A (`=1`) | arm B (`=256`) predicted |
|---|---|---|
| 100,026 | 35.79 ms/tok (B1 anchor) | ~35.25 ms/tok (−0.54) |
| 352,599 | 42.59 ms/tok (B1 anchor) | ~40.14 ms/tok (−2.45) |
| depth delta | +6.80 | ~+4.89 |

Throughput form: 23.48 → ~24.91 tok/s at 352.6K (+6.1%); 27.94 → ~28.37 tok/s at
100K (+1.5%). **The asymmetry (deep ≫ shallow) is the diagnostic fingerprint**, not
the absolute magnitude. Secondary: p90 inter-token gap at 352.6K should collapse
toward p50 (B1: p50 39.16 / p90 61.94; expect p90 −9..10 ms).

**Falsification condition, pre-registered:** no change at 352.6K ⇒ the mechanism is
not on the live critical path, and C3's +1.91 ms/token is a harness artifact of the
stub-MoE schedule. That outcome is to be reported as-is.

---

## 4. R2 control — static half DONE, and it confirms R2's confound is real

R2 (`p3-reviewer-r2-verification-2026-08-23.md`, C3-4) corrected C3 on two points
and asked for a control. Both corrections are honoured in the commit message and
here:

**(a) The safety invariant is `_pool_lengths`, not `_visible_width`.** Confirmed by
reading the code: at cache.py:1920 `_visible_width` is set from `visible`, which
in both branches *is* `self.pooled` in full — pad included. So
`min(P, self._visible_width)` at :2175-2176 returns `P` unchanged and cannot mask a
trailing pad. What masks it is the length mask at :2177-2181,
`pool_idx < pool_lengths`.

**(b) Padding flips the `make_mask` branch — arm B changes TWO things.** Also
confirmed. `make_mask` returns `None` when `all(pl == P)` (cache.py:2185-2186).
Arm A satisfies that; arm B does not.

R2's recommended precondition — *"assert `k < min(_pool_lengths)` holds at both
depths"* — is **verified**, by an arithmetic replica of cache.py:2170-2187:

| depth | arm | max_pool | P (pool width) | pad cols | make_mask returns | all pads masked | k=512 < pool_len |
|---|---|---|---|---|---|---|---|
| 100,026 | A | 25,007 | 25,007 | 0 | `None` | — | yes |
| 100,026 | B | 25,007 | 25,088 | 81 | `valid` array | **yes** | yes |
| 352,599 | A | 88,150 | 88,150 | 0 | `None` | — | yes |
| 352,599 | B | 88,150 | 88,320 | 170 | `valid` array | **yes** | yes |

So: every padded column masks to `False`, and `k = min(512, P)` is far below
`min(_pool_lengths)` at both depths, so pads can never enter the top-k —
**selection is unaffected and the patch is correctness-safe on this axis.** But
the branch flip is real, so arm B removes a concat *and* switches the indexer
from a no-mask path to a masked `mx.where` path (`deepseek_v4.py:3840/3883`).
At decode `L_q=1` that where() is over a `(1,1,P)` tensor, so it is O(P) — small,
but non-zero and not today's path.

**Implication for interpreting the eventual A/B:** a null result is genuinely
ambiguous. It could mean the concat is not on the critical path (the
pre-registered falsification), *or* that the removed concat cost and the added
mask cost roughly cancel. R2's suggested "slice the pad off before returning"
variant would isolate the concat cleanly and is the right follow-up if the A/B
comes back null.

**The remaining half of the control is live and NOT done:** confirming arm B's deep
output is not degraded (text quality / needle behaviour vs arm A). That requires
the cluster.

---

## 5. Why the live A/B was not run *on the first attempt* (historical)

*Kept verbatim for the record. The A/B was subsequently run to completion in a
later session with exclusive cluster time — see §10 onward.*

The task brief stated the cluster was DOWN and that relaunch was pre-approved.
**The cluster was in fact UP and in use by another agent.**

Observed at 17:35-17:38 CDT on `adams-mac-studio-m4-1`:

- `screen -dmS exorun` running continuously since **10:27 AM**; model
  `deepseek-ai/DeepSeek-V4-Flash-0731` placed, TP=2, both runners healthy.
- A concurrent peer subagent (`sa-0-b22f63d2`, different owner session, goal:
  *"Squeeze maximum sustained decode/prefill performance out of the 2-node exo
  cluster"*) firing generations at **17:10, 17:25, 17:26** — i.e. ~9 minutes
  before this check.
- A second screen, `p2prefill2`, from that peer's prefill work (last request
  ended in a runner `signal=9` at 11:10).

Three independent reasons not to proceed unilaterally:

1. **Mutual corruption.** `start_cluster.sh` does `git reset --hard origin/main`
   on both nodes, rebuilds, and restarts runners — it would kill the peer's live
   runner without warning. Conversely, if the peer fires a generation during one
   of the ~19-minute deep probes, both results are contaminated. The ops skill
   records exactly this failure mode: overlapping GPU workloads once produced
   0.3 tok/s readings that were misdiagnosed as a code regression. This task's own
   instructions say *"only one probe at a time."*
2. **The pre-approval rested on a false premise.** "Relaunch is pre-approved
   because the cluster is down" does not transfer to "relaunch is pre-approved
   while a peer is actively using it."
3. **A relaunch is needed regardless, so nothing is lost by pausing.** The running
   cluster's env does **not** contain `EXO_DSV4_MOE_FUSED_GATE_UP=1`, which the
   spec names as part of the production config. The current process could not have
   served as arm A even if it were free.

Checked and worth recording: **neither node has uncommitted local changes** in
`~/repos/exo` or `~/repos/exo/mlx-lm`, so a future `git reset --hard` is
file-safe. Both nodes are currently at exo `6bc843bfc` / mlx-lm `1fea494` —
i.e. **behind** the new commits; the next relaunch by anyone will deploy them.
This is safe: the lever is env-gated and defaults to bit-identical behaviour.

---

## 6. Exact runbook (as written before the run; followed as specified)

Requires ~1.5-2 h of **exclusive** cluster time. This runbook was executed
as-written, with one deviation noted in §10: the probe must be invoked with the
repo venv interpreter (`./.venv/bin/python`), not bare `python3`.

```sh
# ARM A — production config, GROW_STEP unset (default 1)
cd ~/repos/exo && (EXO_DSV4_MOE_FUSED_GATE_UP=1 EXO_DSV4_FENCE_ASYNC=1 \
  ./start_cluster.sh > /tmp/start_cluster_armA.log 2>&1 &) ; sleep 1
# poll for "READY (2/2)"; then verify deployed SHA on BOTH nodes:
#   git -C ~/repos/exo/mlx-lm rev-parse HEAD   -> 643d42d6854e4b6e0fa6e1b7c07cc448c4509c24
# smoke-generate and read the text before trusting anything.

python3 bench/p3_depth_anchor_probe.py --target-tokens 100000 \
  --max-tokens 2000 --out /tmp/poolgrow_armA_100k.json  | tee /tmp/poolgrow_armA_100k.log
python3 bench/p3_depth_anchor_probe.py --target-tokens 352595 \
  --max-tokens 2000 --out /tmp/poolgrow_armA_deep.json  | tee /tmp/poolgrow_armA_deep.log

# ARM B — identical, plus the lever
cd ~/repos/exo && (EXO_DSV4_MOE_FUSED_GATE_UP=1 EXO_DSV4_FENCE_ASYNC=1 \
  EXO_DSV4_POOL_GROW_STEP=256 ./start_cluster.sh > /tmp/start_cluster_armB.log 2>&1 &) ; sleep 1
# MANDATORY before probing — confirm the var actually reached the runner:
#   ssh <node> 'ps eww <runner_pid> | tr " " "\n" | grep EXO_DSV4_POOL_GROW_STEP'
#   must print EXO_DSV4_POOL_GROW_STEP=256 on BOTH nodes. If absent, STOP:
#   the arms are identical and the result is meaningless (see §2).
python3 bench/p3_depth_anchor_probe.py --target-tokens 100000 ... # armB_100k
python3 bench/p3_depth_anchor_probe.py --target-tokens 352595 ... # armB_deep
```

Sanity gate: arm A must reproduce B1's anchors (27.94 tok/s @100K, 23.48 @352.6K)
within ±5%. If not, stop and investigate before reading anything into arm B.
Deep probes take ~19 min each (~17.6 min prefill) — background + poll, do not kill.

Report `decode_tok_s_usage` (not the events-based number, per B1 §1.3), ms/tok,
p50/p90 inter-token gap, and a generated-text snippet for **every** number quoted.

---

## 7. `PERFORMANCE_HISTORY.md` entry

Originally withheld because no measurement existed — `PERFORMANCE_HISTORY.md` is
a record of *measured* outcomes and an entry for an experiment that never ran
would pollute the one document whose purpose is to prevent re-litigating settled
questions.

**The measurement now exists**, so a dated `NEW(...)` entry with the real
numbers has been appended to `docs/PERFORMANCE_HISTORY.md`.

---

## 8. Limitations *of the pre-run state* (historical)

*Superseded by §14, which states the limitations that survive the completed run.*

- **Nothing in §3 was tested.** The signature table is a hypothesis.
- The static R2 control (§4) proves the padded columns are *masked*; it does not
  prove end-to-end output quality is unchanged. That needs the live needle check.
- Arm B confounds two changes (removed concat + masked indexer path). A null
  result cannot distinguish "mechanism absent" from "two effects cancel".
- Default is **not** flipped, per instructions — `EXO_DSV4_POOL_GROW_STEP` is
  opt-in and unset everywhere. That decision remains open pending the live A/B
  and reviewer verification.
- The `uv.lock` mlx-lm pin drift noted in §1 pre-existed this task and may mean
  earlier runs relied on `start_cluster.sh`'s force-reinstall-from-submodule
  fallback rather than the lockfile. **Now resolved in practice** — see §10.3.

---

## 9. Files

- `mlx-lm/mlx_lm/models/cache.py` — the lever (mlx-lm `643d42d`).
- `start_cluster.sh` — env forwarding (exo `8a04cf492`).
- `uv.lock` — mlx-lm pin `5e88545a` → `643d42d`.
- `docs/p3-followup-poolgrow-ab-2026-08-23.md` — this file.
- Bench artifacts from the completed run: `/tmp/ab_arm{A,B}_{100026,352599}.{log,json}`
  (raw stdout + full per-run JSON including every inter-token gap),
  `/tmp/start_cluster_arm{A,B}.log` (relaunch logs),
  `/tmp/smoke_arm{A,B}.json` (post-deploy smoke generations).
- Probe used: `bench/p3_depth_anchor_probe.py`, unmodified. No code was changed
  by the measurement run; the only writes to either studio were the two
  `start_cluster.sh` relaunches.

---

# PART II — THE COMPLETED LIVE A/B (2026-08-23 evening)

## 10. Deployment and verification

### 10.1 Relaunch log

Two relaunches, both clean, both `EXIT=0`, no crashes, no runner deaths, no
`signal=9`, no Metal timeouts, no OOM.

| # | arm | started | READY (2/2) | duration | issues |
|---|---|---|---|---|---|
| 1 | **A** (`GROW_STEP` unset) | 17:46:05 CDT | ~17:54:40 | ~8.6 min | none |
| 2 | **B** (`GROW_STEP=256`) | 18:24:11 CDT | ~18:30:50 | ~6.7 min | none |

Both used the §6 invocation exactly:

```sh
cd ~/repos/exo && EXO_DSV4_MOE_FUSED_GATE_UP=1 EXO_DSV4_FENCE_ASYNC=1 \
  [EXO_DSV4_POOL_GROW_STEP=256] ./start_cluster.sh
```

Both logged `Nodes synchronized on commit 7acf74c57.` and
`Waiting for 2 DeepSeek V4 runner(s) to become Ready........ READY (2/2)`.
Prior to relaunch 1 the cluster was live on **stale** code (exo `6bc843bfc`,
mlx-lm `1fea494`) and without `EXO_DSV4_MOE_FUSED_GATE_UP=1` — i.e. it could not
have served as arm A, exactly as §5 predicted.

### 10.2 Deployed SHAs (verified on BOTH nodes, both arms)

| | node m4-1 | node m4-2 |
|---|---|---|
| `exo` HEAD | `7acf74c5749cd93a42fa12dcda9f2aa400fc3328` | same |
| `mlx-lm` submodule HEAD | `643d42d6854e4b6e0fa6e1b7c07cc448c4509c24` | same |

### 10.3 The `uv.lock` caveat, discharged

Checking the submodule gitlink is **not** sufficient — the runner imports
`mlx_lm` from the venv, not from `./mlx-lm/`. Resolved the actual import path on
each node and hashed the file the runner really executes:

```
$ .venv/bin/python -c "import mlx_lm,os;print(os.path.dirname(mlx_lm.__file__))"
/Users/adam.durham/repos/exo/.venv/lib/python3.13/site-packages/mlx_lm
```

| | venv `models/cache.py` md5 | submodule `models/cache.py` md5 | lever present |
|---|---|---|---|
| node m4-1 | `f6b4201d1fae8634d1b3465445451185` | `f6b4201d1fae8634d1b3465445451185` | yes, `cache.py:1900` |
| node m4-2 | `f6b4201d1fae8634d1b3465445451185` | `f6b4201d1fae8634d1b3465445451185` | yes, `cache.py:1900` |

Identical to the local checkout's `643d42d` file. **The runner is provably
executing the new `cache.py` on both nodes.**

Also confirmed the lever has exactly **one** consumer site in the tree
(`grep -rn EXO_DSV4_POOL_GROW_STEP mlx-lm/ src/` → one hit, `cache.py:1900`), so
the env var cannot be reaching some second, unaccounted code path.

### 10.4 Runner env — the §2 check that makes or breaks the A/B

`ps eww <runner_pid>` on the real runner process (the `spawn_main` child of
`.venv/bin/python -m exo -v`), both nodes, both arms:

| arm | node | `EXO_DSV4_POOL_GROW_STEP` | `EXO_DSV4_MOE_FUSED_GATE_UP` | `EXO_DSV4_FENCE_ASYNC` |
|---|---|---|---|---|
| A | m4-1 (pid 76581) | **absent** ✅ | `1` | `1` |
| A | m4-2 (pid 75375) | **absent** ✅ | `1` | `1` |
| B | m4-1 (pid 81843) | **`256`** ✅ | `1` | `1` |
| B | m4-2 (pid 81022) | **`256`** ✅ | `1` | `1` |

This is the check §2 said would catch a silent null. It passed in the
discriminating direction: the arms are genuinely different, and differ *only* in
this variable.

### 10.5 Smoke generations (post-deploy, pre-probe)

Prompt (both arms, `temperature=0`, `max_tokens=150`):
*"In one sentence, what is the capital of France and why is it notable?"*

- **Arm A:** "The capital of France is Paris, notable for its profound global
  influence in art, fashion, culture, and history, as well as being a major
  political and economic hub."
- **Arm B:** "The capital of France is Paris, notable for its profound global
  influence in art, fashion, gastronomy, and culture, as well as being home to
  iconic landmarks like the Eiffel Tower and the Louvre."

Both coherent, correct, no `<|begin_of_sentence|>` leakage, no U+FFFD.

---

## 11. Raw results — all four probes

All four: `finish_reason=length` with the full 2000 completion tokens (EOS
genuinely banned via `/bench/chat/completions`), `cached_tokens=0` (unique UUID
nonce + `use_prefix_cache=False`, so no prefix-cache shortcut), depth read back
from `usage.prompt_tokens`, decode window >= 60 s.

### 11.1 The eight headline numbers

Reported as `decode_tok_s_usage` per B1 §1.3.

| depth | arm | REAL prompt_tokens | **tok/s** | **ms/tok** | decode window | TTFT |
|---|---|---|---|---|---|---|
| 100K | **A** | 100,022 | **28.09** | **35.60** | 71.17 s | 275.84 s |
| 100K | **B** | 100,023 | **29.06** | **34.41** | 68.79 s | 276.58 s |
| 352.6K | **A** | 352,602 | **23.50** | **42.55** | 85.06 s | 1068.26 s |
| 352.6K | **B** | 352,601 | **25.80** | **38.76** | 77.48 s | 1058.33 s |

**Deltas (B − A):**

| depth | Δ tok/s | Δ % | Δ ms/tok |
|---|---|---|---|
| 100,022 | +0.97 | **+3.46%** | **−1.19** |
| 352,602 | +2.30 | **+9.79%** | **−3.79** |

**Depth delta (100K → 352.6K), ms/tok:** arm A **+6.95**, arm B **+4.35**.

### 11.2 Arm A vs the B1 anchors — the sanity gate

| depth | B1 anchor tok/s | arm A tok/s | deviation | gate ±5% |
|---|---|---|---|---|
| 100,026 | 27.94 | 28.09 | **+0.53%** | **PASS** |
| 352,599 | 23.48 | 23.50 | **+0.09%** | **PASS** |

Arm A reproduces B1's independently-measured anchors to within half a percent at
both depths. This is the single most important validity result in this document:
it establishes that the harness, the cluster, and the code path are all in the
same state B1 measured, and it puts an empirical ceiling of roughly ±0.5% on
run-to-run noise for this instrument. Both arm-B effects are far outside it.

### 11.3 Inter-token gap distribution (ms)

| arm / depth | p10 | **p50** | **p90** | p99 | mean | stdev | p90−p50 |
|---|---|---|---|---|---|---|---|
| A / 100K | 11.16 | 34.39 | 62.59 | 103.68 | 36.46 | 20.59 | 28.20 |
| B / 100K | 14.77 | 33.75 | 54.92 | 97.52 | 34.97 | 18.51 | **21.17** |
| A / 352.6K | 18.71 | **39.53** | **70.53** | 110.81 | 42.81 | 21.77 | 31.00 |
| B / 352.6K | 20.53 | **38.21** | **64.89** | 100.81 | 40.04 | 18.93 | **26.68** |

At 352.6K, arm B: p50 −1.32 ms, **p90 −5.64 ms**, p99 −10.00 ms, stdev −2.84,
and the p90−p50 spread narrows from 31.00 to 26.68 ms. The predicted direction
(p90 collapsing toward p50) is **present**; the predicted magnitude (−9..10 ms)
is **about half-met**. Outlier rate also falls at depth: 0.70% → 0.36% of gaps
above 3× median.

### 11.4 Generated-text snippet for every number quoted

Each probe's own output, so no number in this document is quoted without the
text that produced it.

| arm / depth | snippet (~15 words) |
|---|---|
| A / 100K | *"A synthetic, repetitive corpus of 2,263 templated statements, each linking one of eight system behaviors to a configuration number"* |
| B / 100K | *"The corpus is a long list of sections (0 through 2262) that follow a repetitive pattern"* |
| A / 352.6K | *"The user has provided a very long corpus of text, seemingly generated with a specific pattern"* |
| B / 352.6K | *"The corpus is a long list of sections, each with a similar structure: In practice [topic]..."* |

### 11.5 Foreign-traffic audit

For each of the four probe windows, the API request log on the master node was
grepped for the exact wall-clock span:

| arm / depth | window (CDT) | chat-completion requests in window | verdict |
|---|---|---|---|
| A / 100K | 17:56:15–18:03:27 | 1 (`POST /bench/chat/completions` @ 17:56:18 — this probe) | clean |
| A / 352.6K | 18:03:39–18:23:58 | 1 (`POST /bench/…` @ 18:03:46 — this probe) | clean |
| B / 100K | 18:31:37–18:38:17 | 1 (`POST /bench/…` @ 18:31:40 — this probe) | clean |
| B / 352.6K | 18:38:26–18:58:44 | 1 (`POST /bench/…` @ 18:38:33 — this probe) | clean |

**Zero foreign requests. No probe was rerun.** (A `POST /v1/chat/completions` at
18:31:27 appears just before the arm-B 100K window — that is this task's own
§10.5 smoke generation, which completed at 18:31:32, before the probe started.)

---

## 12. Verdict vs the pre-registered signature: **CONFIRMED**

### 12.1 Scorecard

| pre-registered claim (§3) | predicted | measured | verdict |
|---|---|---|---|
| 352.6K ms/tok change | −2.45 | **−3.79** | **met, exceeded** |
| 100K ms/tok change | −0.54 | **−1.19** | **met, exceeded** |
| 352.6K throughput | 23.48 → ~24.91 (+6.1%) | 23.50 → **25.80 (+9.79%)** | **met, exceeded** |
| 100K throughput | 27.94 → ~28.37 (+1.5%) | 28.09 → **29.06 (+3.46%)** | **met, exceeded** |
| depth delta arm A | +6.80 | **+6.95** | **hit (2.2% off)** |
| depth delta arm B | ~+4.89 | **+4.35** | **hit (11% off)** |
| **asymmetry: deep ≫ shallow** | ~4.5× | **3.2×** | **present, direction correct** |
| p90 at 352.6K collapses toward p50 | −9..10 ms | **−5.64 ms** | **direction met, magnitude ~half** |
| **falsification: no change at 352.6K** | — | −3.79 ms/tok, +9.79% | **NOT triggered** |

### 12.2 Statement of the verdict

**CONFIRMED.** The pre-registered falsification criterion was not met. The
deep≫shallow asymmetry fingerprint is present, and the pre-registered
*depth-delta* values — the numbers that most directly encode the hypothesis —
were hit closely (arm A +6.95 vs predicted +6.80; arm B +4.35 vs predicted
+4.89). Both effects exceed by a wide margin the ~±0.5% run-to-run noise floor
established empirically by arm A's reproduction of the B1 anchors (§11.2).

Two honest qualifications, neither of which changes the verdict:

1. **Absolute magnitudes came in ~2× larger than predicted at both depths.**
   This is a cost-model calibration miss, not a contradiction. The pattern —
   a larger-than-modelled *fixed* component plus a roughly-as-modelled
   *depth-scaling* component — is what you get if the per-flush cost carries
   fixed overhead (allocation, kernel launch, graph rebuild) that C3's estimate
   underweighted relative to the size-dependent copy. That compresses the
   asymmetry ratio from ~4.5× toward the measured 3.2× while preserving the
   asymmetry itself.
2. **The asymmetry ratio (3.2× vs 4.5×) is not resolvable at n=1** and is not
   treated as a miss. Its denominator is the small shallow effect
   (−1.19 ms/tok); ordinary run noise moves that ratio substantially. C3 §8.2
   itself pre-registered that "the asymmetry is the diagnostic fingerprint, not
   the absolute magnitude."

### 12.3 What is confirmed, precisely — and what is not

**Confirmed:** setting `EXO_DSV4_POOL_GROW_STEP=256` causes a substantial
decode-throughput improvement that grows with context depth. That is a causal
claim about the env var, and the controls support it: identical SHAs, identical
launch command modulo the one variable, `ps eww`-verified env difference on both
nodes, byte-identical runner-imported `cache.py`, no foreign traffic, and arm B
measured *after* ~40 min of sustained load (thermally conservative — if anything
biased against arm B).

**Not yet formally isolated:** that the gain comes *specifically* from
eliminating per-flush concats, as opposed to the coupled `make_mask` branch flip
(§4). This is very likely — the branch flip *adds* work (a masked `mx.where`
over a `(1,1,P)` tensor), so it can hide a real gain but cannot manufacture one
— but "very likely" is not "isolated." R2's slice-off-the-pad variant remains
the discriminating follow-up if that distinction matters for a default flip.

Supporting evidence that this is decode-path-only, as the mechanism requires:
**TTFT is unchanged between arms** (352.6K: 1068.26 s arm A vs 1058.33 s arm B;
100K: 275.84 vs 276.58). A confound acting on the whole pipeline would be
expected to move prefill too. It did not.

---

## 13. R2 make_mask control — live half: **PASSED**

The static half (§4) proved the padded columns are always masked out and can
never enter the top-k. The live half asks whether arm B's deep output is
degraded in practice. Both probes ban EOS and run the full 2000 tokens, so any
degeneration has ample room to show.

| arm / depth | U+FFFD | repetition loop | on-task | coherent |
|---|---|---|---|---|
| A / 100K | 0 | none | yes | yes |
| B / 100K | 0 | none | yes | yes |
| A / 352.6K | 0 | none | yes | yes |
| B / 352.6K | 0 | none | yes | yes |

Arm B at 352.6K correctly recovers the corpus structure — the templated
`"In practice [topic] … depends on configuration [number] and on the observed
interaction between stage [n] and stage [m]"` pattern, the cycling of topics,
and the synthetic nature of the numeric fields. That is a genuine
content-dependent read of a 352K-token prompt, i.e. it functions as a
needle-style check: a model whose deep attention had been corrupted by a
mis-masked pad could not describe the corpus's actual structure.

Arm B shows **no** garbling, no repetition, and no nonsense beyond what arm A
shows (which is none). Both arms' 2000-token generations terminate mid-sentence
on the `length` stop, as designed. **The control passes; arm B is not degraded.**

---

## 14. Limitations of the completed run

- **n=1 per cell.** Four probe runs, one per (arm × depth). Deep points cost
  ~19 min each (~17.6 min prefill), so replication was not affordable. Mitigated
  — not eliminated — by arm A reproducing two independent prior anchors to
  within 0.53% / 0.09%, which bounds run-to-run noise well below the effects.
  The measured effects (+3.46%, +9.79%) are ~7× and ~20× that floor.
- **The confound survives, in the benign direction.** Arm B changes both the
  concat chunking and the `make_mask` `None`→`valid` path (§4). Because the
  branch flip adds cost, it cannot fabricate the observed speedup — but the
  clean attribution to concat elimination alone still wants R2's
  slice-off-the-pad variant. That is the right next experiment, now for
  *attribution* rather than for disambiguating a null.
- **`GROW_STEP=256` only.** No sweep over step sizes. 256 was chosen to match
  `PoolingCache.step`; a larger step might buy more (fewer flushes) or less
  (more wasted pad, larger masked `where`). Unmeasured.
- **Two depths only.** No 300K point, so the shape of the gain between 100K and
  352.6K is interpolated, not measured.
- **p90 magnitude under-delivered** (−5.64 vs −9..10 ms predicted) even though
  the direction is right. The secondary signature is only half-confirmed and
  should not be quoted as a clean hit.
- **`MLX_GPU_TIME=1` mod-4 spike check not run.** C3 §8.2 named it as an
  alternative secondary signature; it would have required a third relaunch with
  a different env and a known ~40% perf hit, which would have invalidated the
  A/B parity.
- **The default was NOT changed**, per instructions. `EXO_DSV4_POOL_GROW_STEP`
  remains opt-in and unset in the committed default path. Flipping it is a
  separate reviewed step, and this document is the evidence for that review, not
  the review itself.
- **Cluster left running in arm B** (`EXO_DSV4_POOL_GROW_STEP=256`), since the
  evidence favours it and the output-quality control passed. Note this is a
  *runtime* state, not a committed default: the next relaunch without the
  variable exported reverts to arm A behaviour, bit-for-bit.
