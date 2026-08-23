# P3 follow-up — BatchPoolingCache chunked-growth A/B (`EXO_DSV4_POOL_GROW_STEP`) — 2026-08-23

**Status: LEVER IMPLEMENTED AND DEPLOYED TO GIT. LIVE A/B NOT RUN — BLOCKED ON CLUSTER CONTENTION.**

This document records the code change, two verification results that needed no
cluster, and the exact reason the four probe runs were **not** executed. **No
throughput numbers are reported here, because none were measured.** The
pre-registered signature table from
`p3-worker-c3-donation-failure-insitu-2026-08-23.md` §8.2 is reproduced below as
the *hypothesis to be tested*, not as a result.

---

## 0. What was asked vs what happened

| step | status |
|---|---|
| 1. Code change (env-gated chunked grow) | **DONE** — `mlx-lm` `643d42d` |
| 2. Git (mlx-lm commit+push, exo submodule bump, uv.lock, push) | **DONE** — `exo` `8a04cf492` |
| 2b. *(unplanned, required)* env forwarding in `start_cluster.sh` | **DONE** — see §2 |
| 3. Deploy arm A (relaunch) | **NOT DONE** — cluster contention, §5 |
| 4. Arm A probes (100,026 / 352,599) | **NOT DONE** |
| 5. Deploy arm B + verify env reached runners | **NOT DONE** |
| 6. Analysis vs pre-registered signature | **NOT POSSIBLE** — no data |
| 6b. R2 make_mask control, static half | **DONE** — §4, and it is a real finding |
| 7. This doc | **DONE** |
| 7b. `PERFORMANCE_HISTORY.md` entry | **DELIBERATELY NOT WRITTEN** — §7 |

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

## 5. Why the live A/B was not run

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

## 6. Exact runbook to finish this (nothing else is blocking)

Requires ~1.5-2 h of **exclusive** cluster time.

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

## 7. Why there is no `PERFORMANCE_HISTORY.md` entry

The task asked for a dated `NEW(...)` entry "describing the A/B result (measured,
either direction)". **No measurement exists**, so no entry was written.
`PERFORMANCE_HISTORY.md` is a record of measured outcomes; adding an entry for an
experiment that never ran would pollute the one document whose stated purpose is
to prevent re-litigating settled questions. The entry should be appended by
whoever completes §6, with real numbers.

---

## 8. Limitations

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
  fallback rather than the lockfile. Not investigated further here.

---

## 9. Files

- `mlx-lm/mlx_lm/models/cache.py` — the lever (mlx-lm `643d42d`).
- `start_cluster.sh` — env forwarding (exo `8a04cf492`).
- `uv.lock` — mlx-lm pin `5e88545a` → `643d42d`.
- `docs/p3-followup-poolgrow-ab-2026-08-23.md` — this file.
- No bench artifacts were produced: no probe was run. Nothing on either studio
  was created, edited, or deleted; no runner process was touched.
