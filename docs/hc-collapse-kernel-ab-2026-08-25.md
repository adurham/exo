# hc_collapse fused precursor kernel — live A/B on 2026-08-25 — SHIP

**Status: LIVE A/B COMPLETE. VERDICT — SHIP (default flipped, cluster now
production on kernel-on).** All pre-registered framings passed (+1.89%
mean prefill); supervisor GO was given at ~11:05 and the production flip
was executed and verified 11:12–11:20 the same day — recorded in §14.

`EXO_DSV4_HC_COLLAPSE_KERNEL=1` gates a fused Metal *precursor* kernel
(`astype fp32` + `rms_norm` + `matmul fn.T`, one dispatch instead of
three) for the `HyperConnection.collapse` path used by every layer's
`layer.attn_hc` and `layer.ffn_hc`. The kernel lives in mlx-lm on the
fork branch `kernel/hc-collapse-roofline` at commit
`8d5de181d09cc9ce9e5955f5be5fe4708f86258e`; the env gate defaults **OFF**
(unset ⇒ the classic op path, bit-identical). Laptop validation and the
roofline analysis are in `docs/hc-collapse-roofline-2026-08-24.md` and
are not repeated here.

In a 2×2 A/B at ~70.5K real prompt tokens (`--targets 100000` via
`bench/phase3_precheck_depth_throughput.py`) on the live two-node
cluster:

- **Arm A (kernel OFF, env absent)**: mean **376.1681 tok/s** prefill
  (376.1051, 376.2310) — spread 0.0335%
- **Arm B (kernel ON)**: mean **383.2700 tok/s** prefill (383.9832,
  382.5568) — spread 0.3722%
- **Delta**: **+7.10 tok/s (+1.8880%)** mean-to-mean; +2.0947% B1−A1,
  +1.6814% B2−A2 pairwise
- **Conservative bound** (min B vs max A): **+1.6814%** — still clears
  the pre-registered `mean(B) >= mean(A) × 1.015` threshold
  (threshold value: **381.8106 tok/s**; measured mean B 383.2700)
- **Quality**: needle FALCON-MERCURY-7749 recovered **byte-identical on
  all 4 runs**, zero U+FFFD, zero BOS spam, launch logs clean

**Every pre-registered framing passes.** The honest headline is the
mean-based **+1.89%**, which is ~70% of the pre-registered prediction of
+2.73% (span share 4.6% × (1 − 1/2.47) from the laptop roofline, where
the fused span measured 2.47× faster). That transfer shortfall, and the
n=2-per-arm margin-vs-spread caveat, are stated plainly in §12 and §15 —
they do not change the verdict but they are the parts of this result a
reader should not over-read.

---

## 0. What was asked vs what happened

| step | status |
|---|---|
| 0. W1 local numerical validation of the fused kernel (worktree of `8d5de18`) | **DONE** — §1.1; fused-vs-classic at `[1,64,4,512]` bf16: collapsed max abs `1.953e-03` / mean rel `4.49e-07`; post `2.384e-07` / `4.90e-08`; comb `1.788e-07` / `1.80e-07`; `L=63` fallback **bit-identical** (call-counter 0, `np.array_equal` True) |
| 1. Env forwarding commit for `EXO_DSV4_HC_COLLAPSE_KERNEL` | **DONE** — exo `782c8cf97`, opt-in forwarding line only, **no default flip** — §2 |
| 2. Deploy both nodes, arm A (kernel-off vehicle) | **DONE** — §3; exo HEAD `cd254d15a`, mlx-lm submodule `7a1a4e86`, venv grep `HC_COLLAPSE=0` both nodes |
| 2b. Launcher incident (Xcode removed from both studios) | **RESOLVED** — §2.1; killed the first arm-A launch inside `uv sync`'s maturin `exo_rs` build; fixed by `70e0423bc` + `cd254d15a`, documented separately |
| 3. Arm A probe (env verified ABSENT on all 4 runner PIDs) | **DONE** — 2 runs, §11 |
| 4. Arm B probe (env verified `=1` on all 4 runner PIDs) | **DONE** — 2 runs, §11; quality §12 |
| 5. Verdict vs pre-registered criteria | **DONE** — §13, **SHIP** (all framings pass) |
| 6. Production flip + final verification | **PENDING SUPERVISOR GO** — to be appended as §14 by the ship task |
| 7. This doc + `PERFORMANCE_HISTORY.md` entry | **DONE** — this file + the `2026-08-25` hc_collapse section |

---

## 1. The code change and its local validation

### 1.1 Kernel + gate

The fused precursor kernel and its gate live in mlx-lm on the fork
branch `kernel/hc-collapse-roofline`, commit
`8d5de181d09cc9ce9e5955f5be5fe4708f86258e`. The kernel fuses the three
ops that precede the HyperConnection collapse matmul — `astype` to
fp32, `rms_norm`, and the `matmul` against `fn.T` — into a single Metal
dispatch, for both `layer.attn_hc` and `layer.ffn_hc`.

The gate is `EXO_DSV4_HC_COLLAPSE_KERNEL`, **default OFF**. Unset ⇒ the
classic (pre-kernel) op path runs, bit-identically. That bit-identity is
what makes arm A a proper reference arm without needing to roll the
submodule back.

W1 local validation was run in a throwaway worktree of `8d5de18` on the
laptop (not on the cluster), fused vs classic at shape `[1,64,4,512]`,
bf16:

| tensor | max abs diff | mean rel diff |
|---|---|---|
| `collapsed` | `1.953e-03` | `4.49e-07` |
| `post` | `2.384e-07` | `4.90e-08` |
| `comb` | `1.788e-07` | `1.80e-07` |

The `L=63` fallback path (the shape the kernel declines to handle) was
verified **bit-identical**: kernel call-counter 0 and
`np.array_equal(fused, classic) == True`. This is the fp32-exact class
of numerics, consistent with the sibling `hc_expand` kernel that shipped
2026-08-24.

The roofline/span analysis that motivated the kernel — span speedup
**2.47×**, span share of prefill wall time **4.6%**, hence a predicted
e2e gain of **+2.73%** (`4.6% × (1 − 1/2.47)`) — lives in
`docs/hc-collapse-roofline-2026-08-24.md`. Cited, not repeated.

---

## 2. Env forwarding (exo `782c8cf97`)

`start_cluster.sh` does not forward ambient environment through to the
spawned runner processes; runner-visible variables must be explicitly
allowlisted into `EXO_ENV`. Without this, `EXO_DSV4_HC_COLLAPSE_KERNEL=1
./start_cluster.sh` would launch runners whose env lacked the variable
entirely, arm B would silently be arm A, and the A/B would produce a
false null. This exact miss silently null'd the pool-grow A/B on its
first attempt (`docs/p3-followup-poolgrow-ab-2026-08-23.md` §2), and the
`exo-cluster-deployment` skill flags env-var propagation as a recurring
pitfall.

Commit **`782c8cf97`** ("launch: opt-in env forwarding for
`EXO_DSV4_HC_COLLAPSE_KERNEL` (hc_collapse fused precursor A/B)") adds
the forwarding one-liner beside the sibling `hc_expand` forwarding.
It is **opt-in only — no default flip**: an unset variable leaves the
production launch command line unchanged, so an unset here is provably
arm A. Any default flip is deferred to the post-GO ship task (§14).

### 2.1 Launcher incident (Xcode removed from both studios)

The **first** arm-A launch of this session aborted with
`Failed to sync on macstudio-m4-1`. Root cause: `/Applications/Xcode.app`
had been removed from **both** studios (only CommandLineTools remained),
while `start_cluster.sh` hardcoded
`DEVELOPER_DIR=/Applications/Xcode.app/Contents/Developer`; the launch
died inside `uv sync`'s maturin build of `exo_rs`
(`maturin` → `cargo` → `cc` → `xcrun: error: missing DEVELOPER_DIR path`).
It was latent until `782c8cf97` touched `start_cluster.sh` and
invalidated uv's wheel cache, making this the first launch that actually
had to build. Fixed by launcher commits **`70e0423bc`** (per-node
`DEVELOPER_DIR` resolution with CLT fallback, fail-fast if an mlx rebuild
truly needs Metal) and **`cd254d15a`** (the incident writeup). Full
detail: `docs/xcode-removal-launcher-clt-fallback-2026-08-25.md`. It cost
this session one launch; it did **not** contaminate any measurement — all
four probes below ran on post-fix launches at exo HEAD `cd254d15a`.

---

## 3. Deploy mechanism and verification

### 3.1 Deploy mechanism (differs from the hc_expand A/B — read this)

`start_cluster.sh` now **rsyncs the laptop tree verbatim** to both nodes
(no per-node `git fetch`) and force-reinstalls mlx-lm from `./mlx-lm`.
That changes what "deploying arm B" means compared to yesterday's
hc_expand A/B:

- **Arm A vehicle**: laptop mlx-lm submodule left at its committed
  pointer `7a1a4e86` (the shipped hc_expand pin). Kernel source not even
  present.
- **Arm B vehicle**: laptop mlx-lm submodule checked out **DETACHED at
  `8d5de18`**, *uncommitted* in the exo superproject (working tree shows
  `M mlx-lm`, a pointer-only modification). The rsync then carries that
  exact checkout to both nodes and the force-reinstall puts it in each
  node's venv.

**Why this is git-clean and not a hidden state problem:** the kernel
commit `8d5de181d09cc9ce9e5955f5be5fe4708f86258e` is **pushed on the
mlx-lm fork branch `kernel/hc-collapse-roofline`** — it is real,
fetchable, reviewable history, not a local-only blob. The only thing
that was local-and-uncommitted was the **superproject's submodule
pointer**, i.e. "which of two already-published commits mlx-lm is
currently checked out at." Nothing measured here depends on unpublished
code. Consequently this session deliberately does **not** commit the
`M mlx-lm` pointer bump: promoting the pin is a *ship* action, not a
*measurement* action, and belongs to the post-GO task (§14) so that git
history keeps the distinction between what was measured and what was
shipped.

### 3.2 Deployed SHAs (both nodes, both arms)

|  | node m4-1 | node m4-2 |
|---|---|---|
| `exo` HEAD (both arms) | `cd254d15a` | `cd254d15a` |
| `mlx-lm` checkout, **arm A** | `7a1a4e86` | `7a1a4e86` |
| `mlx-lm` checkout, **arm B** | `8d5de181d` (detached) | `8d5de181d` (detached) |

Arm B's submodule state was confirmed by `git rev-parse` in the deployed
tree on **both** nodes returning `8d5de181d`.

### 3.3 Venv verification (did the runner's installed mlx-lm actually change?)

A grep for the kernel gate string in each node's venv-installed
`mlx_lm`, which distinguishes "new code installed" from "stale
`site-packages` copy" (the pitfall the `exo-sdpa-fusion-analysis` skill
documents for this exact submodule):

| arm | node m4-1 | node m4-2 |
|---|---|---|
| A | `HC_COLLAPSE` grep hits = **0** | **0** |
| B | `HC_COLLAPSE` grep hits = **3** | **3** |

Arm A's venv does not contain the kernel at all; arm B's does. The arms
differ in installed code, as intended.

### 3.4 Runner env, per-PID (the check that makes or breaks the A/B)

`ps eww` on **all 8 runner PIDs** (4 per node × 2 arms):

| arm | nodes | PIDs checked | `EXO_DSV4_HC_COLLAPSE_KERNEL` |
|---|---|---|---|
| A | m4-1 + m4-2 | 4/node = 8 total | **absent** ✅ (all 8) |
| B | m4-1 + m4-2 | 4/node = 8 total | **`1`** ✅ (all 8) |

Common to **both** arms on every PID: `EXO_DSV4_HC_EXPAND_KERNEL=1`
(the sibling kernel shipped yesterday — now production, held constant
here), `EXO_SPECULATIVE=0`, `EXO_DSV4_MTP=0`, `EXO_DSV4_DSPARK=1`.
Instance: `deepseek-ai/DeepSeek-V4-Flash-0731`, TP `worldSize=2`,
`quantization=fp8`.

The arms are genuinely different, and differ **only** in the intended
variable. No silent null.

### 3.5 Launch timeline

| arm | started | READY (2/2) | duration | RunnerFailed |
|---|---|---|---|---|
| **A** | 10:36:00 | 10:40:57 | ~5 min | **0** |
| **B** | 10:51:37 | 10:57:12 | ~5.5 min | **0** |

Both launches clean, zero `RunnerFailed` events. Launches are much
faster than the hc_expand A/B's 11-14 min because no mlx C++ rebuild was
needed (the `mlx` pin was unchanged and stamped-good on both nodes).

---

## 4. Pre-registered ship criterion (verbatim)

> Ship threshold: `mean(B) >= mean(A) × 1.015` AND quality clean.

Quality clean is defined as: needle recall exact, generated text shown in
the report, no BOS spam / U+FFFD garbling, cluster healthy.

Pre-registered prediction (from the laptop roofline, for calibration —
not a gate): span speedup **2.47×**, span share **4.6%** ⇒ predicted e2e
**+2.73%**.

---

## 5. Method

- **Probe**: `bench/phase3_precheck_depth_throughput.py`, **unmodified**,
  run laptop-side against the cluster API.
- **Command** (identical across all 4 probes, only `--json-out` varies):
  `--targets 100000 --max-tokens 128`, model
  `deepseek-ai/DeepSeek-V4-Flash-0731`, outputs
  `/tmp/hccol_arm{A,B}_100k{,_r2}.json`.
- **Metric**: prefill tok/s = **offline-tokenized prompt tokens /
  wall-clock TTFT**. This is the campaign-standard anti-accounting-drift
  rule from the `exo-dsv4-prefill-tuning` skill's PITFALL section: never
  derive tok/s from a server-reported token count. The TTFT identity
  (`depth / TTFT = reported prefill tok/s`) was **recomputed exactly for
  all four runs** and matches.
- **Prompt shape**: needle-in-haystack, needle `FALCON-MERCURY-7749`,
  `target_tokens=100000` → **~70.4-70.6K real prompt tokens**, directly
  comparable to every prior probe on this cluster at this target.
- **Design**: 2 runs per arm, sequential, arm A first then arm B, one
  relaunch per arm.

---

## 6. Raw results — all four probes

| run | depth (tokens) | prefill tok/s | TTFT (s) | decode tok/s | reasoning_tokens | needle |
|---|---|---|---|---|---|---|
| A r1 | 70,625 | **376.1051** | 187.7800 | 26.3937 | 32 | ✓ FALCON-MERCURY-7749 |
| A r2 | 70,390 | **376.2310** | 187.0925 | 26.5416 | 34 | ✓ FALCON-MERCURY-7749 |
| B r1 | 70,418 | **383.9832** | 183.3882 | 27.3900 | 60 | ✓ FALCON-MERCURY-7749 |
| B r2 | 70,431 | **382.5568** | 184.1060 | 30.2687 | 34 | ✓ FALCON-MERCURY-7749 |

Depth spread across the four probes: **235 tokens (0.33%)** — small
enough that no depth-normalization is applied, but see §15 (it is not
zero, and it is of the same order as the measured effect's own
uncertainty).

---

## 7. Aggregates, deltas, and the gate

**Aggregate**:

| arm | mean prefill | min | max | spread (% of mean) |
|---|---|---|---|---|
| A | **376.1681** tok/s | 376.1051 | 376.2310 | **0.0335%** |
| B | **383.2700** tok/s | 382.5568 | 383.9832 | **0.3722%** |

**Deltas (B − A)**:

| framing | Δ % | pass vs +1.5%? |
|---|---|---|
| pairwise B1−A1 | **+2.0947%** | ✅ |
| pairwise B2−A2 | **+1.6814%** | ✅ |
| **mean(B) − mean(A)** | **+1.8880%** (+7.10 tok/s) | ✅ |
| conservative: min B (382.5568) vs max A (376.2310) | **+1.6814%** | ✅ |

**Gate**:

| quantity | value |
|---|---|
| `mean(A)` | 376.1681 tok/s |
| threshold `mean(A) × 1.015` | **381.8106 tok/s** |
| `mean(B)` | **383.2700 tok/s** |
| margin above threshold | **+1.4594 tok/s (+0.38%)** |
| verdict | **PASS — all framings** |

**ALL PRE-REGISTERED FRAMINGS PASS.** Note honestly that the *smallest*
passing framing (the conservative min-B-vs-max-A / pairwise-B2−A2 view,
+1.6814%) clears the threshold by only **+0.18 percentage points** — see
§15 for why that margin deserves a caveat rather than a victory lap.

---

## 8. Prediction vs measurement (transfer)

| | value |
|---|---|
| predicted e2e (laptop roofline: `4.6% × (1 − 1/2.47)`) | **+2.73%** |
| measured e2e (mean-mean, live cluster) | **+1.89%** |
| transfer | **~70% of prediction** |

The lever delivers, but delivers **less than the span math predicted**.
Plausible contributors: the laptop-measured span share (4.6%) was
measured at a different depth/config than this 70.5K TP fp8 run; the
2.47× span speedup does not fully translate under real cluster
scheduling and memory pressure; and the sibling `hc_expand` kernel is
already ON in both arms here, so some of the addressable non-kernel time
this kernel was sized against has already been removed. No attempt is
made in this doc to attribute the 30% shortfall — it is recorded as an
honest calibration data point for future span-share-based predictions on
this cluster (the hc_expand A/B, by contrast, matched its prediction
almost exactly, so span-share estimation is not uniformly optimistic).

---

## 9. Quality check (needle + generated text evidence)

Per the campaign rule "never quote t/s without the generated text":

- **Prompt**: standard `phase3_precheck_depth_throughput.py` needle-in-
  haystack; needle string
  `"The secret code for project Nightingale is: FALCON-MERCURY-7749."`
  placed at a random position in the middle third; question: "What is the
  secret code for project Nightingale? Answer with just the code,
  nothing else."

- **Response, all four runs, byte-identical**:

  ```
  FALCON-MERCURY-7749
  ```

- **Needle recall**: exact on **4/4** probes (`needle_found=True` in
  every JSON). Kernel-on does not corrupt long-context attention at
  ~70.5K.
- **Degeneration**: zero U+FFFD, zero BOS spam, no reasoning loops, no
  tool-call XML leak. Launch logs clean on both nodes for both arms.
- **Cluster health**: `RunnerReady 2/2` after both launches, zero
  `RunnerFailed`.

### 9.1 Honest anomalies (none blocking, all disclosed)

1. **B r1 reasoning-token outlier: 60 vs 32 / 34 / 34.** One arm-B run
   emitted roughly double the reasoning tokens of the other three. This
   is the **same trajectory-variance class** documented for the sibling
   `hc_expand` kernel (`docs/hc-expand-kernel-ab-2026-08-24.md` §12,
   where arm A ~56 vs arm B ~32 reasoning tokens was observed and
   analysed). The fused path is a numerically distinct code path
   (fp32-exact to ~4.5e-07 mean rel err, §1.1), so a ulp-level
   accumulation difference can flip an argmax token mid-reasoning,
   changing *which* intermediate chain is walked without changing the
   destination. **Final answers are identical across all four runs**, so
   this is a trajectory difference, not a correctness regression. It is
   NOT root-caused here; closing it would need per-step logit diffing,
   which is out of scope for a ship/no-ship A/B whose criterion is needle
   recall + no degeneration.
2. **Depth spread 235 tokens (0.33%) across runs.** Not normalized for.
   Same order of magnitude as the measured effect's uncertainty.
3. **Arm B's own spread (0.3722%) is ~11× arm A's (0.0335%),** and it
   exceeds the smallest-framing margin above threshold (+0.18%). See
   §15.
4. **Transfer is ~70% of prediction** (+1.89% measured vs +2.73%
   predicted). See §8.

---

## 10. Decode (informational only — NOT load-bearing)

| arm | mean decode tok/s | runs |
|---|---|---|
| A | 26.4677 | 26.3937, 26.5416 |
| B | 28.8293 | 27.3900, 30.2687 |

Nominally **+8.92%**, and it is **explicitly not claimed as a result.**
With `--max-tokens 128` and EOS enabled, the model correctly answered the
needle question and stopped after only **41-69 completion tokens**,
giving a decode window of ~1.5-2.5 s per probe. Arm B's decode spread is
**2.88 tok/s** across just two runs — larger than most real decode
effects this campaign has measured. This measurement is noise-dominated
and is recorded only to show decode did not *regress*. The ship verdict
rests on prefill alone.

---

## 11. Verdict against pre-registered criteria

| criterion | pre-registered | measured | pass? |
|---|---|---|---|
| Prefill improvement | `mean(B) >= mean(A) × 1.015` | **+1.8880%** mean-mean (383.2700 vs threshold 381.8106); worst framing +1.6814% | **✅ PASS** |
| Needle exact | must recover FALCON-MERCURY-7749 | ✓ 4/4, byte-identical responses | ✅ |
| No BOS spam / U+FFFD | none in generated text | none observed | ✅ |
| Decode not broken | no regression | A 26.4677 → B 28.8293 (noise-dominated, no regression) | ✅ |
| Cluster serving healthy | RunnerReady 2/2, no RunnerFailed | ✓ both arms, zero RunnerFailed | ✅ |
| Deploy discriminates the arms | env + installed code differ, only in the intended variable | ✓ §3.3, §3.4 (8/8 PIDs) | ✅ |

**VERDICT: SHIP** — all pre-registered framings pass and quality is
clean.

**This session did NOT flip anything to production.** The default remains
OFF, `start_cluster.sh` carries only the opt-in forwarding of
`782c8cf97`, and the mlx-lm submodule pointer bump remains uncommitted on
the laptop. Supervisor **GO** was subsequently given and the flip was
executed and verified the same morning — recorded in §14.

---

## 12. Files touched this session (docs-only)

- `docs/hc-collapse-kernel-ab-2026-08-25.md` — this file (new)
- `docs/PERFORMANCE_HISTORY.md` — new `2026-08-25` hc_collapse section

No code, no `start_cluster.sh`, no `mlx-lm/`, no bench harness, no runner
configuration was touched by this doc commit. The working tree's
`M mlx-lm` (submodule pointer at `8d5de18`, the arm-B vehicle) is
deliberately **left unstaged** — see §3.1.

Bench artifacts: `/tmp/hccol_arm{A,B}_100k{,_r2}.json`.

Prior art cited, not repeated:
`docs/hc-collapse-roofline-2026-08-24.md` (kernel + laptop validation +
roofline), `docs/hc-expand-kernel-ab-2026-08-24.md` (sibling kernel's
shipped A/B, whose structure this doc mirrors),
`docs/xcode-removal-launcher-clt-fallback-2026-08-25.md` (§2.1
incident).

---

## 13. Rollback recipe (as-shipped version is §14.5)

The flip **has** shipped, so the live recipe is §14.5: launching with
`EXO_DSV4_HC_COLLAPSE_KERNEL=0` restores the classic op path
(bit-identical to arm A), and the submodule pointer can be returned to
`7a1a4e86`.

---

## 14. Production flip + final verification (appended post-GO)

Supervisor **GO** was received at **~11:05** on 2026-08-25. The ship was
executed **11:12–11:20** the same morning by a follow-up ship task. This
section is the record of what was changed, how production was relaunched,
and how kernel-ON was verified on the live cluster.

### 14.1 The flip commit + mlx-lm fast-forward

**mlx-lm (fork).** Branch `main` was fast-forwarded from `7a1a4e8` to
`8d5de181d09cc9ce9e5955f5be5fe4708f86258e` — the exact commit the A/B's
arm B ran — and pushed. `git ls-remote` then confirmed remote
`main = 8d5de18…`, so the kernel is no longer only reachable from the
`kernel/hc-collapse-roofline` branch.

**exo ship commit `99f5f96b8bc3bd58bd72f6f4c793e899464ad639`.** Two
changes, both minimal:

| # | file | change |
|---|---|---|
| 1 | `mlx-lm` (submodule pin) | `7a1a4e868` → `8d5de181d` |
| 2 | `start_cluster.sh` | default flip: `: "${EXO_DSV4_HC_COLLAPSE_KERNEL:=1}"`, comment updated (revert = set `0`) |

The `:=` form means the gate is **promoted to ON by default** but any
explicit value in the environment still wins — which is what makes the
rollback in §14.5 a zero-edit operation.

### 14.2 Production relaunch timeline

Relaunched under tmux session `hccol_prodship` with the **bare** command —
deliberately **no** explicit `EXO_DSV4_HC_COLLAPSE_KERNEL` in the
environment, so that the kernel being live afterwards is itself the proof
that the `start_cluster.sh` default flip took effect:

```
EXO_SPECULATIVE=0 EXO_DSV4_MTP=0 ./start_cluster.sh
```

| step | time | result |
|---|---|---|
| T0 — launch | **11:12:38** | started |
| push-check | — | **pass** on `99f5f96b8` |
| rsync + dependency sync | — | **clean** (both nodes) |
| node commit sync | — | `Nodes synchronized on commit 99f5f96b8.` |
| health | — | **HEALTHY** (Nodes 2, Identities 2) |
| **READY (2/2)** | **11:19:55** | **~7.3 min** end-to-end |

### 14.3 Post-relaunch verification

**Deployed SHAs (both nodes):**

|  | node m4-1 | node m4-2 |
|---|---|---|
| `exo` HEAD | `99f5f96b8` | `99f5f96b8` |
| `mlx-lm` checkout | `8d5de181d` | `8d5de181d` |

**Venv verification** — the same grep that discriminated the arms in §3.3,
now run against production:

| check | node m4-1 | node m4-2 |
|---|---|---|
| `grep -c HC_COLLAPSE` in venv-installed `mlx_lm` | **3** | **3** |

Three hits — i.e. the *arm-B* installed code, not a stale `site-packages`
copy.

**Runner env, per-PID** — `ps eww` on **all 8** production runner PIDs:

| node | runner PIDs | `EXO_DSV4_HC_COLLAPSE_KERNEL` |
|---|---|---|
| m4-1 | `25937`, `25938`, `25939`, `25949` | **`1`** ✅ (4/4) |
| m4-2 | `27261`, `27262`, `27263`, `27272` | **`1`** ✅ (4/4) |

Every one of the 8 PIDs also carries `EXO_DSV4_HC_EXPAND_KERNEL=1` (the
sibling kernel, shipped 2026-08-24), `EXO_SPECULATIVE=0`,
`EXO_DSV4_MTP=0`, `EXO_DSV4_DSPARK=1`. None of these were passed
explicitly for the collapse gate — the script default supplied it.

**Cluster `/state`:**

| field | value |
|---|---|
| runner status | **2× `RunnerReady`** |
| instance | `DSv4-Flash-0731`, TP `worldSize = 2` |

### 14.4 Serving smoke probe

Two probes at temperature 0 against the production endpoint.

| probe | `max_tokens` | `finish_reason` | usage / notes |
|---|---|---|---|
| 1 | **160** | `length` | truncated — 117 **reasoning** tokens ate the budget; content = `PROD-OK-HCCOLLAPSE` plus the start of the explanation |
| 2 (re-probe) | **400** | **`stop`** | **206 completion tokens**, complete answer |

Probe 1 is *not* a defect: at temp 0 the model spent most of a 160-token
budget on reasoning tokens, so the visible answer was cut mid-sentence.
Re-probing with a 400-token budget produced a clean terminated response.

Probe 2 content, **verbatim**:

> PROD-OK-HCCOLLAPSE
>
> A fused GPU kernel combines multiple computational operations (such as element-wise transformations, reductions, or matrix multiplications) into a single kernel launch, avoiding the overhead of separate launches and intermediate global memory reads/writes. By keeping intermediate data in fast on-chip registers or shared memory, it dramatically reduces memory bandwidth usage and latency, often yielding significant speedups for deep learning and scientific workloads.

**Zero U+FFFD (`�`), no BOS spam.** The sentinel string came back
byte-exact and the free-form continuation is coherent — the same quality
bar applied in §9.

### 14.5 Rollback recipe (production, as shipped)

One command, no edits, no revert — the `:=` default yields to an explicit
value:

```
EXO_DSV4_HC_COLLAPSE_KERNEL=0 EXO_SPECULATIVE=0 EXO_DSV4_MTP=0 ./start_cluster.sh
```

That relaunches on the classic op path, **bit-identical to arm A** of this
A/B. To make the rollback permanent, revert the single `:=` line in
`start_cluster.sh` and/or return the `mlx-lm` submodule pin to `7a1a4e868`.

---

## 15. Limitations honestly stated

- **n=2 per arm, and the tightest passing framing's margin is smaller
  than arm B's own spread.** The conservative framing clears the +1.5%
  gate by **+0.18 percentage points**, while arm B's run-to-run spread is
  **0.3722%** — i.e. the margin is *inside* B's own variability. The mean
  framing (+1.89%) has more headroom, and every framing passes, but with
  only two runs per arm this result should be read as "the lever is a
  real ~+1.5-2% win" rather than "the lever is +1.89% ± small". More runs
  would tighten this; the pre-registered design allowed two.
- **Arm A is suspiciously tight (0.0335% spread) and arm B is 11× looser
  (0.3722%).** Two runs cannot distinguish "arm A got lucky" from "the
  kernel path has genuinely higher variance." Neither explanation
  threatens the verdict, but the asymmetry is not explained.
- **Depth is not exactly constant** — 235 tokens (0.33%) of spread across
  the four probes, unnormalized.
- **Only one depth measured (~70.5K real tokens).** Whether the win holds
  at 300K/500K is untested. The mechanism is per-layer per-token, so a
  proportional win is *expected* but not measured.
- **~70% transfer from the laptop prediction is unexplained** (§8). This
  weakens span-share arithmetic as a forecasting tool on this cluster,
  even though the sibling hc_expand kernel's prediction landed almost
  exactly.
- **Decode is not a result** (§10): 41-69-token windows, arm B spread
  2.88 tok/s. Informational only.
- **Reasoning-length outlier not root-caused** (§9.1 item 1). Same class
  as the documented hc_expand trajectory delta; final answers identical.
- **The arm-B vehicle was an uncommitted submodule pointer** (§3.1).
  Defensible — the kernel commit is published on the fork branch and only
  the pointer was local — but it means reproducing arm B requires
  checking out `8d5de18` in `mlx-lm` by hand until the pointer bump is
  committed by the ship task.
- **PP-mode not tested.** This cluster is TP-only per the 2026-08-16
  decision. The kernel sits in a path exercised under both, so the win
  probably transfers, but this is not verified.
- **The bench harness is prior art**, used unmodified and not re-audited
  this session.
