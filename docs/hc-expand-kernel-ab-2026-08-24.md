# hc_expand fused Metal kernel — live A/B on 2026-08-24 — SHIP

**Status: LIVE A/B COMPLETE. VERDICT — SHIP (default flipped, cluster now
production on kernel-on).**

`EXO_DSV4_HC_EXPAND_KERNEL=1` gates a fused Metal kernel for the
`HyperConnection.expand` op used inside every layer's `attn_residual`
and `ffn_residual` computation (see mlx-lm `mlx_lm/models/
hyper_connection.py`, gate at `_HC_EXPAND_KERNEL_ENABLED`). On this
cluster, in a 2-run A/B at ~70K real prompt tokens (`target_tokens=100000`
via `bench/phase3_precheck_depth_throughput.py`), the kernel path is:

- **Arm A (kernel OFF, unset)**: mean **359.89 tok/s** prefill (355.03, 364.75)
- **Arm B (kernel ON)**: mean **373.80 tok/s** prefill (373.18, 374.43)
- **Delta**: **+13.91 tok/s (+3.87%)** mean-to-mean; +5.11% B1-A1, +2.65%
  B2-A2 pairwise
- **Conservative bound** (worst B vs best A, both arms' extremes): still
  **+2.31%** — every framing clears the pre-registered +1.5% ship threshold
- **Quality**: needle FALCON-MERCURY-7749 recovered exact on all 4 probes,
  no U+FFFD, no BOS spam

The pre-registered ship criterion was **`armB_prefill >= armA_prefill *
1.015`** (per task brief). This was cleared decisively at the mean level
(+3.87%), pairwise on both run-pairs (both above +1.5%), and even in the
"worst B vs best A" conservative framing (+2.31%). Quality is clean. **The
lever ships**; the default has been flipped in `start_cluster.sh` and the
cluster is now serving production with `EXO_DSV4_HC_EXPAND_KERNEL=1`.

Arm B variance was tight (spread 1.25 tok/s = 0.33% of mean); arm A had a
single low outlier on the first run (spread 9.72 tok/s = 2.70%) — the arm
A repeat landed at 364.75, matching the P5 known-good baseline of 366.5
tok/s at this depth exactly, so the A1 low was noise and not a deploy
problem. The +5.11% B1-A1 delta is therefore the upper end of the plausible
range; the mean-based +3.87% is the honest headline number.

The measured e2e gain (~+3.9%) matches the pre-registered expectation from
the T10 span-share math almost exactly: `hc_expand` was measured at 4.4%
of prefill wall time at 220K (`docs/t10-final-decomposition-closed-2026-08-22.md`
Check 1) and the fused kernel is ~7-8x faster than the op path at true
prefill shape (laptop microbench, `~3645µs → 421µs`, 8.66x reduction). A
naive `span_share × kernel_reduction` estimate gives `4.4% × (1 - 1/8.66)
= 3.9%`, matching the +3.87% measured on the live cluster. This is *not*
the bf16-comb-in-hot-path variant rejected earlier on quality grounds
(`docs/hc-expand-rejection-relitigated-multiseed-2026-08-22.md`, 1.08%
mean rel err — dead); this is a fp32-accumulate fused kernel with
laptop-measured 2.77e-7 mean rel err vs the reference op path (i.e.
fp32-exact class).

---

## 0. What was asked vs what happened

| step | status |
|---|---|
| 0. Sanity: cluster health + docs re-read | **DONE** — cluster healthy pre-launch, both nodes RunnerReady, one DSv4-Flash-0731 instance placed, no active tasks; read `known-good-prefill-baseline-2026-08-21.md`, `p3-followup-poolgrow-ab-2026-08-23.md` §10-13, `t10-final-decomposition-closed-2026-08-22.md`, `hc-expand-rejection-relitigated-multiseed-2026-08-22.md`, plus prefill-tuning skill's PITFALL sections on token accounting |
| 1. Env forwarding commit for `EXO_DSV4_HC_EXPAND_KERNEL` | **DONE** — exo `e3df799c0`, one-liner alongside the sibling pool-grow forwarding at `start_cluster.sh:2140`, matches the exact pattern §2 of the pool-grow doc identified as necessary |
| 2. Deploy both nodes at `e3df799c0`, mlx-lm submodule at `7a1a4e8` | **DONE** — verified via `git rev-parse HEAD` + `git submodule status` on both nodes; `mlx_lm.models.hyper_connection` on both nodes' venv imports the new symbol `_make_hc_expand_kernel` (Section 3.1 evidence) |
| 3. Arm A probe (unset env, verified absent on both runner PIDs) | **DONE** — 2 runs, §11 |
| 4. Arm B probe (env=1, verified present on both runner PIDs) | **DONE** — 2 runs, §11; needle-quality check §12 |
| 5. Verdict vs pre-registered criteria | **DONE** — §13, SHIP |
| 6a. Default flip in `start_cluster.sh` | **DONE** — added `: "${EXO_DSV4_HC_EXPAND_KERNEL:=1}"` beside forwarding, comment cites this doc |
| 6b. Cluster left serving in arm-B config (production) | **DONE** — §14 |
| 7. This doc + `PERFORMANCE_HISTORY.md` entry | **DONE** — see also §3.1 of `PERFORMANCE_HISTORY.md` |

---

## 1. The code change (already shipped by a prior worker)

The kernel and its gate live in mlx-lm (submodule pin `7a1a4e8`,
`mlx_lm/models/hyper_connection.py`), landed on the exo main branch by
the submodule bump commit `ecce148ff` ("bump mlx-lm: env-gated fused
Metal kernel for hc_expand (default OFF)"). Not changed by this A/B
session — this session only added the env forwarding (`e3df799c0`) and,
after the ship verdict, the default flip (see below).

The gate is:

```python
_HC_EXPAND_KERNEL_ENABLED = (
    os.environ.get("EXO_DSV4_HC_EXPAND_KERNEL") == "1"
)
_hc_expand_kernel = (
    _make_hc_expand_kernel() if _HC_EXPAND_KERNEL_ENABLED else None
)

def hc_expand(x, residual, post, comb):
    # Default OFF: EXO_DSV4_HC_EXPAND_KERNEL unset ⇒ this is the pre-kernel op path
    if _hc_expand_kernel is None:
        return _hc_expand_op(x, residual, post, comb)
    ...
```

Meaning: unset env is bit-identical to the pre-kernel op (verified
laptop-side by the prior worker, max_abs=0.0). This bit-identity is
what makes arm A a proper reference arm for the A/B without needing to
roll the submodule back.

---

## 2. Env forwarding (real prior blocker, closed here)

`start_cluster.sh` does not forward the ambient environment through to
the spawned runner processes; runner-visible variables must be explicitly
allowlisted into `EXO_ENV` (see `start_cluster.sh:2118-2135` for the
sibling pool-grow forwarding, added for the pool-grow A/B and documented
in `docs/p3-followup-poolgrow-ab-2026-08-23.md` §2 as the exact miss that
silently null'd the pool-grow A/B on its first attempt).

Without this addition, `EXO_DSV4_HC_EXPAND_KERNEL=1 ./start_cluster.sh`
would launch runners whose environment lacked the var entirely, arm B
would silently be arm A, and the A/B result would be a false null. The
skill file `exo-cluster-deployment` calls this out as a repeated pitfall
(env-var propagation recurrence).

The one-liner added in commit `e3df799c0`, placed beside the sibling
pool-grow forwarding at `start_cluster.sh:2140`:

```sh
    # EXO_DSV4_HC_EXPAND_KERNEL: fused Metal kernel for HyperConnection expand
    # (layer.attn_residual / ffn_residual). Env-gated in mlx-lm/mlx_lm/models/
    # hyper_connection.py; default OFF is bit-identical to today's op path.
    # Opt-in forwarding only -- unset var leaves the production launch command
    # line unchanged (matches the pool-grow forwarding above), so an unset here
    # is provably arm A in an A/B. See docs/hc-expand-kernel-ab-2026-08-24.md.
    [ -n "${EXO_DSV4_HC_EXPAND_KERNEL:-}" ] && EXO_ENV="$EXO_ENV EXO_DSV4_HC_EXPAND_KERNEL=$EXO_DSV4_HC_EXPAND_KERNEL"
```

(After the ship verdict, that block was extended with a
`: "${EXO_DSV4_HC_EXPAND_KERNEL:=1}"` line so the script's own default is
now 1; the forwarding one-liner then unconditionally exports it. See §14.)

---

## 3. Deploy verification

### 3.1 Deployed SHAs (both nodes, both arms)

Verified via `git rev-parse HEAD` and `git submodule status` on both
`adams-mac-studio-m4-1.local` and `adams-mac-studio-m4-2.local` after
each of the three relaunches (arm A, arm B, arm A repeat):

|  | node m4-1 | node m4-2 |
|---|---|---|
| `exo` HEAD | `e3df799c04a6d0a5b30c5233fa952f37aa3fa37f` | same |
| `mlx-lm` submodule HEAD | `7a1a4e868564f4a99e5784711367dfc1c09b9bf5` | same |
| `mlx` submodule HEAD | `e40a416b20851d118b061b3a57d8cab70f5756de` | same (unchanged this session) |

Both nodes' launch logs report `Nodes synchronized on commit e3df799c0.`
before `Waiting for 2 DeepSeek V4 runner(s) to become Ready…READY (2/2)`.

Runtime kernel-module check (both nodes):

```
$ .venv/bin/python -c 'import mlx_lm.models.hyper_connection as h; \
    print(hasattr(h, "_make_hc_expand_kernel"), inspect.getfile(h))'
True /Users/adam.durham/repos/exo/.venv/lib/python3.13/site-packages/mlx_lm/models/hyper_connection.py
```

The new symbol `_make_hc_expand_kernel` is present in the venv-installed
`mlx_lm` on both nodes, so we know the runner import is actually seeing
the new code (not a stale `site-packages` copy — the exo-sdpa-fusion-
analysis skill documents that stale-bytecode class of pitfall for this
exact submodule).

### 3.2 Runner env (the §2 check that makes or breaks the A/B)

`ps eww <runner_pid>` on the real `.venv/bin/python -m exo -v` process,
both nodes, both arms and both repeats:

| arm | node | runner pid | `EXO_DSV4_HC_EXPAND_KERNEL` | `EXO_DSV4_MTP` | `EXO_DSV4_DSPARK` | `EXO_SPECULATIVE` |
|---|---|---|---|---|---|---|
| A (r1) | m4-1 | 21459 | **absent** ✅ | `0` | `1` | `0` |
| A (r1) | m4-2 | 19517 | **absent** ✅ | `0` | `1` | `0` |
| B (r1) | m4-1 | 25329 | **`1`** ✅ | `0` | `1` | `0` |
| B (r1) | m4-2 | 24041 | **`1`** ✅ | `0` | `1` | `0` |
| A (r2) | m4-1 | (verified) | **absent** ✅ | `0` | (n/a in grep) | `0` |
| A (r2) | m4-2 | (verified) | **absent** ✅ | `0` | (n/a in grep) | `0` |

This is the check §2 said would catch a silent null. It passed in the
discriminating direction: the arms are genuinely different, and differ
*only* in the intended variable. Production-relevant DSv4 flags
(`EXO_DSV4_MTP=0`, `EXO_SPECULATIVE=0`, `EXO_DSV4_DSPARK=1`) match the
task-brief production config across all arms.

### 3.3 Relaunch justification (task-brief-mandated statement)

A full daemon relaunch is REQUIRED for this A/B because (a) the new
`hyper_connection.py` symbol must load into a fresh Python interpreter
inside each runner process, and (b) the ON arm needs a process-level env
var that cannot be injected into live daemons (they inherited a fixed
env from the prior launch). This matches the pool-grow A/B precedent
exactly (`docs/p3-followup-poolgrow-ab-2026-08-23.md` §10.1). No live
daemon can be re-configured mid-flight to add this env var, so a bounded
relaunch per arm is unavoidable.

### 3.4 Relaunch timeline

Three full relaunches, all clean, all `EXIT=0`, no crashes, no runner
deaths, no `signal=9`, no Metal timeouts, no OOM. All via `tmux
new-session -d` in the background from the laptop (canonical source) so
they survived the tool-call timeout budget:

| # | arm | started | READY (2/2) | duration | commit at launch |
|---|---|---|---|---|---|
| 1 | initial (aborted at 300s bash timeout, mid-build; recovered cleanly by relaunch #2) | 14:00:34 | — | ~n/a | e3df799c0 |
| 2 | **A** (`EXO_SPECULATIVE=0 EXO_DSV4_MTP=0 ./start_cluster.sh`) | 14:22:47 | ~14:36:00 | ~13 min | e3df799c0 |
| 3 | **B** (`EXO_SPECULATIVE=0 EXO_DSV4_MTP=0 EXO_DSV4_HC_EXPAND_KERNEL=1 ./start_cluster.sh`) | 14:41:02 | ~14:55:30 | ~14 min | e3df799c0 |
| 4 | **A repeat** (identical to #2) | 15:04:04 | ~15:15:30 | ~11 min | e3df799c0 |

The initial launch #1 was killed mid-build by the tool-call foreground
timeout — this left m4-1 with zero exo processes and m4-2 still running
the previous production runner at the OLD SHA (a partial split-brain, per
the cluster-deployment skill's documented pattern). The `tmux new-session
-d` pattern used for launches #2-4 sidesteps this class of failure
entirely (matches the recovery pattern the skill recommends). No harm
done: launch #2 re-killed the stale m4-2 runner and successfully brought
both nodes back to the new SHA before any probe fired.

---

## 4. Pre-registered ship criterion (from task brief, verbatim)

> SHIP if `armB_prefill >= armA_prefill × 1.015` (>= +1.5%) AND quality
> clean. If 0 to +1.5%: honest marginal null — keep opt-in, leave cluster
> on arm A config, still document. If negative or quality broken:
> null/regression — same, flag loudly.

Quality clean is defined as: needle recall exact, `finish_reason=stop`,
no BOS spam / U+FFFD garbling, generated text shown in report.

---

## 5. Method

- **Probe**: `bench/phase3_precheck_depth_throughput.py` (unmodified, the
  hardened one that reads `usage.prompt_tokens` for accounting-drift
  detection but uses offline tokenizer-ground-truth for the numerator, per
  Section 55 of that file's own docstring). Wall-clock TTFT is the
  denominator. This is the pattern the exo-dsv4-prefill-tuning skill's
  PITFALL section explicitly mandates ("NEVER derive tok/s from a
  server-reported token count — tokenize the prompt OFFLINE for the
  numerator, use wall-clock for the denominator").
- **Command** (identical across all 4 probes):
  `.venv/bin/python bench/phase3_precheck_depth_throughput.py \
      --model deepseek-ai/DeepSeek-V4-Flash-0731 --targets 100000 \
      --max-tokens 128 --json-out /tmp/hcexp_arm{A,B}_100k[_r2].json`
- **Prompt shape**: needle-in-haystack (needle: FALCON-MERCURY-7749),
  `target_tokens=100000` → ~400K chars → **~70.5K real prompt tokens**
  after tokenizer expansion (this is the standard artifact of the
  `target_chars = target_tokens * 4` estimator used by every prior probe
  on this cluster — the same probe with the same target produced 70,547
  real tokens for the P5 baseline, so my numbers are directly comparable).
- **Depth invariance across the 4 probes**: 70,441 to 70,609 tokens (spread
  168 tokens = 0.24% of depth), so no depth-normalization needed.
- **What each run measures**: prefill tok/s = `prompt_tokens_offline /
  TTFT_wallclock`; ttft_s = time to first streamed token from POST start;
  decode measurement is secondary and structurally weak (`max-tokens=128`
  + EOS-enabled `/v1/chat/completions` produced only 41-67 completion
  tokens because the model correctly answered the needle question and
  stopped — see §12 for how this is handled).

---

## 6. Pre-registered arm-A sanity gate (task brief)

> Expect ~360-370 tok/s prefill (P5 measured 366.5 today on same-code-
> different-docs). If arm A deviates >5% from that, STOP and investigate
> before proceeding (deploy problem, not noise).

- Arm A r1: 355.03 tok/s, deviation from P5 baseline: **-3.13%** (within 5% tolerance ✓, proceeded)
- Arm A r2: 364.75 tok/s, deviation from P5 baseline: **-0.48%** (essentially exact match ✓)
- Arm A mean: 359.89 tok/s, deviation from P5 baseline: **-1.80%** (within 5% tolerance ✓)

The arm A r1 was a slight low outlier; the r2 landed on the P5 baseline
almost exactly. This is the "run-to-run noise" case the task-brief
guardrail anticipated, hence the ONE-repeat-of-each rule was invoked
(§9 below). No investigation needed — deploy is clean.

---

## 7. Skipped

(Kept for parity with the pool-grow doc structure; nothing was skipped
this session beyond the tightening described in §12.)

---

## 8. Files touched this session

- `start_cluster.sh` (exo repo)
  - Commit `e3df799c0`: env forwarding one-liner + comment
  - Commit `deb1c8a6d` (this session's post-verdict commit):
    `: "${EXO_DSV4_HC_EXPAND_KERNEL:=1}"` script default + updated comment
- `docs/hc-expand-kernel-ab-2026-08-24.md` — this file (new)
- `docs/PERFORMANCE_HISTORY.md` — added §3.1 entry with real numbers

Nothing else was touched. No code, no bench harnesses, no runner
configuration. The mlx-lm submodule remains at `7a1a4e8` — the fused
kernel itself was shipped by a prior worker; this session only measured
its live effect and, on positive verdict, flipped the runtime gate's
default.

Bench artifacts: `/tmp/hcexp_arm{A,B}_100k[_r2].{json,log}` (raw stdout +
full per-run JSON including every gap / usage field),
`/tmp/start_cluster_arm{A,A2,B}[_r2].log` (relaunch logs).

---

## 9. Note on repeats (why 2×2 not 1×1)

The task brief allows ONE repeat of each arm (max 4 probes total) "if the
A/B is inconclusive due to run-to-run noise." The first pass (arm A r1:
355.03 tok/s, arm B r1: 373.18 tok/s = **+5.11%**) already cleared the
+1.5% ship threshold with headroom, so the strict letter of the brief
would allow stopping there. But arm A r1 landed 3.13% below the P5
baseline of 366.5 tok/s, which is inside the ±5% sanity gate but
noticeable. If arm A had matched P5 exactly on r1 (366.5), the pairwise
delta would have been only +1.8% — still above threshold but tight.

Rather than declare victory on a single pair, ran ONE repeat of each arm
per the brief's inconclusive-noise clause. Result:

- Arm B repeat (immediate, no relaunch needed — cluster already in arm B
  config): 374.43 tok/s — essentially identical to B r1 (delta 1.25
  tok/s, 0.33%). Arm B is tight.
- Arm A repeat (after relaunching to arm A config): 364.75 tok/s — 9.72
  tok/s (2.70%) higher than A r1, matching the P5 baseline. Confirms A r1
  was a slight low outlier, not a deploy artifact.

Mean-based delta (2 runs each): **+3.87%**. Even the pessimistic "worst B
vs best A" bound is +2.31%. No further repeats needed; the null-repeats
budget (4 probes max) is used and not exceeded.

---

## 10. Runner env — the §2 check that makes or breaks the A/B

See §3.2 above (both nodes, all three runner-startup snapshots).

---

## 11. Raw results — all four probes

Reported as prefill tok/s = `prompt_tokens_offline / TTFT_wallclock`, per
the anti-accounting-drift rule from `exo-dsv4-prefill-tuning`'s PITFALL
section.

| run | depth (tokens) | prefill tok/s | TTFT (s) | decode tok/s | reasoning_tokens | needle |
|---|---|---|---|---|---|---|
| A r1 | 70,597 | **355.03** | 198.85 | 27.14 | 55 | ✓ FALCON-MERCURY-7749 |
| A r2 | 70,555 | **364.75** | 193.43 | 27.00 | 58 | ✓ FALCON-MERCURY-7749 |
| B r1 | 70,441 | **373.18** | 188.76 | 26.79 | 32 | ✓ FALCON-MERCURY-7749 |
| B r2 | 70,609 | **374.43** | 188.58 | 26.34 | 32 | ✓ FALCON-MERCURY-7749 |

**Aggregate**:

| arm | mean prefill | min | max | spread (%mean) |
|---|---|---|---|---|
| A | 359.89 tok/s | 355.03 | 364.75 | 9.72 (2.70%) |
| B | 373.80 tok/s | 373.18 | 374.43 | 1.25 (0.33%) |

**Deltas (B − A)**:

| framing | Δ tok/s | Δ % |
|---|---|---|
| pairwise B1−A1 | +18.15 | **+5.11%** |
| pairwise B2−A2 | +9.68 | **+2.65%** |
| mean(B) − mean(A) | +13.91 | **+3.87%** |
| conservative: worst B (373.18) vs best A (364.75) | +8.43 | **+2.31%** |
| optimistic: best B (374.43) vs worst A (355.03) | +19.39 | +5.46% |

Every framing clears the +1.5% ship threshold. Arm A r1 is a low outlier
(only 355.03 vs the other three sitting between 364.75 and 374.43);
excluding it, the delta between arm A r2 and arm B mean would be
+9.05 tok/s (+2.48%). Even in that outlier-excluded world, the lever
still ships. The mean-based +3.87% is the honest headline.

**Decode weakness disclosed**: with `--max-tokens 128` and EOS enabled at
`/v1/chat/completions`, the model correctly answered the needle question
in 41-67 completion tokens (32-58 reasoning + 9 content) and stopped. The
decode window was only 1.5-2.5s per probe, so the decode tok/s numbers
(26.34-27.14 tok/s) are structurally weak but consistent with prior
baselines at this depth (~28 tok/s @100K in the pool-grow arm A baseline).
Notably decode tok/s **does not regress** between arms (arm A mean 27.07,
arm B mean 26.57, delta -1.85% — inside the noise band for 41-67 token
windows). The task brief prioritizes prefill, and the decode measurement
is not load-bearing for the ship verdict.

---

## 12. Quality check (needle + generation-text evidence)

Per the campaign rule "never quote t/s without the generated text":

- **Prompt** (identical shape across all 4 probes, only the random needle
  position varies): the standard `bench/phase3_precheck_depth_throughput
  .py::build_prompt` needle-in-haystack: ~400K chars of filler paragraphs
  from a small topic pool (`FILLER_TOPICS`), needle string
  `"The secret code for project Nightingale is: FALCON-MERCURY-7749."`
  placed at a random position in the middle third, followed by the
  question "What is the secret code for project Nightingale? Answer with
  just the code, nothing else."

- **Response** (all 4 probes): `'FALCON-MERCURY-7749'`

- **Reasoning stream length** (streamed via `reasoning_content`): arm A r1
  55 tokens, arm A r2 58 tokens, arm B r1 32 tokens, arm B r2 32 tokens.
  Both arms produce coherent chain-of-thought reasoning consistent with
  DSv4-Flash's normal think-block behaviour. No U+FFFD, no `<|begin_of_
  sentence|>` leakage, no infinite reasoning loop.

- **`finish_reason`**: `stop` (EOS emitted after the answer). This is the
  intended behaviour of the `/v1/chat/completions` endpoint at
  `max_tokens=128` — the model finishes answering and stops. Not `length`,
  because we did not need to force the model past its natural end for this
  quality check; the point is the needle was recovered exact, which
  requires the model to have correctly attended to the ~70.5K prompt.

- **Needle recall**: **exact on all 4 probes**. `needle_found=True` in
  every JSON output file. Kernel-on does not silently corrupt long-context
  attention.

If the fused kernel had introduced any numerical divergence bad enough to
matter, we'd expect either:
1. Needle miss (attention to the ~70.5K-token corpus corrupted at some
   layer → answer wrong / hallucinated code)
2. Degeneration (loop, U+FFFD, BOS spam)
3. Structured differences in the reasoning length (would suggest sampling
   flipped tokens due to ulp-level logit drift)

None of these fired. The 55→58 reasoning length variance between A r1 and
A r2, and the 32→32 between B r1 and B r2, are consistent with the model's
own greedy trajectory being stable within each arm; the between-arm
difference (55-58 vs 32) is a larger reasoning-length delta that is
worth flagging honestly:

**Between-arm reasoning-length delta (arm A ~56 tokens vs arm B ~32
tokens) is real, but does NOT indicate corruption.** All 4 probes reached
the same final answer (FALCON-MERCURY-7749). Two mechanisms could explain
the delta at temperature 0: (a) the fused kernel is a NEW code path
(different from the reference op path), so any ulp-level accumulation
difference in fp32 mode could flip an argmax token in the reasoning
stream, changing WHICH intermediate reasoning path is taken but not the
final answer (correctness-preserving trajectory change); or (b) genuine
run-to-run non-determinism inside the reasoning stream from Metal
kernel scheduling (this cluster runs with `MLX_GEMV_BATCH_INVARIANT=1`
which reduces but doesn't eliminate). Since both arms recover the exact
needle and neither degenerates, this is bookkept as a real difference
between kernel-on and kernel-off reasoning paths (expected) but NOT a
quality regression (verified). If a future session wants to close this
tighter, run a byte-equality check via `EXO_DSV4_MTP_REFCHECK`-style
per-step logit diffing across the two arms on a fixed prompt — that
was out of scope for this ship/no-ship A/B, whose criterion was needle
recall + no degeneration, not per-token byte identity.

---

## 13. Verdict against pre-registered criteria

| criterion | pre-registered | measured | pass? |
|---|---|---|---|
| Prefill improvement | `armB_prefill >= armA_prefill × 1.015` (>= +1.5%) | **+3.87%** (mean-mean); worst B vs best A: +2.31% | **✅ SHIP** |
| Needle exact | must recover FALCON-MERCURY-7749 | ✓ all 4 probes | ✅ |
| `finish_reason` | not garbled (`stop` or `length` acceptable) | `stop` all 4 (EOS correctly emitted after answer) | ✅ |
| No BOS spam / U+FFFD | none in generated text | none observed | ✅ |
| Decode not broken | reference: prior baselines ~28 tok/s @100K | 26.34-27.14 tok/s, no arm-to-arm regression | ✅ |
| Cluster serving healthy | RunnerReady 2/2 after each launch | RunnerReady 2/2 after each of 3 launches | ✅ |
| Arm A sanity vs P5 baseline (deploy check) | within ±5% of 366.5 tok/s | mean -1.80% deviation (r2 alone: -0.48%) | ✅ |

**VERDICT: SHIP.** All pre-registered criteria met. Default flipped in
`start_cluster.sh` (script default now `EXO_DSV4_HC_EXPAND_KERNEL=1`),
cluster left serving in arm-B configuration (this is now production).

---

## 14. Default flip + cluster left in production (arm-B) configuration

### 14.1 Flip

Preferred `start_cluster.sh` export flip over an mlx-lm-code-side default
flip, per task brief ("simpler, revertable"). This mirrors the same
choice the pool-grow A/B eventually followed (default forwarding
carrying a script-side `:=1`).

The added line (this session's post-verdict commit) at
`start_cluster.sh:2145`:

```sh
    : "${EXO_DSV4_HC_EXPAND_KERNEL:=1}"
```

Placed immediately before the pre-existing forwarding one-liner (§2 above),
so a bare `./start_cluster.sh` now defaults to kernel-on, while any launch
that explicitly exports `EXO_DSV4_HC_EXPAND_KERNEL=0` on the command line
before running the script gets the pre-kernel op path back (documented
escape hatch — bit-identical to today's arm A).

### 14.2 Cluster state — accurate handoff + final production verification

**Accuracy correction (2026-08-24 PM closing task).** An earlier draft
of this section claimed both nodes had `EXO_DSV4_HC_EXPAND_KERNEL=1`
verified at "session end" of the A/B. That was true immediately after
arm B r2 (launch #3 in §3.4), but the A/B session then ran one more
relaunch — the arm-A repeat (launch #4) — which brought the cluster
back to **arm-A** env before the A/B worker handed off. §3.2's two
arm-A r2 rows corroborate this: `EXO_DSV4_HC_EXPAND_KERNEL` was
**absent** from both runner PIDs at the end of the A/B measurement
session. So the honest state at handoff of the A/B session was:

- Both nodes: exo `e3df799c0`, mlx-lm `7a1a4e8`, mlx `e40a416b2` (unchanged)
- Both nodes: `EXO_DSV4_HC_EXPAND_KERNEL` **absent** from runner env
  (arm-A config); production overrides `EXO_SPECULATIVE=0
  EXO_DSV4_MTP=0 EXO_DSV4_DSPARK=1`
- API on `adams-mac-studio-m4-1.local:52415` — both `RunnerReady`
- No stale foreground/tmux launcher processes still running from the
  A/B session (both `hcexpA` and `hcexpB` tmux sessions killed after
  their respective probe rounds)

**The final production relaunch into the shipped (arm-B) configuration
was performed same-day by the PM's closing task**, after the default
flip (§14.1) landed on origin main. That relaunch used the same
background `tmux new-session -d` launch pattern the A/B used, with
`EXO_DSV4_HC_EXPAND_KERNEL` unset on the command line so the new script
default (`: "${EXO_DSV4_HC_EXPAND_KERNEL:=1}"` at
`start_cluster.sh:2145`) promoted it to 1, and the pre-existing
forwarding one-liner exported it into every runner's environment. The
step-4/5 verification evidence from that closing task (env `ps eww`
lines, `RunnerReady`, smoke completion text) is appended as §14.2.1
in a follow-up docs-only commit — kept separate from the default-flip
commit and the initial A/B doc commit so the git history preserves the
distinction between what the A/B session measured and what the closing
task verified in production.

### 14.2.1 Final production verification (2026-08-24 PM closing task, appended)

Fresh evidence gathered after the closing task's relaunch. Timeline:

- **15:31:09** — background `tmux new-session -d -s hc_prodrelaunch` launched
  `EXO_DSV4_MTP=0 EXO_SPECULATIVE=0 ./start_cluster.sh` on the laptop.
  The script default (this session's `deb1c8a6d` commit)
  `: "${EXO_DSV4_HC_EXPAND_KERNEL:=1}"` promoted the var to 1; no
  explicit override on the command line.
- **~15:43:35** — API on `adams-mac-studio-m4-1.local:52415` began
  responding (before this, the polls got "API not responding" because
  the launcher was still in the Rust `pyo3` rebuild + zenoh discovery
  phase; a bare `Waiting for cluster to stabilize...` was still in
  progress). This is normal for a full launcher run.
- **~15:44:16** — launcher log reported
  `Nodes synchronized on commit 302759bec.`,
  `Memory reclaim on macstudio-m4-1 complete (wired+compressor 3 GB <= 25 GB).`,
  `Memory reclaim on macstudio-m4-2 complete (wired+compressor 3 GB <= 25 GB).`,
  `Waiting for cluster to stabilize...... HEALTHY! (Nodes: 2, Identities: 2)`,
  `Auto-placing DeepSeek V4 Flash (deepseek-ai/DeepSeek-V4-Flash-0731)`,
  and finally `Waiting for 2 DeepSeek V4 runner(s) to become Ready.................... READY (2/2)`.
  Total launch time ~13 minutes (T+00:00 → T+13:07), consistent with the
  A/B session's launches #2-4 (~11-14 min each).
- **15:44:16** — tmux session `hc_prodrelaunch` exited cleanly (return
  code 0, no crash-retry loop, no orphan foreground process).

`/state` verification:

```
$ curl http://adams-mac-studio-m4-1.local:52415/state | jq '.runners'
6446d367-edb...: state={'RunnerReady': {'prefillServerPort': None}}
573c199f-fc7...: state={'RunnerReady': {'prefillServerPort': None}}
```

Both runners `RunnerReady`. The instance
`ce71f238-a68d-4e11-aae9-a678d78cc872` for
`deepseek-ai/DeepSeek-V4-Flash-0731` has both TensorShardMetadata halves
placed (`worldSize=2`, `deviceRank=0/1`, `startLayer=0 endLayer=43`
across 43 total layers, `quantization=fp8`).

`ps eww <runner_pid>` on both nodes (the check §2 said would catch a
silent null):

```
$ ssh adams-mac-studio-m4-1.local 'ps eww $(pgrep -f ".venv/bin/python -m exo -v")'
node m4-1 PID=33871
EXO_DSV4_HC_EXPAND_KERNEL=1
EXO_SPECULATIVE=0
EXO_DSV4_MTP=0
EXO_DSV4_DSPARK=1

$ ssh adams-mac-studio-m4-2.local 'ps eww $(pgrep -f ".venv/bin/python -m exo -v")'
node m4-2 PID=33826
EXO_DSV4_HC_EXPAND_KERNEL=1
EXO_SPECULATIVE=0
EXO_DSV4_MTP=0
EXO_DSV4_DSPARK=1
```

Kernel gate = 1 present on both runners; production overrides
`EXO_SPECULATIVE=0 EXO_DSV4_MTP=0 EXO_DSV4_DSPARK=1` match the A/B's
arm-B configuration. This is genuinely arm-B in production, no silent
null.

Smoke test (short chat completion for serving-sanity, NOT a benchmark):

```
$ curl -X POST http://adams-mac-studio-m4-1.local:52415/v1/chat/completions \
    -H "Content-Type: application/json" -d '{
      "model": "deepseek-ai/DeepSeek-V4-Flash-0731",
      "messages": [{"role":"user","content":
        "Reply with exactly: PROD-OK-HCEXPAND then one sentence about why the sky is blue."}],
      "max_tokens": 64, "temperature": 0.0, "stream": false }'
```

Response:

- `finish_reason: length` (hit max_tokens=64 mid-sentence — expected)
- `usage: {prompt_tokens: 26, completion_tokens: 64, total_tokens: 90,
  reasoning_tokens: 49}`
- `reasoning_content` (49 tokens): coherent thinking that parsed the
  instruction correctly (`"1.  The user asks to reply with exactly:
  \"PROD-OK-HCEXPAND then one sentence about why the sky is blue.\"
  2.  I need to output the exact string \"PROD-OK-HCEXPAND`)
- `content`: `PROD-OK-HCEXPAND The sky appears blue because` (starts
  with the exact required sentinel, then begins the explanation before
  the 64-token cap cut it off — consistent with a healthy DSv4-Flash
  emitting reasoning-then-content in that order)
- No U+FFFD, no BOS spam, no degeneration, no tool-call XML leak

The cluster is now serving production in the shipped kernel-ON
configuration. The only tmux sessions remaining on the laptop are
pre-existing peer sessions (`exo-default2`, `exo-relaunch1`, `p100k`,
`p3e`) from earlier subagents / A/B sessions — the closing task's own
`hc_prodrelaunch` session exited cleanly.

Nodes are at exo `302759bec` (this session's docs commit) — one docs-
only commit will follow this appendage to record §14.2.1 above; the
nodes' commit will lag laptop head by that single docs-only commit,
which matches this campaign's convention (nodes get resynced only at
the next real launcher run).

### 14.3 Rollback recipe (if this ever needs to be reverted)

```sh
cd ~/repos/exo && EXO_DSV4_HC_EXPAND_KERNEL=0 EXO_SPECULATIVE=0 EXO_DSV4_MTP=0 \
    ./start_cluster.sh
```

That will run the pre-kernel op path (bit-identical to arm A of this
A/B). To make the revert stick across future launches, revert the
`: "${EXO_DSV4_HC_EXPAND_KERNEL:=1}"` line in `start_cluster.sh` (this
session's default-flip commit is a single-line change, trivial to revert).

---

## 15. Limitations honestly stated

- **Only one depth measured (~70.5K real tokens, `target_tokens=100000`).**
  The task brief specified 100K. Whether the kernel's win holds at deeper
  context (300K, 500K) is not tested by this A/B. The mechanism (a per-
  layer op that runs once per token per layer during prefill) should
  scale linearly with token count, so a proportional win is expected —
  but not measured. Follow-up work if desired.
- **Only 2 runs per arm.** The task brief's "one repeat if inconclusive"
  clause was invoked and satisfied. Arm B variance was 0.33%, arm A was
  2.70% (single outlier); more runs would tighten confidence intervals
  but wouldn't change the ship verdict — the conservative bound (worst B
  vs best A) already clears +1.5%.
- **Decode measurement is structurally weak.** `max_tokens=128` + EOS
  enabled → 41-67 completion tokens → 1.5-2.5s decode window. Consistent
  with prior baselines and no arm-to-arm regression, but not a rigorous
  decode measurement. The lever's expected effect is prefill, so this is
  acceptable.
- **Reasoning-length delta between arms not root-caused.** ~56 tokens
  reasoning arm A vs ~32 tokens arm B is a real observed difference.
  Both arms produce the same final answer with no degeneration, so this
  does not block the ship, but it's an honest artifact of the kernel path
  being numerically distinct (albeit fp32-exact to laptop-microbench
  precision, 2.77e-7 mean rel err). See §12 for the analysis and how to
  tighten further if needed.
- **The bench harness itself is prior art.** The
  `phase3_precheck_depth_throughput.py` was used unmodified. Its
  offline-tokenized numerator + wall-clock denominator methodology is the
  campaign standard (per the `exo-dsv4-prefill-tuning` skill's Big One
  pitfall). Not re-audited this session.
- **PP-mode not tested.** This cluster runs TP-only per the 2026-08-16
  decision (`exo-sharding-mode-tradeoffs` skill). The kernel is in a
  path exercised under both PP and TP, so the win probably transfers,
  but this is not verified.
