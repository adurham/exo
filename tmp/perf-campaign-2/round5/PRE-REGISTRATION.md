# ROUND 5 — PRE-REGISTRATION (written BEFORE any sweep measurement)

**Written:** 2026-09-03, after Phase 0 (zero cluster cost), before boot 1 of the sweep.
**Commit this file BEFORE the first relaunch.** Any change after the first measurement
must be recorded as an amendment with a timestamp, not an edit.

---

## Phase 0 result that drives the design

### P0-1 — The histogram was NEVER EMITTED (the answer to "check the last 48h")
`EXO_DSV4_MTP_LOG_INTERVAL` is **absent** from all 8 live runner PIDs (4/node × 2 nodes,
`ps eww`). `_LOG_INTERVAL = int(os.environ.get("EXO_DSV4_MTP_LOG_INTERVAL", "0"))` at
`dsv4_mtp.py:118` is a **module-level, import-time** read → it cannot be turned on for a
running process. `start_cluster.sh:2042` only forwards it if already set in the launcher's
shell; there is no default. Result: **0 matches** for `accept_hist=|mean_accept=|k_hist|bypos`
in both nodes' live logs and in all of round1–round4's artifacts.

Per the task's own instruction ("if `_LOG_INTERVAL` was 0 they were never emitted — then
Phase 0 needs ONE short run with the interval set … if read at import, it rides on the
Phase-1 relaunch"), **it rides on the Phase-1 relaunch. No boot is spent on it alone.**

### P0-2 — `bypos` does not exist at all
The per-position curve comes only from `_ShadowStats.summary()` (`dsv4_mtp.py:552-654`,
emit at `:4521`), gated by `EXO_DSV4_SPEC_SHADOW=1`, which appears in **no** surviving
launcher on either node. So the task's "does the curve cliff after position 1" question has
**zero direct evidence** today. It is answered below by *structure* instead of by data.

### P0-3 — The only histogram on disk anywhere (stale, 2026-08-29, off-window)
`macstudio-m4-1:~/exo_von3shard0.log` — `[MTP] cycles=2120 mean_accept=2.561/3
hist=0:119,1:185,2:203,3:1613` → P(0)=5.6% P(1)=8.7% P(2)=9.6% **P(3)=76.1%**, i.e.
**85.4% draft efficiency** — versus production's measured **47%** (a=1.411/3). Two
acceptance regimes are therefore in play and they disagree by ~1.8x.

### P0-4 — Structural facts I verified myself in source (these set the prediction)
`dsv4_mtp.py:3946-3955`, read directly:
```python
if _dspark is not None:
    gamma = _dspark.block_size
    _env_gamma = os.environ.get("EXO_SPECULATIVE_GAMMA", "")
    if _env_gamma:
        ...
        if _env_gamma_int > 0:
            gamma = min(_env_gamma_int, _dspark.block_size)
```
1. **Effective γ = `min(EXO_SPECULATIVE_GAMMA, block_size)` and `block_size = 5`.** The
   arms {2,3,4,5} therefore span *exactly* the reachable range; **γ=5 is a hard ceiling**
   and any γ>5 would silently clamp to 5. Nothing above 5 is worth a boot.
2. **The DSpark head is a 3-stage block trained for width-3** — `dsv4_mtp.py:3913-3914`:
   "the DSpark head is a 3-stage block (n_stages = n_mtp_layers = 3, deepseek_v4.py)
   **trained for width-3 draft/verify**. block_size is 5 (anchor + 4)."
   **Draft positions 4 and 5 are extrapolation beyond the head's trained stage count.**
   This is the structural substitute for the missing `bypos` data: a cliff after position 3
   is predicted *by construction*, not assumed.
3. The same comment block explains the old silent-γ bug: before 2026-08-26 this branch
   "unconditionally re-bound gamma to block_size, SILENTLY IGNORING EXO_SPECULATIVE_GAMMA
   — so a launch that set EXO_SPECULATIVE_GAMMA=2 actually ran width-5 drafts."

### P0-5 — A hypothesis I formed and then REFUTED before spending a boot on it
I hypothesised the 85%-vs-47% gap was `EXO_DSV4_MTP_DEDICATED` (1 in the stale run, 0 in
production) selecting a better draft head. **Refuted by code audit:** DEDICATED has one read
site (`utils_mlx.py:362`) and overlays weights onto `model.model.mtp[0]`, the *classic* MTP
head. Production drafts with `model.model.dspark`, a **different module** that ignores
DEDICATED entirely. So DEDICATED cannot explain the gap and is **not** the lever it looked
like. Recorded so this dead end is not re-walked. (Flipping it is cheap if ever wanted —
`DeepSeek-V4-Flash-MTP-bf16` is already cached on both nodes — but it would not touch the
DSpark path production uses.)

---

## THE MODEL (fixed here; no post-hoc refitting)

Decode = `(1 + a) / cycle`. Per-position acceptance geometric with `p` for the head's
3 trained positions, damped by `d` per position beyond position 3:

    a(γ, p, d) = Σ_{k=1..γ}  p^k · d^max(0, k−3)

Calibration (`a(3)` inverted): regime **A** = production TRUE `a(3)=1.411` → `p=0.6676`;
regime **B** = stale 08-29 `a(3)=2.561` → `p=0.9230`.
Cycle from round-4 ground truth at 89K: `cycle(3)=68.85 ms`, `verify=56.1 ms` (81.5%),
non-verify `12.75 ms / 3 rows = 4.25 ms per draft row` → `cycle(γ) = 56.1 + 4.25·γ`
(V-flat: batched verify at M=γ+1 reads weights once, so marginal per-row verify ≈ 0 —
the round-4 finding that the verify GPU is 117% busy on real compute supports this).

## PRE-REGISTERED PREDICTION MATRIX (ratio vs the γ=3 control)

| regime | damping | γ=2 | γ=3 | γ=4 | γ=5 | ordering | clears 1.10x? |
|---|---|---|---|---|---|---|---|
| A (true, p=.668) | d=1.0 no-cliff | 0.934 | 1.000 | **1.019** | 1.012 | γ4>γ5>γ3>γ2 | **NO** |
| A (true, p=.668) | d=0.5 soft | 0.934 | **1.000** | 0.981 | 0.939 | γ3>γ4>γ5>γ2 | **NO** |
| A (true, p=.668) | d=0.15 hard | 0.934 | **1.000** | 0.954 | 0.902 | γ3>γ4>γ2>γ5 | **NO** |
| B (stale, p=.923) | d=1.0 no-cliff | 0.830 | 1.000 | 1.134 | **1.239** | γ5>γ4>γ3>γ2 | **YES** |
| B (stale, p=.923) | d=0.5 soft | 0.830 | 1.000 | **1.038** | 1.023 | γ4>γ5>γ3>γ2 | **NO** |
| B (stale, p=.923) | d=0.15 hard | 0.830 | **1.000** | 0.971 | 0.921 | γ3>γ4>γ5>γ2 | **NO** |

### HEADLINE PRE-REGISTERED PREDICTION
1. **No arm clears the 1.10x ship band.** 5 of 6 cells say NO. The single YES cell
   (regime B × no-cliff) requires *both* that production acceptance is really ~85% *and*
   that the width-3-trained head extrapolates to width 5 with no degradation — and P0-4.2
   is direct source evidence against the second. **Predicted outcome: NO SHIP CANDIDATE;
   γ=3 stands.**
2. **Predicted ordering: γ3 ≥ γ4 > γ5 > γ2** (the modal ordering across cells).
3. **γ=2 is predicted WORST in every single cell** — this directly contradicts the May-2026
   record, in which γ=2 was champion and "γ=3 is −18% vs γ=2". The reconciliation is the
   async-fence fix (08-22) + batched verify (08-27) removing the per-row verify cost that
   made large γ expensive. **If γ=2 wins, my model is wrong and the May regime never
   actually ended** — that is a real finding, not a failed round.
4. **Hard ceiling on regime A:** `a(∞) = p/(1−p) = 2.009`, so `(1+a) ≤ 3.009` while cycle
   grows without bound. Under regime A the band is unreachable **by construction at any γ**,
   not merely at the four tested.

### THE γ=3 CONTROL BOOT IS THE REGIME DISCRIMINATOR
Boot 1 measures true acceptance at 89K with the counters for the first time ever.
- `mean_accept ≥ 2.0/3` → regime B → γ=5 is a live candidate, run the full sweep.
- `mean_accept < 2.0/3` → regime A → band unreachable by construction; the sweep's job
  becomes *measuring the cliff* (the I8 per-draft-cost question), not finding a winner.

---

## SWEEP PROTOCOL (fixed)

- **Arms and order: 3, 4, 2, 5, 3.** γ=3 twice brackets the sweep and yields a same-config
  boot-variance reading for free. Every arm is a full relaunch (`EXO_SPECULATIVE_GAMMA` is
  read at start).
- **Every boot carries `EXO_DSV4_MTP_LOG_INTERVAL=50`** (rides the relaunch per P0-1;
  it is a log line every 50 cycles — negligible cost).
- **`EXO_DSV4_MTP_PROFILE` is OFF for all timed arms.** Its value is a dump *cadence*, but
  the bracketing `mx.eval` runs **every** cycle ("serialises pipelining — measurements are
  upper bounds on real production walls", `dsv4_mtp.py:240-243`). Turning it on would
  pollute the very t/s the bands are applied to. Consequence, stated plainly: **acceptance
  is the independently-measured quantity; cycle time is derived as `(1+a)/tps`.** Derived
  and measured t/s are therefore algebraically linked, NOT independent — I will report it
  that way rather than dress it up as a cross-check it is not.
- **Phase ratios (the I8 draft-vs-verify growth question)** require the profiler and are
  therefore deferred to at most one extra profiled boot pair (γ=3 and γ=5) run only if the
  main sweep finishes inside the time box. Ratios remain valid under uniform bracketing
  inflation even though absolute walls do not. If unrun, I8 is reported UNMEASURED with
  this reason — not guessed.
- **Depth 89K, n≥3 measured reps per arm, rep 0 discarded as warmup**, `trustworthy` required.
- Per-arm reporting: `mean_accept`, full `hist`, derived cycle ms, derived t/s, measured
  t/s **range** (min–max, never a single number), and both γ=3 boots for boot variance.

## BANDS (copied verbatim from the task; applied without modification)
- derived `(1+a)/cycle` **≥ 1.10x** the γ=3 control (**both** γ=3 boots) **AND** measured
  t/s range **entirely above** γ=3's range → **SHIP CANDIDATE** → quality gate.
- **1.03–1.10x** → "inside boot variance, not shippable on this evidence."
- **< 1.03x** → **closed.**
- Bit-equivalence is a **hard gate**: speculation is lossless, so temp=0 byte-identity vs
  γ=3 must hold **exactly**. Any divergence = a verify bug at that γ → **arm disqualified
  regardless of speed.**

## PRE-REGISTERED DEGRADE PATH (if the time box binds)
Drop **γ=2 (arm 3)** first, giving 3, 4, 5, 3. This preserves both the closing γ=3
bracket (the boot-variance control, the most valuable single arm) and both extension arms.
γ=2 is the lowest-information arm because every model cell agrees it loses. **Never** drop
the closing γ=3 bracket. A partial sweep with honest ranges is a valid round.

## WHAT WOULD FALSIFY ME
- γ=2 winning → the May regime never ended; batched verify did not remove per-row cost.
- γ=5 ≥ 1.10x → regime B is real at 89K **and** the width-3 head extrapolates cleanly;
  the "trained for width-3" comment would then be a poor predictor of acceptance.
- Any arm failing byte-identity → a verify bug at that width, which outranks every
  throughput number in this document.
