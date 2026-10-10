# phase20_q2_gamma — Round Q2-Gamma arm-matrix driver + gate evaluators

**Status: build-only artifact. NOT RUN.** No live arm switch has been executed;
the cluster was touched only by read-only probes while building this. Run it only
after the eval boot and a GO (see *Preconditions*).

Round Q2-Gamma re-prices the dsv41 speculative draft depth `gamma` on the
post-dense production build. It runs a **bracketed arm matrix in ONE boot**:

```
gamma3a -> gamma2 -> gamma4 -> gamma5 -> gamma3b
```

`gamma3a` / `gamma3b` are the same-config determinism control bracketing the two
lever arms; `gamma5` is in the supported set `(2, 3, 4, 5)` so it runs by default
(drop it with `--include-gamma5 no`).

| file | role |
|---|---|
| `q2_gamma_driver.py` | the arm-matrix driver: `switch_arm`, `wait_ready`, rank-consistency assert, `replay_once` (top-2 logprobs), `run_arm`, memory reads, `main` |
| `q2_gamma_eval.py` | the pre-registered gates: `identity_gate`, `drift_control`, `bars`, `memory_gate` (pure stdlib, no cluster) |
| `selftest.py` | offline self-test: reproduces the r1kit frozen anchors + unit-tests every evaluator |
| `../phase20_tests/test_phase20_q2_gamma.py` | pytest wrapper (run with `--noconftest`) |

## DECLARED BUDGET (fixed BEFORE spending; printed in the driver header)

Anchors are the **frozen r1kit control** wall-clocks, measured from
`control_{benign,agentic}.json` (every rep is a COLD prefill because the per-rep
salt differs):

* benign 20K  ≈ **90.5 s/rep** (cold prefill ≈70 s + decode ≈21 s)
* agentic 91K ≈ **385 s/rep** (cold prefill ≈357 s + decode ≈26 s)

Arithmetic for the default 5-arm matrix, benign ×3 + agentic ×2, cap 135 min:

```
agentic:  385 s/rep x 2 reps x 5 arms = 3850 s = 64.2 min
benign:   90.5 s/rep x 3 reps x 5 arms = 1358 s = 22.6 min
overhead: (relaunch 300s + idle 90s) x 5 = 1950 s = 32.5 min
TOTAL:    7158 s = 119.3 min  vs cap 135 min => FITS
DECISION: agentic_reps=2, iqr_enabled=False; DROP disjoint-IQR criterion (agentic reps < 3)
```

**3 agentic reps do NOT fit** (agentic alone = 385×3×5 = 5775 s = 96.3 min;
total 151.4 min > 135 min cap), so the driver **pre-declares 2 agentic reps and
DROPS the disjoint-IQR criterion**. Raise `--budget-wall-min` (e.g. 200) and the
driver re-selects 3 reps + IQR automatically — see the selftest budget cases.
The selected decision is written to `<outdir>/round_budget.json`.

## Preconditions (arm-switch recipe; verified by the PM)

The recipe flips `DSV41_SPEC_GAMMA` by editing `~/relaunch_exo.sh` on BOTH nodes
and gracefully relaunching the node process (a **process** relaunch, NOT a
cluster boot/reboot, so it does not consume an R5 deploy relaunch):

1. **The eval deploy must have been launched WITH `export DSV41_SPEC_GAMMA=3`**
   so `~/relaunch_exo.sh` carries exactly one `DSV41_SPEC_GAMMA=3` token. Without
   the token the `sed` is a silent no-op and **no arm ever flips** — the driver
   asserts this precondition and aborts loudly.
   *(Production `~/relaunch_exo.sh` does NOT carry the token — verified read-only
   2026-10-10 via `grep`: rc=1 on both nodes. This is expected pre-eval-boot.)*
2. `~/relaunch_exo.sh` is on each node; the API is up at `http://192.168.86.48:52415`.
3. `iogpu.wired_limit_mb = 120000` on both nodes (W = 125.829 GB).

The driver, per arm, runs on each node in order:

```bash
ssh studioN "sed -i '' -E 's/DSV41_SPEC_GAMMA=[0-9]+/DSV41_SPEC_GAMMA=N/' ~/relaunch_exo.sh"
ssh studioN "grep -o 'DSV41_SPEC_GAMMA=[0-9]*' ~/relaunch_exo.sh"   # confirm exactly one token == N
ssh studioN '~/relaunch_exo.sh'                                      # graceful relaunch, ~2-3 min
```

then polls `GET /state` until **READY 2/2** (`count_ready_runners` counts
`RunnerReady`), asserts **RANK CONSISTENCY** by grepping both nodes' current-boot
`~/.exo/exo_log/exo.log` for
`[DSV41] spec gamma override: DSV41_SPEC_GAMMA=N -> effective gamma=N ..., rank R`
and aborting if the two ranks' effective gamma differ (or the ranks aren't {0,1}),
idle-guards, then runs the fixed replays.

## Replays (FIXED content, byte-identical across arms)

Salt base **`q2gamma`**, per rep `q2gamma-<rep>` (rep index only — identical
across arms). Requests to `/v1/chat/completions` with `temperature=0`,
`stream=true`, `logprobs=true`, `top_logprobs=2`, `max_tokens=800`; the first
~256 emitted-token `(token, top1_logp, top2_logp)` tuples are captured for the
identity signal. Chunks are ≤15 min (`ChunkGuard(max_wall_s=900)`) with an idle
guard before each. Per rep the driver records `completion_tokens`, `decode_s`,
`ttft_s`, decode tok/s, the r1kit `derive()` metrics, and the generation-stats
deltas → accepted-length histogram + mean accepted/verify step. Per arm it reads
the **corrected peak allocator** (VM `exo_peak_memory_bytes` ÷ **1.073741824**)
and the **peak resident** (`footprint -f bytes -p <runner_pid>`).

## Gates (`q2_gamma_eval`)

* **identity_gate** — vs `gamma3a`, per rep index: first-divergence index + the
  top-1/top-2 logit margin there. PASS if identical OR every divergence is at
  margin < ε (ε = 0.05) AND the position-level divergence rate ≤ ~2× the
  gamma3a-vs-gamma3b rate; **ABORT on any confident-position (margin ≥ ε)
  divergence** (also ABORT if the control itself diverges confidently). If the
  endpoint cannot return logprobs the driver falls back to an
  **acceptance-histogram** identity signal and **says so** (`signal=histogram_fallback`).
  *The rate rule collapses to 0 when the control is byte-identical; a
  pre-registered absolute allowance floor `RATE_FLOOR = 0.05` keeps an isolated
  near-tie flip (the D4-expected behaviour) from aborting the round.*
* **drift_control** — gamma3a vs gamma3b decode tok/s differ > 1.5% ⇒
  `RE-RUN (drift-contaminated)`.
* **bars** — an agentic winner must beat the **median of gamma3a+gamma3b** by
  ≥ 3% OR have disjoint IQRs; benign must not be worse than 1.5% on median and
  its IQR must not fall wholly below gamma3's. A **split** (agentic win + benign
  loss beyond noise) ⇒ verdict `gamma3 confirmed; per-workload gamma is a future
  item` (**do NOT ship**); no winner ⇒ `gamma3 confirmed on the post-dense
  build`. IQR criterion auto-disabled when agentic reps < 3.
* **memory_gate** — any arm whose corrected peak is within **0.5 GB of W** is
  ineligible to win even if faster (worst of the corrected allocator and the
  resident reading over both nodes).

## Commands

```bash
PY=/Users/adam.durham/repos/exo/.venv/bin/python
W=/private/tmp/q2-gamma-driver           # this worktree
KIT=/private/tmp/q1b-driver/bench/phase20_r1kit   # frozen r1kit (or set Q2_GAMMA_KIT)

# offline self-test (no cluster): anchor reproduction + evaluator unit tests
cd $W && PYTHONPATH=bench/phase20_q2_gamma Q2_GAMMA_KIT=$KIT $PY bench/phase20_q2_gamma/selftest.py

# pytest form (bench tests need --noconftest; the repo conftest has the mlx-lm landmine guard)
cd $W && PYTHONPATH=bench Q2_GAMMA_KIT=$KIT $PY -m pytest --noconftest \
    bench/phase20_tests/test_phase20_q2_gamma.py -q -p no:cacheprovider

# dry run: header + budget + exact command plan, touches NOTHING
cd $W && PYTHONPATH=bench/phase20_q2_gamma $PY bench/phase20_q2_gamma/q2_gamma_driver.py \
    --dry-run --outdir /Users/adam.durham/.hermes/cache/scratch/phase20/q2gamma

# LIVE (only after the eval boot + GO)
cd $W && PYTHONPATH=bench/phase20_q2_gamma Q2_GAMMA_KIT=$KIT $PY bench/phase20_q2_gamma/q2_gamma_driver.py \
    --outdir /Users/adam.durham/.hermes/cache/scratch/phase20/q2gamma
```

Outputs (under `--outdir`): `round_budget.json`, `arm_<arm>.json` (records),
`own_requests_<arm>.jsonl` + `guard/` (idle-guard evidence), and `round_eval.json`
(all gate verdicts).

## NOT verified (build-time, no live run)

* No arm switch / relaunch was executed; the recipe is implemented from the PM's
  verified description and exercised only via the `--dry-run` plan + `sed`
  simulation.
* `wait_ready` (`RunnerReady` 2/2) and the effective-gamma log-line regex are
  modelled on read-only `/state` and source probes, not a live eval boot.
* `footprint -f bytes` first-line parsing is defensive (first numeric line, else
  max integer) — not confirmed against a live runner.
* The VM `exo_peak_memory_bytes` query and the ÷1.073741824 correction are per
  the ROUND-Q1E finding, not re-measured here.
* `logprobs`/`top_logprobs` response shape is taken from the API adapter source;
  the live endpoint's honouring of `top_logprobs=2` is unverified.
