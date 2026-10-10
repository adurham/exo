# Q2-FU1 RUNBOOK — stage, eval-boot, capture, restore (exact commands)

All controller-side paths are on the Mac (this repo worktree).  `studio1`
(rank 1) and `studio2` (rank 0).  **Nothing in PART A–B2 was executed while
building this branch** — every command below is for the PM.

Budget: **≤ 2 boots** (eval + restore); process relaunches happen only via the
driver's arm switches (sed `~/relaunch_exo.sh` + graceful relaunch; NOT
reboots).  Declared caps: eval boot ~10 min, arm switches ~5 min, capture
requests (one sweep request + 3 stream requests per arm, agentic prompt)
≈ 6–8 min/arm + pull, ≤ 45 min total for g3/g4.

---

## PART 0 — preconditions (read-only)

```bash
ssh studio1 'grep -c DSV41_SPEC_GAMMA= ~/relaunch_exo.sh; ls /tmp/q2fu1'   # prod today: 0 tokens, no dir
ssh studio2 'grep -c DSV41_SPEC_GAMMA= ~/relaunch_exo.sh; ls /tmp/q2fu1'
```

Production today = `99e2966ee` detached, no `DSV41_SPEC_GAMMA` token.  The
eval boot MUST export the token (below) or the driver's `sed` precondition
fails loudly (and gamma never flips).

---

## PART A — stage the hook on BOTH nodes (no boot)

```bash
scp /private/tmp/q2fu1-instr/bench/phase20_q2_fu1/q2fu1_capture.py studio1:/tmp/q2fu1/q2fu1_capture.py
scp /private/tmp/q2fu1-instr/bench/phase20_q2_fu1/q2fu1_capture.py studio2:/tmp/q2fu1/q2fu1_capture.py
ssh studio1 'mkdir -p /tmp/q2fu1/out && ls -la /tmp/q2fu1'
ssh studio2 'mkdir -p /tmp/q2fu1/out && ls -la /tmp/q2fu1'
```

The hook dir `/tmp/q2fu1` is what the launcher derives into `PYTHONPATH`
(parent of `/tmp/q2fu1/out`).  Do NOT edit it in place on the nodes after this
(no live-editing rule).

## PART B — eval boot (boot #1)

On the CONTROLLER (the checkout whose tree is rsynced to the nodes):

```bash
cd /private/tmp/q2fu1-instr
git fetch origin && git checkout --detach origin/deploy/q2-fu1-capture
export DSV41_SPEC_GAMMA=3                      # so ~/relaunch_exo.sh carries the arm token
export DSV41_Q2FU1_CAPTURE=/tmp/q2fu1/out      # capture dir; PYTHONPATH derives /tmp/q2fu1
EXO_TARGET_BRANCH=deploy/q2-fu1-capture ./start_cluster.sh
```

(The pre-deploy check uses `merge-base --is-ancestor`; a detached checkout at
the branch tip passes.  `start_cluster.sh` regenerates `~/relaunch_exo.sh`
from `EXO_ENV`, so BOTH the gamma token and the capture vars ride node-side
restarts.)

### PART B2 — verify + JIT-warmup (the relaunch spawns NO runner)

```bash
ssh studio1 'grep -a "q2fu1 capture hook ACTIVE" ~/.exo/exo_log/exo.log | tail -1'
ssh studio2 'grep -a "q2fu1 capture hook ACTIVE" ~/.exo/exo_log/exo.log | tail -1'
ssh studio1 'git -C ~/repos/exo rev-parse HEAD'   # == branch tip
ssh studio2 'git -C ~/repos/exo rev-parse HEAD'
curl -s http://192.168.86.48:52415/state | head -c 400   # expect a runner; else warmup POST
ssh studio1 'grep -a "spec gamma override" ~/.exo/exo_log/exo.log | tail -1'   # gamma=3 rank 1
ssh studio2 'grep -a "spec gamma override" ~/.exo/exo_log/exo.log | tail -1'   # gamma=3 rank 0
```

If `/state` shows no runner: a tiny generation POST triggers the JIT model
load (`503 -> 200`, ~120 s) — the driver does this automatically
(`ensure_loaded`), and the 300 s JIT idle reaper means ANY idle-gap wait must
be followed by a re-trigger (the driver registers every POST as own traffic
and re-checks before each arm).  Never gate readiness on `/state` alone.

## PART C — run the capture driver

```bash
cd /private/tmp/q2fu1-instr
PYTHONPATH=bench/phase20_q2_fu1 \
  /Users/adam.durham/repos/exo/.venv/bin/python bench/phase20_q2_fu1/q2fu1_driver.py \
  --outdir /Users/adam.durham/.hermes/cache/scratch/phase20/q2fu1 \
  --arms g3,g4 --ms 24 --rounds 8 --rollback 1
```

Dry-run first if desired: `... q2fu1_driver.py --dry-run` (prints the exact
command plan; touches nothing).

The driver per arm: switch (REUSE if the nodes already log the target gamma —
the warmup POST still runs), READY, rank-consistency assert, sweep request
(`q2fu1;arm=g3;sweep=1`), multi-step request
(`q2fu1;arm=g3;ms=24;rollback=1;rounds=8`), nospec reference request, then
scp-pulls both nodes' `/tmp/q2fu1/out/*.npz` bit-exact, merges rank pairs
(bitwise-equal non-logit keys), runs the analysis + faithfulness gate and
writes `<outdir>/q2fu1_report.json`.

**Read the faithfulness verdict FIRST** (`=== FAITHFULNESS ===`): EXACT ⇒ the
dumps reproduce the recorded round (gamma3 == `arm_gamma3a.json`,
gamma4 == `arm_gamma4.json` div idx 14/12) and are trustworthy; MISMATCH ⇒
stop and report — do not consume the dumps.

If gamma2/gamma5 are wanted (optional): `--arms g2,g3,g4,g5` (adds one arm
switch + 3 requests per extra arm).

## PART D — production restore (boot #2, the FINAL live state)

```bash
cd /Users/adam.durham/repos/exo          # controller checkout = the live repo copy
git fetch origin && git checkout --detach 99e2966eec26891b59e1af5467d8f0ddaad38577
unset DSV41_SPEC_GAMMA DSV41_Q2FU1_CAPTURE DSV41_Q2FU1_MS DSV41_Q2FU1_ROUNDS DSV41_Q2FU1_DMAX DSV41_Q2FU1_KSWEEP
EXO_TARGET_BRANCH=deploy/next19-dense ./start_cluster.sh
```

Verify: `RESTORED 99e2966ee READY 2/2 canary parity`; the new boot's log has NO
`q2fu1 capture hook ACTIVE` line; `grep -c DSV41 ~/relaunch_exo.sh` on both
nodes shows the eval vars gone.

## PART E — cleanup (optional)

```bash
ssh studio1 'rm -rf /tmp/q2fu1'   # after the dumps are pulled + archived
ssh studio2 'rm -rf /tmp/q2fu1'
```

## Failure modes (pre-empted in the design)

* **sed no-op / no token**: driver aborts at the precondition (loud).
* **JIT idle reaper (300 s) unloads mid-run**: every driver POST registers an
  own-request epoch; `ensure_loaded` runs before every arm and repairs an
  unloaded instance with a warmup POST.
* **Watchdog/rank-skew on first call per shape**: the sweep runs at most ~30
  extra forwards per request; warm_guard host-syncs the first 2 calls per
  shape (unchanged semantics, same result values).
* **Sweep failure** (any exception): logged, the request continues WITHOUT
  capture (the wrapper is exception-guarded end to end) — a missing dump is
  visible, a broken serve is not possible.
* **Rank divergence of the diagnostic program**: impossible by construction —
  the extra forwards are driven by the replicated `correlation_id`
  (present in `task_params` on both ranks) at identical cache offsets; the
  collector asserts bitwise equality of every non-logit key across ranks.
