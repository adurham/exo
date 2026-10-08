# PHASE 20 — PRE-REGISTRATION (frozen before any Phase-0 live measurement)

Written 2026-10-07 ~21:30 CDT by the campaign PM. Binding source: `/Users/adam.durham/.hermes/cache/scratch/PHASE20-PLAN.md`
(Fable 5.1 plan + orchestrator appendix). Scope (user, binding): **prefill tok/s and decode tok/s only**.
Gates below were fixed BEFORE the runs they judge; any change after a run is recorded as an amendment with a reason,
never silently applied.

## 0. Deviations from the plan, each resolved against the live system

| id | plan text | live finding | adopted rule |
|---|---|---|---|
| D1 | R1(b) `count(*) from api_calls where ended_at is null` == 0 | `api_calls.ended_at` is `NOT NULL`; rows are inserted at call COMPLETION, so an in-flight user call is invisible. The check is vacuous. R1(a) (POST age >10 min) is also blind to a long in-flight call (a 100K cold prefill runs 6+ min, deep ones 40+ min). | Idle = ALL of: (i) no active TextGeneration task on the cluster (`/state` tasks and/or exo.log start/finish pairing); (ii) no non-own `POST /v1/chat/completions` in exo.log on either node in the last 10 min (own requests are subtracted via a registry); (iii) state.db: no `provider='custom'` row completed in the last 10 min and no exo-model session activity in the last 10 min. Implemented in `bench/phase20_guard.py` (contract: `GUARD-CONTRACT.md`). |
| D2 | R6: nodes `git fetch && git reset --hard f4bb14746` | `start_cluster.sh` **rsyncs the laptop shared checkout `$HOME/repos/exo` (incl. `.git`) to both nodes**; nodes do not pull. A /private/tmp worktree cannot be the deploy source. | Deploy = temporarily `git checkout <deploy branch>` in the shared checkout (untracked files that collide are moved aside and restored), `EXO_TARGET_BRANCH=<branch> ./start_cluster.sh`, verify node `git rev-parse HEAD` == branch tip on BOTH nodes. Restore = checkout `deploy/next13` (f4bb14746) in the shared checkout + `EXO_TARGET_BRANCH=deploy/next13 ./start_cluster.sh` (as `phase19-latency/restore-snapshot-live.md`). |
| D3 | `EXO_DSV41_ROUND_PROF` env, modes unset/1/2 measured "in one campaign" | Env is fixed at boot; 3 relaunches total cannot buy one boot per mode. | Mode is a **per-request API field** `round_prof` in {0,1,2} (pattern = the shipped `spec_gamma` passthrough), env var is only the default. Same for `spec_gamma:"adaptive"` in Phase 2. Task params are identical on both TP ranks, so modes cannot diverge across ranks (a mismatch would deadlock the collectives). |
| D4 | R8: every change byte-identical to f4bb14746 baseline | The committed g3-vs-g5 battery (same build) is **0/25 byte-identical** (prose 0/20; needles 6/6 and tools 10/10 PASS both). Verify chunk shape (M=gamma+1 rows) changes accumulation order -> near-tie argmax flips (documented in `rounds.py` docstring). | **R8a** (fixed-gamma, numerically-neutral changes: PROF timer, host-sync restructure, lazy rollback bookkeeping): battery outputs byte-identical to a baseline captured on the CURRENT f4bb14746 cluster at gamma 3, *conditional on* a same-build determinism replicate (baseline run twice, A1 vs A2) being byte-identical; if the replicate is not byte-identical, R8a degrades to "identical on the deterministic subset (needles/tools raw) + detectors clean on prose". Any R8a diff on a numerically-neutral change = investigate first-divergence margin before revert (moving an `mx.eval` can legitimately reorder fusion). **R8b** (gamma-changing levers = adaptive): cannot be byte-identical by construction; gate = `compare.py` verdict PASS (needles/tools no regression, 0 DIRTY detectors) + unit identity tests (policy never switches == fixed g3; always switches == fixed g5) + adaptive run twice is byte-identical to itself (policy must be a pure function of accepted-history; nothing timing-derived) + per-round KV/ctx-length consistency assertion. REASONING_ONLY rate is a tripwire, not a gate (n=20 has no power). |
| D5 | Phase-1 gate "PROF=1 bracket sum closes >=95% of round time" | Contiguous host brackets sum to the in-function round total by construction (trivially true). | Report it (C1) but the gate is **C2: closure against the client**: sum over rounds of (`round_total_ms` + inter-round `emit_ms`) vs client-observed decode wall (first token -> last token), ratio in [0.95, 1.05], on benign AND agentic. |
| D6 | Phase-2 policy keys on EMA(p4) | At gamma=3 positions 4/5 are never drafted -> p4 is unobservable while running at 3. | Up-switch signal calibrated offline from a gamma=5 PROF=1 per-round pass (conditional hazards P(acc>=4 \| acc>=3), P(acc=5 \| acc>=4)); decision rule is tokens/ms (maximize (1+E[acc_g])/T_g with measured T_3, T_5), probing a gamma=5 round every N rounds if the proxy is uninformative. Thresholds are FROZEN in a commit before the validation A/B. |
| D7 | Rank naming | `/state`: **rank 0 = studio2 (m4-2, 192.168.86.47, jaccl coordinator)**, rank 1 = studio1 (m4-1, 192.168.86.48, API/master). | All per-rank outputs are labelled with both rank and node. |

## 1. Global rules as adopted
R1 idle guard (D1) before every chunk and relaunch; R2 abort-on-arrival watcher on every chunk (tokens `ABORTED_USER_ARRIVED`, `ABORTED_WALL_CAP`);
R3 chunks <=15 min; R4 raw-GPU canary after every boot and before every measurement run (fp16 4096^3 matmul, healthy 14-15 TFLOPS, <5 degraded;
reboot only via `./reboot-node.sh studio1 studio2`, max 2 cycles, never raw shutdown); R5 relaunch budget 3 (#1 next16-instr, #2 next17-levers, #3 restore;
a failed experimental deploy consumes #3 and ends the campaign); R7 graveyard honoured; FORBIDDEN always: `EXO_KV_CACHE_BITS!=0`, `EXO_DSV4_INDEX_TOPK<512`, `repetition_penalty!=1.0`.
No benchmark rep is accepted without the exo.log `turn reuse:` / `prefill controls` evidence of its prefill shape and a unique per-feed salt.
Noise handling: >=5 reps for any A/B claim, medians + IQR, arms interleaved, a g3 control arm re-run at the end of each chunk; control drift >3% marks the chunk "host/thermal drift" and it is reported but not used for gating.
Cold-champion sanity (prior campaign): benign 100K g3 round wall <=145 ms; >=150 ms means investigate before any A/B.

## 2. Phase gates (pre-registered)

**Phase 0**
- 0a: `sum(prefill_s + decode_s)` within +-10% of `sum(latency)=1530.59 s`; remainder = "model-side unaccounted". Components come ONLY from log markers; no estimating (unsplit calls are reported as unsplit). Cache-hit rule: `delta_rows <= 1.2 x (ctx_at_call - ctx_at_prev_call)`; call 1 is the cold start and is excluded from the miss count. Prior audit (`phase19-latency/raw/phase2-delta-audit.md`) found 0/41 misses; 0a must reproduce independently.
- 0b: >=1 miss => top priority (repro with phase19_agentic_measure + session.py cache lookup, 60 min, client-side => report+stop, engine-side => relaunch-#2 candidate). 0 misses => closed with one line.
- 0c: delta rows/s degrading >15% from 20K->110K at 2048-row delta => ctx-depth cost; 256-row rows/s <50% of 4096-row at same ctx => fixed per-call overhead => Phase 4 lever; delta rows/s flat (<10% spread over 1024-8192 rows) => Phase 4 skipped, slope recorded. Fresh-feed 100K reference <255 rows/s => degraded cluster: re-canary, do not record the chunk.
- 0d: powermetrics `GPU HW active residency` minus idle baseline = GPU-busy; decode window starts only after the first SSE token of a decode-only (prefix-cached) request. GPU-busy >=90% both nodes AND host-Python <5% => GPU-serialized (Mode-2 matters most); GPU-busy <85% either => host/comm bound (timer mandatory); nodes differ >10 points => one rank waiting on the other. Residency within 5 points of idle => wrong process, stop, Fallback B.

**Phase 1** (relaunch #1, `deploy/next16-instr`)
- G1.1 deploy regression (round_prof unset): benign g3 >=24.0 t/s, agentic >=20.0 t/s, fresh prefill >=260 rows/s, R8a battery per D4. Any failure => R6 restore, campaign ends.
- G1.2 closure C2 in [0.95,1.05] (D5) on benign AND agentic, top-2 brackets named with ms and share of the round.
- G1.3 PROF=2 `verify` reconciles with phase-17 (~98 ms) within +-20%; if much greater the excess is reported as TP comm.
- G1.4 perturbation (round time PROF=2 vs PROF=1 vs unset) reported, never hidden.
- Falsifier: 0d GPU-busy >=95% both nodes and host-busy <3% => skip Mode-1 analysis, go to a Mode-2 kernel-count audit.

**Phase 2** (relaunch #2, `deploy/next17-levers`): plan gate verbatim: agentic adaptive >=0.99 x agentic g3 (>=20.4 given 20.61) AND benign adaptive >= g3 + 6% (>=26.2), disjoint IQRs, >=5 interleaved reps each of adaptive / fixed g3 / fixed g5, benign and agentic. Falsifier (before coding thresholds): per-round data from the Phase-1 gamma=5 pass shows no stretch of >=20 consecutive rounds with accepted>=4 on agentic content => kill adaptive. R8b per D4.

**Phase 3** (relaunch #2): lever ships only if (microbench) >=3 ms/round measured saving on agentic replay AND R8a clean AND no benign regression. Phase gate: agentic >=22.5 t/s = win; >=30 t/s not expected; remaining ms-gap to 30 t/s reported honestly. Arithmetic fact stated up front: 30 t/s needs BOTH adaptive gamma and >=15 ms/round of Phase-3 savings; neither alone gets there.

**Phase 4** (only if budget allows; rides relaunch #2 if ready, else deferred and reported): 8192-row delta at 50K >=+10% rows/s vs off; no regression at 256/1024; fresh feed unchanged +-2%; battery clean (R8a). Target: 2048-row delta at 50K >=270 rows/s. Skipped if 0c shows rows/s flat vs delta size.

**Phase 5** (no deploy; only if everything else is done and the user has not returned): resolve "40-50% MoE headroom" with one number — measured TFLOPS vs predicted roofline at M=16 within +-10% => headroom claim dead. 1.5 h hard stop.

## 3. Restore ("RESTORED") definition
`RESTORED f4bb14746 READY 2/2 canary X/Y TFLOPS parity decode=<benign g3 t/s> prefill=<fresh rows/s>` requires: shared checkout back on `deploy/next13` @ f4bb14746; both nodes `git rev-parse HEAD`==f4bb14746; READY 2/2; R4 canary both nodes; ps-eww env parity vs the pre-campaign snapshot (`raw/prod-env-pre-relaunch1-m4-*-ps-eww.txt`); parity smoke benign g3 >=24.0 t/s and fresh prefill >=260 rows/s. Until that line exists production is not restored.
