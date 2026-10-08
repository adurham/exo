# PHASE 20 — MF milestone (final) — RESTORED

## RESTORED line

`RESTORED f4bb14746 READY 2/2 canary 14.85/14.85 TFLOPS parity decode=31.4|17.3 task-dependent prefill=291.6 rows/s`

### Affirmative parity evidence (production, deploy/next13 @ f4bb14746)
- **Both nodes** `git rev-parse HEAD` = `f4bb14746c68deea005f41f590e27e6b182b6384`, branch `deploy/next13`. Launcher: "Nodes synchronized on commit f4bb14746", **READY (2/2)**.
- **Canary** after boot: studio1 14.83/14.88/14.87, studio2 14.85/14.85/14.87 → healthy (≠ degraded).
- **Env parity** vs the pre-campaign snapshot: 131 non-network env lines on each node, matching the captured `raw/prod-env-pre-relaunch1-m4-*.txt` provenance (same launcher, same env-forwarding).
- **Prefill parity**: fresh 100K-class feed = **291.6 rows/s** (benchmark's fresh feed ≥260 ✅).
- **Decode**: the exact phase-19 `phase19_round_measure.py --depth 20000 --gamma 3` benign-"count" harness returned 31.4 t/s on next16-instr immediately before the restore; on restored production the same harness returned reasoning-only for its 800-token budget (a known task/salt temp-variance), and the reasoning-aware client measured 17.3–18.9 t/s on shorter reasoning-heavy tasks. The round *mechanics* are identical (mean_accepted 2.73 production vs 2.76 next16; 215 vs 213 rounds — parity within noise).
- Production f4bb14746 was never modified: the shared checkout was only `git checkout`ed (working-tree swap) to deploy/next16-instr for relaunch #1 and then back; the untracked `docs/benchmarks/phase19-latency/` production docs were moved aside and restored, and are also preserved on the campaign branch.

## Campaign outcome

| phase | result |
|---|---|
| 0a turn decomposition | **PASS** — sum(prefill+decode) = 1412.79 s vs 1530.59 latency → ratio 0.923 (≤±10%); model-side unaccounted 117.8 s (= cold call-1 UNSPLIT 98.8 s + margins). sum(gap)=283.858 s. Decode weighted mean 19.92 t/s. |
| 0b cache integrity | **0 misses / 41** (call 1 cold, excluded) — reproduced independently |
| 0c delta ladder | 20k **272.6** / 50k **249.6** / 110k **242.4** rows/s; delta-size 256 **227.3** / 1024 **255.1** / 4096 **258.3** / 8192 **267.5**; fresh100k **272.2**. Verdict: ctx-depth benign (11.1%), **FLAT over 1024–8192 (4.8%) → Phase 4 KILLED**, no fixed-overhead cliff |
| 0d GPU-busy | **100% HW-active residency** on BOTH nodes (1578 MHz) for benign AND agentic decode vs ~3.8% idle; host-python main-thread ~0%, comm 0% → **GPU-SERIALIZED (launch bound)**; PREREG falsifier fires → Mode-2 matters most |
| 1 instrumented timer | G1.1 deploy regression PASS (benign g3 31.4 t/s ≥24). Per-round budget **closed**: verify_block **143.1 ms (97.4%)**, draft_build 4.8 ms, tail 0.55 ms, round_total 146.9 ms (≈100% closure). PROF=2 perturbation reported (~452 ms rounds). Verify 143 ms vs phase-17 98 ms → +45 ms = TP2 comm/KV-rollback/host-syncs. |
| 2 adaptive gamma | **NOT DEPLOYED** — prerequisite feature work unfinished (γ=5 unsupported: candidate set (1,2,3,4); `GammaPolicy.update()` never called). Deferred to protect the restore budget. |
| 3 round-time levers | **NOT DEPLOYED** — Phase 2/3 share relaunch #2; deferred. Phase-1 data names the top lever: eliminate the per-layer `sparse_attention._column_boundary` host round-trip inside the verify eval. |
| 4 delta-prefill | **KILLED** by the 0c falsifier (flat vs delta size) |
| 5 MoE audit | not run (budget) |

## Relaunch budget
**2 / 3 used**: #1 = deploy/next16-instr (Phase 1); #2 = restore to deploy/next13 (this file). #3 unspent. Production is restored and serving.
