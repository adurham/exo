# P4v2 M1 shadow-gate results, incident recovery, and HOLD verdict — 2026-08-24

**Status: CLOSED — shadow measurement captured, gate verdict HOLD, cluster
reverted to production defaults (`EXO_SPECULATIVE=0`).**

## 1. Incident context

The 2026-08-23 P4v2 M0+M1 session (commit `34478792b`, pushed) was cut off
mid-measurement by repeated `HTTP 200: Overloaded` API errors — both the
implementing worker and the orchestrator died non-gracefully. The cluster was
left serving on an **experimental** env (`EXO_SPECULATIVE=1 EXO_DSV4_MTP=1
EXO_DSV4_DSPARK=1 EXO_DSV4_SPEC_SHADOW=1 EXO_DSV4_DSPARK_CONF_TAU=0.9`) with
an apparent 3.4x decode regression (8.51 tok/s @100K vs ~29 known-good) and an
apparently-empty diagnostic log. This doc is the recovery session's
diagnosis + resolution.

Timeline reconstructed from file mtimes and `ps -o lstart`:

| When (Aug 23) | Event |
|---|---|
| ~23:0x | Relaunch #1: shadow env with `EXO_DSV4_SPEC_SHADOW_LOG=/tmp/dspark_shadow.jsonl`, τ at code default 0.5 |
| 23:08–23:09 | `prod_short` / `prod_10k` probes (production build for identity baseline) — *actually run pre-relaunch on the prod build* |
| 23:33–23:41 | `shadow_short`, `shadow_short_rerun`, `shadow_100k` probes against relaunch #1; jsonl grows to 782 cycles |
| 23:50 | Relaunch #2: τ=0.9 added, **`EXO_DSV4_SPEC_SHADOW_LOG` dropped from the env** |
| ~23:5x | API overload kills both agents. No probe ever ran against relaunch #2 |

## 2. The two "mysteries" — both resolved, neither is a bug

**Empty diagnostic log.** The live (relaunch #2) process env on both nodes,
read via `ps eww`, confirms `EXO_DSV4_SPEC_SHADOW_LOG` is absent — the
truncated worker dropped it when it added `CONF_TAU=0.9`. With
`_SPEC_SHADOW_LOG=""` the jsonl append at `dsv4_mtp.py:4168-4170` is skipped
entirely, and since zero probes ran after 23:50 there was nothing to log
anyway. **The real data from relaunch #1 was never lost**: 782 cycles sitting
in `/tmp/dspark_shadow.jsonl` on **m4-1** (173 KB, last write 23:41). The
"empty file" observation came from checking the wrong host (the file never
existed on the MacBook or m4-2 — rank0/API-node placement puts it on m4-1).
Recovered to `bench_data/shadow_gate_20260823/dspark_shadow_relaunch1.jsonl`
along with all probe json/txt files.

**8.51 tok/s @100K.** Not a regression, not a malfunction: it is shadow mode
working exactly as designed and priced. Shadow mode runs the full DSpark
draft (~11.3 ms) + the full γ-row batched verify (~99 ms mean at 100K) every
cycle, then **forces `n_accepted=0`** so exactly one sequential-path token is
committed per cycle. Per-token wall = full speculative cycle cost + shadow
bookkeeping ÷ 1 token. Measured: 117.5 ms/cycle wall (45.12 s window / 384
cycles) vs 110.3 ms mean accounted draft+verify — an ~8 ms/cycle residual
(rollback, fence drain/re-arm, cross-rank broadcast, ctx append), bounded
above by wall-clock triangulation. 1000/117.5 = 8.5 tok/s. The prior
session's alarming "3.4x slower" is the *expected* price of measuring
speculation without emitting speculative tokens; production (non-shadow)
would commit 1+a tokens per cycle for nearly the same cycle cost.

The 385/600 early stop (`finish_reason="stop"`) is likewise benign: the
shadow probe hit natural EOS — the `/bench` EOS-ban applies to the decode
probe mode, and the identity-mode probe intentionally lets the model stop.

## 3. What the shadow data says (the actual M1 deliverable)

Full analysis: 782 cycles, buckets uid=0/1 (2K ctx, byte-identical rerun
pair) and uid=2 (100K). Warmup cycle excluded from means. block_size=5, τ=0.5.

| Depth | n | a (mean accept/cycle) | γ mean | draft_ms | verify_ms mean | accept rate Σa/Σγ |
|---|---|---|---|---|---|---|
| 2K | 198×2 | **2.995** | 3.96 | 11.3 | 100.0 | 0.756 |
| 100K | 383 | **2.256** | 3.31 | 11.3 | 99.0 | 0.681 |

- verify_ms scales linearly with γ at 100K: 53.5 ms (γ=1) → 134.4 ms (γ=5);
  empirical per-row attention cost ≈ **20.2 ms/row @100K**, +22% over the
  cost model's `A@100K = 16.57 ms` (outside the ±15% band of §D.4 — the
  model's verify estimate was optimistic).
- The cost model's D=8 ms verify placeholder is now anchored: real D_cycle
  (draft+verify) = **110.3 ms at 100K** at γ_mean 3.31.

**Break-even arithmetic @100K** (baseline 29 tok/s sequential):
speculative tok/s = 1000·(1+a)/D_cycle = 1000·3.256/110.3 = **29.5 tok/s
(+1.8%)**. Break-even a\* = 110.3·29/1000 − 1 = **2.199**; measured a=2.256
clears it by only **0.057**. Per the §D.4 gate bands
(`a* ≤ a < a* + 0.30` ⇒ HOLD), this is squarely **HOLD — not PROCEED**. At
2K the picture is better (+17.7% headroom, a=2.995 vs a\*=2.395), but the
project north-star is 100K, and at 100K DSpark speculation is on the
knife-edge: within cost-model noise of zero gain, with a verify cost 22%
above the model's assumption.

**Byte-identity gate: FAILS under the shipped config — as it must.**
Shadow output was deterministic across reruns (byte-identical), but diverged
from the production build on the identical temp=0 prompt (the model even
counts the corpus differently: "46 sections/8 topics" prod vs "45 sections/9
topics" shadow). Root cause is documented, not novel: the shipped config runs
`EXO_DSV4_ROWSEQ_FULLBLOCK_MOE=0` + `EXO_DSV4_MOE_PARTS_ROWSEQ=shared`,
which leaves a known ~0.023%/row residual vs bitwise-exact (P4v2 doc §"the
fix is not bitwise-exact in the shipped config"). The `34478792b` commit's
byte-identity claim is conditional on `ROWSEQ_FULLBLOCK(_MOE)` — a condition
the live config doesn't meet. One early low-margin argmax flip (a counting
position) cascades the whole trajectory. This does **not** invalidate the
acceptance stats (a is measured self-consistently against the walked
trajectory's own argmax; distribution drift is ~0.02%/row), but it means the
M1 self-checking gate can't certify losslessness under the production MoE
config. Any future byte-exact shadow validation must set
`EXO_DSV4_ROWSEQ_FULLBLOCK_MOE=1` for the run's duration.

**Measurement debt if M2+ is ever funded:** n=1 per depth, 200/385-token
windows, no 10K shadow point (uid=1 turned out to be a 2K rerun, not 10K),
τ=0.9 never measured (relaunch #2 got zero probes), no 352.6K point.

## 4. PM decision: revert to production, shelve M2+ behind the HOLD

Options considered:
1. Keep diagnosing on the live shadow config (re-add SHADOW_LOG, re-probe,
   measure τ=0.9);
2. Revert to production defaults now, bank the M1 data, close the phase.

Chose **(2)**. Reasoning: M1's purpose was to produce a, D, and the post-τ
k-distribution for the P4v2 gate — that data exists and is sufficient for a
verdict. The verdict is HOLD: at 100K the measured acceptance clears
break-even by 2.6% of a, inside every uncertainty band we have, and the
verify-cost reality is 22% worse than the model assumed. Spending more
cluster time tuning τ on a mechanism that at best breaks even at the
north-star depth is not justified tonight; production has meanwhile been
sitting on an unvalidated experimental config, which is the standing
non-negotiable. τ=0.9 measurement (which prunes harder, trading a for
cheaper verify — plausibly net-positive given the γ-linear verify curve) is
the *first* thing M2 should do **if** the HOLD is ever revisited; it is
explicitly not being done now.

Reverted via full relaunch (`EXO_SPECULATIVE=0 EXO_DSV4_MTP=0
./start_cluster.sh`, relaunch #3 of this phase) rather than tag-pinning:
`34478792b` == origin/main == the known-good code *plus* M0/M1, and both
M0 and M1 are env-gated no-ops under production flags. Relaunching on main
also live-validates M0 (the head-load gate): with `SPECULATIVE=0 MTP=0
DSPARK=1` (script default), the gate's disjunction is false on every
branch, so the ~10 GB/node DSpark head must NOT load.

## 5. Post-revert verification (relaunch #3)

Relaunch #3 (`EXO_SPECULATIVE=0 EXO_DSV4_MTP=0 ./start_cluster.sh`):
`READY (2/2)`, `EXIT=0`. Verified on both nodes via `ps eww`:
`EXO_SPECULATIVE=0 EXO_DSV4_MTP=0 EXO_DSV4_DSPARK=1`, no `SPEC_SHADOW`
vars.

**M0 live-validated:** both nodes log `DSpark head load SKIPPED (~10 GB/node
reclaimed): EXO_DSV4_DSPARK=1 but no runtime consumer is reachable —
EXO_SPECULATIVE=0 EXO_DSV4_MTP=0 sharding=TensorShardMetadata`, zero
`DSpark draft head attached` lines. The gate works exactly as committed.

**Decode health (p4_shadow_gate_probe, real generated text inspected):**

| Probe | tok/s | gap med/p95 ms | Notes |
|---|---|---|---|
| identity @2K, 200 tok | 30.05 | 31.9 / — | Output **byte-identical** to the pre-shadow production baseline (`prod_short.txt`) — trajectory restored exactly |
| decode @100K (100,013 prompt), 600 tok | **28.73** | 34.2 / 60.4 | Coherent corpus-summary text, `finish_reason=length`, vs 8.51/189ms p95 under shadow and ~29 known-good |

Cluster left healthy, serving production config on `34478792b` (origin/main).

