# ROUND-Q1E-Q3ARM.md — experts requant, Round 2: the q3-class arm (GATE 1 RE-DECISION)

Author: Phase-20 PM (delegation). Date: 2026-10-10 (CDT). Worktree `/private/tmp/phase20-campaign`.
Branch `deploy/phase20-campaign`. Round 2 of the experts-requant campaign.

**Offline-first. NO live spend, NO encode spend, NO boots.** Phase A desk checks are ~0 GPU; Phase B
is one offline single-node bench on `studio2` (idle-guarded); Phase C is CPU-only trace mining.
Production is not touched; the only live thing is the idle-guard + canary (read-only). The retained
sampled original shards (`studio1:/tmp/q1d_mem/orig`, layers {0,20,39}, 22.18 GB) are **retained**.

---

## 0. DECLARED BUDGET (fixed BEFORE spending)

- **Boots: 0.** No relaunch, restart, or POST on either node.
- **GPU-hour cap: ~1–2 GPU-h offline** on studio2 (Phase B, idle-guarded). Phase A ≈ 0 GPU (node
  observations + a one-layer packing probe). Phase C = 0 GPU (local CPU numpy).
- **Live spend: 0.** No generation requests. Any node-touching step is idle-guarded.
- **Encode spend: 0.** Arms are built in-memory from the sampled original shards; nothing is written back.
- **Hard stop at the GATE-1 RE-DECISION** even if partial (write what is known). Genuine blocker →
  STOP and report.

### Entry state (verified ~01:35 CDT 2026-10-10)
Production LIVE: exo `deploy/next19-dense @ 99e2966ee` + mlx-lm `689e4ea`; runner env
`DSV41_DENSE=affine6` + `DSV41_DENSE_POLICY=layers.*.ffn.shared_experts.*=q8g64`. Cluster idle,
canary healthy (studio1 14.85 / studio2 14.86 t/s). Rollback: prev prod `fb4f9290b`/`16830e1`;
in-place `DSV41_DENSE=exl3`.

---

## A1. DESK CHECK — live wired limit (the load-bearing constant)

**CONFIRMED (live, both nodes, 2026-10-10).**

| node | `sysctl iogpu.wired_limit_mb` | `hw.memsize` | wired limit |
|---|---:|---:|---:|
| studio1 (`macstudio-m4-1`) | **120000** | 137 438 953 472 B | 125.83 GB = 117.19 GiB |
| studio2 (`macstudio-m4-2`) | **120000** | 137 438 953 472 B | 125.83 GB = 117.19 GiB |

**Setter (mechanism):** `start_cluster.sh:1474` — `_want_wl="${DSV4_WIRED_LIMIT_MB:-120000}"`, applied
per node via `sudo -n sysctl iogpu.wired_limit_mb=$_want_wl` (line 1475) with a NOPASSWD sudoers rule;
the launcher then reads the value back (line 1478) and WARNs on a mismatch. ⇒ the live 120000 is the
**launcher default**, overridable with `DSV4_WIRED_LIMIT_MB`. (A separate code path at line 1452 sets
32000 for a low-memory mode; not active here. The `bench/section17_memory_headroom_check.py` comment
that says "115GB/node" is STALE and is the source of the Round-1 doc's error — see §C0.)

**W_live = 120000 MiB = 125.83 GB = 117.19 GiB.** All Round-2 arithmetic uses this value.

---

## C0. THE BUDGET-ARITHMETIC CORRECTION (recorded for the record; permissive error)

The Round-1 doc (`ROUND-Q1D-MICROBENCH.md` §6) computed its memory-fit limb against the **full node
RAM (128 GiB = 137.44 GB)**, not the production wired limit, and used a stale `115000` MB wired
figure in prose. Verified: the doc's `≤3.72 bpw` budget line = (137.44 − 11.1 − 0.30) / 33.95 exactly,
i.e. it subtracted the non-weight terms from the **physical** 137.44 GB, not from the wired ceiling
the process can actually touch.

**Effect on the Round-1 verdict: NONE.** This is a *permissive* error — the doc's ceiling was too
generous, so a correct computation is *stricter*, and every measured native arm (q4g64 155.4,
q4g32 163.9, mixed 178.0, q5g64 189.4, q6g64 223.3 GB/rank) still fails — and fails harder against the
live wired ceiling (W_live − non-weight ≈ 114.8 GB ⇒ ≤3.38 bpw). **The Round-1 FAIL stands.** The
correction matters only because it re-opens exactly one question the doc dropped: whether a
**q3-class** arm (never run in Round 1) fits. That is what this round measures.

### Correct fit line (per rank, GB), used in the Round-2 rule
Exact from the checkpoint header: per-rank routed-expert weight count `N_rank = 271 790 899 200`;
non-expert rest = 10.71 GB; KV@91K = 0.30 GB.

```
total_GB/rank = 10.71 (rest) + 0.30 (KV@91K) + (N_rank/8/1e9) * bpw
              = 11.01 + 33.974 * bpw          [exact]
             ≈ 11.1  + 33.95  * bpw           [the campaign shorthand; max ~0.4 GB low]
```

Cross-check vs the 5 measured arms (doc §6): q4g32 163.89 ✓, q4g64 155.40 ✓, q5g64 189.37 ✓,
q6g64 223.35 ✓, EXL3 109.19 (doc 109.26) ✓ — all within 0.1 GB. Confirmed.

---

## A2. DESK CHECK — incumbent (EXL3) non-weight peak  → P

**Deliverable:** `P = measured non-weight peak at the production target context (~91K) + 2 GB safety
margin`, derived from the live serve (VM `172.16.0.42:8428`, exo `/state`, node runner logs, or a
short idle-guarded observation), and reconciled with the doc's 10.71 (rest) + 0.30 (KV) = 11.01 GB.

_PENDING — filled from `raw/pricing/q1/q1e/a2_nwpeak.json`._

---

## A3. DESK CHECK — actual 3-bit packing  → B_arm

**Deliverable:** effective bpw of `mx.quantize(expert_weight, bits=3, group_size=128/64)` measured
from `nbytes` on one sampled layer's expert weight (studio2), and the per-arm expert size:

```
B_arm = 33.974 * bpw_measured   (GB/rank, all 40 MoE layers)
```

Discriminates the three candidate packings: **3.125** (dense 3-bit, scale-only), **3.25**
(scale+bias), **3.325** (older 10-vals-per-uint32). Decides B_arm *precisely* — the rule is a
knife-edge (see below), so this measurement is load-bearing.

_PENDING — filled from `raw/pricing/q1/q1e/a3_packing.json`._

---

## R. PRE-REGISTERED DECISION RULE (frozen BEFORE Phase B)

> **The q3 arm runs IF AND ONLY IF `B_arm + P ≤ W_live` (= 125.83 GB).** Otherwise the lead **CLOSES**
> with the measured arithmetic (a valid outcome).

Candidates:
- **q3g128** — the shippable candidate (`bits=3, group_size=128`), built from the sampled ORIGINAL weights.
- **q3g64** — **NON-SHIPPABLE reference only** (`bits=3, group_size=64`).

Arithmetic (evaluate once A2/A3 land):
| arm | bpw (measured) | B_arm GB | + P GB | vs W_live 125.83 | rule |
|---|---:|---:|---:|---|---|
| q3g128 | _pending_ | _pending_ | _pending_ | _pending_ | _pending_ |
| q3g64 (ref) | _pending_ | _pending_ | _pending_ | _pending_ | _pending_ |

---

## B. PHASE B — the one arm (only if R passes)

Same harness/session discipline as Round 1 (studio2 offline, same-session EXL3 baseline, cache-busted
fresh routing per rep + SLC flush outside the timed region, warm, medians, WALL **and** GPU ms).
Extends `bench/p20_q1d_native_experts.py`; arms built from the **sampled original FP8 shards**
(`/tmp/q1d_mem/orig`), not the EXL3 reconstruction.

- **Arms:** EXL3 baseline (same session) | **q3g128** | q3g64 (reference) | q4g64 context (reuse R1,
  re-run only if cheap).
- **Shapes:** sampled layers {0,20,39}; **real trace routing**; at **R=4 AND R=1** (R=1 = 6 experts/layer —
  the fixed overhead share is larger, must clear on its own). Report mean + p95(21) + max(24)
  unique-expert counts.
- **Speed bar (pre-registered):** **≥30 % WALL saved** on the ×40-layer extrapolation vs same-session
  EXL3, at **BOTH R=4 and R=1**.
- **Correctness bar (pre-registered, relative to incumbent — fixed before looking):** vs the bf16
  originals: **q3g128 per-layer rel-MSE ≤ 1.10 × EXL3's rel-MSE on EVERY sampled layer**, **AND
  q3g128 cos ≥ EXL3 cos − 0.002**. (Note for the writeup: EXL3 is trellis-coded and can beat affine q3
  per bit — quality, not speed, is the likely killer; the arm measures it. A per-layer pass clears
  GATE 1 only; end-to-end eval still required before any ship.)
- **Scope guard:** EXISTING `gather_qmm` 3-bit path only — if q3 needs new kernel work, use the
  generic dequant+matmul path as a floor or CLOSE; do not let this become a kernel project.

---

## C. PHASE C — trace mining (free; runs regardless)

From the 531×40 real clamp trace (`raw/…/p30_exl3_trace_clamp.json`; 21 240 real gate decisions):

1. Per-layer **unique-expert UNION** over the full trace + frequency-rank curve.
2. **Hot-set concentration**: share of tokens covered by the top-k experts per layer (k=1…24); is the
   hot set **STABLE** across trace windows/prompts? (strong hot set → future hot-q4/cold-EXL3 mixed
   path; flat → Q4 dead.)
3. Per-rank **max/mean touched-expert IMBALANCE** (wall = slowest rank; if 2×+ → traffic-aware
   placement note).
4. **Batch-to-batch hot-set overlap**.

_PENDING — filled from `raw/pricing/q1/q1e/c_trace_mining.json`._

---

## G. GATE 1 RE-DECISION (pre-registered)

- **PASS** (q3g128 clears speed + correctness + memory rule) → experts continue with a concrete shape
  (q3g128; full encode from original source is a FUTURE round with its own quality/battery gates + a
  source-availability decision).
- **FAIL** (any limb, or a desk check) → **CLOSE** the experts lead with measured cause.

_PENDING._

---

## E. END STATE / BUDGET ACCOUNTING

_PENDING._
