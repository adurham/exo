# DSv4 Structural Path to 35 t/s — mlx/jaccl C++ + mlx-lm changes are FAIR GAME

> **For Hermes:** We own mlx, mlx-lm, jaccl, and exo. Phases A-D in the
> previous to_35tps plan are exhausted (Phase A bisect = moot, Phase B
> ack_sync_pre = chasing a broken-quality number, Phase C topk =
> TOPK=160 produces gibberish per the needle probe). What's left is the
> structural floor: 43 layers x ~1.4ms per-layer all_sum drain = ~60ms
> verify. The 1.4ms is NOT a fundamental constant — it's what jaccl
> currently delivers. We can change jaccl.

**Goal:** >=35 t/s at quality-correct (TOPK=512) gamma=2 100K c=1 MTP-on,
10/10 clean sigma<0.3 0 errors, needle probe PASS.

**Baseline anchor:** `baseline-2026-05-18-mtp-g2-topk512-30.06` (30.062
t/s sigma=0.059 10/10 clean).

**Structural budget to hit 35:**
- Current per-cycle: 4.5ms draft + 57ms verify + 0.8ms accept = 62.65ms
- alpha_2 = 1.04 -> 2.04 tokens/cycle
- t/s = 2.04 / 0.06265 = 32.6 (theoretical, we measure 30.06 with overhead)
- Target: t/s = 35 -> per-cycle = 2.04 / 35 = 58.3ms
- **Need to cut ~4.3ms off the cycle.** Most of that comes from verify.
- Verify decomposition (from Phase B probe): 43 layers x ~1.4ms = 60ms,
  ~~which is the "1 fence drain at end" measurement.
- Per-layer 1.4ms decomposes as (rough estimate):
  - actual data transfer (100KB / 80 Gb/s TB5) ~ 10 us
  - ack_sync_pre RTT ~ 30-50 us
  - ack_sync_post RTT ~ 30-50 us
  - GPU<->CPU encoder dispatch ~ 10-50 us
  - mutex+poll loop ~ 5-20 us
  - rest = ~1.2ms unaccounted -> likely the **lock+driver+poll-loop** overhead
- If we can cut per-layer cost by 0.1ms x 43 = 4.3ms saved -> we hit
  35 t/s.

**Hard constraint preserved:** TOPK=512, model-correct quality. We
will NOT chase numbers at broken TOPK=160.

---

## Critical Context

### Today's confirmed dead ends
- TOPK=160 fails needle test at 100K (BOS-token output) — DO NOT
  benchmark at TOPK<512 unless quality probe passes too.
- mlx-lm compile-boundary collapse (Levers 1, 2) — both regressed.
- gamma=3 — alpha_3 = 0.37 below break-even.
- FENCE=1 (per-layer fence) — adds 6ms verify cost vs FENCE=43.
- mc_ping unconditional file write — removed (no measurable impact).

### What we actually own and can change

```
~/repos/exo/
├── mlx/                          ← we own the mlx C++ collective code
│   └── mlx/distributed/jaccl/
│       ├── mesh.cpp              ← MeshGroup::all_reduce entry (line 524)
│       ├── mesh_impl.h           ← all_reduce body (line 52),
│       │                          ack_sync_pre (573),
│       │                          ack_sync_post (623),
│       │                          drain_acks (683)
│       └── ring_impl.h           ← ring-based fallback for large size
├── mlx-lm/                       ← we own the Python model
│   └── mlx_lm/models/deepseek_v4.py
│       ├── DeepseekV4MoE.__call__ (1106)   ← per-layer all_sum + fence
│       ├── DeepseekV4Model.__call__ (2387) ← top-level forward
│       └── _sparse_pooled_attention (783)
└── src/exo/worker/engines/mlx/
    └── speculative/
        ├── mtp_module.py         ← draft_tokens, per-step fence (line 654)
        └── dsv4_mtp.py           ← _speculative_next (1255)
```

### Per-call cost breakdown (the 1.4ms target)

From mesh_impl.h reading and the Phase B probe (warm steady-state at
FENCE=1, verify L=3, per-layer eval ~1.4ms median):

1. **Two ACK RTTs per call** (`ack_sync_pre` + `ack_sync_post`):
   each is a cross-rank round-trip on the ACK QP. Thunderbolt 5
   RDMA RTT is typically 30-50us. Two RTTs = ~80us per call.
   **At 43 calls/forward = 3.4ms of barrier-only cost per verify.**

2. **Data transfer** (100KB per call at verify, B*L*hc_mult*hidden*2B):
   At TB5 ~80 Gb/s = 10 GB/s effective, 100KB = 10us. Negligible.

3. **GPU<->CPU encoder dispatch** (`encoder.dispatch(...)` in
   mesh.cpp:608): the lambda is queued onto a CPU command encoder,
   meaning the all_sum doesn't happen on the GPU stream — it
   happens on a separate CPU thread that fires after the GPU
   completes the matmul. The CPU<->GPU sync alone is ~20us.

4. **`collective_mutex_`** (mesh.cpp:603): every collective on the
   MeshGroup serializes. 43 layers, 1 group -> all 43 calls go
   through the same mutex.

5. **`ack_pre_fastskip`** is already implemented (mesh_impl.h:580)
   but env-gated to OFF. Enabling could save ~30us per call when
   the chain invariant holds = ~1.3ms per verify.

### What's actually optimizable

| Target | Approach | Best-case saving | Risk |
|--------|----------|------------------|------|
| ack_sync_pre on uniform-config chains | Enable EXO_JACCL_ACK_PRE_FASTSKIP=1 (existing code) | ~1.3ms/verify (~+0.7 t/s) | LOW |
| Combined pre+post barrier | Batch both round-trips into a single 1-RTT exchange | ~1.7ms/verify (~+1 t/s) | MED |
| Eliminate ack_sync_post for non-final layers | Most layers don't need a post-barrier — only the LAST layer's allsum needs to be fully drained before lm_head | ~1.7ms/verify (~+1 t/s) | HIGH (race risk) |
| Move all_sum onto GPU stream (no CPU bounce) | Use mlx's GPU-side queue for the dispatch instead of CPU encoder | ~1ms/verify (~+0.6 t/s) | HIGH (rewrite) |
| Parallel allreduce (no mutex) | Drop collective_mutex_ for the master TP group; ensure call_ids are still ordered | ~0.5ms/verify (~+0.3 t/s) | MED |
| Coalesce layers into batched allreduce | Stack N layers' outputs into one larger collective | Variable | HIGH |

The lowest-hanging is **ack_pre_fastskip** because the code already
exists, is env-gated, and has been validated to be correct (no
regressions in commits since). The user's perf-debugging history
shows it was tried May 17, found to be correct, but never properly
benched at TOPK=512.

---

## Phase F: Enable ack_pre_fastskip + bench at TOPK=512 (~30 min)

**Objective:** prove the existing fastskip code delivers real gains at
quality-correct TOPK=512.

### Step 1: Verify deployed mlx HAS the fastskip code

Both `mesh_impl.h:580` and the `EXO_JACCL_ACK_PRE_FASTSKIP` env are
in the venv. Confirm:

```
SSH_OPTS="-i ~/.ssh/exo_cluster -o IdentitiesOnly=yes"
ssh $SSH_OPTS adam.durham@192.168.86.201 'strings \
  ~/repos/exo/.venv/lib/python3.13/site-packages/mlx/lib/libmlx.dylib \
  | grep EXO_JACCL_ACK_PRE_FASTSKIP'
```

Should return non-empty. If not, the fix branch isn't deployed.

### Step 2: Confirm start_cluster forwards the env

```
grep ACK_PRE_FASTSKIP start_cluster.sh
```

Should show 1 forwarding line. (Already there from earlier
investigation.)

### Step 3: Relaunch cluster with TOPK=512 + fastskip ON

```
EXO_DSV4_FENCE_EVERY_N_LAYERS=43 \
  EXO_DSV4_INDEX_TOPK=512 \
  EXO_JACCL_ACK_PRE_FASTSKIP=1 \
  ./start_cluster.sh
```

### Step 4: Task 0 verification + 3-iter bench

- Inference probe -> "4"
- Quality probe at 100K -> needle PASS
- 3-iter 100K bench at concurrency=1 iterations=3 warmup=1

### Step 5: Decide

If clean 3/3 >= 30.5 t/s, ALSO run a 10-iter for the champion claim.
If <= 30 t/s with no perf lift, deploy was broken or fastskip isn't
hitting the invariant chain on master TP — move to Phase G.

### Phase F success criterion

10/10 iters all >= 30.5 t/s, sigma < 0.3, needle PASS. Tag and push.

### Phase F abort criterion

If fastskip doesn't measurably help (variance band), move to Phase G.

---

## Phase G: Combine ack_sync_pre + ack_sync_post into ONE round-trip (~3-4 hr)

**Objective:** the dedicated pre+post barriers each cost an RTT.
Combine them. A single "bidirectional ACK" exchange at lambda START
serves both purposes:
- ack_sync_pre: confirms peer has entered THIS lambda
- ack_sync_post: confirms peer has drained PRIOR lambda

These are the same constraint expressed twice. If lambda N's
ack_sync_pre = lambda N-1's ack_sync_post (both prove peer is at
boundary between N-1 and N), we can drop one.

The existing `ack_pre_fastskip` (Phase F) is exactly this realization
but conservatively only on a contiguous chain. Phase G generalizes:
**there is only one barrier per inter-lambda boundary, regardless of
whether you call it "pre of N" or "post of N-1".**

### Step 1: Read ack_sync_pre + ack_sync_post side by side

The two functions post-and-wait the same WRs to the same QPs. The
only semantic difference is WHEN they're called (start vs end of
lambda). Re-derive whether a SINGLE call at the boundary suffices.

### Step 2: Design the merged barrier

Sketch (in mesh_impl.h):
- Add `ack_sync_boundary(call_id)` which does the pre+post combined.
- Call sites: at the END of all_reduce (before return), do the
  combined exchange. Drop the pre at the next lambda start.
- Caveat: the very FIRST collective on the group needs an explicit
  setup barrier (handled at MeshGroup ctor).

### Step 3: Microbench locally (before cluster build)

Write a small C++ unit test (`mlx/tests/jaccl_barrier_microbench.cpp`)
that times a tight loop of allreduce-with-old-barrier vs
allreduce-with-merged-barrier. Need to mock the peer side or run on
loopback. Goal: confirm the merge actually halves barrier cost.

### Step 4: Build mlx wheel locally + deploy via local-editable pyproject

(Avoiding the May-18 12:48 deploy regression: edit pyproject to
`path = "/Users/adam.durham/repos/exo/mlx", editable=true` so `uv
sync` picks up the local build instead of fetching a remote wheel.)

### Step 5: Task 0 protocol + 3-iter bench

If 3-iter shows >=1 t/s lift over Phase F baseline, run 10-iter for
champion claim.

### Phase G success criterion

10/10 iters >= 31.5 t/s, sigma < 0.3, needle PASS. Champion tag.

### Phase G abort

If the merge breaks correctness (intermittent stalls, needle fail) or
no meaningful perf lift, revert to Phase F state.

---

## Phase H: Drop ack_sync_post on non-fence layers (~4-6 hr)

**Objective:** with EXO_DSV4_FENCE_EVERY_N_LAYERS=43, only ONE layer
per forward needs a CPU-side fence (mx.eval). The other 42 layers'
all_sums can in principle be FULLY async — no ack_sync_post barrier
needed, just the WR completions.

The current ack_sync_post exists to prevent cross-call FIFO corruption
on UC QPs (the May-18 bistability root cause). But if the lambda
RETURNS without a post-barrier, the NEXT lambda's pre-barrier still
protects us. So the post-barrier is redundant when there's a pre on
the next lambda.

Combined with Phase G's merged barrier, this is: **one boundary
barrier per inter-lambda transition, not two.** Phase G already does
this. So Phase H reduces to: **ensure the merged barrier is the
ONLY barrier; eliminate any explicit `mx.eval` on intermediate
layers.**

Actually this is what FENCE=43 already does. So Phase H is mostly
a verification that Phase G correctly captured all the synchronization
the FENCE was previously providing.

### Step 1: Add per-layer ack_sync_post bypass mode

New env: `EXO_JACCL_SKIP_POST=1` — skip post barrier on non-final
calls. The final layer's mx.eval still flushes everything.

### Step 2: Test in isolation

3-iter bench. If correctness holds (needle PASS), measure the saving.

### Step 3: Combine with Phase F + G

3-iter, then 10-iter.

### Phase H success / abort

Same as Phase G.

---

## Phase I: GPU-stream allreduce (~1-2 days, MED-HIGH RISK)

**Objective:** Currently `encoder.dispatch(...)` queues the lambda
onto a CPU encoder, which means the all_sum runs on a CPU thread
AFTER the GPU matmul completes — that's a forced GPU->CPU sync at
every layer boundary.

If we can fire the all_sum from the GPU command queue directly (or
at least let the CPU dispatch happen while the GPU is still doing
later work), we hide the sync cost.

This is mlx-internal work. Out of scope unless Phases F-H don't get
us to 35.

---

## Phase J: Concurrency / aggregate scaling (~1 hr, ESCAPE HATCH)

**Objective:** If single-stream c=1 is structurally capped, see what
the aggregate metric looks like at c=2 / c=4.

### Step 1: 3-iter bench at c=2 + c=4 on production config

If user accepts aggregate, this is the easy win. If not, document and
move on.

---

## Hard Constraints (Do NOT)

(Inherited from prior plans:)
- DO NOT measure perf at TOPK<512 without quality probe PASS.
- DO NOT retry mlx-lm compile-boundary collapse (Levers 1, 2).
- DO NOT modify pyproject to a remote-branch source until local-editable
  works first.
- DO NOT touch the `mtp_module.py:654` per-step fence.
- DO NOT claim a champion without 10 iters >= goal, sigma < 0.3,
  0 errors, AND needle probe PASS.

(New for this plan:)
- DO NOT modify jaccl C++ without an in-tree unit test that exercises
  the change. The May-18 02:02 false-champion happened because the
  fix-branch was never tested in deployed state. We will not repeat
  that.
- DO NOT use `uv add --source git+url@branch` patterns to deploy mlx
  changes — use local-editable pyproject + local wheel build.

---

## Files Likely To Change

- `mlx/mlx/distributed/jaccl/mesh_impl.h` (Phase G/H — the actual fix)
- `mlx/CMakeLists.txt` or `mlx/tests/CMakeLists.txt` (Phase G Step 3 unit test)
- `mlx/tests/jaccl_barrier_microbench.cpp` (new, Phase G Step 3)
- `pyproject.toml` (Phase G/H — local-editable mlx pin, temporary)
- `uv.lock` (auto-regenerated when pyproject changes)

**Do NOT modify** (this plan):
- mlx-lm model code (already exhausted)
- start_cluster.sh defaults (TOPK=512 stays, FENCE=43 stays)
- mtp_module.py / dsv4_mtp.py
- mlx submodule pointer (we'll work in mlx/ working tree directly)

---

## Tests / Validation

Task 0 protocol unchanged. For Phase G/H specifically:
1. mlx unit test passes locally (cmake + ctest)
2. mlx local wheel build succeeds
3. Cluster deploys via local-editable pyproject without the May-18
   12:48 wedge symptom (READY time < 2 min, iter 0 wall < 500s)
4. py_compile + strings-grep verify mlx changes deployed
5. Inference probe -> "4"
6. Small-prefill 2-iter bench -> 0 errors
7. Needle probe at 100K -> PASS
8. 3-iter 100K bench at c=1 TOPK=512 GAMMA=2 FENCE=43
9. THEN 10-iter for champion claim

Per-bench discipline: 2 LOW iters = answer known, abort + revert.

---

## Risks, Tradeoffs, Open Questions

### Risks

1. **mlx C++ build complexity.** Building mlx from source on macOS
   requires Xcode toolchain. The wheel-build path used by uv may
   differ from the cmake path. Phase G Step 4's local-editable pin
   sidesteps wheel rebuild but requires `pip install -e ./mlx` to
   work cleanly.

2. **jaccl correctness regressions.** Removing or merging barriers
   is exactly the territory where the May-18 02:02 false-champion
   landed (claimed fix, actually broken). MITIGATION: in-tree unit
   test, microbench, and quality probe BEFORE any cluster bench.

3. **GPU<->CPU synchronization assumptions in mlx-lm.** The current
   per-layer fence pattern in DeepseekV4MoE assumes ack_sync_post
   has flushed the previous call. If we change the barrier shape,
   the model code may need to change too.

### Tradeoffs

- **Phase F is cheap and may already deliver +0.5-1 t/s.** Doing it
  first hedges against the riskier C++ changes.
- **Phase G + H together is the structural fix.** Phase G alone may
  save ~1 t/s; combined ~2 t/s.
- **Phase I (GPU-stream allreduce) is the biggest potential win** but
  also the biggest engineering effort. Save for after F-H prove the
  collective path is the bottleneck.

### Open Questions

1. Is the actual per-call cost dominated by the 2 ACK RTTs or by
   something else (encoder dispatch, mutex contention)? **Phase F's
   measured saving will quantify the ACK cost.** If fastskip saves
   ~1 t/s, the ACK barriers are ~3.4ms/verify (validates math). If
   it saves <0.5 t/s, the cost is elsewhere.

2. Can `EXO_JACCL_ACK_PRE_FASTSKIP=1` cause the May-18 12:48 wedge?
   Memory says the wedge happened on `mlx@fix/jaccl-ack-qp-top-level`
   branch which had MORE changes than just fastskip. The fastskip
   code is on `mlx@main` (commit 0b8aca69) and should be safe.

3. Where else in jaccl is there reducible per-call overhead? Probe
   Phase F first to find out.

---

## Time Budget

| Phase | Description                              | Wall  | Best-case t/s lift |
|-------|------------------------------------------|-------|----|
| F     | Enable ack_pre_fastskip + bench TOPK=512 | 30 min | +0.5-1 |
| G     | Merge pre+post into 1 barrier            | 3-4 hr | +1-1.5 |
| H     | Drop redundant post on non-fence layers  | 4-6 hr | +0.5-1 |
| I     | GPU-stream allreduce                     | 1-2 day | +1-2 |
| J     | Concurrency probe (escape hatch)         | 1 hr | varies |

**F+G+H = ~7-10 hr, projected lift +2-3 t/s -> 32-33 t/s.**
**+I = +1-2 day, projected total +3-5 t/s -> 33-35 t/s.**

To HIT 35 t/s exactly at c=1 quality-correct, we likely need F+G+H+I.
**This is a multi-day engineering project, not a tonight job.**

---

## Quick Reference

### Launch — Phase F (fastskip enabled)
```
cd /Users/adam.durham/repos/exo && \
  EXO_DSV4_FENCE_EVERY_N_LAYERS=43 \
  EXO_DSV4_INDEX_TOPK=512 \
  EXO_JACCL_ACK_PRE_FASTSKIP=1 \
  ./start_cluster.sh
```

### Launch — Phase G/H (local mlx build)
```
# After editing pyproject to local-editable mlx + uv lock
cd /Users/adam.durham/repos/exo && \
  EXO_DSV4_FENCE_EVERY_N_LAYERS=43 \
  EXO_DSV4_INDEX_TOPK=512 \
  ./start_cluster.sh
```

### Bench template (3-iter quick)
(same as prior plan; 6.5 min/iter)

### Quality probe
```
ssh $SSH_OPTS adam.durham@192.168.86.201 'zsh -l -c "cd ~/repos/exo && \
  uv run python3 bench/quality_probe_dsv4.py \
    --base-url http://localhost:52415 \
    --model mlx-community/DeepSeek-V4-Flash-8bit \
    --target-tokens 100000 \
    --timeout 1200 \
    --out ~/quality_phaseX.json \
    --label phaseX"'
```

---

## Remember

- Quality probe BEFORE perf claim. Always.
- Phase F is the cheapest experiment. Do it first.
- mlx C++ changes need a unit test. The "deploy and pray" approach
  burned us May-18 02:02 (false champion -> 4.3 t/s redeploy).
- Don't claim a champion on 1-3 iters. 10 minimum.
- TOPK=512 is non-negotiable on quality grounds.
