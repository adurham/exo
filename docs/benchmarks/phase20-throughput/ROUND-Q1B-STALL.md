# ROUND-Q1B-STALL.md — root-cause + fix of the world=2 first-forward affine stall

Author: Phase-20 PM (delegation, ROUND-Q1B). Date: 2026-10-09 (CDT). Worktree `/private/tmp/phase20-campaign`,
branch `deploy/phase20-campaign`. Owner standing authorization: no check-ins; root-cause fixes only; ship gates
unchanged. Prior round `ROUND-Q1-QUANT.md` closed NEGATIVE (control clean, treatment `DSV41_DENSE=affine6`
stalled 4/4 on the first forward).

Entry state (verified this round): production `fb4f9290b` (exo) + mlx-lm `16830e1` on BOTH nodes
(`ssh studio1|studio2 'git -C ~/repos/exo rev-parse --short HEAD'` = fb4f9290b), gates unset, serving.
Eval branches (both pushed to adurham forks): mlx-lm `deploy/q1-dense-qn @ e444cbd`, exo `deploy/q1-dense-qn @ e4c2cb460`.

---

## 0. DECLARED BUDGET (fixed BEFORE spending)

- **Deploy boots: ≤3 total (+1 reserve).** A "boot" = a full `start_cluster.sh` relaunch. A same-boot arm switch
  (exo-process restart with a different `DSV41_DENSE`) is NOT a boot.
  - **Boot #1** — eval build live (new fix sha): SMOKE FIRST (`affine6` + READY + canary + tiny Paris probe on the
    first forward). If smoke passes → same-boot control re-anchor (`exl3`, 2 reps benign) + treatment A/B
    (`affine6`, ≥3 reps benign 20K + agentic 91K, salt `q1b`) + deep-path smoke (>16K prefill + 91K agentic).
  - **Boot #2** — ship (make affine6 default) if G-B/G-C pass; else the single `affine5` retry if budget allows.
  - **Boot #3 / reserve** — only if a genuine retry/re-ship is warranted.
- **Falsifier (pre-registered):** if the smoke boot still stalls after ONE fix iteration → CLOSE the eval negative,
  restore `fb4f9290b`/`16830e1`, no grinding. A second fix iteration is allowed ONLY if the first smoke produced
  DISTINCT NEW evidence AND budget remains.
- Stop at 3 boots regardless. Negative at that point = close + restore.

## 1. GATES (same as the owner-approved Q1 eval — re-stated)

- **G-B (cluster A/B, within-build/same-boot):** agentic round-time **Δ ≥ 15 ms** vs control with `mean_accepted`
  within noise; benign no regression beyond noise. (Target ~18.5–19.4 ms.)
- **G-C (battery):** **CLEAN** (needles 6/6, tools 10/10, prose 0 DIRTY/0 REVIEW, park PASS) + logit drift recorded.
- **G-D (ship):** fresh deploy + canary + parity smoke within noise + both-fork tags (`known-good-dense-q6-<date>`)
  + PERFORMANCE_HISTORY on main same-turn.

## 2. EVIDENCE INVENTORY (what the stall is, precisely)

**Build:** exo `deploy/q1-dense-qn @ e4c2cb460` + mlx-lm `deploy/q1-dense-qn @ e444cbd` (off prod `16830e1`).

**The change (mlx-lm `exl3_build.py`, +131 lines):**
- New `AffineProj.from_weight` / `_set_weight` (quantize a pre-sliced fp16 `[out,in]` weight, group_size=64).
- New `_block_bounds` (rank `[a,b)` of `n` 16-wide tiles on 128-wide blocks — shared by `_slice_dense`/`_slice_weight`).
- New `_slice_weight` (reconstruct → `.T` → slice on 128-wide block boundaries) + `_dense_slice` (per-rank dense
  group in the active mode: `AffineProj` for affineN, `EXL3Linear` for exl3).
- **Gate removal (the load-bearing change):** `attn_tp`/`shared_tp` were gated on `DENSE_MODE == "exl3"`; the gate is
  gone. `tp_on = _DENSE_TP or not DENSE_MODE.startswith("affine")`; `DSV41_DENSE_TP=0` restores replicated affine.
  Previously affine ran REPLICATED (each rank computed the FULL dense slice → ~break-even, not the priced 2.4×).
- `affine5` added to the value set.

**Stall signature (deterministic, 4/4 watchdog cycles):** body loads clean (40/40 layers, DSpark head attached,
hier geometry), then the runner goes silent and the hang watchdog SIGKILLs it — `hung: 1 task(s) in progress, no
event for 66-68s (>45s). SIGKILLing`. Never reaches READY.

**Hang stacks — NEW this round, the decisive detail (thread asymmetry):** `/tmp/exo_hang_*.txt` on the nodes.
The 12:57–13:03 treatment cycles:
- studio1 `/tmp/exo_hang_85628.txt` (12:57): **NO** jaccl frame — main thread only, parked in
  `mlx::core::eval → mlx::core::Event::wait() → __psynch_cvwait` (GPU command-buffer wait).
- studio1 `/tmp/exo_hang_85866.txt` (13:00), `86167.txt` (13:02): main thread in `eval→Event::wait` **plus** a thread
  in `jaccl::MeshGroup::all_sum(void const*, void*, unsigned long, int) → reliable_all_reduce_v2<bfloat16,SumOp>`.
- studio2 `/tmp/exo_hang_95286.txt` (12:58), `95602.txt` (13:00), `95924.txt` (13:03): same — `MeshGroup::all_sum`
  + `eval→Event::wait`.
→ **The two ranks are in DIFFERENT states: one in a bf16 `all_sum`, the other in a GPU-event wait** — an
**ordering / pairing divergence** (rank A enters collective N while rank B is elsewhere), not a generic load hang.
(Contrast: the older 02:00–02:51 stacks on both nodes are the generic `MeshGroup::all_sum` + `eval_impl` shape.)
**The hung collective is bf16** → the attention partial-sum `all_sum` (`attention.py:241`, `out` is bf16), not the
fp32 MoE `all_sum`. `NOT` the known 2026-08-18 `MLX_JACCL_DATA_RECV_POOL` rank-0 hang (no `unordered_map::at`
init-failure lines; stuck frame is a post-load `all_sum`).

## 3. COLLECTIVE-PARITY MAP — call sites to audit for the first warmup forward (world=2)

Registered at build time by `build_block`/`build_model` (mlx-lm `exl3_build.py`):
| flag / field | set when | drives which collective |
|---|---|---|
| `blk.attn.group = group` (also shrinks `n_heads`/`n_groups` to per-rank) | `attn_tp` | `attention.py:241` `out = _coll.all_sum(out, group=self.group)` — **bf16**, span `attn.all_sum` |
| `blk.ffn.group = group` | `world>1` | MoE tail |
| `blk.ffn.shared_sharded = True` | `shared_tp` | `moe.py:180` branch 1 |
| `_SHARD_HEAD`/`group is not None` | head shard | `ShardedHead` one `all_sum` (bf16) |

Forward collective sequence (`moe.py:180`):
- `shared_sharded` True → `y = y + shared_experts(xf)` **then** `_all_sum_tail(y, group)` (span `moe.all_sum`).
- `shared_sharded` False, `group` set → `_all_sum_tail(y, group)` **then** add shared.
- both are **fp32** (`y.astype(mx.float32)`); the hung collective is **bf16** → points at attention.

`collective.py` host-sync guard: `warm_guard(key)` host-syncs collectives for the first
`DSV41_SYNC_WARM_CALLS` (default 2) calls **per shape signature**; `DSV41_SYNC_COLLECTIVES=1` forces always.

**Prime suspects (to falsify with the parity table):**
1. A collective issued by one rank but not the other (rank-divergent conditional / dtype / shape / group mismatch).
2. An `all_sum` now issued where replicated-affine previously issued none, at a point that does not pair with the
   peer (e.g. attention all_sum active on one rank's path only).
3. **Collective ORDERING divergence** between the exl3 and affine paths (e.g. `_fuse_block` is OFF for affine:
   `_FUSE_GROUPS and DENSE_MODE == "exl3"` at `build_block:551` — a fused exl3 block may emit collectives at
   different points than the unfused affine block, shifting the sequence so rank pairs misalign).
4. Warm-guard key/shape coverage: an affine-specific collective shape the guard does not host-sync on the first
   forward (rank-skew → watchdog/hang), where exl3's shapes are covered.

> **Offline win still stands (G-A PASS):** affine q6 2.32× whole / 2.97× K-batched vs prod EXL3 at m=4 on TRUE
> rank-0 sharded shapes (cos 0.99975); q5 2.36×/3.06× (cos 0.99898). Source = EXL3-reconstruction (labeled).

---

## 4. R0 — COLLECTIVE-PARITY TABLE + ROOT-CAUSE HYPOTHESIS

**Verdict: the pre-registered hypothesis (collective ORDERING / PAIRING divergence) is REFUTED.** The collective
sequence is identical across exl3 and affine, and the node logs show both ranks completing collectives in
lockstep right up to the kill. The boot was killed by the **exo supervisor's silence watchdog** during a
**healthy but slow, event-free load warmup**. No rank was deadlocked.

### 4.1 Collective-parity table — first warmup forward, world=2

Path: `Dsv41Builder.load` → `prefill.load_warmup` (inside `sync_collectives()`, so every collective is
host-evaluated) → `warmup` stage `chunk512_a`: `model(ids[1,512], argmax=True, last_logit_only=True)`.
Build flags (`build_block`, both modes, `tp_on=True`): `attn_tp=True`, `shared_tp=True`,
`blk.ffn.group=group`, `blk.ffn.shared_sharded=True`, `ShardedHead` (`_SHARD_HEAD`).
`DSV41_MOE_ALLSUM_BF16` defaults **ON**, so the MoE tail is **bf16** too. (§3 above says "fp32 MoE"; that is wrong
for this build. The traceback below confirms a bf16 `_all_sum_tail`.)

| # (per forward) | site | issued by | exl3 (control) | affine6 (treatment) | dtype / shape (rank-local) |
|---|---|---|---|---|---|
| 0 (connect, pre-load) | `MlxBuilder._probe_data_path` | both ranks | yes | yes | small probe, identical |
| per layer L=0..39: 2L+1 | `attention.py:243` `attn.all_sum` | both ranks (`attn.group` set) | yes | yes | **bf16** `[1,512,5120]` |
| per layer L=0..39: 2L+2 | `moe.py:184` `_all_sum_tail` (shared_sharded branch: shared added BEFORE) | both ranks | yes | yes | **bf16** `[512,5120]` → fp32 |
| 81 | `ShardedHead.combine_argmax` (`exl3_build.py`, ShardedHead is EXL3 in both modes) | both ranks | yes | yes | fp32 `[2,1,2]` |
| — | `_fuse_block` (exl3-only) | — | issues **no** collective (row≤16 GEMM fusion only) | not built | — |
| — | `AffineProj` / `_Grouped` | — | — | issues **no** collective | — |
| later stages | `chunk512_b`, `chunk128`, `decode1`, then the spec-verify `[1,4]` ×2 (+ DSpark `mtp.py:194` per stage) | both ranks | same sequence | same sequence | same shapes as exl3 |

**Parity: identical.** The projection type is never an input to any collective call. Each sharded group produces
the same output shape on both ranks: `wq_b` is out-sliced, `wo_b` and `w2` are in-sliced and then partial-summed,
and `w1`/`w3` are out-sliced. `_block_bounds` is shared by the exl3 and affine slicers. No collective is
conditional on rank, data, or dense format. Pinned statically by
`tests/test_dsv41_q1b_warmup_liveness.py::test_tp_collective_sequence_is_independent_of_dense_format`, which
traces `_coll.all_sum` (site, dtype, shape) through a real TP-wired 4-layer model with EXL3-style vs `AffineProj`
projections and asserts the traces are equal. `test_fused_exl3_helpers_issue_no_collectives` source-checks the fused
helpers.

### 4.2 What the node logs actually show (NEW evidence this round — `raw/pricing/q1/q1b/node_log_excerpts.txt`)

`~/exo.log.prev` on both nodes (affine6 boot) and `~/exo.log` (exl3 production restore) were read with
read-only ssh.

1. **Both ranks keep completing collectives.** After `hier geometry`, both ranks print
   `[Event::wait] slow wait: elapsed=3.0s … (polling; self-abort at 20000ms)` **at the same millisecond**
   every 4-7 s (e.g. 12:56:58.855/.856, 12:57:04.122/.123, …, 12:57:51.445/.446), right up to the kill.
   The line is logged once per `Event::wait` call, so each line is a **new** wait, which means the previous
   one completed. A host-synced warmup does one collective+eval per layer half, so this is the per-layer
   rhythm of a forward that is making progress.
2. **No wait ever wedged.** `MLX_EVENT_WAIT_TIMEOUT_MS=20000` (start_cluster default), and
   `Event::wait` throws `Timed out` after 20 s on a genuinely stuck peer. Count of `Timed out` in both affine
   logs: **0**.
3. **The supervisor killed on silence, not on a hang signature.** For each cycle: `silent for 45-50s; liveness
   probe baseline footprint=110.10GB (114.60 rank0), extending 20s`, then `silent for 66-71s verdict=kill
   stack_class=gpu spin=True growth_gb=+0.00 at_ceiling=False`, then SIGKILL. The footprint is flat by
   construction during a compile-heavy forward, since the weights are already resident.
4. **Survivor traceback (rank0, after its peer was SIGKILLed):** `builder.load → load_warmup →
   warmup:413 (chunk512_a) → model.py:148 _fused_call → moe.py:184 _all_sum_tail → collective.all_sum →
   mx.eval → RuntimeError: [jaccl] Recv failed: peer closed connection (EOF)`. This is the **bf16 MoE tail** of
   some layer in the **first warmup forward**. The rank was blocked only because its peer had been killed.
5. **The control has the SAME shape and barely survives.** The exl3 production restore (13:08) prints the
   **identical** synchronized slow-wait pattern (9 lines, 13:09:24 → 13:09:55). The supervisor's probe fired at
   `silent for 49s` (13:10:02), and warmup finished at 13:10:04.7 with `chunk512_a=48.2 s, total 50.9 s`. That
   is about 52.7 s of silence against a ~66 s kill budget, so **exl3 runs ~13 s from the same cliff**.
6. **The hang-stack "rank asymmetry" is an artifact of sampling a host-synced forward.**
   `sync_collectives` evals every collective, so at any instant one rank can be in
   `MeshGroup::all_sum` (polling for its peer) while the other is in `eval → Event::wait` on its own
   per-layer GPU work. In 5 of the 6 stacks, the SAME process holds both a `MeshGroup::all_sum` stream thread
   and a main thread in `Event::wait`. The 6th, `85628`, has its stream thread in `Fence::wait` with no jaccl
   frame, i.e. between collectives. That is the expected snapshot of a host-synced forward, not evidence of
   a pairing divergence.

### 4.3 Root cause (mechanism)

`Dsv41Builder.load` emits its last event (the `RunnerLoading` progress yield) when the body build finishes. It
then runs `build_draft_head` and `prefill.load_warmup` **without emitting any event**. The first warmup forward
(`chunk512_a`, 512 rows × 40 layers, host-synced collectives, cold Metal pipeline compile for every new shape)
takes **~48 s on exl3** and **>66 s on affine6**. The supervisor's `_check_hang` kills a runner after
`HANG_TIMEOUT_SECONDS=45` plus one 20 s growth probe whenever the footprint is flat and the stack is `gpu`+spin.
Affine crosses that line deterministically. exl3 does not, by about 13 s.

Why affine is slower or heavier going into the warmup (contributing factors, measured offline, NOT proven on
the cluster):

- (a) Building `AffineProj` (reconstruct EXL3 → fp16 `.T` → slice → `mx.quantize`) leaves the fp16
  intermediates in MLX's buffer cache: **~1.5 GiB residue** after one layer's rank-0 dense roster, versus
  **0** for exl3 (`raw/pricing/q1/q1b/affine_build_residue_probe.py`). This matches the higher pre-warmup
  footprint on the cluster: 110.1-114.6 GB affine vs 105.5 GB exl3, at equal post-load `active`
  (104.6 vs 104.9 GB).
- (b) Affine adds a second family of first-use Metal kernels: `affine_qmm/qmv` at bits=6, gs=64 for every new
  (M, K, N). On a laptop the cold dense-roster cost per layer at M=512 is the same in both modes (104 vs 103 ms),
  so (b) alone does not explain +20 s. **The exact source of the extra affine warmup time is an OPEN
  QUESTION.** The fix below does not depend on it.

**Why it was not caught before:** the replicated affine path (`tp_on=False`) issued no attention collective
and was never booted through this warmup at world=2. The control (exl3) passes with a margin that nobody
measured.

## 5. R1 — MINIMAL FIX

_(filled by R0/R1 worker — see below)_

## 6. EXECUTION LEDGER (declare-before-spend)

| # | phase | deploy | purpose | status |
|---|---|---|---|---|
| R0 | offline | — | parity table + hypothesis + pre-reg | IN PROGRESS |
| R1 | offline | — | minimal fix + unit test (exl3 byte-identical) | PENDING |
| R2 | boot #1 | fixed build | smoke-first → same-boot control re-anchor + treatment A/B (salt q1b) + deep path | HELD |
| R3 | same boot | treatment | R8a battery + logit drift | HELD |
| R4 | boot #2/#3 | affine6 default | ship (or affine5 retry) | HELD |
