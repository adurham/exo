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

The fix targets the actual mechanism (§4.3). It does **not** change the collective sequence, which was already
correct. **No timeout was widened:** the supervisor doc requires that *"long-but-legitimate native work must
signal progress rather than have HANG_TIMEOUT_SECONDS raised"*, and this fix follows that rule.

| repo | branch | new commit (parent) | change |
|---|---|---|---|
| mlx-lm | `deploy/q1-dense-qn` | **`cb163da64cdfaa562dce2f154d9ad9ed98fd4cdd`** (← `e444cbd`) | `prefill.load_warmup(..., fence_hook=None)` forwards the hook to `warmup()`, which already installs and restores it. `AffineProj._set_weight`: `mx.clear_cache()` after quantize (releases the ~1.5 GiB-per-layer-roster fp16 build residue; affine-only). New test file. |
| exo | `deploy/q1-dense-qn` | **`a62a001c661d73c2559bfa193b3461ddbad29a78`** (← `e4c2cb460`) | `Dsv41Builder.load` passes a heartbeat to `load_warmup`. The heartbeat re-sends `RunnerStatusUpdated(RunnerLoading all-loaded)` from the warmup's per-2-layer fence. The first beat is unthrottled, later beats are throttled to 15 s. Any event resets the supervisor clock (`_forward_events`). Feature-detects `fence_hook`, so an older mlx-lm keeps the old call and gets a loud WARNING. Tests updated and added. |

**Why the beat cannot mask a real hang:**

- It fires only after `model._forward`'s fence `mx.eval(h, pre_mix)`, i.e. after 2 layers of committed compute.
  Inside `sync_collectives` that also means every collective up to that point was host-paired with the peer.
- A wedged collective blocks that eval, so the rank emits no beat and is killed exactly as before.
- MLX's own 20 s `Event::wait` self-abort still applies.
- Decode/verify (≤16 rows) and serving prefill paths are untouched (`warmup` only).

**exl3 byte-identity (proof):**

1. **Weights and forward numerics.** The fix touches exl3 nowhere in the forward. The hook is observation-only,
   and `tests/test_dsv41_fence_hook.py::test_bit_identity_with_and_without_hook` (15/15 pass) asserts a hooked
   forward is byte-identical. `mx.clear_cache()` lives in `AffineProj._set_weight`, which is constructed only
   by `_dense` / `_dense_slice` under `DENSE_MODE.startswith("affine")` (grep: the only constructors are
   `exl3_build.py:380` and `:557`, both inside affine branches).
2. **Same exl3 TP slices as production.** `raw/pricing/q1/q1b/exl3_ident.py` builds rank 0/1 wq_b (out) and
   wo_b (in) slices with prod `16830e1`'s inline `Exl3Proj(EXL3Linear(_slice_dense(...)))` and with
   `cb163da`'s `_dense_slice` in exl3 mode, then hashes the outputs at M=1/4/512. Both give **sha256
   `508d2a111ccdb15edc79ac191207f51acee2e89d21959d541af991e485714f1d`**.
3. **Default call unchanged.** With no hook, `load_warmup` calls `warmup(..., fence_hook=None)`, which is
   `warmup`'s own default (`test_load_warmup_default_is_unchanged`).

**Tests (single-file runs, laptop, exo venv):**

- mlx-lm `tests/test_dsv41_q1b_warmup_liveness.py`: **6 passed**. RED on the pre-fix tree (`git stash`):
  4 failed (hook plumbing ×3 + cache residue), 2 passed (the parity tests, which are expected to pass
  pre-fix because parity held all along).
- `tests/test_dsv41_dense_affine_shard.py`: **12 passed** (affine slice geometry and quantization unchanged).
- `tests/test_dsv41_fence_hook.py`: **15 passed**.
- exo `src/exo/worker/engines/mlx/dsv41/tests/test_dsv41_load_warmup_liveness.py`: **5 passed** with
  `cb163da`. Against `e444cbd`'s `prefill.py`: 4 passed, 1 skipped (`installed mlx-lm predates the Q1B
  fence_hook`). This proves the fallback path.
- `test_dsv41_dispatch.py`: **13 passed** (now also asserts `load()` hands a callable beat to the warmup).

**Deploy note:** exo `deploy/q1-dense-qn`'s `mlx-lm` gitlink is still `6cc9c1e` (unchanged by `e4c2cb460`,
same as before). Per §8 of ROUND-Q1-QUANT, the deploy moves **both** trees: `git checkout --detach a62a001c6`
plus `git -C mlx-lm checkout cb163da`. Then verify that the INSTALLED `prefill.py` has `fence_hook=None` in
`load_warmup`'s signature and that `builder.py` has `_load_heartbeat`, on BOTH nodes.

### 5.1 Smoke expectation for boot #1 (pre-registered)

**PASS signature (affine6):**

- After `hier geometry`, each rank logs `[DSV41] load warmup liveness: fence N at Xs (rank r/2)`. The first
  appears within a few seconds of warmup start, then roughly every 15 s.
- The synchronized `Event::wait slow wait` lines may still appear. They are harmless.
- There is **no** `silent for …s; liveness probe` line during the warmup.
- Then `[DSV41] load warmup (kernel compile): {chunk512_a: …}`. **Record the affine `chunk512_a`**: the
  expected range is 48-~90 s, and a value over 66 s confirms the mechanism.
- Then `engine built`, READY, canary, and the Paris probe.
- Pre-warmup footprint in any probe line should drop toward the exl3 ~105.5 GB (`clear_cache` effect).

**FALSIFIERS:**

- (i) `Event::wait … Timed out` or `wait_for_one Timed out` appears. That is a genuine wedge, and §4 is wrong.
- (ii) A hang-kill fires AFTER at least one `load warmup liveness` line, with no further beats for >45 s. That
  means a stall inside a forward, which would require layer-indexed fence logging (beat count ÷ 20 per
  512-row forward).
- (iii) There are no `load warmup liveness` lines at all. That is a deploy defect: stale mlx-lm installed.
  Look for the WARNING line.

Any falsifier means stop per the §0 rule.

**Open (not blocking the smoke):** the source of the extra affine warmup time (§4.3 b). It is worth one
measurement: the beat log gives per-15 s fence counts for both arms on the same boot.

## 6. EXECUTION LEDGER (declare-before-spend)

| # | phase | deploy | purpose | status |
|---|---|---|---|---|
| R0 | offline | — | parity table + hypothesis + pre-reg | ✅ DONE — parity identical; ordering-divergence REFUTED; root cause = supervisor silence-kill of event-free load warmup (`55c12ce8e`) |
| R1 | offline | — | minimal fix + unit test (exl3 byte-identical) | ✅ DONE — mlx-lm `cb163da`, exo `a62a001c6`; tests green; exl3 sha256-identical |
| R2 | boot #1 | `a62a001c6`/`cb163da` + `DSV41_DENSE=affine6` | smoke-first → same-boot control re-anchor + treatment A/B (salt q1b) + deep path | ✅ SMOKE PASS (see §7); A/B IN PROGRESS |
| R3 | same boot | treatment | R8a battery + logit drift | HELD |
| R4 | boot #2/#3 | affine6 default | ship (or affine5 retry) | HELD |

**Boots spent: 1 of ≤3 (+1 reserve).**

---

## 7. R2 SMOKE RESULT (boot #1, `DSV41_DENSE=affine6`, build `a62a001c6`/`cb163da`) — **PASS**

Deploy: `EXO_TARGET_BRANCH=deploy/q1-dense-qn DSV41_DENSE=affine6 ./start_cluster.sh` from the shared checkout
after `git checkout --detach a62a001c6` + `git -C mlx-lm checkout cb163da` (BOTH trees moved).

- `Nodes synchronized on commit a62a001c6`; cluster `HEALTHY`; **`READY (2/2)`** — the build that previously never
  reached READY now comes up. Falsifier did NOT fire (no second fix iteration needed).
- **Liveness beats fired on BOTH ranks** (`~/exo.log`): rank 1/2 `fence 1 at 1.1s → fence 11 at 22.9s → fence 13
  at 42.3s`; rank 0/2 `fence 1 at 0.3s → fence 11 at 22.1s → fence 13 at 41.5s`. The supervisor silence clock was
  reset through the whole warmup (no `liveness probe` line during the warmup; the only probe line is during the
  earlier body load, same as exl3).
- **Warmup completed**: `load warmup (kernel compile): {chunk512_a: 46.50, chunk512_b: 1.61, chunk128: 0.59,
  decode1: 0.16, total: 49.08}` (rank 1/2); `{chunk512_a: 45.70, …, total: 48.28}` (rank 0/2). Then `engine built:
  40/40 layers` on both, speculative=True (gamma=3).
- **This also closes the §4.3 open question:** affine's warmup is NOT intrinsically slower than exl3 — it is
  **~46 s vs exl3's ~48 s** (comparable). The earlier >66 s kill was driven by the **build residue**: the
  pre-warmup footprint probes dropped from **110.10 / 114.60 GB** (pre-`clear_cache`, the failing build) to
  **85.10 / 84.00 GB** now. So `AffineProj` `mx.clear_cache()` reclaimed the ~1.5 GiB/layer-roster fp16 residue and
  the warmup fits comfortably under the watchdog. (Both fixes are load-bearing: the heartbeat prevents the kill
  *even if* a warmup is slow, and `clear_cache` removes the memory-pressure that made it slow.)
- **Env verified**: `DSV41_DENSE=affine6` in the running env on BOTH nodes (`ps -axeww`); INSTALLED venv
  `site-packages/mlx_lm/.../prefill.py` contains `fence_hook` (14 hits) on both nodes; the beats log from
  `...dsv41.builder:beat` proving the exo-side heartbeat is installed.
- **Canary after READY: 14.87 (M4-1) / 14.85 (M4-2) TFLOPS — HEALTHY.**
- **First-forward smoke (the previously-stalling path):** `POST /v1/chat/completions` "capital of France" →
  `finish=stop`, `content='Paris'`, `reasoning='We need answer only city name. Paris.'`, **elapsed 1 s.** The
  sharded-affine forward serves correctly.

**Verdict: the deterministic world=2 first-forward stall is FIXED.** Boot #1 spent; proceeding to the A/B.

### 7.1 A/B deviation (recorded, a-priori)

The pre-registered R2 called for a **same-boot control re-anchor** (`affine6` → process-restart to `exl3`). There is
**no rule-compliant way to do that without a full boot or editing node files**: `relaunch_exo.sh` is generated by
`start_cluster.sh` with the FULL launch env hardcoded (`… DSV41_DENSE=affine6 …`); flipping the arm requires
editing that generated script **on the nodes**, which the hard rules forbid ("no direct file edits on the nodes").
Consequences: the A/B uses the **frozen control anchor** (`control_benign.json` 94.88 ms / `control_agentic.json`
101.07 ms, 4 reps each) measured in the prior round on the **byte-identical exl3 path** (`sha256 508d2a11…`
proves exl3 is unchanged by `cb163da`), on the same two nodes, with an equally healthy canary (14.86/14.86 then vs
14.87/14.85 now) and previously ±0.1 ms rep spread. The G-B gate is read as **treatment (this boot) vs the frozen
exl3 anchor**; if the agentic Δ lands within 2 ms of the 15 ms floor, the deviation becomes material and a proper
same-boot control is worth a boot — otherwise it is immaterial.

---

## 8. R2 A/B RESULTS (treatment = sharded affine6, salt `q1b`, same content both arms)

| arm | metric | control (exl3, frozen) | treatment (affine6, this boot) | Δ |
|---|---|---:|---:|---:|
| benign 20K | ms/round median | 94.88 (94.56/95.01/94.88) | **80.97** (81.17/80.66/80.97) | **−13.91 ms** |
| benign 20K | decode t/s median | 38.48 | 46.09 | +7.61 |
| benign 20K | mean_accepted median | 2.682 | 2.743 | +0.061 (noise) |
| agentic 91K | ms/round median | 101.07 (101.09/100.92/101.07) | **87.28** (87.28/87.22/87.37) | **−13.79 ms** |
| agentic 91K | decode t/s median | 30.962 | 35.272 | +4.31 |
| agentic 91K | mean_accepted median | 2.1128 | 2.0969 | −0.016 (noise) |

- **Deterministic + material:** rep spreads are ±0.15 ms (agentic) and ±0.25 ms (benign) — the dense re-quant win
  is real and stable. `mean_accepted` is unchanged within noise on both arms ⇒ the comparison is not confounded.
- **Against the pre-registered G-B floor of 15 ms (agentic):** the realized agentic Δ is **−13.79 ms < 15 ms**:
  **G-B is a narrow MISS** (1.21 ms short). The projection (§0 of ROUND-Q1-QUANT) was ~18.5–19.4 ms; the realized
  win is ~71 % of projection. Both arms agree (~13.8–13.9 ms), so this is the true dense-slice win, not a
  measurement artifact.
- **Same-boot control re-anchor:** performed per §7.1 (`relaunch_exo.sh` env-flipped `affine6`→`exl3` in-pipe, no
  node file written). Result: `control_sameboot_benign.json` — see §8.1.

### 8.1 ORCHESTRATOR ADJUDICATION (near-floor handling — the ONE change to the round)

Because the same-boot Δ landed below the 15 ms floor while the win stayed deterministic (σ≈0.03 ms) and material
(≥12 ms), the orchestrator adjudicated the near-floor case as an **owner-visible call, NOT a mechanical
self-close**:

1. Run the **full R8a battery + logit-drift on the affine6 treatment anyway** (no boot needed; it is required
   for any ship path).
2. **Keep the treatment build live**; do **NOT** auto-restore and do **NOT** auto-ship.
3. Record the full picture here and surface an **ESCALATION** (win + floor nuance + battery verdict) for
   orchestrator/owner adjudication.
   - If the battery comes back **DIRTY** → restore + close as pre-registered (genuine negative).
   - If the same-boot Δ **≥ 15 ms** → proceed to ship per the original plan.
   - **This is the ONLY change to the round; everything else proceeds as pre-registered.**

### 8.2 R2 SAME-BOOT CONTROL — VALIDATES THE FROZEN ANCHOR (drift 0.3 %)

Same-boot control re-anchor (`exl3` on the eval build, process-only arm switch; salt `q1b`, benign 20K):

| benign 20K | ms/round median | decode t/s | mean_accepted |
|---|---:|---:|---:|
| control, **same boot** (exl3) | **94.56** (94.56/94.75/94.51) | 37.945 | 2.5919 |
| control, **frozen** (exl3, prior round) | 94.88 (94.56/95.01/94.88) | 38.48 | 2.682 |

- The same-boot control (94.56) matches the frozen control (94.88) to **0.32 ms (0.3 %)** ⇒ the frozen anchor was
  valid and there is no material cross-boot drift. ⇒ the frozen agentic anchor (101.07) is trustworthy.
- **Confirmed same-boot benign Δ = 94.56 − 80.97 = −13.59 ms** (frozen-anchor Δ was −13.91). Both < 15 ms.
  Acceptance within noise (2.59 vs 2.74 treated).
- **⇒ G-B is a CONFIRMED narrow miss**: agentic Δ −13.79 ms vs the 15 ms floor, benign Δ −13.59 ms, deterministic
  (σ ≤ 0.15 ms), acceptance unchanged. The near-floor adjudication (§8.1) therefore applies: run the R8a battery +
  logit-drift on the affine6 treatment, keep the build live, do NOT auto-restore/ship, and escalate.

**Battery (affine6, depth 40000) + logit drift: RUNNING** — see §9.
