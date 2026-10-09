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

_(filled by R0/R1 worker — see below)_

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
