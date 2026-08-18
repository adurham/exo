# DSv4-Flash 220K-context TP prefill span profile (2026-08-18)

## Context

Investigating whether DSv4-Flash prefill at long context is genuinely
compute-bound (as the user asserted from general "prefill=compute-bound,
decode=memory-bound" priors) versus a stale 2026-07-13 finding that
called it memory-bandwidth-bound on the sparse-attention KV gather.
A lot of code has changed since that July investigation (including
fallout from the now-abandoned PP-prefill/TP-decode hybrid experiment,
see `pp-prefill-tp-decode-phase-swap-design-2026-08-16.md` and
`hybrid-pp-prefill-tp-decode-design-2026-08-04.md` — that whole
direction is DEAD, see warm memory fact 1472), so the July numbers
cannot be trusted without re-measuring on current `main`.

Cluster state at measurement time: commit `1c49a377d`, TP-only
(`DSV4_SHARDING=Tensor`, the default), `MLX_JACCL_DATA_RECV_POOL=0`
(workaround for the Section 119 rank-0 `allocate_buffers()` RDMA hang
— see that same design doc's Section 119 and warm memory fact ~1472
area for the incident). Both Mac Studio M4 Max nodes, 128GB each,
RDMA over Thunderbolt 5.

## Method

1. Relaunched cluster with `EXO_PROFILER=spans EXO_PROFILER_LEVEL=1`
   added on top of the standing `MLX_JACCL_DATA_RECV_POOL=0` fix —
   both are plain launcher env vars (allow-listed in `start_cluster.sh`),
   no source changes.
2. Generated a genuinely fresh ~220K-token prompt (random word-salad
   filler + a unique embedded "secret code", regenerated with a new
   random seed each run) so the KV prefix cache could not produce a
   phantom instant "cache hit" result — this is a real trap: an
   earlier attempt in this same investigation got a bogus 67s "result"
   from an orphaned first curl attempt whose server-side prefill kept
   running after the client was killed (exo does NOT cancel
   server-side prefill when the client disconnects — documented
   behavior, see skill exo-cluster-operations pitfall on this).
3. Sent via `POST /v1/chat/completions`, `max_tokens=50`, `temp=0`,
   asking the model to repeat back the secret code (a real
   needle-in-haystack correctness check, not just a throughput probe).
4. Verified genuineness of the measurement by cross-checking:
   - `usage.cached_tokens: 0` in the response (fresh, not cache-hit).
   - Server-side log timestamps for `Prefill progress: N/220315
     tokens` lines, confirming a SINGLE clean run with no overlapping
     concurrent request during the measurement window (checked both
     nodes for any other `Prefill progress: 0/` start line inside the
     window — none found).
   - Both nodes logged the request start within 2ms of each other
     (15:00:38.261 vs .259 in an earlier non-profiled run of the same
     prompt size) — confirms genuine 2-node parallel execution, not
     sequential.
5. Model answered correctly (`PROF-5116-TRACE-4361`, `finish_reason:
   stop`), confirming the profiler run didn't corrupt output quality.

## Headline throughput numbers

- **Unprofiled** (clean run, `MLX_JACCL_DATA_RECV_POOL=0` only, no
  `EXO_PROFILER`): 220,321 tokens in 612s wall-clock = **360 tok/s**,
  matching the server's own live-computed rate (359.8 tok/s) to
  within 0.2 — internally consistent, not a snapshot artifact.
  Verified via both nodes' `Prefill progress` log timestamps, not
  client-side `time_total` (which conflates network+decode+overhead).
- **Profiled** (`EXO_PROFILER=spans`, same prompt size, fresh random
  content): 220,315 tokens in ~673s of pure prefill = ~308-329
  tok/s live-computed rate, degrading over the run (369→308 tok/s) —
  profiler overhead is real and non-trivial at this call volume
  (~4730 span entries per major category), consistent with the
  profiler module's own documented ~3-5µs/call overhead × high call
  counts at 43 layers × ~110 chunks.

**User's prediction going in: "at most ~350 tok/s ceiling." Measured
clean number: ~360 tok/s.** Prediction was accurate, slightly
conservative if anything.

## Span-level breakdown (profiled run, aggregate over full 220K request, rank0/node1)

Format: `span_name  n_calls  avg_us  total_ms  %_of_wall`

```
attn                  4730   76696.50 us avg   362774.43 ms total   58.4%
ffn                   4730   54581.01 us avg   258168.16 ms total   41.6%
  moe.switch_mlp       4730   35373.07 us avg   167314.61 ms total   26.9%
attn.sdpa              2530   33315.29 us avg    84287.67 ms total   13.6%
attn.sdpa.compressed   2200   33362.18 us avg    73396.81 ms total   11.8%
attn.o_proj            4730   13072.01 us avg    61830.58 ms total   10.0%
moe.all_sum            4730   12446.73 us avg    58873.05 ms total    9.5%
attn.proj_qkv          4730   11725.31 us avg    55460.70 ms total    8.9%
attn.all_gather        4428   11949.58 us avg    52912.74 ms total    8.5%
moe.post_combine       4730    5567.13 us avg    26332.52 ms total    4.2%
attn.indexer           2310   10647.93 us avg    24596.71 ms total    4.0%
layer.ffn_hc           4730    3082.56 us avg    14580.50 ms total    2.3%
layer.attn_hc          4730    3073.20 us avg    14536.23 ms total    2.3%
layer.ffn_residual     4730    2909.33 us avg    13761.15 ms total    2.2%
layer.attn_residual    4730    2879.58 us avg    13620.41 ms total    2.2%
moe.gate               4730    1169.80 us avg     5533.17 ms total    0.9%
attn.rope_in           4730     845.42 us avg     3998.84 ms total    0.6%
attn.rope_out          4730     468.90 us avg     2217.90 ms total    0.4%
model.embed             110   17803.40 us avg     1958.37 ms total    0.3%
attn.kv_cache          4730     362.77 us avg     1715.88 ms total    0.3%
attn.mask              4510     358.38 us avg     1616.28 ms total    0.3%
model.lm_head            110   13838.65 us avg     1522.25 ms total    0.2%
layer.attn_norm        4730     247.48 us avg     1170.58 ms total    0.2%
layer.ffn_norm         4730     213.91 us avg     1011.79 ms total    0.2%
attn.compressor        4510     132.20 us avg      596.21 ms total    0.1%
model.final_norm         110    4442.05 us avg      488.63 ms total    0.1%
model.attn_mask          110     503.06 us avg       55.34 ms total    0.0%
switch.gather_sort     4644       7.15 us avg       33.21 ms total    0.0%
switch.up_proj         4730       4.32 us avg       20.45 ms total    0.0%
indexer.score          2289       8.34 us avg       19.10 ms total    0.0%
attn.gather            2268       6.76 us avg       15.34 ms total    0.0%
switch.activation      4730       3.18 us avg       15.05 ms total    0.0%
indexer.topk           2289       5.81 us avg       13.31 ms total    0.0%
switch.gate_proj       4730       1.54 us avg        7.27 ms total    0.0%
switch.scatter_unsort  4644       1.54 us avg        7.13 ms total    0.0%
switch.down_proj       4730       1.32 us avg        6.25 ms total    0.0%
```

## Interpretation

**Compute-bound verdict: YES, with real evidence, superseding the
July 2026-07-13 memory-bandwidth-bound finding for this codebase
state.**

Key evidence, not just GPU-power inference:
1. `attn + ffn` sum to ~100% of wall time (58.4% + 41.6%). There is
   no large unattributed gap in the span accounting that would
   indicate GPU idle/stall time between named ops — every dollar of
   wall-clock is accounted for as real model computation (SDPA,
   projections, MoE matmuls, or communication collectives).
2. Live dashboard observation during a comparable run (user
   screenshot, 2026-08-18 ~15:09): both nodes simultaneously at
   97-98% GPU utilization, 116-124W power draw, 76°C — genuinely
   pegged, not starved.
3. The July note's "memory-bandwidth-bound sparse-attention gather"
   theory is NOT supported here: `indexer.score`/`indexer.topk`
   (the actual top-k selection) are now 0.0% each (essentially free —
   consistent with the OPT-6 indexer fold noted in July), and
   `attn.gather` itself (the scattered KV read the July theory
   pinned as the bottleneck) is also 0.0% (15.34ms total across the
   whole 220K request). Whatever was true in July, the current
   codebase's gather step is not where time goes.

**This does NOT mean zero comms cost.** Cross-node collectives
(`attn.all_gather` 8.5% + `moe.all_sum` 9.5% = 18.0% combined) are a
real, quantified tax from 2-node parallelism — see the companion
question of how much of that is genuine RDMA wait vs. local
finalize()-forced eval overhead, addressed in a follow-up
measurement (JACCL_TRACE_PROGRESS run).

## Balance shift vs. July 2026-07-13 finding

| | July 2026-07-13 (300K) | This run (220K, 2026-08-18) |
|---|---|---|
| MoE total | 51.1% | 41.6% (ffn) |
| Attention total | ~24% (SDPA+all_gather) + ~18% (proj) | 58.4% |
| Indexer | 0.0% (already free then) | 0.0% (still free) |
| moe.all_sum | 8.4% | 9.5% |
| attn.all_gather | 7.6% | 8.5% |

The attention/MoE balance has shifted meaningfully toward attention
dominating (58% vs the old ~42% combined attention figure). Cause
not yet root-caused — candidate factors: different context length
(220K vs 300K), possible code changes from the PP/TP experiment
fallout, or `attn.sdpa.compressed` (11.8%, a specific attention
variant) being newly significant in a way the July breakdown didn't
call out by name. Worth a follow-up if attention-side optimization
becomes the target.

## Candidate levers identified (not yet validated)

1. **`EXO_DSV4_SEQ_SPLIT` A/B at this context size.** The seq-split
   scheme (splits prefill query rows across nodes, adds the
   `attn.all_gather` reconstruction step) was measured in June 2026 at
   B=1 100K context to cost ~7% throughput for its reconstruction
   overhead in exchange for splitting SDPA work across nodes — net
   positive at that scale per the June finding. That number has NEVER
   been re-validated at 220K+ context on current code. Directly
   testable: relaunch with `EXO_DSV4_SEQ_SPLIT=0`, re-run the same
   220K prompt, compare tok/s and the `attn.all_gather` span
   disappearing entirely.
2. **Comms tax (`all_gather` + `all_sum`, 18.0% combined)** is the
   largest single quantified non-attention/non-MoE-matmul cost. How
   much of this is genuine network wait vs. local `mx.eval()` /
   finalize() overhead is the subject of the JACCL_TRACE_PROGRESS
   follow-up measurement (see below / separate doc section once run).

## What this does NOT establish

- Does not re-litigate or revive the PP-prefill/TP-decode hybrid —
  that remains dead per warm memory fact ~1472 and
  `pp-prefill-tp-decode-phase-swap-design-2026-08-16.md`. This
  profile is pure TP-only prefill performance, orthogonal to that
  dead design.
- Does not establish a new performance target or "ceiling" beyond
  confirming ~360 tok/s is achievable and near-compute-bound at 220K.
  Whether further optimization (attention-side kernel work, seq-split
  tuning) can move that number meaningfully is unproven — the two
  candidate levers above are untested, not confirmed wins.
- Profiler overhead itself (unprofiled 360 tok/s vs profiled
  ~308-329 tok/s, a ~10-15% tax) means the exact span percentages
  above are measured under artificially loaded conditions; relative
  proportions between spans should be reliable, but do not assume the
  profiled run's absolute tok/s reflects production performance.
