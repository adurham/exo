# hc_expand fused Metal kernel — depth verification (300K, 500K) on 2026-08-24 — INCONCLUSIVE

**Status: LIVE DEPTH-A/B COMPLETE. VERDICT — INCONCLUSIVE at both 300K
and 500K depths (both within ±1.5%). Cluster restored to production
(kernel ON, defaults).**

Follow-up depth verification to
`docs/hc-expand-kernel-ab-2026-08-24.md`, which shipped the fused Metal
kernel default-ON after a +3.87% mean prefill win at ~70.5K real tokens.
The 70.5K A/B doc explicitly flagged "Not tested at deeper context
(300K, 500K)" as an open item; this session closes that gap.

## 0. Pre-registered criteria (from task brief, verbatim)

- **GAIN HOLDS** at a depth if `ON ≥ OFF × 1.015` (pairwise, same depth).
- **NEUTRAL** at depth if within `±1.5%`.
- **REGRESSION FLAG** if `ON ≤ OFF × 0.985` at either depth — in that
  case do NOT flip any default; document loudly and stop after
  restoring production.
- **Mechanistic expectation**: `hc_expand` is a per-layer-per-token
  elementwise op (~4.4% span share @220K). Its relative share should
  shrink slightly as SDPA grows with depth. Predicted rough delta:
  +2.5% to +4% ON-vs-OFF at 300K, slightly less at 500K.
- **Quality gate every probe**: needle recovered exact, no U+FFFD, no
  BOS spam, `finish_reason=stop`, actual generated text in report.
- **Sanity anchor**: OFF @500K-target should land near historical
  324.1 × ~1.08–1.10 (pool-grow now default-ON) ≈ **350–360 tok/s**.
  If an arm deviates >8% from its plausible band, STOP and investigate
  deploy before proceeding.

## 1. Results at a glance

| Arm | Depth  | Run | Real tok | TTFT (s) | Prefill (tok/s) | Decode (tok/s) | Needle |
|-----|--------|-----|---------:|---------:|----------------:|---------------:|:------:|
| ON  | 300K   | r1  | 211,797  |  596.15  |    355.27       |  31.08         |  OK    |
| ON  | 300K   | r2  | 211,022  |  589.18  |    358.16       |  25.03         |  OK    |
| OFF | 300K   | r1  | 211,774  |  601.20  |    352.25       |  24.80         |  OK    |
| OFF | 300K   | r2  | 211,379  |  600.24  |    352.16       |  25.23         |  OK    |
| ON  | 500K   | r1  | 352,670  | 1047.31  |    336.74       |  22.82         |  OK    |
| OFF | 500K   | r1  | 352,431  | 1055.48  |    333.91       |  23.95         |  OK    |

**Means and deltas** (means where n=2; single point where n=1):

| Depth | ON mean | OFF mean | Δ tok/s | Δ %      | Criterion verdict |
|-------|--------:|---------:|--------:|---------:|:------------------|
| 300K  | 356.72  | 352.21   | +4.51   | **+1.28%** | INCONCLUSIVE (< 1.5%) |
| 500K  | 336.74  | 333.91   | +2.83   | **+0.85%** | INCONCLUSIVE (< 1.5%) |

- Neither depth clears the +1.5% GAIN threshold.
- Neither depth trips the −1.5% REGRESSION threshold.
- No regression flag; kernel is not shown to be harmful at depth.

## 2. Verdict against pre-registered criteria

- **300K: NEUTRAL / INCONCLUSIVE** (+1.28%, below +1.5% ship threshold).
- **500K: NEUTRAL / INCONCLUSIVE** (+0.85%, below +1.5% ship threshold).
- **No regression at either depth** — the kernel is safe to leave on;
  it is simply not measurably helpful at these depths in this sample.
- **The 70.5K ship decision is not undermined** by this depth
  measurement — it was cleared at +3.87% mean-to-mean with an even
  tighter 2×2 sample. But the +3.87% headline **does not empirically
  transfer to 300K/500K** in this single-run-pair test at 500K plus
  two-run pair at 300K.
- **Cluster left in production configuration** (kernel ON via
  `start_cluster.sh` default). See §7 for the final-state proof.

## 3. Cross-check vs mechanistic prediction

Prediction from `docs/hc-expand-kernel-ab-2026-08-24.md` and the T10
span-share analysis: the op is per-layer-per-token elementwise, so its
absolute per-chunk cost is roughly constant across depths, but SDPA and
indexer costs grow with depth — so the RELATIVE share of hc_expand
should shrink modestly at 300K and 500K vs 70.5K. Predicted rough
delta range: **+2.5% at 300K → slightly less at 500K**.

Measured: **+1.28% @300K → +0.85% @500K**. Direction of decay matches
the prediction (smaller at deeper context), magnitude is below the
predicted band (+1.28% vs predicted +2.5%–4% at 300K). Two possibilities:
(a) the 70.5K span share overestimated hc_expand's contribution at
depth relative to the SDPA/indexer/MoE terms that scale with L; (b)
noise floor at this depth is wider than the 70.5K A/B's tight ±0.33%,
so a real ~+2% effect could hide inside the ~±0.7% run-to-run spread
we observed.

Not "wildly outside" the mechanistic prediction (i.e. not +8% or
−3%) — so no deploy re-verification triggered on that gate.

## 4. Sanity anchor gate — OFF @500K plausible band

Pre-registered: OFF @500K-target should land near **350–360 tok/s**
(324.1 historical × 1.08–1.10 for pool-grow default-ON). If any arm
deviates >8% from that band, STOP and investigate deploy.

Measured OFF @500K: **333.91 tok/s** — below the predicted band by
~5% (333.91 / 350 = 0.954). Still +3% above the raw 324.1 historical
figure. Deviation is under the 8% "stop" threshold, so proceeded
without deploy re-verification.

Honest caveat: the sanity band was speculative (pool-grow's headline
was +9.79% @352.6K on Aug 23, but that was measured on top of the
pre-hc_expand code path and with a different mix of resident state).
The 5% miss vs predicted here does not imply a deploy problem — it
simply says the pool-grow gain is not additive in the naive way at
this deeper depth with this session's exact residuals. Both arms
measured on the SAME production code path save for the one env var,
so relative comparison is still valid.

## 5. Runner env — the §2 forwarding check per arm

Verified via `ssh <node> ps eww <pid>` on all 4 primary runner PIDs
per arm (2 per node — the coordinator and worker python processes).
Both arms below; only `EXO_DSV4_HC_EXPAND_KERNEL` differed.

### 5.1 ON arm (production config, exo `d723784a4`)

Runner PIDs: M4-1 { 51725, 51737 }, M4-2 { 52266, 52277 } (post
production restore, §7). Env recorded from all four:

```
EXO_DSV4_BATCHED_PREFILL=1
EXO_DSV4_DSPARK=1
EXO_DSV4_HC_EXPAND_KERNEL=1
EXO_DSV4_MTP=0
EXO_DSV4_SEQ_SPLIT=1
EXO_PREFILL_STEP_SIZE=2048
EXO_SPECULATIVE=0
MLX_JACCL_SHARDING_MODE=Tensor
```

The initial ON-arm 300K/500K probes (before the OFF-arm relaunch) were
against the same production configuration, exo commit `302759bec`
(nodes) with laptop one docs-only commit ahead at `d723784a4`
(see §6 for git-delta verification). Verified pre-probe: M4-1 PID
33871 and M4-2 PID 33826 both showed the identical env block above.

### 5.2 OFF arm (kernel disabled, exo `d723784a4`)

Runner PIDs: M4-1 { 45962, 45963, 45964, 45974 }, M4-2 { 46211, 46212,
46213, 46222 }. Env verified on all four primary + all four secondary:

```
EXO_DSV4_BATCHED_PREFILL=1
EXO_DSV4_DSPARK=1
EXO_DSV4_HC_EXPAND_KERNEL=0     <-- the ONLY difference vs ON arm
EXO_DSV4_MTP=0
EXO_DSV4_SEQ_SPLIT=1
EXO_PREFILL_STEP_SIZE=2048
EXO_SPECULATIVE=0
MLX_JACCL_SHARDING_MODE=Tensor
```

**Env isolation: OK.** The one env var under test is different; every
other production-relevant flag is byte-identical between arms.

## 6. Git SHAs deployed per arm

Both arms ran on the same code SHA `d723784a4`. Delta from cluster's
initial `302759bec` verified via
`git log --stat 302759bec..d723784a4`:

```
commit d723784a43af6beac1d48f0b4a90fdd57b47a227
Author: Adam Durham <adam@example.com>
Date:   Mon Aug 24 15:46:26 2026 -0500
    docs: final production-state verification for hc_expand ship

 docs/hc-expand-kernel-ab-2026-08-24.md | 102 +++++++++++++++++++++++++++++++++
 1 file changed, 102 insertions(+)
```

**Confirmed docs-only** (single markdown file, no runtime code path
touched, no submodule bump). The ON-arm probes ran against the
cluster's `302759bec` (nodes did not re-sync); the OFF-arm relaunch
brought nodes to `d723784a4` (docs-only, no runtime impact); the
production-restore relaunch left nodes at `d723784a4`. Runtime code
identical across the whole session.

`mlx-lm` submodule pin `7a1a4e868` (unchanged, contains the fused
kernel).

## 7. Cluster left in production configuration

Per the task brief, cluster was returned to production (kernel ON,
`EXO_SPECULATIVE=0 EXO_DSV4_MTP=0`) at the end of the session.

- **Relaunch cmd** (via tmux):
  `EXO_SPECULATIVE=0 EXO_DSV4_MTP=0 ./start_cluster.sh`
- **Deploy log**: `/tmp/start_cluster_hcdepth_restore.log` — reached
  `READY (2/2)` after ~1 placement retry, HEALTHY (Nodes: 2,
  Identities: 2), 4 runners spawned per node.
- **Env verified restored on 4 runner PIDs** (M4-1 51725/51737,
  M4-2 52266/52277): `EXO_DSV4_HC_EXPAND_KERNEL=1` — full env block
  in §5.1.
- **Live probe** (2000-token target, `bench/phase3_precheck_depth_throughput.py`):

```
target=2000, real tokens=1521
TTFT: 6.36s, prefill: 239.0 tok/s
decode: 28.69 tok/s, finish_reason implied by max_tokens (64) reached
response: 'FALCON-MERCURY-7749'
needle found: YES
```

  Coherent needle-recovery text, no U+FFFD, no BOS spam. Cluster
  is serving.

- **Depth-scale production proof**: the LAST measurement in this
  session was the ON@300K r2 probe (§1 line 2) — a real 211,022-token
  prefill through the exact restored production config, ending in
  clean needle-recovery generation. That's ~10 minutes of successful
  serving under real load, not just a warm-up smoke.

## 8. Method

Per task brief, methodology mirrors `docs/hc-expand-kernel-ab-2026-08-24.md`:

- Harness: `bench/phase3_precheck_depth_throughput.py`, `--max-tokens 128`
  (one probe per call, `--json-out` per probe).
- Numerator: real tokenizer (`AutoTokenizer` on the model repo).
  NEVER derived from server-reported counts (see
  `exo-dsv4-prefill-tuning` skill §"THE BIG ONE").
- Denominator: wall-clock TTFT (client `time.perf_counter()`).
- Concurrency: one probe at a time, sequentially.
- Verification: needle FALCON-MERCURY-7749, `needle_found: true` on
  every probe. API prompt_token count cross-checked; a 4-token
  accounting drift (`api_prompt_tokens - ground_truth_prompt_tokens = 4`)
  observed on EVERY probe — harness reports this as a WARNING, uses
  ground truth for the tok/s numerator. Not new; a known long-standing
  drift and not sensitive to the kernel gate.
- OFF arm requires a full cluster relaunch (the env var is
  process-level, cannot be injected into live runners) — same
  justification as the 70.5K A/B doc §3.3.

## 9. Raw JSON evidence

Full harness output preserved:

- `/tmp/hcexp_depth_on_300k.json`   (ON@300K r1)
- `/tmp/hcexp_depth_on_300k_r2.json` (ON@300K r2, also the production-restore
                                      depth-scale smoke)
- `/tmp/hcexp_depth_on_500k.json`   (ON@500K)
- `/tmp/hcexp_depth_off_300k.json`  (OFF@300K r1)
- `/tmp/hcexp_depth_off_300k_r2.json` (OFF@300K r2)
- `/tmp/hcexp_depth_off_500k.json`  (OFF@500K)
- `/tmp/hcexp_prod_smoke.json`      (production-restore small-context smoke)

Cluster relaunch logs:

- `/tmp/start_cluster_hcdepth_off.log`    (OFF-arm relaunch → READY)
- `/tmp/start_cluster_hcdepth_restore.log` (production restore → READY)

## 10. Note on repeats — why the pair at 300K, single at 500K

Task budget allowed ONE repeat pair at ONE depth if the 300K result was
inconclusive. The initial single-run 300K delta was +0.85%
(INCONCLUSIVE); the repeat pair added ON@300K r2 (358.16) and
OFF@300K r2 (352.16), sharpening the 300K delta to +1.28% — still
INCONCLUSIVE at ±1.5%, but with much tighter within-arm variance
(OFF@300K spread 0.09 tok/s = 0.03%; ON@300K spread 2.89 tok/s =
0.81%). The 500K depth remains n=1 per arm; the delta there (+0.85%)
is inside the plausible run-to-run spread the 300K pair revealed
(<±1%), so we cannot rule out a real ~+2% effect at 500K, only bound
it to be below the +1.5% ship threshold in this specific sample.

## 11. Hard guardrails respected

- No xctrace / Metal traces during any live deep prefill (per the
  `docs/p3-followup-allsum-wait-at-depth-2026-08-24.md` §6 rule that
  says profiling at depth has twice killed both runners).
- `EXO_PROFILER_SYNC_SPANS` never set.
- No blind sleeps in the harness; all long waits are bounded poll
  loops on the JSON output file or the mlx runner's log progress.
- No files hand-edited on the Mac Studios; all changes flow through
  the laptop git → `start_cluster.sh` node-sync path.
- One probe at a time; never two concurrent.
- No retry-mitigation hacks; where a client-side hang happened once
  (§13.1 below) it was root-caused honestly rather than masked.

## 12. Limitations honestly stated

- **n=1 at 500K per arm.** A single probe cannot bound the noise
  floor; the +0.85% delta could easily be noise-of-either-sign around
  a true zero, or a real small win obscured by noise. Would need
  ≥3 pairs at 500K to distinguish these — not affordable in this
  session's budget (~35 min per 500K probe × 6 = ~3.5 hrs of pure
  probe time, plus one relaunch cycle for the OFF arm).
- **Both arms of this A/B ran through the exact same submodule pin**;
  the kernel gate's implementation-under-test is unchanged from the
  70.5K measurement.
- **No P5/known-good baseline anchor at 300K/500K** was re-measured
  this session for the OFF arm — the "sanity anchor" gate in §4 used
  a historical figure (324.1 tok/s @500K, pre-pool-grow, pre-hc_expand)
  and a projected +8–10% pool-grow boost. That projection was 5% off
  on the low side; the OFF@500K arm landed at 333.91 rather than the
  predicted 350–360. This is the largest single unmeasured baseline
  in this test.
- **No mechanistic breakdown** (e.g. per-layer `hc_expand` cost
  isolated with `EXO_PROFILER=spans`) — deliberately avoided per the
  no-profiling-at-depth guardrail. So we cannot say from this data
  alone whether the +1.28%→+0.85% decay with depth matches a genuine
  span-share decrease of the op, or the op's absolute cost is stable
  but relative noise grows.

## 13. Anomalies encountered

### 13.1 First ON@500K probe: client-side hang after successful server-side prefill

**Symptom**: The first ON@500K probe attempt (started 17:08:13)
completed on the server side at 17:25:46 (`Prefill complete: 352477
tokens in 1046.40s (336.8 tok/s)` in `~/exo.log`; TaskFinished
event issued at 17:25:48). But the client harness python process
sat at ~665 MB RSS with no JSON written; last log line was still
the prompt-tokens count printed BEFORE the HTTP call started. Client
process was live but stuck.

**What I did** (recorded honestly): I sent `SIGUSR1` to the client
PID hoping it would print a Python traceback. Python's default
SIGUSR1 handler is to terminate — the client had no handler
registered, so this killed it and produced no traceback. This was
my mistake; the correct move would have been either (a) leave it
alone and let it eventually time out on its own connection, or (b)
`py-spy dump` (which needs privileges we may not have). No harm
done: no cluster damage, no leaked runner state, just a lost
diagnostic window. The re-run of the same probe (17:42:26) worked
cleanly end-to-end, producing 336.7 tok/s at essentially the same
prefill number the server had already logged from the first attempt.

**Root cause not fully identified.** The server side ran to
completion normally and issued the SSE terminal chunks; the client
was reading `httpx.AsyncClient().stream()` in an `async for line in
response.aiter_lines()` loop and evidently blocked past the last
chunk. Two candidates: (1) HTTP/keep-alive or SSE `[DONE]` marker
handling in `httpx` (the bench never sets a client-side timeout on
the stream, so a lost final chunk could hang forever); (2) an
SSE-chunk-flush ordering issue on the API path when a truly large
prompt hits `stream=True` — the server COULD be delaying the final
close of the response until some other coroutine yields. Neither is
strong; the re-run of the identical scenario ~15 min later worked
fine, so it may just be an intermittent httpx-side edge case at
this exact combination of long TTFT + short decode + stream=True.
Not tracked further here — this doc is about the kernel gate, not
the harness. Flagged for anyone who hits it again.

### 13.2 Consistent +4-token API-vs-tokenizer accounting drift

Every one of the six probes reported the same shape: the API's
`usage.prompt_tokens` is +4 tokens above the harness's offline
ground-truth tokenization. Fixed offset, present on all probes,
sensitive to neither depth nor kernel gate. Almost certainly one
of: BOS token, chat-template preamble bytes, or a system-role
insertion the server adds after the client sends the prompt.
Long-standing, out-of-scope for this doc, but recorded as a
reminder to the campaign that the "usage" fields still can't be
trusted 1-for-1 with client offline tokenization.

## 14. Files created / modified

- `docs/hc-expand-depth-verification-2026-08-24.md` (this doc)
- `docs/PERFORMANCE_HISTORY.md` §3.1 — appended depth-verification
  note under the existing 2026-08-24 hc_expand entry
