# hc_collapse fused precursor kernel — depth verification (300K, 500K) on 2026-08-25 — GAIN HOLDS

**Status: LIVE DEPTH-A/B COMPLETE. VERDICT — GAIN HOLDS at both 300K
(+1.97%) and 500K (+1.89%) depths. Cluster restored to production
(kernel ON, defaults).**

Depth follow-up to `docs/hc-collapse-kernel-ab-2026-08-25.md`, which
shipped the fused Metal precursor kernel (`astype fp32` + `rms_norm` +
`matmul fn.T` per layer, gate `EXO_DSV4_HC_COLLAPSE_KERNEL`) default-ON
after a +1.89% mean prefill win at ~70.5K real tokens. The ship doc's
§14 explicitly carried the open item: the win was measured at ~70.5K
only, with the +2.73% roofline prediction made at 70.5K — depth
behavior (300K/500K) was untested. This session closes that gap.

Headline result: **the +1.9% gain FULLY HOLDS at depth — no
attenuation.** Measured +1.97% @300K and +1.89% @500K vs the same-depth
OFF arm, both clearing the pre-registered `ON ≥ OFF × 1.015` HOLDS
threshold, and both matching the 70.5K ship number (+1.89%) within
noise. This is the opposite of the sibling `hc_expand` kernel's decay
pattern (+3.87% @70.5K → +1.28% @300K → +0.85% @500K) and of this
session's own pre-registered mechanistic prediction (~+0.6%/+0.4%) —
the prediction was wrong in the pleasant direction (§3).

## 0. Pre-registered criteria (from task brief, verbatim)

Written to `/tmp/hccol_depth_preregistration.txt` at **2026-08-25
11:46:07 CDT**, before any probe. Quoted verbatim:

> Pre-registered 2026-08-25 by FABLE before any depth probe. Per-depth
> verdict vs same-depth OFF arm: HOLDS if ON >= OFF x 1.015;
> ATTENUATED/INCONCLUSIVE if within +/-1.5% (positive-but-subthreshold
> noted as attenuated); REGRESSION FLAG if ON <= OFF x 0.985 -> restore
> production, document loudly, stop. Mechanistic prediction: baseline
> +1.89% @70.5K real; predicted ~+0.6% @300K-target, ~+0.4%
> @500K-target (sibling hc_expand decay scaling). Sanity anchors
> (either arm, STOP and report if outside by >8%): 340-370 tok/s
> @300K-target, 320-350 tok/s @500K-target. Quality gate every probe:
> needle FALCON-MERCURY-7749 recovered exactly, no U+FFFD, no BOS
> spam, generated text captured.
> 2026-08-25 11:46:07 CDT
```

The repeat branch (one repeat pair at the first inconclusive depth) was
pre-registered for the inconclusive case only; because both deltas
cleared +1.5%, no repeats were triggered and none ran.

## 1. Results at a glance

n=1 per arm per depth; no repeats were triggered (both deltas cleared
+1.5%, so the pre-registered repeat branch — reserved for the
inconclusive case — never fired).

| Arm | Target | Real prompt tok | TTFT (s) | Prefill tok/s | Decode tok/s | reasoning_tokens | Needle |
|-----|--------|----------------:|---------:|--------------:|-------------:|-----------------:|:------:|
| ON  | 300K   | 211,670         |  579.97  |  364.9700     |  25.1234     | 55               | OK     |
| ON  | 500K   | 352,277         | 1019.52  |  345.5335     |  22.8427     | 32               | OK     |
| OFF | 300K   | 211,239         |  590.16  |  357.9329     |  24.7175     | 34               | OK     |
| OFF | 500K   | 352,070         | 1038.19  |  339.1177     |  22.6526     | 35               | OK     |
| Restored prod smoke | 2K | 1,520 | 5.50 | 276.2442 | 29.3494 | 33 | OK |

All 5 responses byte-identical `'FALCON-MERCURY-7749'`, zero U+FFFD,
zero BOS spam, `needle_found: true` on every probe.

## 2. Verdict against pre-registered criteria

| Depth | ON tok/s | OFF tok/s | Gate (OFF × 1.015) | Δ tok/s | Δ % | Verdict |
|-------|---------:|----------:|-------------------:|--------:|----:|:--------|
| 300K  | 364.9700 | 357.9329  | 363.30             | +7.04   | **+1.9660%** | **HOLDS** (≥ +1.5%) |
| 500K  | 345.5335 | 339.1177  | 344.20             | +6.42   | **+1.8919%** | **HOLDS** (≥ +1.5%) |

- Both depths clear the pre-registered `ON ≥ OFF × 1.015` HOLDS
  threshold (300K by +0.47 pp, 500K by +0.39 pp).
- No regression flag at either depth (`ON ≤ OFF × 0.985` never close:
  ON is 1.0197× / 1.0189× of OFF).
- The shipped 70.5K number for comparison: **+1.8880%** mean-to-mean.
  Measured depth behavior: **+1.89% @70.5K → +1.97% @300K → +1.89%
  @500K** — the gain holds flat, with no attenuation at either depth.
- Cluster left in production configuration (kernel ON via
  `start_cluster.sh` default). See §7.

## 3. Cross-check vs mechanistic prediction — prediction was WRONG in the good direction

The pre-registered mechanistic prediction, extrapolated from the
sibling `hc_expand` kernel's measured decay (+3.87% @70.5K → +1.28%
@300K → +0.85% @500K), was **~+0.6% @300K-target, ~+0.4% @500K-target**
— i.e., a modest, decaying gain at depth.

Measured: **+1.97% @300K, +1.89% @500K** — essentially flat at the
70.5K ship number, nowhere near the predicted decay. The prediction
was wrong in the pleasant direction (underestimated; the gain does not
shrink with depth in this sample).

Plausible reading — two non-exclusive candidates:

(a) **Cost-scaling symmetry**: `hc_collapse`'s precursor ops (`astype
fp32` + `rms_norm` + `matmul fn.T` per layer) are per-token-per-layer
work whose absolute cost scales with prompt length exactly like the
rest of prefill, so their relative share does NOT shrink as SDPA grows
with depth the way `hc_expand`'s did. (b) **The `hc_expand` decay model
does not transfer between the two ops** — the sibling kernel's measured
decay was its own story; extrapolating it to `hc_collapse` was a
heuristic that this sample refutes.

Honest n=1 caveat: with the ~0.3–0.8% run-to-run spreads seen in prior
sessions at these depths, each individual point estimate carries maybe
±1% uncertainty. But both points landing at ~+1.9% — matching the 70.5K
number — is consistent, and both pass the pre-registered gate as
written. The HOLDS verdict stands per the pre-registered n=1 criterion;
a tighter bound on the depth behavior would need paired repeats, which
the pre-registered design only budgeted for the inconclusive case.

## 4. Sanity anchor gate — no miss this session

Pre-registered: 340–370 tok/s @300K-target, 320–350 tok/s @500K-target,
STOP and investigate if either arm deviates >8% from its band.

| Arm | Band | Measured | In band? |
|-----|------|---------:|:--------:|
| ON  @300K | 340–370 | 364.9700 | YES |
| OFF @300K | 340–370 | 357.9329 | YES |
| ON  @500K | 320–350 | 345.5335 | YES |
| OFF @500K | 320–350 | 339.1177 | YES |

No stop triggered. Note: unlike the `hc_expand` depth session (where
OFF@500K landed ~5% below its speculative band), no anchor missed here
at all — all four arms sat inside their pre-registered bands.

## 5. Runner env — per-arm verification

Verified via `ssh <node> ps eww <pid>` on all 8 runner PIDs per arm (4
per node) and captured to `/tmp/hccol_depth_env_{on,off,restored}.txt`.

### 5.1 ON arm (production config)

Runner PIDs: m4-1 { 25937, 25938, 25939, 25949 }, m4-2 { 27261, 27262,
27263, 27272 }. `EXO_DSV4_HC_COLLAPSE_KERNEL=1` on **8/8 PIDs**.

### 5.2 OFF arm (gate=0)

Runner PIDs: m4-1 { 32804, 32805, 32806, 32816 }, m4-2 { 34443, 34444,
34445, 34454 }. `EXO_DSV4_HC_COLLAPSE_KERNEL=0` on **8/8 PIDs**;
`EXO_DSV4_HC_EXPAND_KERNEL=1` intact. A diff of the ON vs OFF blocks
shows **ONLY the gate var differs** — byte-identical otherwise.

### 5.3 Restored arm (production)

Runner PIDs: m4-1 { 37753, 37754, 37755, 37765 }, m4-2 { 39911, 39912,
39913, 39922 }. `EXO_DSV4_HC_COLLAPSE_KERNEL=1` on **8/8 PIDs**.

### 5.4 Constant env across all arms

```
EXO_DSV4_HC_EXPAND_KERNEL=1
EXO_SPECULATIVE=0
EXO_DSV4_MTP=0
EXO_DSV4_DSPARK=1
EXO_DSV4_BATCHED_PREFILL=1
EXO_DSV4_SEQ_SPLIT=1
EXO_PREFILL_STEP_SIZE=2048
MLX_JACCL_SHARDING_MODE=Tensor
```

Cluster: TP worldSize=2, fp8, `deepseek-ai/DeepSeek-V4-Flash-0731`.

**Env isolation: OK.** The one variable under test differs between arms;
every other production-relevant flag is byte-identical.

## 6. Git SHAs deployed per arm

All arms, both nodes, unchanged throughout: **exo `f7ef1180e`, mlx-lm
`8d5de181d`**.

The OFF arm did **not** roll the mlx-lm submodule back — the kernel code
stayed installed and the env gate `=0` selected the classic
bit-identical path (the sanctioned rollback mechanism per
`docs/hc-collapse-kernel-ab-2026-08-25.md` §13/§14.5). This is a
cleaner isolation than the hc_expand depth run: **zero code delta
between arms, pure env flip.**

The initial ON-arm env capture was taken while the laptop sat one
docs-only commit ahead (`99f5f96b8`); the delta `99f5f96b8 →
f7ef1180e` is verified docs-only (`PERFORMANCE_HISTORY.md` +
`hc-collapse-kernel-ab-2026-08-25.md`), no runtime code path touched.
Nodes synced to `f7ef1180e` for the OFF relaunch and stayed there
through the production restore.

## 7. Cluster left in production configuration

Per the task brief, cluster was returned to production (kernel ON,
`EXO_SPECULATIVE=0 EXO_DSV4_MTP=0`) at the end of the session.

- **Relaunch cmd** (tmux `hccol_depth_restore`):
  `EXO_SPECULATIVE=0 EXO_DSV4_MTP=0 ./start_cluster.sh` — **no explicit
  gate variable**. The script's default `: "${EXO_DSV4_HC_COLLAPSE_KERNEL:=1}"`
  promoted the var to 1 on its own; the bare relaunch IS the proof the
  flip holds (mirrors the ship-day §14.5 procedure).
- **Deploy log**: `/tmp/start_cluster_hccol_depth_restore.log` —
  "Nodes synchronized on commit f7ef1180e", HEALTHY (Nodes: 2,
  Identities: 2), `READY (2/2)` in ~330s, zero `RunnerFailed`.
- **Env verified restored on all 8 runner PIDs** (§5.3):
  `EXO_DSV4_HC_COLLAPSE_KERNEL=1`.
- **Live smoke probe** (2000-token target, §1 last row): needle-exact
  `'FALCON-MERCURY-7749'`, zero U+FFFD, zero BOS spam, coherent.
- **Independent /state check** post-restore: 2× `RunnerReady`,
  DSv4 placed. Cluster is serving in production config.

## 8. Method

Per task brief, methodology mirrors `docs/hc-collapse-kernel-ab-2026-08-25.md`
§13 and the sibling `docs/hc-expand-depth-verification-2026-08-24.md`
§8:

- Harness: `bench/phase3_precheck_depth_throughput.py` unmodified,
  laptop-side, `--targets {300000|500000}` `--max-tokens 128`; one
  probe at a time, sequentially, `--json-out` per probe.
- Numerator: real tokenizer ground truth (offline `AutoTokenizer` on
  the model repo). NEVER derived from server-reported counts (see
  `exo-dsv4-prefill-tuning` skill "THE BIG ONE").
- Denominator: wall-clock TTFT (client `time.perf_counter()`).
- Known +4-token API-vs-tokenizer drift observed on every probe
  (long-standing, harness warns, ground truth used). Present on all
  five probes, sensitive to neither depth nor gate.
- OFF arm required a full cluster relaunch (env is process-level —
  cannot be injected into live runners); same justification as the
  70.5K A/B doc §3.3.
- Sequence: ON@300K → ON@500K → OFF relaunch → OFF@300K → OFF@500K →
  production restore → smoke probe.

## 9. Raw artifacts

Harness JSONs:

- `/tmp/hccol_depth_on_300k.json`
- `/tmp/hccol_depth_on_500k.json`
- `/tmp/hccol_depth_off_300k.json`
- `/tmp/hccol_depth_off_500k.json`
- `/tmp/hccol_prod_smoke.json`

Runner env captures (all 8 PIDs per arm):

- `/tmp/hccol_depth_env_on.txt`
- `/tmp/hccol_depth_env_off.txt`
- `/tmp/hccol_depth_env_restored.txt`

Pre-registration + working notes:

- `/tmp/hccol_depth_preregistration.txt`
- `/tmp/hccol_depth_results.md`

Cluster relaunch logs:

- `/tmp/start_cluster_hccol_depth_off.log`     (OFF-arm relaunch → READY)
- `/tmp/start_cluster_hccol_depth_restore.log` (production restore → READY)

## 10. Note on repeats — why none ran

The pre-registered procedure allowed ONE repeat pair at ONE depth
(ON/OFF @ the first depth) **only if** the 300K delta fell inside
±1.5% (inconclusive). The measured 300K delta was +1.97% — clearly
outside the inconclusive band — so the repeat branch was never
triggered, and by design no repeats ran at either depth. Both depths
are n=1 per arm, per the pre-registered n=1 criterion; the HOLDS
verdicts stand exactly as the gate was written.

## 11. Limitations honestly stated

- **n=1 per arm per depth.** A single probe per arm cannot bound the
  run-to-run noise floor. Prior sessions at these depths observed
  ~0.3–0.8% spreads; individual point estimates here carry maybe ±1%
  uncertainty. The strength of this session is CONSISTENCY — both
  depths independently landing at ~+1.9%, matching the 70.5K ship
  number — not the tightness of any single delta.
- **The pre-registered mechanistic prediction was wrong** in the
  pleasant direction (predicted ~+0.6%/+0.4% decay; measured
  +1.97%/+1.89%). We do not claim the prediction failure is
  understood; §3 offers two plausible readings, neither proven.
- **No per-layer breakdown** (e.g. `EXO_PROFILER=spans` or xctrace)
  was taken — profiling at depth is forbidden by the hard guardrail
  (see §12). So we cannot attribute the held gain to the precursor
  ops vs. some interaction effect from this data alone.
- **OFF arm isolation is env-flip only** — which is also its
  strength: zero code delta between arms (the kernel stayed
  installed; the gate selected the classic path). The cost: the OFF
  arm and ON arm are not fully independent in the sense of a
  submodule-rolled-back comparison, but the classic path is
  bit-identical to pre-ship behavior and was itself the object of the
  70.5K A/B's baseline.

## 12. Anomalies / incidents

**Zero.** No client-side hang occurred at 500K this time (the
hc_expand session's §13.1 hang did not recur), no runner failure, no
stop-trigger, no deploy re-verification needed. Hard guardrails
respected throughout: no xctrace/Metal traces during any live deep
prefill (per `docs/p3-followup-allsum-wait-at-depth-2026-08-24.md`
§6), `EXO_PROFILER_SYNC_SPANS` never set, one probe at a time, no
blind sleeps, no files hand-edited on the studios, no retry-mitigation
hacks.

## 13. Files created / modified

- `docs/hc-collapse-depth-verification-2026-08-25.md` (this doc)
- `docs/PERFORMANCE_HISTORY.md` — appended depth-verification note
  under the existing 2026-08-25 hc_collapse entry (before the
  "Quick-reference" divider)
