# P05 Phase A Live A/B — Run→Instance→Model→Sharding→Knob Attribution & Misattribution Verdict

**Date:** 2026-08-30 · **Author:** LEAD 1 forensics (read-only) · **Logs:** `/tmp/exo_1753.log` (08-29 17:53 → 08-30 11:08), `/tmp/exo_1108.log` (11:08 → 12:45:54) on `adams-mac-studio-m4-1.local`

---

## Attribution Table (Q1)

| file | request_time | model_id | instance_id | sharding | knob_fired |
|---|---|---|---|---|---|
| live_ab/warmup_b1.json | 10:47 | mlx-community/DeepSeek-V4-Flash | ed3a0993-367e-4de0-b8dd-7ed58df2b898 | tensor | no |
| live_ab/warmup_b2.json | 10:47 | mlx-community/DeepSeek-V4-Flash | ed3a0993-367e-4de0-b8dd-7ed58df2b898 | tensor | no |
| live_ab/base_5k_1.json | 10:47 | mlx-community/DeepSeek-V4-Flash | ed3a0993-367e-4de0-b8dd-7ed58df2b898 | tensor | no |
| live_ab/base_5k_2.json | 10:48 | mlx-community/DeepSeek-V4-Flash | ed3a0993-367e-4de0-b8dd-7ed58df2b898 | tensor | no |
| live_ab/base_5k_3.json | 10:48 | mlx-community/DeepSeek-V4-Flash | ed3a0993-367e-4de0-b8dd-7ed58df2b898 | tensor | no |
| live_ab/base_100k_1.json | 10:53 | mlx-community/DeepSeek-V4-Flash | ed3a0993-367e-4de0-b8dd-7ed58df2b898 | tensor | no |
| live_ab/base_100k_2.json | 10:58 | mlx-community/DeepSeek-V4-Flash | ed3a0993-367e-4de0-b8dd-7ed58df2b898 | tensor | no |
| live_ab/base_100k_3.json | 11:03 | mlx-community/DeepSeek-V4-Flash | ed3a0993-367e-4de0-b8dd-7ed58df2b898 | tensor | no |
| live_ab/warmup_a1.json | 11:17 | mlx-community/DeepSeek-V4-Flash | *(none — HTTP 503)* | — | no |
| live_ab/warmup_a2.json | 11:21 | mlx-community/DeepSeek-V4-Flash | ea0d6b4d-317a-477d-af7e-0cc0c54a2131 | pipeline | no |
| live_ab/quant_5k_1.json | 11:23 | mlx-community/DeepSeek-V4-Flash | ea0d6b4d-317a-477d-af7e-0cc0c54a2131 | pipeline | no |
| live_ab/quant_5k_2.json | 11:26 | mlx-community/DeepSeek-V4-Flash | ea0d6b4d-317a-477d-af7e-0cc0c54a2131 | pipeline | no |
| live_ab/quant_5k_3.json | 11:28 | mlx-community/DeepSeek-V4-Flash | ea0d6b4d-317a-477d-af7e-0cc0c54a2131 | pipeline | no |
| live_ab/quant_100k_1.json | 11:36 | mlx-community/DeepSeek-V4-Flash | ea0d6b4d-317a-477d-af7e-0cc0c54a2131 | pipeline | no |
| live_ab/quant_100k_2.json | 11:43 | mlx-community/DeepSeek-V4-Flash | ea0d6b4d-317a-477d-af7e-0cc0c54a2131 | pipeline | no |
| live_ab/quant_100k_3.json | 11:50 | mlx-community/DeepSeek-V4-Flash | ea0d6b4d-317a-477d-af7e-0cc0c54a2131 | pipeline | no |
| live_ab/same_prompt_A.json | 11:54 | mlx-community/DeepSeek-V4-Flash | ea0d6b4d-317a-477d-af7e-0cc0c54a2131 | pipeline | no |
| live_ab/quant_8k_diag.json | 12:03 | mlx-community/DeepSeek-V4-Flash | b20ca9e4-9472-4bb3-b6d7-81edbdcdb7de | tensor | no |
| live_ab/quant_16k_diag.json | 12:03 | mlx-community/DeepSeek-V4-Flash | b20ca9e4-9472-4bb3-b6d7-81edbdcdb7de | tensor | no |
| live_ab/quant_100k_retry1.json | 12:08 | mlx-community/DeepSeek-V4-Flash | b20ca9e4-9472-4bb3-b6d7-81edbdcdb7de | tensor | no |
| live_ab/quant_100k_retry2.json | 12:13 | mlx-community/DeepSeek-V4-Flash | b20ca9e4-9472-4bb3-b6d7-81edbdcdb7de | tensor | no |
| live_ab/same_prompt_A_0731_dryrun.json | 12:37 | deepseek-ai/DeepSeek-V4-Flash-0731 | 99e6a0a5-11cf-4cb2-9986-ba660dc85437 | tensor | **yes** |
| live_ab_v2/A_warmup_1.json | 12:41 | deepseek-ai/DeepSeek-V4-Flash-0731 | 99e6a0a5-11cf-4cb2-9986-ba660dc85437 | tensor | **yes** |
| live_ab_v2/A_warmup_2.json | 12:41 | deepseek-ai/DeepSeek-V4-Flash-0731 | 99e6a0a5-11cf-4cb2-9986-ba660dc85437 | tensor | **yes** |
| live_ab_v2/A_5k_1.json | 12:41 | deepseek-ai/DeepSeek-V4-Flash-0731 | 99e6a0a5-11cf-4cb2-9986-ba660dc85437 | tensor | **yes** |
| live_ab_v2/A_5k_2.json | 12:41 | deepseek-ai/DeepSeek-V4-Flash-0731 | 99e6a0a5-11cf-4cb2-9986-ba660dc85437 | tensor | **yes** |
| live_ab_v2/A_5k_3.json | 12:42 | deepseek-ai/DeepSeek-V4-Flash-0731 | 99e6a0a5-11cf-4cb2-9986-ba660dc85437 | tensor | **yes** |
| live_ab_v2/A_100k_1.json | 12:45 | deepseek-ai/DeepSeek-V4-Flash-0731 | 99e6a0a5-11cf-4cb2-9986-ba660dc85437 | tensor | **yes** (killed mid-prefill) |
| live_ab_v2/A_100k_2.json | 12:45 | *(none — conn refused)* | — | — | — |
| live_ab_v2/A_100k_3.json | 12:45 | *(none — conn refused)* | — | — | — |

**Key structural fact:** Every `base_*` and `quant_*` run in `live_ab/` was served by **`mlx-community/DeepSeek-V4-Flash`** (the 8-bit conversion), NOT the 0731 production checkpoint. The knob-quantized 0731 head was only ever exercised by the `live_ab_v2/` runs (12:41-12:45) and the 11:13 probe.

---

## Q2 — Why did mlx-community load with PIPELINE at 11:19 (and TENSOR at 12:01)?

**The knob did NOT fire for mlx-community because its `lm_head` is already quantized — not because of `model_type`.**

- The knob condition (`mlx_lm/utils.py:610-620`) requires `EXO_DSV4_LMHEAD_MXFP8=="1"` **AND** `config.model_type=="deepseek_v4"` **AND** `isinstance(mod, nn.Linear)` **AND** `not hasattr(mod, "scales")`.
- The mlx-community `config.json` on the node has `model_type: 'deepseek_v4'` — so the model_type gate **passes**. The env var `EXO_DSV4_LMHEAD_MXFP8=1` was set in `/tmp/p05_lmhead_launch.sh` for the whole A/B window.
- The gate that actually blocks it is `not hasattr(mod, "scales")`: the 8-bit conversion's `lm_head` is **already quantized** — `model.safetensors.index.json` lists `lm_head.scales` (and `lm_head.biases`). So `mod` is not a plain `nn.Linear` (it has `.scales`), the guard fails, and the knob silently no-ops. **No `[LMHEAD_MXFP8]` line appears for mlx-community loads** (11:19, 12:01) — exactly as observed. The knob only fired on the two 0731 loads (11:13:24, 12:32:44).

**Sharding difference (11:19 Pipeline vs 12:01 Tensor):**
- 11:19:08 — `JIT auto-placing mlx-community/DeepSeek-V4-Flash (sharding=Pipeline, meta=MlxRing, min_nodes=1)`. This was a **single-node** placement (`min_nodes=1`), chosen because the 0731 instance `6a7f098e` was still resident and the JIT placement poller had been failing since 11:15:46 with `no admissible placement (No cycles found with sufficient memory)`. The 0731 instance was only evicted at 11:19:07 (`JIT idle reaper unloading instance 6a7f098e ... idle 302s >= 300s`), freeing memory, and the mlx-community load then went in as a **single-node Pipeline** ring.
- 12:01:12 — `JIT auto-placing mlx-community/DeepSeek-V4-Flash (sharding=Tensor, meta=MlxJaccl, min_nodes=2)`. By then the previous mlx-community instance `ea0d6b4d` had been evicted (11:59:57) and the cluster had converged to a 2-node Tensor placement — the production configuration.

So the 11:19-11:50 "quant" runs were served by a **single-node Pipeline-parallel** mlx-community instance — a configuration with a documented depth-dependent spec-decode collapse (see VERDICT).

---

## Q3 — What killed the API at ~12:45?

**Clean manual shutdown, not a crash.** The cluster was stopped (SIGTERM) at 12:45:53 and a fresh cluster started at 12:46:02.

- Current cluster pid `65573` started **Sun Aug 30 12:46:02 2026** (`ps -o lstart -p 65573`), launched by `bash /tmp/verbon3_launch.sh` inside screen `65564.exo`.
- The prior cluster's log tail (`/tmp/exo_1108.log`) shows a **clean teardown**:
  - `12:45:53.215 | INFO | exo.worker.runner.bootstrap:_graceful_sigterm:220 ] Runner received SIGTERM — exiting cleanly to tear down RDMA.`
  - `12:45:53.301 | INFO | ... _release_gpu_memory_before_exit:193 ] Released MLX buffers before exit`
  - `12:45:54.196 | INFO | exo.worker.runner.supervisor:run:352 ] Runner process successfully terminated: 0`
  - `12:45:54.200 | INFO | exo.main:main_inner:379 ] EXO Shutdown complete`
- The `anyio.BrokenResourceError` / `Error in ASGI Framework` tracebacks at 12:45:50 are **downstream symptoms** of the teardown (the in-flight A_100k_1 SSE stream's send channel was closed as the process shut down), not the cause.
- **Classification: clean manual stop** (SIGTERM, exit 0, no crash traceback). The A_100k_1 run was killed mid-prefill at `processed_tokens=88064 / total_tokens=111074`; A_100k_2/3 then got `Connection refused` because the API was already down.

---

## Q4 — Was there EVER a request ≥8192 prompt tokens against the knob-quantized 0731 head?

**No.** The knob-quantized 0731 instance `99e6a0a5` (active 12:32:40 → 12:45:50) served only:
- trivial-context probes (360-392 tokens: A_warmup_1/2, the 12:33 "Say OK" probe),
- 5.6K-context runs (5562-5619 tokens: A_5k_1/2/3),
- one 100K run (A_100k_1) that was **killed mid-prefill at 88064/111074** by the 12:45 shutdown.

**The batched-verify regime (`EXO_DSV4_VERIFY_BATCH=1`, `EXO_DSV4_VERIFY_BATCH_MIN_CTX=8192`) was NEVER exercised against the quantized head.** This is the explicit **unmeasured regime**. The 100K regime on the quantized head is also unmeasured (killed/conn-refused).

**Healthy quantized-head evidence (12:32-12:45):** see `v2_0731_acceptance.json`. Summary:
- `[MTP] cycles=237 mean_accept=1.890/3 hist=0:43,1:37,2:60,3:97` — sustained ~1.89/3 acceptance, dominated by 2-3 token accepts. **No zero-acceptance collapse.**
- `[MTP-PROF]` draft mean ~9ms, verify mean ~66ms, total ~78ms (cycles 50-200). **No ~550ms draft-phase signature.**
- Decode ~200-223 tok/s at 5.6K prompt tokens.

---

## Q5 — The 11:13-11:14 "Say OK" probe (instance 6a7f098e, 0731, knob ON)

- Load: `11:13:26.463 loading model from .../deepseek-ai--DeepSeek-V4-Flash-0731 with tensor parallelism`; knob fired `11:13:24.411 [LMHEAD_MXFP8] quantized lm_head to mxfp8`.
- Request: `11:14:05.248 received chat request ... instance_id='6a7f098e-f342-4e4f-b536-f9a48e7004e1' ... model='deepseek-ai/DeepSeek-V4-Flash-0731' ... content='Say OK.'`.
- Acceptance (trivial ctx, ~10 tokens): `[MTP] cycles=13 mean_accept=1.308/3 hist=0:3,1:4,2:5,3:1` — **healthy** (mean ~1.3-1.6/3, no collapse). The first cycle `mean_accept=0.000/3` is a single cold-start cycle, not a regime.
- **Eviction:** yes — the 0731 instance `6a7f098e` was evicted by the JIT idle reaper at `11:19:07.341` (`idle 302s >= 300s`), immediately before the mlx-community Pipeline load at 11:19:08. The JIT placement poller had been blocked since 11:15:46 (`no admissible placement (No cycles found with sufficient memory)`) because the 0731 instance was still resident.

---

## Q6 — The ~550ms draft-phase signature lived on the mlx-community PP instance, not the quantized 0731 head

**mlx-community Pipeline instance `ea0d6b4d` (11:19-11:50):** `[MTP-PROF]` shows a catastrophic draft-phase collapse:
- cycles=50 (11:21:17): `draft mean=499.87ms`, `verify mean=873.68ms`, `total mean=1417.37ms`
- cycles=100 (11:23:09): `draft mean=527.21ms`, `verify mean=1049.70ms`, `total mean=1606.27ms`
- cycles=500 (11:50:36): `draft mean=551.83ms`, `verify mean=1218.82ms`, `total mean=1789.72ms`

Draft mean ~500-552ms — the **~550ms draft-phase signature** is exactly this. Acceptance was `mean_accept=0.000/3` (hist=0:500) — **100% zero-acceptance** across the entire 11:19-11:50 window.

**mlx-community Tensor instance `b20ca9e4` (12:01-12:13):** the same model under Tensor parallelism was **fast and healthy**:
- cycles=250 (12:08:42): `draft mean=14.32ms`, `verify mean=58.08ms`, `total mean=74.93ms`
- cycles=300 (12:08:46): `draft mean=14.32ms`, `verify mean=58.25ms`, `total mean=75.09ms`

So the slowness and the zero-acceptance were properties of the **Pipeline-parallel** mlx-community instance, not of the quantized 0731 head. (The 12:13:41 cycles=350 line shows `draft mean=37.12ms max=7996.62ms` — a single outlier spike, not the sustained ~550ms collapse.)

---

## VERDICT

**The catastrophic 0.05-0.06x zero-acceptance regression was never a property of the mxfp8-quantized 0731 head.** It was measured against a **different model** — `mlx-community/DeepSeek-V4-Flash` (8-bit) — JIT-loaded under **pipeline parallelism** (single-node, `min_nodes=1`) for the 11:19-11:50 window, a configuration with a documented depth-dependent spec-decode collapse (100% zero-acceptance, draft ~550ms). The knob-quantized 0731 head's only live measurements — trivial ctx (360-392) and 5.6K ctx (5562-5619), rowseq verify — were **healthy**: mean_accept ~1.89/3, draft ~9ms, verify ~66ms, decode ~200-223 tok/s.

**Unmeasured regimes on the quantized head:**
- **≥8K batched verify** (`EXO_DSV4_VERIFY_BATCH=1`, `MIN_CTX=8192`) — never exercised.
- **100K** — A_100k_1 killed mid-prefill (88064/111074) by the 12:45 shutdown; A_100k_2/3 connection-refused.

**Misattribution mechanism (root cause of the false "QuantizedLinear draft/verify bug" claim):** `bench/ab_probe_tier1.py` hardcoded `MODEL='mlx-community/DeepSeek-V4-Flash'` until commit `de925720e` (13:04) added `--model`. Every "quant" run in `live_ab/` therefore hit the 8-bit mlx-community model — which the knob **cannot** quantize (its `lm_head` already has `.scales`) — under a degenerate single-node Pipeline placement. The 0.05-0.06x decode and the ~550ms draft signature were the Pipeline-parallel mlx-community instance's spec-decode collapse, and were wrongly attributed to the mxfp8-quantized 0731 head. The knob itself was never even active on the model being measured.

**Corroborating note:** the 10:47 `base_5k` run (knob OFF, mlx-community, Tensor) ALSO showed `mean_accept=0.000/3` early — confirming the zero-acceptance is a property of the mlx-community model/placement, independent of the knob.
