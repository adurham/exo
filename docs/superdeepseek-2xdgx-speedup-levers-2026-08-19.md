# SuperDeepseek-2xDGX serving-stack review — transferable speed levers for the MLX cluster (2026-08-19)

Context: user asked whether `Jiunsong/SuperDeepseek-V4-Flash-abliterated-MQ-2xDGX` (HF) offers
anything for the 2x Mac Studio M4 Max exo cluster running `deepseek-ai/DeepSeek-V4-Flash-0731`.
Verdict on the weights themselves: **no speed delta** — the repo's own benchmark reports a median
matched decode ratio of **1.0012x vs the parent**, i.e. its 118–123 tok/s headline is 100% serving
stack (vLLM + FlashInfer + CUDA graphs + spec decode on 2x DGX Spark / CX-7 RoCE), 0% weights.
The checkpoint is byte-identical -0731 parent shards (pinned commit `9e165c30`) plus a ~2.6GB
abliteration overlay (92 `attn.wo_b` fp8 tensors + 1 bf16 head) and a redirecting index.
The abliteration trial is a separate thread (user deferred download; see warm memory fact 1484).

This doc maps their six serving techniques onto our fork and ranks what to actually do.

## Their stack -> our fork, line by line

| # | Their technique | Our status | Transferable? |
|---|---|---|---|
| 1 | DSpark spec decode, K=1 greedy draft | Implemented; prod runs `EXO_SPECULATIVE=0` since 2026-08-03 | **YES — top lever** |
| 2 | NVFP4 DS-MLA KV cache | `EXO_KV_CACHE_BITS` + TurboQuant exist, both default-off (prod KV bf16) | **YES — second lever** |
| 3 | Regular CUDA graphs (their 1.21–1.27x) | No MLX analog (no graph capture); mx.compile/async_eval partially cover | Idea only: re-measure per-step host overhead |
| 4 | Prefix caching | `KVPrefixCache` exists in batch_generate.py | Verify actually engaged in prod path |
| 5 | Chunked prefill + async scheduling | Equivalents effectively present at c=2 | Nothing to take |
| 6 | FlashInfer b12x MoE kernels | CUDA-only; MLX gate+up fusion tried & deleted 2026-06-18 (B>1 degeneration) | No — known dead end |

## Lever 1 — DSpark A/B with -0731's NATIVE head (never run)

Why it's first: fork history has DSpark PP at **27–33 tok/s vs 15.6 sequential** (facts 1116/1410),
but prod turned speculation off after the 2026-08-03 A/B showed parity (24.68 off vs 23.62 on).
That A/B used the PREVIEW-vintage draft head against -0731's re-post-trained hidden states
(64% acceptance) — the mismatched-head asterisk was never removed. Since then:

- The **native-head attach code now exists** in `utils_mlx.py` (~line 855): scans the checkpoint's
  own `model.safetensors.index.json` for `mtp.*` shards, loads only those, sanitizes, and remaps
  onto the DSpark module (`stages.{n}` + main_proj/main_norm/hc_head/markov/confidence specials).
  -0731 bundles its own trained MTP stack (`num_nextn_predict_layers` in config; 3 MTP layers
  per the SuperDeepseek card).
- The SuperDeepseek repo is **third-party evidence spec decode pays on this exact checkpoint**
  when the draft head matches the weights (their whole 118–123 tok/s profile runs it).
- Their **K=1 greedy** choice is itself a reusable idea: minimal draft depth = cheapest verify,
  highest acceptance, and the smallest possible exposure surface for the batched-verify numerics
  class root-caused in August (ROWSEQ fixed it — facts 1140/1141 — but L=2 verify is a much
  smaller surface than block=5).

Recipe (prepared 2026-08-15, fact 1409, still unrun):
```
DSV4_MODEL_ID=deepseek-ai/DeepSeek-V4-Flash-0731 DSV4_SHARDING=Pipeline \
  EXO_PP_METAFRAME=1 JACCL_TRACE_PROGRESS=1 ./start_cluster.sh
# NOTE: omit EXO_PP_BATCHED_DECODE (defaults 0) so the speculation path is reachable.
# Confirm engagement in runner log: "PP speculation enabled in BatchGenerator"
#                                 + "PP speculation using DSpark"
```
Sweep: native head vs preview head, draft depth K=1 / K=2 vs block=5. Prize: the 15.6 -> ~27–33
class jump without the mismatched-head asterisk.

Validation is NON-NEGOTIABLE per standing rules: needle check + `finish_reason` + actual output
text end-to-end. A spec-decode regression shows up as non-termination, never as a slow number.
tok/s alone proves nothing.

## Lever 2 — quantized KV cache at depth

Prod KV is bf16 (`KV_BITS=None`, `EXO_KV_CACHE_BITS` unset). At 100K context, MLA latent KV is
~50KB/token/stream (576 dims x 43 layers x bf16) — ~5GB of reads per decoded token, the main
reason decode sags from ~38 tok/s short-context to ~25–30 at 100K (fact 235). 4-bit KV cuts that
read 4x. Their prod profile shipping FP4 MLA KV on this same architecture is a meaningful external
quality signal for exactly what TurboQuant (`EXO_TURBOQUANT`, rotation-aware 3-bit + residual)
was built for.

Plan: `EXO_KV_CACHE_BITS=4` first (simpler affine path), then TurboQuant; needle + output-quality
validation at 100K before trusting either. Prize: flattens the long-context decode sag.

## Lever 3 — per-step host overhead re-profile

Their CUDA-graphs delta says **21–27% of their decode step was host/launch overhead** on a stack
already far more optimized than a Python/MLX loop. At 30 tok/s, every 3ms of per-step host time
is ~10% throughput. MLX has no graph capture-replay; levers are mx.compile placement (mind the
B>1 mis-specialization that got whole-model fusion deleted 2026-06-18, and the cross-thread
multi-output compile bug, fact 1202) and deeper async_eval pipelining. Third priority: measure
before touching.

## Lever 4 — confirm KVPrefixCache engages for the agent workload

For Hermes-style traffic (every turn re-prefills system prompt + full conversation), prefix-cache
hits are the biggest WALL-CLOCK lever available, independent of decode tok/s. Check
`GenerationStats.prefix_cache_hit` in prod before assuming it works.

## Ranked plan

1. DSpark native-head A/B (everything already built; launch-env change + bench runs, no code edits).
2. 4-bit KV / TurboQuant validation at 100K.
3. Host-overhead re-profile against their 21–27% datapoint.
4. Prefix-cache engagement check (cheap, can ride along with any bench session).

Non-goals: FlashInfer-style MoE kernel work (CUDA-only; MLX equivalent is a documented dead end),
chunked prefill / async scheduling (already effectively present), the SuperDeepseek weights as a
speed play (proven no-op by their own numbers).

Cluster note (2026-08-19): the API did not answer on adams-mac-studio-m4-1.local:52415 nor
192.168.86.201:52415 when probed from the laptop — verify cluster state before bench planning.
