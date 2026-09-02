# Pre-registration: Temperature arm (temp=1.0 vs temp=0.8 baseline)

Written BEFORE any temp=1.0 request was sent to the cluster. The decision rule
below was fixed by the PM and is reproduced VERBATIM — numbers and bands are
not to be altered by any dispatch, including this one.

## Decision rule (VERBATIM, PM-fixed)

  Baseline B = temp=0.8 pooled repetitive arm: n=7, mean 33.58 t/s, sd 1.72, per-iteration [31.72, 31.79, 33.02, 34.00, 33.33, 37.23, 33.96].
  Treatment T = temp=1.0 arm mean decode t/s, n=5 scored.
  - CONFIRMED: T <= 32.0 AND Welch one-sided p < 0.05. => temperature materially lowers decode. Report share of the 14 t/s real-usage gap as (33.58 - T)/14.
  - NOT-CAUSE: T >= 33.0, OR (Welch p >= 0.05 AND the 95% CI upper bound on the drop is < 2.5 t/s). => temperature is NOT the cause; the '11-16% of decode wall outside profiled phases' suspect takes priority.
  - PARTIAL: anything else (32.0 < T < 33.0, or bands and significance disagreeing).
  Mandatory in ALL branches: report (B-T)/B as a percentage, the share of the 14 t/s gap, and the minimum detectable effect.
  MDE: with pooled sd 1.72, n1=7, n2=5, se = 1.72*sqrt(1/7+1/5) = 1.007 t/s; at alpha=0.05 one-sided and 80% power the MDE is 2.49 t/s (~7.4% relative). A null result therefore rules out drops larger than ~2.5 t/s, NOT small drops.
  Mechanism check (secondary, non-gating): if decode falls, cycle time ([MTP-PROF] verify+draft+accept+rollback) staying flat (<10% change vs baseline) while decode moves implicates ACCEPTANCE. Cycle time moving in proportion to the decode drop implicates CYCLE COST, and the temperature-acceptance story is wrong even if decode falls.
  Validity gates: G1 runner PIDs+lstart byte-identical before AND after the arm. G2 probe rc=0 and 5/5 scored iterations returned. G3 zero DEGENERATION/PPSpec/crash/SIGKILL matches in the arm's log window; benign HF-404 warnings classified and counted separately. G4 server-reported prompt_tokens within 0.5% of 89,408. G5 idle gate (<0.10 gpu_usage_ratio both nodes) satisfied before the arm.

Statistics and verdict are OWNED BY A SEPARATE DISPATCH. This file plus
raw/ artifacts contain only measurements; no Welch test or branch decision
is performed here.

## Pre-flight facts (PM-verified, established before this arm)

- Cluster: 2-node Mac Studio M4 Max exo cluster, DeepSeek-V4-Flash-0731 via
  MLX, MTP speculative decoding. API host 192.168.86.201:52415. Nodes:
  macstudio-m4-1 (192.168.86.201), macstudio-m4-2 (192.168.86.202).
- entropy_probe.py sends NO temperature/top_p in its request body (only
  model/messages/stream/max_tokens) — verified at entropy_probe.py:190-195.
- Server resolves sampling PER FIELD: request -> instance -> card ->
  cluster-env -> hardcoded (src/exo/worker/engines/mlx/sampling.py,
  resolve_sampling(), docstring literally says "Resolution order per field",
  implemented via _first_non_none per key).
- Model card resources/inference_model_cards/deepseek-ai--DeepSeek-V4-Flash-0731.toml
  [sampling_defaults]: temperature=0.8, top_p=0.9, top_k=40, min_p=0.0.
  Instance defaultTemperature=None, EXO_DEFAULT_TEMPERATURE unset.
- THEREFORE the temp=0.8 baseline is real, and sending ONLY temperature=1.0
  in the request body changes ONLY temperature (top_p/top_k/min_p stay at
  card values). Clean single-variable contrast.
- Temperature plumbing is LIVE end-to-end (canary: temp=0.0 reproduced
  byte-identically twice; temp=2.0 differed).
- Runner PIDs verified: m4-1 PID 83029, lstart "Tue Sep  1 16:19:35 2026";
  m4-2 PID 85554, lstart "Tue Sep  1 16:19:37 2026".
  EXO_DSV4_MTP_PROFILE=50 and EXO_DSV4_RB_PROFILE=1 live on both genuine
  runner PIDs. GPU idle ~0.027-0.029 both nodes.
- Baseline expected server-reported prompt_tokens for mode=repetitive /
  75000 words: 89,408 (G4 gate).

## STEP 0 finding: does a REAL Hermes session send an explicit temperature?

Checked 2026-09-02, before any temp=1.0 request was sent:

1. Config files: `/Users/adam.durham/.hermes/config.yaml` and
   `/Users/adam.durham/.hermes/profiles/exo/config.yaml` — grep for
   `temperature` (case-insensitive) across both: ZERO matches. The exo
   provider block (providers.exo, base_url
   http://192.168.86.201:52415/v1) sets only api_mode/base_url/
   discover_models/extra_body(use_prefix_cache, service_tier)/models/
   request_timeout_seconds. No temperature anywhere in either config.

2. Client source (live install /Users/adam.durham/.hermes/hermes-agent,
   mirrored at /Users/adam.durham/repos/hermes-agent):
   - Provider profile: plugins/model-providers/exo/__init__.py registers
     `exo = ProviderProfile(...)` with NO `fixed_temperature` (defaults to
     None) and no request_overrides from config contain temperature.
   - transports/chat_completions.py build_kwargs temperature logic (L874-882):
     if profile.fixed_temperature is OMIT_TEMPERATURE: pass  # Don't include temperature at all
     elif profile.fixed_temperature is not None:
         api_kwargs["temperature"] = profile.fixed_temperature
     else:
         # Use caller's temperature if provided
         temp = params.get("temperature")
         if temp is not None:
             api_kwargs["temperature"] = temp
     For exo, fixed_temperature is None, and the main-loop caller
     (agent/chat_completion_helpers.py build_api_kwargs) never passes a
     `temperature` param — the only temperature values it ever emits are on
     summary/auxiliary paths (auxiliary_client _fixed_temperature_for_model,
     which returns None for DSv4/exo; the only fixed-temperature entries are
     Kimi=OMIT and Arcee Trinity Thinking=0.5, neither applicable).
   - `agent.adaptive_sampling` is enabled in both configs; the config key
     names (fp_max/pp_max/ema_alpha/cold_start_turns/history_window) appear
     NOWHERE in the installed client code — grep of the entire
     .hermes/hermes-agent tree (py files, venv/node_modules excluded) has
     zero hits. No code reads it, so it cannot inject temperature. (A past
     session note in @session:default/20260610_120311_d86fa3 described it as
     "auto-tunes sampling params like temperature/top-p per turn", but no
     implementing code exists in the installed client; treat the config key
     as inert.)
   - Config providers.exo.extra_body = {use_prefix_cache, service_tier} —
     no temperature there either.

**Verbatim conclusion: NO — a real Hermes session does NOT send an explicit
temperature to this cluster.** Neither config nor client code emits a
temperature field on chat-completions requests to the exo endpoint. Since the
exoserver applies the card default temperature=0.8 to any request omitting
temperature, real Hermes sessions on DSv4-Flash run at **temp=0.8 — the same
as the bench baseline.** The premise of a bench-vs-real temperature
difference is therefore FALSE as far as the client is concerned; the observed
real-usage slowdown vs the bench cannot be explained by real sessions
sampling at temp=1.0 while the bench sampled at 0.8.

Per the task instruction, the temp=1.0 arm was still executed: it measures
decode sensitivity to temperature on its own merits (and would matter if the
user ever configures an explicit temperature, or if another client hitting
the cluster sends one).

## Arm specification (as executed)

- ONE arm only: mode=repetitive, words=75000, tag=temp10, iterations=5,
  warmup=1, max_tokens=256, seed=1234, temperature=1.0.
- Probe: temperature/temp_probe.py — byte-copy of entropy_probe.py plus a
  single additive `--temperature` float arg injected into the three POSTed
  body dicts. No top_p/top_k/min_p added (single-variable).
- Driver: temperature/run_temp_arm.sh — mirrors entropy/run_entropy_ab.sh
  run_arm(): idle gate (<0.10 both nodes) -> runner PIDs+lstart BEFORE ->
  exo.log byte offsets BEFORE -> probe from cwd /tmp with rc captured on the
  immediately-following line -> offsets AFTER -> PIDs AFTER -> byte-window
  harvest of [MTP-PROF] lines, error-pattern lines, and the ~200KB
  pre-window anchor, on BOTH nodes. Plus periodic gpu_usage_ratio sampling
  on both nodes during the arm.
- All raw outputs under temperature/raw/.