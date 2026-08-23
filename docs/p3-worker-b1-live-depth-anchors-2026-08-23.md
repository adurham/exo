# P3 worker B1 — live decode-throughput depth anchors (2026-08-23)

**Task**: fresh, real, end-to-end decode-throughput anchors at two context
depths (~100K real tokens and the previously-validated deep point, ~352K real
tokens, labeled "500K" in prior benches), with a decode window >= 60s of pure
decode and EOS banned so length is the only stop signal. These anchors are the
delta a later P3 phase decomposes.

**Cluster**: unchanged, untouched, healthy throughout. 1 `MlxJacclInstance`,
`deepseek-ai/DeepSeek-V4-Flash-0731`, TP=2, both runners `RunnerReady` before
and after every probe. No relaunch, no runner restart, no model load/unload, no
config change, no ssh writes. Every probe is a single HTTP POST.

**Repo HEAD**: `6bc843bfc`. The `usage.prompt_tokens` prompt-TAIL bug fix
(`7d14daea7`) is an ancestor of HEAD, so `usage.prompt_tokens` is trustworthy
here — verified explicitly with `git merge-base --is-ancestor`.

---

## 0. Headline result

| depth (REAL `usage.prompt_tokens`) | decode tok/s | ms/token | decode window | completion tokens | finish_reason |
|---|---|---|---|---|---|
| 520 (short control) | **29.63** | **33.75** | 67.47 s | 2000 / 2000 | `length` |
| 100,026 | **27.94** | **35.79** | 71.55 s | 2000 / 2000 | `length` |
| 352,599 (deep) | **23.48** | **42.59** | 85.13 s | 2000 / 2000 | `length` |

All three points: EOS genuinely banned (`finish_reason=length`, full 2000
tokens delivered), decode window >= 60s, depth read from the API's own usage
field, decode-only throughput (TTFT/prefill excluded by construction).

**T1's anchors reproduced within noise, with one qualification**: short and
100K reproduce cleanly; the deep point comes in **+9.2% ABOVE** T1's 21.51
tok/s. That gap is very likely methodological, not a real regression or
improvement — see §4.

---

## 1. Methodology, and a real probe bug found and worked around

### 1.1 The bug: `decode_probe.py` does NOT ban EOS

`bench/decode_probe.py` posts `{"bench": true}` to `/v1/chat/completions` and
its docstring claims this "bans EOS so length is the only stop signal".

**It does not.** `ChatCompletionRequest` (`src/exo/api/types/api.py:243`) has
**no `bench` field**. It is a plain pydantic `BaseModel`, so the extra key is
silently dropped and the request runs as an ordinary completion. The EOS ban
lives at `batch_generate.py:2658`:

```python
if is_bench:
    eos_ids = eos_ids_from_tokenizer(self.tokenizer)
    logits_processors = [ban_token_ids(eos_ids)] + logits_processors
```

and `is_bench = task_params.bench` is only ever set True by the **separate**
`/bench/chat/completions` route (`main.py:1183-1198`), which builds a
`BenchChatCompletionRequest` and force-sets `"bench": True`.

**Verified live, side by side** (prompt "Say hello.", `max_tokens=60`, one
request each, 2026-08-23):

```
/v1/chat/completions    extra={'bench': True}           -> events=52 finish=stop   usage={'prompt_tokens': 7, 'completion_tokens': 56, ...}
/bench/chat/completions extra={'use_prefix_cache': False} -> events=58 finish=length usage={'prompt_tokens': 7, 'completion_tokens': 60, ...}
```

`/v1` + `bench:true` stopped **early at 56 < 60** with `finish_reason=stop`.
`/bench` ran to exactly 60 with `finish_reason=length`.

**This is precisely the failure that invalidated the prior T5 capture down to a
~9s decode window.** `docs/PERFORMANCE_HISTORY.md` (T5 entry, line ~1980)
already records the symptom — *"decode window was only ~9s (a `bench=True`-
routing quirk meant EOS wasn't banned as intended)"* — but not the mechanism.
The mechanism is above: a dropped pydantic field on the wrong route. Any past
result produced by `decode_probe.py` on a thinking-model prompt where the model
could naturally EOS is suspect for exactly this reason.

### 1.2 Why a new script

Neither existing probe can produce this anchor:

- `bench/decode_probe.py` — hits the non-EOS-banning route (§1.1), records **no
  usage** at all, and captures no per-token latencies.
- `bench/pertoken_latency_probe.py` — has the right measurement methodology
  (usage capture, TTFT excluded, full inter-token gap distribution, explicitly
  refuses to present a lone mean) but also posts to `/v1`, so a thinking model
  can EOS out and collapse the window.
- `bench/decode_depth_sweep.py` — correct depth/nonce/needle discipline, but
  reads decode cost from runner-log `[LAYER_PHASE]` lines via ssh and defaults
  to `max_tokens=10`; not an end-to-end >= 60s decode-window instrument, and I
  was not going to ssh-poke the live studios.

So `bench/p3_depth_anchor_probe.py` = `pertoken_latency_probe.py`'s measurement
methodology pointed at the endpoint that actually bans EOS. It is read-only
w.r.t. the cluster: one HTTP POST per depth point, nothing else. Inherited
guardrails, all of them deliberate:

- **Depth is read back from `usage.prompt_tokens`**, never from the nominal
  label. Locally the prompt is binary-searched to the target with the real model
  tokenizer (`~/.exo/models/deepseek-ai--DeepSeek-V4-Flash`, chat-template
  overhead measured live at 4 tokens); the local number is a sanity check only.
  It landed within 4 tokens of the API's number at every depth.
- **Unique UUID nonce at the FRONT of every prompt** + `use_prefix_cache=False`,
  so a KV prefix-cache hit cannot silently turn a deep measurement into a
  shallow one. Confirmed clean: `prompt_tokens_details.cached_tokens = 0` on all
  three runs.
- **Non-degenerate filler** (`decode_depth_sweep._filler`'s 8 rotating topics
  with varying numeric fields), not one repeated sentence — a degenerate prompt
  exercises different attention/routing behaviour.
- **Decode window = last streamed event − first streamed event.** TTFT/prefill
  is outside the window by construction, so the reported tok/s is decode-only.
- **A hard `--depth-cap` of 355,000** that raises `SystemExit` before any
  request is issued. The deep point was targeted at exactly 352,595 (the prior
  validated footprint) and landed at 352,599 real tokens.

### 1.3 Two tok/s numbers, and which one to use

The probe reports both `decode_tok_s_usage` (from `usage.completion_tokens`)
and `decode_tok_s_events` (from counted SSE events). **The usage-based number
is the anchor.** Streamed events are not 1:1 with tokens — the server can
coalesce, and 1844 events for 2000 tokens at short context shows it does. At
100K and 352K they agree to within 0.6% (1988 and 1987 events vs 2000 tokens),
so the distinction only matters materially at short context, where the
event-based number (27.32) understates the real 29.63.

---

## 2. Raw probe output

### 2.1 Short-context control (520 real tokens)

```
probing http://adams-mac-studio-m4-1.local:52415/bench/chat/completions model=deepseek-ai/DeepSeek-V4-Flash-0731 max_tokens=2000

building depth target ~512 tokens: 2,719 chars, locally predicted 520 tokens

=== depth target ~512 tokens ===
  target_tokens:        512 (locally predicted 520)
  REAL prompt_tokens:   520
  completion_tokens:    2000
  finish_reason:        length  (must be 'length' -- EOS banned)
  streamed events:      1844
  TTFT (prefill):       3.37s
  DECODE WINDOW:        67.47s
  total wall clock:     70.84s
  decode tok/s (usage): 29.63  -> 33.75 ms/tok
  decode tok/s (events):27.32  -> 36.61 ms/tok
  full usage:           {'prompt_tokens': 520, 'completion_tokens': 2000, 'total_tokens': 2520, 'prompt_tokens_details': {'cached_tokens': 0, 'audio_tokens': 0}, 'completion_tokens_details': {'reasoning_tokens': 0, 'audio_tokens': 0, 'accepted_prediction_tokens': 0, 'rejected_prediction_tokens': 0}}
  text head:            '1.  **Analyze the Request**:\n    *   Reference identifier: 5d46df0516a04db0a1c2637f3cfbde9b (ignore, just a reference).\n    *   Corpus: 11 sections (0 to 10).\n    *   Task: Briefly summarise the corpu'
  text tail:            'ements by 3 modulo 13). Overall, it is a highly structured, synthetic set of statements linking system behaviours to num'
  inter-token gap distribution (n=1843):
    min        0.01 ms
    p10       18.02 ms
    p50       31.85 ms
    p90       61.11 ms
    p99       96.74 ms
    max     2111.30 ms
    mean      36.61 ms
    stdev     55.55 ms
  implied steady-state from p50 gap: 31.39 tok/s
  gaps > 3x median: 23 / 1843 (1.25%)
  slow-gap values (ms, first 20): [2111.3, 904.8, 132.8, 123.9, 120.4, 116.4, 116.3, 113.5, 113.3, 106.4, 106.2, 105.7, 104.2, 103.4, 100.0, 99.6, 98.1, 97.7, 96.7, 96.4]
  first 25 gaps (ms): [29.6, 66.3, 23.5, 34.1, 28.9, 30.0, 32.1, 29.8, 37.3, 26.7, 29.7, 31.2, 28.7, 72.1, 25.0, 27.4, 35.3, 23.9, 31.1, 59.2, 0.1, 42.6, 103.4, 0.0, 0.0]
  last 25 gaps (ms):  [1.0, 29.2, 54.1, 22.2, 95.7, 0.0, 13.2, 40.1, 42.3, 18.6, 47.3, 59.0, 40.1, 23.5, 44.7, 34.1, 29.0, 37.1, 38.6, 32.9, 30.2, 40.1, 904.8, 33.7, 120.4]
```

### 2.2 100K anchor (100,026 real tokens)

```
building depth target ~100,000 tokens: 544,306 chars, locally predicted 100,026 tokens

=== depth target ~100,000 tokens ===
  target_tokens:        100,000 (locally predicted 100,026)
  REAL prompt_tokens:   100026
  completion_tokens:    2000
  finish_reason:        length  (must be 'length' -- EOS banned)
  streamed events:      1988
  TTFT (prefill):       273.71s
  DECODE WINDOW:        71.55s
  total wall clock:     345.26s
  decode tok/s (usage): 27.94  -> 35.79 ms/tok
  decode tok/s (events):27.77  -> 36.01 ms/tok
  full usage:           {'prompt_tokens': 100026, 'completion_tokens': 2000, 'total_tokens': 102026, 'prompt_tokens_details': {'cached_tokens': 0, 'audio_tokens': 0}, 'completion_tokens_details': {'reasoning_tokens': 0, 'audio_tokens': 0, 'accepted_prediction_tokens': 0, 'rejected_prediction_tokens': 0}}
  text head:            'We need to summarize the corpus. The corpus is a long list of sections, each with a template sentence: "In practice [topic] ...; the resulting behaviour depends on configuration [number] and on the ob'
  text tail:            'n number and on the interaction between two numbered stages of a surrounding system. The topics cycle in a fixed order, '
  inter-token gap distribution (n=1987):
    min        0.01 ms
    p10       22.77 ms
    p50       34.28 ms
    p90       48.59 ms
    p99       93.64 ms
    max      138.23 ms
    mean      36.01 ms
    stdev     16.28 ms
  implied steady-state from p50 gap: 29.17 tok/s
  gaps > 3x median: 11 / 1987 (0.55%)
  slow-gap values (ms, first 20): [138.2, 119.6, 114.6, 109.6, 106.7, 105.4, 104.6, 104.0, 103.6, 103.0, 102.9]
  first 25 gaps (ms): [68.6, 3.8, 33.5, 34.0, 43.2, 29.9, 33.3, 34.0, 46.9, 34.0, 38.5, 28.1, 42.0, 33.2, 33.4, 33.8, 41.4, 34.4, 34.7, 37.0, 35.0, 33.7, 33.4, 35.2, 43.4]
  last 25 gaps (ms):  [37.6, 28.7, 90.0, 0.1, 19.0, 33.1, 39.6, 33.7, 63.9, 3.8, 38.6, 33.7, 33.1, 33.5, 67.6, 32.2, 32.7, 32.8, 48.2, 46.1, 22.6, 32.5, 43.1, 30.9, 102.9]
```

### 2.3 Deep anchor (352,599 real tokens — the depth prior benches labeled "500K")

```
building depth target ~352,595 tokens: 1,908,184 chars, locally predicted 352,599 tokens

=== depth target ~352,595 tokens ===
  target_tokens:        352,595 (locally predicted 352,599)
  REAL prompt_tokens:   352599
  completion_tokens:    2000
  finish_reason:        length  (must be 'length' -- EOS banned)
  streamed events:      1987
  TTFT (prefill):       1058.62s
  DECODE WINDOW:        85.13s
  total wall clock:     1143.75s
  decode tok/s (usage): 23.48  -> 42.59 ms/tok
  decode tok/s (events):23.33  -> 42.86 ms/tok
  full usage:           {'prompt_tokens': 352599, 'completion_tokens': 2000, 'total_tokens': 354599, 'prompt_tokens_details': {'cached_tokens': 0, 'audio_tokens': 0}, 'completion_tokens_details': {'reasoning_tokens': 0, 'audio_tokens': 0, 'accepted_prediction_tokens': 0, 'rejected_prediction_tokens': 0}}
  text head:            'The user wants a brief summary of the corpus. The corpus is a long list of sections (0 to 7922) that follow a repetitive pattern. Each section describes a practice (e.g., "distributed inference schedu'
  text tail:            'ic or generated data.</think>The corpus is a long, repetitive collection of templated statements (sections 0–7922) descr'
  inter-token gap distribution (n=1986):
    min        0.01 ms
    p10       26.43 ms
    p50       39.16 ms
    p90       61.94 ms
    p99      113.11 ms
    max      197.49 ms
    mean      42.86 ms
    stdev     19.62 ms
  implied steady-state from p50 gap: 25.54 tok/s
  gaps > 3x median: 15 / 1986 (0.76%)
  slow-gap values (ms, first 20): [197.5, 158.1, 154.5, 140.6, 137.6, 136.7, 130.9, 129.8, 127.9, 127.1, 122.6, 121.9, 119.0, 118.6, 118.0]
  first 25 gaps (ms): [35.5, 41.6, 35.0, 51.9, 41.3, 38.5, 42.4, 44.9, 41.3, 38.2, 48.8, 39.7, 40.1, 37.6, 39.0, 52.3, 39.6, 39.6, 35.5, 54.8, 118.0, 0.0, 53.8, 41.0, 32.5]
  last 25 gaps (ms):  [22.4, 38.8, 51.2, 38.9, 38.8, 38.0, 52.6, 38.0, 76.2, 51.0, 38.9, 40.2, 37.2, 50.8, 39.7, 39.3, 42.6, 44.8, 38.9, 39.1, 37.7, 51.0, 38.7, 39.1, 108.0]
```

---

## 3. Per-token latency distribution — the depth penalty is UNIFORM, not bursty

This is the part that constrains a later decomposition phase, so state it
precisely. Comparing the three distributions (all ms):

| statistic | 520 tok | 100,026 tok | 352,599 tok | deep − short |
|---|---|---|---|---|
| p10 | 18.02 | 22.77 | 26.43 | +8.41 |
| **p50** | **31.85** | **34.28** | **39.16** | **+7.31** |
| p90 | 61.11 | 48.59 | 61.94 | +0.83 |
| p99 | 96.74 | 93.64 | 113.11 | +16.37 |
| max | 2111.30 | 138.23 | 197.49 | −1913.81 |
| mean | 36.61 | 36.01 | 42.86 | +6.25 |
| stdev | 55.55 | 16.28 | 19.62 | −35.93 |
| gaps > 3× median | 1.25% | 0.55% | 0.76% | — |

**The depth slowdown is a uniform per-token shift, not a stall/burst
phenomenon.** Evidence:

1. **The whole distribution translates rightward.** p10 +8.4ms, p50 +7.3ms,
   mean +6.3ms — the fast tokens slow down by about as much as the median does.
   A bursty mechanism would leave p10 flat and inflate only the upper tail.
2. **The tail does not fatten with depth.** Outlier rate is 1.25% at short ctx
   and *lower* (0.55%, 0.76%) at 100K and 352K. The single 2111ms and 904ms
   spikes are at SHORT context, not deep.
3. **Dispersion does not grow with depth.** stdev falls 55.55 → 16.28 → 19.62.
   The deep run is the *more* metronomic one.
4. **p50-implied throughput tracks the measured anchor.** 31.39 / 29.17 / 25.54
   tok/s from the p50 gap alone, monotone in depth and within ~9% of the
   usage-derived numbers.

This **independently corroborates T5's conclusion by a completely different
instrument**. T5 (a two-rank `xctrace` GPU-occupancy capture) concluded the
context-scaling drop is "straightforward increased real per-token compute cost
from larger KV/pooled attention shapes at depth", not an idle-time problem —
and did so on a ~9s window it flagged as too short. This is a 60-85s
client-side wall-clock capture with EOS genuinely banned, and it reaches the
same shape conclusion. T5's caveat is now discharged.

**For the later decomposition phase**: the target is a ~+8.8 ms/token uniform
cost that grows with depth, spread across essentially every token. There is no
bursty subset of "slow tokens" to isolate and attribute — a mechanism proposed
to explain this decay must explain a shift of the entire distribution
including its fast decile.

---

## 4. Comparison vs T1 anchors (2026-08-22, post-async-fence-fix)

| depth | T1 tok/s | T1 ms/tok | B1 real depth | B1 tok/s | B1 ms/tok | Δ tok/s |
|---|---|---|---|---|---|---|
| short (512-2000 tok prompt) | 29.2–31.1 | 32.15–34.25 | 520 | 29.63 | 33.75 | in range |
| 100K | 26.91 | 37.16 | 100,026 | 27.94 | 35.79 | **+3.8%** |
| 300K | 24.44 | 40.92 | — | not run | — | — |
| "500K" = 352,595 real | 21.51 | 46.49 | 352,599 | 23.48 | 42.59 | **+9.2%** |

**Short: reproduced.** 29.63 sits inside T1's 29.2–31.1 band and matches
today's independently re-verified 30.29.

**100K: reproduced within noise.** +3.8% on a single sample, against a metric
whose own short-ctx band spans ±3.2%, is not a distinguishable difference.

**Deep: +9.2%, which I do NOT claim as an improvement.** Honest read — the most
likely explanation is methodological, in my favour on window length:

- T1's deep number was measured under the same `bench` routing (§1.1) that
  produced T5's ~9s window. A short window at depth is dominated by the first
  tokens after prefill, which is exactly where warmup/allocator effects land.
  My window is 85s and 2000 tokens; the first-25-gaps list shows the deep run
  is already at steady state by token ~2, so a short window would sample a
  *different*, likely slower, region.
- Prompt content differs. T1's deep prompt was a needle prompt; mine is
  8-topic rotating filler. Both are non-degenerate, but MoE expert-routing
  distribution is content-dependent and could plausibly move decode cost by a
  few percent at fixed depth.
- Both are n=1 at this depth. I ran one deep point (each costs ~19 minutes,
  ~17.6 of it prefill) rather than burn an hour of live-cluster time on
  repetitions the task did not ask for.

I did not re-run T1's exact prompt at depth, so I cannot separate these. The
defensible statement is: **the decay SHAPE reproduces (monotone decrease with
depth, ~21% short→deep), the deep MAGNITUDE is ~9% higher than T1's under a
window 9x longer with EOS verifiably banned.** Treat B1's number as the better
anchor and T1's deep point as likely slightly pessimistic, but note the
confound rather than declaring a win.

### The decay itself, as measured today

| segment | Δ ms/token | Δ tok/s |
|---|---|---|
| 520 → 100,026 (+99.5K tokens) | +2.04 | −1.69 |
| 100,026 → 352,599 (+252.6K tokens) | +6.80 | −4.46 |
| 520 → 352,599 (total) | **+8.84** | **−6.15 (−20.8%)** |

Per 100K tokens of context, the cost is **+2.05 ms/token** over the first 100K
and **+2.69 ms/token** over the next 253K — i.e. the marginal per-token cost of
depth is mildly *super*-linear in this range, not saturating.

---

## 5. Prefill, incidentally (not the deliverable, but recorded)

TTFT is measured to the first streamed event, so it includes queueing/scheduling
overhead as well as pure prefill — treat these as a lower bound on end-to-end
time-to-first-token, not as a clean prefill-kernel number.

| depth | TTFT | implied tok/s | T1 prefill reference |
|---|---|---|---|
| 520 | 3.37 s | 154.5 | — (dominated by fixed overhead at this size) |
| 100,026 | 273.71 s | 365.4 | 359.7 |
| 352,599 | 1058.62 s | 333.1 | 324.1 |

Both deep points land slightly above T1's prefill figures, consistent with T1's
prefill numbers being at parity and normal run-to-run variance.

---

## 6. Honest limitations

- **n=1 per depth.** The task asked for two clean anchors with long windows, and
  each deep point costs ~19 minutes of live cluster time. Per-depth medians over
  n>=3 (the `decode_depth_sweep.py` standard) were not affordable here. The
  within-run gap distributions (n≈1990 gaps each) give confidence in the
  *within-run* stability, not in run-to-run variance.
- **300K was not run.** Not requested; the task specified 100K + the deep point.
  That leaves T1's 24.44 tok/s @ 300K unrefreshed, so the curve has a 253K-wide
  gap between my two deep-ish points.
- **The +9.2% deep delta is unattributed.** See §4 — window length, prompt
  content, and n=1 are all live confounds and I did not run the experiment that
  would separate them.
- **TTFT includes non-prefill overhead**, so §5 is indicative only.
- **A new bench script was added** (`bench/p3_depth_anchor_probe.py`). It is
  additive; no existing script was modified. `bench/decode_probe.py` still has
  the §1.1 bug — I deliberately did not fix it, since patching a shared bench
  script mid-investigation while a parallel worker may be using it is exactly
  the kind of change that invalidates someone else's in-flight measurement.
  **Recommend a follow-up to repoint `decode_probe.py` at
  `/bench/chat/completions`, and to re-examine any past result that depended on
  it.**
- **Nothing was committed to git.**

---

## 7. Files

- `bench/p3_depth_anchor_probe.py` — new, additive probe (read-only vs cluster).
- `/tmp/p3b1_short.json`, `/tmp/p3b1_100k.json`, `/tmp/p3b1_deep.json` — raw
  per-run JSON including every inter-token gap.
- `/tmp/p3b1_short.log`, `/tmp/p3b1_100k.log`, `/tmp/p3b1_deep.log` — raw stdout
  reproduced verbatim in §2.
