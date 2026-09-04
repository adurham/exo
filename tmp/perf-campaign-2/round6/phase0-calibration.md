# Phase 0 — Mechanical Calibration Gate (Campaign 2, Round 6)

Run on the **current production boot**. No relaunch performed. No sweep arms run.

## Verdict: **PASS**

Both probes land inside their pre-registered bands. The server-side
`stats.generation_tps` measurement path is sane and trusted for the round.

| probe | server `generation_tps` | band | result |
|---|---|---|---|
| 2K prompt | **29.06 t/s** | 20.6 – 32.6 | PASS |
| 89K prompt (achieved 143,964 tok) | **32.82 t/s** | 24.0 – 40.0 | PASS |

---

## 1. Cluster state verification (unchanged, no relaunch)

### Node .201 (192.168.86.201) — real runner PID
```
$ ssh adam.durham@192.168.86.201 'pgrep -af "exo -v"'
31596  SCREEN -dmS exorun ...
31597  login -pflq adam.durham /bin/zsh -l -c ...
31598  zsh -l -c ...
31608  .venv/bin/python -m exo -v      <- real runner
```
`ps eww 31608` env (filtered):
```
EXO_SPECULATIVE_GAMMA=3
EXO_DSV4_MTP=1
EXO_DSV4_DSPARK=1
```
`EXO_DSV4_MTP_LOG_INTERVAL` — **not present** (unset).
`EXO_DSV4_MTP_PROFILE` — **not present** (unset).

### Node .202 (192.168.86.202) — real runner PID
```
$ ssh adam.durham@192.168.86.202 'pgrep -af "exo -v"'
39802  SCREEN -dmS exorun ...
39803  login -pflq adam.durham /bin/zsh -l -c ...
39804  zsh -l -c ...
39813  .venv/bin/python -m exo -v      <- real runner
```
`ps eww 39813` env (filtered):
```
EXO_SPECULATIVE_GAMMA=3
EXO_DSV4_MTP=1
EXO_DSV4_DSPARK=1
```
`EXO_DSV4_MTP_LOG_INTERVAL` — **not present** (unset).
`EXO_DSV4_MTP_PROFILE` — **not present** (unset).

**Result: matches expectation exactly.** gamma=3 on both nodes, LOG_INTERVAL
unset, PROFILE unset — no confound from the round-5 diagnostic env is present
on this boot.

### API health
```
$ curl -s -o /dev/null -w "HTTP %{http_code}\n" http://192.168.86.201:52415/v1/models
HTTP 200
```
Model list includes `deepseek-ai/DeepSeek-V4-Flash-0731` (confirmed via
successful probe completions below, which named this model and returned
valid completions).

---

## 2. Server-side throughput field (proof of measurement path)

`src/exo/worker/engines/mlx/generator/batch_generate.py`:

```python
648:    generation_start_time: float = 0.0
649:    generation_time_at_start: float = 0.0
...
1255:            generation_start_time=time.perf_counter(),
1257:            generation_time_at_start=_mlx_gen_elapsed_seconds(self._mlx_gen),
```

Computed inside the generator at completion time (non-pp-spec path):

```python
4567:                else:
4568:                    gen_time_delta = (
4569:                        _mlx_gen_elapsed_seconds(self._mlx_gen)
4570:                        - state.generation_time_at_start
4571:                    )
4572:                    generation_tps = (
4573:                        state.completion_tokens / gen_time_delta
4574:                        if gen_time_delta > 0
4575:                        else 0.0
4576:                    )
...
4597:                stats = GenerationStats(
4598:                    prompt_tps=state.prefill_tps,
4599:                    generation_tps=generation_tps,
4600:                    prompt_tokens=len(state.all_prompt_tokens),
4601:                    generation_tokens=state.completion_tokens,
```

`GenerationStats` (containing `generation_tps`) rides on `TokenChunk.stats`
(`src/exo/shared/types/chunks.py:29`), which is set only when
`is_done`/`chunk.finish_reason is not None`.

**Streaming vs non-streaming:** `stats` IS present on the streaming path, but
NOT inside the `data:` JSON chunks. `generate_chat_stream()`
(`src/exo/api/adapters/chat_completions.py:229-296`) emits it as an **SSE
comment line** on the final chunk:

```python
293:                if chunk.finish_reason is not None:
294:                    if chunk.stats is not None:
295:                        yield f": generation_stats {chunk.stats.model_dump_json()}\n\n"
296:                    yield "data: [DONE]\n\n"
297:                    return
```

Because SSE comment lines (`: ...`) are conventionally ignored by clients,
the original probe (which only parsed `data: ` lines) never captured it.
No fallback to non-streaming was needed — the stats object IS obtainable on
the streaming path, just via a different line prefix.

---

## 3. Probe edit (the one permitted change)

`bench/long_decode_probe.py` — diff:

```diff
@@ -94,11 +94,24 @@
     reasoning_parts: list[str] = []
     usage: dict[str, object] = {}
     finish_reason = None
+    # Server-side stats (perf_counter-timed INSIDE the generator at
+    # batch_generate.py:1255-1257/4559-4576), emitted on the streaming
+    # path as an SSE comment line (": generation_stats {...}") rather
+    # than inside a "data: " chunk -- see chat_completions.py
+    # generate_chat_stream(). This is the ONLY trusted throughput
+    # number; client-side decode_tps below remains a cross-check only.
+    server_stats: dict[str, object] | None = None
 
     with httpx.Client(timeout=1800.0) as client:
         with client.stream("POST", f"{API}/v1/chat/completions", json=body) as r:
             r.raise_for_status()
             for line in r.iter_lines():
+                if line and line.startswith(": generation_stats "):
+                    try:
+                        server_stats = json.loads(line[len(": generation_stats "):])
+                    except json.JSONDecodeError:
+                        pass
+                    continue
                 if not line or not line.startswith("data: "):
                     continue
                 payload = line[6:]
@@ -164,6 +164,13 @@
         # honesty flag: the standing rule is that t/s from a short
         # generation is startup noise, not a throughput measurement.
         "decode_sample_trustworthy": bool(ctok and ctok >= 400),
+        # Server-side, perf_counter-timed decode throughput -- the
+        # trusted number. decode_tps above is a client-side cross-check
+        # only, never the decision input (round-6 phase-0 rule).
+        "server_stats": server_stats,
+        "server_generation_tps": (
+            server_stats.get("generation_tps") if server_stats else None
+        ),
         "finish_reason": finish_reason,
```

No timing arithmetic added. No request-issuing change (streaming kept — the
fallback clause was not triggered because `stats` IS present on the stream).
No prompt-construction change. Existing client-side `decode_tps` retained
as-is, used only as a cross-check below.

---

## 4. Probe runs (current boot, no relaunch)

Both probes were run directly against `http://192.168.86.201:52415` from the
laptop (no cluster-side execution needed — the probe is a plain HTTP client).
No relaunch of the cluster occurred; both nodes' runner PIDs (31608, 39813)
were unchanged throughout.

### a) SHORT (~2K target)
Argument: `2000` → achieved `prompt_tokens = 2262` (>= ~2K, as intended).
`--max-tokens 1500` (well above the 400-token trustworthy threshold).

| rep | server `generation_tps` | prompt_tokens | completion_tokens | trustworthy | finish_reason | client `decode_tps` (cross-check) |
|---|---|---|---|---|---|---|
| 1 | **29.058** | 2262 | 1500 | true | length | 29.15 |

Raw JSON: `tmp/perf-campaign-2/round6/results/p0_2k.json`

### b) DEEP (~89K target, iterated argument)
Only one iteration was needed. Argument `128000` (chosen up front,
anticipating round 5's undershoot — round 5's `--depth 89000` produced only
62K actual, i.e. ~70% of target) achieved:

**Depth argument that hit >=85K: `128000` → achieved `prompt_tokens = 143,964`.**

(Overshoot vs the 89K nominal target; well clear of the >=85,000 floor, so no
further iteration was required. Note for the next phase: the ~4 chars/token
heuristic is running well below 1:1 at this scale — 128,000 requested →
143,964 actual, i.e. the achieved/argument ratio is >1.0 here, the opposite
direction from round 5's undershoot at a different scale. Argument-to-actual
is not linear/stable across depths; recalibrate per depth target, don't reuse
a fixed multiplier.)

`--max-tokens 1200` (clears the 400-token trustworthy floor with margin).

| rep | server `generation_tps` | prompt_tokens | completion_tokens | trustworthy | finish_reason | client `decode_tps` (cross-check) |
|---|---|---|---|---|---|---|
| 1 | **32.816** | 143,964 | 1200 | true | length | 32.88 |

Raw JSON: `tmp/perf-campaign-2/round6/results/p0_deep_r1.json`

Prefill sanity check: 346.44s for 143,964 prompt tokens = 415.6 tok/s server
prefill_tps — consistent with the cluster's known prefill ceiling (~350-420
tok/s range), not a suspiciously fast prefix-cache-hit number like round 5's
misread 3s TTFT on a 62K prompt.

---

## 5. Gate application (pre-registered, not renegotiated)

| probe | measured `generation_tps` | pre-registered band | verdict |
|---|---|---|---|
| 2K | 29.058 | 20.6 – 32.6 | **PASS** (inside band) |
| 89K (143,964 achieved) | 32.816 | 24.0 – 40.0 | **PASS** (inside band) |

Both numbers land inside their bands, server-side `generation_tps` is
physically sane relative to the cluster's known 20-35 t/s decode envelope
(no 300+ t/s burst artifact, no chunk-rate mislabeling), and the client-side
`decode_tps` cross-check is consistent with the server number in both runs
(29.15 vs 29.058; 32.88 vs 32.816 — agreement within noise, as expected when
the client wall-clock window and the server's internal decode window
approximately coincide for a single non-concurrent request).

## GATE VERDICT: **PASS**

The measurement path is sane. Phase 1 (sweep arms) may proceed using
`bench/long_decode_probe.py` as edited here, with **depth argument 128000**
as the calibrated value for the ~89K-token deep probe target for this boot.

No sweep arms were run. No cluster relaunch occurred. No git commit was made.
