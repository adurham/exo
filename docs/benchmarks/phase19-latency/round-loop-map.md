# DSv4.1 MTP speculative-decode round loop — mechanism + file:line map

**Read at commit:** exo `deploy/next13` = `f0840af1c50ecfdc85236a68b60a4c341805f8ce`;
mlx-lm submodule `main` = `6cc9c1e8709e228fca99ac152cd5e681ddcce65d`
(pinned in `uv.lock:1501`, installed into `.venv` as `mlx_lm-0.31.3.dist-info`).

**Read-only.** No cluster contact, no tests, no source edits. One file written, this one.

---

## 0. HEADLINE: the cluster does NOT run the MTP-PROF path

The MTP-PROF per-cycle timer (`draft/verify/accept/rollback`) lives in
`src/exo/worker/engines/mlx/speculative/dsv4_mtp.py` inside
`DSv4MTPBatchGenerator._speculative_next` / `_speculative_next_batch`
(`dsv4_mtp.py:3844`, `dsv4_mtp.py:2589`). That class is the **generic MLX
BatchGenerator** speculative path. It is selected by the *generic* `MlxBuilder`,
not by the DSv4.1 engine.

The DSv4.1-Flash EXL3 card on this cluster declares `engine = "dsv41"`
(`resources/inference_model_cards/dealignai--DeepSeek-V4.1-Flash-UNCENSORED-EXL3-2.9bpw.toml:56`), and dispatch sends it to `Dsv41Builder`
(`src/exo/worker/engines/mlx/dsv41/dispatch.py:51-58`). The DSv4.1 engine runs its
own round loop in `src/exo/worker/engines/mlx/dsv41/rounds.py` — it never
imports `dsv4_mtp.py` (grep: only `model_output_parsers.py` and
`generator/batch_generate.py` import it).

Consequences up front, each expanded below:

- **The brackets the user's MTP-PROF numbers come from wrap code the DSv4.1
  engine does not execute.** `[MTP-PROF]` never appears in a DSv4.1-engine
  runner log. The 123.5 ms client-visible round is `rounds._one_round`
  (`rounds.py:328`).
- **The DSv4.1 round has exactly ONE `mx.eval` sync and NO per-phase brackets**
  (`rounds.py:429`, `rounds.py:318` in `spec.generate`). There is no profiler on
  this path at all.
- **The adaptive-gamma policy that actually runs** is
  `mlx_lm.models.deepseek_v41.spec.GammaPolicy` (`spec.py:80`), invoked every
  round at `rounds.py:402` via `_spec_policy` (`rounds.py:291`), and it is **ON
  by default** (`engine.py:281 adaptive_gamma: bool = True`) and evaluates
  candidate gammas up to **4** (`spec.py:87`).
- **`VERIFY_MS` is the literal dict the phase-17 campaign produced**, hardcoded
  into that policy (`spec.py:77`).

---

## 1. THE SPEC ROUND LOOP

### 1a. The DSv4.1 engine round loop (what the cluster runs)

`Dsv41Engine._rounds` is the outer cycle loop
(`src/exo/worker/engines/mlx/dsv41/engine.py:971`):

- One **cycle start** = the top of the `while n < max_tokens:` body,
  `engine.py:989`.
- One **cycle end** = `yield batch, lps` (`engine.py:1019`) — the generator
  suspends there and the runner drains it.
- Per cycle it calls `_one_round(...)` (`engine.py:993`), which returns
  `(committed_tokens, round_ms, accepted, gamma)`.
- Spec eligibility: `head = self.loaded.head if self.speculative else None`
  (`engine.py:981`); the policy is built once per request
  (`engine.py:982-986`).

`_one_round` is the actual **round** (`src/exo/worker/engines/mlx/dsv41/rounds.py:328`).
Inside it, mapped to the requested bracket names:

| bracket | file:line | exact expression wrapped |
|---|---|---|
| **draft** | `rounds.py:408-414` | `drafted = head.draft(anchor.reshape(-1), model.embed, model.head, draft_state, width=gamma)` |
| **verify** | `rounds.py:419-429` | `verify_in = mx.concatenate([anchor.reshape(1,1), drafted.reshape(1,gamma)], axis=1)` → `logits, taps = model(verify_in, cache, return_taps=True, argmax=True)` → `mx.eval(logits)` |
| **accept** | `rounds.py:431-447` | readback `target=[int(v) for v in logits[0]]`, `draft=[int(v) for v in drafted[0]]`, then the first-mismatch loop `while accepted < gamma and accepted < len(target) and target[accepted]==draft[accepted]: accepted += 1`, then `committed = draft[:accepted] + [target[min(accepted, len(target)-1)]]` |
| **rollback** | `rounds.py:422`, `rounds.py:453` | snapshot: `snapshot = spec.snap(cache, position)`; rollback: `spec.rollback(cache, snapshot, committed_position, stashes)` with `stashes = spec.stashes(cache)` (`rounds.py:430`) |
| round wall | `rounds.py:362` / `rounds.py:457` | `started = time.perf_counter()` … `(time.perf_counter() - started) * 1e3` |
| capacity (unbracketed tail) | `rounds.py:406` | `_ensure_capacity_for_round(cache, 1 + gamma)`, i.e. `cache.ensure_capacity(int(cache.offset) + rows)` (`rounds.py:320`) — can **reallocate** the compressed/index/engram buffers per round |

The **only GPU sync in the round** is `mx.eval(logits)` at `rounds.py:429`
(comment at `rounds.py:429`: "the one sync"). `logits`' dependency graph reaches
`drafted` (via `verify_in`), so the draft forward is materialized here too; the
readbacks at `rounds.py:431-432` are therefore host-side and free. There is **no
`mx.eval` after the rollback and none after `head.append_ctx`** — that work is
left lazy and billed to the *next* round's `mx.eval(logits)`.

There is also a **greedy-only** branch with no head: `rounds.py:364-370`
(`logits = model(anchor, cache, last_logit_only=True)`, then
`int(mx.argmax(logits...).item())` at `rounds.py:368`) and the **first-round
warm-up** with a head but no draft window yet (`rounds.py:385-400`,
`head.append_ctx(tapcat(taps), draft_state)` at `rounds.py:393`).

`spec.generate` (`mlx-lm/mlx_lm/models/deepseek_v41/spec.py:149`) is the
harness/`serve.py` twin of the same loop, structurally identical; its round sync
is `mx.eval(am, d)` at `spec.py:318` and its rollback at `spec.py:340`. The
engine copies this structure deliberately (module docstring `rounds.py:15-19`).

### 1b. The MTP-PROF brackets (generic path — NOT the cluster's engine)

The emitter is `_PhaseTimer.dump` (`src/exo/worker/engines/mlx/speculative/dsv4_mtp.py:815`),
which emits `[MTP-PROF] cycles=…` (`dsv4_mtp.py:817`) and per-phase lines
(`dsv4_mtp.py:830`/`836`). `_PhaseTimer` is a module singleton built when
`EXO_DSV4_MTP_PROFILE > 0` (`dsv4_mtp.py:244`, `dsv4_mtp.py:842`).

Bracket call sites (all guarded by `if prof is not None:`):

**Single-uid path `_speculative_next` (`dsv4_mtp.py:3844`):**

| bracket | record | bracket open (fence) | source |
|---|---|---|---|
| draft | `dsv4_mtp.py:4091` | `mx.eval(*draft_ids)` at `dsv4_mtp.py:4089`, close `t_after_draft` `4090`; cycle start `t_cycle_start` `3901` | call `draft_tokens(...)` `4078-4086` (or DSpark `_dspark.draft` `4008`) |
| verify | `dsv4_mtp.py:4227` | `mx.eval(verify_pre_norm, verify_logits)` at `dsv4_mtp.py:4225`, close `t_after_verify` `4226` | `dsv4_speculative_forward(...)` `4166-4171`; snapshot/arm block between `4127`-`4161` (not separately bracketed unless RB_PROFILE) |
| accept | `dsv4_mtp.py:4944` | **NO mx.eval** — comment at `dsv4_mtp.py:4937-4941` explicitly says so; close `t_after_accept` `4943` | accept/bonus derivation ~`4229-4930` |
| rollback | `dsv4_mtp.py:5448` | **NO mx.eval** — comment at `dsv4_mtp.py:5441-5445`; close `t_after_rollback` `5447` | rollback body `~4960-5440`; sub-phases rb_snap/rb_gate/rb_drain/rb_ring/rb_pool/rb_commitfwd/rb_tail under `EXO_DSV4_RB_PROFILE` (`dsv4_mtp.py:4989-5074`), each fenced with its own `mx.synchronize()` (`5020`,`5042`,`5060`,`5073`) |
| total | `dsv4_mtp.py:5449` | `end_cycle(1)` `dsv4_mtp.py:5450` | |

`rb_snap` is recorded at `dsv4_mtp.py:4157` (single path), fenced by
`mx.synchronize()` at `dsv4_mtp.py:4127` and `4156`.

**Batched path `_speculative_next_batch` (`dsv4_mtp.py:2589`):** draft
`dsv4_mtp.py:4091`→`2669` (`mx.eval(*draft_ids_list)` `2667`); verify
`dsv4_mtp.py:2724` (`mx.eval(verify_pre_norm, verify_logits)` `2722`); accept
`dsv4_mtp.py:3102` (**no eval**, comment `3091-3099`); rollback
`dsv4_mtp.py:3427` (**no eval**, comment `3416-3424`); total `3428`;
`end_cycle(N)` `3429`.

The two "no mx.eval" comments are the load-bearing fact for the user's
attribution question: the accept and rollback brackets on the generic path close
**without a fence**, so anything lazily queued during them drains at the *next*
round's draft/verify `mx.eval` — billing their compute to the wrong bracket. The
comment says this exact mistake "produced a multi-round phantom during the perf
campaign" (`dsv4_mtp.py:3091-3099`, repeated `3416-3424`, `5437-5445`).

---

## 2. SYNC SPANS

**`EXO_PROFILER_SYNC_SPANS` — EXISTS.** Read at hook construction, i.e.
**per-hook-instantiation** (once per runner process, since `SpanProfilerHook()`
is built once at entrypoint):
`mlx-lm/mlx_lm/profiler.py:215` → `self._sync = os.environ.get("EXO_PROFILER_SYNC_SPANS", "").strip() in ("1","true","yes")`.
**DEFAULT: OFF (falsy).** When on, `span()` calls `mx.synchronize()` at BOTH
boundaries (`profiler.py:223-225` entry, `profiler.py:230-231` exit). Docstring:
"By default spans only bracket graph-build time … Set `EXO_PROFILER_SYNC_SPANS=1`
to `mx.synchronize()` at BOTH span boundaries" (`profiler.py:204-210`).

Important qualifier: the `SpanProfilerHook` is a **span-timing** hook for model
code (`profiler.span(...)` calls), *not* the `_PhaseTimer` bracket emitter. The
two are different instruments. There is **no `SYNC_SPANS` variant of the
MTP-PROF brackets**; the MTP bracket emitter has no sync toggle at all. The
closest analogue is `EXO_DSV4_RB_PROFILE` (below).

**`EXO_DSV4_MTP_PROFILE` — EXISTS.** `dsv4_mtp.py:244`:
`_PROFILE_INTERVAL = int(os.environ.get("EXO_DSV4_MTP_PROFILE", "0"))`.
**DEFAULT: 0 (off; `_phase_timer is None`, `dsv4_mtp.py:842`).** Read **once at
module import** (module-level statement, `dsv4_mtp.py:244`); toggling needs a
runner restart. Generic path only.

**`EXO_DSV4_RB_PROFILE` — EXISTS.** `dsv4_mtp.py:260`:
`_RB_PROFILE = os.environ.get("EXO_DSV4_RB_PROFILE", "0") == "1"`.
**DEFAULT: off.** Read once at import. **Requires `EXO_DSV4_MTP_PROFILE>0`**
(`dsv4_mtp.py:247`). Splits the rollback bracket with `mx.synchronize()`
sub-boundaries (`dsv4_mtp.py:4989-5074`). Generic path only.

**`EXO_DSV4_MTP_LOG_INTERVAL` — EXISTS.** `dsv4_mtp.py:118`:
`_LOG_INTERVAL = int(os.environ.get("EXO_DSV4_MTP_LOG_INTERVAL", "0"))`.
**DEFAULT: 0.** Read once at import. Gates the `[MTP] cycles=… mean_accept=…`
line (`dsv4_mtp.py:2168-2177`). Generic path only. (Note: the comment at
`dsv4_mtp.py:116` claims "when `EXO_DSV4_MTP_LOG=1`", but the code only reads
`EXO_DSV4_MTP_LOG_INTERVAL` — `EXO_DSV4_MTP_LOG` itself is read nowhere.)

**`EXO_DSV4_SECTION_TIME` — EXISTS** (related sync instrument, mlx-lm side).
`mlx-lm/mlx_lm/models/deepseek_v4.py:233`:
`_SECTION_TIME_ENABLED = bool(os.environ.get("EXO_DSV4_SECTION_TIME"))`.
**DEFAULT: off.** Read once at import; dumps on SIGUSR2 (`deepseek_v4.py:498`)
or every `EXO_DSV4_SECTION_TIME_LOG_EVERY` forwards (`deepseek_v4.py:234`,
`8072`). Docstring `deepseek_v4.py:226-231`: uses `mx.synchronize()` boundaries
so per-section SHARE is accurate; ~4 syncs/layer when on. This is the
"sync-span" instrument for a model forward.

**Does an EXO_DSV4_MTP_PROFILE sync variant exist?** No. `EXO_DSV4_MTP_PROFILE`
itself always inserts `mx.eval` at the draft and verify bracket boundaries
(`dsv4_mtp.py:4089`, `4225`) but **not** at accept/rollback (`4937`, `5441`).
`EXO_DSV4_RB_PROFILE` adds sync sub-boundaries inside rollback only.

**`VERIFY_MS` — EXISTS, module constant (not an env flag).** See §5.

---

## 3. WHAT IS NOT BRACKETED (per-round code between the brackets)

### 3.1 On the cluster path (DSv4.1 engine) — there are no brackets, so scope this to "what runs outside the single `mx.eval`"

Per-round, between the round's sync at `rounds.py:429` and the next round's sync
at the *next* `rounds.py:429`, `_rounds` runs:

1. **`session.maybe_checkpoint()`** — `engine.py:1018` → `session.py:753`.
   Cadence-gated (`self._spacing`, default 1024 rows,
   `session.py:112-113`). When it fires: `self._checkpoint()`
   (`session.py:802`) does `self.cache.snapshot()` (`session.py:804`), which
   runs `_spec.snap(self.cache, pos)` + `lc.ring_snapshot()` for every layer +
   `mx.eval([...rings...])` (`session_cache.py:367-374`), plus a **draft-window**
   snapshot that materializes `w.win_kv` via `mx.eval([...])`
   (`session.py:805-807`). **Classification: neither per-token nor strictly
   per-request — per (default) 1024 committed rows**, i.e. roughly once per
   ~340 rounds at 3 tokens/round. Still not inside any bracket.
2. **`self._check_cancel(...)`** — `engine.py:991` → `_check_cancel`
   `engine.py:1025` → `agreement.agree_on_cancellations_fast(cancel_receiver.collect())`.
   `collect()` is `MpReceiver.collect` (`channels.py:400`) = drain a
   multiprocessing queue until `WouldBlock`. **Classification: per-round**
   (`engine.py:991`). Host work only; no GPU sync.
3. **`session.maybe_checkpoint()`** — see 1.
4. **Yield / detokenize / SSE emission** — see 3.2.

There is **no per-round logging** on the DSv4.1 engine path: `logger.` in
`rounds.py` appears only at `rounds.py:395` (the once-per-request "draft window
primed" INFO). `session.py` logs at `333` (once per prefill), `1004` (per
store-`get`), `1045/1062/1072/1112/1120/1125` (store/park events). Nothing
per-round.

### 3.2 The sampler path

**GPU, greedy, host-light.** The DSv4.1 engine is greedy-only: `_resolve_sampler`
REFUSES any `temperature > 0` (`engine.py:1044-1050`) and the method is **never
called** (grep for `_resolve_sampler(` returns only the definition). There is no
`top_p` / `top_k` / temperature branch anywhere in `engine.py` or `rounds.py`
(grep returned only the field defaults at `engine.py:274-277`). Token selection is
`mx.argmax` inside the model forward, e.g. `rounds.py:368`
(`int(mx.argmax(logits.reshape(-1), axis=-1).item())`) and
`rounds.py:424/427` (`argmax=True` kwarg into `model(...)`). Because `argmax=True`
is passed, the model head emits argmax ids, **not** a full vocab logits tensor
(`mlx-lm/mlx_lm/models/deepseek_v41/spec.py:19-31` describes the argmax /
`combine_argmax` head path). **No full-vocab sort, no `.tolist()` of logits, no
`np.array` of logits in the round.**

The one host readback in the round is `rounds.py:431-432`
(`target = [int(v) for v in logits[0]]`, `draft = [int(v) for v in drafted[0]]`)
— these are length-`gamma` integer rows (the model returned argmax ids), so no
`.item()` and no sync. **Classification: per-round, but O(gamma) host ints.**

The only real GPU→host pulls are the optional-logprobs paths and the
`(temp>0)`-only sampling module:

- `_row_logprobs` — `rounds.py:270-278`: `mx.eval(sel, tid, tlp)` (`rounds.py:277`)
  then `float(sel[0].item())`, `tid[0].tolist()`, `tlp[0].tolist()` (`rounds.py:278`).
  Called for the anchor (`engine.py:791`) and per round only when logprobs were
  requested (`rounds.py:423-425`, `rounds.py:448-451`). **Classification:
  per-request (anchor) + per-round when `logprobs`/`top_logprobs` set; otherwise
  skipped.** The `k` is bounded by `lp_k = max(top_n, 1)` (`engine.py:789`), and
  `logprobs.from_logits` reduces the row to a `[lse, max, argmax, top_k]` summary
  (`mlx-lm/mlx_lm/models/deepseek_v41/logprobs.py:16-27`) — **no full-vocab sort on
  the hot path.** `_rows_logprobs` (`rounds.py:281-288`) does `mx.eval(sel, tid, tlp)`
  (`rounds.py:286`) + `.tolist()` (`rounds.py:287`).
- `sampling.py` (`mlx_lm/models/deepseek_v41/sampling.py`) is the temp>0 engine:
  `spec_generate` at `sampling.py:433`, the round sync at `sampling.py:486`
  (`dev.eval(lg, d, sampler.pool, *probe.qs)`), then **the full logits row is
  pulled host-side**: `p_rows = sampler.probs(dev.host(lg)[0])` (`sampling.py:487`)
  and `q_rows = [dev.host(q)[0] ...]` (`sampling.py:488`), where `dev.host` is
  `np.asarray` (`sampling.py:210-211`). That path is **NOT reachable from the
  serving engine** (`engine.py:1044` refuses temp>0; `spec.generate` raises for
  `cache=` on this path at `spec.py:232-237`). **Classification: per-round, but
  dead in production.**

### 3.3 The SSE / streaming emission path

**Not a socket write; a bounded in-process `mp.Queue` put, and it is NOT
awaited inside the round.** Chain:

- `_rounds` yields a whole round's batch (`engine.py:1019`).
- `_generate` consumes it token-by-token in `committed_tokens()`
  (`engine.py:793-798`), detokenizes (`engine.py:830-831`), and yields a
  `GenerationResponse` (`engine.py:856`).
- `dsv41_output_parser` wraps that in the thinking/DSML/chunk pipeline
  (`output.py:87-108`); `step()` pulls through it (`engine.py:419-429`) and
  returns a list of `(task_id, chunk)` (`engine.py:455`).
- The runner loop sends each: `results = self.generator.step()` (`runner.py:653`)
  → per result `self.send_chunk(other, ...)` (`runner.py:729`) →
  `runner.send_chunk` (`runner.py:877`) → `self.event_sender.send(ChunkGenerated(...))`
  (`runner.py:884`).
- `MpSender.send` (`channels.py:260-267`): tries non-blocking `put`, falls back
  to `buffer.put(item, block=True)` (`channels.py:267`). The buffer is a
  `multiprocessing.Queue`; `mp_channel(..., max_buffer_size=inf)` maps `inf → 0`
  in `MpState` (`channels.py:224-225`) and `mp.Queue(0)` is **unbounded**
  (`channels.py:231`). So the put essentially never blocks.

**Classification: per-emitted-token (so per-round yields `1+accepted` of them),
all outside `_one_round` and outside the round wall.** The round timing at
`rounds.py:457` closes before any of this. There is an optional
`MLX_GPU_TIME` probe that measures step-vs-send wall separately
(`runner.py:634-638`, `runner.py:779-792`) — **per-runner-loop-iteration**,
gated by `MLX_GPU_TIME`, default off.

### 3.4 `mx.eval` / `.item()` syncs in the un-bracketed tail

- `rounds.py:429` — the round's one sync (billed to the round). ✔ bracketed by
  round wall, not by any phase bracket.
- `session.py:393` — `mx.eval(handle, *(taps.values() if taps else []))`:
  **per prefill chunk** (prefill only).
- `session.py:807` — `mx.eval([kv for kv,_ in saved])`: **per checkpoint**
  (default every 1024 rows).
- `session_cache.py:371` — `mx.eval([...rings...])`: **per checkpoint**.
- `rounds.py:277`, `rounds.py:286` — **per round when logprobs requested**.
- `engine.py:724` (image `mx.eval(embeddings)`), `engine.py:968`
  (`int(mx.argmax(turn.anchor_logits...).item())`): **per-request / per-turn**.
- `channels.py` etc.: no GPU syncs.

---

## 4. REQUEST-LEVEL PHASE TIMESTAMPS

The six phase events, PRESENT/ABSENT. All on the DSv4.1 engine path.

| phase event | status | evidence |
|---|---|---|
| **request received** | **ABSENT as a timestamped log line.** The nearest signals are the admission INFO in the runner — `deferring task …` (`runner.py:832`) or dispatch — and the engine's `_active` activation via `_activate_next` (`engine.py:480`). None records a wall-clock timestamp for "request received". | `runner.py:832`, `engine.py:496-513` |
| **tokenization done** | **ABSENT.** `encode_prompt(tokenizer, prompt)` + `.tolist()` at `engine.py:730-734`; `prompt_len = int(prompt_tokens.shape[0])` `engine.py:731`. No log line, no timestamp. | `engine.py:730-734` |
| **prefix-cache match done (+matched/refed row counts)** | **PARTIAL / no explicit "done" timestamp.** Row counts ARE surfaced: `[DSV41] session reuse: this {len(ids)}-token prompt matches a resident conversation on {best_lcp} rows` (`session.py:1004-1007`, `session.py:1005-1006` fmt) and `[DSV41] turn reuse: {turn}` (`engine.py:957`, where `TurnOutcome.__str__` = `prompt=.. prefill=.. reuse=.. cache=..` at `session.py:449-450`), plus `[DSV41] reuse undershoot: refed={turn.prefill_tokens} rows (reused={turn.reused_tokens})` (`engine.py:964-967`). But the match itself is timed nowhere and there is no "match done at T" timestamp. | `session.py:1004-1007`, `engine.py:957`, `engine.py:964-967` |
| **prefill start** | **ABSENT as a start marker.** `[DSV41] prefill controls: fence_every=… transient_budget_mb=… score_row_bytes=… fence_hook=… (rows=…, base=…)` (`session.py:333-339`) fires once per `engine_prefill` call, before the loop (`session.py:349 t0 = time.perf_counter()`). It is a config dump, not a start timestamp, though it is emitted at the start. | `session.py:333-339` |
| **prefill end** | **ABSENT as an explicit line.** The end is only inferable from `turn reuse:` (`engine.py:957`) / the next request's line. `_start_turn` (`engine.py:925`) captures `turn.prefill_seconds` (from `session.py:661 pre_s`) but never logs it on the success path. | `engine.py:925-969`, `session.py:661`, `session.py:696` |
| **first token** | **ABSENT as a timestamp.** `_start_turn` computes `anchor = int(mx.argmax(turn.anchor_logits.reshape(-1), axis=-1).item())` (`engine.py:968`) — this IS the first token — but it is not logged and not timestamped. The anchor is emitted as a normal mid-response (`engine.py:794`). | `engine.py:968`, `engine.py:794` |
| **last token** | **ABSENT as a timestamp.** The turn close is `_end_turn` (`engine.py:878` → `_final_response` `rounds.py:183`), which attaches usage/stats but no wall timestamp. | `engine.py:878-888`, `rounds.py:183-249` |
| **response closed** | **ABSENT.** `FinishedResponse` is appended at `engine.py:447`; the final chunk carries `GenerationStats` (`rounds.py:213-231`) with `prefix_cache_hit` = `"partial"`/`"none"` (`rounds.py:212`) and cumulative `mtp_cycles`/`mtp_accepted` (`engine.py:852-853`, `874-875`). No "closed at T". | `engine.py:447`, `rounds.py:212-231` |

**Bottom line for a later instrumentation phase:** of the six phase events,
**zero** carry a wall-clock timestamp on the DSv4.1 engine path. The useful
*existing* anchors you can build on: `[DSV41] prefill controls:` (prefill
start-ish; `session.py:333`), `[DSV41] turn reuse:` (`engine.py:957`), `[DSV41]
session reuse:` (`session.py:1005`), `[DSV41] reuse undershoot:` (`engine.py:965`),
`[DSV41] engine built:` (`builder.py:102-107`), `[DSV41] warmup: … in {s}s` (`engine.py:384-387`).
Everything else must be added.

Note also: the generic-path `[MTP-PROF]` and `[MTP] cycles=…` lines are **not**
emitted by the DSv4.1 engine, so no existing per-cycle instrumentation exists on
the cluster path.

---

## 5. ADAPTIVE GAMMA / GammaPolicy

**It EXISTS, it is ON by default, and it can RAISE gamma to 4.**

- Class: `mlx_lm.models.deepseek_v41.spec.GammaPolicy`
  (`mlx-lm/mlx_lm/models/deepseek_v41/spec.py:80`).
- **Default candidate set is `gammas=(1, 2, 3, 4)`** (`spec.py:87`); default
  `start=3` (`spec.py:87`).
- **`next()` search** (`spec.py:112-125`): after a `warmup=4`-round warm-up
  returning `self.g`, it iterates `for g in self.gammas`, computes
  `e = sum_k prod_{j<=k} q_j` (expected committed tokens, `spec.py:117-120`) and
  `t = dms[0] + dms[1]*g + self.v[g+1] + self.oh` (`spec.py:121`), and picks the
  `g` maximizing `e/t`. With `gammas` containing 4 and `v` containing `{1..6}`,
  **gamma 4 is reachable and is the max selectable.**
- **On by default in the engine:** `Dsv41Engine.adaptive_gamma: bool = True`
  (`src/exo/worker/engines/mlx/dsv41/engine.py:281`). The policy is built once
  per request in `_rounds` (`engine.py:982-986`):
  `policy = _spec_policy(self.gamma) if (head is not None and self.adaptive_gamma) else None`,
  via `_spec_policy` → `GammaPolicy(start=gamma)` (`rounds.py:291-295`).
- **Consumed every round:** `gamma = int(policy.next()) if policy is not None else 1`
  (`rounds.py:402`); after verify, `pol.update(g, n)` (`spec.py:335` in
  `spec.generate`; in `_one_round` the update is not called — see caveat below).
- **`VERIFY_MS`** IS the timing table the policy minimizes over:
  `mlx-lm/mlx_lm/models/deepseek_v41/spec.py:77`
  `VERIFY_MS = {1: 58.5, 2: 74.9, 3: 87.9, 4: 97.7, 5: 111.7, 6: 120.1}`,
  with the comment at `spec.py:76` "Measured on 2x M4 Max TP=2 (exo phase
  16/17): verify forward ms by rows." It is bound as `self.v = verify_ms or VERIFY_MS`
  (`spec.py:91`) and used only in `next()` at `spec.py:121` (via `self.v[g+1]`).
  **These are exactly the phase-17 numbers** in the task brief (R1..R6 =
  58.5/74.9/87.9/97.7/111.7/120.1). `dms` (draft cost model) defaults
  `draft_ms=(8.5, 0.9)` (`spec.py:87`), `overhead_ms=4.0` (`spec.py:88`).

**Caveat that is itself a finding:** `GammaPolicy.update()` **requires the
caller to call it each round** (`spec.py:96-103`). `spec.generate` does
(`spec.py:335`), but `rounds._one_round` — the cluster path — **never calls
`policy.update()`**. Grep of `rounds.py` for `update` / `pol.` shows only
`policy.next()` at `rounds.py:402`. So on the engine path the policy's
`self.tried`/`self.acc` stay all-zero, `_q()` returns the 0.7 prior
(`spec.py:106`), and `next()` returns the `best_rate` argmax of a *constant*
objective — i.e. **the adaptive policy is effectively a static pick driven by
`VERIFY_MS`/`dms`, not adaptively tracking acceptance**, unless a future
revision adds the `update()` call. This is worth flagging to the parent as a
likely live bug or at least a divergence from `spec.generate`.

**Env flags:** there is **no `EXO_DSV4_MTP_GAMMA`** and **no `EXO_SPECULATIVE_GAMMA`
read for the DSv4.1 engine** — `adaptive_gamma` and `gamma` are dataclass fields
with no env binding on the dsv41 path (`engine.py:280-281`; `_engine_kwargs_from_instance`
at `builder.py:140-166` only forwards `max_kv_tokens`, `prefill_step_size`,
`prefill_transient_budget_mb`). `EXO_SPECULATIVE_GAMMA` is read on the *generic*
path (`generator/batch_generate.py:845`, `dsv4_mtp.py:3969`) and by `spec.generate`
only as the `gamma=` argument from its caller. `start_cluster.sh` exports
`EXO_SPECULATIVE_GAMMA=3` (`start_cluster.sh:242`, forwarded `:2090`) and it is
set in the live launch env (`docs/benchmarks/phase22-…/raw/production-launch-cmd-m4-1.txt:1`),
but on the dsv41 engine it is inert for gamma selection.

---

## 6. THE MTP stats-block FORMAT STRING

The line the user quotes (`gamma=` / `mean_accept=` / `committed` / `MTP cycles`
/ `hist=`, seen at 150 cycles mean 2.447/4 and 2400 cycles mean 1.272/3) is the
**`[MTP]` acceptance line**:

```
[MTP] cycles={self._spec_cycles} mean_accept={mean:.3f}/{self.gamma} hist={hist}
```

- **Exact format string:** `src/exo/worker/engines/mlx/speculative/dsv4_mtp.py:2173-2177`
  (f-string assembled at `2174-2176`; `mean = self._spec_total_accepted / self._spec_cycles`
  at `2171`; `hist = ",".join(f"{i}:{c}" for i, c in enumerate(self._spec_accept_hist))`
  at `2172`).
- **Emit site / method:** `DSv4MTPBatchGenerator._record_acceptance`
  (`dsv4_mtp.py:2150`), gated by `_LOG_INTERVAL > 0` and
  `self._spec_cycles % _LOG_INTERVAL == 0` (`dsv4_mtp.py:2168-2170`).
  `_LOG_INTERVAL` = `EXO_DSV4_MTP_LOG_INTERVAL` default 0 (`dsv4_mtp.py:118`).
  The `/{self.gamma}` denominator matching the user's "…/4" and "…/3" is
  `dsv4_mtp.py:2175`. Call sites: batch path `dsv4_mtp.py:3080` (once per stream
  per cycle), single-uid `dsv4_mtp.py:4648`, tree `dsv4_mtp.py:5675`.

**A second, distinct stats block** (also present):

```
[MTP-PROF] cycles={self.cycles} {bs_summary}                       # dsv4_mtp.py:817
[MTP-PROF]   B={b} {phase:10s} mean={mean:6.2f}ms min=…ms max=…ms n=…
                                                                   # dsv4_mtp.py:830
[MTP-PROF]   B={b} {phase:10s} mean={mean:6.2f} min=… max=… n=…     # dsv4_mtp.py:836
```

emitted by `_PhaseTimer.dump` (`dsv4_mtp.py:815`), phase order
`("draft","verify","accept","commit","rollback","total")` + extras
(`dsv4_mtp.py:818, 821`), every `_PROFILE_INTERVAL` cycles (`dsv4_mtp.py:812`).

**And a third**, the shadow block: `[DSPARK-SHADOW] cycles=… ctx=… a_mean=… k_mean=…
accept_hist=… k_hist=… bypos=… draft_ms_mean=… … cycle_ms_mean=…`
(`dsv4_mtp.py:646-652`), plus `[DSPARK-SHADOW-GUARD] …` (`dsv4_mtp.py:657-662`).

**Rank/node specificity — the honest answer: the emitting code has NO rank
gate.** `_record_acceptance` (and `_PhaseTimer.dump`) call `logger.warning`
unconditionally (`dsv4_mtp.py:2173`, `dsv4_mtp.py:817/829/835`). The method is
called once per stream per cycle (docstring `dsv4_mtp.py:2153-2162`). Nothing
checks `device_rank`. So **on a TP=2 2-node cluster this line prints on EVERY
rank that reaches the spec path** — i.e. one `[MTP]` line per node per interval,
not rank-0-only. (Contrast the runner's chunk emitter, which IS rank-gated:
`runner.py:877-884` drops `ChunkGenerated` on `device_rank != 0`.) This matters
because "both nodes log the same `[MTP]` line" can be mistaken for two
independent measurement points. If the observed stats block shows only one
copy, that is evidence the DSv4.1 engine (not this code) is what ran — see §0.

---

## 7. PREFIX-CACHE MATCH

**Structure: tokenize-once, then a LINEAR SCAN of resident conversation token
lists computing a contiguous common-prefix length; a blake2b hash is used only
as the cold-open store key, NOT for matching.**

- The engine asks the session store for a conversation:
  `session = self._sessions.get(tokens_list, self._conversation_key(params))`
  (`engine.py:761`).
- `Dsv41Sessions.get` (`src/exo/worker/engines/mlx/dsv41/session.py:979`):
  when the client names a conversation (`key`), it keys on `f"id:{key}"`
  (`session.py:983-976`) — O(1) dict hit. When it does not, it **linear-scans
  every resident entry**:
  `for k, entry in self._entries.items(): lcp = _sc.common_prefix_len(ids, entry.session.tokens)`
  (`session.py:997-1000`), keeping the max, then requires
  `best_lcp >= MIN_REUSE_TOKENS` (`session.py:1001`). Emits the `session reuse:`
  INFO (`session.py:1004-1007`). On miss it consults the SSD park store
  (`session.py:1011`) then cold-opens (`session.py:1014-1015`).
- `common_prefix_len` (`mlx-lm/mlx_lm/models/deepseek_v41/session_cache.py:123-132`):
  converts both to numpy int64 (`_as_ids`), takes `n = min(len)`, computes
  `diff = x[:n] != y[:n]`, and returns `n` or `int(np.argmax(diff))`. **O(n)
  vectorized compare, no trie, no hash on this path.**
- The per-conversation rewind decision is inside `SessionCache.plan`
  (`session_cache.py:460-473`): `lcp = common_prefix_len(ids, self._ids)`, then
  `boundary = cache.offset` if `lcp >= offset` else `self._boundary_le(lcp)`
  (`session_cache.py:469-472`), where `_boundary_le` scans the checkpoint dict
  (`session_cache.py:383-391`, O(#checkpoints)). `append_turn` rewinds +
  prefills only `ids[boundary:]` (`session_cache.py:605-611`).
- The hash: `prefix_hash` (blake2b, `session_cache.py:112-120`) is used by
  `key_for` as the cold-open / park key (`session.py:972-977`), not for match.

**Complexity at 100K rows:** the match is **O(100K) numpy element compares per
call, per resident conversation** (`session_cache.py:129`), plus the
per-conversation `plan`/`common_prefix_len` again in `append_turn`
(`session_cache.py:468`). With `max_sessions = 2` (`engine.py:289`, default 2),
that is ~2×100K int64 compares per request — sub-millisecond in numpy, but it is
**linear, not a trie/hash-index**, and it re-runs the full-length compare on the
hot path of every request that does not carry a conversation id. `MIN_REUSE_TOKENS`
comes from `session.py` (imported at `park.py:80`) and gates whether a match is
used at all (`session.py:1001`). A subtlety worth flagging: `common_prefix_len`
allocates an O(n) boolean `diff` array (`session_cache.py:129`) even on an exact
hit, so it is O(n) memory churn per lookup, not O(1) fast-path.

---

## 8. `MTP_ACCEPT_LOGPROBS` + `MTP_TIEBREAK_FIX` SEMANTICS

Both flags live on the **generic MLX batch-generator** path
(`dsv4_mtp.py`), NOT the DSv4.1 engine. Neither is read anywhere under
`dsv41/`. They are active in the cluster's launch env
(`EXO_DSV4_MTP_ACCEPT_LOGPROBS=1`, `EXO_DSV4_MTP_TIEBREAK_FIX=0`;
`docs/benchmarks/phase22-…/raw/production-launch-cmd-m4-1.txt:1`) — which is
consistent with §0: if those flags are set but the dsv41 engine is what serves
the card, they are inert for that card.

### `EXO_DSV4_MTP_ACCEPT_LOGPROBS`

- Gate: `_ACCEPT_LOGPROBS = os.environ.get("EXO_DSV4_MTP_ACCEPT_LOGPROBS", "0") == "1"`
  (`dsv4_mtp.py:340`). **Read once at import. DEFAULT 0.**
- Mechanism (comment `dsv4_mtp.py:321-339`): at temp=0 the plain MTP-off
  generator picks its token by `argmax` over **log-sum-exp-normalized logprobs**
  in native bf16, while this file's accept/bonus historically picked `argmax`
  over the **raw verify logits**. bf16 subtraction can collapse two near-tied
  logits to the SAME value, and first-index argmax then picks the lower id —
  a different token with bitwise-identical logits.
  - `=1`: the greedy accept/bonus argmaxes are taken over the SAME normalized
    logprobs the generator samples from, making MTP-on token selection
    **rule-identical** to MTP-off. Code: `logprobs_all = verify_logits - mx.logsumexp(...)`
    then `target_tokens = mx.argmax(logprobs_all[:, :gamma, :], ...)` and
    `all_next = mx.argmax(logprobs_all, ...)` — `dsv4_mtp.py:2743-2747`
    (single-uid) and `dsv4_mtp.py:4243-4246` (batched).
  - `=0`: argmax over the raw `verify_logits` (`dsv4_mtp.py:2748-2753`,
    `4248-4251`).
- So: **probabilistic-vs-greedy is NOT what this flag controls** (this path is
  greedy either way). It controls *which greedy rule* — normalized-logprob
  argmax (`=1`, generator-identical) vs raw-logits argmax (`=0`). It is a
  **losslessness / token-identity** switch, not a sampling switch.

### `EXO_DSV4_MTP_TIEBREAK_FIX`

- Gate: `if os.environ.get("EXO_DSV4_MTP_TIEBREAK_FIX", "1") != "0":` —
  i.e. **DEFAULT ON** (code default `"1"`), read **per call** inside
  `_speculative_next` (`dsv4_mtp.py:4275`; eps via `EXO_DSV4_MTP_TIEBREAK_EPS`
  default `0.5`, `dsv4_mtp.py:4276`). Comment `dsv4_mtp.py:4251-4274`.
- Mechanism: the batched verify forward differs from a sequential single-token
  decode by ~1 ulp; at temp 0 that flips *tied* tokens, and one flipped tie
  cascades the whole generation onto a different (often degenerate/repetition)
  trajectory. With the fix on, the **bonus-token** selection applies a
  deterministic tie-break: among tokens within `eps` logits of the per-position
  max (`_tied = _vl0 >= (_maxlogit - _tb_eps)`, `dsv4_mtp.py:4281`), pick the
  **lowest token id** (`mx.where(_tied, ids, big)` then `argmin`,
  `dsv4_mtp.py:4285-4287`). It touches only `all_next` (the BONUS token, which is
  NOT in the KV cache this cycle), never the accepted drafts (which ARE cached,
  so rewriting them would desync KV). `=0` skips the tie-break and keeps plain
  `argmax` (`dsv4_mtp.py:4289`).
- Note the interaction: `_ACCEPT_LOGPROBS=1` **supersedes** the tie-break fix
  (comment `dsv4_mtp.py:337-339`), and production runs `ACCEPT_LOGPROBS=1` with
  `TIEBREAK_FIX=0`. Both are read on the generic path only.

---

## DEAD ENDS (grepped for and NOT found)

These are absences verified by grep across `src/`, `mlx-lm/mlx_lm/`, `bench/`,
`start_cluster.sh`, `docs/` (excluding `.venv/` and `mlx-lm/build/`). Absence is a
result — several of these are things a later phase might otherwise assume exist.

1. **`EXO_DSV4_MTP_GAMMA`** — does not exist anywhere. The dsv41 engine's gamma
   is a plain dataclass field (`engine.py:280`), never env-bound.
2. **`SYNC_SPANS` / any "sync-span" variant of the MTP-PROF brackets** — no such
   flag. `EXO_PROFILER_SYNC_SPANS` exists but only on the independent
   `SpanProfilerHook` span timer (`profiler.py:215`), which is not the bracket
   emitter. The bracket emitter has no sync mode.
3. **`EXO_DSV4_MTP_PROFILE_SYNC` / `MTP_PROFILE_SYNC`** — does not exist.
4. **`mx.synchronize()` inside `_PhaseTimer`** — none. Brackets use `mx.eval`
   only (`dsv4_mtp.py:4089`, `4225`), and only at draft+verify.
5. **`mx.eval` at the MTP-PROF accept / rollback bracket closes** — deliberately
   absent, with an explicit comment saying so (`dsv4_mtp.py:3091-3099`,
   `3416-3424`, `5437-5445`). The accept and rollback brackets close UNFENCED.
6. **Any `logger.` / per-round log line inside the DSv4.1 `_one_round` round
   loop** — none per-round; only `rounds.py:395` (once per request, draft-window
   priming) and the store/prefill lines in `session.py`.
7. **`[MTP]` / `[MTP-PROF]` emitted from the DSv4.1 engine path** — the dsv41
   package never imports `dsv4_mtp`. Those lines cannot appear for this card
   while it is served by `Dsv41Builder`.
8. **`EXO_DSV4_MTP_LOG` (unqualified)** — the comment at `dsv4_mtp.py:116`
   references it but the code only reads `EXO_DSV4_MTP_LOG_INTERVAL`
   (`dsv4_mtp.py:118`). `EXO_DSV4_MTP_LOG` itself is read nowhere.
9. **`_resolve_sampler` call sites** — the method exists (`engine.py:1040`) but is
   **never invoked**; there is no temp>0 sampler wired into the engine.
10. **A top-p / top-k / temperature branch in the dsv41 engine or `rounds.py`** —
    none; those names appear only as unused dataclass field defaults
    (`engine.py:274-277`).
11. **A trie- or hash-indexed prefix match** — none. Matching is the linear
    `common_prefix_len` scan (`session_cache.py:123-132`); blake2b
    (`prefix_hash`, `session_cache.py:112`) is only the cold-open/park key.
12. **`VERIFY_MS` as an env var** — no. It is a module-level literal dict
    (`spec.py:77`); the only override is the `verify_ms=` constructor arg
    (`spec.py:88, 91`), which no caller in-tree passes.
13. **Any request-level wall-clock timestamp for received / tokenized /
    prefill-start / prefill-end / first-token / last-token / closed** — none on
    the dsv41 path (see §4). Zero of the six are present as timestamps.
14. **`[DSV41] session reuse:` / `[DSV41] turn reuse:` / `[DSV41] prefill
    controls:`** — CONFIRMED PRESENT (not dead ends, listed here only to note
    they were checked against the user's examples):
    `session.py:1005`, `engine.py:957`, `session.py:334`.
