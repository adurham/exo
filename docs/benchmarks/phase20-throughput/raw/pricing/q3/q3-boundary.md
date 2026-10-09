# Q3 — The 7.26 ms client↔server per-round boundary: attribution & provenance

**Author:** sr-coder (Q3 dispatch). **Date:** 2026-10-09 CDT. **Worktree:** `/private/tmp/phase20-campaign`,
read-only; exo read at `/Users/adam.durham/repos/exo` (`git show <commit>:<path>`), **no boot, no POST, no deploy**.
**Deliverables:** this doc + `q3-boundary.json`. Raw evidence = already-collected files under `raw/p3/` and `raw/p3b/`
(no new measurement).

---

## 0. VERDICT

**The 7.26 ms figure is an arithmetic artifact, not a boundary cost on the production critical path.**
it is `101.06 − 93.8` where the minuend is a **client-observed, agentic-shaped, next17-HIER0-build** number and the
subtrahend is a **server-side, benign-shaped, next16-instr-build** number (different boot, different request shape,
different statistic). Measured **same-boot, same-request** (next16-instr OFF, `raw/p3/`), the client-observed
ms/round sits **+0.23…+0.51 ms** above the server loop's own mean per-round wall (loop wall/n, the quantity the
round loop actually measures) — **both shapes** — and stays within **≤2.4 ms** of it under *any* honest server
statistic convention tried (`round_total` mean → +0.86…+1.14; `round_total` median → +1.18…+2.42; wall/(n−1) →
+0.42…+1.70). I.e. the true client↔server per-round gap is ≈ a few tenths of a ms, ≤8 % of the claimed 7.26 ms,
and it is delivery-offset noise at the two ends of the client's measurement span, not a per-round handshake.
**Addressable as a production lever: ~0 ms.** The engine decode loop is a **server-internal loop that streams
tokens out**; the next round's dispatch is gated **only on the local consumer drain** (measured 0.63–0.69 ms/round
median, in-process, includes a required cross-rank cancel collective), **never on the client**. The doc's "1.40 ms
in-round + 7.26 ms boundary" split is mis-assigned: the same-boot decomposition is **~1.5 ms in-round (rt−verify,
inside the round) + ~0.65 ms inter-round in-engine gap + ~0.3–0.5 ms client-statistic delivery offset**, and the
dominant remaining component of the doc's 7.26 is simply **agentic-vs-benign shape** (raw server medians 99.23/99.28
vs 93.69/93.71 → ≈ +5.55 ms; verify_block 97.8 vs 92.2) — not a client boundary at all.

---

## 1. Loop mechanism (verified at the deployed commit `fb4f9290b`)

**Which code runs.** The deployed DeepSeek-V4.1 instance is served by `Dsv41Engine`
(`src/exo/worker/engines/mlx/dsv41/`), selected by the model card via `make_mlx_family_builder`
(`dsv41/dispatch.py:36-56` → `Dsv41Builder` → `Dsv41Engine`, `dsv41/builder.py:89-108`). The brief-named
`batch_generate.py` / `speculative/dsv4_mtp.py` / `pp_batched_*` are the **batched-generator / PP path** — `git grep`
shows the dsv41 engine imports neither (only a docstring reference to `GeneratorQueue.gen` semantics,
`dsv41/rounds.py:124`). The MTP (DSpark) draft+verify for this deployment lives in `dsv41/rounds.py._one_round`
(`:328`) calling the mlx-lm fork (`from mlx_lm.models.deepseek_v41 import spec`, `rounds.py:372`).

**The round loop** (`dsv41/engine.py:1027-1076`, `_rounds`): per round — `_check_cancel` (`:1047`, cross-rank
agreement collective) → `_one_round` (`:1049`, draft+verify forward) → `yield batch, lps` (`:1075`). The generator
**suspends at the per-round yield** and is resumed only when the local consumer pulls it again.

**Who drives the pull** (all in-process): `_generate`'s `committed_tokens()` (`engine.py:849-855`) iterates the
rounds and yields one `_mid_response` per committed token (`:912`); the `dsv41_output_parser` pipeline wraps that
stream; `_queue_of` interleaves a `None` flush sentinel after every response (`rounds.py:115-133`); `Dsv41Engine.step()`
(`engine.py:451`) pulls through the parser **until a flush point, never into the client** (`:471-485`).

**The runner loop** (`runner/runner.py`): `while self.active_tasks: results = self.generator.step()` (`:649-653`);
each result is pushed via `send_chunk` (`:869-884`) as `ChunkGenerated` into `event_sender.send(...)` (`:884`), which is
`MpSender.send` → `mp.Queue.put` (`utils/channels.py:260-268`). **The channel is UNBOUNDED** (`mp_channel` default
`max_buffer_size=inf` → `mp.Queue(0)`, `channels.py:222-231`, `:442-455`), so `send` never blocks for space. The
supervisor forwards events to master from a **separate async task** (`_forward_events`, `runner/supervisor.py:844-886`);
HTTP/SSE and the client are behind master/API in other processes. Client read speed can buffer downstream but can
**never block the worker decode loop**.

**Answer to the gating question:** there is **no request/response-per-round**. One HTTP request → one long SSE stream.
Round N+1's dispatch waits **only** on the in-process consumer drain of round N + the cancel-agreement collective
(`agreement.py:65-75` → `mx_any` = distributed all_sum+eval, `utils_mlx.py:2147-2154`) — **not on any client-side
receipt/return**. Measured inter-round gap (same boot, both ranks, both shapes; derived from the next16 PROF where
`emit_ms` is cumulative-from-loop-start: `Δemit_k − round_total_{k−1}`): **median 0.63–0.69 ms**, and it contains the
detokenize/parse/send drain and the cancel collective.

## 2. How t/s and ms/round are computed — server side vs client side

- **Server:** the `: generation_stats` SSE frame carries a `generation_tps` field, but on this engine path it is
  **structurally 0.0**: `_RoundStats` (`engine.py:263`) is defined but **never instantiated/passed** — `_generate`
  calls `_final_response(...)` without `round_stats` (`engine.py:898-910`, `:920`), so `rounds.py:213` takes the
  `else` branch (`generation_tps=0.0`, `rounds.py:222-231`). Confirmed in the live frames: `generation_tps: 0.0`
  in every rec of `raw/p3b/next17h0_agentic.json.jsonl` and `raw/p3/levers_off_recs.jsonl`. The server does **not**
  compute or report t/s for this deployment.
- **Client (all published campaign t/s and ms/round):** `decode_tps = (completion_tokens−1)/decode_s` and
  `ms_per_round = decode_s·1000/rounds`, where `decode_s` = client wall from the first to the last decoded
  content/reasoning SSE delta and `rounds = Δmtp_cycles_cumulative/3` (a server counter read out of the stats frame).
  Citations: `phase19_round_measure.py:96-105` (`decode_s`), `:120-135` (`ms_per_round`); agentic variant
  `phase19_agentic_measure.py:201-217`, `:248-263`; driver `p3b_driver.py:36-77`, `:122`. So every "94.94",
  "99.46", "101.06" is a **client-statistic applied over a server counter** — the honest production number is
  already client-observed.

## 3. Provenance of 101.06 (verified)

- **Build:** `deploy/next17-levers` @ **`576e9d279`** (mlx-lm `3bf8316`), relaunch #2 2026-10-08 14:46
  (`raw/p3b/deploy_next17_hier0.log`); run as agentic 91K g3, 800 tok, 4 reps (3 timed: 101.06, 101.07, 100.67),
  median 101.06 / 30.24 t/s. Harness: `phase19_agentic_measure.py` driven by `p3b_driver.py`
  (`raw/p3b/next17h0_agentic.{json,jsonl,log}`; doc cite `PHASE3B-SHIP-VALIDATION.md:161`;
  PM history `PERFORMANCE_HISTORY.md:11439`). **Not the current build**, but a direct ancestor:
  `git merge-base --is-ancestor 576e9d279 fb4f9290b` = TRUE. Current production `fb4f9290b` (next18) keeps the
  lever-1 gate + adds the mlx-lm `16830e1` bump (lever-2 L2-full default-ON + capture harness); its live numbers
  are **94.94 ms benign / 99.46 ms agentic** (`PHASE5-R1C-SHIP.md:124-125`, `/tmp/p5r1c/ship_smoke_*.json`).
- **The doc's 93.8 partner number:** `round_total 93.8` is the **next16-instr build (`f234b0f6d`), both levers OFF,
  benign** median (`PHASE3-M3.md:49`; raw re-derivation: benign `round_total` median 93.69/93.71 rank1/rank0,
  `verify_block` 92.21/92.25 → in-round residual **1.45–1.47 ms** [doc says 1.40 from the pooled 93.8−92.4]).
  Same-session agentic server numbers on that boot: `round_total` 99.22/99.27, `verify_block` 97.77/97.81
  (residual 1.46 both) — i.e. the agentic-vs-benign shape term is ≈ **+5.55 ms** on the same boot.

**So 101.06 − 93.8 = agentic-client(next17) − benign-server(next16).** The doc's own §1.4 said "(b) is bounded,
not measured" — correct — but its *direction* is misleading: the number mostly measures shape, not boundary.

## 4. Same-boot, same-request measurement (the decisive check; no new measurement needed)

`raw/p3/` contains **both sides of the same boot**: client recs (`levers_off_recs.jsonl`, round_prof=1 OFF arm)
and per-round server PROF (`off_s{1,2}_rank{0,1}.round_prof.jsonl`). Matching client reps to server segments by
round count / `decode_s`≈loop-wall:

| arm | rep | client ms/round | server loop wall/n | server `round_total` med | client − wall/n |
|---|---|---:|---:|---:|---:|
| benign | 1 | 94.93 | 94.62 | 93.65 | **+0.31** |
| benign | 2 | 95.04 | 94.62 | 93.70 | **+0.42** |
| benign | 3 | 94.90 | 94.62 | 93.72 | **+0.28** |
| benign | 4 | 94.87 | 94.59 | 93.64 | **+0.28** |
| agentic | 0 | 101.41 | 100.90 | 99.28 | **+0.51** |
| agentic | 1 | 101.18 | 100.93 | 99.22 | **+0.25** |
| agentic | 2 | 101.45 | 101.18 | 99.27 | **+0.27** |
| agentic | 3 | 101.72 | 101.49 | 99.30 | **+0.23** |

**Client ≈ server mean round wall + 0.23–0.51 ms, both shapes.** (The offset vs other honest server statistics:
vs `round_total` mean +0.86…+1.14; vs `round_total` *median* — the doc's pairing — +1.18…+2.42, because the median
excludes the inter-round gap (0.63–0.69) and the cold first round (122–157 ms benign, 328–468 ms agentic,
amortized ≈ +0.14–0.22 / +0.86–1.46 ms/round); vs wall/(n−1) +0.42…+1.70. The choice of server statistic moves the
answer by up to ~2 ms; the true boundary is the +0.2–0.5 ms floor.)

**Decomposition of the doc's 7.26 ms** (101.06 − 93.8): ≈ **+5.5 ms agentic-vs-benign server shape**
(raw medians 99.23/99.28 − 93.69/93.71 ≈ 5.5; it is verify_block 97.8 vs 92.2) + **+0.63 ms** inter-round in-engine gap + **+0.3–0.5 ms**
client-statistic offset + median-vs-mean/warm-round slack. All MECE, none of it a client↔server handshake.

## 5. Addressability

- **7.26 ms as a lever: none.** The loop is server-internal and streams; nothing waits on the client; the only
  recurring cross-boundary cost is the ~0.3–0.5 ms end-offset in the client statistic (not a lever — it is the
  delivery latency of the first anchor delta and the final batch).
- The only real in-engine inter-round cost is the **0.63–0.69 ms** suspend/resume window (consumer drain +
  `agree_on_cancellations_fast` collective). It is already inside the client number, is bounded, and contains a
  **required** cross-rank agreement collective (`mx_any` = all_sum+eval) that cannot be dropped without touching
  correctness machinery. Even a perfect remove of all of it buys **<0.7 ms/round**.
- **The server `round_total 93.8` is NOT "the honest production number"** — it is a different build + benign shape
  and its bracket **excludes** the inter-round consumer window. The honest per-round production cost is the
  client-observed 94.94 / 99.46 (this build), which already sits within ~+0.2…+2.4 ms of the engine's own
  same-boot numbers for every choice of server statistic.
- Footnote for the review note in the brief ("live benign 94.94 ≈ 1.1 above 93.8, consistent with the 1.40 in-round
  figure"): coincidence of magnitudes, not the same quantity. 94.94 − 93.8 ≈ **gap 0.65 + first-round amortization
  ~0.2 + client offset ~0.3 ± cross-build noise**, whereas 1.40/1.5 = `round_total − verify_block` (draft+tail+host,
  inside the round). Different pair, different meaning.

## 6. Replay (not needed)

No bounded replay is required to settle Q3 — the same-boot client+PROF pair already exists in `raw/p3`. A live-build
PROF would require re-basing the round_prof timer onto `fb4f9290b` (code change + boot) to chase a ≤1.5 ms
statistic-definition effect; not justified. If ever wanted at zero risk, the spec would be: one boot of a build
carrying the next16 PROF timer on the current base, benign+agentic pair, compare per-request client `decode_s/rounds`
against that request's own `Σ(round_total + emit-gap)` in the same rank JSONL — acceptance: |diff| ≤ ~1 ms.

## 7. Caveats

- Segment↔rep matching used round-count + `decode_s`≈loop-wall; where two server segments had equal n (benign
  211/216), both candidates score within 0.06 ms of each other — conclusion unchanged.
- The `raw/p3` OFF boot ran next16-instr with both gates `0`; the live build is next18 (gates' fix default-ON).
  The client-vs-server statistic relationship is a property of the harness + engine loop, not of the gates.
- `emit_ms` in the PROF file is **cumulative from loop start** to each round's start (verified against data:
  monotone, Δ = prev `round_total` + gap); the per-round gap cited is the difference, not the raw field.
