# Post-reboot remaining levers — campaign deliverable (2026-10-07)

Branch `deploy/next15-levers` (off `deploy/next14-gamma` @ `71a8c94c2`).
Cluster: 2× Mac Studio M4 Max, TP2 over jaccl RDMA. Model
`dealignai/DeepSeek-V4.1-Flash-UNCENSORED-EXL3-2.9bpw`, engine `dsv41` (NOT
`dsv4_mtp`; all `EXO_DSV4_MTP_*` knobs dormant), greedy-only, gamma pinned at 3.

## 0. Executive summary — what the reboot blocked, and what changed anyway

| # | Phase | Status | Headline result |
|---|---|---|---|
| 0 | Cold baseline + drift gate | **BLOCKED** — nodes at FileVault pre-boot | Cluster down the whole window; no cold rep possible. See §0.1. |
| 1 | Token accounting + `reasoning_effort` | **DONE** | Turn reconciled to 0.0 s. `reasoning_effort` = **PARTIAL** (prompt-level only). Brief's 118–160K reasoning-token claim = **FALSE** (real: 12,399). |
| 2 | Delta-prefill audit + knob matrix | **DONE** | **0 cache-miss flags** (0/41 ratio > 1.5). Chunk-4096 go-condition met (41% of delta rows in >4096-row calls). 1 missed image-path entry point. |
| 3 | Kernel go/no-go | **DONE** (source-read + synthetic microbench; no cluster needed) | Verdict **INCONCLUSIVE**: no ≥2 ms/round dense lever; **does NOT clear the GO gate**. M=4→5 step +5.1 ms/round found. |
| 4 | Payload-cap validation | **DONE as recommendation** (replay needs cluster) | `tool_output.max_bytes` 50000→20000 would have cut 13,560 tok = 13.5% of delta rows ≈ **2.8–3.5% of turn wall**. NOT applied (idle-guard: user traffic). |
| 5 | Reasoning-budget test | **SKIPPED** — conditional on Phase-1 = WIRED | Phase 1 = PARTIAL (no effort gradient above xhigh); no engine-side budget exists. |
| 6 | Chunk-4096 endpoint arm | **SKIPPED** — needs ≥1 relaunch; budget unusable (cluster down) | Go-condition met but experiment deferred. |
| 7 | Kernel implementation | **SKIPPED** — conditional on Phase-3 GO | Phase 3 did not open GO. |
| 8 | Wall-decomposition rebuild | **OPTIONAL/LAST** — not reached | — |

**Relaunch budget: 3 available, 0 spent.** The post-reboot production relaunch
(budget line 1) never happened because the nodes never came back. No cluster
state was touched.

### 0.1 The blocker (recorded as required)

The reboot was issued as a raw `sudo shutdown -r now` over SSH, **not** via the
repo's `reboot-node.sh` (which uses `fdesetup authrestart` to escrow the
FileVault key). Consequence: with FileVault enabled the Studios stop at the
**pre-boot disk-unlock screen**, which has **no network stack** — Tailscale
shows both peers `offline` and LAN TCP/22 is dead. This is **"blocked on local
physical unlock"**, not node failure; remote unlock is impossible.

Poll evidence (`~/.hermes/cache/scratch/levers-poll.log`, one probe/min):

```
16:09:55 | 192.168.86.201 TCP:closed | 192.168.86.202 TCP:closed
...
16:22:22 | 192.168.86.201 TCP:closed | 192.168.86.202 TCP:closed
```

Both nodes TCP-closed for the entire window (16:04 → end). `tailscale status`
shows `macstudio-m4-1/.2  offline, last seen ~13m ago` throughout — no peer ever
reappeared. **No rescue/reboot was attempted, per instruction.**

**When the owner unlocks the disks:** the cluster is DOWN (a reboot wipes all
processes). The relaunch is *already known-good* — the identical
`EXO_TARGET_BRANCH=deploy/next13 ./start_cluster.sh` from the laptop brought both
nodes to `READY (2/2)` on `deploy/next13 @ f0840af1c` at 15:26 earlier today
(`restore-snapshot-live.md`). Run it, then run the Phase-0 harnesses
(`bench/phase19_round_measure.py --depth 100000 --reps 3`,
`bench/phase19_agentic_measure.py --arm agentic --reps 3`) from the
`deploy/next14-gamma` worktree `/private/tmp/next14-gamma` (harnesses are not on
next13). **Phase-0 gate:** benign 100K round wall PASS if ≤ ~140 ms (morning
135.6; slow-afternoon 154.1). **If ≥150 ms cold, STOP — do not A/B against a
phantom champion.**

---

## 1. Phase 1 — reconciled token accounting + `reasoning_effort` verdict

Full table: `docs/benchmarks/phase19-latency/raw/phase1-accounting.md`.
Session `20261007_092009_9a2ed7`, 42 calls, all `provider=custom`.

### 1.1 Accounting (reconciles)

| quantity | value | source |
|---|---:|---|
| Σ `latency_seconds` (model time) | **1530.59 s** | ledger |
| turn span (first start → last end) | 1814.45 s | ledger |
| Σ inter-call gaps (client/tool) | 283.86 s | ledger |
| `Σlat + Σgaps − span` residual | **0.0 s** | exact |
| Σ `output_tokens` | 21,799 | ledger |
| Σ `reasoning_tokens` | **12,399 (56.9 % of output)** | ledger |
| Σ `prompt_tokens_total` | 3,091,048 | ledger |
| Σ `cache_read_tokens` | 2,990,220 | ledger |
| Σ `delta_rows` (prefill) | 100,828 = Σ `input_tokens` (0 mismatches vs A1 exo log) | A1 + ledger |
| decode_s = Σout / 20.66 t/s | **1055.1 s** | agentic rate |
| prefill_s = 1530.59 − 1055.1 | **475.5 s** (ceiling) | derived |
| implied prefill rate | 212 rows/s (vs 244 cold / 267.5 @100K / 268.6 @30K) | derived |
| bound: prefill_s ∈ | **[377 s, 475 s]** ⇒ decode 18.9–20.66 t/s | derived |

Reasoning share: mean **295 tok/call**, max 1411 (call 40), min 0 (5 calls).
12,399 reasoning tokens = **600 s decode = 39.2 % of model time, 33.1 % of turn wall.**

**Brief claim tested — "42 calls × 2.8–3.8K hidden reasoning = 118–160K tok ≈
1.6–2.2 h": FALSE.** Real total is **12,399 tok** (10–13× lower); real time
≈ 10.3 min. The 2.8–3.8K/call figure matches only a handful of outliers (top-5:
1411/1228/1023/945/786); the max single call is half the claimed per-call
minimum. The brief's contradiction (Section 0.3) is resolved: the reasoning
lever's *token* upside is small; the *prefill* side is the larger term.
(Note: config `agent.reasoning_effort = 'ultra'`, per-model `xhigh`.)

### 1.2 `reasoning_effort` — verdict **PARTIAL** (prompt-level knob, coarse)

Chain: `text_generation.py:15-53` (ReasoningEffort Literal tops at `xhigh`;
`REASONING_EFFORT_CEILING="xhigh"`; `clamp_reasoning_effort` L36) → `ultra`→`xhigh`
→ `api/types/api.py:308-320` validator → `resolve_reasoning_params` (`text_generation.py:75-94`:
effort≠none ⇒ `enable_thinking=True`) → `api/adapters/chat_completions.py:183-202`
→ prompt render `dsv41/engine.py:592-616` → `utils_mlx.py:1859-1898` V4 branch
(`thinking_mode`, `_v4_reasoning_effort:1774-1780` maps `xhigh`→`"max"`) →
`vendor/deepseek_v4_encoding.py:82-95,307-313` prepends the `"max"` effort
preamble (~526 chars) at prompt index 0.

- **WIRED at the prompt level** (not a no-op): it changes the rendered prompt.
- **But coarse**: only 3 rungs (`low`/no-prefix, `high`, `max`); `xhigh` and
  `ultra` collapse to `max`. **No gradient above `xhigh`.**
- **Sampler unaffected**: engine greedy-only (`dsv41/rounds.py:21-27`).
- **dsv41 round/session/output code never re-reads it.** Only literal
  `enable_thinking` in the package is the warmup call (`engine.py:377`).

⇒ **Phase 5 is moot:** there is no stronger or weaker Hermes-side setting to
test (xhigh/ultra are identical on the wire), and no engine-side thinking budget
exists. A real budget would be an output-affecting engine change (recommend-only).

### 1.3 System-prompt stability

`system_prompt_hash = 13466837d495ef2fda6d113fbc7656ab13431a876c7efe9e40a2d706a99a708a`,
one row, 22,770 chars, used by exactly 1 session. **Byte-identical across all 42
calls.** No per-call wall-clock / session-id / cwd churn (0 matches); only fixed
baked dates/paths. ⇒ the cached prefix is **not** invalidated by per-call content.

---

## 2. Phase 2 — delta-prefill audit + entry-point × knob matrix

Full: `docs/benchmarks/phase19-latency/raw/phase2-delta-audit.md`.

### 2.1 Audit — hypothesis refuted (0 misses)

Denominator: `expected_delta(i) = prompt[i] − prompt[i−1]` (client-added tokens;
the reuse boundary is pinned at the previous prompt end by the prompt-end
checkpoint, `dsv41/session.py:684-686`).

- **Misses (ratio > 1.5): 0** of 41 warm calls. Ratio min/median/max =
  **0.004 / 1.000 / 1.000**; largest positive excess = 0.
- **5 warm calls under-feed** vs the prompt delta (3, 8, 20, 21, 41) — reuse
  hits *past* the boundary (benign under-prefill).
- Σ actual_prefill = 100,828 (all) / 76,733 (warm). Σ expected_delta warm =
  80,785 (**actual 4.0 % below client delta**).
- The 39/42 `reuse undershoot` warnings are a **BPE-seam rewind artifact**
  (`engine.py:971-980`, fires when `prefill_tokens > 256`), **not** a miss signal.
- The 32/42 "prefill-dominated" calls are so because the **client delta itself**
  is large relative to short generations — not cache miss or prompt churn.
- **Chunk-4096 go-condition MET**: calls with `delta_rows > 4096` = {1 cold
  24095, 6 9115, 7 14271, 24 8325}; share of Σ delta rows = **55.3 % all-calls /
  41.3 % warm-only** (threshold 25 %). Max warm delta = 14,271 rows.
  (Experiment itself deferred — needs a relaunch.)

### 2.2 Entry-point × knob matrix (findings)

14 entry points enumerated. The one knob-honoring gap:

- **Missed entry point — image-span / `chunk_plan` path.** `_start_turn` passes a
  `chunk_plan` only when images are present (`engine.py:954`) → routes to
  `SessionCache._prefill_planned` (mlx-lm `session_cache.py:506`), which pins
  `long_threshold=10**9` (`:529-530`) → selects the fixed-crossover branch
  (`session.py:317,361-362`) and **never calls `choose_prefill_step`**
  (`session.py:372`). ⇒ **`EXO_PREFILL_TRANSIENT_BUDGET_MB` is inert on any image
  request** (the chunk no longer shrinks with context). The fence knob is *not*
  affected. Not exercised this turn (text-only), but a real latent gap.
- **Latent fallback gap**: mlx-lm `chunked_prefill` ignores `fence_every`/`async_depth`
  (`session_cache.py:154-157`); not live (the engine's partial is always installed).
- **Launcher allow-list gaps**: `EXO_DSV41_PARK*`, `EXO_DSV41_{LAYERS,VISION,SPECULATIVE,
  THINK_MARKERS,ENGRAM_*}`, and `DSV41_INDEXER_TILE*` are read in code but not
  forwarded by `start_cluster.sh` — overrides silently dropped.
- **No consumer-skip gap**: `DSV41_INDEXER_CONSUMER_SKIP` is an import-time
  constant (uniformly active).

Chunk: deployed base = **2048** (`EXO_PREFILL_STEP_SIZE=2048`, launcher:88),
shrinking as `budget//(row_bytes·offset)` (budget 2048 MB, `row_bytes=1` under
the M2 HIER block=8), floor 128. Delta histogram: ≤256→2, 257–1024→13,
1025–4096→23, 4097–16384→3, >16384→1.

---

## 3. Phase 3 — kernel go/no-go: **INCONCLUSIVE** (not EXHAUSTED, not GO)

Two artifacts: source-read `raw/phase3-kernel-sourceread.md`; synthetic microbench
`raw/phase3-kernel-microbench.md` (+ JSON), script `bench/exl3_dense_smallm_probe.py`.
Microbench machine: **MacBook Pro M4 Max** (same chip class), mlx 0.32.0.dev;
**synthetic weights** at the real per-rank TP=2 shapes (141.0 M params/layer,
52.9 MB trellis/layer, k=3); validity cos(A,B)=0.9999999.

### 3.1 Dispatch + dequant (source)

`EXL3Linear.__call__` (`mlx_lm/models/exl3/exl3_linear.py`): `rows==1`→fused GEMV
(L123); **`2≤rows≤16`→fused trellis GEMM `inner_gemm_mlx` (L146) — the M=4 verify
band**; `rows≤64`→same (L157); `_WCACHE` branch **dead by default** (`EXL3_WCACHE=0`,
L154); `>64`→transient decode+matmul / striped. **No persistent fp16 W** in the
M≤8 band — the trellis is decoded once per `mt`-group and **amortized over M**
(not per row). `_M_TILE=8`; `simd_mt = {1:1, 2:2, ≤4:4, else 8}` (`gemv_metal.py:1044,
1410, 1422`). Per-row FMA/store is **guarded by `mm<batch`** (`gemv_metal.py:648,
1330`) — no M-tile work padding. Trellis re-read stays **1× for all M≤8**,
doubling only at **M=9**.

### 3.2 Microbench table (µs/layer, arm A EXL3Linear / arm B fp16 ceiling / arm C roofline ×40)

| M | A µs/layer | A ×40 ms | B µs/layer | B ×40 ms | C ×40 ms | A eff GB/s |
|--:|---:|---:|---:|---:|---:|---:|
| 1 | 666.8 | 26.67 | 472.8 | 18.91 | 4.70 | 79.3 |
| 4 | **962.6** | **38.51** | 996.1 | 39.85 | 4.70 | **54.9** |
| 5 | **1089.1** | **43.56** | 1021.4 | 40.86 | 4.70 | 48.6 |
| 6 | 1097.3 | 43.89 | 1028.6 | 41.15 | 4.70 | 48.2 |
| 8 | 1094.8 | 43.79 | 1041.0 | 41.64 | 4.70 | 48.3 |
| 16 | 2070.3 | 82.81 | 1118.4 | 44.74 | 4.70 | 25.5 |

### 3.3 The three findings and the rubric

1. **Not at the roofline** (achieved 15–26 % of 450 GB/s) — but **arm A ≈ arm B at
   M=4** (962.6 vs 996.1 µs) despite reading 5.3× fewer bytes ⇒ **decode is not the
   recoverable bottleneck**; the machine's own ceiling here is ~55 GB/s. Even a
   perfect A→B (decode-free) move is a **+3 % regression** at M=4.
2. **M=4→5 step is VISIBLE: +126.4 µs/layer = +5.1 ms/round** across 40 layers
   (reproduces in every linear). `A(6)−A(5)` is only +8.2 µs; A(8)<A(6). Mechanism
   = `simd_mt` 4→8 (`gemv_metal.py:1422`). This is a **plausible dense-side
   contributor** to the real R4→R5 `VERIFY_MS` +14.0 ms (the other ~9 ms is
   MoE/attention, out of scope).
3. **Rubric:** GO requires `A(4) ≥ 1.6·A(1)` **and** saving ≥2 ms/round. Ratio
   **1.44 < 1.6** (FAILS; saving clause 5.17 ms ≥ 2 ms passes). EXHAUSTED requires
   `A(4) ≤ 1.25·max(A(1),C)` (962.6 ≤ 833.5 FALSE) **and** `A(5)−A(4) ≈ A(6)−A(5)`
   (126.4 ≈ 8.2 FALSE). **Both fail → INCONCLUSIVE.**

### 3.4 Operational conclusion

**No demonstrated ≥2 ms/round recoverable dense-GEMM lever.** The dense path is
neither a large hidden lever nor at a *recoverable* ceiling; the small-M penalty
(5.17 ms) exists but the GO ratio gate does not open, and the biggest headroom
(33.8 ms to the paper roofline) is the machine's distance from its own bandwidth
(arm B can't reach it either), not a removable kernel mistake. Per the campaign's
own rubric, this track is **not cleared to implement**. Residual uncertainty is a
**loaded-machine / eval-floor confound** (load avg 5–8/16, streaming BW ~320 GB/s).
*One extra measurement would settle it:* time the entire 40-layer × 40-projection
dense set at M=4 as **one batched eval (K≥256) on an idle machine** — if the ~55
GB/s plateau persists it is EXHAUSTED; if it jumps toward the roofline the lever
is GO (fix = launch/batch shape, not decode). **This is the natural first thing to
run once the cluster — or a quiet machine — is available.**

---

## 4. Phase 4 — payload discipline: measured numbers, NOT applied

**Recommendation only.** The validation replay (`phase19_agentic_measure.py`
with truncated payloads) needs the cluster, which is down; and the Hermes config
is write-guarded + must not change under live user traffic. **Nothing was applied.**

Offline measurement from the real session (`state.db` messages, 53 tool results):
Σ tool payload = **195,473 chars**; 3 payloads exceed 20 KB (two `search_files`
29,334/28,238 and one `read_file` 48,396). `search_files`/`read_file` returned
`/Users/adam.durham/.hermes/config.yaml` **4 times** (duplicate reads).

Config keys (top-level, confirmed via `hermes config get`):
`file_read_max_chars = 100000`; `tool_output.max_bytes = 50000` (+ `max_lines 2000`,
`max_line_length 2000`).

Candidate caps and the cost model (chars→tokens from the exo log: call 7's 48,396-char
`read_file` = 14,271 delta rows ⇒ **3.39 chars/token**):

| change | old | proposed | chars removed | tokens | % of turn Δrows (100,828) |
|---|---|---:|---:|---:|---:|
| `tool_output.max_bytes` | 50000 | **20000** | 45,968 | 13,560 | 13.5 % |
| `file_read_max_chars` | 100000 | 40000 | 8,396 | 2,477 | 2.5 % |

Estimated turn-wall saving (prefill only): **50.7–63.9 s = 2.8–3.5 % of the
1814 s turn wall** (3.3–4.2 % of model time). The `tool_output` cap (20 K)
subsumes the 40 K read cap for the >20 K payloads.

**Verdict: recommend-with-numbers; do not apply blind.** The cap sits right at
the campaign's ≥3 % adoption bar (2.8–3.5 %), so it is genuinely marginal and
needs the interleaved replay to confirm. Behaviour risk: truncation can force
more calls (a `read_file` truncated at 20 K may be re-read with offsets) — this
can only be seen live.

**Exact change + revert (for the owner to apply after the cluster is verified):**
```
hermes config set tool_output.max_bytes 20000        # revert: hermes config set tool_output.max_bytes 50000
hermes config set file_read_max_chars 40000          # revert: hermes config set file_read_max_chars 100000
```
(No quality battery is required for pure truncation caps; a battery **is**
required if the system prompt is edited — not proposed here.) Note: this affects
**all** sessions.

Complementary free win (no config): the system prompt already invites re-reads;
a prompt addendum "never re-read a file already in context; use offset/limit;
prefer grep-style targeted reads" would have removed the 4× `config.yaml` re-read
directly (but is a system-prompt edit ⇒ requires the quality battery).

---

## 5. Branches, artifacts, reproducibility

- **Branch**: `deploy/next15-levers` (off `deploy/next14-gamma`). Working tree
  `/private/tmp/levers-wt` (`/private/tmp` is tmpfs). **Push: see final summary**
  — the branch is committed + pushed from the parent worktree session; SHA in the
  final report.
- **Artifacts** (all in the worktree; `docs/benchmarks/phase19-latency/…`):
  - `levers-postreboot-2026-10-07.md` (this file)
  - `raw/phase1-accounting.md` — accounting table + reasoning_effort PARTIAL
  - `raw/phase2-delta-audit.md` — 0-miss audit + entry-point×knob matrix
  - `raw/phase3-kernel-sourceread.md` — EXL3 dispatch/dequant/padding source-read
  - `raw/phase3-kernel-microbench.md` + `raw/phase3-kernel-microbench.json` — the run
  - `bench/exl3_dense_smallm_probe.py` — the (synthetic) microbench script
- **Relaunches: 0 of 3 used.** No cluster env changed; no file on either Studio edited.
- Predecessors (for context, on `deploy/next14-gamma`): `README.md`,
  `agentic-replay.md`, `gamma-optimality.md`, `round-loop-map.md`,
  `raw/A1-real-turn-split.md`.

## 6. Ranked "pullable next" (for when the cluster returns)

1. **Unlock the disks + relaunch** `deploy/next13 @ f0840af1c`, then **Phase-0
   cold baseline** (benign + agentic 100K, 3 reps) and the drift rep. Gate: stop
   if cold ≥150 ms.
2. **Settle Phase 3** with the one batched mega-eval (§3.4) — cheap, decisive,
   and answers whether the dense lever exists at all.
3. **Payload-cap replay** (§4) if a ≥3 % confirmation is wanted before applying.
4. **Chunk-4096 endpoint arm** — go-condition already met (41 % of delta rows in
   >4096 calls), 1–2 relaunches, weak design; only if the prefill term is the target.
5. **Image-path knob fix** (Phase 2 finding) — a real latent bug
   (transient-budget inert on image requests); low risk, no perf number attached.
   → **DONE in Part II (§11), committed `c7fc2a9b4`.**

---

# PART II — RESUMED RUN (2026-10-07 16:25 → 17:50 CDT)

The nodes came back (owner unlocked the disks) and this PM resumed the campaign
that Part I had to stop. **Every live phase below was executed.** Cluster:
`deploy/next13 @ f4bb14746`, 2× Mac Studio M4 Max, TP2 over jaccl RDMA.

| # | Phase | Part-I status | Part-II result |
|---|---|---|---|
| 0 | Cold baseline + drift gate | BLOCKED | **PASS** — benign 100K cold **139.13 ms/round** (vs 135.6 morning champion, 154.1 degraded). §8. |
| 3 | Kernel go/no-go | INCONCLUSIVE | **SETTLED = EXHAUSTED** — one batched mega-eval; the ~55 GB/s plateau persists. §9. |
| 4 | Payload-cap validation | recommend-only | **REPLAYED** — 20K cap saves **49.9 s cold prefill = 3.26 % model-time / 2.75 % turn-wall**. Marginal vs the ≥3 % bar. §10. |
| 2b | Image-path knob gap | open bug | **FIXED + committed** `c7fc2a9b4`; 208 tests pass. §11. |

**Relaunch budget: 3 available, 2 used** (one VOID, one healthy), 1 remaining.

## 7. The reboot did NOT clear the degraded-clock state — and a mid-prefill jaccl hang

This is the most important operational finding of the resumed run.

The **raw-GPU canary rule was violated** at the 16:28 relaunch: no canary was run
after the 15:50 boot before benching. The first baseline then produced a **void**
result, and the hardware was diagnosed live:

- **studio2 was in the pre-existing DEGRADED-CLOCK state at ~618 MHz** (lowest bin),
  ~92–97 % residency, ~3–6 W. PM's own canary, 3× fp16 4096³ matmul:
  **studio1 = 14.09 TFLOPS** (~1578 MHz) vs **studio2 = 4.07 TFLOPS** (3.49/4.07/4.09).
  `powermetrics` during the window: studio2 active residency parked on the 618 MHz
  bin, *no thermal warning* — the exact signature of the 2026-09-22 note.
- **This was AFTER the 15:50 reboot**, so the standing "a reboot clears it" rule
  **did NOT hold this time.** Flagged as new information.
- The workload then **hung in a jaccl collective mid-prefill**: both runners'
  stderr froze at 16:50:48, jaccl `call_id` stuck at **570 on BOTH nodes**, all
  threads blocked in `__psynch_cvwait`/`pthread_cond_wait` (waiting, not
  computing), TB nets still pingable → a rendezvous/collective hang, not link-down.

**Recovery (PM-owned):** killed the client, then **rebooted BOTH nodes with the
correct tool** — `cd ~/repos/exo && ./reboot-node.sh studio1 studio2`
(`fdesetup authrestart` + FileVault auto-unlock; `--check` dry-run passed on both
first). **NEVER** raw `sudo shutdown -r now` — that left them at the FileVault
screen for ~30 min earlier today. Both returned ssh-able + auto-unlocked in ~70 s.

**Canary after the reboot — HEALTHY on both** (3× fp16 4096³):
`studio1 9.7 ms 14.22 TFLOPS | studio2 9.7 ms 14.23 TFLOPS` (run0 each ~7.3–7.4,
the other two ~14.2–14.5 — warm-up transient only). Then relaunched, and
canaried **again** post-`READY 2/2`: `14.07 / 14.15 TFLOPS`. Final idle canary at
17:46: `studio1 14.13 | studio2 14.82 TFLOPS`.

**Rules recorded:** (a) run the raw-GPU canary after **every** boot and after
`READY`, before any bench — the 16:28 window was void precisely because it was
skipped; (b) a degraded node is a **reboot target**, not something to bench
through; (c) use `./reboot-node.sh`, never raw `shutdown -r now`.

## 8. Phase 0 — cold baseline (healthy cluster)

Post-reboot, canaries healthy, warm-up + settle done. `phase19_round_measure.py
--depth 100000 --reps 3` on `deploy/next13 @ f4bb14746`:

| rep | prompt_tok | completion | ttft_s | decode_s | decode_tps | ms/round | mean_acc |
|---:|---:|---:|---:|---:|---:|---:|---:|
| 0 (cold) | 100039 | 955 | **372.65** | 35.09 | 27.18 | – | – |
| 1 | 100039 | 955 | 4.22 | 34.70 | 27.49 | **139.37** | 2.8353 |
| 2 | 100039 | 955 | 4.17 | 34.58 | 27.59 | **138.89** | 2.8353 |

**GATE: PASS.** median **139.13 ms/round** (≤ ~140). Cold prefill 372.65 s =
**268.4 rows/s**. `mean_accepted 2.8353`, `gamma_implied 1.521`.

Interpretation: the reboot **cleared the thermal/driver state** — 139.13 is the
champion (morning 135.6, degraded 154.1). **VOID prior window** (16:28): rep0
ttft 1066 s and rep1 143.5 ms/round — that was the degraded node2, recorded here
only to mark it unusable.

## 9. Phase 3 — SETTLED: **EXHAUSTED** (one batched mega-eval)

The Part-I §3.4 / §6 item-2 decisive test: build the **entire dense set (15
linears × K) as ONE `mx.eval`**, no per-eval amortisation, on an idle machine.
Artifacts: `raw/phase3-kernel-megaeval.{md,json,stdout.txt}`,
`bench/exl3_dense_smallm_probe_megaeval.py` (laptop M4 Max, mlx 0.32.0.dev; the
nodes' 128 GB is fully resident with the model so the synthetic probe ran on the
laptop; load 5.07–5.40/14).

```
M=4: mega-eval 60.70 us/call  vs  prior amortised K=64 64.18 us/call  ->  ratio 0.946  [PLATEAU PERSISTS]
     per-op floor inside one 600-op graph = 1.818 us/op  (~3% of a call)
     A 58.1 GB/s (12.9% of 450) | B 247.5 GB/s | roofline C 117.53 us/L | compute E 75.22 us/L
     x40L:  A 36.419 ms   B 45.593 ms   C 4.701 ms   E 3.009 ms
```

- Removing the amortisation assumption moved per-call cost by **5.4 %** (inside
  noise) — **no hidden submission cost** was revealed. K=1→80 per-call is flat
  (~77.5→60.7 µs, saturating by K≈8) while the graph grows 10×.
- Arm B (fp16, decode-free, 5.3× the bytes) is also pinned at 247 GB/s — a plain
  matmul can't reach 450 at these shapes either; measured streaming BW was
  **343.7 GB/s**. So 450 GB/s is a large-stream number, not reachable here.
- `dispatch_count`/`gpu_time_ns` still report 0 (not wired on this build).

**Verdict: EXHAUSTED.** The ~55 GB/s plateau is a genuine kernel throughput, not a
per-eval launch-shape artefact. **No ≥2 ms/round dense-GEMM lever exists**; a
small-M rewrite / batched-eval change does **not** help. This is not a claim of an
irreducible hardware bound (arm B fails 450 too) — what is ruled out is
batching/launch-shape and a decode small-M rewrite. Phase-7 (kernel impl) stays
closed.

## 10. Phase 4 — payload cap replayed on the cluster (interleaved)

Harness `bench/phase19_agentic_trunc_cap.py` (new; `--cap-chars` truncates each
tool result's content before rendering, emulating Hermes `tool_output.max_bytes`).
Verified: the cap truncates **exactly the 3 payloads >20 KB**
(48,396 / 29,334 / 28,238 chars), removing **45,968 chars** — −23.5 % of the
195,473-char tool payload. Interleaved arms (orig, then cap; 3 timed reps each):

| arm | prompt_tok | ms/round (reps) | decode_tps | mean_acc | rep0 prefill |
|---|---:|---|---:|---:|---:|
| ORIGINAL | 91,044 | 153.22 / 153.68 / **153.40** | 20.609 | 2.1614 | 357.96 s |
| CAP 20K | 78,225 | 151.69 / 151.87 / **151.87** | 21.288 | 2.2395 | 308.03 s |

- prompt delta = **12,819 tok (−14.08 %)**. Prefill rate identical (254.3 vs
  254.0 rows/s) → the delta is purely size.
- Cold-prefill saving = **49.93 s** = **3.26 % of model time** (1530.59 s) /
  **2.75 % of the real 1814.45 s turn wall**. Delta-row share 45,968/100,828 =
  **13.45 %** (chars; 12,819/100,828 = 12.7 % in measured tokens).
- ms/round medians differ 1.0 % with **disjoint** 3-rep IQRs (153.22 > 152.20),
  but the arms also differ in `mean_accepted` (2.16 vs 2.24) → decode-side
  acceptance is partly responsible, so the ms/round gap overstates the prefill win.

**Verdict: recommend-with-numbers; not a ≥3 % turn-wall win.** 2.75 % turn-wall
(3.26 % model-time) sits **just below** the campaign's ≥3 % adoption bar → **the
cap is NOT applied**. It is trivially reversible and affects **all** sessions, so
the write-guarded config change is left as a recommendation:

```
hermes config set tool_output.max_bytes 20000     # revert: hermes config set tool_output.max_bytes 50000
```

Behaviour caveat: a payload truncated at 20 K can be **re-fetched** (a `read_file`
re-read with offsets), which this static replay cannot capture — the live risk can
only increase the cost, not reduce it.

## 11. Phase 2b — image-path knob gap: **FIXED** and committed (`c7fc2a9b4`)

Verdict: the `long_threshold=10**9` pin in mlx-lm `_prefill_planned` is
**deliberate** (cold==reused bitwise op-sequence contract) and is **left intact**.
But the contract is **inert on the exo image path** (`engine.py:765` `keep = … and
embeddings is None`; image caches are closed every turn), so the fix is exo-side
and touches no fork file: the plan the engine passes in is now **budget-shaped**
via a new pure function `plan_image_prefill_pieces()` (`session.py`), used by
`engine._image_prefill_plan` (`engine.py`). Under the shipped 2048 MB budget it is
**byte-identical to the old uniform tail** (no regression possible); only a
tightened `EXO_PREFILL_TRANSIENT_BUDGET_MB` changes behaviour — exactly the fix.
Span atomicity beats the budget (a boundary inside an image span is a hard error).

Committed + pushed `origin/deploy/next15-levers` **`c7fc2a9b4`** (+303/−14, 4
files). **Verified independently by the PM** in a clean worktree
(`PYTHONPATH=<wt>/src:<wt>/mlx-lm … pytest`): **208 passed** (dsv41 tests dir),
33 in `test_dsv41_session.py`. Not deployed (needs a relaunch + sign-off; it is
behaviour-neutral at the default budget).

## 12. Phase 0 drift probe

Re-ran the benign 100K baseline after ~50 min of campaign load (same healthy
process): **144.79 ms/round** (144.74/144.83), 27.099 t/s, `mean_accepted 2.9289`,
cold prefill 373.69 s (267.8 rows/s).

| | ms/round | decode_tps | mean_acc |
|---|---:|---:|---:|
| baseline (17:17) | 139.13 | 27.491 | 2.8353 |
| drift (17:46) | 144.79 | 27.099 | 2.9289 |

Δ ms/round **+4.1 %**, but that is driven by a **+3.3 % acceptance change**;
**throughput Δ = −1.4 %** (27.491 → 27.099 t/s). No large drift — this is a
healthy ~1–4 % band, not the 154 ms degraded regime. The acceptance variation is
the decode-side knob and is why ms/round moves more than throughput.

## 13. Relaunch budget, final cluster state, artifacts

- **Relaunch budget: 3 available — USED 2.** (1) 16:28 production relaunch →
  VOID (degraded studio2 + jaccl hang). (2) 17:01 post-reboot relaunch → healthy
  canaries + Phase-0 PASS. **1 remaining.**
- **Final cluster:** `deploy/next13 @ f4bb14746`, **READY 2/2**, both runners
  Ready, canaries **14.13 / 14.82 TFLOPS** at idle, no active user/API traffic
  (idle-guard clean: last exo POST ≤10 min during the window, 0 open `api_calls`
  in the last hour, last `provider=custom` call ended ~09:55). Cluster left
  serving production.
- **Artifacts** (branch `deploy/next15-levers`, origin/adurham/exo):
  - `raw/phase3-kernel-megaeval.{md,json,stdout.txt}` — the decisive kernel test
  - `bench/exl3_dense_smallm_probe_megaeval.py` — the mega-eval probe
  - `bench/phase19_agentic_trunc_cap.py` — the Phase-4b truncating replay harness
  - `raw/phase0-live/{benign100k_cold2,agentic_orig,agentic_cap,benign100k_drift}.{json,log}`
    — the raw live-run outputs
  - `src/exo/worker/engines/mlx/dsv41/{session.py,engine.py,vision.py}` +
    `tests/test_dsv41_session.py` — the Phase-2b fix (via `c7fc2a9b4`)
- **Not applied / still open:** Payload cap (§10 — recommendation only, below the
  ≥3 % bar); Phase-6 chunk-4096 endpoint arm (never needed; go-condition was met
  but the kernel verdict closed the dense track).
