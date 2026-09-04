# I11 STEP 1B — did step 1 measure the DEPLOYED kernel? (no) + the 3-bit extension

**Two answers, both measured:**

1. **Step 1 did NOT measure the deployed kernel.** Its harness hardcoded `mode="affine"` on
   every arm. Production routed experts run `mode="mxfp4"`, which is a **different compiled
   kernel family** with a **different byte footprint**. The deployed configuration is
   **339.42 µs/call**, not the 390.29 µs/call step 1 published as "4-bit" — the original number
   is **15.4 % too slow**. The published ratios need restating, and they are restated in §5.
2. **3-bit is on the fast path, but it is a weak lever.** `affine` 3-bit measures
   **325.82 µs/call = 0.9599× the deployed mxfp4 arm — a 4.0 % gain, not the 25 % the naive
   0.75× byte expectation implies.** The kernel is fine; the *format* eats the saving. §4
   shows exactly where.

Everything below is measured on the live cluster with the step-1 method of record
(chained-graph, no serial sync, interleaved arms, n=5, ranges). Nothing here is extrapolated
except one clearly-labelled arithmetic projection in §4, which is marked **NOT MEASURED**.

---

## 1. THE GAP — what mode/group_size did the original harness actually use?

**Read from source, not assumed.** `tmp/perf-campaign-2/round8/i11_gather_qmm_bits_microbench.py`
hardcodes affine in two places, and there is no CLI flag to change it:

| Site | Line | Code |
|---|---|---|
| weight construction | `:96` | `packed = mx.quantize(src, group_size=group_size, bits=bits, mode="affine")` |
| the timed call | `:145` | `mode="affine",` inside `Projection.__call__`'s `mx.gather_qmm(...)` |
| results metadata | `:292` | `"mode": "affine",` — hardcoded into the JSON blob |
| CLI surface | `:266-267` | `--bits` and `--group-size` exist; **no `--mode`** |

`group_size` defaulted to **32**, which *does* match production.

So the original arms were:

| step-1 arm | mode used | group_size | matches production? |
|---|---|---|---|
| `bits=6` | **affine** | 32 | n/a — no 6-bit is deployed |
| `bits=5` | **affine** | 32 | n/a — no 5-bit is deployed |
| **`bits=4`** | **affine** | 32 | ❌ **NO — production is `mxfp4`** |

**Verdict: the mode mismatch is real, and it matters.** Production routed experts are
`{"group_size": 32, "bits": 4, "mode": "mxfp4"}` (`mlx-lm/mlx_lm/models/deepseek_v4.py`
`make_quantization_config()`, applied to every `.ffn.switch_mlp.*_proj`; confirmed live by
`QuantizedSwitchLinear()` in the runner log). Step 1's own §7 correctly *identified* that
production is mxfp4 — but its **measurement still used affine for the bits=4 arm**, so the
number it published never described the deployed kernel.

This is not a cosmetic label difference. `affine` and `mxfp4` dispatch **different compiled
Metal kernels from different source files**:

- `affine` → `affine_gather_qmv_fast_bfloat16_t_gs_32_b_4`, instantiated in
  `mlx/backend/metal/kernels/quantized.metal:174-182`.
- `mxfp4` → `mxfp4_gather_qmv_fast_bfloat16_t_gs_32_b_4`, instantiated in
  `mlx/backend/metal/kernels/fp_quantized.metal` via
  `instantiate_quantized_modes(type, mxfp4, 32, 4)`.

The kernel name is assembled as `mode + "_gather_qmv_fast_" + ...` at
`mlx/backend/metal/quantized.cpp:1084-1092`, where `mode` comes from
`quantization_mode_to_string(mode_)` (`:1862`). Same dispatch *decision* (§3), different
compiled binary, and — decisively — **different bytes moved** (§4).

Step 1's structural conclusion that the fast/slow gate depends on N and K only, never on
`bits`, is **re-verified and still correct**, and it now extends to mode: at M=4 / B=24 /
`sorted_indices=False`, every mode falls through the same chain in `GatherQMM::eval_gpu`
(`quantized.cpp:1889` needs `M==1`; `:1914` and `:1940` need `right_sorted_`; `:1962` needs
`M >= vector_limit ≈ 10-12`) and lands on `gather_qmv` at `:1983`, whose only fast/slow gate is
`N % 8 == 0 && K % 512 == 0` (`:1084`). All six arms below take the `_gather_qmv_fast_` variant.

---

## 2. SUPPORTED / UNSUPPORTED — probed, never assumed

The harness calls real `mx.quantize` **and** real `mx.gather_qmm` for each candidate and
records the actual exception. **No unsupported combination was substituted with affine.**

**UNSUPPORTED (real error text from the live run):**

| combination | error |
|---|---|
| `mxfp3` g32 b3 | `ValueError: [quantize] Invalid quantization mode 'mxfp3'.` |
| `mxfp4` g32 **b3** | `ValueError: [quantize] mxfp4 quantization requires bits to be 4 but got 3.` |
| `mxfp5` g32 b5 | `ValueError: [quantize] Invalid quantization mode 'mxfp5'.` |
| `mxfp6` g32 b6 | `ValueError: [quantize] Invalid quantization mode 'mxfp6'.` |
| `mxfp4` **g64** b4 | `ValueError: [quantize] mxfp4 quantization requires group size 32 but got 64.` |

**SUPPORTED:** `mxfp4/g32/b4`, `mxfp8/g32/b8`, `nvfp4/g16/b4`, and `affine/g32/b{3,4,5,6,8}`.

This is corroborated structurally: the mode enum is closed at
`mlx/primitives.h:155` — `enum class QuantizationMode { Affine, Mxfp4, Mxfp8, Nvfp4 };` — and
the MX kernels are instantiated at exactly three (mode, gs, bits) triples in
`fp_quantized.metal`: `(nvfp4,16,4)`, `(mxfp8,32,8)`, `(mxfp4,32,4)`. There is no
mxfp3/mxfp5/mxfp6 to fall back to, and each MX mode is locked to one bit width and one group
size (`mlx/ops.cpp:4975-4989`).

**Finding for the decision package: there is no MX-mode option at any precision other than 4
and 8 bits.** Any 3-, 5- or 6-bit deployment is necessarily an **affine** deployment, which
(§4) carries a materially heavier metadata footprint than the MX modes.

---

## 3. METHOD

Harness: `tmp/perf-campaign-2/round8/i11_step1b_mode_and_3bit.py` — a **copy-and-extend** of the
step-1 harness; the original committed artifact is untouched. Method of record preserved
verbatim:

1. **Chained-graph construction. SERIAL SYNC IS NOT USED.** `CHAIN_LEN=300` SwitchGLU-equivalent
   forwards, each a genuine data dependency of the next (`carry = x + 1e-9*mean(out, axis=-2)`),
   with a **single `mx.eval()` at the end** followed by `mx.synchronize()`. No `mx.eval()` inside
   any timing loop.
2. **Rotated routing indices**, pool of 64 independently drawn `(4,6)` index sets, so no expert
   weight set sits warm in cache across the chain.
3. **Measured empty-graph baseline subtracted.** Median **13.65 µs/call** (range 12.13–19.06),
   the identical carry chain with the three `gather_qmm` calls removed. All µs/call below are
   **net**. Baseline is 2–4 % of signal.
4. **Elision detector.** µs/call flat across chain length 100/200/300 for every arm (e.g.
   mxfp4 349.68 / 349.32 / 349.49; affine-3 338.57 / 338.26 / 338.48). The 300 dependent calls
   really execute.
5. **Production call signature**, unchanged except `mode`: `mx.gather_qmm(..., rhs_indices=idx,
   transpose=True, group_size=32, bits=B, mode=MODE, sorted_indices=False)` —
   `QuantizedSwitchLinear.__call__` (`switch_layers.py:76-91`). `sorted_indices=False` is correct
   at M=4 (`indices.size = 24 < 64`, `switch_layers.py:182`).
6. **Bytes from real `.nbytes`**, never assumed — weight + scales + biases summed off the actual
   quantized arrays, ÷256 experts × 24 (row, expert) pairs.
7. **n=5 reps per arm; ranges reported, never bare means.**
8. **group_size = 32 on every arm** (the one supported MX group size, and production's).

**Interleaving upgrade.** Step 1 had 3 arms and alternated forward/reverse. With **6** arms that
would pin each arm to only 2 of 6 positions, so the order is now **rotated by rep index and
reversed on odd reps** — each arm visits a spread of positions across the 5 reps. Recorded in
the JSON as `rep_order`:

```
rep1: b3, mxfp4, b4, b5, b6, mxfp8
rep2: b3, mxfp8, b6, b5, b4, mxfp4
rep3: b4, b5, b6, mxfp8, b3, mxfp4
rep4: b4, mxfp4, b3, mxfp8, b6, b5
rep5: b6, mxfp8, b3, mxfp4, b4, b5
```

**One arm resident at a time** (built / measured / freed inside each rep), the memory discipline
step 1 root-caused. Interleaving is preserved at the rep level. The OOM back-off guard did not
fire during this run.

**Run location:** macstudio-m4-2 (the non-API node), via that node's own
`~/repos/exo/.venv/bin/python`, script staged with `scp` then run directly. **No cluster process
was killed, restarted, relaunched or reconfigured** (§7).

---

## 4. RESULTS

`chain_len=300`, `n_pool=64`, `reps=5`, `group_size=32`, M=4, deployed per-rank shapes
(gate/up `(256,1024,4096)`, down `(256,4096,1024)`), net of the 13.65 µs/call baseline. Ranges
are min–max over n=5.

### 4a. Head-to-head: deployed mxfp4 vs the original affine-4 arm

| arm | mode | µs/call (range, n=5) | median | bytes/call | achieved GB/s | % of 546 |
|---|---|---|---|---|---|---|
| **DEPLOYED** | **mxfp4** g32 b4 | **338.63 – 340.48** | **339.42** | 160,432,128 | 472.7 | 86.6 % |
| step-1's "4-bit" | affine g32 b4 | 389.31 – 392.54 | 391.62 | 188,743,680 | 482.0 | 88.3 % |

**They differ materially: the deployed mxfp4 kernel is 52.20 µs/call faster — the original
affine-4 number is 15.4 % too slow (1.1538×).** The two n=5 ranges do not come close to
overlapping (338.63–340.48 vs 389.31–392.54). **339.42 µs/call is the production-relevant
number.**

**Why — and it is not "mxfp4 is a better kernel".** It is byte accounting, and the harness read
it off the real arrays:

| arm | weight bytes/expert | scales | biases | total/expert | metadata share |
|---|---|---|---|---|---|
| mxfp4 g32 b4 | 2,097,152 (uint32 pack) | 131,072 (**uint8** e8m0) | **none** | 2,228,224 | **5.9 %** |
| affine g32 b4 | 2,097,152 (identical) | 262,144 (bfloat16) | 262,144 (bfloat16) | 2,621,440 | **20.0 %** |

Identical 4-bit weight payload. affine carries **4× the metadata** — bf16 scales *and* a bf16
bias array that the MX modes simply do not have. Per-byte efficiency actually *favours* affine
slightly (482.0 vs 472.7 GB/s), but mxfp4 moves **15 % fewer bytes**, and this path is
bandwidth-bound, so mxfp4 wins by 13.3 %.

**Cross-validation against step 1.** This run independently reproduced all three of step 1's
affine arms on a different day with a rewritten harness:

| affine arm | step 1 median | step 1b median | delta |
|---|---|---|---|
| b6 | 543.37 | 542.73 | −0.12 % |
| b5 | 471.04 | 469.11 | −0.41 % |
| b4 | 390.29 | 391.62 | +0.34 % |

All within 0.5 %. **Step 1's measurements were sound; only its choice of arm was wrong.**

### 4b. The 3-bit extension

| arm | µs/call (range, n=5) | median | bytes/call | GB/s | ratio vs deployed | byte ratio | band |
|---|---|---|---|---|---|---|---|
| affine g32 **b3** | 325.81 – 327.76 | 325.82 | 150,994,944 | 463.4 | **0.9599** | 0.9412 | **FAST** |

`mxfp3` — **UNSUPPORTED** (§2). 3-bit is affine-only.

**Band verdict: FAST, on both framings.**
- vs deployed mxfp4: time ratio 0.9599 against byte ratio 0.9412 → time/byte = **1.020**.
- Like-for-like vs affine-4 (same mode, so the cleanest kernel comparison): 325.82/391.62 =
  **0.8320** against byte ratio 0.8000 → time/byte = **1.040**, comfortably inside step 1's
  pre-registered `FAST ≤ 0.90×` band.

Time tracks bytes in both framings. A generic/slow path would show time *decoupled* from bytes;
it does not. Achieved 463.4 GB/s (84.9 % of spec) is the lowest of the six arms but sits in the
same 85–89 % band as everything else — 3-bit's 8-values-in-3-bytes packing is very slightly less
bandwidth-efficient, not a fast-path fall-off.

**But the lever is much weaker than the byte expectation suggests, and this is the finding that
matters.** The brief's expected 0.75× byte ratio **holds exactly on the weight payload**
(1,572,864 vs 2,097,152 bytes/expert = 0.7500) — and then dies, because dropping to 3 bits means
dropping to **affine**, which re-adds bf16 scales + bf16 biases:

| arm | weight/expert | metadata/expert | total/expert | vs mxfp4 |
|---|---|---|---|---|
| affine b3 | 1,572,864 | 524,288 (**25.0 %**) | 2,097,152 | 0.9412× |
| mxfp4 b4 | 2,097,152 | 131,072 (5.9 %) | 2,228,224 | 1.0000× |

The 25 % weight saving shrinks to a **5.9 % total-byte saving**, which buys a **4.0 % measured
speedup**. The metadata share *rises* as bits fall, so this only gets worse below 3 bits.

> **NOT MEASURED — arithmetic projection only, flagged as such.** A hypothetical MX-style 3-bit
> (uint8 e8m0 scales, no biases) would be 1,703,936 bytes/expert = **0.765× of mxfp4**, which at
> the ~1.02 time/byte factor observed here would land near 0.78×. **No such mode exists** (§2),
> no such kernel is compiled, and this number was **not measured**. It is stated only to make the
> point that the weak 3-bit result is a *format* limitation, not a kernel limitation — closing it
> would require new mlx work, not a config change.

---

## 5. RESTATED TABLE — deployed mxfp4 = 1.000×

All six arms, one interleaved n=5 run, net of the same 13.65 µs/call baseline, group_size=32
throughout, M=4, deployed per-rank shapes.

| arm | **mode** | bits | µs/call (range, n=5) | median | GB/s | % of 546 | **ratio vs deployed** | byte ratio | time/byte | verdict |
|---|---|---|---|---|---|---|---|---|---|---|
| 3-bit | affine ⚠ | 3 | 325.81 – 327.76 | 325.82 | 463.4 | 84.9 % | **0.9599** | 0.9412 | 1.020 | FAST — the only downward lever |
| **DEPLOYED** | **MX (mxfp4)** | **4** | **338.63 – 340.48** | **339.42** | 472.7 | 86.6 % | **1.000** | 1.000 | 1.000 | **reference** |
| 4-bit affine | affine ⚠ | 4 | 389.31 – 392.54 | 391.62 | 482.0 | 88.3 % | 1.1538 | 1.1765 | 0.981 | ⚠ **not deployed** — step 1's arm |
| 5-bit | affine ⚠ | 5 | 468.57 – 470.98 | 469.11 | 482.8 | 88.4 % | **1.3821** | 1.4118 | 0.979 | upward / quality |
| 6-bit | affine ⚠ | 6 | 540.33 – 547.02 | 542.73 | 486.9 | 89.2 % | 1.5990 | 1.6471 | 0.971 | upward |
| 8-bit | **MX (mxfp8)** | 8 | 637.57 – 658.66 | 641.35 | 485.6 | 88.9 % | 1.8895 | 1.9412 | 0.973 | deployed for shared_experts + attention |

**Row-comparability flags:**

- ⚠ **The four `affine` rows are not mode-matched to the deployed row.** They are the honest
  cost of those precisions *as they would actually have to be deployed* (no MX mode exists at 3,
  5 or 6 bits), so the ratios are decision-valid — but they conflate a bits change with a mode
  change. The affine-4 row is included precisely so the mode effect can be separated from the
  bits effect: **affine-4 vs mxfp4 = pure mode, 1.1538×; affine-3 vs affine-4 = pure bits,
  0.8320×.**
- ⚠ **`mxfp8` is a reference point, not an option for routed experts.** It is what
  shared_experts and attention already run. Moving routed experts to it is not on the table; it
  is here to price the MX-mode 8-bit kernel on the same axis.
- Every row took `<mode>_gather_qmv_fast_bfloat16_t_gs_32_b_<bits>` — the fast templated path
  (§1). No row fell off it.
- All rows land at 84.9–89.2 % of the 546 GB/s M4 Max spec. No row exceeds peak (which would
  indicate a cache artifact or byte-accounting error) and none is implausibly low.

### The two practical deltas, in plain terms

**Moving UP from deployed 4-bit to 5-bit (the quality upgrade): the routed-expert MoE kernel gets
38.2 % slower.** 339.42 → 469.11 µs/call, +129.69 µs per call, ratio **1.3821×**. Step 1's
published "5-bit is 0.8669× of 6-bit" made this sound cheap; against the arm production actually
runs it is not a 13 % discount, it is a **38 % surcharge**. Two things stack: 5 bits carries 25 %
more weight payload than 4, *and* leaving mxfp4 for affine adds the bf16 scales+biases the MX
format does not have. A 5-bit deployment cannot avoid the second cost — no `mxfp5` exists.

**Moving DOWN to 3-bit (the remaining throughput lever): the kernel gets 4.0 % faster.**
339.42 → 325.82 µs/call, −13.60 µs per call, ratio **0.9599×**. The kernel is genuinely on the
fast path — but the naive "3/4 of the bytes → 25 % faster" expectation is wrong by a factor of
six, because leaving mxfp4 for affine-3 gives back most of the weight saving in metadata. **On
this axis, at these shapes, downward precision is close to exhausted.** A 4 % kernel-level gain
is before any end-to-end dilution by attention, shared experts, MTP and communication, and comes
at whatever 3-bit costs in quality.

---

## 6. CONTENTION

**Could live-cluster GPU contention have biased the arms? Yes in principle. Controlled for and
measured, exactly as in step 1.**

- The cluster was **LIVE throughout** — a hard constraint of this round, not an oversight. One
  other round-8 subagent shares these nodes.
- **Idle-state check.** `exo_gpu_usage_ratio` was **0.028 on the bench node (m4-2) before the
  run** and, sampled after the benchmark exited, **0.026 on m4-2 / 0.030 on m4-1** — the live
  runner was essentially idle. The ~0.97–0.99 readings recorded *during* the run are my own
  benchmark saturating the GPU; the ratio returns to 0.026 the moment it exits. Sampled before
  and after **every** timed block and stored per-rep in the JSON so this is checkable, not
  asserted.
- **Arms interleaved with rotation** (§3) — with 6 arms, rotation is what actually decorrelates
  arm identity from position. Any monotonic drift (thermal, or a competing job ramping) is spread
  across all six arms.
- **Spread is tiny and the ranges do not overlap.** Every arm's full n=5 range is **under 0.6 %
  wide** except mxfp8 (3.3 %, one slow rep at 658.66), and every pair of adjacent arms is cleanly
  separated: 325.81–327.76 / 338.63–340.48 / 389.31–392.54 / 468.57–470.98 / 540.33–547.02 /
  637.57–658.66. The two numbers the conclusions turn on — deployed mxfp4 and affine-4 — are
  separated by 49 µs with ranges 2 µs wide. A contention artifact large enough to manufacture
  that would have to land on the same arm in all five reps despite rotated ordering, while
  leaving each arm's own spread under 1 %.
- **Residual honest caveat, unchanged from step 1:** a microbench on a node that is also hosting
  a live inference runner is never as clean as a quiesced node. The **ratios** (what every verdict
  here turns on) are trustworthy; the **absolute** µs/call could shift a few percent on a fully
  idle machine. Note the step-1 reproduction in §4a is itself evidence the absolutes are stable
  across days.

---

## 7. CLUSTER UNTOUCHED

No process was killed, restarted, relaunched or reconfigured. Nothing was written to the models
directory and **no real model weights were quantized** — every array in this benchmark is
synthetic `mx.random.uniform` at the deployed *shapes* only. The script was staged to `/tmp` via
`scp` and run directly with the node's venv python (the scp-then-run pattern step 1 adopted after
its quoting incident).

exo PIDs advanced monotonically with no PID change: m4-1 PID **9390** at etime `01:36:17` at task
start → `01:43:55` at the end. A restart would have reset etime and changed the PID. The live PID
line is pasted in the final summary.

---

## 8. BOTTOM LINE

1. **Step 1's bits=4 number does not describe the deployed kernel.** It measured `affine`;
   production runs `mxfp4`. Deployed = **339.42 µs/call**, 15.4 % faster than the published
   390.29. Everything keyed to that old 4-bit arm needs the §5 table substituted.
2. **The MX family offers nothing at 3, 5 or 6 bits** — mxfp3/mxfp5/mxfp6 do not exist, and
   mxfp4 is locked to bits=4/g32. Any non-4/8-bit deployment is affine, and affine's bf16
   scales+biases cost ~14 percentage points of extra metadata versus MX.
3. **5-bit costs +38.2 %** against the true baseline, not the ~15 % the old 6-bit-keyed table
   implied.
4. **3-bit is FAST but nearly spent: +4.0 %.** The remaining downward precision lever on this
   axis is small, and it is small for a *format* reason (affine metadata), not a kernel reason.

---

## FILES

- `i11_step1b_mode_and_3bit.py` — this task's harness (copy-and-extend of the step-1 harness;
  the step-1 file is unmodified). Adds mode as a first-class arm dimension, a real
  supported/unsupported probe, rotated 6-arm interleaving, and deployed-mxfp4-keyed ratios.
- `i11b_results.json` — full raw results: per-rep samples, per-rep GPU-usage readings,
  `rep_order`, scaling check, the support probe with real error strings, real array
  shapes/dtypes/nbytes.
- `i11b_run.log` — complete stderr of the reported run.
