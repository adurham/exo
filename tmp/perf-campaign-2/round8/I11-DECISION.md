# I11-DECISION.md — expert precision: the decision package
**Round 8, campaign 2. NOTHING WAS SHIPPED. The cluster is unchanged and healthy.**

---

## ⚠️ READ THIS FIRST: the premise of I11 was false

I11 was scoped as *"drop routed-expert precision from **6-bit** to 4-bit for a -33% active-bytes
decode win."* That decision does not exist, because **the routed experts are already 4-bit.**

There is no 6-bit configuration deployed, and the record contains no evidence there ever was one.
The `-33%` win the R7 review projected is **already banked** — it is in the 30-34 t/s the cluster
serves today, not sitting on the table.

**Evidence (verified directly, not taken from a worker report):**
`mlx-lm/mlx_lm/models/deepseek_v4.py:952-982`, `make_quantization_config()`:
```python
mxfp4 = {"group_size": 32, "bits": 4, "mode": "mxfp4"}
mxfp8 = {"group_size": 32, "bits": 8, "mode": "mxfp8"}
experts        = {k: mxfp4 for k,_ in flat_modules
                  if ".ffn.switch_mlp." in k and k.endswith("_proj")}   # ROUTED EXPERTS -> 4-bit
shared_experts = {k: mxfp8 ...}
attn           = {k: mxfp8 for ... ".attn.w" or ".attn.indexer.wq"}
return {"group_size": 64, "bits": 8, "mode": "affine", **experts, **shared_experts, **attn, ...}
```
- Applied at **load time** (`mlx-lm/mlx_lm/utils.py:549-556`), so the on-disk checkpoint's
  `quantization: null` is not the deployed precision — the mxfp4 packing exists only in RAM.
- Live runner log confirms `QuantizedSwitchLinear()` on every `ffn.switch_mlp.{gate,up,down}_proj`
  (`~/exo.log:849-852`).
- Two subagents reached this independently by different routes (on-disk tensor-shape analysis;
  load-path + live log). I then verified the config site and the live log myself.

**Where "6-bit" came from:** it is a label mismatch, not a config. `resources/inference_model_cards/
mlx-community--DeepSeek-V4-Flash-{6,4}bit.toml` exist but **fail validation at load** (missing
`backends` field, `model_cards.py:167`) — they are inert and describe nothing that runs. The serving
model is `deepseek-ai/DeepSeek-V4-Flash-0731`. The bare alias `deepseek-ai/DeepSeek-V4-Flash` 503s
("no admissible placement").

**Consequence for this package:** steps 3 and 4 of the brief (measure decode t/s for a 6-bit arm;
diff 6-bit vs 4-bit output quality across 8 prompts) were **not run**, because both require a 6-bit
baseline that has never been deployed. Building one would mean *manufacturing a regression* and then
measuring the cost of undoing it — a number that describes nothing real and answers no question the
user has. **The honest deliverable is the reframed decision below.**

---

## The decision that actually exists

Production sits at **mxfp4** (4-bit MX, group_size=32) for routed experts. The real options are one
step UP (quality headroom, costs throughput) or one step DOWN (the last throughput lever on this
axis).

### Measured kernel cost — MoE `gather_qmm`, M=4, deployed shapes
Deployed mxfp4 = 1.000x reference. Chained-graph method, n=5, arms rotated+reversed, ranges
non-overlapping, empty-graph baseline (13.65 us) subtracted.

| precision | mode | us/call (range) | median | GB/s | **ratio vs deployed** |
|---|---|---|---|---|---|
| 3-bit | affine ⚠ | 325.81–327.76 | 325.82 | 463.4 | **0.960** (−4.0%) |
| **4-bit (DEPLOYED)** | **MX mxfp4** | **338.63–340.48** | **339.42** | 472.7 | **1.000** |
| 5-bit | affine ⚠ | 468.57–470.98 | 469.11 | 482.8 | **1.382** (+38.2%) |
| 6-bit | affine ⚠ | 540.33–547.02 | 542.73 | 486.9 | 1.599 |
| 8-bit | MX mxfp8 | 637.57–658.66 | 641.35 | 485.6 | 1.890 |

⚠ = **not mode-matched to production.** MX format exists ONLY at 4 and 8 bits, so any 3/5/6-bit
deployment is forced onto `affine`, which carries 4x the metadata (bf16 scales *and* biases, vs
mxfp4's uint8 e8m0 scales and no biases). These rows therefore price those precisions *as they would
actually deploy* — bits and mode change together, and that is the real cost.

**Verified constraint:** `mlx/mlx/primitives.h:155` — `enum class QuantizationMode { Affine, Mxfp4,
Mxfp8, Nvfp4 }`; `kernels/fp_quantized.metal:191-193` instantiates exactly `(nvfp4,16,4)`,
`(mxfp8,32,8)`, `(mxfp4,32,4)`. **`mxfp3`, `mxfp5`, `mxfp6` do not exist** (confirmed by real
dispatch errors, not inference). I checked this myself.

### The two options, in plain terms

**UP to 5-bit (more quality headroom): costs +38.2% expert-kernel time.**
Far worse than the ~15% a naive bits-only reading suggests, because 5-bit forces the drop from MX to
affine — you pay the bit increase *and* the metadata penalty. On a bandwidth-bound decode path this
is a large, real regression. **Not recommended on throughput grounds.**

**DOWN to 3-bit (the last throughput lever): gains only +4.0%.**
The 25% weight-payload saving is real, but affine's metadata re-adds most of it (25% weight cut ->
5.9% total-byte cut -> 4.0% measured). And 4.0% is a *kernel-level* number, before end-to-end
dilution by attention, shared experts, MTP, and collectives — the e2e figure would be smaller still.
Against a known quality cost at 3-bit, **this is a bad trade.** The ceiling here is a *format*
limitation, not a kernel-quality one: 3-bit is confirmed on the fast path (time/byte 1.020).

---

## RECOMMENDATION: no precision change. I11 closes.

Production is at the sweet spot of the available format lattice. The only mode-matched neighbour is
mxfp8 (+89% time). The one remaining downward step buys ~4% kernel-level for a real quality cost.
**The lever I11 was created to evaluate was already pulled, before this campaign began.**

**This is not a quality tradeoff being deferred to the user — it is a decision the measurements
resolve.** No quality battery was needed, because no precision change is worth running one for.

### What could reopen it
Writing a new `mxfp3` mode in mlx (does not exist today) would decouple the 3-bit gain from the
affine metadata penalty — projected ~0.765x bytes vs deployed. That is a **NOT MEASURED arithmetic
projection**, included only to show the ceiling is a format limit. It is a real engineering project
in mlx's Metal backend, not a config flag.

## Disk / rollback (for completeness)
No weights were converted; nothing was deleted or moved. Binding headroom = min(node1 111 GiB,
node2 134 GiB). Analytical target-set sizes: 4-bit ~82.1 GiB (would pass, ~8.9 GiB margin), 5-bit
~103.7 GiB (would fail node1). Moot — no conversion is recommended. The serving checkpoint is
untouched and remains its own rollback path.
