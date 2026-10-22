# oQ2.5e vs EXL3 1017 — cross-format parity, layer 0

Measured 2026-10-22 on the laptop, layer 0 only, from real bytes of both packs.
This answers the one question the 43 unit tests cannot: *does the oQ loader read
the same weights the EXL3 path reads*, or does it permute/transpose something the
synthetic fixtures would not catch.

## Packs

| side | source | what was used locally |
|---|---|---|
| oQ | `mlx-community/DeepSeek-V4.1-Flash-1017-oQ2.5e-ngram-mtp` | mini pack, 9.7 GiB: all of layer 0 + `embed`/`norm`/`head` + all of `mtp.0` |
| EXL3 | `dealignai/DeepSeek-V4.1-Flash-1017-EXL3-2.9bpw-UNCENSORED` on m4-1 | shards `00001`, `00002`, `00041` (1.32 + 5.32 + 5.72 GB) copied read-only from the Studio |

The EXL3 side is the **abliterated** build, so a handful of edited tensors are
expected to differ from the base model the oQ pack was made from.

## Dense projections (oQ 8-bit affine g64 vs EXL3)

| tensor | shape | cos | rel Frobenius |
|---|---|---|---|
| `attn.wq_a` | (1280, 5120) | 0.999779 | 0.0210 |
| `attn.wq_b` | (32768, 1280) | 0.999056 | 0.0435 |
| `attn.wkv` | (512, 5120) | 0.999776 | 0.0212 |
| `attn.wo_b` | (5120, 8192) | 0.999048 | 0.0436 |
| `ffn.shared_experts.w1` | (2304, 5120) | 0.999151 | 0.0412 |
| `ffn.shared_experts.w2` | (5120, 2304) | 0.999151 | 0.0412 |
| `ffn.shared_experts.w3` | (2304, 5120) | 0.999152 | 0.0412 |
| `attn.wo_a` (8 EXL3 slices, group-stacked) | (8192, 4096) | 0.999206 | 0.0398 |

Dense min 0.999048 / median 0.999151. `attn.wo_b` is one of the tensors
abliteration touches and still lands at 0.99905, i.e. the two packs agree to
quantization noise there as well.

## Routed experts, 8 sampled of 384

| stack | oQ bits | cos range | rel Frobenius |
|---|---|---|---|
| `ffn.experts.w1` | 2 | 0.93233 – 0.93275 | 0.380 |
| `ffn.experts.w3` | 2 | 0.93233 – 0.93272 | 0.380 |
| `ffn.experts.w2` | 3 | 0.96924 – 0.97505 | 0.225 – 0.253 |

This is the quantizer talking, not the loader: every sampled expert sits in a
0.4e-3 band, and a layout fault (block permutation, transposed projection, wrong
expert index) produces cos near 0 on the affected stack, not a uniform 0.93.
w2 is 3-bit in both packs and reads correspondingly higher than the 2-bit w1/w3.
Still, 0.93 is the price of oQ's 2-bit expert allocation, and it is the number to
watch in the quality battery — see "Open" below.

## Plain tensors (unquantized in both) — bit-exact expected

| tensor | result | max abs diff |
|---|---|---|
| `attn_norm`, `ffn_norm`, `attn.q_norm`, `attn.kv_norm` | bit-exact | 0 |
| `attn.attn_sink`, `hc_attn_*`, `hc_ffn_*` | bit-exact | 0 |
| `embed.weight` | bit-exact | 0 |
| `ffn.gate.weight` | not exact | 2.980e-08 |
| `ffn.gate.bias` | not exact | 3.882e-03 |

Everything matches bit-for-bit except the routing gate. The gate weight delta is
bf16 rounding (`2.98e-8` on values of order 1), but the **bias delta of 3.9e-3 is
larger than rounding** and is unexplained. It is a 384-vector of router biases, so
its practical effect is a small nudge to expert selection; it is not a layout bug
(a layout fault would show up in the projections, which are exact). Note it as an
open question, and check whether the abliteration touched router biases.

## Two oracle bugs found and fixed on the way

1. `reconstruct_public_mlx` returns exllamav3's stored `[in, out]`; the oQ reader
   and MLX `nn.Linear` use `[out, in]`. The first run therefore reported a shape
   mismatch on every dense tensor and printed `nan` cosines (a missing-cos format
   artifact, not NaN data). `exl3_dense` now transposes.
2. `attn.wo_a` is stored by EXL3 as 8 per-group slices `[heads_per_group*head_dim,
   o_lora]`; once transposed they stack on axis 0 into the oQ `[groups*o_lora,
   heads*head_dim]` layout. Concatenating on the old axis produced `[1024, 32768]`.

## Reproduce

```sh
# EXL3 layer-0 shards, read-only from the Studio
mkdir -p ~/.hermes/cache/scratch/exl3_1017_l0 && cd $_
for s in 00001 00002 00041; do
  scp macstudio-m4-1:"~/.exo/models/dealignai--DeepSeek-V4.1-Flash-1017-EXL3-2.9bpw-UNCENSORED/model-$s-of-00043.safetensors" .
done
# the subset fetch renamed the card; the EXL3 reader wants config.json
ln -s exl3_config.json config.json

cd ~/repos/exo-oq
PYTHONPATH=$PWD/src:$PWD/mlx-lm ~/repos/exo/.venv/bin/python \
  tests/dsv41_oq/crossformat_oracle.py \
  --oq-pack ~/.hermes/cache/scratch/oqmini \
  --exl3-pack ~/.hermes/cache/scratch/exl3_1017_l0 \
  --experts 8 --json /tmp/crossformat.json
```

## Verdict

The oQ loader reads the 1017 pack the same way the EXL3 path reads the 1017 pack,
to within the difference between the two quantizers: dense ≥ 0.9990, plain
bit-exact except the router gate, expert stacks uniformly at the level their bit
widths predict. No permutation, transpose or indexing fault survived.

## Open

1. `ffn.gate.bias` 3.9e-3 delta — explain (abliteration? EXL3 gate handling?).
2. Layer 0 only; layers 1/14 (engram), the 6 layers whose experts are 3-bit, and
   the MTP head are not covered by this oracle.
3. Production-shape kernel bench (oQ vs EXL3 vs Q1d affine-3 reference) still
   unrun — the second half of the interrupted validation task.
4. Full-model build/boot is gated on ~211 GB free per Studio (disk decision).
