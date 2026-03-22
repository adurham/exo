#!/usr/bin/env python3
"""Download and quantize MTP weights from Qwen3.5-397B-A17B.

Downloads only the safetensor files containing MTP weights from the original
bf16 model, extracts and quantizes them to 4-bit, and saves as a standalone
safetensors file that can be loaded alongside the quantized main model.

Usage:
    uv run python scripts/quantize_mtp_weights.py [--output-dir DIR]

The output file (mtp_weights_4bit.safetensors) should be placed in the
model's cache directory alongside the main model weights.
"""

import argparse
import json
import os
import sys
from pathlib import Path

import mlx.core as mx
import mlx.nn as nn
from huggingface_hub import hf_hub_download


MODEL_ID = "Qwen/Qwen3.5-397B-A17B"
GROUP_SIZE = 64
BITS = 4


def main():
    parser = argparse.ArgumentParser(description="Quantize MTP weights")
    parser.add_argument(
        "--output-dir",
        type=str,
        default=".",
        help="Output directory for quantized weights",
    )
    parser.add_argument(
        "--bits",
        type=int,
        default=BITS,
        help="Quantization bits (default: 4)",
    )
    args = parser.parse_args()

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    output_path = output_dir / f"mtp_weights_{args.bits}bit.safetensors"

    if output_path.exists():
        print(f"Output already exists: {output_path}")
        sys.exit(0)

    # 1. Find which safetensor files contain MTP weights
    print("Downloading model index...")
    idx_path = hf_hub_download(MODEL_ID, "model.safetensors.index.json")
    with open(idx_path) as f:
        idx = json.load(f)

    mtp_key_to_file: dict[str, str] = {}
    for key, filename in idx["weight_map"].items():
        if "mtp." in key:
            mtp_key_to_file[key] = filename

    mtp_files = sorted(set(mtp_key_to_file.values()))
    print(f"Found {len(mtp_key_to_file)} MTP weight tensors across {len(mtp_files)} files")

    # 2. Download and extract MTP weights using MLX (handles bf16 natively)
    all_mtp_weights: dict[str, mx.array] = {}
    for filename in mtp_files:
        print(f"Downloading {filename}...")
        filepath = hf_hub_download(MODEL_ID, filename)
        print(f"  Loading and extracting MTP weights...")
        file_weights = mx.load(filepath)
        keys_in_file = [k for k, f in mtp_key_to_file.items() if f == filename]
        for key in keys_in_file:
            if key in file_weights:
                all_mtp_weights[key] = file_weights[key]
        del file_weights  # free non-MTP weights
        mx.clear_cache()
        print(f"  Extracted {len(keys_in_file)} tensors")

    print(f"\nTotal MTP weights: {len(all_mtp_weights)}")
    total_bytes = sum(v.nbytes for v in all_mtp_weights.values())
    print(f"Total size (bf16): {total_bytes / 1e9:.1f} GB")

    # 3. Quantize weights that are large enough (skip norms, biases)
    print(f"\nQuantizing to {args.bits}-bit (group_size={GROUP_SIZE})...")
    quantized: dict[str, mx.array] = {}
    for key, weight in sorted(all_mtp_weights.items()):
        if weight.ndim < 2 or weight.shape[0] < GROUP_SIZE:
            # Small weights (norms, biases) — keep as-is
            quantized[key] = weight
        else:
            # Quantize
            q_weight, q_scales, q_biases = mx.quantize(weight, group_size=GROUP_SIZE, bits=args.bits)
            mx.eval(q_weight, q_scales, q_biases)
            quantized[key] = q_weight
            quantized[key.replace(".weight", ".scales")] = q_scales
            quantized[key.replace(".weight", ".biases")] = q_biases

    # 4. Save using MLX's native save (handles all dtypes)
    print(f"Saving to {output_path}...")
    mx.save_safetensors(str(output_path), quantized)
    out_size = output_path.stat().st_size
    print(f"Done! Output: {out_size / 1e9:.1f} GB ({out_size / total_bytes * 100:.0f}% of original)")


if __name__ == "__main__":
    main()
