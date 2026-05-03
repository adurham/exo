#!/usr/bin/env bash
# Convert upstream deepseek-ai/DeepSeek-V4-Flash to MLX 8-bit-tail
# format INCLUDING the mtp.* weights for self-speculative decode.
#
# Output is a drop-in replacement for mlx-community/DeepSeek-V4-Flash-8bit
# but ships the additional ~3-4 GB of mtp.0.* weights that the
# mlx-community conversion stripped. Quality policy matches: experts
# stay FP4 (already pre-quantized upstream), non-expert tail is
# quantized to MLX 8-bit (group_size=64).
#
# Disk requirements (whichever node runs this):
#   - upstream FP4+FP8 download:  ~160 GB
#   - MLX 8-bit-mtp output:       ~155 GB
#   - HF cache headroom:          ~5 GB
#   TOTAL:                        ~320 GB free space needed
#
# Runtime: ~30-60 min on M4 Max (CPU-only, the convert step is mostly
# I/O + dequant + requantize). Runs fine on the laptop.
#
# After running, rsync the output to both cluster nodes' ~/.exo/models/
# and update start_cluster.sh to use it.
set -euo pipefail

# Output location — put it under ~/.exo/models so exo discovers it
# the same way as the existing variants. Name must NOT collide with
# anything mlx-community publishes.
OUT_DIR="${OUT_DIR:-$HOME/.exo/models/local--DeepSeek-V4-Flash-8bit-mtp}"
HF_REPO="${HF_REPO:-deepseek-ai/DeepSeek-V4-Flash}"

if [[ -d "$OUT_DIR" ]]; then
    echo "Output dir already exists: $OUT_DIR"
    echo "Delete it or pass OUT_DIR=... to convert into a different path."
    exit 1
fi

# Free-space check.
AVAIL_KB=$(df -k "$(dirname "$OUT_DIR")" | awk 'NR==2 {print $4}')
NEEDED_KB=$((320 * 1024 * 1024))
if [[ "$AVAIL_KB" -lt "$NEEDED_KB" ]]; then
    echo "Insufficient disk space at $(dirname "$OUT_DIR"):"
    echo "  available: $((AVAIL_KB / 1024 / 1024)) GB"
    echo "  needed:    320 GB"
    echo "Free up space (e.g., delete unused models in ~/.exo/models/)"
    echo "or run on the laptop where disk is roomier."
    exit 1
fi

# Activate the EXO_DSV4_MTP gate so sanitize() KEEPS the mtp.* weights
# during conversion. Without this the output checkpoint will be
# identical to mlx-community's 8bit (no mtp.*).
export EXO_DSV4_MTP=1

cd "$(dirname "$0")/.."

uv run python -c "
import os
os.environ['EXO_DSV4_MTP'] = '1'
from mlx_lm import convert

convert(
    hf_path='$HF_REPO',
    mlx_path='$OUT_DIR',
    quantize=True,
    q_group_size=64,
    q_bits=8,
)
print('conversion complete:', '$OUT_DIR')
"

echo
echo "Verifying mtp.* keys present in output..."
uv run python -c "
import json
with open('$OUT_DIR/model.safetensors.index.json') as f:
    d = json.load(f)
mtp_keys = [k for k in d['weight_map'] if k.startswith('model.mtp.')]
print(f'  model.mtp.* keys: {len(mtp_keys)}')
if not mtp_keys:
    raise SystemExit('FAIL: no mtp.* keys in output. EXO_DSV4_MTP gate may not have fired.')
print('  PASS')
"

echo
echo "Next steps:"
echo "  1. Rsync to cluster nodes:"
echo "       rsync -avh --progress $OUT_DIR/ macstudio-m4-1:$OUT_DIR/"
echo "       rsync -avh --progress $OUT_DIR/ macstudio-m4-2:$OUT_DIR/"
echo "  2. In start_cluster.sh set DSV4_MODEL_ID=local/DeepSeek-V4-Flash-8bit-mtp"
echo "     (or whatever local model card name you choose; create a"
echo "     custom_model_cards/ entry pointing at the new dir)."
echo "  3. Boot the cluster with EXO_DSV4_MTP=1 EXO_SPECULATIVE=1"
echo "       EXO_DSV4_MTP=1 EXO_SPECULATIVE=1 ./start_cluster.sh"
echo "  4. Smoke test c=1 chat completion."
echo "  5. Smoke test c=2 chat completion (target: per-stream ≥30 tok/s)."
