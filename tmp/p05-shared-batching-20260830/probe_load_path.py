# Load ONE real shared_experts tensor pair through the PRODUCTION path
# (mlx_lm's _load_safetensors with the F8_E8M0 fallback) to see exactly
# what dtype/shape comes out, then build the mxfp8 QuantizedLinear the
# same way load_model's _quantize does.
import os
import sys

sys.path.insert(0, os.path.expanduser("~/repos/exo/mlx-lm"))

from pathlib import Path

import mlx.core as mx
import mlx.nn as nn
from mlx_lm.utils import _load_safetensors

CKPT = Path.home() / ".exo/models/deepseek-ai--DeepSeek-V4-Flash-0731"
w = _load_safetensors(str(CKPT / "model-00005-of-00048.safetensors"))
for k, v in w.items():
    if "layers.3.ffn.shared_experts" in k:
        print(k, v.dtype, v.shape)
    if "layers.3.hc_attn_fn" in k or k == "layers.3.ffn.gate.weight":
        print(k, v.dtype, v.shape)
        break