# P05: knob smoke test — construct the real DSv4 Model + load ONLY the
# lm_head weights (head.weight), then run the load_model's quantization
# branch manually, verifying:
#   - unset env: lm_head stays nn.Linear (bit-identical path)
#   - env=1: lm_head becomes QuantizedLinear with mxfp8 packing
#   - a forward through the quantized head produces sane logits
# Runs on the studio beside production (standalone, no relaunch).
import os
import sys

sys.path.insert(0, os.path.expanduser("~/repos/exo/mlx-lm"))

import json
import struct
from pathlib import Path

import mlx.core as mx
import mlx.nn as nn

CKPT = Path.home() / ".exo/models/deepseek-ai--DeepSeek-V4-Flash-0731"


def load_bf16(shard, key):
    with open(CKPT / shard, "rb") as fh:
        n = struct.unpack("<Q", fh.read(8))[0]
        hdr = json.loads(fh.read(n))
        info = hdr[key]
        o0, o1 = info["data_offsets"]
        fh.seek(n + 8 + o0)
        raw = fh.read(o1 - o0)
    u16 = __import__("numpy").frombuffer(raw, dtype=__import__("numpy").uint16)
    f32 = (u16.astype(__import__("numpy").uint32) << 16).view(__import__("numpy").float32)
    return f32.reshape(info["shape"])


from mlx_lm.models.deepseek_v4 import Model, ModelArgs

cfg = json.loads((CKPT / "config.json").read_text())
args = ModelArgs.from_dict(cfg)
print(f"model_type={args.model_type} vocab={args.vocab_size} hidden={args.hidden_size}")

# --- env OFF: knob must be a no-op ---
for val, expect_q in ((None, False), ("1", True), ("bogus", False), ("0", False)):
    if val is None:
        os.environ.pop("EXO_DSV4_LMHEAD_MXFP8", None)
    else:
        os.environ["EXO_DSV4_LMHEAD_MXFP8"] = val
    model = Model(args)
    w = load_bf16("model-00045-of-00048.safetensors", "head.weight")
    model.eval()
    model.load_weights([("lm_head.weight", mx.array(w).astype(mx.bfloat16))], strict=False)
    # replicate the load_model branch manually (it keys on config model_type
    # which we emulate here — the real load path is exercised on relaunch)
    if os.environ.get("EXO_DSV4_LMHEAD_MXFP8", "0") == "1" and args.model_type == "deepseek_v4":
        mod = getattr(model, "lm_head", None)
        if (isinstance(mod, nn.Linear) and mod is not None
                and mod.weight.shape[-1] % 32 == 0 and not hasattr(mod, "scales")):
            model.update_modules(
                {"lm_head": mod.to_quantized(group_size=32, bits=8, mode="mxfp8")}
            )
    is_q = hasattr(model.lm_head, "scales")
    status = "OK" if is_q == expect_q else "MISMATCH"
    print(f"env={val!r}: lm_head is {'QuantizedLinear' if is_q else 'Linear'} -> {status}")
    if is_q:
        x = mx.random.normal((1, 4, args.hidden_size))
        x = x / mx.sqrt(mx.mean(x * x, axis=-1, keepdims=True))
        logits = model(mx.array(x).astype(mx.bfloat16), cache=None)
        mx.eval(logits)
        print(f"  quantized forward: shape={logits.shape} "
              f"std={float(mx.std(logits.astype(mx.float32))):.2f} "
              f"argmax={int(mx.argmax(logits[0, -1]).item())}")