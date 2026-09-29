#!/usr/bin/env python3
"""pV5 -- pin the DSv4.1 prompt/placeholder facts the engine wiring needs.

CPU-only (no Metal work beyond import), but run it under the lock anyway.

  1. the exact image placeholder TOKEN string (id 129264) and its tokenization,
  2. whether ``tokenizer.encode`` adds BOS by default (the template says NOT to),
  3. what the checkpoint's own chat_template.jinja renders for a 2-turn chat,
  4. how it differs from exo's vendored V4 encoder (the model id contains
     'deepseek-v4', so exo's generic apply_chat_template routes there),
  5. ``prepare_vl_inputs`` expansion on a real small image (span layout),
  6. the TokenizerWrapper surface serve/output need (think marks, detokenizer).
"""
from __future__ import annotations

import io
import json
import os
import sys

import numpy as np

HOME = os.path.expanduser("~")
sys.path.insert(0, os.path.join(HOME, "repos", "exo", "src"))

M = os.path.join(HOME, ".exo", "models",
                 "dealignai--DeepSeek-V4.1-Flash-UNCENSORED-EXL3-2.9bpw")

from transformers import AutoTokenizer  # noqa: E402

tj = json.load(open(os.path.join(M, "tokenizer.json")))
add = {t["id"]: t["content"] for t in tj.get("added_tokens", [])}
IMAGE_ID = 129264
PLACEHOLDER = add[IMAGE_ID]
print("[1] placeholder id", IMAGE_ID, "=", ascii(PLACEHOLDER), "len", len(PLACEHOLDER))
print("[1] codepoints", [hex(ord(c)) for c in PLACEHOLDER])

tok = AutoTokenizer.from_pretrained(M, trust_remote_code=False)
enc = tok.encode(PLACEHOLDER, add_special_tokens=False)
print("[1] encode(add_special_tokens=False) ->", enc)
print("[1] single-token placeholder:", enc == [IMAGE_ID])

q = "Hello"
print("[2] encode('Hello') default ->", tok.encode(q),
      "| no-special ->", tok.encode(q, add_special_tokens=False))

TURNS = [
    {"role": "user", "content": "Hello, what is 2+2?"},
    {"role": "assistant", "content": "It is 4."},
    {"role": "user", "content": "And times 3?"},
]
try:
    p_tmpl = tok.apply_chat_template(TURNS, tokenize=False, add_generation_prompt=True,
                                    enable_thinking=False)
    print("[3] checkpoint template render (2 turns):")
    print(repr(p_tmpl[:700]))
    print("[3] template token len:", len(tok.encode(p_tmpl, add_special_tokens=False)))
except Exception as e:
    print("[3] template render FAILED:", type(e).__name__, e)

try:
    from exo.worker.engines.mlx.vendor.deepseek_v4_encoding import encode_messages as v4enc
    v4 = v4enc(messages=TURNS, thinking_mode="chat")
    print("[4] vendored V4 encoder render:", repr(v4[:700]))
    print("[4] identical to checkpoint template:", v4 == p_tmpl)
except Exception as e:
    print("[4] vendored encoder FAILED:", type(e).__name__, e)

# --- 5. image expansion ---------------------------------------------------
import mlx.core as mx  # noqa: E402
from PIL import Image  # noqa: E402

from mlx_lm.models.deepseek_v41 import image_processor as mip  # noqa: E402
from mlx_lm.models.deepseek_v41 import vision as mv  # noqa: E402

arr = np.random.default_rng(11).integers(0, 256, (240, 320, 3), dtype=np.uint8)
buf = io.BytesIO()
Image.fromarray(arr).save(buf, format="PNG")
rec = {"data": buf.getvalue()}

tower, cfg = mv.load_vision_tower(M, dtype=mx.bfloat16)
print("[5] vision tower loaded: text_dim", cfg.text_dim, "image_token_id", cfg.image_token_id,
      "patch", cfg.patch_size, "down", cfg.downsample_ratio, "max_tok", cfg.max_image_tokens)
print("[5] cfg.vision_enabled:", cfg.vision_enabled)


class _Tok:
    """Minimal encode() adapter so prepare_vl_inputs can expand a real prompt."""

    def __init__(self, hf):
        self.hf = hf

    def encode(self, text):
        return self.hf.encode(text, add_special_tokens=False)


prompt = "Look at this and describe it.\n" + PLACEHOLDER + "\nWhat do you see?"
ids, types, imgs = mip.prepare_vl_inputs(prompt, [rec], _Tok(tok), cfg)
print("[5] expanded token count:", len(ids), "(was", len(tok.encode(prompt, add_special_tokens=False)), ")")
print("[5] n image inputs:", None if imgs is None else len(imgs))
if imgs:
    img = imgs[0]
    print("[5] span:", img.start, "->", img.start + len(img.types),
          "n_vit", img.n_vit_h, "x", img.n_vit_w,
          "patch shape", img.patches.shape, "types", np.asarray(img.types).tolist()[:12], "...")

blk = tower.build_image_block(img)
import mlx.core as mx  # noqa: E402
mx.eval(blk)
print("[5] image block:", blk.shape, blk.dtype)

h = mx.zeros((1, len(ids), cfg.text_dim), dtype=mx.bfloat16)
merged = tower.merge_image_embeddings(h, imgs, sample=0)
mx.eval(merged)
m = np.array(merged.astype(mx.float32))
span = m[0, img.start:img.start + len(img.types)]
print("[5] merged span nonzero rows:", int((np.abs(span).sum(axis=1) > 0).sum()),
      "of", span.shape[0])
print("[5] outside-span max|v|:", float(np.abs(np.delete(m[0], slice(img.start, img.start + len(img.types)), axis=0)).max()))

# --- 6. TokenizerWrapper surface -----------------------------------------
from mlx_lm.tokenizer_utils import TokenizerWrapper  # noqa: E402
import inspect  # noqa: E402

print("[6] TokenizerWrapper attrs:",
      [a for a in ("eos_token_ids", "think_start", "think_end", "has_thinking",
                   "chat_template", "detokenizer", "apply_chat_template")
       if hasattr(TokenizerWrapper, a)])
w = TokenizerWrapper(tok)
print("[6] eos_token_ids:", w.eos_token_ids)
print("[6] has_thinking:", w.has_thinking, "think_start:", getattr(w, "think_start", None),
      "think_end:", getattr(w, "think_end", None))
print("[6] chat_template is not None:", w.chat_template is not None,
      "len", len(w.chat_template or ""))
print("[6] encode exists:", hasattr(w, "encode"),
      "| stream:", hasattr(w, "stream") if True else None)
try:
    d = w.detokenizer
    d.add_token(100)
    print("[6] detokenizer ok:", repr(d.last_segment))
except Exception as e:
    print("[6] detokenizer FAILED:", type(e).__name__, e)
print("[6] apply_chat_template sig:", inspect.signature(w.apply_chat_template))
try:
    r = w.apply_chat_template(TURNS, tokenize=False, add_generation_prompt=True,
                              enable_thinking=False)
    print("[6] wrapper apply_chat_template == direct:", r == p_tmpl)
except Exception as e:
    print("[6] wrapper apply_chat_template FAILED:", type(e).__name__, e)
print("OK")
