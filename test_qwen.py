import sys
from typing import Any
import mlx.core as mx

messages = [{"role": "user", "content": [{"type": "text", "text": "What is in this image?"}, {"type": "image_url", "image_url": {"url": "data:image/jpeg;base64,..."}}]}]

from exo.worker.engines.mlx.vision import _format_vlm_messages
try:
    print(_format_vlm_messages(messages, "qwen_vl"))
except Exception as e:
    print(f"CRASH: {type(e).__name__}: {e}")
    import traceback
    traceback.print_exc()
