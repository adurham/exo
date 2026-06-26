import sys
from typing import Any
import mlx.core as mx

messages = [{"role": "user", "content": [{"type": "text", "text": "What is in this image?"}, {"type": "image_url", "image_url": {"url": "data:image/jpeg;base64,..."}}]}]

from exo.worker.engines.mlx.vendor.deepseek_v4_encoding import encode_messages

try:
    prompt = encode_messages(messages, thinking_mode="chat")
    print("Success")
except Exception as e:
    print(f"CRASH: {type(e).__name__}: {e}")
    import traceback
    traceback.print_exc()
