import sys
from transformers import AutoTokenizer

tokenizer = AutoTokenizer.from_pretrained("mlx-community/Qwen2.5-Coder-32B-Instruct-4bit")
messages = [{"role": "user", "content": [{"type": "text", "text": "What is in this image?"}, {"type": "image_url", "image_url": {"url": "data:image/jpeg;base64,..."}}]}]

try:
    print(tokenizer.apply_chat_template(messages, tokenize=False))
except Exception as e:
    print(f"CRASH: {type(e).__name__}: {e}")
    import traceback
    traceback.print_exc()
