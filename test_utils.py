import sys
from transformers import AutoTokenizer
from exo.worker.engines.mlx.utils_mlx import apply_chat_template
from exo.worker.engines.mlx.generator.batch_generate import TextGenerationTaskParams

tokenizer = AutoTokenizer.from_pretrained("mlx-community/Qwen2.5-Coder-32B-Instruct-4bit")
messages = [{"role": "user", "content": [{"type": "text", "text": "What is in this image?"}, {"type": "image_url", "image_url": {"url": "data:image/jpeg;base64,..."}}]}]

params = TextGenerationTaskParams(chat_template_messages=messages, input=[], model="qwen", request_id="123")

try:
    print(apply_chat_template(tokenizer, params))
except Exception as e:
    print(f"CRASH: {type(e).__name__}: {e}")
    import traceback
    traceback.print_exc()
