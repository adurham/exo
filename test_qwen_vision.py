import sys
import asyncio
from exo.worker.engines.mlx.utils_mlx import load_model
from exo.shared.types.text_generation import TextGenerationTaskParams
from exo.api.types import ChatCompletionMessage, ChatCompletionMessageText, ChatCompletionMessageImageUrl
from exo.api.adapters.chat_completions import chat_request_to_text_generation, ChatCompletionRequest
from exo.worker.engines.mlx.vision import prepare_vision

async def test():
    req = ChatCompletionRequest(
        model="mlx-community/Qwen3.6-35B-A3B-8bit",
        messages=[
            ChatCompletionMessage(role="user", content=[
                ChatCompletionMessageText(type="text", text="What is in this image?"),
                ChatCompletionMessageImageUrl(type="image_url", image_url={"url": "data:image/jpeg;base64,/9j/4AAQSkZJRgABAQEASABIAAD/2wBDAP//////////////////////////////////////////////////////////////////////////////////////wgALCAABAAEBAREA/8QAFBABAAAAAAAAAAAAAAAAAAAAAP/aAAgBAQABPxA="})
            ])
        ]
    )
    task_params = await chat_request_to_text_generation(req)
    
    # Load model
    model, tokenizer, vision_processor = load_model(
        "mlx-community/Qwen3.6-35B-A3B-8bit",
        {"max_kv_size": 2048, "gpu_memory_utilization": 0.8}
    )
    
    print("Model loaded. Preparing vision...")
    vision_result = prepare_vision(
        images=task_params.images,
        chat_template_messages=task_params.chat_template_messages,
        vision_processor=vision_processor,
        tokenizer=tokenizer,
        model=model,
        model_id="mlx-community/Qwen3.6-35B-A3B-8bit",
        task_params=task_params
    )
    print("Success!")

asyncio.run(test())
