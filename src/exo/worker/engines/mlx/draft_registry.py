"""Process-global registry for draft token serving.

The runner registers its already-loaded MLX model here after load.
The API wraps it in a DraftHandler to serve /v1/draft requests — no
second model copy is loaded.
"""
import threading
from typing import Any

import mlx.core as mx
from mlx_lm.models.cache import make_prompt_cache, trim_prompt_cache

_lock = threading.Lock()
_registry: dict[str, Any] = {}  # model_id -> MLX model


class DraftHandler:
    """KV-cache-stateful wrapper around an already-loaded MLX model."""

    def __init__(self, model: Any, model_id: str):
        self.model = model
        self.model_id = model_id
        self._cache = make_prompt_cache(model)
        self.cache_len = 0
        self._op_lock = threading.Lock()

    def reset(self) -> None:
        with self._op_lock:
            self._cache = make_prompt_cache(self.model)
            self.cache_len = 0

    def prefill(self, token_ids: list[int]) -> int:
        if not token_ids:
            return self.cache_len
        with self._op_lock:
            tokens = mx.array([token_ids])
            logits = self.model(tokens, cache=self._cache)
            mx.eval(logits)
            self.cache_len += len(token_ids)
            return self.cache_len

    def draft(self, token_id: int, num_tokens: int, trim: int = 0) -> list[int]:
        with self._op_lock:
            if trim > 0:
                trim_prompt_cache(self._cache, trim)
                self.cache_len = max(0, self.cache_len - trim)
            results: list[int] = []
            tok = token_id
            for _ in range(num_tokens):
                logits = self.model(mx.array([[tok]]), cache=self._cache)
                mx.eval(logits)
                self.cache_len += 1
                tok = int(logits[0, -1].argmax().item())
                results.append(tok)
            return results


# Cached handlers (one per model_id, created lazily on first /v1/draft request)
_handlers: dict[str, DraftHandler] = {}


def register(model_id: str, model: Any) -> None:
    """Called by the runner after model load."""
    with _lock:
        _registry[model_id] = model
        # Invalidate any cached handler so it's rebuilt with the new model
        _handlers.pop(model_id, None)


def get_handler(model_id: str) -> DraftHandler | None:
    """Return a DraftHandler for the given model, or None if not registered."""
    with _lock:
        if model_id in _handlers:
            return _handlers[model_id]
        model = _registry.get(model_id)
        if model is None:
            return None
        handler = DraftHandler(model, model_id)
        _handlers[model_id] = handler
        return handler


def unregister(model_id: str) -> None:
    """Called by the runner on shutdown."""
    with _lock:
        _registry.pop(model_id, None)
        _handlers.pop(model_id, None)
