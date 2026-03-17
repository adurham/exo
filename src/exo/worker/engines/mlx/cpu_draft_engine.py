"""CPU Draft Engine: runs a small model on CPU for speculative decoding.

Loads weights once at startup, runs forward pass via C/Accelerate in
a background thread. Zero GPU contention — proven to overlap perfectly
with GPU operations on Apple Silicon unified memory.

Usage:
    engine = CPUDraftEngine("mlx-community/Qwen3-0.6B-8bit")
    engine.start_draft(token_id=1234, num_tokens=2)
    # ... GPU does other work ...
    draft_tokens = engine.get_draft()  # blocks until CPU finishes
"""
import ctypes
import os
import threading
import numpy as np
import mlx.core as mx


class CPUDraftEngine:
    """Manages CPU-only draft model inference in a background thread."""

    def __init__(self, model_id: str, max_seq: int = 256):
        from mlx_lm.utils import load
        from exo.download.download_utils import build_model_path
        from exo.shared.types.common import ModelId

        # Find and load C library
        lib_dir = os.path.dirname(os.path.abspath(__file__))
        lib_path = os.path.join(lib_dir, "cpu_draft.dylib")
        if not os.path.exists(lib_path):
            raise FileNotFoundError(
                f"CPU draft library not found at {lib_path}. "
                "Build with: clang -O3 -shared -DACCELERATE_NEW_LAPACK "
                "-o cpu_draft.dylib cpu_draft.c -framework Accelerate -arch arm64"
            )
        self._lib = ctypes.CDLL(lib_path)
        self._setup_ctypes()

        # Load model and extract float32 weights
        model_path = build_model_path(ModelId(model_id))
        model, self.tokenizer = load(str(model_path), lazy=True, strict=False)
        mx.eval(model.parameters())
        self.args = model.args

        self._extract_weights(model)
        del model
        mx.clear_cache()

        # KV cache
        self.max_seq = max_seq
        self._init_kv_cache()

        # Threading
        self._thread = None
        self._draft_result = None
        self._lock = threading.Lock()

    def _setup_ctypes(self):
        class LayerWeights(ctypes.Structure):
            _fields_ = [
                ("in_norm", ctypes.c_void_p), ("post_norm", ctypes.c_void_p),
                ("q", ctypes.c_void_p), ("k", ctypes.c_void_p),
                ("v", ctypes.c_void_p), ("o", ctypes.c_void_p),
                ("gate", ctypes.c_void_p), ("up", ctypes.c_void_p),
                ("down", ctypes.c_void_p),
                ("q_norm", ctypes.c_void_p), ("k_norm", ctypes.c_void_p),
                ("n_heads", ctypes.c_int), ("n_kv", ctypes.c_int),
                ("head_dim", ctypes.c_int), ("hidden", ctypes.c_int),
                ("inter", ctypes.c_int), ("scale", ctypes.c_float),
            ]

        class KVCache(ctypes.Structure):
            _fields_ = [
                ("k", ctypes.c_void_p), ("v", ctypes.c_void_p),
                ("seq_len", ctypes.c_int), ("max_seq", ctypes.c_int),
            ]

        self._LayerWeights = LayerWeights
        self._KVCache = KVCache

        self._lib.cpu_draft_forward.argtypes = [
            ctypes.c_int, ctypes.c_void_p, ctypes.POINTER(LayerWeights),
            ctypes.c_int, ctypes.c_void_p, ctypes.c_void_p,
            ctypes.c_int, ctypes.c_int, ctypes.POINTER(KVCache),
            ctypes.c_int, ctypes.c_float, ctypes.c_void_p,
        ]
        self._lib.cpu_draft_forward.restype = None

    def _to_np32(self, arr):
        return np.array(arr.astype(mx.float32))

    def _dq(self, proj):
        return np.ascontiguousarray(self._to_np32(mx.dequantize(
            proj.weight, proj.scales, getattr(proj, 'biases', None),
            proj.group_size, proj.bits)))

    def _extract_weights(self, model):
        inner = model.model
        args = model.args

        # Embedding
        emb = inner.embed_tokens
        self._embed = np.ascontiguousarray(self._to_np32(
            mx.dequantize(emb.weight, emb.scales, emb.biases, emb.group_size, emb.bits)))

        n_layers = len(inner.layers)
        self._n_layers = n_layers
        self._hidden = args.hidden_size
        self._vocab = args.vocab_size
        self._head_dim = args.head_dim
        self._theta = args.rope_theta

        # Layer weights
        self._layer_arrays = []
        self._c_layers = (self._LayerWeights * n_layers)()

        for i, layer in enumerate(inner.layers):
            attn = layer.self_attn
            mlp = layer.mlp
            arrays = {}

            arrays['in_norm'] = np.ascontiguousarray(self._to_np32(layer.input_layernorm.weight))
            arrays['post_norm'] = np.ascontiguousarray(self._to_np32(layer.post_attention_layernorm.weight))
            for name, proj in [('q', attn.q_proj), ('k', attn.k_proj),
                               ('v', attn.v_proj), ('o', attn.o_proj),
                               ('gate', mlp.gate_proj), ('up', mlp.up_proj),
                               ('down', mlp.down_proj)]:
                arrays[name] = self._dq(proj)
            arrays['q_norm'] = np.ascontiguousarray(self._to_np32(attn.q_norm.weight)) if hasattr(attn, 'q_norm') else None
            arrays['k_norm'] = np.ascontiguousarray(self._to_np32(attn.k_norm.weight)) if hasattr(attn, 'k_norm') else None

            self._layer_arrays.append(arrays)
            lw = self._c_layers[i]
            lw.in_norm = arrays['in_norm'].ctypes.data
            lw.post_norm = arrays['post_norm'].ctypes.data
            for name in ['q', 'k', 'v', 'o', 'gate', 'up', 'down']:
                setattr(lw, name, arrays[name].ctypes.data)
            lw.q_norm = arrays['q_norm'].ctypes.data if arrays['q_norm'] is not None else 0
            lw.k_norm = arrays['k_norm'].ctypes.data if arrays['k_norm'] is not None else 0
            lw.n_heads = attn.n_heads
            lw.n_kv = attn.n_kv_heads
            lw.head_dim = self._head_dim
            lw.hidden = self._hidden
            lw.inter = args.intermediate_size
            lw.scale = attn.scale

        self._final_norm = np.ascontiguousarray(self._to_np32(inner.norm.weight))
        self._lm_head = self._embed if args.tie_word_embeddings else self._dq(model.lm_head)
        self._logits = np.zeros(self._vocab, dtype=np.float32)

    def _init_kv_cache(self):
        self._kv_arrays = []
        self._c_caches = (self._KVCache * self._n_layers)()
        for i in range(self._n_layers):
            nkv = self._c_layers[i].n_kv
            k = np.zeros((nkv, self.max_seq, self._head_dim), dtype=np.float32)
            v = np.zeros((nkv, self.max_seq, self._head_dim), dtype=np.float32)
            self._kv_arrays.append((k, v))
            self._c_caches[i].k = k.ctypes.data
            self._c_caches[i].v = v.ctypes.data
            self._c_caches[i].seq_len = 0
            self._c_caches[i].max_seq = self.max_seq

    def reset_cache(self):
        """Reset KV cache (call when starting new generation)."""
        for i in range(self._n_layers):
            self._c_caches[i].seq_len = 0

    @property
    def cache_seq_len(self):
        return self._c_caches[0].seq_len

    def _forward_one(self, token_id: int) -> int:
        """Run one token through draft model on CPU. Returns predicted next token."""
        offset = self._c_caches[0].seq_len
        self._lib.cpu_draft_forward(
            token_id,
            self._embed.ctypes.data,
            self._c_layers,
            self._n_layers,
            self._final_norm.ctypes.data,
            self._lm_head.ctypes.data,
            self._vocab,
            self._hidden,
            self._c_caches,
            offset,
            ctypes.c_float(self._theta),
            self._logits.ctypes.data,
        )
        return int(self._logits.argmax())

    def draft_sync(self, start_token: int, num_tokens: int) -> list[int]:
        """Draft tokens synchronously (blocking). Returns list of predicted token IDs."""
        tokens = []
        tok = start_token
        for _ in range(num_tokens):
            tok = self._forward_one(tok)
            tokens.append(tok)
        return tokens

    def start_draft(self, start_token: int, num_tokens: int):
        """Start drafting in a background thread. Non-blocking."""
        if self._thread is not None and self._thread.is_alive():
            self._thread.join()  # wait for previous draft
        self._draft_result = None

        def _work():
            self._draft_result = self.draft_sync(start_token, num_tokens)

        self._thread = threading.Thread(target=_work, daemon=True)
        self._thread.start()

    def get_draft(self, timeout: float = 5.0) -> list[int] | None:
        """Get draft results. Blocks until background thread finishes."""
        if self._thread is None:
            return None
        self._thread.join(timeout=timeout)
        if self._thread.is_alive():
            return None  # timed out
        result = self._draft_result
        self._draft_result = None
        self._thread = None
        return result

    def trim_cache(self, n: int):
        """Remove last n tokens from draft KV cache (for rejection rewind)."""
        for i in range(self._n_layers):
            self._c_caches[i].seq_len = max(0, self._c_caches[i].seq_len - n)
