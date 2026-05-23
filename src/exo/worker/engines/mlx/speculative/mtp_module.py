#!/usr/bin/env python3
"""MTP (Multi-Token Prediction) module for Qwen3.5-27B.

Architecture (from llama.cpp build_mtp_head + HuggingFace config):
  1. Normalize: pre_fc_norm_hidden(hidden_state) || pre_fc_norm_embedding(embed(token))
  2. Combine: fc(concat([e_norm, h_norm])) → 5120
  3. 1 GQA decoder layer (same config as main model's full-attention layers)
     - Attention with Q/K RMSNorm + partial RoPE + output gate
  4. Final norm → shared lm_head → vocab logits

Predicts token t+2 given the main model's hidden state at position t
and the token sampled at position t+1.

Usage:
    from .mtp_module import MTPPredictor
    mtp = MTPPredictor(model, "mtp_weights.safetensors")
    # During decode:
    pre_norm, normed = mtp.get_hidden_state(input_tokens, cache)
    logits_t1 = mtp.apply_lm_head(normed)            # token t+1
    logits_t2 = mtp.predict(pre_norm, token_t1)       # token t+2
"""

import os
from typing import Any, Optional

import mlx.core as mx
import mlx.nn as nn

# Token-tree drafting alpha-distribution probe. Gated by
# EXO_DSV4_TREE_ALPHA_PROBE=1. When ON, draft_tokens() appends a record per
# draft step into _TREE_ALPHA_PROBE_STEPS containing the MTP head's top-5
# argmax IDs (as lazy mx.arrays). _speculative_next then drains the list,
# joins the target argmax, and writes JSONL. Zero-cost when env is unset
# (the gate is a single env lookup at module-import time below).
TREE_ALPHA_PROBE = os.environ.get("EXO_DSV4_TREE_ALPHA_PROBE") == "1"
_TREE_ALPHA_PROBE_STEPS: list[dict] = []


def speculative_forward(model, inputs, cache, speculative=False):
    """Run model forward pass, optionally capturing GDN per-step states for rollback.

    This is the shared core for both MTP and draft-model speculative decoding.
    It manually iterates model layers to:
    1. Wrap GDN caches in SpeculativeArraysCache when speculative=True
    2. Patch gated_delta_update to use the speculative kernel
    3. Capture per-step recurrent states and reconstruct conv_input

    Args:
        model: the loaded model (e.g. from mlx_lm.load)
        inputs: (B, S) int token ids
        cache: cache list from make_prompt_cache()
        speculative: if True, saves per-step GDN states for rollback

    Returns:
        (pre_norm, logits) — pre-RMSNorm hidden states and vocab logits
    """
    inner = getattr(model, 'model', None) or model.language_model.model
    text_model = getattr(model, 'model', None) or model.language_model
    S = inputs.shape[1]
    do_spec = speculative and S > 1

    if hasattr(inner, 'embed_tokens'):
        hidden_states = inner.embed_tokens(inputs)
    else:
        hidden_states = inputs

    cache_list = cache if cache is not None else [None] * len(inner.layers)

    gdn_spec_data = []
    if do_spec:
        from .speculative_cache import SpeculativeArraysCache
        for i, c in enumerate(cache_list):
            if c is not None and hasattr(c, 'cache') and not hasattr(c, 'offset'):
                cache_list[i] = SpeculativeArraysCache(c, S=S)
        if cache is not None:
            for i in range(len(cache)):
                cache[i] = cache_list[i]

    spec_all_states = []
    if do_spec:
        import mlx_lm.models.qwen3_5 as _qwen3_5_mod
        _orig_gdu = _qwen3_5_mod.gated_delta_update
        _qwen3_5_mod.gated_delta_update = _make_speculative_gdu(spec_all_states)

    from mlx_lm.models.qwen3_5 import create_attention_mask, create_ssm_mask
    fa_mask = create_attention_mask(hidden_states, cache_list[inner.fa_idx])
    ssm_mask = create_ssm_mask(hidden_states, cache_list[inner.ssm_idx])

    for layer, c in zip(inner.layers, cache_list):
        mask = ssm_mask if layer.is_linear else fa_mask

        if do_spec and layer.is_linear:
            from .speculative_cache import SpeculativeArraysCache as _SAC
            if isinstance(c, _SAC):
                pre_conv = c[0]
                if pre_conv is None:
                    gdn = layer.linear_attn
                    pre_conv = mx.zeros(
                        (hidden_states.shape[0], gdn.conv_kernel_size - 1,
                         gdn.conv_dim), dtype=hidden_states.dtype)
                gdn_spec_data.append((hidden_states, pre_conv, c, layer))

        hidden_states = layer(hidden_states, mask=mask, cache=c)

    if do_spec:
        _qwen3_5_mod.gated_delta_update = _orig_gdu

        gdn_idx = 0
        for layer_input, pre_conv, spec_cache, parent_layer in gdn_spec_data:
            if gdn_idx < len(spec_all_states):
                spec_cache.all_states = spec_all_states[gdn_idx]
            gdn_idx += 1

            gdn = parent_layer.linear_attn
            normed = parent_layer.input_layernorm(layer_input)
            if hasattr(gdn, 'in_proj_qkv'):
                qkv = gdn.in_proj_qkv(normed)
            else:
                q, k, v, z, b, a = gdn.fix_query_key_value_ordering(
                    gdn.in_proj_qkvz(normed), gdn.in_proj_ba(normed))
                B_dim = normed.shape[0]
                qkv = mx.concatenate(
                    [q.reshape(B_dim, S, -1), k.reshape(B_dim, S, -1),
                     v.reshape(B_dim, S, -1)], axis=-1)
            spec_cache.conv_input = mx.concatenate([pre_conv, qkv], axis=1)

    pre_norm = hidden_states
    normed = inner.norm(hidden_states)

    if hasattr(text_model, 'lm_head'):
        logits = text_model.lm_head(normed)
    else:
        logits = inner.embed_tokens.as_linear(normed)

    return pre_norm, logits


def _make_speculative_gdu(all_states_list):
    """Create a gated_delta_update replacement that uses the speculative kernel.

    The speculative kernel is identical to the original but also outputs
    per-step recurrent states (all_states). These are appended to
    all_states_list for later assignment to SpeculativeArraysCache wrappers.

    Returns (y, state_out) — same interface as original gated_delta_update.
    """
    from .speculative_gdn_kernel import speculative_gated_delta_kernel
    from mlx_lm.models.gated_delta import compute_g

    def speculative_gated_delta_update(q, k, v, a, b, A_log, dt_bias,
                                        state=None, mask=None, use_kernel=True):
        beta = mx.sigmoid(b)
        g = compute_g(A_log, a, dt_bias)
        if state is None:
            B, _, Hk, Dk = q.shape
            Hv, Dv = v.shape[-2:]
            state = mx.zeros((B, Hv, Dv, Dk), dtype=q.dtype)
        y, state_out, all_states = speculative_gated_delta_kernel(
            q, k, v, g, beta, state, mask)
        all_states_list.append(all_states)
        return y, state_out

    return speculative_gated_delta_update


class MTPPredictor:
    """MTP draft predictor for speculative decoding.

    Wraps the MTP module weights and provides:
      - get_hidden_state(): extract pre-lm_head hidden state from main model
      - predict(): run MTP to get next-next-token logits
    """

    def __init__(self, model, mtp_weights_path, quantize=True, skip_mlp=False):
        """Load MTP weights and attach to the main model.

        Args:
            model: loaded Qwen3.5-27B model
            mtp_weights_path: path to mtp_weights.safetensors
            quantize: quantize MTP linears to 8-bit gs=64
            skip_mlp: skip MoE/MLP weights (saves ~13GB for PP mode)
        """
        self.model = model
        self._inner = getattr(model, 'model', None) or model.language_model.model
        self._text_model = getattr(model, 'model', None) or model.language_model

        # Shared components
        self.embed_tokens = self._inner.embed_tokens
        if hasattr(self._text_model, 'lm_head'):
            self.lm_head = self._text_model.lm_head
        else:
            # tie_word_embeddings case
            self.lm_head = None

        # Load MTP weights
        weights = mx.load(mtp_weights_path)

        # ---- Sanitize norm weights ----
        # CRITICAL: Qwen3.5 HuggingFace format stores ALL norm weights as (actual - 1.0).
        # mlx-lm's TextModel.sanitize() adds +1.0 back for the main model norms, but
        # MTP weights are stripped before sanitize runs. We must apply the same shift
        # to ALL 1-D norm weights in the MTP.
        #
        # Evidence: pre_fc_norm_hidden has mean=-0.17 raw → 0.83 after shift (plausible).
        # Linear projection weights (2-D) are NOT shifted.
        shifted = []
        for k in list(weights.keys()):
            if weights[k].ndim == 1:
                weights[k] = weights[k] + 1.0
                shifted.append(k)
        if shifted:
            print(f"  Sanitized {len(shifted)} norm weights (+1.0 shift)")

        # Detect pre-quantized weights (have .scales/.biases companions)
        _is_prequantized = any(k.endswith('.scales') for k in weights)

        # Infer all dimensions from weight shapes (works for any Qwen3.5 size)
        # For pre-quantized 4-bit: shape[0] = output_dims (unchanged),
        # shape[1] = input_dims * bits / 32 (packed). Unpack with * 32 / bits.
        def _dim(w, axis):
            """Get original dimension, unpacking if pre-quantized.
            Only axis 1 is packed (input_dims * bits / 32). Axis 0 is output_dims (unchanged).
            """
            d = w.shape[axis]
            if _is_prequantized and w.dtype == mx.uint32 and axis == 1:
                d = d * 32 // 4  # 4-bit packing: unpack input_dims
            return d

        fc_w = weights['mtp.fc.weight']
        hidden_size = _dim(fc_w, 0)                    # 4096 (9B) or 5120 (27B)
        fc_in = _dim(fc_w, 1)                          # 2 * hidden_size

        q_w = weights['mtp.layers.0.self_attn.q_proj.weight']
        q_out = _dim(q_w, 0)                           # num_heads * head_dim * 2 (gate)
        k_w = weights['mtp.layers.0.self_attn.k_proj.weight']
        kv_out = _dim(k_w, 0)                          # num_kv_heads * head_dim
        o_w = weights['mtp.layers.0.self_attn.o_proj.weight']
        o_in = _dim(o_w, 1)                            # num_heads * head_dim

        # Detect MoE vs dense MLP
        self.is_moe = 'mtp.layers.0.mlp.gate.weight' in weights

        if not self.is_moe:
            gate_w = weights['mtp.layers.0.mlp.gate_proj.weight']
            intermediate = gate_w.shape[0]
        else:
            intermediate = 0  # MoE experts handle this

        # head_dim from q_norm weight (always per-head)
        head_dim = weights.get('mtp.layers.0.self_attn.q_norm.weight',
                               mx.ones(256)).shape[0]
        num_heads = o_in // head_dim
        num_kv_heads = kv_out // head_dim

        print(f"  Dims: hidden={hidden_size}, heads={num_heads}, kv_heads={num_kv_heads}, "
              f"head_dim={head_dim}, MLP={'MoE' if self.is_moe else f'dense({intermediate})'}")

        # Build layers from weights — all dimension-agnostic
        def make_linear(w, key_prefix: str = ''):
            if _is_prequantized and f'{key_prefix}.scales' in weights:
                # Pre-quantized: build QuantizedLinear and update weights via module.update()
                scales = weights[f'{key_prefix}.scales']
                biases = weights[f'{key_prefix}.biases']
                in_dims = w.shape[1] * 32 // 4  # unpack packed input_dims
                out_dims = w.shape[0]
                ql = nn.QuantizedLinear(in_dims, out_dims, bias=False, group_size=64, bits=4)
                ql.update({'weight': w, 'scales': scales, 'biases': biases})
                return ql
            out_dim, in_dim = w.shape
            l = nn.Linear(in_dim, out_dim, bias=False)
            l.weight = w
            return l

        self.pre_fc_norm_hidden = nn.RMSNorm(hidden_size)
        self.pre_fc_norm_hidden.weight = weights['mtp.pre_fc_norm_hidden.weight']

        self.pre_fc_norm_embedding = nn.RMSNorm(hidden_size)
        self.pre_fc_norm_embedding.weight = weights['mtp.pre_fc_norm_embedding.weight']

        self.fc = make_linear(fc_w, 'mtp.fc')
        self.q_proj = make_linear(q_w, 'mtp.layers.0.self_attn.q_proj')
        self.k_proj = make_linear(k_w, 'mtp.layers.0.self_attn.k_proj')
        self.v_proj = make_linear(weights['mtp.layers.0.self_attn.v_proj.weight'], 'mtp.layers.0.self_attn.v_proj')
        self.o_proj = make_linear(o_w, 'mtp.layers.0.self_attn.o_proj')

        self.q_norm = nn.RMSNorm(head_dim)
        self.k_norm = nn.RMSNorm(head_dim)
        q_norm_key = 'mtp.layers.0.self_attn.q_norm.weight'
        k_norm_key = 'mtp.layers.0.self_attn.k_norm.weight'
        if q_norm_key in weights:
            self.q_norm.weight = weights[q_norm_key]
            self.k_norm.weight = weights[k_norm_key]

        self.input_layernorm = nn.RMSNorm(hidden_size)
        self.input_layernorm.weight = weights['mtp.layers.0.input_layernorm.weight']

        self.post_attention_layernorm = nn.RMSNorm(hidden_size)
        self.post_attention_layernorm.weight = weights['mtp.layers.0.post_attention_layernorm.weight']

        self.skip_mlp = skip_mlp

        if self.is_moe and not skip_mlp:
            # Reuse mlx-lm's SparseMoeBlock from the target model
            moe_layer = None
            for layer in self._inner.layers:
                if hasattr(layer, 'mlp') and hasattr(layer.mlp, 'gate'):
                    moe_layer = layer.mlp
                    break
            if moe_layer is None:
                raise RuntimeError("MTP has MoE weights but target model has no MoE layer")

            # Create a fresh MoE block with same class/config as target
            moe_class = type(moe_layer)
            args = (getattr(self._text_model, 'args', None)
                    or getattr(getattr(self._text_model, 'model', None), 'args', None)
                    or getattr(self.model, 'args', None))
            self.mlp = moe_class(args)
            # Quantize the fresh block to match the pre-quantized weight format
            # Skip layers that were too small to quantize (min dim < 64)
            if _is_prequantized:
                def _q_predicate(path, m):
                    if not hasattr(m, 'to_quantized'):
                        return False
                    w = getattr(m, 'weight', None)
                    if w is not None and min(w.shape) < 64:
                        return False
                    return True
                nn.quantize(self.mlp, group_size=64, bits=4, class_predicate=_q_predicate)

            # Load MTP MoE weights — remap HF expert names to mlx-lm SwitchLinear
            prefix = 'mtp.layers.0.mlp.'
            direct_keys = {}
            expert_weights = {}  # {proj_name: {expert_idx: weight}}
            for k, v in weights.items():
                if not k.startswith(prefix):
                    continue
                name = k[len(prefix):]
                if name.startswith('experts.'):
                    # experts.N.{gate,up,down}_proj.{weight,scales,biases}
                    parts = name.split('.')
                    idx = int(parts[1])
                    proj = parts[2]  # gate_proj, up_proj, down_proj
                    suffix = '.'.join(parts[3:])  # weight, scales, or biases
                    key = f'{proj}.{suffix}'
                    if key not in expert_weights:
                        expert_weights[key] = {}
                    expert_weights[key][idx] = v
                else:
                    direct_keys[name] = v

            # Stack individual expert weights into SwitchLinear format
            moe_weights = []
            for proj_key, idx_map in expert_weights.items():
                n_experts = max(idx_map.keys()) + 1
                stacked = mx.stack([idx_map[i] for i in range(n_experts)])
                moe_weights.append((f'switch_mlp.{proj_key}', stacked))

            for name, v in direct_keys.items():
                moe_weights.append((name, v))

            print(f"  MoE loading {len(moe_weights)} weight groups, keys: {[k for k,_ in moe_weights][:10]}...")
            _missing = self.mlp.load_weights(moe_weights, strict=False)
            if _missing:
                print(f"  MoE note: {len(_missing)} module-level keys unmatched: {list(_missing)[:5]}")
            # Verify quantized weights were loaded
            if hasattr(self.mlp, 'switch_mlp') and hasattr(self.mlp.switch_mlp, 'gate_proj'):
                gp = self.mlp.switch_mlp.gate_proj
                print(f"  MoE verify: switch_mlp.gate_proj.weight={gp.weight.dtype} {gp.weight.shape}")
            if hasattr(self.mlp, 'gate'):
                print(f"  MoE verify: gate.weight={self.mlp.gate.weight.dtype} {self.mlp.gate.weight.shape}")
            print(f"  MoE MLP: {len(moe_weights)} weight groups loaded "
                  f"({len(expert_weights)} stacked expert projections)")
        elif skip_mlp:
            print(f"  MLP skipped (skip_mlp=True)")
        else:
            self.gate_proj = make_linear(gate_w, 'mtp.layers.0.mlp.gate_proj')
            self.up_proj = make_linear(weights['mtp.layers.0.mlp.up_proj.weight'])
            self.down_proj = make_linear(weights['mtp.layers.0.mlp.down_proj.weight'])

        self.norm = nn.RMSNorm(hidden_size)
        self.norm.weight = weights['mtp.norm.weight']

        # RoPE from main model's GQA layers
        for layer in self._inner.layers:
            if not layer.is_linear:
                self.rope = layer.self_attn.rope
                break

        # GQA config
        self.num_heads = num_heads
        self.num_kv_heads = num_kv_heads
        self.head_dim = head_dim
        self.scale = head_dim ** -0.5

        # MTP KV cache (separate from main model)
        self.kv_cache = None

        mx.eval(self.pre_fc_norm_hidden.weight, self.pre_fc_norm_embedding.weight,
                self.fc.weight, self.input_layernorm.weight,
                self.post_attention_layernorm.weight, self.norm.weight,
                self.q_norm.weight, self.k_norm.weight)

        if quantize:
            self._quantize_linears()

        total_params = sum(w.size for w in weights.values())
        q_label = ' (pre-quantized 4-bit)' if _is_prequantized else (' (quantized 8-bit gs=64)' if quantize else ' (bf16)')
        print(f"  MTP loaded: {len(weights)} tensors, {total_params / 1e6:.1f}M params{q_label}")

    def _quantize_linears(self):
        """Quantize all MTP linear layers to 8-bit gs=64."""
        for name in ['fc', 'q_proj', 'k_proj', 'v_proj', 'o_proj',
                      'gate_proj', 'up_proj', 'down_proj']:
            linear = getattr(self, name, None)
            if linear is None:
                continue  # MoE models don't have dense MLP projections
            _cdtype = mx.bfloat16 if os.environ.get("EXO_COMPUTE_DTYPE", "fp16") == "bf16" else mx.float16
            linear.weight = linear.weight.astype(_cdtype)
            q = nn.QuantizedLinear.from_linear(linear, group_size=64, bits=8)
            mx.eval(q.parameters())
            setattr(self, name, q)
        # For MoE, quantize expert weights in-place to reduce memory
        # (nn.quantize on the whole block OOMs — quantize one expert at a time)
        if self.is_moe and hasattr(self, 'mlp'):
            if hasattr(self.mlp, 'switch_mlp'):
                nn.quantize(self.mlp.switch_mlp, group_size=64, bits=8)
                mx.eval(self.mlp.switch_mlp.parameters())
            if hasattr(self.mlp, 'shared_expert'):
                nn.quantize(self.mlp.shared_expert, group_size=64, bits=8)
                mx.eval(self.mlp.shared_expert.parameters())

    def reset_cache(self):
        """Reset the MTP KV cache (call at start of generation)."""
        from mlx_lm.models.cache import KVCache
        self.kv_cache = KVCache()

    def get_hidden_state(self, inputs, cache, speculative=False):
        """Run main model and return pre-norm hidden states + logits.

        Delegates to the shared speculative_forward() function.
        """
        return speculative_forward(self.model, inputs, cache, speculative)

    def _attn_mlp(self, h):
        """Run GQA attention + MLP. Shared by predict, predict_hidden, predict_from_hidden."""
        B, S = h.shape[0], h.shape[1]

        residual = h
        h = self.input_layernorm(h)

        q_out = self.q_proj(h)
        q_out, gate = mx.split(
            q_out.reshape(B, S, self.num_heads, -1), 2, axis=-1
        )
        gate = gate.reshape(B, S, -1)

        queries = self.q_norm(q_out).transpose(0, 2, 1, 3)
        keys = self.k_norm(
            self.k_proj(h).reshape(B, S, self.num_kv_heads, self.head_dim)
        ).transpose(0, 2, 1, 3)
        values = self.v_proj(h).reshape(
            B, S, self.num_kv_heads, self.head_dim
        ).transpose(0, 2, 1, 3)

        if self.kv_cache is not None:
            offset = self.kv_cache.offset
            queries = self.rope(queries, offset=offset)
            keys = self.rope(keys, offset=offset)
            keys, values = self.kv_cache.update_and_fetch(keys, values)
        else:
            queries = self.rope(queries)
            keys = self.rope(keys)

        mask = None
        if S > 1:
            total_kv = keys.shape[2]
            q_pos = mx.arange(S) + (total_kv - S)
            k_pos = mx.arange(total_kv)
            mask = mx.where(k_pos[None, :] <= q_pos[:, None],
                           mx.array(0, dtype=queries.dtype),
                           mx.array(-1e9, dtype=queries.dtype))

        output = mx.fast.scaled_dot_product_attention(
            queries, keys, values, scale=self.scale, mask=mask
        )
        output = output.transpose(0, 2, 1, 3).reshape(B, S, -1)

        h = residual + self.o_proj(output * mx.sigmoid(gate))

        if self.skip_mlp:
            return h  # post-attention, skip FFN (lightweight mode for PP)

        residual = h
        h = self.post_attention_layernorm(h)
        if self.is_moe:
            h = residual + self.mlp(h)
        else:
            h = residual + self.down_proj(nn.silu(self.gate_proj(h)) * self.up_proj(h))

        return h  # post-FFN, pre-norm

    def _combine(self, hidden_state, token_ids):
        """Combine hidden state + token embedding → fc input."""
        B, S = hidden_state.shape[0], hidden_state.shape[1]
        embed = self.embed_tokens(token_ids.reshape(B, S))
        h_norm = self.pre_fc_norm_hidden(hidden_state)
        e_norm = self.pre_fc_norm_embedding(embed)
        return self.fc(mx.concatenate([e_norm, h_norm], axis=-1))

    def predict(self, hidden_state, token_ids, return_hidden=False, draft_mode=False):
        """Predict next-next-token logits using MTP.

        Args:
            hidden_state: (B, S, D) bf16 — PRE-NORM hidden states
            token_ids: (B, S) or (S,) int — tokens at each position
            return_hidden: if True, also return pre-norm hidden for chaining
            draft_mode: if True, use truncated lm_head (32K vocab) for speed
        Returns:
            logits: (B, S, vocab_size) if S>1, (B, vocab_size) if S=1
            If return_hidden: (logits, hidden)
        """
        S = hidden_state.shape[1]
        h = self._combine(hidden_state, token_ids)
        pre_norm_out = self._attn_mlp(h)

        normed = self.norm(pre_norm_out)
        if draft_mode:
            logits = normed @ self.draft_lm_head_weight.T
        elif self.lm_head is not None:
            logits = self.lm_head(normed)
        else:
            logits = self.embed_tokens.as_linear(normed)

        if S == 1:
            logits = logits.squeeze(1)

        if return_hidden:
            return logits, pre_norm_out
        return logits

    def predict_hidden(self, hidden_state, token_ids):
        """Like predict() but returns only post-FFN hidden state (no lm_head)."""
        h = self._combine(hidden_state, token_ids)
        return self._attn_mlp(h)

    def predict_from_hidden(self, prev_hidden):
        """MTP step using post_norm of prev_hidden instead of token embedding.

        Replaces embed_tokens + pre_fc_norm_embedding with just norm(prev_hidden).
        This skips the lm_head → argmax → embed_tokens roundtrip.
        """
        post_norm = self.norm(prev_hidden)
        h_norm = self.pre_fc_norm_hidden(prev_hidden)
        h = self.fc(mx.concatenate([post_norm, h_norm], axis=-1))
        return self._attn_mlp(h)


def broadcast_from_canonical(arr, sync_group):
    """Broadcast ``arr`` from ``sync_group`` rank 0 to every rank.

    Implementation: ``mx.distributed.all_gather`` concatenates every
    rank's array along axis 0; we slice off rank 0's contribution. This
    has identical operator dependencies on every rank (``arr →
    all_gather → slice``) so MLX schedules the collective at the same
    point in each rank's call sequence — keeping JACCL in lock-step.

    A previous implementation used masked all_sum (``arr * indicator``
    where indicator was 1 on rank 0 and 0 elsewhere), but the
    multiply-by-zero on non-rank-0 evidently let MLX desynchronise the
    chained-MTP outputs at temp>0 (every chained-step output collapsed
    to BOS/0 across the cluster, observed at c=1 temp=0.7 and c=2 long
    same temp=0.7). all_gather + slice has no zero-multiply trap and
    is the canonical broadcast primitive in MLX.

    Used to force cross-rank determinism on stochastic outputs (random
    samples, uniforms, rejection-sampled corrections) and on argmax
    outputs that drift due to MLX-level numerical non-determinism in
    unsharded MTP modules. Cheaper than broadcasting full distributions
    and avoids the ``all_min`` bias toward small token IDs that breaks
    temp>0 sampling. Memory: jaccl_phase_f_outcome_2026_05_06.md.

    Caller must ensure ``sync_group`` is non-None and ``size() > 1``;
    bypass the call when running single-rank. ``arr`` must have at
    least one dimension (axis 0 is the rank-stride after the gather).
    """
    gathered = mx.distributed.all_gather(arr, group=sync_group)
    n0 = arr.shape[0]
    return gathered[:n0]


def _compute_eagle_soft_emb(
    logits: mx.array,
    embed_tokens: nn.Embedding,
    K: int,
) -> mx.array:
    """Probability-weighted top-K soft embedding from a draft logit dist.

    Eagle-style mixture: instead of feeding the hard-argmax token's
    embedding into the next chained MTP step, feed
    ``sum_i topk_probs[i] * embed_tokens(topk_ids[i])`` where
    ``topk_*`` are the top-K entries of ``softmax(logits)``, re-normalized
    over the top-K so the mixture weights sum to 1.

    Args:
        logits: ``(B, S, vocab)`` or ``(B, vocab)`` (squeezed S=1) — the
            previous chain step's logits. Both shapes are accepted to
            match the squeezed return of :meth:`DSv4MTPPredictor.predict`.
        embed_tokens: the model's input ``nn.Embedding``.
        K: top-K width. Caller ensures K > 0; K=0 callers skip this fn.
    Returns:
        ``(B, S, hidden)`` soft embedding suitable for the MTP module's
        ``_EAGLE_CTX["soft_emb"]`` slot — same shape as
        ``embed_tokens(token_ids)`` for matching B/S.
    """
    if logits.ndim == 2:
        logits = logits[:, None, :]
    probs = mx.softmax(logits, axis=-1)
    topk_ids = mx.argsort(-logits, axis=-1)[..., :K]  # (B, S, K) int32
    topk_probs = mx.take_along_axis(probs, topk_ids, axis=-1)  # (B, S, K)
    topk_probs = topk_probs / topk_probs.sum(axis=-1, keepdims=True)
    topk_embs = embed_tokens(topk_ids)  # (B, S, K, hidden)
    return (topk_embs * topk_probs[..., None]).sum(axis=-2)  # (B, S, hidden)


def draft_tokens(mtp_pred, hidden, first_token_arr, gamma, temp, fast_lm_head=False, sync_group=None):
    """Draft γ tokens by chaining MTP predictions — fully lazy, no mx.eval.

    The entire chain stays in the MLX computation graph. Draft token ids
    are lazy mx.arrays (argmax/categorical results), not Python ints.

    Args:
        first_token_arr: mx.array of shape (1,1) — the token to start from
        sync_group: optional TP sharding group. When provided and size>1,
            broadcast each step's token ID from rank 0 to all other ranks
            (and the per-step ``q`` softmax used for the acceptance ratio).
            Forces draft determinism across ranks since the MTP module's
            logits can drift by ~1ulp due to MLX-level numerics even with
            bit-exact inputs and unsharded weights. Memory:
            jaccl_phase_f_outcome_2026_05_06.md.
    Returns: (draft_ids, draft_probs) where draft_ids[i] is a lazy mx.array
             scalar, draft_probs[i] is the full draft distribution (or None if greedy)
    """
    draft_ids = []
    draft_probs = []
    h = hidden
    tok_arr = first_token_arr
    sync_drafts = sync_group is not None and sync_group.size() > 1

    # Temporary diagnostic gate: EXO_DSV4_MTP_NO_BROADCAST=1 disables
    # the cross-rank broadcast on temp=0 drafts. Used to isolate whether
    # broadcast_from_canonical is the source of the 2026-05-06 broken-
    # English regression. Remove once the regression is rooted.
    import os as _os_local
    _disable_broadcast = _os_local.environ.get("EXO_DSV4_MTP_NO_BROADCAST") == "1"

    # Eagle soft-embedding chain — opt-in via EXO_DSV4_MTP_EAGLE_K>0 on
    # the predictor. When ON, every chained predict() beyond i=0 has its
    # input embedding replaced by a probability-weighted top-K mixture
    # built from the PREVIOUS step's logits, captured as ``prev_logits``.
    # The predictor exposes ``set_eagle_soft_emb()`` which installs into
    # the model's ``_EAGLE_CTX`` side channel; the channel is cleared in
    # the ``finally`` block so a non-DSv4 caller (or any predict() raise)
    # never leaves the channel populated.
    _eagle_k = int(getattr(mtp_pred, "eagle_k", 0) or 0)
    _eagle_embed = getattr(mtp_pred, "embed_tokens", None) if _eagle_k > 0 else None
    _eagle_active = _eagle_k > 0 and _eagle_embed is not None
    prev_logits: Optional[mx.array] = None

    # Break the chained-collective dependency between successive MTP
    # draft steps. Without an explicit fence, gamma chained predicts
    # queue up gamma lazy `all_sum`s (one per MTP MoE forward) inside
    # the GPU/comm-stream command buffer; each subsequent all_sum is
    # gated on the previous one's CQE delivery. Empirically (2026-05-16
    # diagnostic data, JACCL_POLL_INSTRUMENT + MLX_SIGNAL_PROBE) this
    # produces a peer-CQE arrival tail (~75-107 ms outliers) that
    # collapses gamma>=2 decode into the documented bistability. Forcing
    # `mx.eval(tok_arr)` between iterations drains the buffer one step
    # at a time, eliminating the chained-collective queue buildup.
    # Cost at gamma=1: zero (loop never iterates a second time).
    # Cost at gamma=2: one extra sync per cycle (microseconds).
    # Benefit: eliminates the iter-1+ stall mechanism on gamma>=2.
    for i in range(gamma):
        # Eagle: install soft-emb computed from previous step's logits
        # before the predict() call; clear it after so non-Eagle callers
        # downstream see a clean channel. ``prev_logits is None`` at i=0
        # so the first chain step always uses hard-embed (matching the
        # verify-side input prefix).
        _eagle_installed = _eagle_active and prev_logits is not None
        if _eagle_installed:
            # Cross-rank determinism: prev_logits is rank-local; under
            # MLX's documented per-rank drift the rank-local argmax can
            # flip for tied/near-tied positions, so any rank-local
            # soft-emb diverges between ranks. The tok_arr broadcast
            # below (line 713-716) ALREADY synchronizes argmax across
            # ranks at the end of each iter — reuse that broadcast as
            # the determinism source instead of stacking a second large
            # bf16 collective on the chain critical path. See commit
            # 21ba40db post-mortem (.hermes/plans/2026-05-22 reports)
            # for why broadcasting the soft_emb tensor itself induces a
            # ~17x slowdown.
            assert prev_logits is not None  # narrow for type-checker
            assert _eagle_embed is not None  # _eagle_active gate
            if _eagle_k == 1:
                # K=1: soft-emb == embed_tokens(argmax(prev_logits)) and
                # tok_arr is exactly broadcast(argmax(prev_logits)).
                soft_emb = _eagle_embed(tok_arr)  # (1, 1, hidden)
            else:
                # K>1: broadcast tiny topk_ids + topk_probs and rebuild
                # the mixture locally on every rank.
                _logits3d = (
                    prev_logits
                    if prev_logits.ndim == 3
                    else prev_logits[:, None, :]
                )
                _probs = mx.softmax(_logits3d, axis=-1)
                _topk_ids = mx.argsort(-_logits3d, axis=-1)[..., :_eagle_k]
                _topk_probs = mx.take_along_axis(_probs, _topk_ids, axis=-1)
                _topk_probs = _topk_probs / _topk_probs.sum(
                    axis=-1, keepdims=True
                )
                if sync_drafts and not _disable_broadcast:
                    _topk_ids = broadcast_from_canonical(
                        _topk_ids.astype(mx.int32), sync_group
                    )
                    _topk_probs = broadcast_from_canonical(
                        _topk_probs, sync_group
                    )
                _topk_embs = _eagle_embed(_topk_ids)
                soft_emb = (_topk_embs * _topk_probs[..., None]).sum(axis=-2)
            mtp_pred.set_eagle_soft_emb(soft_emb)
        try:
            logits, h = mtp_pred.predict(h, tok_arr, return_hidden=True,
                                          draft_mode=fast_lm_head)
        finally:
            if _eagle_installed:
                mtp_pred.set_eagle_soft_emb(None)
        prev_logits = logits

        if temp == 0:
            tok_arr = mx.argmax(logits, axis=-1).reshape(1, 1)
            if sync_drafts and not _disable_broadcast:
                tok_arr = broadcast_from_canonical(
                    tok_arr.astype(mx.int32), sync_group
                )
            draft_ids.append(tok_arr.reshape(-1))
            draft_probs.append(None)
            # Token-tree alpha probe (greedy path only). Capture MTP top-5
            # token IDs at this draft step in SORTED-BY-LOGIT order so
            # rank-1=argmax, rank-2=top-2, etc. _speculative_next pairs this
            # with the verify-target argmax and writes JSONL.
            #
            # IMPORTANT: we MATERIALISE top5 to Python ints HERE (synchronous
            # .tolist()) so the queue only holds plain data, never lazy
            # mx.arrays. Earlier versions of this probe stored mx.arrays in
            # the queue and called .tolist() later in _speculative_next; that
            # held lazy refs into the MTP forward graph alive across cycles
            # and produced all-zero MTP logits during multi-request decode
            # (verified 2026-05-19: probe-on→gibberish, probe-off→correct).
            # Cost: one synchronous sync per draft step at vocab~129k argsort
            # — small relative to the MTP MoE forward.
            if TREE_ALPHA_PROBE:
                logits_flat = logits.reshape(-1)
                top5 = mx.argsort(-logits_flat)[:5]
                top5_ids = top5.tolist()  # synchronous; materialise NOW
                _TREE_ALPHA_PROBE_STEPS.append({
                    "step": i,
                    "top5_ids": top5_ids,
                })
        else:
            q = mx.softmax(logits / temp, axis=-1)
            tok_arr = mx.random.categorical(logits * (1.0 / temp)).reshape(1, 1)
            if sync_drafts:
                tok_arr = broadcast_from_canonical(
                    tok_arr.astype(mx.int32), sync_group
                )
                # NOTE: q (softmax) stays per-rank. The downstream
                # acceptance check (ratio = p/q with verify-side p) will
                # diverge by ~1ulp; we sync n_accepted at the end of the
                # spec cycle rather than broadcasting a vocab-sized q
                # every draft step.
            draft_ids.append(tok_arr.reshape(-1))
            draft_probs.append(q)

        # Per-step fence — see comment above the loop.
        if i + 1 < gamma:
            mx.eval(tok_arr)

    return draft_ids, draft_probs


def draft_tokens_topk(
    mtp_pred,
    hidden,
    first_token_arr,
    gamma,
    K,
    sync_group=None,
):
    """Draft a TREE of K^gamma candidate token paths via top-K MTP expansion.

    Phase 2 of the token-tree drafting plan
    (.hermes/plans/2026-05-19_token_tree_drafting.md). Linear `draft_tokens`
    chains gamma sequential argmax MTP forwards; this builds a tree where
    each depth-d MTP forward expands K children from each depth-(d-1)
    parent. For K=2 gamma=2: 1 (root) + 2 (depth-1) + 4 (depth-2) = 7 nodes.

    Args:
        mtp_pred: DSv4MTPPredictor (or Qwen MTPPredictor) — has `predict()`
            and `_cache` with `.trim(n)`.
        hidden: pre-norm hidden shape `(1, 1, D)` at the last verify position
            (= the seed for the root MTP forward).
        first_token_arr: mx.array `(1, 1)` int — last committed token. This
            is tree node 0's token (the verify-input prefix).
        gamma: tree depth (number of draft steps). Production: 2.
        K: branching factor (top-K per parent). Production: 2.
        sync_group: optional TP coord subgroup. When >1 rank, broadcast each
            MTP-emitted top-K token vector from rank 0 to keep all ranks
            bit-exact on the tree structure (same rationale as
            `draft_tokens`).

    Returns: (tree_tokens, parent_idx, depth) where each is a Python
        ``list[int]`` of length ``n_nodes = sum(K**d for d in range(gamma+1))``.
        BFS order: node 0 is the root, nodes 1..K are depth-1 children of
        root in MTP-logit-rank order, etc.

    Cache semantics: between depth-1 siblings (and between depth-2 siblings)
    we call ``mtp_pred._cache.trim(1)`` to roll back the MTP KV cache before
    re-running from the same parent. After the tree is built, the cache is
    LEFT at offset = L_kv + 1 (root MTP forward consumed, but no extant
    branch's depth-2 forward consumed). _speculative_next is responsible
    for trimming back to L_kv (root) on cycle end so the next cycle starts
    fresh -- mirrors the linear `trim(rollback)` semantics.

    Greedy only (temp=0) for v1. Stochastic tree drafting requires per-
    branch q-tracking and is out of scope for the production decode path.
    """
    sync_drafts = sync_group is not None and sync_group.size() > 1
    import os as _os_local
    _disable_broadcast = (
        _os_local.environ.get("EXO_DSV4_MTP_NO_BROADCAST") == "1"
    )
    # Greedy tree: only expand depth-2 children for the TOP-1 depth-1
    # sibling, not all K. Cuts tree from 1+K+K^2 to 1+K+K nodes (e.g.,
    # 7 -> 5 for K=2 gamma=2). The intuition: top-2 d1 is already
    # unlikely; spending verify budget on its depth-2 children pays
    # off rarely at long context where MTP correlation decays. The
    # 5-node tree still preserves the top-1 chain (which dominates
    # acceptance at greedy temp=0) and the top-2 d1 candidate (which
    # lifts depth-1 acceptance over linear's single chain). Enable
    # with EXO_DSV4_TREE_GREEDY=1.
    _greedy = _os_local.environ.get("EXO_DSV4_TREE_GREEDY") == "1"

    # Tree node lists. Node 0 = root = first_token_arr.
    # We materialise first_token_arr to a Python int (sync) so we can put
    # it in tree_tokens as a plain int. This is cheap (the token was just
    # committed, already on host).
    root_tok = int(first_token_arr.reshape(-1)[0].item())
    tree_tokens: list[int] = [root_tok]
    parent_idx: list[int] = [-1]
    depth_arr: list[int] = [0]

    # node_hidden[i] holds the MTP post-block hidden state to use as the
    # NEXT MTP forward's input hidden (i.e., the hidden output of the
    # MTP forward that PRODUCED node i's token). For the root, this is
    # the seed `hidden` passed in (verify-side pre_norm at last position).
    # Only used for nodes that will spawn children — depth-(gamma-1) and
    # shallower. We keep it indexed by node id for clarity.
    node_hidden: list[Any] = [hidden]  # type: ignore[name-defined]

    # MTP forward at the root: input = (hidden, first_token). Output: top-K
    # token logits + a new hidden h_root. The h_root becomes the seed for
    # all K depth-1 MTP forwards.
    logits_root, h_root = mtp_pred.predict(
        hidden, first_token_arr, return_hidden=True
    )
    # Cache offset is now L_kv + 1 (root MTP forward consumed one step).
    # Top-K (sorted by logit desc). Materialise to Python ints
    # IMMEDIATELY — keeping lazy mx.arrays across the rest of this
    # function would entangle with the MTP cache buffer (see 2026-05-19
    # probe bug). top_K_arr is shape (K,).
    top_K_arr = mx.argsort(-logits_root.reshape(-1))[:K]
    if sync_drafts and not _disable_broadcast:
        # Broadcast the K-vector of token IDs from rank 0 so all ranks
        # agree on the same tree shape downstream.
        top_K_arr = broadcast_from_canonical(
            top_K_arr.astype(mx.int32), sync_group
        )
    depth1_ids = top_K_arr.tolist()  # sync; cheap (K ints)

    for k in range(K):
        tok_id = int(depth1_ids[k])
        tree_tokens.append(tok_id)
        parent_idx.append(0)
        depth_arr.append(1)
        node_hidden.append(h_root)

    # If gamma == 1 we're done: tree is depth-1 only.
    if gamma <= 1:
        return tree_tokens, parent_idx, depth_arr

    # Depth-2 expansion. For each depth-1 node (indices 1..K), seed the
    # MTP with (h_root, that node's token) → get a fresh logits and
    # post-block hidden. Take top-K of those logits → K depth-2 children
    # of this depth-1 node.
    #
    # CRITICAL: cache fanout. After processing depth-1 node 1, the MTP
    # cache is at offset L_kv + 2 (root step + node-1 step). Before
    # processing depth-1 node 2 we must trim(1) back to L_kv + 1 so the
    # node-2 forward sees the same context as node-1 did.
    #
    # Greedy mode (EXO_DSV4_TREE_GREEDY=1): only expand the TOP-1
    # depth-1 sibling. Top-2+ d1 siblings get no children (verify
    # budget saved). Caller trims MTP cache by gamma-n_accepted as
    # usual; the cache will be at L_kv+2 only when d1_node=1 actually
    # ran a depth-2 forward, which always happens, so the cache trim
    # math is unchanged.
    d1_range = (1, 2) if _greedy else (1, K + 1)
    for d1_node in range(d1_range[0], d1_range[1]):
        tok_id = tree_tokens[d1_node]
        tok_arr_in = mx.array([[tok_id]])
        # The hidden input for this MTP forward is h_root (the output
        # of the root MTP forward), shared across all depth-1 siblings.
        # Each call re-runs MTP from offset L_kv + 1.
        logits_d1, h_d1 = mtp_pred.predict(
            node_hidden[d1_node], tok_arr_in, return_hidden=True
        )
        # Cache offset is now L_kv + 2.
        top_K_d1 = mx.argsort(-logits_d1.reshape(-1))[:K]
        if sync_drafts and not _disable_broadcast:
            top_K_d1 = broadcast_from_canonical(
                top_K_d1.astype(mx.int32), sync_group
            )
        d2_ids = top_K_d1.tolist()  # sync; K ints

        for k in range(K):
            tree_tokens.append(int(d2_ids[k]))
            parent_idx.append(d1_node)
            depth_arr.append(2)
            node_hidden.append(h_d1)  # would seed depth-3 children if gamma>2

        # Trim back to offset L_kv + 1 so the NEXT depth-1 sibling sees
        # the same context. Skip the trim after the LAST sibling we'll
        # iterate over -- the cache stays at L_kv + 2, and the caller
        # trims by gamma=2 at cycle end (see _speculative_next's tree
        # rollback).
        #
        # In greedy mode the loop only iterates d1_node=1, so the "last"
        # check is `d1_node + 1 < d1_range[1]` not `d1_node < K`. The
        # old `d1_node < K` would trim after the only iteration in
        # greedy mode -> MTP cache at L_kv + 1 instead of L_kv + 2,
        # which the caller's trim(gamma) then over-trims to L_kv - 1.
        # Caught by 2026-05-20 bench bistability (29.5/8.6/22.3 t/s).
        if d1_node + 1 < d1_range[1]:
            cache = getattr(mtp_pred, "_cache", None)
            if cache is not None and hasattr(cache, "trim"):
                cache.trim(1)
            elif cache is not None and hasattr(cache, "offset"):
                cache.offset -= 1

    # NOTE: gamma >= 3 fan-out not implemented in v1. Per plan section 7,
    # K=2 gamma=2 and K=3 gamma=2 are the realistic operating points.
    if gamma > 2:
        raise NotImplementedError(
            "draft_tokens_topk: gamma > 2 not implemented in v1 (plan "
            "section 7 lists K=2 gamma=2 and K=3 gamma=2 as the target "
            "configs; deeper trees fail the wall-cost gate)."
        )

    # Reorder BFS -> DFS-prefix so the MOST-LIKELY path (top-1 d1 -> top-1 d2)
    # lives at contiguous columns [0, 1, 2..gamma]. This lets the
    # _speculative_next_tree post-accept fast path skip the commit forward
    # when the most-likely chain is accepted (the common case in spec decoding).
    #
    # BFS layout (full, K=2 gamma=2):
    #   [root, d1[0], d1[1], d2[0,0], d2[0,1], d2[1,0], d2[1,1]]   (7 nodes)
    # DFS-prefix (full):
    #   [root, d1[0], d2[0,0], d2[0,1], d1[1], d2[1,0], d2[1,1]]
    #
    # BFS layout (greedy, K=2 gamma=2): top-1 d1 expands, top-2 d1 has no kids:
    #   [root, d1[0], d1[1], d2[0,0], d2[0,1]]                     (5 nodes)
    # DFS-prefix (greedy):
    #   [root, d1[0], d2[0,0], d2[0,1], d1[1]]
    #
    # The MTP forwards above are independent of column order (they use
    # `node_hidden[d1_node]` keyed on the BFS-id), so we can permute the
    # final lists without re-running anything. parent_idx is rewritten in
    # the new col-id space; depth_arr is unchanged in values but reordered.
    if gamma == 2 and K >= 1:
        # Number of d2 children PER d1 sibling: K in full mode, K only
        # under d1[0] in greedy mode (d1[1..K-1] get zero d2 children).
        # We assign d2_children[k] = K (k=0 always; k>0 only if not greedy).
        n_d2_children = [K] + [(0 if _greedy else K) for _ in range(K - 1)]
        # BFS cols of d2 children of d1[k]: contiguous block starting at
        # offset 1 + K + sum(n_d2_children[:k]).
        d2_bfs_base = [0] * K
        running = 1 + K
        for k in range(K):
            d2_bfs_base[k] = running
            running += n_d2_children[k]

        # Build permutation: BFS col -> DFS col.
        bfs_to_dfs = [0] * len(tree_tokens)
        dfs_col = 1
        for k in range(K):
            d1_bfs = 1 + k
            bfs_to_dfs[d1_bfs] = dfs_col
            dfs_col += 1
            for j in range(n_d2_children[k]):
                d2_bfs = d2_bfs_base[k] + j
                bfs_to_dfs[d2_bfs] = dfs_col
                dfs_col += 1
        # Apply permutation.
        n = len(tree_tokens)
        new_tokens = [0] * n
        new_parent = [0] * n
        new_depth = [0] * n
        for bfs_col in range(n):
            new_col = bfs_to_dfs[bfs_col]
            new_tokens[new_col] = tree_tokens[bfs_col]
            new_depth[new_col] = depth_arr[bfs_col]
            old_parent = parent_idx[bfs_col]
            new_parent[new_col] = -1 if old_parent < 0 else bfs_to_dfs[old_parent]
        tree_tokens = new_tokens
        parent_idx = new_parent
        depth_arr = new_depth

    return tree_tokens, parent_idx, depth_arr
