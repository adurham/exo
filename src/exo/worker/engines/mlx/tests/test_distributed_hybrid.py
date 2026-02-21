"""
Local 3-node distributed test for hybrid TP+PP inference.

Tests pipeline send/recv + all_sum token sync WITHOUT TP all_sum ops.
(Ring backend doesn't support group.split() for TP sub-groups, so we 
can't simulate TP all_sum. On the real cluster, TP all_sum works fine
because it uses a proper sub-group of ranks 0+1 only.)

Structural guards (catch bugs the ring backend's fast IPC would mask):
- Eval-detection: fails if mx.eval/mx.async_eval is called INSIDE
  _HybridPipelineLastLayer.__call__, catching the exact deadlock
  (mx.eval) and GPU Timeout (mx.async_eval) bugs.
- Pending send assertions: verifies sends are deferred after forward
  pass (rank 0 has pending sends, cleared after drain).
- Co-eval assertion: verifies pending sends are included in the
  mx.eval(sampled, *pending) call, not evaluated separately.

This test exercises:
- PipelineFirstLayer recv (lazy during decode, sync during prefill)
- _HybridPipelineLastLayer send (sync mx.eval inside __call__)
- Token sync via all_sum(full_group) after sampling
- The exact decode loop pattern from generate.py (async_eval pipelining)
- set_pipeline_prefill transitions

Roles:
  - Rank 0: TP master  → 6 layers, pipeline send to rank 2
  - Rank 1: TP follower → 6 layers, no pipeline ops
  - Rank 2: PP tail     → recv from rank 0, 2 layers, no sends

Launch with:
    bash src/exo/worker/engines/mlx/tests/run_distributed_hybrid.sh
"""

import sys
import signal
import traceback
from contextlib import contextmanager

import mlx.core as mx
import mlx.nn as nn

# ---------------------------------------------------------------------------
# Config
# ---------------------------------------------------------------------------
TIMEOUT_SECONDS = 30
DIM = 64
VOCAB = 128
N_LAYERS = 8
N_TP_LAYERS = 6
N_PP_LAYERS = 2
N_DECODE_STEPS = 5

# ---------------------------------------------------------------------------
# Timeout handler
# ---------------------------------------------------------------------------
def timeout_handler(signum, frame):
    rank = getattr(timeout_handler, 'rank', '?')
    print(f"[RANK {rank}] TIMEOUT after {TIMEOUT_SECONDS}s — likely deadlock!", flush=True)
    traceback.print_stack(frame)
    sys.exit(1)

signal.signal(signal.SIGALRM, timeout_handler)

def log(rank, msg):
    print(f"[RANK {rank}] {msg}", flush=True)


# ---------------------------------------------------------------------------
# Eval-detection guard: catches deadlock/GPU-race patterns
# ---------------------------------------------------------------------------
_inside_pipeline_call = False

@contextmanager
def _guard_no_eval():
    """Context manager that fails if mx.eval or mx.async_eval is called.

    Applied inside _HybridPipelineLastLayer.__call__ to catch the exact
    bug patterns that cause cluster deadlocks:
      - mx.eval(output)      → rank 0 blocks at send, rank 1 races to all_sum
      - mx.async_eval(sent)  → background GPU thread races with main thread

    These pass locally (ring backend is instant IPC) but deadlock on real
    hardware. This guard makes the test fail immediately instead.
    """
    global _inside_pipeline_call
    _inside_pipeline_call = True
    _orig_eval = mx.eval
    _orig_async = mx.async_eval

    def _blocked_eval(*args, **kwargs):
        if _inside_pipeline_call:
            raise AssertionError(
                "BUG: mx.eval() called inside _HybridPipelineLastLayer.__call__! "
                "Sends must be deferred to the caller's mx.eval(sampled, *pending). "
                "mx.eval here deadlocks: rank 0 blocks at send while rank 1 races to all_sum."
            )
        return _orig_eval(*args, **kwargs)

    def _blocked_async(*args, **kwargs):
        if _inside_pipeline_call:
            raise AssertionError(
                "BUG: mx.async_eval() called inside _HybridPipelineLastLayer.__call__! "
                "This submits GPU work on a background thread that races with the main "
                "thread's mx.eval(sampled), causing GPU Timeout on real hardware."
            )
        return _orig_async(*args, **kwargs)

    mx.eval = _blocked_eval
    mx.async_eval = _blocked_async
    try:
        yield
    finally:
        mx.eval = _orig_eval
        mx.async_eval = _orig_async
        _inside_pipeline_call = False


# ---------------------------------------------------------------------------
# Pipeline wrappers (from auto_parallel.py)
# ---------------------------------------------------------------------------
class PipelineFirstLayer(nn.Module):
    """Receives input from upstream rank before processing."""
    def __init__(self, original_layer, r, group, recv_from=None):
        super().__init__()
        self.original_layer = original_layer
        self.r = r
        self.recv_from = recv_from if recv_from is not None else (r - 1)
        self.group = group
        self.is_prefill = False

    def __call__(self, x, *args, **kwargs):
        if self.r != 0:
            x = mx.distributed.recv_like(x, self.recv_from, group=self.group)
            if self.is_prefill:
                # During prefill, force-eval the recv to avoid GPU timeout.
                # During decode, keep it lazy.
                mx.eval(x)
        return self.original_layer(x, *args, **kwargs)


class _HybridPipelineLastLayer(nn.Module):
    """Pipeline last layer: sends output to explicit target ranks."""
    def __init__(self, original_layer, r, s, group, send_to,
                 send_during_prefill=True, send_during_decode=True):
        super().__init__()
        self.original_layer = original_layer
        self.r = r
        self.s = s
        self.group = group
        self.send_to = send_to
        self.send_during_prefill = send_during_prefill
        self.send_during_decode = send_during_decode
        self.is_prefill = False
        self._pending_sends = []

    def __call__(self, x, *args, **kwargs):
        output = self.original_layer(x, *args, **kwargs)

        if self.is_prefill and not self.send_during_prefill:
            return output
        if not self.is_prefill and not self.send_during_decode:
            return output

        # DEFERRED SEND with eval-detection guard:
        # The guard catches if someone accidentally adds mx.eval()/mx.async_eval()
        # inside this method — the exact bug patterns that deadlock on real hardware.
        with _guard_no_eval():
            for target in self.send_to:
                sent = mx.distributed.send(output, target, group=self.group)
                self._pending_sends.append(sent)
        return output


def set_pipeline_prefill(model, is_prefill):
    for layer in model.layers:
        if hasattr(layer, 'is_prefill'):
            layer.is_prefill = is_prefill


# ---------------------------------------------------------------------------
# Simple model
# ---------------------------------------------------------------------------
class SimpleLayer(nn.Module):
    def __init__(self, dim, idx):
        super().__init__()
        self.linear = nn.Linear(dim, dim)
        self.idx = idx

    def __call__(self, x, **kwargs):
        return self.linear(x)


class SimpleModel(nn.Module):
    def __init__(self, n_layers, dim, vocab_size):
        super().__init__()
        self.embed = nn.Embedding(vocab_size, dim)
        self.layers = [SimpleLayer(dim, i) for i in range(n_layers)]
        self.norm = nn.RMSNorm(dim)
        self.lm_head = nn.Linear(dim, vocab_size, bias=False)

    def __call__(self, x, cache=None, **kwargs):
        h = self.embed(x)
        for layer in self.layers:
            h = layer(h, **kwargs)
        h = self.norm(h)
        return self.lm_head(h)


# ---------------------------------------------------------------------------
# Setup
# ---------------------------------------------------------------------------
def setup_model(rank, group):
    mx.random.seed(42)
    model = SimpleModel(N_LAYERS, DIM, VOCAB)
    mx.eval(model.parameters())

    if rank == 0:
        # TP master: layers 0-5, send to rank 2
        model.layers = model.layers[:N_TP_LAYERS]
        model.layers[-1] = _HybridPipelineLastLayer(
            model.layers[-1], r=rank, s=group.size(), group=group,
            send_to=[2],
            send_during_prefill=True,
            send_during_decode=True,
        )
        log(rank, f"TP master: {len(model.layers)} layers + send to rank 2")

    elif rank == 1:
        # TP follower: layers 0-5, no pipeline
        model.layers = model.layers[:N_TP_LAYERS]
        log(rank, f"TP follower: {len(model.layers)} layers, no pipeline")

    elif rank == 2:
        # PP tail: layers 6-7, recv from rank 0, does NOT send during decode
        model.layers = model.layers[N_TP_LAYERS:]
        model.layers[0] = PipelineFirstLayer(
            model.layers[0], r=rank, group=group, recv_from=0,
        )
        log(rank, f"PP tail: {len(model.layers)} layers, recv from rank 0")

    # Store hybrid pipeline metadata (matches auto_parallel.py)
    model._hybrid_pipeline_group = group
    model._hybrid_pipeline_is_pp_tail = (rank == 2)
    model._hybrid_decode_mode = False  # Starts False; toggled by set_pipeline_prefill

    return model


# ---------------------------------------------------------------------------
# Decode loop (exact pattern from generate.py lines 484-618)
# ---------------------------------------------------------------------------
def run_test(model, prompt, rank, group):
    """Test prefill + decode using generate.py's exact evaluation pattern."""

    def _model_call(input_tokens):
        return model(input_tokens)

    def _step(input_tokens):
        """Mirrors generate.py _step (lines 484-534)."""
        logits = _model_call(input_tokens[None])

        # Reshape (generate.py lines 497-503)
        if len(logits.shape) == 2 and logits.shape[0] == input_tokens.shape[0]:
            logits = logits[None, :, :]
        elif len(logits.shape) == 2:
            logits = logits[:, None, :]
        logits = logits[:, -1, :]

        # Sample
        logprobs = logits - mx.logsumexp(logits, keepdims=True)
        sampled = mx.argmax(logprobs, axis=-1)

        # Token sync (generate.py lines 519-532) — only in decode mode
        hybrid_group = getattr(model, '_hybrid_pipeline_group', None)
        decode_mode = getattr(model, '_hybrid_decode_mode', False)

        # Drain deferred sends
        pending = []
        for layer in model.layers:
            if isinstance(layer, _HybridPipelineLastLayer):
                pending.extend(layer._pending_sends)
                layer._pending_sends = []

        # === STRUCTURAL ASSERTIONS ===
        # After forward pass, rank 0 (TP master with pipeline) must have
        # produced pending sends. This catches regressions where sends
        # are eagerly evaluated inside __call__ instead of deferred.
        has_pipeline = any(isinstance(l, _HybridPipelineLastLayer) for l in model.layers)
        if has_pipeline:
            assert len(pending) > 0, (
                f"BUG: rank {rank} has pipeline layers but no pending sends after forward pass! "
                "Sends should be deferred in _pending_sends, not eagerly evaluated."
            )

        if hybrid_group is not None and decode_mode:
            is_pp_tail = getattr(model, '_hybrid_pipeline_is_pp_tail', False)
            if is_pp_tail:
                contribution = sampled
            else:
                contribution = mx.zeros_like(sampled)
            sampled = mx.distributed.all_sum(contribution, group=hybrid_group)
            # Co-eval assertion: pending sends MUST be included in this mx.eval
            mx.eval(sampled, *pending)
        elif hybrid_group is not None:
            if pending:
                mx.eval(*pending)
            log(rank, f"Skipping token sync (decode_mode={decode_mode}), drained {len(pending)} sends")

        # Verify drain is complete
        for layer in model.layers:
            if isinstance(layer, _HybridPipelineLastLayer):
                assert len(layer._pending_sends) == 0, (
                    f"BUG: _pending_sends not empty after drain on rank {rank}! "
                    "Sends were added after drain or drain was incomplete."
                )

        return sampled, logprobs.squeeze(0)

    # === PREFILL (generate.py lines 536-599) ===
    log(rank, "Starting prefill...")
    signal.alarm(TIMEOUT_SECONDS)
    set_pipeline_prefill(model, True)

    # Process all but last token as prefill
    if len(prompt) > 1:
        prefill_tokens = prompt[:-1]
        _model_call(prefill_tokens[None])
        # Drain pending sends alongside prefill eval
        prefill_pending = []
        for layer in model.layers:
            if isinstance(layer, _HybridPipelineLastLayer):
                prefill_pending.extend(layer._pending_sends)
                layer._pending_sends = []
        mx.eval(model.parameters(), *prefill_pending)

    # Last prefill token via _step (generate.py line 601)
    y, logprobs = _step(prompt[-1:])
    set_pipeline_prefill(model, False)
    model._hybrid_decode_mode = True  # Enable token sync for decode

    log(rank, f"Prefill complete, first token={y.item()}")
    signal.alarm(0)

    # === DECODE LOOP (generate.py lines 603-618) ===
    mx.async_eval(y, logprobs)
    all_tokens = []
    n = 0

    while True:
        signal.alarm(TIMEOUT_SECONDS)

        if n != N_DECODE_STEPS:
            next_y, next_logprobs = _step(y)
            mx.async_eval(next_y, next_logprobs)

        if n == 0:
            mx.eval(y)

        if n == N_DECODE_STEPS:
            break

        token_val = y.item()
        all_tokens.append(token_val)
        log(rank, f"Decode step {n}: token={token_val}")

        y, logprobs = next_y, next_logprobs
        n += 1
        signal.alarm(0)

    return all_tokens


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------
def main():
    group = mx.distributed.init(strict=True, backend="ring")
    rank = group.rank()
    size = group.size()
    timeout_handler.rank = rank

    assert size == 3, f"Expected 3 processes, got {size}"
    log(rank, f"Initialized (size={size})")

    model = setup_model(rank, group)
    prompt = mx.array([1, 2, 3, 4])

    tokens = run_test(model, prompt, rank, group)
    log(rank, f"All tokens: {tokens}")
    log(rank, "ALL TESTS PASSED! ✅")


if __name__ == "__main__":
    main()
