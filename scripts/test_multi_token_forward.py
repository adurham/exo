#!/usr/bin/env python3
"""Test whether multi-token forward pass produces identical logits to sequential single-token.

This tests the core assumption of batch verify in speculative decoding:
that model(tokens[0:K], cache) produces the same logits as K sequential
model(token[i], cache) calls.

For TP models, this must be run on ALL TP nodes simultaneously.

Usage:
  # On both Studios simultaneously (use start_distributed_test.py or manual):
  # Studio 1: MLX_RANK=0 MLX_WORLD_SIZE=2 python3 scripts/test_multi_token_forward.py
  # Studio 2: MLX_RANK=1 MLX_WORLD_SIZE=2 python3 scripts/test_multi_token_forward.py

  # Or for single-node testing with a small model:
  # python3 scripts/test_multi_token_forward.py --model mlx-community/Qwen3-1.7B-8bit

  # Simplest: just use the API (doesn't test multi-token directly, but tests
  # that the model is deterministic and produces clean output):
  # python3 scripts/test_multi_token_forward.py --api-only
"""
import argparse
import json
import sys
import time
import urllib.request


def test_via_api(api: str, model: str):
    """Test model determinism and output quality via the API.

    This doesn't directly test multi-token forward, but confirms the model
    produces correct, deterministic output at various context sizes.
    """
    FILLER = (
        "The architecture of modern distributed systems relies on careful coordination "
        "between multiple nodes. Each node maintains its own local state while participating "
        "in global consensus protocols. The challenge lies in balancing consistency with "
        "availability, as described by the CAP theorem. Network partitions are inevitable "
        "in any sufficiently large system, requiring careful design of failure recovery "
        "mechanisms. Replication strategies must account for both synchronous and asynchronous "
        "communication patterns, with trade-offs in latency and durability. "
    )

    def generate(prompt: str, max_tokens: int = 20) -> tuple[str, int]:
        payload = json.dumps({
            "model": model,
            "messages": [{"role": "user", "content": prompt}],
            "max_tokens": max_tokens,
            "temperature": 0,
            "enable_thinking": False,
        }).encode()
        req = urllib.request.Request(
            f"{api}/v1/chat/completions", data=payload,
            headers={"Content-Type": "application/json"},
        )
        with urllib.request.urlopen(req, timeout=600) as resp:
            r = json.loads(resp.read())
        return r["choices"][0]["message"]["content"], r["usage"]["prompt_tokens"]

    print("=== Determinism Test (same prompt twice, temp=0) ===\n")
    all_pass = True
    for mult in [3, 30, 150, 750]:
        text = (FILLER * mult)
        prompt = f"Summarize in 20 words: {text}"
        out1, ptoks = generate(prompt)
        time.sleep(1)
        out2, _ = generate(prompt)
        match = out1 == out2
        if not match:
            all_pass = False
        status = "PASS" if match else "FAIL"
        print(f"  [{status}] {ptoks} tokens: {out1[:60]}...")
        if not match:
            print(f"         vs: {out2[:60]}...")

    print(f"\n=== Output Quality Test ===\n")
    for mult in [3, 15, 75, 300, 750, 1200]:
        text = (FILLER * mult)
        prompt = f"Summarize in 20 words: {text}"
        out, ptoks = generate(prompt)
        has_garbage = "<|" in out or "\x00" in out
        if has_garbage:
            all_pass = False
        status = "PASS" if not has_garbage else "FAIL"
        print(f"  [{status}] {ptoks} tokens: {out[:60]}...")

    return all_pass


def test_multi_token_direct(model_id: str):
    """Direct test: compare single-token vs multi-token logits.

    Loads the model (TP if MLX_RANK is set) and runs the comparison.
    """
    import mlx.core as mx
    from mlx_lm.utils import load
    from mlx_lm.models.cache import make_prompt_cache

    print(f"Loading model: {model_id}")
    model, tokenizer = load(model_id)
    print("Model loaded.\n")

    # Encode a test prompt
    prompt_text = "The quick brown fox jumps over the lazy dog. " * 50
    messages = [{"role": "user", "content": f"Repeat this: {prompt_text}"}]
    tokens = tokenizer.apply_chat_template(messages, add_generation_prompt=True)
    prompt = mx.array(tokens)
    print(f"Prompt: {len(tokens)} tokens")

    all_pass = True

    for test_context in [100, 500, 1000]:
        if test_context > len(tokens):
            continue

        print(f"\n--- Context size: {test_context} tokens ---")
        test_prompt = prompt[:test_context]

        # Prefill + single-token decode
        cache_single = make_prompt_cache(model)
        model(test_prompt[:-1][None], cache=cache_single)
        mx.eval([c.state for c in cache_single])

        single_tokens = []
        single_logits = []
        y = test_prompt[-1].item()
        for i in range(10):
            out = model(mx.array([[y]]), cache=cache_single)
            mx.eval(out)
            single_logits.append(out[0, -1])
            y = out[0, -1].argmax().item()
            single_tokens.append(y)

        # Prefill again + multi-token decode
        cache_multi = make_prompt_cache(model)
        model(test_prompt[:-1][None], cache=cache_multi)
        mx.eval([c.state for c in cache_multi])

        multi_input = mx.array([[test_prompt[-1].item()] + single_tokens[:9]])
        multi_out = model(multi_input, cache=cache_multi)
        mx.eval(multi_out)

        # Compare logits at each position
        mismatches = 0
        for i in range(10):
            single_top = single_logits[i].argmax().item()
            multi_top = multi_out[0, i].argmax().item()
            if single_top != multi_top:
                mismatches += 1
                if mismatches <= 3:
                    print(f"  MISMATCH at position {i}: single={single_top} multi={multi_top}")

        if mismatches == 0:
            print(f"  [PASS] All 10 positions match")
        else:
            print(f"  [FAIL] {mismatches}/10 positions differ")
            all_pass = False

    return all_pass


def main():
    parser = argparse.ArgumentParser(description="Test multi-token forward pass correctness")
    parser.add_argument("--api", default="http://192.168.86.201:52415")
    parser.add_argument("--model", default="mlx-community/Qwen3-235B-A22B-Instruct-2507-6bit")
    parser.add_argument("--api-only", action="store_true",
                        help="Only run API-based tests (no direct model loading)")
    parser.add_argument("--direct-only", action="store_true",
                        help="Only run direct model tests (loads model locally)")
    args = parser.parse_args()

    all_pass = True

    if not args.direct_only:
        print("=" * 60)
        print("API-BASED TESTS")
        print("=" * 60)
        try:
            if not test_via_api(args.api, args.model):
                all_pass = False
        except Exception as e:
            print(f"API test failed: {e}")
            all_pass = False

    if not args.api_only:
        print("\n" + "=" * 60)
        print("DIRECT MULTI-TOKEN FORWARD TESTS")
        print("=" * 60)
        try:
            if not test_multi_token_direct(args.model):
                all_pass = False
        except Exception as e:
            print(f"Direct test failed: {e}")
            all_pass = False

    print("\n" + "=" * 60)
    print(f"RESULT: {'ALL PASS' if all_pass else 'SOME FAILURES'}")
    print("=" * 60)
    sys.exit(0 if all_pass else 1)


if __name__ == "__main__":
    main()
