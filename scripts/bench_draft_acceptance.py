#!/usr/bin/env python3
"""Measure speculative decode acceptance rate for draft/target model pairs.

Generates text with the target model, then checks how many tokens
the draft model would have predicted. No speculative decode needed —
just token-by-token comparison.

Usage (on a node with both models loaded):
  uv run python scripts/bench_draft_acceptance.py \
    --target mlx-community/Qwen3-235B-A22B-Instruct-2507-6bit \
    --draft mlx-community/Qwen3-0.6B-8bit

Or via the API (target generates, draft checks locally):
  uv run python scripts/bench_draft_acceptance.py \
    --target-api http://192.168.86.201:52415 \
    --draft mlx-community/Qwen3-0.6B-8bit
"""
import argparse
import json
import time
import urllib.request


def get_target_tokens_via_api(api_url, model_id, prompt, max_tokens=100):
    """Generate tokens from the target model via API."""
    data = json.dumps({
        "model": model_id,
        "messages": [{"role": "user", "content": prompt}],
        "max_tokens": max_tokens,
        "temperature": 0.0,  # greedy for reproducibility
    }).encode()
    req = urllib.request.Request(
        f"{api_url}/v1/chat/completions",
        data=data,
        headers={"Content-Type": "application/json"},
    )
    with urllib.request.urlopen(req, timeout=300) as resp:
        result = json.loads(resp.read())
    return result["choices"][0]["message"]["content"]


def measure_acceptance(target_text, draft_model, tokenizer, max_check=50):
    """Measure how many tokens the draft model would have predicted correctly.

    Simulates speculative decoding: for each position in the target text,
    check if the draft model's greedy prediction matches the target's token.
    """
    import mlx.core as mx
    from mlx_lm.models.cache import make_prompt_cache

    # Tokenize the target text
    target_tokens = tokenizer.encode(target_text)
    if len(target_tokens) < 5:
        return 0.0, 0, len(target_tokens)

    # Create draft cache
    cache = make_prompt_cache(draft_model)

    # Feed tokens one at a time, check if draft predicts the next one
    matches = 0
    checked = 0

    # Prefill with first few tokens
    prefill_tokens = mx.array([target_tokens[:3]])
    draft_model(prefill_tokens, cache=cache)
    mx.eval([c.state for c in cache])

    for i in range(3, min(len(target_tokens) - 1, max_check)):
        # Feed current token, get prediction
        current = mx.array([[target_tokens[i]]])
        logits = draft_model(current, cache=cache)
        mx.eval(logits)

        # Draft's greedy prediction
        predicted = logits[0, -1].argmax().item()
        actual = target_tokens[i + 1]

        if predicted == actual:
            matches += 1
        checked += 1

    rate = matches / checked if checked > 0 else 0
    return rate, matches, checked


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--target-api", default="http://192.168.86.201:52415")
    parser.add_argument("--target-model", default="mlx-community/Qwen3-235B-A22B-Instruct-2507-6bit")
    parser.add_argument("--draft", default="mlx-community/Qwen3-0.6B-8bit")
    parser.add_argument("--max-tokens", type=int, default=100)
    parser.add_argument("--max-check", type=int, default=80)
    args = parser.parse_args()

    prompts = [
        "Write a Python function to compute fibonacci numbers.",
        "Explain how a CPU cache works in simple terms.",
        "What are the main differences between Python and Rust?",
        "Write a haiku about programming.",
        "List the first 10 prime numbers and explain why they matter.",
    ]

    # Load draft model locally
    print(f"Loading draft model: {args.draft}")
    from mlx_lm.utils import load
    draft_model, tokenizer = load(args.draft)
    print(f"Draft loaded: {len(draft_model.model.layers)} layers")
    print()

    total_matches = 0
    total_checked = 0

    print(f"{'Prompt':>50}  {'Accept':>7}  {'Match':>6}  {'Total':>6}")
    print("-" * 80)

    for prompt in prompts:
        # Get target output
        target_text = get_target_tokens_via_api(
            args.target_api, args.target_model, prompt, args.max_tokens)

        # Measure acceptance
        rate, matches, checked = measure_acceptance(
            target_text, draft_model, tokenizer, args.max_check)

        total_matches += matches
        total_checked += checked

        print(f"{prompt[:50]:>50}  {rate:>6.1%}  {matches:>6}  {checked:>6}")

    overall = total_matches / total_checked if total_checked > 0 else 0
    print("-" * 80)
    print(f"{'OVERALL':>50}  {overall:>6.1%}  {total_matches:>6}  {total_checked:>6}")
    print()

    # Calculate break-even analysis
    print("Break-even analysis (assuming 35ms baseline, 40ms verify):")
    for rate_name, rate_val in [("Measured", overall), ("30%", 0.30), ("50%", 0.50)]:
        # Average tokens per step with 3 draft tokens
        avg_tokens = 1 + rate_val + rate_val**2 + rate_val**3
        effective_ms = 40 / avg_tokens  # verify cost / tokens produced
        speedup = 35 / effective_ms
        print(f"  {rate_name} acceptance: {avg_tokens:.2f} tok/step → "
              f"{effective_ms:.1f}ms/tok → {speedup:.2f}× vs baseline")


if __name__ == "__main__":
    main()
