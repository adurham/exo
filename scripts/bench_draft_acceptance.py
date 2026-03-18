#!/usr/bin/env python3
"""Measure speculative decode acceptance rate for draft/target model pairs.

Generates tokens from the target model (235B via API), then prefills the
draft model with the SAME prompt tokens and checks if the draft model
predicts the same output tokens.

This must run on a machine that can load the draft model AND reach the
cluster API. The MacBook is ideal.

Usage:
  ssh macbook-m4 'cd ~/repos/exo && .venv/bin/python scripts/bench_draft_acceptance.py'
  ssh macbook-m4 'cd ~/repos/exo && .venv/bin/python scripts/bench_draft_acceptance.py \
    --draft mlx-community/Qwen3-30B-A3B-4bit'
"""
import argparse
import json
import sys
import time
import urllib.request

API = "http://192.168.86.201:52415"
TARGET_MODEL = "mlx-community/Qwen3-235B-A22B-Instruct-2507-6bit"

FILLER = (
    "The architecture of modern distributed systems relies on careful coordination "
    "between multiple nodes. Each node maintains its own local state while participating "
    "in global consensus protocols. The challenge lies in balancing consistency with "
    "availability, as described by the CAP theorem. Network partitions are inevitable "
    "in any sufficiently large system, requiring careful design of failure recovery "
    "mechanisms. Replication strategies must account for both synchronous and asynchronous "
    "communication patterns, with trade-offs in latency and durability. "
)


def get_target_output(api: str, model: str, prompt: str, max_tokens: int) -> str:
    """Generate from the target model via API."""
    payload = json.dumps({
        "model": model,
        "messages": [{"role": "user", "content": prompt}],
        "max_tokens": max_tokens,
        "temperature": 0,
        "enable_thinking": False,
    }).encode()
    req = urllib.request.Request(
        f"{api}/v1/chat/completions",
        data=payload,
        headers={"Content-Type": "application/json"},
    )
    with urllib.request.urlopen(req, timeout=600) as resp:
        r = json.loads(resp.read())
    return r["choices"][0]["message"]["content"]


def measure_acceptance(
    prompt: str,
    target_text: str,
    draft_model,
    tokenizer,
) -> tuple[float, int, int]:
    """Measure acceptance rate by prefilling draft with prompt, then comparing predictions.

    Returns (acceptance_rate, matches, total_checked).
    """
    import mlx.core as mx
    from mlx_lm.models.cache import make_prompt_cache

    # Tokenize the full prompt (same chat template as target)
    messages = [{"role": "user", "content": prompt}]
    prompt_tokens = tokenizer.apply_chat_template(
        messages, add_generation_prompt=True, tokenize=True
    )

    # Tokenize the target output
    output_tokens = tokenizer.encode(target_text, add_special_tokens=False)
    if len(output_tokens) < 3:
        return 0.0, 0, 0

    # Prefill the draft model with the full prompt
    cache = make_prompt_cache(draft_model)
    prompt_array = mx.array([prompt_tokens])
    logits = draft_model(prompt_array, cache=cache)
    mx.eval(logits)

    # The model's prediction for the first output token
    first_pred = logits[0, -1].argmax().item()

    matches = 0
    checked = 0

    # Check first token
    if first_pred == output_tokens[0]:
        matches += 1
    checked += 1

    # Feed each output token and check if draft predicts the next one
    for i in range(len(output_tokens) - 1):
        tok = mx.array([[output_tokens[i]]])
        logits = draft_model(tok, cache=cache)
        mx.eval(logits)
        predicted = logits[0, -1].argmax().item()
        actual = output_tokens[i + 1]
        if predicted == actual:
            matches += 1
        checked += 1

    rate = matches / checked if checked > 0 else 0
    return rate, matches, checked


def main():
    parser = argparse.ArgumentParser(description="Measure draft model acceptance rate")
    parser.add_argument("--api", default=API)
    parser.add_argument("--target-model", default=TARGET_MODEL)
    parser.add_argument("--draft", default="mlx-community/Qwen3-1.7B-8bit")
    parser.add_argument("--max-tokens", type=int, default=100)
    args = parser.parse_args()

    # Load draft model
    print(f"Loading draft model: {args.draft}")
    from mlx_lm.utils import load
    draft_model, tokenizer = load(args.draft)
    print(f"Loaded.\n")

    # Test prompts at various context sizes
    test_cases = [
        ("Write a Python fibonacci function.", "short"),
        ("Explain distributed consensus in detail.", "short"),
        (f"Summarize: {FILLER * 3}", "~1K"),
        (f"Summarize: {FILLER * 15}", "~5K"),
        (f"Summarize: {FILLER * 30}", "~10K"),
        (f"Summarize: {FILLER * 75}", "~25K"),
    ]

    print(f"{'Prompt':>40} {'Context':>8} {'Accept':>8} {'Match':>6} {'Total':>6}")
    print("-" * 75)

    total_matches = 0
    total_checked = 0
    results_by_size = {}

    for prompt, size_label in test_cases:
        # Generate from target
        target_text = get_target_output(args.api, args.target_model, prompt, args.max_tokens)

        # Measure acceptance
        rate, matches, checked = measure_acceptance(
            prompt, target_text, draft_model, tokenizer
        )

        total_matches += matches
        total_checked += checked
        results_by_size.setdefault(size_label, []).append(rate)

        prompt_short = prompt[:37] + "..." if len(prompt) > 40 else prompt
        print(f"{prompt_short:>40} {size_label:>8} {rate:>7.1%} {matches:>6} {checked:>6}")

    print("-" * 75)
    overall = total_matches / total_checked if total_checked > 0 else 0
    print(f"{'OVERALL':>40} {'':>8} {overall:>7.1%} {total_matches:>6} {total_checked:>6}")

    print(f"\nBy context size:")
    for size, rates in results_by_size.items():
        avg = sum(rates) / len(rates)
        print(f"  {size}: {avg:.1%}")

    # Break-even analysis
    print(f"\nBreak-even analysis (baseline 36ms/tok at 1K, 81ms/tok at 100K):")
    for k in [3, 5]:
        # At acceptance rate `r`, average tokens per step = sum(r^i for i in 0..k) = (1 - r^(k+1)) / (1 - r)
        # But we always get at least 1 token (the verified one)
        for label, baseline_ms in [("1K ctx", 36), ("25K ctx", 48), ("100K ctx", 81)]:
            r = overall
            avg_accepted = sum(r ** i for i in range(1, k + 1))
            tokens_per_step = 1 + avg_accepted  # 1 verified + accepted drafts
            # Cost per step: draft_compute + verify_compute
            # Sequential verify: draft_compute + tokens_per_step * verify_ms
            # Batch verify: draft_compute + 1 * verify_ms
            draft_ms = k * 7.5  # 1.7B at 7.5ms/tok
            # Sequential verify
            seq_cost = draft_ms + tokens_per_step * baseline_ms
            seq_ms_per_tok = seq_cost / tokens_per_step
            seq_speedup = baseline_ms / seq_ms_per_tok
            print(f"  K={k}, {label}, {overall:.0%} accept: "
                  f"{tokens_per_step:.1f} tok/step, "
                  f"seq verify {seq_ms_per_tok:.0f}ms/tok ({seq_speedup:.2f}x)")


if __name__ == "__main__":
    main()
