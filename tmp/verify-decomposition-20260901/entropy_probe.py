"""
Measures server-side decode throughput at a fixed context depth under prompts of different entropy.

Example invocation:
    PYTHONPATH=/Users/adam.durham/repos/exo/tools/src /Users/adam.durham/repos/exo/.venv/bin/python /Users/adam.durham/repos/exo/tmp/verify-decomposition-20260901/entropy_probe.py \
        --mode natural --target-tokens 89408 --iterations 5
"""

import argparse
import json
import random
import statistics
import sys
import time
from pathlib import Path
from typing import Any

from exo_tools.client import ExoClient

def _fixed_prompt(approx_words: int) -> str:
    """EXACT reproduction of concurrent_bench.py's _fixed_prompt logic."""
    base = (
        "Write a concise technical explanation of how a CPU pipeline handles "
        "branch prediction, including speculative execution and how mispredicts "
        "are recovered. Be precise and avoid filler. "
    )
    repeats = max(1, approx_words // len(base.split()))
    return (base * repeats).strip()

def _natural_prompt(word_count: int, rng: random.Random) -> str:
    """
    High-entropy English-like prose.
    Uses a large varied vocabulary to avoid triggering server-side repetitive 
    output kill-switches.
    """
    # 100+ distinct words to ensure high entropy and avoid degeneration
    vocab = [
        "the", "a", "an", "and", "or", "but", "if", "then", "else", "while",
        "although", "because", "since", "unless", "until", "whether", "which",
        "who", "whom", "whose", "quantum", "complexity", "distributed", "parallel",
        "consensus", "entropy", "throughput", "latency", "optimization", "heuristic",
        "algorithmic", "stochastic", "deterministic", "asynchronous", "synchronous",
        "idempotent", "recursive", "iterative", "polynomial", "exponential", "logarithmic",
        "orthogonal", "convergent", "divergent", "invariant", "covariant", "contravariant",
        "monadic", "isomorphic", "homomorphic", "topology", "manifold", "vector",
        "tensor", "matrix", "scalar", "gradient", "eigenvalue", "eigenvector",
        "orthonormal", "hermitian", "unitary", "symplectic", "kähler", "riemannian",
        "euclidean", "minkowski", "affine", "projective", "compact", "connected",
        "simply-connected", "complete", "separable", "dense", "sparse", "discrete",
        "continuous", "differentiable", "integrable", "holomorphic", "meromorphic",
        "analytic", "algebraic", "transcendental", "rational", "irrational", "prime",
        "composite", "modular", "primitive", "fundamental", "canonical", "axiomatic",
        "empirical", "theoretical", "conceptual", "abstract", "concrete", "formal",
        "rigorous", "heuristic", "intuitive", "probabilistic", "statistical", "Bayesian",
        "frequentist", "markovian", "ergodic", "stationary", "non-stationary", "chaos",
        "fractal", "attractor", "bifurcation", "equilibrium", "stability", "dynamics",
        "kinetics", "thermodynamics", "entropy", "enthalpy", "gibbs", "boltzmann",
        "schrödinger", "dirac", "feynman", "planck", "einstein", "hawking", "penrose"
    ]
    
    words = []
    while len(words) < word_count:
        # Vary sentence length from 5 to 15 words
        sentence_len = rng.randint(5, 15)
        for _ in range(sentence_len):
            if len(words) < word_count:
                words.append(rng.choice(vocab))
        words.append(".") # End sentence
    
    # Join and clean up: capitalize start of sentences, remove space before dots
    text = " ".join(words).replace(" .", ".")
    # Basic capitalization for "natural" look
    parts = text.split(". ")
    capitalized = [p[0].upper() + p[1:] if p else p for p in parts]
    return ". ".join(capitalized)

def _random_prompt(word_count: int, rng: random.Random) -> str:
    """Maximum entropy: random letter-strings as words."""
    alphabet = "abcdefghijklmnopqrstuvwxyz"
    words = []
    for _ in range(word_count):
        # Random word length 3-10
        length = rng.randint(3, 10)
        word = "".join(rng.choice(alphabet) for _ in range(length))
        words.append(word)
    return " ".join(words)

def get_prompt(mode: str, word_count: int, seed: int) -> str:
    rng = random.Random(seed)
    if mode == "repetitive":
        return _fixed_prompt(word_count)
    elif mode == "natural":
        return _natural_prompt(word_count, rng)
    elif mode == "random":
        return _random_prompt(word_count, rng)
    else:
        raise ValueError(f"Unknown mode: {mode}")

def _calibrate(client, args):
    """Server-probe calibration (fallback when --fixed-words is not given).

    Each probe costs a full deep-context prefill, so prefer offline tokenizer
    calibration + --fixed-words whenever cluster time matters.
    """
    current_word_count = args.target_tokens // 2
    print(f"Calibrating prompt length for mode={args.mode}...", end=" ", flush=True)
    for _ in range(5):
        prompt = get_prompt(args.mode, current_word_count, args.seed)
        try:
            out = client.post_bench_chat_completions({
                "model": args.model,
                "messages": [{"role": "user", "content": prompt}],
                "stream": False,
                "max_tokens": 1,
            })
            observed = int(out.get("generation_stats", {}).get("prompt_tokens", 0))
            if observed == 0:
                break
            if abs(observed - args.target_tokens) / args.target_tokens <= 0.02:
                print(f"Done. Achieved {observed} tokens ({current_word_count} words).")
                return current_word_count, prompt, observed
            current_word_count = max(
                1000, int(current_word_count * args.target_tokens / observed)
            )
        except Exception as e:
            print(f"Calibration error: {e}")
            sys.exit(1)
    final_prompt = get_prompt(args.mode, current_word_count, args.seed)
    return current_word_count, final_prompt, None


def main():
    ap = argparse.ArgumentParser(description="Measure decode throughput under different prompt entropy.")
    ap.add_argument("--host", default="192.168.86.201")
    ap.add_argument("--port", type=int, default=52415)
    ap.add_argument("--model", default="deepseek-ai/DeepSeek-V4-Flash-0731")
    ap.add_argument("--mode", choices=["repetitive", "natural", "random"], required=True)
    ap.add_argument("--target-tokens", type=int, default=89408)
    ap.add_argument("--iterations", type=int, default=5)
    ap.add_argument("--warmup", type=int, default=1)
    ap.add_argument("--max-tokens", type=int, default=256)
    ap.add_argument("--timeout", type=float, default=3600.0)
    ap.add_argument("--seed", type=int, default=1234)
    ap.add_argument("--json-out", type=str, default=None)
    ap.add_argument(
        "--fixed-words",
        type=int,
        default=None,
        help=(
            "Skip server-probe calibration and use this exact word count. "
            "Calibrate OFFLINE against the real tokenizer instead -- each "
            "server probe costs a full deep-context prefill (~3.5 min at 89K), "
            "so a 5-attempt calibration would burn ~18 min of cluster time per "
            "mode for information the local tokenizer gives for free."
        ),
    )
    args = ap.parse_args()

    client = ExoClient(args.host, args.port, timeout_s=args.timeout)

    if args.fixed_words is not None:
        current_word_count = args.fixed_words
        final_prompt = get_prompt(args.mode, current_word_count, args.seed)
        achieved_tokens = None  # filled in from the first real request below
        print(
            f"Using pre-calibrated word count for mode={args.mode}: "
            f"{current_word_count} words (offline-tokenizer calibrated)."
        )
    else:
        current_word_count, final_prompt, achieved_tokens = _calibrate(client, args)

    # --- EXECUTION ---
    results = []
    
    # Warmup
    for i in range(args.warmup):
        try:
            client.post_bench_chat_completions({
                "model": args.model,
                "messages": [{"role": "user", "content": final_prompt}],
                "stream": False,
                "max_tokens": args.max_tokens,
            })
        except Exception as e:
            print(f"Warmup error: {e}")

    # Scored iterations
    for i in range(args.iterations):
        try:
            out = client.post_bench_chat_completions({
                "model": args.model,
                "messages": [{"role": "user", "content": final_prompt}],
                "stream": False,
                "max_tokens": args.max_tokens,
            })
            stats = out.get("generation_stats", {})
            if achieved_tokens is None:
                achieved_tokens = int(stats.get("prompt_tokens", 0))
            results.append({
                "iteration": i,
                "generation_tps": float(stats.get("generation_tps", 0.0)),
                "prompt_tps": float(stats.get("prompt_tps", 0.0)),
                "prompt_tokens": int(stats.get("prompt_tokens", 0)),
                "generation_tokens": int(stats.get("generation_tokens", 0)),
            })
        except Exception as e:
            print(f"Iteration {i} error: {e}")

    if not results:
        print("No successful iterations.")
        sys.exit(1)

    gen_tps_vals = [r["generation_tps"] for r in results]
    prompt_tps_vals = [r["prompt_tps"] for r in results]
    
    summary = {
        "mode": args.mode,
        "achieved_prompt_tokens": achieved_tokens,
        "generation_tps_mean": statistics.mean(gen_tps_vals),
        "generation_tps_median": statistics.median(gen_tps_vals),
        "generation_tps_min": min(gen_tps_vals),
        "generation_tps_max": max(gen_tps_vals),
        "prompt_tps_mean": statistics.mean(prompt_tps_vals),
    }

    print(f"\nMode: {summary['mode']}")
    print(f"PROMPT TOKENS: {summary['achieved_prompt_tokens']}")
    print(f"Generation TPS: mean={summary['generation_tps_mean']:.2f}, median={summary['generation_tps_median']:.2f}, min={summary['generation_tps_min']:.2f}, max={summary['generation_tps_max']:.2f}")
    print(f"Prompt TPS: mean={summary['prompt_tps_mean']:.2f}")

    if args.json_out:
        out_data = {
            "config": vars(args),
            "achieved_prompt_tokens": achieved_tokens,
            "iterations": results,
            "summary": summary
        }
        Path(args.json_out).write_text(json.dumps(out_data, indent=2))

if __name__ == "__main__":
    main()
