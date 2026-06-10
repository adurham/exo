# type: ignore
#!/usr/bin/env python3
"""Dual-model concurrent 100K-context aggregate-throughput benchmark.

Measures the exact target scenario: 2 concurrent streams PER MODEL at 100K
context, for both Qwen3.6 and DSv4 simultaneously (4 streams total), and
reports aggregate decode tok/s over the request set, plus per-model and
per-stream breakdown, needle recall, and BOS-spam.

Reuses the proven 100K methodology from mtp_longctx_probe.py verbatim
(build_prompt + run_once: streaming decode_tps, prefix-cache disabled,
content+reasoning_content parsing, needle + special-token scan) so the
numbers are comparable to the single-stream MTP A/B runs.

Aggregate t/s is the SUM of every concurrent stream's decode tok/s in an
iteration (what the cluster delivers in total while all streams run).

Usage (on a node, where localhost:52415 is the master):
    .venv/bin/python bench/dual_model_c2_100k.py \
        --target-tokens 100000 --iters 5 --max-tokens 200 \
        --streams-per-model 2
"""
from __future__ import annotations

import argparse
import statistics
import time
from concurrent.futures import ThreadPoolExecutor, as_completed

# Reuse the exact prompt builder + single-request runner the fork already
# uses for its 100K MTP probes.
from mtp_longctx_probe import build_prompt, run_once, SPECIAL_TOKEN_MARKERS

QWEN = "mlx-community/Qwen3.6-35B-A3B-8bit"
DSV4 = "mlx-community/DeepSeek-V4-Flash"


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--base-url", default="http://localhost:52415")
    ap.add_argument("--target-tokens", type=int, default=100000)
    ap.add_argument("--iters", type=int, default=5)
    ap.add_argument("--max-tokens", type=int, default=200)
    ap.add_argument("--streams-per-model", type=int, default=2)
    ap.add_argument(
        "--char-ratio-fix",
        type=float,
        default=1.9,
        help="Multiplier on target-tokens passed to build_prompt to correct "
        "for the probe's optimistic 2.9 chars/token assumption. The English "
        "filler prose actually tokenizes at ~5.5 chars/token, so the probe "
        "undershoots by ~1.9x. Set 1.0 to use the probe's raw sizing.",
    )
    ap.add_argument(
        "--models",
        default=f"{QWEN},{DSV4}",
        help="Comma-separated model ids to drive concurrently.",
    )
    ap.add_argument(
        "--seed",
        type=int,
        default=7749,
        help="Base prompt seed; each stream gets a distinct derived seed so "
        "the server-side prefix cache can't serve a warm prefill.",
    )
    args = ap.parse_args()

    models = [m.strip() for m in args.models.split(",") if m.strip()]
    spm = args.streams_per_model

    # Build a distinct 100K prompt per stream up front (defeat prefix cache
    # AND avoid timing the prompt construction inside the concurrent window).
    # stream key = (model, idx)
    import random

    build_target = int(args.target_tokens * args.char_ratio_fix)
    prompts: dict[tuple[str, int], tuple[str, str]] = {}
    for m in models:
        for s in range(spm):
            random.seed(args.seed + hash((m, s)) % 100000)
            prompts[(m, s)] = build_prompt(build_target)

    n_streams = len(models) * spm
    print(
        f"=== dual-model c={spm}/model 100K agg-throughput: "
        f"{len(models)} models x {spm} streams = {n_streams} concurrent streams, "
        f"{args.iters} iters, ~{args.target_tokens} tok ctx, max_tokens={args.max_tokens} ==="
    )

    # Per-iteration aggregate t/s, and per-model accumulators.
    agg_rates: list[float] = []
    per_model_rates: dict[str, list[float]] = {m: [] for m in models}
    needle_hits: dict[str, int] = {m: 0 for m in models}
    spam_hits: dict[str, int] = {m: 0 for m in models}
    errors = 0

    for it in range(args.iters):
        results: dict[tuple[str, int], dict] = {}

        def fire(key: tuple[str, int]) -> tuple[tuple[str, int], dict]:
            model, _ = key
            prompt, _needle = prompts[key]
            r = run_once(args.base_url, model, prompt, args.max_tokens)
            return key, r

        t0 = time.perf_counter()
        with ThreadPoolExecutor(max_workers=n_streams) as ex:
            futs = [ex.submit(fire, k) for k in prompts]
            for f in as_completed(futs):
                key, r = f.result()
                results[key] = r
        wall = time.perf_counter() - t0

        # Aggregate = sum of all stream decode rates this iteration.
        iter_agg = 0.0
        line_parts = []
        for m in models:
            m_rate = 0.0
            for s in range(spm):
                r = results.get((m, s), {})
                tps = r.get("decode_tps", 0.0) or 0.0
                ntok = r.get("n_tokens", 0)
                full = r.get("full", "")
                needle = prompts[(m, s)][1]
                if needle in full:
                    needle_hits[m] += 1
                if any(mk in full for mk in SPECIAL_TOKEN_MARKERS):
                    spam_hits[m] += 1
                if ntok <= 1:
                    errors += 1
                m_rate += tps
            per_model_rates[m].append(m_rate)
            iter_agg += m_rate
            line_parts.append(f"{m.split('/')[-1]}={m_rate:.1f}tps")
        agg_rates.append(iter_agg)
        print(
            f"iter {it+1} (wall {wall:.1f}s): AGG={iter_agg:.1f} tok/s  ::  "
            + "  ".join(line_parts)
        )

    print(f"\n=== SUMMARY ({args.iters} iters, {n_streams} concurrent streams @ ~{args.target_tokens} tok) ===")
    print(
        f"  AGGREGATE decode tok/s: mean={statistics.mean(agg_rates):.1f}  "
        f"median={statistics.median(agg_rates):.1f}  "
        f"min={min(agg_rates):.1f}  max={max(agg_rates):.1f}"
        + (f"  stdev={statistics.pstdev(agg_rates):.2f}" if len(agg_rates) > 1 else "")
    )
    for m in models:
        rates = per_model_rates[m]
        denom = args.iters * spm
        print(
            f"  {m.split('/')[-1]}: per-model agg mean={statistics.mean(rates):.1f} tok/s "
            f"(~{statistics.mean(rates)/spm:.1f}/stream)  "
            f"needle={needle_hits[m]}/{denom}  BOS-spam={spam_hits[m]}/{denom}"
        )
    print(f"  degenerate streams (<=1 tok): {errors}/{args.iters * n_streams}")


if __name__ == "__main__":
    main()
