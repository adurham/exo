# type: ignore
#!/usr/bin/env python3
"""Bit-equivalence probe for the c=2 batched-prefill rework.

Fires N c=1 deterministic-decode requests at temperature=0 with seed=42 and
captures the generated tokens. Two back-to-back runs should produce
byte-identical token streams. The plan calls for this rig at every phase
end that ships a behavior change (Phase 2/3/4/5).

Usage:
    # Capture baseline (commit before the change):
    uv run python bench/c2_bitequiv_probe.py \
        --model mlx-community/DeepSeek-V4-Flash-DSv4-8bit \
        --out bench/results/bitequiv_baseline.json

    # Capture post-change run, then diff:
    uv run python bench/c2_bitequiv_probe.py \
        --model mlx-community/DeepSeek-V4-Flash-DSv4-8bit \
        --out bench/results/bitequiv_post.json
    uv run python bench/c2_bitequiv_probe.py \
        --diff bench/results/bitequiv_baseline.json bench/results/bitequiv_post.json

The cluster must already be running with the model deployed.
"""

from __future__ import annotations

import argparse
import json
import sys
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path
from typing import Any

from harness import ExoClient, ExoHttpError
from loguru import logger

# Fixed deterministic prompts. Mix lengths so we exercise short and
# medium contexts in c=1; for c=2 we test paired-stream determinism in
# the same script via --concurrency 2.
_PROMPTS: list[str] = [
    "Write a one-paragraph technical explanation of how a CPU pipeline handles branch prediction.",
    "List the first 20 prime numbers, comma-separated.",
    "Explain the difference between RDMA and TCP for cluster interconnect, in 3 short paragraphs.",
    "Describe the M4 Max memory hierarchy: registers, L1, L2, system memory. One paragraph each.",
    "Define tensor parallelism vs pipeline parallelism for LLM inference. Short and concrete.",
    "Walk through how the Bully algorithm elects a leader in a distributed cluster.",
    "Summarize how event sourcing differs from CRUD persistence, with one example each.",
    "Write a short proof that the square root of 2 is irrational.",
]


def _post_one(
    client: ExoClient,
    model: str,
    prompt: str,
    max_tokens: int,
    seed: int,
) -> dict[str, Any]:
    payload: dict[str, Any] = {
        "model": model,
        "messages": [{"role": "user", "content": prompt}],
        "stream": False,
        "max_tokens": max_tokens,
        "temperature": 0.0,
        "seed": seed,
    }
    out = client.post_bench_chat_completions(payload)
    # Best-effort: capture both the message text AND raw choices if present.
    choices = out.get("choices") or []
    text = ""
    if choices:
        msg = (choices[0] or {}).get("message") or {}
        text = msg.get("content") or ""
    return {
        "prompt": prompt,
        "max_tokens": max_tokens,
        "seed": seed,
        "text": text,
        "stats": out.get("generation_stats") or {},
    }


def _capture(
    host: str,
    port: int,
    timeout_s: float,
    model: str,
    max_tokens: int,
    seed: int,
    concurrency: int,
) -> dict[str, Any]:
    client = ExoClient(host, port, timeout_s=timeout_s)

    # Sanity check: server reachable, model present.
    try:
        state = client.request_json("GET", "/state")
    except ExoHttpError as e:
        raise RuntimeError(f"Cannot reach exo at {host}:{port}: {e}") from e

    matching: list[str] = []
    for inst_id, inst in (state.get("instances") or {}).items():
        for variant_key in ("MlxRingInstance", "MlxJacclInstance", "MlxInstance"):
            variant = inst.get(variant_key) or {}
            sa = variant.get("shardAssignments") or {}
            if sa.get("modelId") == model:
                matching.append(inst_id)
                break
    if not matching:
        raise RuntimeError(
            f"No running instance for model {model}. Place at least one first."
        )

    results: list[dict[str, Any]] = []

    if concurrency == 1:
        # Serial — exercises the c=1 prefill path exactly as production.
        for prompt in _PROMPTS:
            r = _post_one(client, model, prompt, max_tokens, seed)
            logger.info(
                f"[c=1] prompt='{prompt[:48]}...' tokens={r['stats'].get('generation_tokens')} "
                f"len(text)={len(r['text'])}"
            )
            results.append(r)
    else:
        # Concurrent — fires `concurrency` prompts in parallel each batch.
        # We chunk _PROMPTS into batches of size `concurrency`, run each
        # batch in parallel, and preserve per-prompt determinism by pinning
        # the same seed to the same prompt regardless of arrival order.
        for batch_start in range(0, len(_PROMPTS), concurrency):
            batch = _PROMPTS[batch_start : batch_start + concurrency]
            with ThreadPoolExecutor(max_workers=len(batch)) as pool:
                futures = {
                    pool.submit(
                        _post_one,
                        ExoClient(host, port, timeout_s=timeout_s),
                        model,
                        prompt,
                        max_tokens,
                        seed,
                    ): prompt
                    for prompt in batch
                }
                batch_results: dict[str, dict[str, Any]] = {}
                for fut in as_completed(futures):
                    p = futures[fut]
                    batch_results[p] = fut.result()
            # Re-order by original prompt order so the output is
            # stable across runs.
            for prompt in batch:
                r = batch_results[prompt]
                logger.info(
                    f"[c={concurrency}] prompt='{prompt[:48]}...' tokens={r['stats'].get('generation_tokens')} "
                    f"len(text)={len(r['text'])}"
                )
                results.append(r)

    return {
        "model": model,
        "max_tokens": max_tokens,
        "seed": seed,
        "concurrency": concurrency,
        "instance_count_at_start": len(matching),
        "results": results,
    }


def _diff(a_path: str, b_path: str) -> int:
    a = json.loads(Path(a_path).read_text())
    b = json.loads(Path(b_path).read_text())

    if a["model"] != b["model"]:
        logger.error(f"model mismatch: a={a['model']} b={b['model']}")
        return 2
    if a["max_tokens"] != b["max_tokens"] or a["seed"] != b["seed"]:
        logger.error("max_tokens/seed mismatch — runs not comparable")
        return 2
    if a["concurrency"] != b["concurrency"]:
        logger.warning(
            f"concurrency differs: a={a['concurrency']} b={b['concurrency']} "
            "— this is fine if you're cross-checking c=1 vs c=2 outputs of the same prompts"
        )

    a_results = a["results"]
    b_results = b["results"]
    if len(a_results) != len(b_results):
        logger.error(f"result count mismatch: a={len(a_results)} b={len(b_results)}")
        return 2

    mismatches: list[tuple[int, str]] = []
    for i, (ra, rb) in enumerate(zip(a_results, b_results, strict=False)):
        if ra["prompt"] != rb["prompt"]:
            mismatches.append((i, "prompt order changed"))
            continue
        if ra["text"] != rb["text"]:
            # Find the first mismatching character index for diagnostics.
            ta = ra["text"]
            tb = rb["text"]
            common = 0
            for ca, cb in zip(ta, tb, strict=False):
                if ca != cb:
                    break
                common += 1
            mismatches.append(
                (
                    i,
                    f"text diverges at char {common}: "
                    f"a={ta[common : common + 32]!r} b={tb[common : common + 32]!r}",
                )
            )

    if not mismatches:
        logger.info(
            f"OK — {len(a_results)} prompts byte-identical between {a_path} and {b_path}"
        )
        return 0

    logger.error(f"{len(mismatches)}/{len(a_results)} prompts differ:")
    for idx, msg in mismatches:
        prompt = a_results[idx]["prompt"]
        logger.error(f"  [{idx}] '{prompt[:60]}...' — {msg}")
    return 1


def main() -> int:
    ap = argparse.ArgumentParser(prog="c2-bitequiv-probe")
    ap.add_argument("--host", default="192.168.86.201")
    ap.add_argument("--port", type=int, default=52415)
    ap.add_argument("--timeout", type=float, default=600.0)
    ap.add_argument(
        "--model",
        help="Model id deployed on the cluster.",
    )
    ap.add_argument("--max-tokens", type=int, default=64)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument(
        "--concurrency",
        type=int,
        default=1,
        help="1 (serial baseline) or N>1 (per-batch parallel — exercises c=N path).",
    )
    ap.add_argument(
        "--out",
        help="Path to write per-prompt JSON results (capture mode).",
    )
    ap.add_argument(
        "--diff",
        nargs=2,
        metavar=("A", "B"),
        help="Diff two captured JSON files for byte-identity. Skips capture.",
    )
    args = ap.parse_args()

    if args.diff is not None:
        return _diff(args.diff[0], args.diff[1])

    if not args.model or not args.out:
        ap.error("--model and --out are required in capture mode (no --diff).")

    captured = _capture(
        args.host,
        args.port,
        args.timeout,
        args.model,
        args.max_tokens,
        args.seed,
        args.concurrency,
    )

    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps(captured, indent=2))
    logger.info(f"wrote {len(captured['results'])} results to {out_path}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
