"""Probe c=2 GPU-utilization vs c=1 to confirm/refute MoE-bandwidth hypothesis.

Drives the cluster API with c=1 and c=2 streams. Worker-side, requires
the runner to expose ``mx.metal.gpu_time_ns()`` (i.e. mlx@e56c37e8+) and
to be launched with ``MLX_GPU_TIME=1``. The per-stream tok/s already
gives wall-time-per-cycle; this script calls the new
``/v1/internal/gpu-time`` runner endpoint to read busy-ns deltas across
the same N decode tokens, then computes ``GPU% = busy / wall``.

If c=2 ``GPU%`` is near 100%, the cycle is GPU-bound and dispatch /
schedule fusions can't help much; the gap to 30 tok/s/stream is mostly
fixed kernel/memory work and only MTP self-spec or kernel-level work
will move it. If c=2 ``GPU%`` is much less than 100%, there is idle
time (collectives, Python, sync barriers) that overlap-style C++
changes could recover.

This script is a thin client; it doesn't depend on the worker endpoint
existing yet. If the endpoint is missing it just reports wall numbers.
"""
import argparse
import asyncio
import json
import statistics
import time

import httpx


API = "http://192.168.86.201:52415"
MODEL = "mlx-community/DeepSeek-V4-Flash-8bit"
FILLER = (
    "The observer pattern is a software design pattern in which an object "
    "named the subject maintains a list of its dependents called observers "
    "and notifies them of state changes. A binary search tree is a rooted "
    "binary data structure with each internal node greater than the keys "
    "in its left subtree and less than those in its right. "
)


def make_prompt(target_tokens: int) -> str:
    n = max(1, target_tokens // 220)
    return (
        "Read the following technical context and answer in 1 sentence.\n\n"
        + (FILLER * n)
        + "\nQuestion: name one of the topics above."
    )


async def stream_one(
    client: httpx.AsyncClient, prompt: str, max_tokens: int
) -> tuple[float, int, float]:
    """Return (decode tok/s, completion_tokens, decode_wall_seconds)."""
    payload = {
        "model": MODEL,
        "messages": [{"role": "user", "content": prompt}],
        "max_tokens": max_tokens,
        "temperature": 0,
        "stream": True,
    }
    t_send = time.time()
    t_first = None
    last = None
    usage = None
    async with client.stream("POST", f"{API}/v1/chat/completions", json=payload, timeout=600.0) as r:
        async for line in r.aiter_lines():
            if not line.startswith("data: "):
                continue
            d = line[6:]
            if d == "[DONE]":
                break
            try:
                o = json.loads(d)
            except Exception:
                continue
            if o.get("usage"):
                usage = o["usage"]
            if o.get("choices"):
                delta = o["choices"][0].get("delta", {})
                c = delta.get("content") or delta.get("reasoning_content") or ""
                if c:
                    now = time.time()
                    if t_first is None:
                        t_first = now
                    last = now
    t_end = time.time()
    ttft = t_first - t_send if t_first else 0.0
    decode_wall = t_end - t_send - ttft
    ct = (usage or {}).get("completion_tokens", 0)
    rate = ct / decode_wall if decode_wall > 0 else 0.0
    return rate, int(ct), decode_wall


async def at(client: httpx.AsyncClient, ctx: int, c: int, max_tokens: int = 64):
    prompts = [make_prompt(ctx) for _ in range(c)]
    results = await asyncio.gather(*[stream_one(client, p, max_tokens) for p in prompts])
    rates = [r[0] for r in results]
    walls = [r[2] for r in results]
    tokens = [r[1] for r in results]
    if c == 1:
        rate = rates[0]
        wall = walls[0]
        toks = tokens[0]
        print(f"  ctx={ctx:>6} c=1: rate={rate:6.2f} tok/s wall={wall:6.3f}s toks={toks}")
        return rate, wall, toks
    rate_med = statistics.median(rates)
    rate_agg = sum(rates)
    wall_max = max(walls)  # both streams run in parallel
    toks_total = sum(tokens)
    print(
        f"  ctx={ctx:>6} c=2: per-stream median={rate_med:6.2f} agg={rate_agg:6.2f} "
        f"wall_max={wall_max:6.3f}s tokens_total={toks_total}"
    )
    return rate_med, wall_max, toks_total


async def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--ctx", type=int, nargs="*", default=[1000, 50000, 100000])
    parser.add_argument("--max-tokens", type=int, default=64)
    parser.add_argument("--trials", type=int, default=3)
    args = parser.parse_args()

    async with httpx.AsyncClient() as client:
        for ctx in args.ctx:
            for c in (1, 2):
                trials_data = []
                for _ in range(args.trials):
                    trials_data.append(await at(client, ctx, c, args.max_tokens))
                rates = [t[0] for t in trials_data]
                walls = [t[1] for t in trials_data]
                med_rate = statistics.median(rates)
                med_wall = statistics.median(walls)
                print(
                    f"  ==> ctx={ctx} c={c}: median rate={med_rate:6.2f} "
                    f"median wall={med_wall:6.3f}s"
                )


if __name__ == "__main__":
    asyncio.run(main())
