#!/usr/bin/env python3
"""GPU-utilization probe during a real long-context prefill run.

Starts a `macmon` sampler on every cluster node (sudoless, 1 Hz), fires ONE
needle-in-haystack chat-completion request at the API node, then stops the
samplers and correlates the GPU-busy time series against the server log's
`Prefill progress:` bracket.

Why this exists: span-time percentages measure wall-clock of a CODE REGION,
which can include CPU-side graph build/dispatch while the GPU is idle. This
measures GPU business directly, to rule in/out hidden idle bubbles.

Usage:
    python bench/gpu_util_needle_probe.py --target-tokens 200000
"""

from __future__ import annotations

import argparse
import json
import random
import string
import subprocess
import sys
import time
from datetime import datetime, timezone
from pathlib import Path

NODES = ["macstudio-m4-1", "macstudio-m4-2"]
API_HOST = "adams-mac-studio-m4-1.local"
API_PORT = 52415
MACMON = "~/.cargo/bin/macmon"
CHARS_PER_TOKEN = 5.68  # measured for this tokenizer on English prose

WORDS = [
    "".join(random.choices(string.ascii_lowercase, k=random.randint(3, 9)))
    for _ in range(4000)
]


def make_prompt(target_tokens: int, secret: str) -> str:
    target_chars = int(target_tokens * CHARS_PER_TOKEN)
    rng = random.Random(1234)
    parts: list[str] = []
    n = 0
    while n < target_chars:
        w = rng.choice(WORDS)
        parts.append(w)
        n += len(w) + 1
    mid = len(parts) // 2
    parts.insert(
        mid,
        f"The secret access code is {secret}. Remember it.",
    )
    return " ".join(parts)


def start_samplers(run_dir: Path, interval_ms: int) -> None:
    for node in NODES:
        remote = f"/tmp/gpu_util_{node}.jsonl"
        subprocess.run(
            ["ssh", node, f"pkill -f 'macmon pipe' ; rm -f {remote}"],
            check=False,
            capture_output=True,
        )
        subprocess.Popen(
            [
                "ssh",
                node,
                f"nohup {MACMON} pipe -i {interval_ms} -s 0 > {remote} 2>/dev/null &",
            ],
            stdout=subprocess.DEVNULL,
            stderr=subprocess.DEVNULL,
        )
    time.sleep(3)


def stop_and_fetch(run_dir: Path) -> dict[str, Path]:
    out: dict[str, Path] = {}
    for node in NODES:
        remote = f"/tmp/gpu_util_{node}.jsonl"
        subprocess.run(["ssh", node, "pkill -f 'macmon pipe'"], check=False,
                       capture_output=True)
        local = run_dir / f"gpu_util_{node}.jsonl"
        with local.open("wb") as fh:
            subprocess.run(["ssh", node, f"cat {remote}"], stdout=fh, check=True)
        out[node] = local
    return out


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--target-tokens", type=int, default=200_000)
    ap.add_argument("--interval-ms", type=int, default=1000)
    ap.add_argument("--outdir", default="/tmp/gpu_util_probe")
    args = ap.parse_args()

    run_dir = Path(args.outdir)
    run_dir.mkdir(parents=True, exist_ok=True)

    secret = "".join(random.choices(string.ascii_uppercase + string.digits, k=10))
    prompt = make_prompt(args.target_tokens, secret)
    (run_dir / "prompt.txt").write_text(prompt)
    print(f"secret={secret} prompt_chars={len(prompt)} "
          f"est_tokens~{len(prompt)/CHARS_PER_TOKEN:.0f}", flush=True)

    body = {
        "model": "deepseek-ai/DeepSeek-V4-Flash-0731",
        "messages": [
            {"role": "user",
             "content": prompt + "\n\nWhat is the secret access code? "
                                 "Answer with the code only."},
        ],
        "max_tokens": 50,
        "temperature": 0,
    }
    (run_dir / "request.json").write_text(json.dumps(body))

    start_samplers(run_dir, args.interval_ms)
    t0 = datetime.now(timezone.utc)
    print(f"request_start_utc={t0.isoformat()}", flush=True)
    started = time.monotonic()
    proc = subprocess.run(
        ["curl", "-s", "-m", "3600", "-w", "\\nHTTP:%{http_code}",
         "-H", "Content-Type: application/json",
         "-X", "POST", f"http://{API_HOST}:{API_PORT}/v1/chat/completions",
         "--data-binary", f"@{run_dir/'request.json'}"],
        capture_output=True, text=True,
    )
    elapsed = time.monotonic() - started
    t1 = datetime.now(timezone.utc)
    print(f"request_end_utc={t1.isoformat()} elapsed_s={elapsed:.1f}", flush=True)
    (run_dir / "response.txt").write_text(proc.stdout)

    files = stop_and_fetch(run_dir)
    meta = {
        "secret": secret,
        "prompt_chars": len(prompt),
        "request_start_utc": t0.isoformat(),
        "request_end_utc": t1.isoformat(),
        "elapsed_s": elapsed,
        "util_files": {k: str(v) for k, v in files.items()},
    }
    (run_dir / "meta.json").write_text(json.dumps(meta, indent=2))
    print(json.dumps(meta, indent=2), flush=True)
    print(proc.stdout[-2000:], flush=True)
    return 0


if __name__ == "__main__":
    sys.exit(main())
