#!/usr/bin/env python3
"""Long-context quality harness for DSv4-Flash — the gate for
numerics-perturbing decode/prefill changes (fp8 indexer pool, head-shared
SDPA, draft-head variants, ...).

Per context rung (default 100K / 256K / 500K):
  * builds a fixed-seed word-soup body (longctx_bench style) with K=10
    needles ("The secret code for <name> is <6 digits>.") planted at
    controlled depths (5%..95%)
  * one temp=0 request asks for ALL K codes -> needle recall = fraction
    of codes exactly reproduced in the final content
  * the same generation is LONG (max_tokens 3000), so its decode tok/s
    doubles as the acceptance-band metric (per-token acceptance noise
    averages out over a long fixed-prompt generation)
  * body content is identical across runs (fixed seed); only a tiny uuid
    salt prefix defeats the prefix cache, so A/B variants prefill the
    same needles at the same depths

Also parses MTP acceptance windows from ~/exo.log for the request's time
range when EXO_DSV4_MTP_LOG_INTERVAL is active (opportunistic — absent in
prod config, present in relaunch_exo_v2.sh).

Output: one JSON line per rung + a summary block; use --out FILE to also
append JSON lines to a results file for baseline/variant comparison.

Usage: quality_harness.py [--label NAME] [--out FILE] [ctx_tokens ...]
Runs ON m4-1 against localhost. Runtime ~55 min for the default ladder
(dominated by the 500K prefill).
"""
import argparse
import json
import random
import re
import sys
import time
import urllib.request
import uuid

DEFAULT_URL = "http://localhost:52415/v1/chat/completions"
DEFAULT_MODEL = "mlx-community/DeepSeek-V4-Flash"
DEFAULT_LADDER = [100000, 256000, 500000]
K_NEEDLES = 10
GEN_TOKENS = 8000
WORDS_PER_TOKEN = 0.77  # word-soup ratio measured from longctx_bench runs
PESSIMISTIC_PREFILL_TPS = 40.0

WORDS = (
    "alpha bravo charlie delta echo foxtrot golf hotel india juliet kilo lima "
    "mike november oscar papa quebec romeo sierra tango uniform victor whiskey "
    "yankee zulu ocean river mountain forest desert island valley harbor "
    "bridge tunnel castle garden market station library museum theater anchor "
    "signal beacon compass lantern meadow canyon glacier prairie lagoon summit"
).split()

# needle names use vocab DISJOINT from the body words so retrieval is
# unambiguous (no accidental collisions with the noise)
NEEDLE_COLORS = "crimson amber violet teal indigo scarlet cobalt jade onyx pearl".split()
NEEDLE_GEMS = "quartz topaz garnet opal beryl zircon spinel agate flint jasper".split()


def build_prompt(n_words: int, seed: int, k: int):
    rng = random.Random(seed)
    body_words = [WORDS[rng.randrange(len(WORDS))] for _ in range(n_words)]
    needles = []
    for i in range(k):
        name = f"{NEEDLE_COLORS[i]}-{NEEDLE_GEMS[i]}"
        code = f"{rng.randrange(100000, 999999)}"
        needles.append((name, code))
        depth_frac = (i + 0.5) / k
        pos = int(depth_frac * len(body_words))
        body_words.insert(
            pos, f". The secret code for {name} is {code} ."
        )
    body = " ".join(body_words)
    salt = uuid.uuid4().hex[:12]
    names_list = ", ".join(n for n, _ in needles)
    prompt = (
        f"[session {salt}] Below is a long transcript of radio callsigns "
        f"and waypoints. Hidden inside it are secret codes for these "
        f"entities: {names_list}. Read all of it carefully.\n\n{body}\n\n"
        f"Now, without any analysis or preamble, immediately list the secret "
        f"code for each of the following, one per line in the format "
        f"'name: code' and nothing else: {names_list}."
    )
    return prompt, needles


def run_rung(target: int, k: int, label: str, url: str, model: str):
    n_words = max(1000, int(target * WORDS_PER_TOKEN))
    prompt, needles = build_prompt(n_words, seed=target, k=k)
    body = json.dumps({
        "model": model,
        "messages": [{"role": "user", "content": prompt}],
        "max_tokens": GEN_TOKENS,
        "temperature": 0.0,
        "stream": True,
        "stream_options": {"include_usage": True},
    }).encode()
    timeout = target / PESSIMISTIC_PREFILL_TPS + 1800
    req = urllib.request.Request(
        url=url, data=body, headers={"Content-Type": "application/json"}
    )
    t_req_start = time.time()
    t0 = time.monotonic()
    t_first = None
    t_last = None
    n_deltas = 0
    usage = None
    content_parts = []
    reasoning_parts = []
    err = None
    try:
        with urllib.request.urlopen(req, timeout=timeout) as resp:
            for raw in resp:
                line = raw.decode("utf-8", "replace").strip()
                if not line.startswith("data:"):
                    continue
                payload = line[5:].strip()
                if payload == "[DONE]":
                    break
                try:
                    obj = json.loads(payload)
                except json.JSONDecodeError:
                    continue
                if obj.get("usage"):
                    usage = obj["usage"]
                elif "usage" in payload and usage is None:
                    print(f"[dbg-usage-line] {payload[:300]}", file=sys.stderr, flush=True)
                for ch in obj.get("choices", []):
                    delta = ch.get("delta", {})
                    # count BOTH reasoning and content deltas for tok/s
                    piece = delta.get("content") or ""
                    rpiece = delta.get("reasoning_content") or ""
                    if piece or rpiece:
                        now = time.monotonic()
                        if t_first is None:
                            t_first = now
                        t_last = now
                        n_deltas += 1
                    if piece:
                        content_parts.append(piece)
                    if rpiece:
                        reasoning_parts.append(rpiece)
    except Exception as exc:  # noqa: BLE001
        err = f"{type(exc).__name__}: {exc}"
    t_req_end = time.time()

    content = "".join(content_parts)
    reasoning = "".join(reasoning_parts)
    combined = reasoning + "\n" + content
    per_needle = []
    hits = 0
    for name, code in needles:
        found = code in combined
        # stricter: code appears near the name (same line or within 80 chars)
        near = bool(re.search(re.escape(name) + r".{0,80}" + re.escape(code),
                              combined, re.DOTALL))
        per_needle.append({"name": name, "code": code,
                           "found": found, "near_name": near})
        hits += 1 if found else 0

    prompt_tokens = (usage or {}).get("prompt_tokens")
    completion_tokens = (usage or {}).get("completion_tokens")
    ttft = (t_first - t0) if t_first else None
    decode_s = (t_last - t_first) if (t_first and t_last) else None
    result = {
        "label": label,
        "target": target,
        "prompt_tokens": prompt_tokens,
        "prefill_tps": (
            round(prompt_tokens / ttft, 1) if (prompt_tokens and ttft) else None
        ),
        "completion_tokens": completion_tokens,
        "decode_tps": (
            round((n_deltas - 1) / decode_s, 2)
            if (decode_s and n_deltas > 1) else None
        ),
        "needle_recall": f"{hits}/{len(needles)}",
        "recall_frac": round(hits / len(needles), 3),
        "per_needle": per_needle,
        "t_start_epoch": round(t_req_start, 1),
        "t_end_epoch": round(t_req_end, 1),
        "err": err,
        "content_tail": content[-200:],
        "reasoning_len": len(reasoning),
        "content_len": len(content),
        "in_content": sum(1 for _, c in needles if c in content),
    }
    return result


def parse_acceptance_windows(t_start: float, t_end: float):
    """Opportunistic: pull [MTP] mean_accept windows from ~/exo.log within
    the request's wall-clock range (present only when
    EXO_DSV4_MTP_LOG_INTERVAL is set on the runner)."""
    import os
    import datetime
    path = os.path.expanduser("~/exo.log")
    windows = []
    try:
        with open(path, "rb") as f:
            f.seek(max(0, f.seek(0, 2) - 8_000_000))
            text = f.read().decode("utf-8", "replace")
        for m in re.finditer(
            r"\[ (\d{4}-\d{2}-\d{2} \d{2}:\d{2}:\d{2})\.\d+.*?"
            r"mean_accept[= ]([0-9.]+)", text
        ):
            ts = datetime.datetime.strptime(
                m.group(1), "%Y-%m-%d %H:%M:%S"
            ).timestamp()
            if t_start <= ts <= t_end:
                windows.append(float(m.group(2)))
    except Exception:  # noqa: BLE001
        pass
    return windows


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--label", default="unlabeled")
    ap.add_argument("--out", default=None)
    ap.add_argument("--needles", type=int, default=K_NEEDLES)
    ap.add_argument("--url", default=DEFAULT_URL)
    ap.add_argument("--model", default=DEFAULT_MODEL)
    ap.add_argument("ladder", nargs="*", type=int, default=None)
    args = ap.parse_args()
    ladder = args.ladder or DEFAULT_LADDER

    results = []
    for target in ladder:
        print(f'{{"start": {target}, "t": "{time.strftime("%H:%M:%S")}"}}',
              flush=True)
        r = run_rung(target, args.needles, args.label, args.url, args.model)
        acc = parse_acceptance_windows(
            r["t_start_epoch"], r["t_end_epoch"]
        )
        if acc:
            r["mtp_accept_windows"] = {
                "n": len(acc),
                "mean": round(sum(acc) / len(acc), 3),
                "min": min(acc),
            }
        results.append(r)
        print(json.dumps(r), flush=True)
        if args.out:
            with open(args.out, "a") as f:
                f.write(json.dumps(r) + "\n")

    print("\n=== SUMMARY ===", flush=True)
    for r in results:
        acc = r.get("mtp_accept_windows")
        print(
            f'ctx {r["target"]:>7}: recall {r["needle_recall"]}  '
            f'decode {r["decode_tps"]} t/s  prefill {r["prefill_tps"]} t/s'
            + (f'  accept {acc["mean"]}' if acc else "")
            + (f'  ERR {r["err"]}' if r["err"] else ""),
            flush=True,
        )
    bad = [r for r in results if r["err"]]
    print("HARNESS_DONE" + ("_WITH_ERRORS" if bad else ""), flush=True)


if __name__ == "__main__":
    main()
