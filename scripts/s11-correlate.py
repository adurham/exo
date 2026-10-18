#!/usr/bin/env python3
"""Join each client row (run-s11.py output) to the runner's own [DSV41] obs lines.

§11.6 documents the chain: the response `id` is the command_id; the master/runner
logs print `Starting task TextGeneration(task_id='<turn>', ..., command_id='<id>')`;
the obs lines key on `turn=<task_id>`. This script implements exactly that chain
(command_id -> task_id -> obs first-token / obs turn-end), with a prompt-token +
time-window fallback when the log has rotated past the command line.

Server decode definition matches §9.1: (committed - 1) / (turn-end - first-token).
Prefill tok/s (uncached) = (prompt_tokens - cached_tokens) / prefill_s.

Usage: python3 s11-correlate.py --out ./out [--nodes macstudio-m4-1 macstudio-m4-2] [--refresh]
"""

from __future__ import annotations

import argparse
import glob
import json
import os
import re
import subprocess
import sys
from datetime import datetime

FIRST_RE = re.compile(
    r"^\[\s*(?P<ts>\d{4}-\d\d-\d\d \d\d:\d\d:\d\d\.\d+)\s*\|[^]]*\]\s*\[DSV41\] obs first-token "
    r"turn=(?P<turn>[0-9a-f-]+) latency=(?P<lat>[\d.]+)s \(prefill (?P<prefill>[\d.]+)s, "
    r"(?P<prompt>\d+) prompt tokens, reuse=(?P<reuse>\d+)\)"
)
END_RE = re.compile(
    r"^\[\s*(?P<ts>\d{4}-\d\d-\d\d \d\d:\d\d:\d\d\.\d+)\s*\|[^]]*\]\s*\[DSV41\] obs turn-end "
    r"turn=(?P<turn>[0-9a-f-]+) committed=(?P<committed>\d+) emitted=(?P<emitted>\d+) "
    r"elapsed=(?P<elapsed>[\d.]+)s tok_s=(?P<tok_s>[\d.]+) reason=(?P<reason>\S+)"
)
CMD_RE = re.compile(r"command_id='(?P<cmd>[0-9a-f-]+)'")
TASK_RE = re.compile(r"task_id='(?P<turn>[0-9a-f-]+)'")
TS = "%Y-%m-%d %H:%M:%S.%f"
ANSI_RE = re.compile(r"\x1b\[[0-9;]*m")


def clean(line: str) -> str:
    """Runner log lines carry ANSI colour codes (`\\x1b[1mINFO\\x1b[0m`); strip them
    before matching or every regex misses."""
    return ANSI_RE.sub("", line).strip()


def ssh_lines(node: str, pattern: str):
    cmd = ["ssh", "-o", "ConnectTimeout=10", node, f"grep -hE {json.dumps(pattern)} ~/exo.log | tail -800"]
    try:
        out = subprocess.run(cmd, capture_output=True, text=True, timeout=120)
    except subprocess.TimeoutExpired:
        return []
    return out.stdout.splitlines()


def parse_first(lines):
    rows = []
    for ln in lines:
        m = FIRST_RE.match(clean(ln))
        if m:
            rows.append(
                {
                    "node": None,
                    "epoch": datetime.strptime(m["ts"], TS).timestamp(),
                    "ts": m["ts"],
                    "turn": m["turn"],
                    "latency_s": float(m["lat"]),
                    "prefill_s": float(m["prefill"]),
                    "prompt_tokens": int(m["prompt"]),
                    "reuse": int(m["reuse"]),
                }
            )
    return rows


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default=os.path.join(os.path.dirname(os.path.abspath(__file__)), "out"))
    ap.add_argument("--nodes", nargs="+", default=["macstudio-m4-1", "macstudio-m4-2"])
    ap.add_argument("--slack-before", type=float, default=300.0)
    ap.add_argument("--slack-after", type=float, default=1800.0)
    ap.add_argument("--refresh", action="store_true", help="re-correlate rows that already have a server block")
    args = ap.parse_args()

    firsts = []
    for node in args.nodes:
        for row in parse_first(ssh_lines(node, r"\[DSV41\] obs first-token")):
            row["node"] = node
            firsts.append(row)

    ends = {}
    for node in args.nodes:
        for ln in ssh_lines(node, r"\[DSV41\] obs turn-end"):
            m = END_RE.match(clean(ln))
            if m:
                ends.setdefault(m["turn"], []).append(
                    {
                        "node": node,
                        "epoch": datetime.strptime(m["ts"], TS).timestamp(),
                        "committed": int(m["committed"]),
                        "emitted": int(m["emitted"]),
                        "elapsed_s": float(m["elapsed"]),
                        "tok_s_field": float(m["tok_s"]),
                        "reason": m["reason"],
                    }
                )

    # command_id -> task_id, straight from the line §11.6 names.
    task_map = {}
    for node in args.nodes:
        for ln in ssh_lines(node, r"Starting task TextGeneration"):
            c = clean(ln)
            cmd, task = CMD_RE.search(c), TASK_RE.search(c)
            if cmd and task:
                task_map.setdefault(cmd["cmd"], (task["turn"], node))

    updated = 0
    for path in sorted(glob.glob(os.path.join(args.out, "*.json"))):
        with open(path) as fh:
            res = json.load(fh)
        if res.get("server", {}).get("matched") and not args.refresh:
            continue
        prompt_tokens = res.get("prompt_tokens")
        t0, t1 = res.get("started_epoch"), res.get("ended_epoch")
        hit = None
        method = None
        rid = res.get("response_id")
        if rid and rid in task_map:
            turn, _node = task_map[rid]
            cands = [r for r in firsts if r["turn"] == turn]
            if cands:
                hit, method = cands[0], "command_id->task_id"
        if hit is None:
            cand = [
                r
                for r in firsts
                if prompt_tokens is not None
                and r["prompt_tokens"] == prompt_tokens
                and r["epoch"] >= (t0 - args.slack_before)
                and r["epoch"] <= (t1 + args.slack_after)
            ]
            if cand:
                cand.sort(key=lambda r: abs(r["epoch"] - t0))
                hit, method = cand[0], "prompt_tokens+time_window"
        if hit is None:
            res["server"] = {"matched": False, "reason": "no obs first-token line matched"}
            with open(path, "w") as fh:
                json.dump(res, fh, indent=1)
            continue
        end = (ends.get(hit["turn"]) or [None])[0]
        server = {
            "matched": True,
            "method": method,
            "node": hit["node"],
            "turn": hit["turn"],
            "first_token_epoch": hit["epoch"],
            "first_token_ts": hit["ts"],
            "latency_s": hit["latency_s"],
            "prefill_s": hit["prefill_s"],
            "prompt_tokens": hit["prompt_tokens"],
            "reuse": hit["reuse"],
        }
        cached = res.get("cached_tokens") or 0
        if hit["prefill_s"] > 0:
            server["prefill_tok_s_uncached"] = round((hit["prompt_tokens"] - cached) / hit["prefill_s"], 1)
        if end:
            server["committed"] = end["committed"]
            server["emitted"] = end["emitted"]
            server["server_elapsed_s"] = end["elapsed_s"]
            server["server_reason"] = end["reason"]
            server["tok_s_field"] = end["tok_s_field"]
            dec_window = end["epoch"] - hit["epoch"]
            if dec_window > 0 and end["committed"] > 1:
                server["server_decode_tok_s"] = round((end["committed"] - 1) / dec_window, 2)
        res["server"] = server
        with open(path, "w") as fh:
            json.dump(res, fh, indent=1)
        updated += 1
        print(
            json.dumps(
                {
                    "row": res["label"],
                    "method": method,
                    "turn": server["turn"],
                    "node": server["node"],
                    "prefill_s": server["prefill_s"],
                    "reuse": server["reuse"],
                    "server_decode_tok_s": server.get("server_decode_tok_s"),
                    "prefill_tok_s_uncached": server.get("prefill_tok_s_uncached"),
                }
            )
        )
    print(f"correlated {updated} row(s); first-token lines seen: {len(firsts)}; command->task ids: {len(task_map)}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
