#!/usr/bin/env python3
"""Localize the first divergence / collapse in an MTP spec trace.

Pairs with:
  * the EXO_DSV4_SPEC_TRACE=1 per-cycle dump
    (/tmp/dsv4_spec_trace_pid<PID>.jsonl), and
  * optionally a plain-greedy ground-truth capture from
    spec_degen_capture.py (run on an MTP-OFF cluster).

What it does
------------
1. Loads the spec trace, flattens each cycle's ``committed`` token ids
   (rank 0) into a single decode stream per uid.
2. Detokenizes the stream (transformers tokenizer for the model).
3. Flags the FIRST cycle where:
     (a) a special token (BOS/EOS/<|begin_of_sentence|>) is committed, or
     (b) a short-period repetition loop begins (period<=max_period,
         repeats>=min_repeats), or
     (c) [if ground-truth token_ids provided] the committed stream
         diverges from the greedy stream.
   and prints that cycle's full record: n_accepted, draft vs
   target_argmax, bonus, verify_input, and ALL cache offsets.

The first-divergence cycle's offsets + acceptance pattern tell us which
subsystem corrupted state:
  * n_accepted high right before collapse + draft==target on a special
    token  -> verify forward itself produced the bad logits (RoPE/mask
    offset on the system-prefix length) — root-cause is verify, not accept.
  * offsets stepping unevenly across cache labels (KV vs pooling/indexer)
    -> min-strategy rollback drift (dsv4_mtp.py:1498-1501).
  * pre_norm/bonus mismatch with the committed prefix -> handoff bug.

Usage:
  python3 bench/spec_degen_diff.py \\
    --trace /tmp/dsv4_spec_trace_pid12345.jsonl \\
    --model mlx-community/DeepSeek-V4-Flash-8bit \\
    [--ground-truth ~/spec_degen_groundtruth.json] \\
    [--max-period 8 --min-repeats 6]
"""
from __future__ import annotations

import argparse
import json
import sys
from typing import Any, Optional


def _load_jsonl(path: str) -> tuple[Optional[dict], list[dict]]:
    header: Optional[dict] = None
    recs: list[dict] = []
    with open(path) as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            try:
                rec = json.loads(line)
            except json.JSONDecodeError:
                continue
            if rec.get("type") == "header":
                header = rec
            else:
                recs.append(rec)
    return header, recs


def _flatten_streams(recs: list[dict]) -> dict[int, list[tuple[int, int]]]:
    """Return {uid -> [(cycle, token_id), ...]} in commit order."""
    streams: dict[int, list[tuple[int, int]]] = {}
    for rec in recs:
        uids = rec.get("uids") or []
        committed = rec.get("committed") or []
        cyc = rec.get("cycle", -1)
        for n, uid in enumerate(uids):
            if n >= len(committed):
                continue
            streams.setdefault(uid, [])
            for tid in committed[n]:
                streams[uid].append((cyc, int(tid)))
    return streams


def _find_loop_start(ids: list[int], max_period: int,
                     min_repeats: int) -> Optional[int]:
    """Index where a period<=max_period cycle repeated >=min_repeats begins."""
    n = len(ids)
    for i in range(n):
        for period in range(1, max_period + 1):
            if i + period * min_repeats > n:
                continue
            block = ids[i:i + period]
            ok = True
            for r in range(1, min_repeats):
                if ids[i + r * period:i + (r + 1) * period] != block:
                    ok = False
                    break
            if ok:
                return i
    return None


def _load_tokenizer(model_id: str) -> Any:
    try:
        # Reuse the bench helper (handles snapshot fallback).
        from exo_bench import load_tokenizer_for_bench  # type: ignore
        return load_tokenizer_for_bench(model_id)
    except Exception:
        try:
            from transformers import AutoTokenizer
            return AutoTokenizer.from_pretrained(model_id,
                                                 trust_remote_code=True)
        except Exception as e:  # noqa: BLE001
            print(f"WARN: no tokenizer ({e}); ids only", file=sys.stderr)
            return None


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--trace", required=True,
                    help="dsv4_spec_trace_pid*.jsonl")
    ap.add_argument("--model", default="mlx-community/DeepSeek-V4-Flash-8bit")
    ap.add_argument("--ground-truth", default=None,
                    help="spec_degen_capture.py JSON (optional)")
    ap.add_argument("--max-period", type=int, default=8)
    ap.add_argument("--min-repeats", type=int, default=6)
    args = ap.parse_args()

    header, recs = _load_jsonl(args.trace)
    if not recs:
        print("no cycle records in trace", file=sys.stderr)
        return 1
    print(f"trace: {len(recs)} cycles"
          + (f"  env={header['env']}" if header else ""))

    tok = _load_tokenizer(args.model)
    streams = _flatten_streams(recs)
    by_cycle = {r.get("cycle"): r for r in recs}

    # Identify special token ids for leak detection.
    special_ids: set[int] = set()
    if tok is not None:
        for attr in ("bos_token_id", "eos_token_id", "pad_token_id"):
            v = getattr(tok, attr, None)
            if isinstance(v, int):
                special_ids.add(v)
        extra = getattr(tok, "all_special_ids", None)
        if isinstance(extra, list):
            special_ids.update(int(x) for x in extra)

    rc = 0
    for uid, seq in streams.items():
        ids = [t for (_c, t) in seq]
        cyc_of = [c for (c, _t) in seq]
        print(f"\n===== uid {uid}: {len(ids)} committed tokens =====")
        if tok is not None:
            try:
                text = tok.decode(ids)
                print(f"decoded[:160]: {text[:160]!r}")
            except Exception as e:  # noqa: BLE001
                print(f"(decode failed: {e})")

        culprit_idx: Optional[int] = None
        reason = ""

        # (a) first special-token leak
        for i, t in enumerate(ids):
            if t in special_ids:
                culprit_idx, reason = i, f"special token id={t}"
                break

        # (b) loop start (only if earlier than any leak)
        loop_i = _find_loop_start(ids, args.max_period, args.min_repeats)
        if loop_i is not None and (culprit_idx is None or loop_i < culprit_idx):
            culprit_idx = loop_i
            blk = ids[loop_i:loop_i + args.max_period]
            reason = f"repetition loop starts (block head ids={blk[:4]}...)"

        # (c) divergence from ground truth, if its token_ids are present
        if args.ground_truth:
            try:
                with open(args.ground_truth) as f:
                    gt = json.load(f)
                # match by nothing reliable here (labels differ from uids);
                # we just report GT availability. Token-id GT requires the
                # server to have returned logprobs ids (often null) — when
                # present, compare positionally.
                gtids = None
                for r in gt.get("results", []):
                    tids = r.get("token_ids")
                    if tids and all(isinstance(x, int) for x in tids):
                        gtids = tids
                        break
                if gtids:
                    for i, t in enumerate(ids):
                        if i >= len(gtids):
                            break
                        if t != gtids[i]:
                            if culprit_idx is None or i < culprit_idx:
                                culprit_idx = i
                                reason = (f"diverges from greedy at pos {i}: "
                                          f"spec={t} greedy={gtids[i]}")
                            break
                else:
                    print("(ground-truth has no token_ids; "
                          "using leak/loop detection only)")
            except Exception as e:  # noqa: BLE001
                print(f"(ground-truth load failed: {e})")

        if culprit_idx is None:
            print("CLEAN: no leak / loop / divergence detected")
            continue

        rc = 2
        bad_cycle = cyc_of[culprit_idx]
        print(f"\n*** FIRST PROBLEM at committed pos {culprit_idx} "
              f"(cycle {bad_cycle}): {reason}")
        # Show the suspect cycle and the one before it.
        for c in (bad_cycle - 1, bad_cycle):
            r = by_cycle.get(c)
            if not r:
                continue
            print(f"\n--- cycle {c} ---")
            print(f"  n_accepted   : {r.get('n_accepted')}")
            print(f"  draft        : {r.get('draft')}")
            print(f"  target_argmax: {r.get('target_argmax')}")
            print(f"  committed    : {r.get('committed')}")
            print(f"  bonus        : {r.get('bonus')}")
            print(f"  verify_input : {r.get('verify_input')}")
            print(f"  offsets      : {r.get('offsets')}")

    if rc == 0:
        print("\nALL STREAMS CLEAN.")
    return rc


if __name__ == "__main__":
    raise SystemExit(main())
