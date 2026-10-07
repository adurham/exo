#!/usr/bin/env python3
"""Phase-19 AGENTIC replay measurement (variant of phase19_round_measure.py).

NO RELAUNCH.  Public API only.  Read-only state.db access.

Why: the campaign measured 28.08 t/s @100K on synthetic filler, but REAL
Hermes agentic turns the same boot era ran ~16-19 t/s.  This variant replays
REAL agentic content reconstructed from state.db (session 20261007_092009_9a2ed7)
instead of filler, and compares against the original benign 'count' task run
in the same process (same prefix-cache state), so content is the only variable.

Prompt construction (agentic arm):
  preamble (the session's real system prompt) + every session message rendered
  in order (content + reasoning_content + tool_calls + api_content), flattened
  into ONE user message exactly like the synthetic harness does, so the request
  SHAPE is identical and only the token CONTENT differs.  A unique salt is
  prepended (as in the synthetic harness) so rep0 is a genuine cold prefill and
  later reps are prefix-cache hits.

Stats frame: read the ': generation_stats' SSE COMMENT frame (not a data: line)
for mtp_cycles_cumulative / mtp_accepted_drafts_cumulative.  gamma is pinned at 3
on the dsv41 path, so rounds = d_cycles/3 and mean_accepted = 3*d_accepted/d_cycles.

Usage:
  python3 phase19_agentic_measure.py --arm agentic --reps 4 --max-tokens 1000 --out ...
  python3 phase19_agentic_measure.py --arm benign  --reps 4 --depth 100000 --task count --out ...
"""
from __future__ import annotations

import argparse
import contextlib
import json
import os
import random
import sqlite3
import statistics
import sys
import time
import urllib.request

API = os.environ.get("PHASE19_API", "http://macstudio-m4-1.tail19c543.ts.net:52415/v1/chat/completions")
MODEL = "dealignai/DeepSeek-V4.1-Flash-UNCENSORED-EXL3-2.9bpw"
STATE_DB = "file:/Users/adam.durham/.hermes/state.db?mode=ro"
SESSION_ID = "20261007_092009_9a2ed7"

# real session system-prompt hash (see state.db system_prompts)
SYS_HASH = "13466837d495ef2fda6d113fbc7656ab13431a876c7efe9e40a2d706a99a708a"

# instruction appended as the final user turn of the agentic arm: forces a long
# reasoning + tool-JSON-shaped answer rather than a one-line reply.
AGENTIC_INSTRUCTION = (
    "\n\n=== CONTINUATION REQUEST ===\n"
    "You have just been resumed. Do NOT run anything yet.\n"
    "First, inside your reasoning, audit the investigation transcript above: list "
    "every concrete finding with the command/evidence that produced it, then identify "
    "the single largest unexplained gap in the evidence, and work out the exact next "
    "shell command that would close it and why that command rather than any other.\n"
    "Then answer ONLY with a JSON object of the form "
    '{"findings":[{"claim":"...","evidence":"...","command":"..."}],'
    '"biggest_gap":"...","next_command":"...","why_this_command":"..."}.\n'
    "Be exhaustive and cite specific numbers from the transcript."
)

_SENTENCES = [
    "The observer pattern defines a one-to-many dependency so that when one object changes state all its dependents are notified automatically.",
    "A binary search tree keeps the key of every internal node greater than all keys in its left subtree and less than those in its right subtree.",
    "Garbage collection reclaims memory that was allocated by a program but is no longer reachable from any live root reference.",
    "MapReduce expresses a computation as a map phase that emits intermediate key value pairs and a reduce phase that aggregates them.",
    "The CAP theorem states a distributed data store can simultaneously guarantee only two of consistency availability and partition tolerance.",
    "Vector clocks track causal ordering between events in a distributed system without relying on a global wall clock.",
    "A bloom filter answers set membership queries with no false negatives and a tunable rate of false positives.",
    "Consistent hashing places both keys and nodes on a ring so that adding or removing a node moves only a small fraction of keys.",
    "The two-phase commit protocol coordinates an atomic transaction across participants at the cost of blocking on coordinator failure.",
    "A skip list provides expected logarithmic search time using a hierarchy of linked lists with geometrically decreasing density.",
    "Content addressable storage names an object by a hash of its bytes so identical content is stored only once.",
    "A write ahead log records intent before mutating state so a crash can be replayed to a consistent point.",
]
TASKS = {
    "count": "\n\nTask: List the integers from 1 to 450 in order, each on its own line, and nothing else.",
    "done": "\n\nTask: Reply with ONLY the single word DONE and nothing else.",
}


# ---------------------------------------------------------------- prompt build
def load_session():
    con = sqlite3.connect(STATE_DB, uri=True)
    con.row_factory = sqlite3.Row
    cur = con.cursor()
    cur.execute("SELECT prompt FROM system_prompts WHERE hash=?", (SYS_HASH,))
    row = cur.fetchone()
    preamble = row[0] if row else ""
    cur.execute(
        "SELECT id,role,content,reasoning_content,tool_calls,tool_name,timestamp "
        "FROM messages WHERE session_id=? ORDER BY timestamp,id", (SESSION_ID,))
    msgs = [dict(r) for r in cur.fetchall()]
    con.close()
    return preamble, msgs


#: Phase-4b payload cap (characters) applied to each tool result's `content`
#: BEFORE it is rendered into the agentic prompt.  This is a CLIENT-side
#: truncation emulating Hermes `tool_output.max_bytes` (50000 -> 20000 chars;
#: 20000 is the proposed cap, matching the raw-bytes cap on ASCII payloads).
TOOL_CAP_CHARS: int | None = None
#: Counter: how many tool results the cap actually truncated this run.
_CAP_STATS = {"truncated": 0, "removed_chars": 0, "n_tool": 0}


def render_msg(m: dict) -> str:
    role, c = m["role"], (m["content"] or "")
    rc = m["reasoning_content"] or ""
    tc = m["tool_calls"] or ""
    if role == "tool":
        _CAP_STATS["n_tool"] += 1
        if TOOL_CAP_CHARS is not None and len(c) > TOOL_CAP_CHARS:
            _CAP_STATS["truncated"] += 1
            _CAP_STATS["removed_chars"] += len(c) - TOOL_CAP_CHARS
            c = c[:TOOL_CAP_CHARS] + "\n…[truncated]"
        return f"\n\n[tool_result name={m['tool_name']}]\n{c}"
    if role == "assistant":
        parts = []
        if rc:
            parts.append("<reasoning>\n" + rc + "\n</reasoning>")
        if c:
            parts.append(c)
        if tc:
            parts.append("<tool_calls>\n" + tc + "\n</tool_calls>")
        body = "\n".join(parts)
        return f"\n\n[assistant]\n{body}"
    if role == "tool":
        return f"\n\n[tool_result name={m['tool_name']}]\n{c}"
    if role == "system":
        return f"\n\n[system]\n{c}"
    return f"\n\n[{role}]\n{c}"


def build_agentic_prompt(salt: str, char_budget: int | None):
    """preamble + all real session messages, truncated to char_budget, + instruction."""
    preamble, msgs = load_session()
    head = f"[SESSION-SALT {salt}]\n"
    body = head + preamble
    used = 0
    for m in msgs:
        piece = render_msg(m)
        if char_budget is not None and len(body) + len(piece) > char_budget:
            break
        body += piece
        used += 1
    body += AGENTIC_INSTRUCTION
    return body, {"n_messages_total": len(msgs), "n_messages_used": used,
                  "preamble_chars": len(preamble), "prompt_chars": len(body),
                  "pred_tokens_3p48": int(len(body) / 3.479)}


def build_benign_prompt(target_tokens: int, salt: str, task: str) -> str:
    rng = random.Random(20261007)
    target_chars = int(target_tokens * 5.59)
    head = f"[SESSION-SALT {salt}]\n"
    parts = [head]
    n = len(head)
    while n < target_chars:
        s = _SENTENCES[rng.randrange(len(_SENTENCES))]
        parts.append(s + " ")
        n += len(s) + 1
    parts.append(TASKS[task])
    return "".join(parts)


# ---------------------------------------------------------------- measurement
def stream_once(prompt: str, max_tokens: int, effort: str | None,
                spec_gamma: int | None = None) -> dict:
    send_epoch = time.time()
    body = {"model": MODEL, "messages": [{"role": "user", "content": prompt}],
            "max_tokens": max_tokens, "temperature": 0, "stream": True}
    if effort:
        body["reasoning_effort"] = effort
    # deploy/next14-gamma: per-request speculative draft length, clamped to
    # [1,6] server-side.  Absent (None) -> engine default gamma 3 and a
    # byte-identical body, so existing callers are unchanged.
    if spec_gamma is not None:
        body["spec_gamma"] = spec_gamma
    req = urllib.request.Request(API, data=json.dumps(body).encode(),
                                 headers={"Content-Type": "application/json"})
    t0 = time.perf_counter()
    ttft = first = last = None
    fr = fc = lc = None                 # first reasoning, first content, last content
    rchars = cchars = 0
    usage = None
    stats = None
    finish = None
    created = None
    with urllib.request.urlopen(req, timeout=3600) as resp:
        for raw in resp:
            line = raw.decode("utf-8", "replace").rstrip("\n")
            if line.startswith(": generation_stats"):
                with contextlib.suppress(Exception):
                    stats = json.loads(line.split(" ", 2)[2])
                continue
            if not line.startswith("data:"):
                continue
            payload = line[5:].strip()
            if payload == "[DONE]":
                break
            try:
                obj = json.loads(payload)
            except Exception:
                continue
            now = time.perf_counter()
            if created is None:
                created = obj.get("created")
            if usage is None and obj.get("usage"):
                usage = obj["usage"]
            for ch in obj.get("choices", []):
                if ch.get("finish_reason"):
                    finish = ch["finish_reason"]
                d = ch.get("delta") or {}
                rtxt = d.get("reasoning_content") or d.get("reasoning") or ""
                ctxt = d.get("content") or ""
                if rtxt:
                    if fr is None:
                        fr = now
                    rchars += len(rtxt)
                if ctxt:
                    if fc is None:
                        fc = now
                    cchars += len(ctxt)
                    lc = now
                if rtxt or ctxt:
                    if ttft is None:
                        ttft, first = now - t0, now
                    last = now                 # last token of ANY kind
    wall = time.perf_counter() - t0
    decode_s = (last - first) if (first is not None and last is not None and last > first) else None
    return {"wall_s": round(wall, 3),
            "request_created_epoch": created,
            "request_send_epoch": send_epoch,
            "ttft_s": round(ttft, 4) if ttft else None,
            "decode_s": round(decode_s, 4) if decode_s else None,
            "reasoning_chars": rchars, "content_chars": cchars,
            "reason_first_s": round(fr - t0, 4) if fr else None,
            "content_first_s": round(fc - t0, 4) if fc else None,
            "content_last_s": round(lc - t0, 4) if lc else None,
            "finish_reason": finish, "usage": usage, "stats": stats}


def derive(rec, prev_cyc, prev_acc, gamma):
    st = rec.get("stats") or {}
    u = rec.get("usage") or {}
    cyc = st.get("mtp_cycles_cumulative")
    acc = st.get("mtp_accepted_drafts_cumulative")
    gen = u.get("completion_tokens")
    det = u.get("completion_tokens_details") or {}
    out = {"prompt_tokens": u.get("prompt_tokens"), "completion_tokens": gen,
           "reasoning_tokens": det.get("reasoning_tokens"),
           "cached_tokens": (u.get("prompt_tokens_details") or {}).get("cached_tokens"),
           "cycles_cum": cyc, "accepted_cum": acc,
           # raw process-cumulative per-position acceptance histogram ([int]*7,
           # index k = rounds that accepted exactly k drafts); None if absent.
           "hist_cum": st.get("mtp_accepted_histogram_cumulative"),
           "prefix_hit": st.get("prefix_cache_hit"),
           "prompt_tps": round(st.get("prompt_tps") or 0, 1),
           "decode_tps": None, "rounds": None, "mean_accepted": None,
           "ms_per_round": None, "gamma_implied": None}
    if rec.get("decode_s") and gen and gen > 1 and rec["decode_s"] > 0:
        out["decode_tps"] = round((gen - 1) / rec["decode_s"], 3)
    if cyc is not None and prev_cyc is not None and cyc > prev_cyc:
        d_cyc = cyc - prev_cyc
        d_acc = (acc - (prev_acc or 0)) if acc is not None else None
        rounds = d_cyc / gamma
        out["rounds"] = round(rounds, 1)
        if d_acc is not None:
            out["mean_accepted"] = round(gamma * d_acc / d_cyc, 4)
            mc = 1.0 + d_acc / d_cyc
            if gen and mc > 0:
                out["rounds_est"] = round(gen / mc, 1)
                if out["rounds_est"]:
                    out["gamma_implied"] = round(d_cyc / out["rounds_est"], 3)
        if rec.get("decode_s") and rounds:
            out["ms_per_round"] = round(rec["decode_s"] * 1000.0 / rounds, 2)
    return out


def main():
    global TOOL_CAP_CHARS
    ap = argparse.ArgumentParser()
    ap.add_argument("--arm", required=True, choices=["agentic", "benign"])
    ap.add_argument("--reps", type=int, default=4)
    ap.add_argument("--max-tokens", type=int, default=1000)
    ap.add_argument("--depth", type=int, default=100000)
    ap.add_argument("--char-budget", type=int, default=None)
    ap.add_argument("--cap-chars", type=int, default=None,
                    help="truncate each tool result to this many chars "
                         "(emulates Hermes tool_output.max_bytes)")
    ap.add_argument("--gamma", type=int, default=3)
    ap.add_argument("--label", default="run")
    ap.add_argument("--effort", default=None)
    ap.add_argument("--task", default="count", choices=list(TASKS))
    ap.add_argument("--out", default=None)
    a = ap.parse_args()

    TOOL_CAP_CHARS = a.cap_chars
    salt = os.urandom(4).hex()
    if a.arm == "agentic":
        prompt, meta = build_agentic_prompt(salt, a.char_budget)
        meta["tool_cap_chars"] = a.cap_chars
        meta["cap_stats"] = dict(_CAP_STATS)
        print(f"[{a.label}] AGENTIC salt={salt} chars={len(prompt)} "
              f"n_msgs_used={meta['n_messages_used']}/{meta['n_messages_total']} "
              f"pred_tokens~{meta['pred_tokens_3p48']} "
              f"cap_chars={a.cap_chars} cap_stats={_CAP_STATS}", flush=True)
    else:
        prompt = build_benign_prompt(a.depth, salt, a.task)
        meta = {"prompt_chars": len(prompt), "depth_target": a.depth}
        print(f"[{a.label}] BENIGN depth={a.depth} salt={salt} chars={len(prompt)}", flush=True)

    recs = []
    prev_cyc = prev_acc = None
    for i in range(a.reps):
        r = stream_once(prompt, a.max_tokens, a.effort)
        r.update(derive(r, prev_cyc, prev_acc, a.gamma))
        r["rep"] = i
        recs.append(r)
        if r.get("cycles_cum") is not None:
            prev_cyc, prev_acc = r["cycles_cum"], r["accepted_cum"]
        print(json.dumps({k: r.get(k) for k in (
            "rep", "prompt_tokens", "completion_tokens", "reasoning_tokens",
            "content_chars", "reasoning_chars", "ttft_s", "decode_s", "decode_tps",
            "rounds", "mean_accepted", "ms_per_round", "gamma_implied",
            "prefix_hit", "finish_reason")}), flush=True)
        time.sleep(2)

    # timed reps exclude rep0 (the cold prefill)
    timed = recs[1:] if len(recs) > 1 else recs
    dec = [r["decode_tps"] for r in timed if r.get("decode_tps")]
    mar = [r["mean_accepted"] for r in timed if r.get("mean_accepted") is not None]
    msr = [r["ms_per_round"] for r in timed if r.get("ms_per_round")]
    gimp = [r["gamma_implied"] for r in timed if r.get("gamma_implied")]
    rtk = [r["reasoning_tokens"] for r in timed if r.get("reasoning_tokens") is not None]
    summary = {"label": a.label, "arm": a.arm, "reps": len(recs),
               "meta": meta,
               "prompt_tokens": recs[0].get("prompt_tokens"),
               "timed_decode_tps_median": round(statistics.median(dec), 3) if dec else None,
               "timed_decode_tps_all": dec,
               "timed_mean_accepted_median": round(statistics.median(mar), 4) if mar else None,
               "timed_ms_per_round_median": round(statistics.median(msr), 2) if msr else None,
               "timed_gamma_implied_median": round(statistics.median(gimp), 3) if gimp else None,
               "timed_reasoning_tokens_median": round(statistics.median(rtk), 1) if rtk else None}
    print("SUMMARY", json.dumps(summary), flush=True)
    if a.out:
        with open(a.out, "w") as f:
            json.dump({"summary": summary, "reps": recs}, f, indent=1)
    return 0


if __name__ == "__main__":
    sys.exit(main())
