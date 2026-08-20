#!/usr/bin/env python3
"""Controlled A/B harness for EXO_PREFILL_CHUNK_OVERLAP correctness.

DESIGN DOC: docs/prefill-chunk-overlap-ab-harness-design-2026-08-20.md
Read it before running. It contains the pre-flight checklist, the
statistical decision rule, and the honest limitations.

WHAT THIS DOES
--------------
Generates N paired long-context prompts, each carrying MULTIPLE distinct
secret strings ("needles") placed at deliberately chosen token offsets that
straddle KV-cache chunk boundaries (multiples of EXO_PREFILL_STEP_SIZE),
plus interior CONTROL needles far from any boundary. Runs every prompt under
flag=0 and flag=1, plus a flag=0 repeat arm to measure the cluster's own
nondeterminism floor. Captures per-token logprobs for a logit-level
comparison. Then applies a pre-registered three-gate decision rule.

STANDARD SINGLE-SESSION RUN (one approved cluster session, two relaunches)
--------------------------------------------------------------------------
  # 0. offline, no cluster needed
  python3 scripts/prefill_overlap_ab.py --generate-only

  # 1. relaunch cluster, flag OFF, step size 2048, MTP/spec OFF
  DSV4_KV_CACHE_BITS=0 EXO_DSV4_MTP=0 EXO_SPECULATIVE=0 \
    EXO_PREFILL_STEP_SIZE=2048 ./start_cluster.sh
  python3 scripts/prefill_overlap_ab.py --run --arm flag0
  python3 scripts/prefill_overlap_ab.py --run --arm flag0_repeat

  # 2. relaunch cluster, flag ON (BOTH RANKS -- verify!)
  DSV4_KV_CACHE_BITS=0 EXO_DSV4_MTP=0 EXO_SPECULATIVE=0 \
    EXO_PREFILL_STEP_SIZE=2048 EXO_PREFILL_CHUNK_OVERLAP=1 ./start_cluster.sh
  python3 scripts/prefill_overlap_ab.py --preflight     # asserts flag on both ranks
  python3 scripts/prefill_overlap_ab.py --run --arm flag1

  # 3. revert cluster to standing baseline, then analyze (offline)
  python3 scripts/prefill_overlap_ab.py --analyze-only

Results land in results/prefill_overlap_ab/*.jsonl and can be analyzed
offline any time.
"""

from __future__ import annotations

import argparse
import json
import math
import os
import random
import re
import statistics
import sys
import time
import urllib.request
from dataclasses import asdict, dataclass
from pathlib import Path

# ---------------------------------------------------------------- config

API = os.environ.get("EXO_AB_API", "http://192.168.86.201:52415/v1/chat/completions")
MODEL = os.environ.get("EXO_AB_MODEL", "mlx-community/DeepSeek-V4-Flash")

#: Must match EXO_PREFILL_STEP_SIZE on the cluster. prefill_batched() uses
#: this UNDIVIDED (generate.py ~L1424). The serial prefill() path divides by
#: min(4, group.size()) (~L496) -- the overlap flag is not in that path.
CHUNK = int(os.environ.get("EXO_AB_CHUNK", "2048"))

#: Needle text starts/ends this many tokens from the boundary.
DELTA = 24

#: Generously high so no trial is truncation-ambiguous. DSv4 is a thinking
#: model: reasoning tokens come out of this same budget. The live 2026-08-20
#: test's trial 2 was ruined by max_tokens=50.
MAX_TOKENS = 4096

TOP_LOGPROBS = 20
TEMPERATURE = 0.0
REQUEST_TIMEOUT = 2400

OUT_DIR = Path(__file__).resolve().parent.parent / "results" / "prefill_overlap_ab"
PROMPTS_FILE = OUT_DIR / "prompts.json"

#: (n_prompts, approx_context_tokens) tiers. The 40K tier reproduces the
#: regime of the original anomaly (38K).
TIERS = [(4, 20_000), (4, 40_000), (4, 80_000)]

ARMS = ("flag0", "flag1", "flag0_repeat")

WORDS = ["system", "config", "module", "handler", "request", "buffer", "thread", "socket", "packet", "kernel", "daemon", "cache", "index", "pointer", "segment", "registry", "cluster", "node", "payload", "session", "token", "vector", "matrix", "gradient", "tensor", "pipeline", "queue", "scheduler", "allocator", "mutex", "semaphore", "latency", "throughput", "bandwidth", "protocol", "adapter", "interface", "schema", "migration", "rollback", "checkpoint", "ledger", "shard", "replica", "quorum", "lease"]

CODEWORDS = [
    "ZEPHYR", "OBSIDIAN", "MARLIN", "QUARTZ", "NIMBUS", "FALCON", "BASALT",
    "LANTERN", "COBALT", "TAMARIN", "VESPER", "GRANITE", "HALCYON", "PELICAN",
    "IRIDIUM", "SABLE", "THISTLE", "MERIDIAN", "ONYX", "KESTREL",
]

# --------------------------------------------------------------- tokenizer


def get_tokenizer():
    """Local tokenizer for exact offset placement. No cluster needed."""
    try:
        from transformers import AutoTokenizer  # type: ignore
    except ImportError:
        sys.exit("need `transformers` locally for token-offset placement")
    path = os.environ.get("EXO_AB_TOKENIZER", MODEL)
    return AutoTokenizer.from_pretrained(path, trust_remote_code=True)


def ntok(tok, text: str) -> int:
    return len(tok.encode(text, add_special_tokens=False))


# ------------------------------------------------------------ prompt build


@dataclass
class Needle:
    slot: int
    secret: str
    #: "before" | "after" | "straddle" | "interior"
    kind: str
    #: which chunk boundary (k) this needle is anchored to; None for interior
    boundary_k: int | None
    #: intended token offset of the needle's first token, pre-calibration
    target_offset: int
    #: measured token offset after assembly
    actual_offset: int = -1


@dataclass
class PromptSpec:
    prompt_id: str
    tier_tokens: int
    seed: int
    needles: list[Needle]
    prompt: str
    approx_tokens: int
    #: token offset of the needle-free prefix nonce, guarantees cold cache
    nonce: str
    template_overhead: int = 0

    def to_json(self) -> dict:
        d = asdict(self)
        return d


def make_secret(rng: random.Random, used: set[str]) -> str:
    """Fixed-format, high-entropy, partial-credit-scorable.

    NDL-<slot>-<WORD>-<4 digits>-<WORD>. The observed live corruption mutated
    only PART of the string, so scoring must be field-aware.
    """
    while True:
        s = (
            f"{rng.choice(CODEWORDS)}-{rng.randint(1000, 9999)}-"
            f"{rng.choice(CODEWORDS)}"
        )
        if s not in used:
            used.add(s)
            return s


def filler_line(rng: random.Random, idx: int) -> str:
    return f"block {idx:06d}: " + " ".join(rng.choice(WORDS) for _ in range(18))


def needle_sentence(slot: int, secret: str) -> str:
    return (
        f"IMPORTANT RECORD {slot}: the authorization code for slot {slot} "
        f"is {secret}. Record it exactly as written."
    )


def plan_offsets(tier_tokens: int, rng: random.Random) -> list[tuple[str, int | None, int]]:
    """Choose (kind, boundary_k, target_offset) for 8 needles.

    6 boundary needles spread across the available chunk boundaries
    (2 "before", 2 "after", 2 "straddle") + 2 interior CONTROL needles.
    The control needles are what make a within-arm, within-prompt contrast
    possible -- see Gate B in the design doc.
    """
    n_chunks = max(2, tier_tokens // CHUNK)
    # skip k=1 (too close to the template/prefix calibration noise) and the
    # final boundary (may not be reached if the prompt lands short)
    candidates = list(range(2, n_chunks))
    if len(candidates) < 6:
        candidates = candidates * ((6 // max(1, len(candidates))) + 1)
    ks = sorted(rng.sample(candidates, 6)) if len(set(candidates)) >= 6 else candidates[:6]

    plan: list[tuple[str, int | None, int]] = []
    kinds = ["straddle", "before", "after", "straddle", "before", "after"]
    for kind, k in zip(kinds, ks, strict=False):
        b = k * CHUNK
        if kind == "before":
            # needle text ENDS just before the boundary
            plan.append(("before", k, b - DELTA - 40))
        elif kind == "after":
            plan.append(("after", k, b + DELTA))
        else:
            # needle text SPANS the boundary: the only placement whose tokens
            # live in BOTH the producing and the consuming chunk.
            plan.append(("straddle", k, b - 20))

    # 2 interior controls: mid-chunk, maximally far from any boundary
    used_k = set(ks)
    interior_ks = [k for k in range(2, n_chunks) if k not in used_k] or list(range(2, n_chunks))
    for k in rng.sample(interior_ks, min(2, len(interior_ks))) or [2, 3]:
        plan.append(("interior", None, k * CHUNK + CHUNK // 2))
    while len(plan) < 8:
        plan.append(("interior", None, CHUNK * 2 + CHUNK // 2))

    plan.sort(key=lambda t: t[2])
    return plan


QUESTION_TMPL = """

--- END OF RECORDS ---

Above are {n} IMPORTANT RECORD entries, numbered by slot. Report the
authorization code for every slot. Think as long as you need to, then end
your reply with exactly {n} lines in this format and nothing after them:

{lines}

Each value must be reproduced EXACTLY as it appeared, character for
character. If you cannot find a slot's code, write UNKNOWN for that slot.
"""


def build_prompt(tok, tier_tokens: int, seed: int, template_overhead: int) -> PromptSpec:
    rng = random.Random(seed)
    used: set[str] = set()
    plan = plan_offsets(tier_tokens, rng)

    needles = [
        Needle(slot=i + 1, secret=make_secret(rng, used), kind=kind,
               boundary_k=k, target_offset=off - template_overhead)
        for i, (kind, k, off) in enumerate(plan)
    ]

    parts: list[str] = []
    cur = 0
    nonce = f"SESSION-NONCE-{rng.getrandbits(64):016x}"
    head = f"Reference log dump. {nonce}\n"
    parts.append(head)
    cur += ntok(tok, head)

    blk = 0
    for nd in needles:
        while cur < nd.target_offset:
            line = filler_line(rng, blk) + "\n"
            t = ntok(tok, line)
            if cur + t > nd.target_offset:
                break
            parts.append(line)
            cur += t
            blk += 1
        nd.actual_offset = cur
        s = needle_sentence(nd.slot, nd.secret) + "\n"
        parts.append(s)
        cur += ntok(tok, s)

    while cur < tier_tokens:
        line = filler_line(rng, blk) + "\n"
        t = ntok(tok, line)
        if cur + t > tier_tokens:
            break
        parts.append(line)
        cur += t
        blk += 1

    lines = "\n".join(f"SLOT{n.slot}=<code>" for n in needles)
    parts.append(QUESTION_TMPL.format(n=len(needles), lines=lines))
    prompt = "".join(parts)

    return PromptSpec(
        prompt_id=f"t{tier_tokens // 1000}k-s{seed}",
        tier_tokens=tier_tokens,
        seed=seed,
        needles=needles,
        prompt=prompt,
        approx_tokens=ntok(tok, prompt),
        nonce=nonce,
        template_overhead=template_overhead,
    )


def generate_prompts(template_overhead: int = 0) -> list[PromptSpec]:
    tok = get_tokenizer()
    specs: list[PromptSpec] = []
    seed = 20260820
    for n, tier in TIERS:
        for i in range(n):
            specs.append(build_prompt(tok, tier, seed + i + tier, template_overhead))
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    PROMPTS_FILE.write_text(json.dumps([s.to_json() for s in specs], indent=1))
    print(f"wrote {len(specs)} prompts -> {PROMPTS_FILE}")
    for s in specs:
        bd = [f"{n.kind}@k={n.boundary_k}:{n.actual_offset}" for n in s.needles]
        print(f"  {s.prompt_id}: ~{s.approx_tokens} tok, needles {bd}")
    return specs


def load_prompts() -> list[PromptSpec]:
    raw = json.loads(PROMPTS_FILE.read_text())
    out = []
    for d in raw:
        nds = [Needle(**n) for n in d.pop("needles")]
        out.append(PromptSpec(needles=nds, **d))
    return out


# ------------------------------------------------------------------ client


def post(body: dict, timeout: int = REQUEST_TIMEOUT) -> dict:
    req = urllib.request.Request(
        API, data=json.dumps(body).encode(),
        headers={"Content-Type": "application/json"},
    )
    with urllib.request.urlopen(req, timeout=timeout) as r:
        return json.load(r)


def chat(prompt: str, max_tokens: int = MAX_TOKENS, logprobs: bool = True) -> dict:
    return post({
        "model": MODEL,
        "messages": [{"role": "user", "content": prompt}],
        "max_tokens": max_tokens,
        "temperature": TEMPERATURE,
        "stream": False,
        "logprobs": logprobs,
        "top_logprobs": TOP_LOGPROBS,
    })


def preflight() -> bool:
    """Fail loudly BEFORE burning an approved session on an invalid run."""
    ok = True
    print("== preflight ==")
    try:
        d = chat("Reply with exactly: PREFLIGHT-OK", max_tokens=64)
    except Exception as e:  # noqa: BLE001
        print(f"FAIL: cluster unreachable: {e}")
        return False
    ch = d["choices"][0]
    print(f"  reachable, finish_reason={ch.get('finish_reason')}")
    lp = ch.get("logprobs")
    if not lp or not lp.get("content"):
        print("FAIL: logprobs came back empty -- Gate C (logit comparison) is "
              "DEAD. Do not proceed silently; report this.")
        ok = False
    else:
        print(f"  logprobs OK ({len(lp['content'])} positions, "
              f"{len(lp['content'][0].get('top_logprobs') or [])} top-k)")
    print("  MANUAL CHECKS (not automatable from here):")
    print("   [ ] EXO_PREFILL_CHUNK_OVERLAP identical on BOTH ranks")
    print(f"   [ ] EXO_PREFILL_STEP_SIZE == {CHUNK} on BOTH ranks")
    print("   [ ] EXO_DSV4_MTP=0 and EXO_SPECULATIVE=0 (both arms)")
    print("   [ ] no concurrent bench/probe process on the cluster")
    print("   [ ] prefill_batched() path confirmed in runner trace")
    return ok


# -------------------------------------------------------------------- run


def parse_answers(text: str, n: int) -> dict[int, str]:
    out: dict[int, str] = {}
    for i in range(1, n + 1):
        m = re.findall(rf"SLOT{i}\s*=\s*([A-Za-z0-9\-]+)", text)
        if m:
            out[i] = m[-1].strip()
    return out


def score_needle(expected: str, got: str | None) -> dict:
    if got is None:
        return {"status": "MISSING", "exact": False, "fields": 0, "edit": None}
    if got.upper() == expected.upper():
        return {"status": "CORRECT", "exact": True, "fields": 3, "edit": 0}
    ef, gf = expected.upper().split("-"), got.upper().split("-")
    fields = sum(1 for a, b in zip(ef, gf, strict=False) if a == b)
    return {
        "status": "WRONG", "exact": False, "fields": fields,
        "edit": _lev(expected.upper(), got.upper()),
    }


def _lev(a: str, b: str) -> int:
    prev = list(range(len(b) + 1))
    for i, ca in enumerate(a, 1):
        cur = [i]
        for j, cb in enumerate(b, 1):
            cur.append(min(prev[j] + 1, cur[j - 1] + 1, prev[j - 1] + (ca != cb)))
        prev = cur
    return prev[-1]


def run_arm(arm: str, only: str | None = None) -> None:
    assert arm in ARMS, f"arm must be one of {ARMS}"
    specs = load_prompts()
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    out = OUT_DIR / f"{arm}.jsonl"
    done = set()
    if out.exists():
        done = {json.loads(ln)["prompt_id"] for ln in out.read_text().splitlines() if ln.strip()}

    with out.open("a") as fh:
        for s in specs:
            if only and s.prompt_id != only:
                continue
            if s.prompt_id in done:
                print(f"skip {s.prompt_id} (already recorded)")
                continue
            print(f"[{arm}] {s.prompt_id} (~{s.approx_tokens} tok) ...", flush=True)
            t0 = time.time()
            try:
                d = chat(s.prompt)
            except Exception as e:  # noqa: BLE001
                rec = {"arm": arm, "prompt_id": s.prompt_id, "error": repr(e)}
                fh.write(json.dumps(rec) + "\n")
                fh.flush()
                print(f"  ERROR {e}")
                continue
            dt = time.time() - t0
            ch = d["choices"][0]
            msg = ch["message"]
            text = (msg.get("content") or "") + "\n" + (msg.get("reasoning_content") or "")
            usage = d.get("usage", {}) or {}
            cached = (usage.get("prompt_tokens_details") or {}).get("cached_tokens",
                                                                    usage.get("cached_tokens", 0))
            finish = ch.get("finish_reason")

            answers = parse_answers(text, len(s.needles))
            scores = []
            for nd in s.needles:
                sc = score_needle(nd.secret, answers.get(nd.slot))
                # cross-contamination: did another slot's secret show up here?
                sc["cross"] = next(
                    (o.slot for o in s.needles
                     if o.slot != nd.slot and answers.get(nd.slot)
                     and answers[nd.slot].upper() == o.secret.upper()),
                    None,
                )
                sc.update(slot=nd.slot, kind=nd.kind, boundary_k=nd.boundary_k,
                          offset=nd.actual_offset)
                scores.append(sc)

            # INVALID, not WRONG. Truncation must never be scored as a miss.
            invalid = None
            if finish == "length":
                invalid = "truncated_at_max_tokens"
            elif cached:
                invalid = f"warm_cache cached_tokens={cached}"

            lp = ch.get("logprobs") or {}
            lp_content = lp.get("content") or []
            rec = {
                "arm": arm, "prompt_id": s.prompt_id, "tier": s.tier_tokens,
                "elapsed_s": round(dt, 1), "finish_reason": finish,
                "prompt_tokens": usage.get("prompt_tokens"),
                "completion_tokens": usage.get("completion_tokens"),
                "cached_tokens": cached, "invalid": invalid,
                "answers": answers, "scores": scores,
                "logprobs": [
                    {"t": c.get("token"), "lp": c.get("logprob"),
                     "top": [(x.get("token"), x.get("logprob"))
                             for x in (c.get("top_logprobs") or [])]}
                    for c in lp_content
                ],
                "text_tail": text[-2000:],
            }
            fh.write(json.dumps(rec) + "\n")
            fh.flush()
            nc = sum(1 for x in scores if x["status"] == "CORRECT")
            print(f"  {nc}/{len(scores)} correct, finish={finish}, "
                  f"cached={cached}, {dt:.0f}s"
                  + (f"  INVALID: {invalid}" if invalid else ""))


# --------------------------------------------------------------- analysis


def _binom_p_ge(k: int, n: int) -> float:
    """One-sided exact binomial P(X >= k | n, 0.5)."""
    if n == 0:
        return 1.0
    return sum(math.comb(n, i) for i in range(k, n + 1)) / (2 ** n)


def _fisher_one_sided(a: int, b: int, c: int, d: int) -> float:
    """P(>= a errors in group 1) for table [[a,b],[c,d]]."""
    n = a + b + c + d
    r1, c1 = a + b, a + c
    p = 0.0
    for x in range(a, min(r1, c1) + 1):
        p += (math.comb(c1, x) * math.comb(n - c1, r1 - x)) / math.comb(n, r1)
    return min(1.0, p)


def _mannwhitney_p(xs: list[float], ys: list[float]) -> float:
    """Normal-approx two-sided Mann-Whitney U. Adequate at n=12."""
    if not xs or not ys:
        return 1.0
    allv = sorted([(v, 0) for v in xs] + [(v, 1) for v in ys])
    i = 0
    r = [0.0] * len(allv)
    while i < len(allv):
        j = i
        while j + 1 < len(allv) and allv[j + 1][0] == allv[i][0]:
            j += 1
        avg = (i + j) / 2 + 1
        for k in range(i, j + 1):
            r[k] = avg
        i = j + 1
    r1 = sum(r[k] for k in range(len(allv)) if allv[k][1] == 0)
    n1, n2 = len(xs), len(ys)
    u1 = r1 - n1 * (n1 + 1) / 2
    mu = n1 * n2 / 2
    sd = math.sqrt(n1 * n2 * (n1 + n2 + 1) / 12) or 1e-9
    z = abs(u1 - mu) / sd
    return math.erfc(z / math.sqrt(2))


def _load(arm: str) -> dict[str, dict]:
    p = OUT_DIR / f"{arm}.jsonl"
    if not p.exists():
        return {}
    out = {}
    for line in p.read_text().splitlines():
        if line.strip():
            r = json.loads(line)
            if "error" not in r:
                out[r["prompt_id"]] = r
    return out


def _lp_delta(a: dict, b: dict) -> tuple[float, float, float]:
    """(mean |dlogprob|, max |dlogprob|, top1 agreement) over aligned prefix."""
    la, lb = a.get("logprobs") or [], b.get("logprobs") or []
    n = min(len(la), len(lb))
    if n == 0:
        return (float("nan"), float("nan"), float("nan"))
    ds, agree = [], 0
    for i in range(n):
        if la[i]["lp"] is not None and lb[i]["lp"] is not None:
            ds.append(abs(la[i]["lp"] - lb[i]["lp"]))
        if la[i]["t"] == lb[i]["t"]:
            agree += 1
        else:
            break  # after divergence the sequences aren't comparable
    if not ds:
        return (float("nan"), float("nan"), float("nan"))
    return statistics.mean(ds), max(ds), agree / n


def analyze() -> None:
    f0, f1, fr = _load("flag0"), _load("flag1"), _load("flag0_repeat")
    print("=" * 72)
    print("EXO_PREFILL_CHUNK_OVERLAP A/B -- pre-registered analysis")
    print("=" * 72)
    print(f"trials: flag0={len(f0)} flag1={len(f1)} flag0_repeat={len(fr)}")

    inv = [(a, r["prompt_id"], r["invalid"])
           for a, m in (("flag0", f0), ("flag1", f1), ("flag0_repeat", fr))
           for r in m.values() if r.get("invalid")]
    total = len(f0) + len(f1) + len(fr)
    print(f"\nINVALID trials (excluded, NOT scored as errors): {len(inv)}/{total}")
    for a, pid, why in inv:
        print(f"   {a} {pid}: {why}")
    if total and len(inv) / total > 0.10:
        print("  !! >10% invalid -- ABORT and raise MAX_TOKENS / fix cold-cache.")

    ok = lambda m: {k: v for k, v in m.items() if not v.get("invalid")}  # noqa: E731
    f0v, f1v, frv = ok(f0), ok(f1), ok(fr)
    shared = sorted(set(f0v) & set(f1v))

    # ---- Gate A: paired McNemar over NEEDLES
    b = c = 0
    for pid in shared:
        s0 = {x["slot"]: x for x in f0v[pid]["scores"]}
        s1 = {x["slot"]: x for x in f1v[pid]["scores"]}
        for slot in s0:
            a_ok = s0[slot]["status"] == "CORRECT"
            b_ok = s1.get(slot, {}).get("status") == "CORRECT"
            if a_ok and not b_ok:
                b += 1
            elif b_ok and not a_ok:
                c += 1
    npairs = sum(len(f0v[p]["scores"]) for p in shared)
    p_a = _binom_p_ge(b, b + c)
    gate_a = (p_a <= 0.05) and (b - c >= 4)
    print(f"\n[Gate A] paired needle-level McNemar, {npairs} paired needles")
    print(f"  b (ok@flag0, wrong@flag1) = {b}")
    print(f"  c (wrong@flag0, ok@flag1) = {c}")
    print(f"  one-sided exact binomial p = {p_a:.4f}")
    print(f"  -> {'FIRE (real regression)' if gate_a else 'no fire'}"
          + ("" if (b + c) > 2 else "   [b+c<=2: underpowered, not evidence of safety]"))

    # ---- Gate B: boundary vs interior WITHIN flag=1
    be = bt = ie = it = 0
    hist: dict[int, int] = {}
    for _pid, r in f1v.items():
        for x in r["scores"]:
            bad = x["status"] != "CORRECT"
            if x["kind"] == "interior":
                it += 1
                ie += bad
            else:
                bt += 1
                be += bad
                if bad and x["boundary_k"] is not None:
                    hist[x["boundary_k"]] = hist.get(x["boundary_k"], 0) + 1
    p_b = _fisher_one_sided(be, bt - be, ie, it - ie)
    rb = be / bt if bt else 0.0
    ri = ie / it if it else 0.0
    gate_b = p_b <= 0.05 and (ri == 0 and be > 0 or (ri > 0 and rb >= 3 * ri))
    print("\n[Gate B] boundary vs interior, WITHIN flag=1 (immune to arm baseline)")
    print(f"  boundary needles: {be}/{bt} wrong ({rb:.1%})")
    print(f"  interior control: {ie}/{it} wrong ({ri:.1%})")
    print(f"  Fisher one-sided p = {p_b:.4f}")
    print(f"  -> {'FIRE (boundary-localized)' if gate_b else 'no fire'}")
    if hist:
        print("  per-boundary error histogram (k -> count):")
        for k in sorted(hist):
            print(f"     k={k:<4} (tok {k*CHUNK:>7})  {'#' * hist[k]} {hist[k]}")
        print("  ^ THIS is the localization the test exists to produce.")

    # ---- Gate C: logit divergence vs measured noise floor
    d_noise, d_flag, ag_noise, ag_flag = [], [], [], []
    for pid in sorted(set(f0v) & set(frv)):
        m, _mx, ag = _lp_delta(f0v[pid], frv[pid])
        if not math.isnan(m):
            d_noise.append(m)
            ag_noise.append(ag)
    for pid in shared:
        m, _mx, ag = _lp_delta(f0v[pid], f1v[pid])
        if not math.isnan(m):
            d_flag.append(m)
            ag_flag.append(ag)
    print("\n[Gate C] logit divergence vs cluster nondeterminism floor")
    gate_c = False
    if d_noise and d_flag:
        mn, mf = statistics.median(d_noise), statistics.median(d_flag)
        p_c = _mannwhitney_p(d_noise, d_flag)
        print(f"  noise floor (flag0 vs flag0): median {mn:.5f}, max {max(d_noise):.5f}")
        print(f"  flag0 vs flag1            : median {mf:.5f}, max {max(d_flag):.5f}")
        print(f"  Mann-Whitney p = {p_c:.4f}")
        print(f"  top1 agreement: noise min {min(ag_noise):.3f} | flag {min(ag_flag):.3f}")
        gate_c = (mf > max(d_noise)) or (p_c <= 0.05 and mf >= 2 * mn)
        if ag_flag and ag_noise and min(ag_flag) < min(ag_noise):
            print("  top1 agreement between arms is BELOW the noise floor -> flag C")
            gate_c = True
    else:
        print("  INSUFFICIENT DATA -- flag0_repeat arm missing. Gate C cannot be "
              "evaluated; do NOT substitute a comparison against zero.")
    print(f"  -> {'FIRE (numerics perturbed)' if gate_c else 'no fire'}")

    # ---- verdict
    print("\n" + "=" * 72)
    if gate_a and gate_b:
        v = ("REAL BUG, LOCALIZED to specific chunk boundaries. Do NOT enable. "
             "Fix the depth-1 fence in prefill_batched().")
    elif gate_a:
        v = ("REAL correctness regression, NOT boundary-localized. Do NOT enable; "
             "widen the investigation beyond the double buffer.")
    elif gate_b:
        v = ("REAL boundary-localized effect at sub-threshold global rate. "
             "Do NOT enable.")
    elif gate_c:
        v = ("Numerics measurably perturbed; correctness impact unproven. "
             "Do NOT enable; extend to N=24.")
    elif (b + c) <= 2:
        v = ("NO EFFECT DETECTED at this power. This is NOT 'proven safe' -- "
             "see the minimum-detectable-effect note below.")
    else:
        v = "INCONCLUSIVE -- extend to N=24 prompts in a follow-up session."
    print(f"VERDICT: {v}")
    base = sum(1 for p in f0v for x in f0v[p]["scores"] if x["status"] != "CORRECT")
    print(f"\nPower honesty: baseline (flag0) needle error rate = {base}/{npairs}. "
          f"With {npairs} paired needles this design has ~80% power against an "
          f"induced per-needle error rate of ~6-7%, and POOR power against ~1%. "
          f"Any 'no effect' conclusion must be stated with that bound.")
    print("=" * 72)


# ------------------------------------------------------------------- main


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--generate-only", action="store_true")
    ap.add_argument("--preflight", action="store_true")
    ap.add_argument("--run", action="store_true")
    ap.add_argument("--arm", choices=ARMS)
    ap.add_argument("--only", help="run a single prompt_id")
    ap.add_argument("--analyze-only", action="store_true")
    ap.add_argument("--template-overhead", type=int, default=0,
                    help="measured chat-template token prefix; recalibrate "
                         "offsets after the first real prompt_tokens reading")
    a = ap.parse_args()

    if a.generate_only:
        generate_prompts(a.template_overhead)
    elif a.preflight:
        sys.exit(0 if preflight() else 1)
    elif a.run:
        if not a.arm:
            sys.exit("--run requires --arm")
        if not preflight():
            sys.exit("preflight failed -- refusing to burn an approved session")
        run_arm(a.arm, a.only)
    elif a.analyze_only:
        analyze()
    else:
        ap.print_help()


if __name__ == "__main__":
    main()
