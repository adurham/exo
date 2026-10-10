#!/usr/bin/env python3
"""Round Q2-Gamma arm-matrix driver — the γ re-price (build-only, PREPARED).

Re-prices the dsv41 speculative draft depth ``gamma`` on the post-dense
production build by running a *bracketed* arm matrix in ONE boot:

    gamma3a -> gamma2 -> gamma4 -> gamma5 (if supported) -> gamma3b

Arms are switched by the PM-VERIFIED recipe: edit ``~/relaunch_exo.sh`` on BOTH
nodes (``sed`` the single ``DSV41_SPEC_GAMMA=N`` token), graceful node-process
relaunch, TRIGGER THE JIT MODEL LOAD (a tiny warmup POST — the relaunch spawns NO
runner, so ``GET /state`` alone never reaches READY; the POST returns 503 at the
~120 s load timeout then 200; its epoch is registered as an own request), wait
READY 2/2, assert RANK CONSISTENCY of the logged effective gamma, idle-guard, then
run the FIXED replays (benign 20K x3 + agentic 91K x2) with the FIXED salt base
``q2gamma`` so the content is byte-identical across arms.  If both nodes' latest
logged effective gamma is ALREADY the target arm the relaunch+warmup is SKIPPED
(REUSE) and control goes straight to the READY check.

This module is DELIVERED NOT-RUN: no live arm switch is executed here.  Run it
only after the eval boot (see ``README.md`` for preconditions).  ``--dry-run``
prints the header + the exact command sequence without touching the cluster.

Gates live in ``q2_gamma_eval`` (pure stdlib); this file only produces records.
"""
from __future__ import annotations

import argparse
import contextlib
import json
import os
import statistics
import subprocess
import sys
import time
import urllib.parse
import urllib.request

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)  # q2_gamma_eval
import q2_gamma_eval as EV  # noqa: E402

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------
NODES = ("studio1", "studio2")  # m4-1 (rank 1) + m4-2 (rank 0); recipe applies to both
API_BASE = os.environ.get("Q2_GAMMA_API", "http://192.168.86.48:52415")
VM_BASE = os.environ.get("Q2_GAMMA_VM", "http://172.16.0.42:8428")
RELAUNCH_PATH = "~/relaunch_exo.sh"
SPEC_GAMMA_ENV = "DSV41_SPEC_GAMMA"
SUPPORTED_GAMMA = EV.SUPPORTED_GAMMA  # (2, 3, 4, 5)
MODEL = "dealignai/DeepSeek-V4.1-Flash-UNCENSORED-EXL3-2.9bpw"

SALT_BASE = "q2gamma"
DEPTH = 20000
MAX_TOKENS = 800
CAPTURE_POSITIONS = 256  # top-2 logps captured for the identity signal
TOP_N = 2                # request top_logprobs = 2

# Arm matrix (bracketed): the two gamma-3 arms are the determinism control.
MATRIX = (("gamma3a", 3), ("gamma2", 2), ("gamma4", 4), ("gamma5", 5), ("gamma3b", 3))

# Budget anchors (measured, frozen r1kit control_*.json):
#   benign 20K  wall ~ 90.5 s/rep  (cold prefill ~70 s + decode ~21 s)
#   agentic 91K wall ~ 385 s/rep   (cold prefill ~357 s + decode ~26 s)
ANCHOR_BENIGN_WALL_S = 90.5
ANCHOR_AGENTIC_WALL_S = 385.0
LAUNCH_WALL_S = 300.0   # graceful relaunch both nodes, sequential (~2-3 min/node)
WARMUP_WALL_S = 120.0   # tiny POST that triggers the JIT model load after relaunch (503 -> 200)
IDLE_SETTLE_S = 90.0    # idle-guard settle + inter-chunk sleeps per arm
ROUND_BUDGET_WALL_MIN = 135.0  # declared cap for the measurement wall-clock

# --- warmup trigger (post-relaunch JIT-load kick) ---------------------------
# ``~/relaunch_exo.sh`` restarts the exo *process* but spawns NO runner (no
# LoadModel), so ``GET /state`` never reaches RunnerReady on its own.  A tiny
# generation POST triggers the JIT model load; the first attempt blocks until the
# ~120 s load timeout and returns 503, a repeat then succeeds 200 (live-measured
# 2026-10-10: 200 in 119.6 s).  Mirrors the prior-round ``arm_switch2.py`` loop.
WARMUP_SETTLE_S = 20.0        # short settle after the relaunch, before the POST
WARMUP_TIMEOUT_S = 300.0      # total retry budget (503 -> 200)
WARMUP_HTTP_TIMEOUT_S = 300.0  # per-attempt socket timeout (> the ~120 s load timeout)
WARMUP_RETRY_SLEEP_S = 5.0    # pause between retries
WARMUP_BODY = {"model": MODEL, "messages": [{"role": "user", "content": "Say OK."}],
               "max_tokens": 8, "temperature": 0.0, "stream": False}

SCRATCH = "/Users/adam.durham/.hermes/cache/scratch/phase20/q2gamma"
DEFAULT_REGISTRY = os.path.join(SCRATCH, "own_requests.jsonl")

_OVERRIDE_RE = __import__("re").compile(
    r"spec gamma override:\s+" + SPEC_GAMMA_ENV + r"=(\d+)\s*->\s*effective gamma=(\d+)"
    r".*?rank\s+(\d+)")


# ---------------------------------------------------------------------------
# Budget declaration (printed in the header + written into the round doc)
# ---------------------------------------------------------------------------
def budget_arithmetic(arms, benign_reps: int, agentic_reps: int) -> dict:
    n = len(arms)
    agentic_s = ANCHOR_AGENTIC_WALL_S * agentic_reps * n
    benign_s = ANCHOR_BENIGN_WALL_S * benign_reps * n
    warmup_s = WARMUP_WALL_S * n  # post-relaunch JIT-load trigger, ~120 s per relaunched arm
    overhead_s = (LAUNCH_WALL_S + IDLE_SETTLE_S) * n + warmup_s
    total_s = agentic_s + benign_s + overhead_s
    return {"n_arms": n, "benign_reps": benign_reps, "agentic_reps": agentic_reps,
            "agentic_s": agentic_s, "benign_s": benign_s, "warmup_s": warmup_s,
            "overhead_s": overhead_s, "total_s": total_s, "total_min": total_s / 60.0}


def choose_agentic_reps(arms, benign_reps: int, budget_wall_min: float) -> int:
    """Pre-declare the agentic rep count: 3 if it fits the wall cap, else 2."""
    for reps in (3, 2):
        if budget_arithmetic(arms, benign_reps, reps)["total_min"] <= budget_wall_min:
            return reps
    return 1


def declare_budget(arms, benign_reps: int = 3, budget_wall_min: float = ROUND_BUDGET_WALL_MIN) -> dict:
    agentic_reps = choose_agentic_reps(arms, benign_reps, budget_wall_min)
    a = budget_arithmetic(arms, benign_reps, agentic_reps)
    iqr_enabled = agentic_reps >= 3
    drop = [] if iqr_enabled else ["disjoint-IQR criterion (agentic reps < 3)"]
    lines = [
        "Q2-Gamma DECLARED BUDGET (fixed BEFORE spending)",
        f"  boot: 1 (eval boot)  |  arm switches: {len(arms)} (graceful node-process relaunches, NOT reboots)",
        f"  measurement wall cap: {budget_wall_min:.0f} min",
        f"  anchors: benign 20K {ANCHOR_BENIGN_WALL_S:.1f} s/rep, agentic 91K {ANCHOR_AGENTIC_WALL_S:.1f} s/rep "
        f"(frozen r1kit control)",
        f"  agentic: {ANCHOR_AGENTIC_WALL_S:.0f} s/rep x {agentic_reps} reps x {a['n_arms']} arms "
        f"= {a['agentic_s']:.0f} s = {a['agentic_s']/60:.1f} min",
        f"  benign:  {ANCHOR_BENIGN_WALL_S:.1f} s/rep x {benign_reps} reps x {a['n_arms']} arms "
        f"= {a['benign_s']:.0f} s = {a['benign_s']/60:.1f} min",
        f"  warmup:  {WARMUP_WALL_S:.0f} s/arm x {a['n_arms']} arms (JIT-load POST after each relaunch) "
        f"= {a['warmup_s']:.0f} s = {a['warmup_s']/60:.1f} min",
        f"  overhead: (relaunch {LAUNCH_WALL_S:.0f}s + idle {IDLE_SETTLE_S:.0f}s + warmup {WARMUP_WALL_S:.0f}s) "
        f"x {a['n_arms']} = {a['overhead_s']:.0f} s = {a['overhead_s']/60:.1f} min",
        f"  TOTAL: {a['total_s']:.0f} s = {a['total_min']:.1f} min  vs cap {budget_wall_min:.0f} min "
        f"=> {'FITS' if a['total_min'] <= budget_wall_min else 'OVER'}",
        f"  DECISION: agentic_reps={agentic_reps}, iqr_enabled={iqr_enabled}"
        + (f"; DROP {', '.join(drop)}" if drop else ""),
    ]
    return {"agentic_reps": agentic_reps, "benign_reps": benign_reps, "iqr_enabled": iqr_enabled,
            "budget_wall_min": budget_wall_min, "arithmetic": a,
            "dropped_criteria": drop, "lines": lines}


def render_header(arms, budget: dict) -> str:
    arm_str = " -> ".join(a for a, _ in arms)
    return "\n".join([
        "=" * 78,
        "Round Q2-Gamma — dsv41 speculative draft-depth (gamma) re-price",
        f"  arm matrix (bracketed): {arm_str}",
        f"  salt base: {SALT_BASE} (per-rep {SALT_BASE}-<rep>; FIXED => byte-identical across arms)",
        f"  benign 20K x{budget['benign_reps']} reps + agentic 91K x{budget['agentic_reps']} reps, "
        f"max_tokens={MAX_TOKENS}, temperature 0, stream, top_logprobs={TOP_N}",
        f"  API {API_BASE}   VM {VM_BASE}   nodes {', '.join(NODES)}",
        "",
        *budget["lines"],
        "=" * 78,
    ])


# ---------------------------------------------------------------------------
# Arm-switch recipe (VERIFIED by the PM) — pure command builders + simulation
# ---------------------------------------------------------------------------
def sed_cmd(n: int) -> str:
    return f"sed -i '' -E 's/{SPEC_GAMMA_ENV}=[0-9]+/{SPEC_GAMMA_ENV}={n}/' {RELAUNCH_PATH}"


def grep_cmd() -> str:
    return f"grep -o '{SPEC_GAMMA_ENV}=[0-9]*' {RELAUNCH_PATH}"


def relaunch_cmd() -> str:
    return f"{RELAUNCH_PATH}"


def count_tokens(text: str) -> list[int]:
    import re
    return [int(x) for x in re.findall(rf"{SPEC_GAMMA_ENV}=([0-9]+)", text)]


def simulate_sed(text: str, n: int) -> str:
    """In-memory model of the recipe's ``sed -E 's/DSV41_SPEC_GAMMA=[0-9]+/.../N/'``."""
    import re
    return re.sub(rf"{SPEC_GAMMA_ENV}=[0-9]+", f"{SPEC_GAMMA_ENV}={n}", text)


def _ssh(node: str, cmd: str, *, timeout: float = 30.0) -> tuple[int, str]:
    try:
        p = subprocess.run(["ssh", "-o", "BatchMode=yes", "-o", "ConnectTimeout=8", node, cmd],
                           capture_output=True, text=True, timeout=timeout)
    except (subprocess.TimeoutExpired, OSError) as exc:
        return 255, f"<ssh error: {exc}>"
    return p.returncode, (p.stdout or "") + (p.stderr or "")


def assert_precondition(dry: bool) -> None:
    """The eval deploy must carry the DSV41_SPEC_GAMMA token in relaunch_exo.sh.

    (Production relaunch_exo.sh does NOT — the eval launch adds it.  Without the
    token, ``sed`` is a silent no-op and the arm never flips.)
    """
    for node in NODES:
        if dry:
            print(f"[dry] {node}: {grep_cmd()}  # expect exactly one 'DSV41_SPEC_GAMMA=<N>'")
            continue
        rc, out = _ssh(node, grep_cmd())
        toks = [t for t in out.splitlines() if t.strip().startswith(SPEC_GAMMA_ENV)]
        if rc != 0 or len(toks) != 1:
            raise SystemExit(
                f"PRECONDITION FAILED on {node}: expected exactly ONE '{SPEC_GAMMA_ENV}=<N>' "
                f"token in {RELAUNCH_PATH}; got {toks!r} (rc={rc}). The eval deploy must be "
                f"launched with `export {SPEC_GAMMA_ENV}=3` so the token exists.")


def switch_arm(n: int, *, dry: bool = False, arm: str | None = None,
               registry_path: str | None = None, settle_s: float = WARMUP_SETTLE_S,
               allow_reuse: bool = True) -> dict:
    """Set arm ``n`` on BOTH nodes via the verified recipe (sed -> confirm -> relaunch).

    After the relaunch the driver TRIGGERS THE JIT MODEL LOAD with a tiny warmup
    POST (the relaunch spawns NO runner), registers that POST in the own-request
    registry so the next idle guard treats it as own traffic, and only then
    returns for :func:`wait_ready`.

    REUSE optimisation (safe): if BOTH nodes' latest logged effective gamma is
    ALREADY ``n`` (e.g. the first arm right after a boot), the relaunch+warmup is
    SKIPPED and control goes straight to the READY check + rank assert.  Returns
    ``{"mode": "reuse"|"relaunch", ...}`` so the caller can log/record which.
    """
    if n not in SUPPORTED_GAMMA:
        raise ValueError(f"gamma {n} not in supported set {SUPPORTED_GAMMA}")
    label = f"gamma{arm}" if arm and arm.startswith("gamma") else (arm or f"gamma{n}")

    if dry:
        print(f"[dry] ssh <node> \"{_LOG_TAIL_CMD}\"  # if latest effective gamma already == {n}: "
              f"REUSE (skip relaunch+warmup)")
        for node in NODES:
            print(f"[dry] ssh {node} \"{sed_cmd(n)}\"")
            print(f"[dry] ssh {node} \"{grep_cmd()}\"  # confirm one token == {n}")
            print(f"[dry] ssh {node} '{relaunch_cmd()}'  # graceful relaunch, ~2-3 min")
        print(f"[dry] sleep {settle_s:.0f}s settle, then warmup POST {API_BASE}/v1/chat/completions "
              f"({json.dumps(WARMUP_BODY)})  # retry 503->200, <= {WARMUP_TIMEOUT_S:.0f}s")
        print(f"[dry] register warmup epoch in own-request registry "
              f"({registry_path or 'own_requests_<arm>.jsonl'})")
        return {"mode": "relaunch", "dry": True}

    if allow_reuse:
        per = latest_effective_gamma()
        reuse, msg = reuse_verdict(per, n)
        if reuse:
            print(f"[{time.strftime('%T')}] {label}: REUSE — {msg}; "
                  f"skipping relaunch+warmup", flush=True)
            return {"mode": "reuse", "per": per, "reason": msg}

    for node in NODES:
        rc, out = _ssh(node, sed_cmd(n))
        if rc != 0:
            raise SystemExit(f"switch_arm({n}) sed failed on {node}: rc={rc} {out!r}")
        rc, out = _ssh(node, grep_cmd())
        toks = count_tokens(out)
        if rc != 0 or toks != [n]:
            raise SystemExit(f"switch_arm({n}) confirm failed on {node}: tokens={toks!r}")
        print(f"[{time.strftime('%T')}] {node}: {SPEC_GAMMA_ENV} set to {n}; relaunching...", flush=True)
        _ssh(node, relaunch_cmd(), timeout=600)

    if settle_s > 0:
        print(f"[{time.strftime('%T')}] {label}: settle {settle_s:.0f}s before warmup POST", flush=True)
        time.sleep(settle_s)
    warm = warmup_post(registry_path=registry_path, label=label)
    return {"mode": "relaunch", "warmup": warm}


# ---------------------------------------------------------------------------
# Warmup trigger (post-relaunch JIT model load) + own-request registration
# ---------------------------------------------------------------------------
def register_own_request(registry_path: str | None, t: float | None = None,
                         label: str = "q2gamma_warmup") -> float:
    """Append a ``{"label", "t"}`` own-request line to the registry (guard format).

    Mirrors ``phase20_guard.ChunkGuard.register_own_request`` so the same file is
    readable by ``_load_own`` / the idle guard.  Returns the registered epoch.
    """
    t = time.time() if t is None else t
    if not registry_path:
        return t
    try:
        d = os.path.dirname(registry_path)
        if d:
            os.makedirs(d, exist_ok=True)
        with open(registry_path, "a", encoding="utf-8") as fh:
            fh.write(json.dumps({"label": label, "t": t}) + "\n")
    except OSError:
        pass
    return t


def _http_warmup_post(body: dict | None = None, *, timeout_s: float = WARMUP_HTTP_TIMEOUT_S) -> int | None:
    """Send the warmup generation POST; return the HTTP status (None on transport error).

    The first attempt blocks until the ~120 s JIT-load timeout and returns 503;
    a repeat succeeds 200.  A connection error (API not yet up after the
    relaunch) returns ``None`` so the retry loop keeps waiting.
    """
    import urllib.error
    import urllib.request
    body = WARMUP_BODY if body is None else body
    req = urllib.request.Request(API_BASE + "/v1/chat/completions",
                                 data=json.dumps(body).encode(),
                                 headers={"Content-Type": "application/json"})
    try:
        with urllib.request.urlopen(req, timeout=timeout_s) as resp:  # noqa: S310 (fixed http LAN)
            resp.read()
            return resp.status
    except urllib.error.HTTPError as exc:
        return exc.code
    except Exception:  # noqa: BLE001 (transport: API still coming up)
        return None


def warmup_post(*, registry_path: str | None = None, label: str = "warmup",
                timeout_s: float = WARMUP_TIMEOUT_S, retry_sleep_s: float = WARMUP_RETRY_SLEEP_S,
                post_fn=None, sleep_fn=time.sleep, now_fn=time.time, log_fn=print) -> dict:
    """Trigger the JIT model load with a tiny POST; retry on 503 until 200.

    The POST epoch is registered in ``registry_path`` BEFORE sending so the next
    idle guard treats this driver traffic as own.  HTTP 200 => success.  HTTP
    5xx / transport errors are retried until ``timeout_s``; other non-200 codes
    are a hard failure (config error, not a slow load).
    """
    post_fn = _http_warmup_post if post_fn is None else post_fn
    t_reg = register_own_request(registry_path, now_fn(), label)
    t0 = now_fn()
    deadline = t0 + timeout_s
    attempt = 0
    statuses: list = []
    while True:
        attempt += 1
        a0 = now_fn()
        status = post_fn()
        elapsed = now_fn() - a0
        statuses.append(status)
        log_fn(f"[{time.strftime('%T')}] warmup {label}: attempt {attempt} -> HTTP {status} "
               f"in {elapsed:.1f}s (total {now_fn() - t0:.1f}s)", flush=True)
        if status == 200:
            return {"ok": True, "attempts": attempt, "statuses": statuses,
                    "elapsed_s": round(now_fn() - t0, 3), "registered": bool(registry_path),
                    "t_reg": t_reg}
        if status is not None and not (500 <= status < 600):
            raise SystemExit(f"warmup {label}: hard HTTP failure {status} "
                             f"(not a retryable JIT-load timeout)")
        if now_fn() >= deadline:
            raise SystemExit(f"warmup {label}: timed out after {timeout_s:.0f}s "
                             f"({attempt} attempts, statuses={statuses}) — JIT load never succeeded")
        sleep_fn(retry_sleep_s)


def latest_effective_gamma(*, dry: bool = False) -> dict:
    """Read each node's latest ``[DSV41] spec gamma override`` line -> {rank/env/eff} or None."""
    per: dict = {}
    for node in NODES:
        if dry:
            per[node] = None
            continue
        rc, out = _ssh(node, _LOG_TAIL_CMD)
        lines = parse_override_lines(out)
        per[node] = lines[-1] if lines else None
    return per


def reuse_verdict(per: dict, n: int) -> tuple[bool, str]:
    """Pure decision: can the relaunch+warmup be SKIPPED (already at effective gamma ``n``)?

    Reuse is safe (bracketed determinism semantics preserved) only when BOTH
    nodes' latest logged effective gamma == ``n`` AND the ranks are {0,1}.
    """
    if not per:
        return False, "no nodes probed"
    missing = [k for k, v in per.items() if not v]
    if missing:
        return False, f"no 'spec gamma override' line on {', '.join(sorted(missing))}"
    effs = {d.get("eff") for d in per.values()}
    if effs != {n}:
        return False, f"latest effective gammas {effs} != {{{n}}}"
    ranks = [d.get("rank") for d in per.values() if d.get("rank") is not None]
    if ranks and set(ranks) != {0, 1}:
        return False, f"expected ranks {{0,1}}, got {ranks}"
    return True, f"both ranks already log effective gamma={n}"


# ---------------------------------------------------------------------------
# Readiness + rank consistency
# ---------------------------------------------------------------------------
def count_ready_runners(state: dict) -> int:
    """Count runners reporting a ready/idle state in a ``GET /state`` payload."""
    runners = state.get("runners")
    if not isinstance(runners, dict):
        return 0
    n = 0
    for v in runners.values():
        if isinstance(v, dict) and any(k in ("RunnerReady", "RunnerIdle") for k in v):
            n += 1
    return n


def wait_ready(*, timeout_s: float = 600, poll_s: float = 10) -> int:
    """Poll ``GET /state`` until READY 2/2 (both runners placed + idle)."""
    deadline = time.time() + timeout_s
    last = 0
    while True:
        try:
            st = _fetch_json(f"{API_BASE}/state")
            last = count_ready_runners(st)
        except Exception as exc:  # noqa: BLE001
            last = -1
            print(f"[{time.strftime('%T')}] /state poll error: {exc!r}", flush=True)
        if last >= 2:
            return last
        if time.time() >= deadline:
            raise SystemExit(f"wait_ready timed out; last ready count={last}")
        time.sleep(poll_s)


def parse_override_lines(text: str) -> list[dict]:
    """Parse ``[DSV41] spec gamma override: ... rank R`` lines -> {env, eff, rank}."""
    out = []
    for m in _OVERRIDE_RE.finditer(text):
        out.append({"env": int(m.group(1)), "eff": int(m.group(2)), "rank": int(m.group(3))})
    return out


_LOG_TAIL_CMD = ('tail -n 5000 "$HOME/.exo/exo_log/exo.log" 2>/dev/null || true')


def rank_consistency_verdict(per: dict, n: int) -> tuple[bool, str]:
    """Pure decision: do the per-node override lines agree on effective gamma=N?

    ``per`` maps node -> {"env", "eff", "rank"}.  Aborts (False) if the two ranks'
    logged effective gamma differ, if either != N, or if the ranks are not {0,1}.
    """
    if not per:
        return False, "no per-node override lines"
    effs = {d.get("eff") for d in per.values()}
    envs = {d.get("env") for d in per.values()}
    if effs != {n} or envs != {n}:
        return False, f"per-node override lines {per!r} do not both report effective gamma={n}"
    ranks = [d.get("rank") for d in per.values() if d.get("rank") is not None]
    if ranks and set(ranks) != {0, 1}:
        return False, f"expected ranks {{0,1}}, got {ranks}"
    return True, "ok"


def confirm_rank_consistency(n: int, *, dry: bool = False) -> dict:
    """Assert BOTH ranks logged ``effective gamma=N`` and the two ranks agree."""
    per = {}
    for node in NODES:
        if dry:
            print(f"[dry] ssh {node} \"{_LOG_TAIL_CMD}\"  # grep 'spec gamma override' -> effective gamma=N")
            per[node] = {"env": n, "eff": n, "rank": None}
            continue
        rc, out = _ssh(node, _LOG_TAIL_CMD)
        lines = parse_override_lines(out)
        if not lines:
            raise SystemExit(f"rank-consistency: no 'spec gamma override' line on {node}; "
                             f"was the boot launched with {SPEC_GAMMA_ENV} set?")
        per[node] = lines[-1]
    ok, msg = rank_consistency_verdict(per, n)
    if not ok:
        raise SystemExit("RANK CONSISTENCY FAILED: " + msg)
    return per


# ---------------------------------------------------------------------------
# Replay (FIXED content; top-2 logprobs captured for the identity signal)
# ---------------------------------------------------------------------------
def _fetch_json(url: str, *, timeout: float = 10.0) -> dict:
    req = urllib.request.Request(url, method="GET")
    with urllib.request.urlopen(req, timeout=timeout) as resp:  # noqa: S310 (fixed http LAN)
        return json.loads(resp.read().decode("utf-8"))


def replay_once(prompt: str, max_tokens: int, *, top_n: int = TOP_N,
                capture_positions: int = CAPTURE_POSITIONS) -> dict:
    """One fixed replay; SSE parser mirroring the r1kit ``stream_once`` plus logprobs.

    Requests ``logprobs=True, top_logprobs=top_n`` and captures the first
    ``capture_positions`` (token, (lp_top1, lp_top2)) pairs so the identity
    gate's first-divergence margin is computable.  Returns the same keys the
    r1kit driver produces (``wall_s/ttft_s/decode_s/content_chars/usage/stats/
    finish_reason``) plus ``tokens`` and ``top2_logprobs``.
    """
    body = {"model": MODEL, "messages": [{"role": "user", "content": prompt}],
            "max_tokens": max_tokens, "temperature": 0, "stream": True,
            "logprobs": True, "top_logprobs": top_n}
    req = urllib.request.Request(API_BASE + "/v1/chat/completions",
                                 data=json.dumps(body).encode(),
                                 headers={"Content-Type": "application/json"})
    t0 = time.perf_counter()
    ttft = first = last = None
    chars = 0
    usage = stats = finish = None
    tokens: list = []
    top2: list = []
    lp_seen = False
    with urllib.request.urlopen(req, timeout=3600) as resp:  # noqa: S310
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
            if usage is None and obj.get("usage"):
                usage = obj["usage"]
            for ch in obj.get("choices", []):
                if ch.get("finish_reason"):
                    finish = ch["finish_reason"]
                d = ch.get("delta") or {}
                t = d.get("content") or d.get("reasoning_content") or ""
                if t:
                    if ttft is None:
                        ttft, first = now - t0, now
                    chars += len(t)
                    last = now
                lps = ch.get("logprobs")
                if isinstance(lps, dict) and lps.get("content"):
                    lp_seen = True
                    for item in lps["content"]:
                        if len(tokens) >= capture_positions:
                            continue
                        tops = item.get("top_logprobs") or []
                        lp1 = item.get("logprob")
                        lp2 = tops[1].get("logprob") if len(tops) >= 2 else None
                        tokens.append(item.get("token"))
                        top2.append((lp1, lp2))
    wall = time.perf_counter() - t0
    return {"wall_s": round(wall, 3),
            "ttft_s": round(ttft, 4) if ttft else None,
            "decode_s": round(last - first, 4) if (first and last and last > first) else None,
            "content_chars": chars, "usage": usage, "stats": stats, "finish_reason": finish,
            "tokens": tokens, "top2_logprobs": top2, "logprobs_available": lp_seen}


# ---------------------------------------------------------------------------
# Memory reads
# ---------------------------------------------------------------------------
def parse_vm_value(resp: dict) -> float | None:
    try:
        res = resp["data"]["result"]
        if not res:
            return None
        return float(res[0]["value"][1])
    except (KeyError, IndexError, TypeError, ValueError):
        return None


def query_vm_peak(instance: str, window_s: int = 3600) -> float | None:
    q = f'max_over_time(exo_peak_memory_bytes{{instance="{instance}"}}[{window_s}s])'
    url = f"{VM_BASE}/api/v1/query?query=" + urllib.parse.quote(q)
    try:
        return parse_vm_value(_fetch_json(url, timeout=10))
    except Exception:  # noqa: BLE001
        return None


def parse_footprint_bytes(text: str) -> int | None:
    """First line whose leading token is numeric => bytes; else max integer seen."""
    for line in text.splitlines():
        s = line.strip()
        if not s:
            continue
        try:
            return int(float(s.split()[0]))
        except (ValueError, IndexError):
            continue
    import re
    nums = [int(n) for n in re.findall(r"\d+", text)]
    return max(nums) if nums else None


_FOOTPRINT_CMD = ("pid=$(pgrep -f 'multiprocessing.spawn import spawn_main' | head -1); "
                  'echo "RUNNER_PID=$pid"; [ -n "$pid" ] && footprint -f bytes -p $pid 2>/dev/null')


def read_peak_resident(node: str, *, timeout: float = 60.0) -> int | None:
    rc, out = _ssh(node, _FOOTPRINT_CMD, timeout=timeout)
    if rc != 0:
        return None
    return parse_footprint_bytes(out)


def collect_memory() -> dict:
    """Peak allocator (VM gauge, CORRECTED /1.073741824) + peak resident per node."""
    alloc, resident = {}, {}
    for node in NODES:
        raw = query_vm_peak(node)
        alloc[node] = (raw / EV.ALLOC_CORRECTION) if raw is not None else None
        resident[node] = read_peak_resident(node)
    alloc = {k: v for k, v in alloc.items() if v is not None}
    resident = {k: v for k, v in resident.items() if v is not None}
    return {"peak_alloc_bytes": alloc, "peak_resident_bytes": resident}


# ---------------------------------------------------------------------------
# Run one arm (idle-guarded chunks, <=15 min)
# ---------------------------------------------------------------------------
def _load_kit():
    """Resolve the frozen fixed-replay kit lazily (so import never needs it)."""
    cands = [os.environ.get("Q2_GAMMA_KIT"),
             "/private/tmp/q1b-driver/bench/phase20_r1kit",
             "/private/tmp/next16-instr/bench",
             HERE]
    for c in cands:
        if c and os.path.exists(os.path.join(c, "phase19_round_measure.py")):
            if c not in sys.path:
                sys.path.insert(0, c)
            import importlib
            RM = importlib.import_module("phase19_round_measure")
            AM = importlib.import_module("phase19_agentic_measure")
            return RM, AM
    raise SystemExit("frozen r1kit not found (set Q2_GAMMA_KIT); expected "
                     "phase19_round_measure.py + phase19_agentic_measure.py")


def _load_guard():
    sys.path.insert(0, os.path.dirname(HERE))  # repo bench/ (phase20_guard.py)
    import phase20_guard as G
    return G


def _load_own(registry_path: str) -> list[float]:
    out: list[float] = []
    if registry_path and os.path.exists(registry_path):
        with open(registry_path, encoding="utf-8") as fh:
            for line in fh:
                line = line.strip()
                if line:
                    with contextlib.suppress(Exception):
                        out.append(float(json.loads(line)["t"]))
    return out


def arm_registry_path(out_dir: str, arm: str) -> str:
    """Canonical per-arm own-request registry (shared by switch_arm + run_arm)."""
    return os.path.join(out_dir, f"own_requests_{arm}.jsonl")


def run_arm(arm: str, gamma: int, budget: dict, *, out_dir: str) -> dict:
    """Run benign + agentic fixed replays for one arm; return the arm record."""
    RM, AM = _load_kit()
    G = _load_guard()
    os.makedirs(out_dir, exist_ok=True)
    registry = arm_registry_path(out_dir, arm)
    log_dir = os.path.join(out_dir, "guard")
    own = _load_own(registry)

    rec: dict = {"arm": arm, "gamma": gamma, "logprobs_available": True,
                 "benign": [], "agentic": [], "peak_alloc_bytes": {}, "peak_resident_bytes": {}}

    def run_chunk(kind: str, reps: int) -> None:
        nonlocal own
        label = f"q2gamma_{arm}_{kind}"
        print(f"[{time.strftime('%T')}] wait-idle before {label} ({reps} reps)...", flush=True)
        G.wait_for_idle(max_wait_s=1500, own_requests=own)
        prev_cyc = prev_acc = prev_hist = None
        with G.ChunkGuard(label, max_wall_s=900, log_dir=log_dir,
                          registry_path=registry, own_requests=own) as guard:
            for j in range(reps):
                if guard.cancel_event.is_set():
                    break
                salt = f"{SALT_BASE}-{j}"
                if kind == "agentic":
                    prompt, meta = AM.build_agentic_prompt(salt, None)
                else:
                    prompt = RM.build_prompt(DEPTH, salt, "count")
                    meta = {"prompt_chars": len(prompt)}
                guard.register_own_request(time.time())
                r = replay_once(prompt, MAX_TOKENS)
                r.update(RM.derive(r, prev_cyc, prev_acc, gamma))
                hist_cur = (r.get("stats") or {}).get("mtp_accepted_histogram_cumulative")
                r["hist_delta"] = EV.hist_delta(hist_cur, prev_hist)
                r["mean_accepted_hist"] = EV.mean_accepted_from_hist(r["hist_delta"])
                r.update({"rep": j, "arm": arm, "chunk": label, "salt": salt, "meta": meta})
                if r.get("cycles_cum") is not None:
                    prev_cyc, prev_acc = r["cycles_cum"], r["accepted_cum"]
                if hist_cur is not None:
                    prev_hist = hist_cur
                rec[kind].append(r)
                rec["logprobs_available"] = rec["logprobs_available"] and bool(r.get("logprobs_available"))
                print(json.dumps({k: r.get(k) for k in (
                    "rep", "salt", "prompt_tokens", "completion_tokens", "ttft_s", "decode_s",
                    "decode_tps", "rounds", "mean_accepted", "ms_per_round", "gamma_implied",
                    "mean_accepted_hist", "logprobs_available")}), flush=True)
                time.sleep(2)
        if guard.aborted:
            print(f"[{time.strftime('%T')}] chunk {label} ABORTED: {guard.reason}", flush=True)
        own = _load_own(registry)
        time.sleep(10)

    run_chunk("benign", budget["benign_reps"])
    run_chunk("agentic", budget["agentic_reps"])
    rec.update(collect_memory())
    with open(os.path.join(out_dir, f"arm_{arm}.json"), "w", encoding="utf-8") as fh:
        json.dump(rec, fh, indent=1)
    return rec


# ---------------------------------------------------------------------------
# Evaluate + main
# ---------------------------------------------------------------------------
def evaluate(arms_records: dict, budget: dict) -> dict:
    mem = EV.memory_gate(arms_records)
    bars = EV.bars(arms_records, mem=mem, iqr_enabled=budget["iqr_enabled"])
    identity = EV.identity_gate(arms_records)
    drift = EV.drift_control(arms_records.get("gamma3a", {}).get("agentic"),
                             arms_records.get("gamma3b", {}).get("agentic"))
    return {"memory_gate": mem, "bars": bars, "identity_gate": identity, "drift_control": drift}


def _arm_records_from_dir(out_dir: str, arms) -> dict:
    recs = {}
    for arm, _ in arms:
        p = os.path.join(out_dir, f"arm_{arm}.json")
        if os.path.exists(p):
            recs[arm] = json.load(open(p, encoding="utf-8"))
    return recs


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(prog="q2_gamma_driver")
    ap.add_argument("--matrix", default=None, help="e.g. '3a,2,4,5,3b' (default the bracketed matrix)")
    ap.add_argument("--include-gamma5", choices=["auto", "yes", "no"], default="auto",
                    help="gamma5 in the matrix (supported set is (2,3,4,5) => auto=yes)")
    ap.add_argument("--benign-reps", type=int, default=3)
    ap.add_argument("--agentic-reps", type=int, default=None,
                    help="override the pre-declared count (default: chosen from the budget)")
    ap.add_argument("--budget-wall-min", type=float, default=ROUND_BUDGET_WALL_MIN)
    ap.add_argument("--outdir", default=SCRATCH)
    ap.add_argument("--dry-run", action="store_true", help="print header + command plan; touch nothing")
    a = ap.parse_args(argv)

    arms = list(MATRIX)
    if a.include_gamma5 == "no":
        arms = [x for x in arms if x[0] != "gamma5"]
    if a.matrix:
        name_by_g = {name: g for name, g in MATRIX}

        def resolve(tok):
            tok = tok.strip()
            if tok in name_by_g:
                return tok, name_by_g[tok]
            t = tok[5:] if tok.startswith("gamma") else tok
            if t in ("3a", "3b"):
                return f"gamma{t}", 3
            if t.isdigit() and int(t) in SUPPORTED_GAMMA:
                return f"gamma{t}", int(t)
            raise SystemExit(f"unknown arm token {tok!r}")

        arms = [resolve(t) for t in a.matrix.split(",")]

    budget = declare_budget(arms, benign_reps=a.benign_reps, budget_wall_min=a.budget_wall_min)
    if a.agentic_reps is not None:
        budget["agentic_reps"] = a.agentic_reps
        budget["iqr_enabled"] = a.agentic_reps >= 3

    print(render_header(arms, budget), flush=True)
    os.makedirs(a.outdir, exist_ok=True)
    with open(os.path.join(a.outdir, "round_budget.json"), "w", encoding="utf-8") as fh:
        json.dump(budget, fh, indent=1)

    if a.dry_run:
        print("\n--- DRY RUN: command plan (nothing executed) ---", flush=True)
        assert_precondition(dry=True)
        for arm, g in arms:
            print(f"\n# arm {arm} (gamma={g})")
            switch_arm(g, dry=True, arm=arm, registry_path=arm_registry_path(a.outdir, arm))
            print(f"[dry] poll GET {API_BASE}/state until READY 2/2")
            confirm_rank_consistency(g, dry=True)
            print(f"[dry] idle-guard then: benign {DEPTH} x{budget['benign_reps']} + "
                  f"agentic 91K x{budget['agentic_reps']} (salt {SALT_BASE}-<rep>)")
        print("\n[dry] done — no cluster interaction performed.")
        return 0

    assert_precondition(dry=False)
    records = {}
    switches = {}
    for arm, g in arms:
        print(f"\n[{time.strftime('%T')}] === arm {arm} (gamma={g}) ===", flush=True)
        outcome = switch_arm(g, arm=arm, registry_path=arm_registry_path(a.outdir, arm))
        switches[arm] = outcome["mode"]
        print(f"[{time.strftime('%T')}] arm {arm}: switch mode = {outcome['mode']}", flush=True)
        wait_ready()
        confirm_rank_consistency(g)
        records[arm] = run_arm(arm, g, budget, out_dir=a.outdir)
        records[arm]["switch_mode"] = outcome["mode"]
    with open(os.path.join(a.outdir, "arm_switches.json"), "w", encoding="utf-8") as fh:
        json.dump(switches, fh, indent=1)

    ev = evaluate(records, budget)
    print("\n=== GATE RESULTS ===")
    print("drift_control:", json.dumps(ev["drift_control"]))
    print("identity_gate verdict:", ev["identity_gate"]["verdict"],
          "signal:", ev["identity_gate"].get("signal"))
    print("memory_gate:", json.dumps({k: v["eligible"] for k, v in ev["memory_gate"].items()}))
    print("bars verdict:", ev["bars"]["verdict"])
    with open(os.path.join(a.outdir, "round_eval.json"), "w", encoding="utf-8") as fh:
        json.dump(ev, fh, indent=1)
    return 0


if __name__ == "__main__":
    sys.exit(main())
