#!/usr/bin/env python3
# Copyright © 2026 Adam Durham (hermes-gw).
"""Round Q2-FU1 capture driver -- run the VERIFY-SHAPE arms, pull, gate, analyze.

PREPARED ARTIFACT.  It drives the live cluster: run it ONLY after the eval boot
(see RUNBOOK.md) and an explicit GO.  ``--dry-run`` prints the plan and touches
nothing.

WHAT IT DOES (one eval boot, gamma3 then gamma4 via the verified arm-switch
recipe -- the SAME recipe as q2_gamma_driver.py, reimplemented here so this
driver is self-contained):

  gamma3a arm:
    1. shape-sweep request  (correlation `q2fu1;arm=g3;sweep=1`, salt q2fu1-0):
       the hook runs the M=1/3/4/5/6 + repeat + bf16/fp32 + fence + rollback
       A/B program at the fixed prefix and continues decoding normally.
    2. multi-step request   (correlation `q2fu1;arm=g3;ms=24;rollback=1;
       rounds=8;nospec=0`, agentic salt `q2gamma-0`, max_tokens=256,
       top_logprobs=2): passive capture of the first 24 real verify rounds +
       the A/B probes inside the first 8 rounds.
    3. nospec reference     (correlation `q2fu1;arm=g3n;nospec=1;sweep=0`,
       salt `q2gamma-0`): pure-greedy stream of the same prompt.
  gamma4 arm: switch via sed ~/relaunch_exo.sh + relaunch + warmup POST +
    rank-consistency, then the same three requests with arm=g4/g4n.
  gamma2 + gamma5 arms: multi-step only (optional, --include-2-5).

  Pull: scp the per-rank npz + summary json off BOTH nodes bit-exact, merge
  (logits: NaN-spliced vocab halves; every other key must be BITWISE equal
  across ranks), then evaluate:

  * FAITHFULNESS GATE (loud): the gamma3 multi-step request must reproduce the
    recorded round artifact arm_gamma3a.json exactly (first 256 tokens + top-2
    logprobs) and gamma4 must reproduce arm_gamma4.json (divergence at idx
    14/12).  FAIL => the dumps are untrustworthy; the driver says so loudly.
  * noise floor: M=4 vs M=4 repeat delta.
  * H1/H2/H3 read-out per M: row-0 |delta-logit| vs M=1, routing identity
    (idx_L bitwise), router margins at the delta positions.
  * allsum bf16-vs-fp32, fence 0-vs-default arms.
  * rollback A/B: production vs +ring-restore (per gamma, per accepted count).

Outputs under --outdir: captures/<arm>_M<M>_<tag>.npz (merged, both ranks),
arm_<arm>_capture.json (per-request metrics + gate verdicts), and
q2fu1_report.md (the read-out table).
"""
from __future__ import annotations

import argparse
import contextlib
import json
import os
import re
import subprocess
import sys
import time
import urllib.error
import urllib.parse
import urllib.request

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
NODES = ("studio1", "studio2")          # studio1=rank1, studio2=rank0
API_BASE = os.environ.get("Q2FU1_API", "http://192.168.86.48:52415")
MODEL = "dealignai/DeepSeek-V4.1-Flash-UNCENSORED-EXL3-2.9bpw"
RELAUNCH_PATH = "~/relaunch_exo.sh"
SPEC_GAMMA_ENV = "DSV41_SPEC_GAMMA"
SUPPORTED_GAMMA = (2, 3, 4, 5)
NODE_OUTDIR = os.environ.get("Q2FU1_NODE_OUT", "/tmp/q2fu1/out")
NODE_HOOK_DIR = os.path.dirname(NODE_OUTDIR)        # staged hook dir (PYTHONPATH)
SALT_SWEEP = "q2fu1-0"
SALT_STREAM = "q2gamma-0"
CAPTURE_POSITIONS = 256
RECORDED_DIR = os.environ.get(
    "Q2FU1_RECORDED",
    "/private/tmp/phase20-campaign/docs/benchmarks/phase20-throughput/raw/pricing/q2/round")
DEFAULT_OUTDIR = "/Users/adam.durham/.hermes/cache/scratch/phase20/q2fu1"

WARMUP_BODY = {"model": MODEL,
               "messages": [{"role": "user", "content": "Say OK."}],
               "max_tokens": 8, "temperature": 0.0, "stream": False}
WARMUP_TIMEOUT_S = 300.0
WARMUP_HTTP_TIMEOUT_S = 300.0
WARMUP_RETRY_SLEEP_S = 5.0
WARMUP_SETTLE_S = 20.0
LOG_TAIL_CMD = ('tail -n 8000 "$HOME/.exo/exo_log/exo.log" 2>/dev/null || true')

_OVERRIDE_RE = re.compile(
    r"spec gamma override:\s+" + SPEC_GAMMA_ENV + r"=(\d+)\s*->\s*effective gamma=(\d+)"
    r".*?rank\s+(\d+)")
_HOOK_RE = re.compile(r"q2fu1 capture hook ACTIVE -> (\S+)")


# ---------------------------------------------------------------------------
# ssh / scp / http
# ---------------------------------------------------------------------------
def _ssh(node: str, cmd: str, *, timeout: float = 60.0) -> tuple[int, str]:
    try:
        p = subprocess.run(["ssh", "-o", "BatchMode=yes", "-o", "ConnectTimeout=8",
                            node, cmd], capture_output=True, text=True, timeout=timeout)
    except (subprocess.TimeoutExpired, OSError) as exc:
        return 255, f"<ssh error: {exc}>"
    return p.returncode, (p.stdout or "") + (p.stderr or "")


def _scp_from(node: str, remote_glob: str, local_dir: str, *, timeout: float = 300.0
              ) -> tuple[int, list[str]]:
    os.makedirs(local_dir, exist_ok=True)
    before = set(os.listdir(local_dir))
    try:
        p = subprocess.run(["scp", "-q", "-o", "BatchMode=yes",
                            f"{node}:{remote_glob}", local_dir],
                           capture_output=True, text=True, timeout=timeout)
    except (subprocess.TimeoutExpired, OSError) as exc:
        return 255, [f"<scp error: {exc}>"]
    after = set(os.listdir(local_dir)) - before
    return p.returncode, sorted(after)


def _fetch_json(url: str, *, timeout: float = 10.0) -> dict:
    with urllib.request.urlopen(urllib.request.Request(url, method="GET"),
                                timeout=timeout) as resp:  # noqa: S310 (fixed LAN)
        return json.loads(resp.read().decode("utf-8"))


def _post_http(body: dict, *, timeout_s: float) -> int | None:
    req = urllib.request.Request(API_BASE + "/v1/chat/completions",
                                 data=json.dumps(body).encode(),
                                 headers={"Content-Type": "application/json"})
    try:
        with urllib.request.urlopen(req, timeout=timeout_s) as resp:  # noqa: S310
            resp.read()
            return resp.status
    except urllib.error.HTTPError as exc:
        return exc.code
    except Exception:  # noqa: BLE001 (transport: API coming up)
        return None


# ---------------------------------------------------------------------------
# own-request registry + idle guard (mirror of the q2-gamma machinery)
# ---------------------------------------------------------------------------
def register_own_request(registry_path: str | None, t: float | None = None,
                         label: str = "q2fu1") -> float:
    t = time.time() if t is None else t
    if registry_path:
        try:
            os.makedirs(os.path.dirname(registry_path), exist_ok=True)
            with open(registry_path, "a", encoding="utf-8") as fh:
                fh.write(json.dumps({"label": label, "t": t}) + "\n")
        except OSError:
            pass
    return t


def warmup_post(*, registry_path: str | None = None, label: str = "warmup",
                timeout_s: float = WARMUP_TIMEOUT_S) -> dict:
    t0 = time.time()
    attempt = 0
    statuses: list = []
    while True:
        attempt += 1
        register_own_request(registry_path, time.time(), label)
        status = _post_http(WARMUP_BODY, timeout_s=WARMUP_HTTP_TIMEOUT_S)
        statuses.append(status)
        print(f"[{time.strftime('%T')}] warmup {label}: attempt {attempt} -> "
              f"HTTP {status}", flush=True)
        if status == 200:
            return {"ok": True, "attempts": attempt, "statuses": statuses}
        if status is not None and not (500 <= status < 600):
            raise SystemExit(f"warmup {label}: hard HTTP failure {status}")
        if time.time() - t0 >= timeout_s:
            raise SystemExit(f"warmup {label}: timed out; statuses={statuses}")
        time.sleep(WARMUP_RETRY_SLEEP_S)


_RUNNER_STATES = ("RunnerReady", "RunnerIdle", "RunnerLoading")


def state_has_runner(state: dict) -> bool:
    runners = state.get("runners")
    if isinstance(runners, dict):
        for v in runners.values():
            if isinstance(v, dict) and any(k in v for k in _RUNNER_STATES):
                return True
    inst = state.get("instances")
    if isinstance(inst, (list, tuple, dict)):
        return len(inst) > 0
    if isinstance(inst, int):
        return inst > 0
    return False


def count_ready_runners(state: dict) -> int:
    runners = state.get("runners")
    n = 0
    if isinstance(runners, dict):
        for v in runners.values():
            if isinstance(v, dict) and any(k in v for k in ("RunnerReady", "RunnerIdle")):
                n += 1
    return n


def ensure_loaded(*, registry_path: str | None = None, label: str = "ensure_loaded"
                  ) -> tuple[bool, str]:
    try:
        st = _fetch_json(f"{API_BASE}/state")
    except Exception as exc:  # noqa: BLE001
        st = {}
        print(f"[{time.strftime('%T')}] ensure_loaded {label}: /state error {exc!r}",
              flush=True)
    if state_has_runner(st):
        return True, "runner present; no POST needed"
    print(f"[{time.strftime('%T')}] ensure_loaded {label}: no runner -> warmup POST",
          flush=True)
    warm = warmup_post(registry_path=registry_path, label=label)
    return True, f"warmup ok (attempts={warm['attempts']}, statuses={warm['statuses']})"


def wait_ready(*, timeout_s: float = 600, poll_s: float = 10, warmup_ok: bool = False) -> int:
    deadline = time.time() + timeout_s
    last = 0
    while True:
        try:
            last = count_ready_runners(_fetch_json(f"{API_BASE}/state"))
        except Exception:  # noqa: BLE001
            last = -1
        if last >= 2:
            return last
        if time.time() >= deadline:
            if warmup_ok:
                print(f"[{time.strftime('%T')}] WARNING: /state not READY 2/2 but the "
                      f"warmup POST served 200; continuing", flush=True)
                return max(last, 0)
            raise SystemExit(f"wait_ready timed out; ready={last}")
        time.sleep(poll_s)


# ---------------------------------------------------------------------------
# arm switch (same recipe as q2_gamma_driver; VERIFIED by the PM)
# ---------------------------------------------------------------------------
def _sed_cmd(n: int) -> str:
    return (f"sed -i '' -E 's/{SPEC_GAMMA_ENV}=[0-9]+/{SPEC_GAMMA_ENV}={n}/' "
            f"{RELAUNCH_PATH}")


def _grep_cmd() -> str:
    return f"grep -o '{SPEC_GAMMA_ENV}=[0-9]*' {RELAUNCH_PATH}"


def latest_effective_gamma() -> dict:
    per: dict = {}
    for node in NODES:
        rc, out = _ssh(node, LOG_TAIL_CMD)
        lines = _OVERRIDE_RE.findall(out)
        per[node] = ({"env": int(lines[-1][0]), "eff": int(lines[-1][1]),
                      "rank": int(lines[-1][2])} if lines else None)
    return per


def confirm_rank_consistency(n: int) -> dict:
    per = latest_effective_gamma()
    effs = {d.get("eff") for d in per.values() if d}
    ranks = {d.get("rank") for d in per.values() if d}
    if len(effs) != 1 or effs != {n} or (ranks and ranks != {0, 1}):
        raise SystemExit(f"RANK CONSISTENCY FAILED: per-node override lines {per!r} "
                         f"(want effective gamma={n} on both ranks)")
    return per


def confirm_hook_active() -> dict:
    per: dict = {}
    for node in NODES:
        rc, out = _ssh(node, LOG_TAIL_CMD)
        hits = _HOOK_RE.findall(out)
        per[node] = hits[-1] if hits else None
    if not all(per.values()):
        raise SystemExit(
            f"CAPTURE HOOK NOT ACTIVE on both nodes: {per!r}. The eval boot must "
            f"export DSV41_Q2FU1_CAPTURE={NODE_OUTDIR} (see RUNBOOK.md) and stage "
            f"q2fu1_capture.py in {NODE_HOOK_DIR}.")
    return per


def switch_arm(n: int, *, registry_path: str | None = None, arm: str | None = None,
               allow_reuse: bool = True, dry: bool = False) -> dict:
    """gamma n on BOTH nodes via sed -> confirm -> relaunch -> warmup -> loaded."""
    label = arm or f"gamma{n}"
    if dry:
        for node in NODES:
            print(f"[dry] ssh {node} \"{_sed_cmd(n)}\"")
            print(f"[dry] ssh {node} \"{_grep_cmd()}\"  # confirm one token == {n}")
            print(f"[dry] ssh {node} '{RELAUNCH_PATH}'  # graceful relaunch ~2-3 min")
        print(f"[dry] sleep {WARMUP_SETTLE_S:.0f}s; warmup POST (503->200); "
              f"ensure_loaded; READY; rank consistency (want effective gamma={n})")
        return {"mode": "dry"}
    if allow_reuse:
        per = latest_effective_gamma()
        effs = {d.get("eff") for d in per.values() if d}
        if effs == {n}:
            print(f"[{time.strftime('%T')}] {label}: REUSE (already gamma={n}); "
                  f"warmup still runs", flush=True)
            warm = warmup_post(registry_path=registry_path, label=label)
            ok, detail = ensure_loaded(registry_path=registry_path, label=label)
            if not ok:
                raise SystemExit(f"{label}: ensure_loaded failed: {detail}")
            return {"mode": "reuse", "warmup": warm}
    for node in NODES:
        rc, out = _ssh(node, _sed_cmd(n))
        if rc != 0:
            raise SystemExit(f"{label}: sed failed on {node}: {out!r}")
        rc, out = _ssh(node, _grep_cmd())
        toks = [int(x.split("=")[1]) for x in out.split() if x.startswith(SPEC_GAMMA_ENV)]
        if rc != 0 or toks != [n]:
            raise SystemExit(f"{label}: confirm failed on {node}: tokens={toks!r}")
        print(f"[{time.strftime('%T')}] {node}: {SPEC_GAMMA_ENV}={n}; relaunching...",
              flush=True)
        _ssh(node, RELAUNCH_PATH, timeout=900)
    time.sleep(WARMUP_SETTLE_S)
    warm = warmup_post(registry_path=registry_path, label=label)
    ok, detail = ensure_loaded(registry_path=registry_path, label=label)
    if not ok:
        raise SystemExit(f"{label}: ensure_loaded failed: {detail}")
    return {"mode": "relaunch", "warmup": warm}


# ---------------------------------------------------------------------------
# capture requests (SSE stream with logprobs)
# ---------------------------------------------------------------------------
def stream_request(prompt: str, correlation_id: str, *, max_tokens: int,
                   registry_path: str | None = None, label: str = "capture",
                   capture_positions: int = CAPTURE_POSITIONS,
                   timeout_s: float = 3600.0) -> dict:
    register_own_request(registry_path, time.time(), label)
    body = {"model": MODEL, "messages": [{"role": "user", "content": prompt}],
            "max_tokens": max_tokens, "temperature": 0, "stream": True,
            "logprobs": True, "top_logprobs": 2, "correlation_id": correlation_id}
    req = urllib.request.Request(API_BASE + "/v1/chat/completions",
                                 data=json.dumps(body).encode(),
                                 headers={"Content-Type": "application/json"})
    t0 = time.perf_counter()
    tokens: list = []
    top2: list = []
    stats = None
    finish = None
    with urllib.request.urlopen(req, timeout=timeout_s) as resp:  # noqa: S310
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
            with contextlib.suppress(Exception):
                obj = json.loads(payload)
                for ch in obj.get("choices", []):
                    if ch.get("finish_reason"):
                        finish = ch["finish_reason"]
                    lps = ch.get("logprobs")
                    if isinstance(lps, dict) and lps.get("content"):
                        for item in lps["content"]:
                            if len(tokens) >= capture_positions:
                                continue
                            tops = item.get("top_logprobs") or []
                            lp2 = tops[1].get("logprob") if len(tops) >= 2 else None
                            tokens.append(item.get("token"))
                            top2.append([item.get("logprob"), lp2])
    return {"wall_s": round(time.perf_counter() - t0, 3), "tokens": tokens,
            "top2_logprobs": top2, "stats": stats, "finish_reason": finish,
            "correlation_id": correlation_id}


def build_agentic_prompt(salt: str):
    """The frozen kit's fixed agentic prompt (same as round Q2-Gamma)."""
    cands = [os.environ.get("Q2FU1_KIT"),
             "/private/tmp/q1b-driver/bench/phase20_r1kit",
             "/private/tmp/next16-instr/bench"]
    for c in cands:
        if c and os.path.exists(os.path.join(c, "phase19_agentic_measure.py")):
            if c not in sys.path:
                sys.path.insert(0, c)
            import importlib
            AM = importlib.import_module("phase19_agentic_measure")
            return AM.build_agentic_prompt(salt, None)
    raise SystemExit("frozen r1kit (phase19_agentic_measure.py) not found; "
                     "set Q2FU1_KIT")


# ---------------------------------------------------------------------------
# pull + merge
# ---------------------------------------------------------------------------
def pull_captures(local_dir: str, *, label: str) -> list[str]:
    """Pull each rank's npz/json into its own subdir (no name collisions)."""
    got: list[str] = []
    for node in NODES:
        rank_dir = os.path.join(local_dir, f"rank_{node}")
        os.makedirs(rank_dir, exist_ok=True)
        rc, files = _scp_from(node, f"{NODE_OUTDIR}/*.npz", rank_dir)
        if rc != 0:
            raise SystemExit(f"scp failed from {node}: {files}")
        got += [f"rank_{node}/{f}" for f in files if str(f).endswith(".npz")]
        rc2, _ = _scp_from(node, f"{NODE_OUTDIR}/*.json", rank_dir)
        if rc2 != 0:
            print(f"[warn] json pull from {node} failed", flush=True)
    print(f"[{time.strftime('%T')}] pulled {len(got)} npz for {label}", flush=True)
    return got


def merge_all(local_dir: str) -> dict:
    """Merge rank_studio1/<f> + rank_studio2/<f> -> <f>.merged.npz in local_dir.

    scp is byte-exact; the merge splices the two logits vocab halves and
    requires every other key to be bitwise equal across ranks.  Any problem is
    reported loudly (the analyzer will not trust a mismatched pair)."""
    s1 = os.path.join(local_dir, "rank_studio1")
    s2 = os.path.join(local_dir, "rank_studio2")
    problems: dict[str, list[str]] = {}
    merged: list[str] = []
    names = sorted(set(os.listdir(s1)) & set(os.listdir(s2))) if (
        os.path.isdir(s1) and os.path.isdir(s2)) else []
    for name in names:
        if not name.endswith(".npz"):
            continue
        out_path = os.path.join(local_dir, name.replace(".npz", ".merged.npz"))
        try:
            res = merge_rank_pair(os.path.join(s1, name), os.path.join(s2, name),
                                  out_path)
        except Exception as exc:  # noqa: BLE001
            problems[name] = [f"merge raised: {exc!r}"]
            continue
        if res["problems"]:
            problems[name] = res["problems"]
        else:
            merged.append(name)
    print(f"[{time.strftime('%T')}] merged {len(merged)} rank pairs; "
          f"{len(problems)} with problems", flush=True)
    return {"merged": merged, "problems": problems}


def _load(path: str) -> dict:
    with np.load(path) as z:
        return {k: z[k] for k in z.files}


def merge_rank_pair(path_a: str, path_b: str, out_path: str) -> dict:
    """Merge two ranks' npz bit-exact: logits spliced; all else must be equal."""
    a, b = _load(path_a), _load(path_b)
    keys = set(a) & set(b)
    problems: list[str] = []
    merged = {}
    for k in keys:
        if k == "logits":
            continue
        va, vb = a[k], b[k]
        if va.dtype.kind in "fiu" and va.shape == vb.shape:
            if not np.array_equal(va, vb):
                problems.append(f"key {k} differs across ranks")
        elif va.dtype.kind in "US":
            if str(va) != str(vb):
                problems.append(f"key {k} differs across ranks")
        merged[k] = va
    la, lb = a["logits"], b["logits"]
    if la.shape == lb.shape:
        both = np.where(np.isnan(la), lb, la)
        n_overlap = int(np.sum(~np.isnan(la) & ~np.isnan(lb)))
        if n_overlap:
            problems.append(f"logits vocab slices overlap on {n_overlap} columns")
        merged["logits"] = both.astype(np.float32)
    else:
        problems.append(f"logits shapes differ: {la.shape} vs {lb.shape}")
    with open(out_path + ".tmp", "wb") as fh:
        np.savez(fh, **merged)
    os.replace(out_path + ".tmp", out_path)
    return {"problems": problems}


# ---------------------------------------------------------------------------
# analysis / gates (pure numpy)
# ---------------------------------------------------------------------------
def _row(d: dict, key: str, r0: int = 0) -> np.ndarray:
    return np.asarray(d[key]).reshape(np.asarray(d["M"]).item() if np.asarray(d["M"]).ndim else -1, -1)[r0]


def analyze_captures(capdir: str) -> dict:
    """The full read-out over merged npz files in ``capdir``.

    Prefers ``*.merged.npz`` (both ranks spliced); falls back to plain npz
    (single-rank dumps, e.g. an offline synthetic run)."""
    all_npz = sorted(f for f in os.listdir(capdir) if f.endswith(".npz")
                     and not f.startswith("._"))
    files = [f for f in all_npz if f.endswith(".merged.npz")] or all_npz
    by_tag: dict[str, str] = {}
    for f in files:
        m = re.match(r"(?P<arm>.+)_M(?P<M>\d+)_(?P<tag>.+?)(\.merged)?\.npz$", f)
        if m:
            by_tag.setdefault(m.group("tag"), os.path.join(capdir, f))
    out: dict = {"noise_floor": None, "shape_readout": {}, "rollback_ab": {},
                 "allsum": {}, "fence": {}, "routing_identity": {}}

    def _get(tag):
        return _load(by_tag[tag]) if tag in by_tag else None

    seq = [_get(f"s{i}") for i in range(6)]
    for M in (1, 3, 4, 5, 6):
        d = _get(f"M{M}")
        if d is None:
            continue
        rows = int(np.asarray(d["M"]).item())
        entry = {"max_abs_delta_logits_row0_vs_M1": None,
                 "max_abs_delta_logits_per_row_vs_seq": None,
                 "routing_identical_row0": None, "min_margin_at_changed": None}
        if seq[0] is not None:
            l0_m = np.asarray(d["logits"])[0]
            l0_1 = np.asarray(seq[0]["logits"])[0]
            finite = np.isfinite(l0_m) & np.isfinite(l0_1)
            entry["max_abs_delta_logits_row0_vs_M1"] = float(
                np.max(np.abs(l0_m[finite] - l0_1[finite])))
        ds = []
        for r in range(min(rows, len(seq))):
            if seq[r] is None:
                continue
            lm = np.asarray(d["logits"])[r]
            l1 = np.asarray(seq[r]["logits"])[0]
            finite = np.isfinite(lm) & np.isfinite(l1)
            ds.append(float(np.max(np.abs(lm[finite] - l1[finite]))))
        entry["max_abs_delta_logits_per_row_vs_seq"] = max(ds) if ds else None
        # routing identity: every captured layer's ids for row 0 vs M=1 chain
        same = None
        if seq[0] is not None:
            same = True
            for k in d:
                if k.startswith("idx_") and k in seq[0]:
                    if not np.array_equal(np.asarray(d[k])[0], np.asarray(seq[0][k])[0]):
                        same = False
                        break
        entry["routing_identical_row0"] = same
        out["shape_readout"][f"M{M}"] = entry
    rep = _get("M4rep2")
    if rep is not None and _get("M4") is not None:
        a = np.asarray(_get("M4")["logits"])
        b = np.asarray(rep["logits"])
        f = np.isfinite(a) & np.isfinite(b)
        out["noise_floor"] = {"M": 4, "max_abs_delta": float(np.max(np.abs(a[f] - b[f])))}
    for gm in (2, 3, 4, 5):
        fa, fb = _get(f"g{gm}_A_m1"), _get(f"g{gm}_B_m1")
        fwd = _get(f"g{gm}_fwd")
        if fa is None or fb is None:
            continue
        la, lb = np.asarray(fa["logits"])[0], np.asarray(fb["logits"])[0]
        f = np.isfinite(la) & np.isfinite(lb)
        rec = {"gamma": gm, "max_abs_delta_A_vs_B": float(np.max(np.abs(la[f] - lb[f])))}
        if fwd is not None:
            # accepted count is not recorded; report the verify row ids so the
            # analyzer can pair with the multi-step captures.
            rec["verify_rows"] = np.asarray(fwd["rows_ids"]).astype(int).tolist()
        out["rollback_ab"][f"g{gm}"] = rec
    for nm, tag0, tag1 in (("allsum", "M1", "M1fp32"), ("allsum_M4", "M4", "M4fp32"),
                           ("fence", "M1", "M1fence0"), ("fence_M4", "M4", "M4fence0")):
        a, b = _get(tag0), _get(tag1)
        if a is None or b is None:
            continue
        la, lb = np.asarray(a["logits"]), np.asarray(b["logits"])
        f = np.isfinite(la) & np.isfinite(lb)
        out.setdefault("arms", {})[nm] = {"max_abs_delta": float(np.max(np.abs(la[f] - lb[f])))}
    return out


def faithfulness_gate(stream_recs: dict, recorded_dir: str) -> dict:
    """gamma3 must equal arm_gamma3a.json; gamma4 must equal arm_gamma4.json."""
    res: dict = {"checks": [], "verdict": "UNKNOWN"}
    for arm_name, rec_arm in (("g3", "gamma3a"), ("g4", "gamma4")):
        rec = stream_recs.get(arm_name)
        if rec is None:
            res["checks"].append({"arm": arm_name, "status": "MISSING_STREAM"})
            continue
        path = os.path.join(recorded_dir, f"arm_{rec_arm}.json")
        if not os.path.exists(path):
            res["checks"].append({"arm": arm_name, "status": "NO_RECORDED",
                                  "recorded": path})
            continue
        with open(path, encoding="utf-8") as fh:
            recorded = json.load(fh)
        agentic = (recorded.get("agentic") or [{}])[0]
        rt, rr = recorded.get("agentic", []), rec
        r_tokens = agentic.get("tokens")
        r_top2 = agentic.get("top2_logprobs")
        div = next((i for i, (a, b) in enumerate(zip(rec["tokens"], r_tokens or []))
                    if a != b), None)
        lp_equal = (rec["top2_logprobs"] == r_top2)
        res["checks"].append({
            "arm": arm_name, "recorded": rec_arm,
            "n_tokens": len(rec["tokens"]),
            "first_divergence_vs_recorded": div,
            "top2_logprobs_exact": bool(lp_equal),
            "status": ("EXACT" if (div is None and lp_equal) else "MISMATCH"),
        })
    st = [c.get("status") for c in res["checks"]]
    if all(s == "EXACT" for s in st) and st:
        res["verdict"] = "EXACT (dumps trustworthy)"
    elif any(s == "MISMATCH" for s in st):
        res["verdict"] = ("*** MISMATCH: the instrumented build does NOT reproduce "
                          "the recorded round; dumps are UNTRUSTWORTHY ***")
    else:
        res["verdict"] = "INCOMPLETE (recorded artifacts or streams missing)"
    return res


# ---------------------------------------------------------------------------
# per-arm orchestration
# ---------------------------------------------------------------------------
def run_arm(gamma: int, arm: str, *, outdir: str, registry: str, sweep: bool,
            ms: int, rollback: bool, rounds: int, max_tokens_stream: int,
            include_multi_arm: bool, dry: bool) -> dict:
    rec: dict = {"arm": arm, "gamma": gamma, "captures": {}, "streams": {}}
    if dry:
        print(f"[dry] arm {arm} (gamma={gamma}): sweep={sweep} ms={ms} "
              f"rollback={rollback} rounds={rounds}")
        return rec
    prompt_sweep, _ = build_agentic_prompt(SALT_SWEEP)
    prompt_stream, meta_stream = build_agentic_prompt(SALT_STREAM)
    rec["prompt_meta"] = meta_stream
    if sweep:
        cid = f"q2fu1;arm={arm};sweep=1;ms=0;ksweep=1,3,4,5,6"
        print(f"[{time.strftime('%T')}] {arm}: shape-sweep request ({cid})", flush=True)
        ok, detail = ensure_loaded(registry_path=registry, label=f"{arm}_sweep")
        rec["sweep"] = stream_request(prompt_sweep, cid, max_tokens=8,
                                      registry_path=registry, label=f"{arm}_sweep",
                                      capture_positions=8)
        rec["sweep"].pop("stats", None)
        print(f"[{time.strftime('%T')}] {arm}: sweep request done "
              f"({rec['sweep']['wall_s']}s)", flush=True)
    if include_multi_arm:
        cid = (f"q2fu1;arm={arm};sweep=0;ms={ms};rollback={1 if rollback else 0};"
               f"rounds={rounds}")
        print(f"[{time.strftime('%T')}] {arm}: multi-step request ({cid})", flush=True)
        rec["streams"][arm] = stream_request(prompt_stream, cid,
                                             max_tokens=max_tokens_stream,
                                             registry_path=registry, label=f"{arm}_ms")
        print(f"[{time.strftime('%T')}] {arm}: multi-step done, "
              f"{len(rec['streams'][arm]['tokens'])} logprob positions", flush=True)
    if include_multi_arm:
        cid = f"q2fu1;arm={arm}n;sweep=0;ms=0;nospec=1"
        print(f"[{time.strftime('%T')}] {arm}n: nospec reference request ({cid})",
              flush=True)
        rec["streams"][arm + "n"] = stream_request(
            prompt_stream, cid, max_tokens=max_tokens_stream,
            registry_path=registry, label=f"{arm}n_nospec")
    return rec


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(prog="q2fu1_driver")
    ap.add_argument("--outdir", default=DEFAULT_OUTDIR)
    ap.add_argument("--dry-run", action="store_true")
    ap.add_argument("--arms", default="g3,g4",
                    help="comma list from {g2,g3,g4,g5} (default g3,g4)")
    ap.add_argument("--ms", type=int, default=24, help="passive verify rounds to capture")
    ap.add_argument("--rounds", type=int, default=8, help="rollback-arm rounds cap")
    ap.add_argument("--rollback", type=int, default=1)
    ap.add_argument("--skip-switch", action="store_true",
                    help="do not sed/relaunch (the boot is already at the arm)")
    ap.add_argument("--skip-faithfulness", action="store_true")
    ap.add_argument("--max-tokens-stream", type=int, default=256)
    a = ap.parse_args(argv)

    arms = [(x.strip(), int(x.strip()[1:])) for x in a.arms.split(",") if x.strip()]
    capdir = os.path.join(a.outdir, "captures")
    os.makedirs(capdir, exist_ok=True)
    registry = os.path.join(a.outdir, "own_requests.jsonl")

    print("=" * 78)
    print("Round Q2-FU1 capture driver")
    print(f"  arms={arms} ms={a.ms} rounds={a.rounds} rollback={a.rollback}")
    print(f"  API {API_BASE}  nodes {NODES}  node outdir {NODE_OUTDIR}")
    print(f"  outdir {a.outdir}")
    print("=" * 78)

    if a.dry_run:
        for arm, gamma in arms:
            if not a.skip_switch:
                switch_arm(gamma, arm=arm, registry_path=registry, dry=True)
            run_arm(gamma, arm, outdir=a.outdir, registry=registry, sweep=True,
                    ms=a.ms, rollback=bool(a.rollback), rounds=a.rounds,
                    max_tokens_stream=a.max_tokens_stream, include_multi_arm=True,
                    dry=True)
        print("[dry] scp both nodes' /tmp/q2fu1/out/*.npz -> captures/; merge; "
              "analyze_captures; faithfulness_gate vs "
              f"{RECORDED_DIR}; write q2fu1_report.md")
        print("[dry] done -- no cluster interaction performed.")
        return 0

    confirm_hook_active()
    records = {}
    for arm, gamma in arms:
        print(f"\n[{time.strftime('%T')}] === arm {arm} (gamma={gamma}) ===", flush=True)
        if not a.skip_switch:
            switch_arm(gamma, arm=arm, registry_path=registry)
        wait_ready(warmup_ok=True)
        confirm_rank_consistency(gamma)
        records[arm] = run_arm(gamma, arm, outdir=a.outdir, registry=registry,
                               sweep=True, ms=a.ms, rollback=bool(a.rollback),
                               rounds=a.rounds, max_tokens_stream=a.max_tokens_stream,
                               include_multi_arm=True, dry=False)
        # pull immediately after each arm (bounded blast radius)
        pull_captures(capdir, label=arm)

    # merge each rank pair (bitwise-equal non-logit keys; NaN-spliced logits)
    mres = merge_all(capdir)
    if mres["problems"]:
        print("*** RANK-MERGE PROBLEMS (non-bitwise-identical across ranks):")
        print(json.dumps(mres["problems"], indent=1))

    analysis = analyze_captures(capdir)
    fgate = {"verdict": "SKIPPED"}
    if not a.skip_faithfulness:
        streams = {}
        for arm, rec in records.items():
            for k, v in rec.get("streams", {}).items():
                streams[k] = v
        fgate = faithfulness_gate(streams, RECORDED_DIR)
    report = {"analysis": analysis, "faithfulness": fgate,
              "arms": {k: {kk: vv for kk, vv in v.items() if kk != "streams"}
                       for k, v in records.items()},
              "merge": {"merged": mres["merged"], "problems": mres["problems"]}}
    with open(os.path.join(a.outdir, "q2fu1_report.json"), "w", encoding="utf-8") as fh:
        json.dump(report, fh, indent=1, default=str)
    md = ["# Q2-FU1 capture report", "",
          f"arms: {[x for x, _ in arms]}  ms={a.ms}  rounds={a.rounds} "
          f"rollback={a.rollback}", "",
          f"## Faithfulness — {fgate.get('verdict')}", ""]
    for c in fgate.get("checks", []):
        md.append(f"- {c}")
    md += ["", "## Shape read-out (row0 vs M=1; routing identity)", ""]
    for k, v in analysis.get("shape_readout", {}).items():
        md.append(f"- {k}: {json.dumps(v)}")
    md += ["", f"## Noise floor: {json.dumps(analysis.get('noise_floor'))}", "",
           "## Rollback A/B (production vs +ring-restore)", ""]
    for k, v in analysis.get("rollback_ab", {}).items():
        md.append(f"- {k}: {json.dumps(v)}")
    md += ["", f"## Other arms: {json.dumps(analysis.get('arms', {}))}", "",
           f"## Merge: {len(mres['merged'])} pairs ok, "
           f"{len(mres['problems'])} with problems", ""]
    with open(os.path.join(a.outdir, "q2fu1_report.md"), "w", encoding="utf-8") as fh:
        fh.write("\n".join(md) + "\n")
    print("\n=== FAITHFULNESS ===\n" + json.dumps(fgate, indent=1))
    print("\n=== SHAPE READ-OUT ===\n" + json.dumps(analysis.get("shape_readout"),
                                                    indent=1))
    print("\n=== NOISE FLOOR ===\n" + json.dumps(analysis.get("noise_floor")))
    print("\n=== ROLLBACK A/B ===\n" + json.dumps(analysis.get("rollback_ab"), indent=1))
    print(f"\nreport -> {os.path.join(a.outdir, 'q2fu1_report.json')}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
