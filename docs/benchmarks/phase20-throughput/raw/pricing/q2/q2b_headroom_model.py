#!/usr/bin/env python3
"""q2b: peak-memory-vs-context model for the SHIPPED dsv41 serve, and the context N* where it crosses W.

DESK ONLY, stdlib only, no network, no cluster.  Reads ONLY the raw files committed next to this script:
  q2b_raw_vm_instances_dsv41.json       VM exo_peak_memory_bytes up-steps (+ prompt-token / hit-kind deltas), 104 instances
  q2b_raw_studio1_exo_log_excerpts.txt  studio1 exo.log (current boot): `prefill controls ... rows=` per request
  q2b_raw_vm_sys_series.json            exo_memory_ram_used_bytes / swap (system) for the current boot
  q2b_raw_vm_ram_soak13_window.json     same, soak13 window (2026-10-07)
  q2b_raw_node_studio1_vmmap_footprint.txt   studio1 runner `footprint -f bytes` (exact bytes)
  q2b_raw_state_capacity.json           GET /state capacity fields
Writes q2b-memory-headroom.json and q2b-memory-headroom.md next to it (the .md is generated from the same
numbers, so the document cannot drift from the model).   Run:  python3 q2b_headroom_model.py

UNITS (the point of this round):
  * VM gauge exo_peak_memory_bytes = true_bytes * (1024**3/1e9): dsv41/engine.py:284 + rounds.py:227 store
    Memory.from_gb(mx.get_peak_memory()/1e9) and Memory.from_gb(v) = round(v*1024**3).  true = labelled / K.
  * `footprint` default output is BINARY (MiB/GiB printed as MB/GB); `-f bytes` / libproc are exact bytes.
  * Everything is DECIMAL GB (1e9 B) unless suffixed GiB.  W = 120000 MiB = 125.82912 GB.
"""

from __future__ import annotations

import datetime as dt
import json
import math
import pathlib
import re

HERE = pathlib.Path(__file__).resolve().parent
CDT = dt.timezone(dt.timedelta(hours=-5))
K = 1024**3 / 1e9                        # 1.073741824  gauge inflation (executed in q2b_memory_unit_probe.py)
W = 120000 * 1048576 / 1e9               # iogpu.wired_limit_mb=120000 on BOTH nodes (live)  -> 125.82912 GB
PHYS = 137_438_953_472 / 1e9             # hw.memsize
GC = 0.95 * W                            # mlx metal allocator gc_limit_ = min(0.95*max_recommended_working_set, block_limit_)
CAP = 1_048_576                          # live GET /state maxKvTokens
KV_B = 3200                              # comp_kv 2560 + index_k 640 B per capacity-row (q2b_kv_per_token_probe.out.txt)
TAPS_B = 3 * 5120 * 2                    # 3 DSpark tap layers x hidden 5120 x bf16 (bf16 CONFIRMED by the free fit below)
WEIGHTS = 104.7                          # loader line active=104.7 GB (mx.get_active_memory()/1e9, both ranks)
HERMES_COMPRESS_AT = int(0.7 * CAP)      # ~/.hermes/config.yaml compression.threshold 0.7 x context_length 1,048,576
MAX_SEEN_PROMPT = 126_527                # state.db: 96 real calls to this model, max prompt_tokens_total


# ----------------------------------------------------------------------------------------------- tiny OLS
def ols(X: list[list[float]], y: list[float]):
    n, p = len(X), len(X[0])
    A = [[sum(X[k][i] * X[k][j] for k in range(n)) for j in range(p)] for i in range(p)]
    b = [sum(X[k][i] * y[k] for k in range(n)) for i in range(p)]
    M = [row[:] + [1.0 if i == j else 0.0 for j in range(p)] for i, row in enumerate(A)]
    for c in range(p):
        piv = max(range(c, p), key=lambda r: abs(M[r][c]))
        M[c], M[piv] = M[piv], M[c]
        d = M[c][c]
        M[c] = [v / d for v in M[c]]
        for r in range(p):
            if r != c:
                f = M[r][c]
                M[r] = [a - f * bb for a, bb in zip(M[r], M[c])]
    inv = [row[p:] for row in M]
    beta = [sum(inv[i][j] * b[j] for j in range(p)) for i in range(p)]
    res = [y[k] - sum(X[k][i] * beta[i] for i in range(p)) for k in range(n)]
    sigma = math.sqrt(sum(r * r for r in res) / max(n - p, 1))
    se = [math.sqrt(max(inv[i][i], 0.0)) * sigma for i in range(p)]
    return beta, res, sigma, se


# ----------------------------------------------------------------------------------------------- capacity model
def ensure(cap: int, req: int, max_seq: int = CAP) -> int:
    """ModelCache.ensure_capacity (mlx-lm cache.py:188): new_cap = min(max(req, 2*cap), max_seq) when req > cap."""
    return cap if req <= cap else min(max(req, cap * 2), max_seq)


def cap_cold(n: int) -> int:
    """Fresh ModelCache (initial capacity 65,536) then ONE up-front ensure_capacity(offset+delta) (SessionCache.append_turn)."""
    return ensure(65536, n)


# ----------------------------------------------------------------------------------------------- raw loading
def load_vm() -> dict:
    return json.loads((HERE / "q2b_raw_vm_instances_dsv41.json").read_text())


def vm_step(vm: dict, iid: str, t_prefix: str) -> dict:
    for inst in vm["instances"]:
        if inst["instance_id"].startswith(iid):
            for s in inst["up_steps"]:
                if s["t"].startswith(t_prefix):
                    return s
    raise KeyError((iid, t_prefix))


def excerpt_rows() -> list[tuple[dt.datetime, int]]:
    out = []
    for ln in (HERE / "q2b_raw_studio1_exo_log_excerpts.txt").read_text().splitlines():
        m = re.search(r"\[ (\d{4}-\d\d-\d\d \d\d:\d\d:\d\d)\.\d+ .*prefill controls: .*\(rows=(\d+),", ln)
        if m:
            out.append((dt.datetime.strptime(m.group(1), "%Y-%m-%d %H:%M:%S"), int(m.group(2))))
    return out


def node_footprint() -> dict[str, float]:
    txt = (HERE / "q2b_raw_node_studio1_vmmap_footprint.txt").read_text()
    g = lambda pat: int(re.search(pat, txt).group(1))  # noqa: E731
    now = g(r"phys_footprint:\s*(\d+) B")
    peak = g(r"phys_footprint_peak:\s*(\d+) B")
    ioa = g(r"(\d+) B\s+0 B\s+0 B\s+\d+\s+IOAccelerator \(graphics\)")
    return dict(now=now / 1e9, peak=peak / 1e9, ioaccel=ioa / 1e9, host=(now - ioa) / 1e9)


# ----------------------------------------------------------------------------------------------- observations
def build_obs(vm: dict):
    c160 = cap_cold(160_006)
    c500 = ensure(c160, 499_992)
    c750 = ensure(c500, 749_994)
    c1m = ensure(c750, 1_039_974)
    # id, instance, step-time prefix, N (rows in cache at the end), D (rows PREFILLED by that request), capacity, note
    spec = [
        ("cur_20K_a", "7aa2dbd3", "2026-10-09T23:09", 20_076, 20_076, cap_cold(20_076), "current boot (next19), cold"),
        ("cur_20K_b", "7aa2dbd3", "2026-10-09T23:12", 20_076, 20_076, cap_cold(20_076), "current boot, cold (3rd 20K)"),
        ("cur_18K", "7aa2dbd3", "2026-10-10T00:29", 18_387, 18_387, cap_cold(18_387), "current boot, cold"),
        ("cur_45K", "7aa2dbd3", "2026-10-10T00:33", 45_702, 45_702, cap_cold(45_702), "current boot, cold"),
        ("cur_100K", "7aa2dbd3", "2026-10-10T00:40", 100_497, 100_497, cap_cold(100_497),
         "current boot, cold = THE request behind A2's 'peak at ~102K'"),
        ("n13_100K", "38ed8ddb", "2026-10-07T04:16", 100_012, 100_012, cap_cold(100_012), "soak13 (next13), cold"),
        ("n13_160K", "38ed8ddb", "2026-10-07T04:26", 160_006, 160_006, c160, "soak13, cold (r160)"),
        ("n13_r500", "38ed8ddb", "2026-10-07T04:57", 499_992, 340_248, c500, "soak13 delta 340,248 on 159,744 reused (r500)"),
        ("n13_r750", "38ed8ddb", "2026-10-07T07:08", 749_994, 250_282, c750, "soak13 delta 250,282 on 499,712 reused (r750)"),
        ("n13_r1m", "38ed8ddb", "2026-10-07T07:48", 1_039_974, 290_406, c1m, "soak13 delta 290,406 on 749,568 reused (r1m)"),
    ]
    rows = excerpt_rows()
    fit = []
    for oid, iid, tp, N, D, cap, note in spec:
        s = vm_step(vm, iid, tp)
        pt = s["prompt_tokens_in_scrape_window"]
        assert pt is None or abs(pt - N) <= 2, (oid, pt, N)          # VM prompt-token counter delta must equal N
        if iid == "7aa2dbd3":                                         # current boot: log must show the same rows= line
            t = dt.datetime.fromisoformat(s["t"])
            assert any(abs((t - rt).total_seconds()) < 600 and r == D for rt, r in rows), (oid, D)
        fit.append(dict(id=oid, N=N, D=D, cap=cap, y=s["labelled_B"] / 1e9 / K, labelled=s["labelled_B"] / 1e9, t=s["t"], note=note))
    # validation (NOT fitted): cold / hit-kind 'none' requests from other instances & builds
    vspec = [
        ("n18 b8d90d57 188K", "b8d90d57", "2026-10-09T02:17", 188_261, "next18 (shipped lineage), cold"),
        ("n13 75fa21d8 160K", "75fa21d8", "2026-10-07T08:19", 160_007, "next13-era other instance, cold"),
        ("n17 49984332 91K", "49984332", "2026-10-08T09:26", 91_043, "next17 (lever-1 era), cold"),
        ("old 247fb3c8 160K", "247fb3c8", "2026-10-06T12:30", 159_995, "10-06 build, cold"),
        ("old d301ea85 350K", "d301ea85", "2026-10-06T04:56", 350_124, "10-06 build, cold"),
        ("old 5689f0ba 350K", "5689f0ba", "2026-10-06T11:40:01", 350_124, "10-06 build, cold"),
        ("old 30b396d2 350K", "30b396d2", "2026-10-07T01:21:09", 350_124, "10-07 build, cold"),
    ]
    val = []
    for name, iid, tp, N, note in vspec:
        s = vm_step(vm, iid, tp)
        assert (s["hit_kind_increments"] or {}).get("none", 0) >= 1, (name, s)   # cold only
        val.append(dict(name=name, N=N, D=N, cap=cap_cold(N), y=s["labelled_B"] / 1e9 / K, note=note))
    env = []   # upper-envelope reference only (pre-bf16-row / pre-M2 builds on 2026-10-04)
    for name, iid, tp, N in (("OLD 867ef937 cold 500K", "867ef937", "2026-10-04T14:45", 499_981),
                             ("OLD 867ef937 cold 750K", "867ef937", "2026-10-04T17:33", 749_983)):
        s = vm_step(vm, iid, tp)
        env.append(dict(name=name, N=N, y=s["labelled_B"] / 1e9 / K))
    return fit, val, env


def hhmm(a: str) -> dt.datetime:
    return dt.datetime.strptime(a, "%Y-%m-%d %H:%M:%S").replace(tzinfo=CDT)


def ram_windows() -> dict:
    """System RAM-used (ram_total - ram_available) maxima per event window, from the committed raw series."""
    sysd = json.loads((HERE / "q2b_raw_vm_sys_series.json").read_text())["series"]
    soak = json.loads((HERE / "q2b_raw_vm_ram_soak13_window.json").read_text())["series"]

    def cur(inst: str, node_prefix: str):
        for r in sysd[f"exo_memory_ram_used_bytes|{inst}"]:
            if r["labels"].get("node_id", "").startswith(node_prefix):
                return [(t / 1000.0, v) for t, v in zip(r["timestamps_ms"], r["values"]) if v is not None]
        raise KeyError(inst)

    c1, c2 = cur("macstudio-m4-1", "f8754a0e"), cur("macstudio-m4-2", "e662cd77")

    def cur_swap(inst: str, node_prefix: str):
        for r in sysd[f"exo_memory_swap_used_bytes|{inst}"]:
            if r["labels"].get("node_id", "").startswith(node_prefix):
                return [(t / 1000.0, v) for t, v in zip(r["timestamps_ms"], r["values"]) if v is not None]
        raise KeyError(inst)

    w1, w2 = cur_swap("macstudio-m4-1", "f8754a0e"), cur_swap("macstudio-m4-2", "e662cd77")

    def mx(series, a: str, b: str) -> float:
        ta, tb = hhmm(a).timestamp(), hhmm(b).timestamp()
        return max(v for t, v in series if ta <= t <= tb) / 1e9

    cur_w = [("cold 20K (23:07:30)", "2026-10-09 23:07:30", "2026-10-09 23:09:00"),
             ("cold 20K (23:10:31)", "2026-10-09 23:10:31", "2026-10-09 23:12:00"),
             ("cold 18K", "2026-10-10 00:28:41", "2026-10-10 00:29:46"),
             ("cold 45.7K", "2026-10-10 00:30:35", "2026-10-10 00:33:18"),
             ("cold 100.5K", "2026-10-10 00:34:10", "2026-10-10 00:41:06"),
             ("idle 01:30-02:00", "2026-10-10 01:30:00", "2026-10-10 02:00:00")]
    soak_w = [("100K cold", "2026-10-07 04:09:00", "2026-10-07 04:17:00"),
              ("160K cold", "2026-10-07 04:19:00", "2026-10-07 04:27:00"),
              ("r500 delta (to 500K)", "2026-10-07 04:26:30", "2026-10-07 04:58:00"),
              ("r750 delta (to 750K)", "2026-10-07 06:38:00", "2026-10-07 07:10:00"),
              ("r1m delta (to 1.04M)", "2026-10-07 07:08:00", "2026-10-07 07:50:00")]
    s1 = [(t, v) for t, v in soak["macstudio-m4-1"]]
    s2 = [(t, v) for t, v in soak["macstudio-m4-2"]]
    sw = json.loads((HERE / "q2b_raw_vm_ram_soak13_window.json").read_text())["swap"]
    sw1 = [(t, v) for t, v in sw["macstudio-m4-1"]]
    sw2 = [(t, v) for t, v in sw["macstudio-m4-2"]]
    def mx_sum(ram_s, swap_s, a: str, b: str) -> float:
        """max over CONCURRENT samples (same scrape timestamp) of ram_used + swap_used."""
        ta, tb = hhmm(a).timestamp(), hhmm(b).timestamp()
        sw_by_t = {t: v for t, v in swap_s}
        return max(v + sw_by_t[t] for t, v in ram_s if ta <= t <= tb and t in sw_by_t) / 1e9

    return dict(
        current=[(n, mx(c1, a, b), mx(c2, a, b), mx(w1, a, b), mx(w2, a, b)) for n, a, b in cur_w],
        soak13=[(n, mx(s1, a, b), mx(s2, a, b), mx(sw1, a, b), mx(sw2, a, b)) for n, a, b in soak_w],
        soak13_ram_plus_swap=[(n, mx_sum(s1, sw1, a, b), mx_sum(s2, sw2, a, b)) for n, a, b in soak_w])


# ----------------------------------------------------------------------------------------------- main
def main() -> int:
    vm = load_vm()
    fit, val, env = build_obs(vm)
    fp = node_footprint()
    cap_state = json.loads((HERE / "q2b_raw_state_capacity.json").read_text())
    live_cap = next(iter(cap_state["instances"].values()))["maxKvTokens"]
    assert live_cap == CAP, live_cap

    # ---- 2-parameter fit: y = true_peak - KV(cap) - taps(D) = A + t*N        (t in GB per Mtok == KB per token)
    X = [[1.0, o["N"] / 1e6] for o in fit]
    yv = [o["y"] - KV_B * 1e-9 * o["cap"] - TAPS_B * 1e-9 * o["D"] for o in fit]
    (A, t_gbm), res, sigma, se = ols(X, yv)
    t_B = t_gbm * 1e3                                   # bytes per token
    def pred(N: int, D: int, cap: int) -> float:
        return A + t_gbm * N / 1e6 + KV_B * 1e-9 * cap + TAPS_B * 1e-9 * D

    # ---- 3-parameter FREE fit (checks the taps mechanism independently): y3 = true - KV(cap) = A + cD*D + cN*N
    y3 = [o["y"] - KV_B * 1e-9 * o["cap"] for o in fit]
    X3 = [[1.0, o["D"] / 1e6, o["N"] / 1e6] for o in fit]
    (A3, cD, cN), res3, sig3, se3 = ols(X3, y3)
    cD_B, cN_B, seD_B, seN_B = cD * 1e3, cN * 1e3, se3[1] * 1e3, se3[2] * 1e3

    def solve(target: float, D_of_N, cap_of_N, extra: float = 0.0, hi: int = 6_000_000, A_=None, t_=None):
        """Smallest N (tokens) with pred(N, D(N), cap(N)) + extra >= target; None if not reached by `hi`."""
        A_ = A if A_ is None else A_
        t_ = t_gbm if t_ is None else t_
        f = lambda N: A_ + t_ * N / 1e6 + KV_B * 1e-9 * cap_of_N(N) + TAPS_B * 1e-9 * D_of_N(N) + extra  # noqa: E731
        if f(hi) < target:
            return None
        lo_, hi_ = 1_000.0, float(hi)
        for _ in range(70):
            m = (lo_ + hi_) / 2
            if f(int(m)) >= target:
                hi_ = m
            else:
                lo_ = m
        return hi_

    cold = (lambda N: N, cap_cold)
    warm = lambda D: ((lambda N: D), (lambda N: min(CAP, max(cap_cold(N), N))))   # noqa: E731
    N_a = solve(W, *cold)
    N_dbl = solve(W, lambda N: N, lambda N: min(CAP, 2 * N))        # cold, IF capacity were rounded up 2x (worst case)
    slope_cold_B = TAPS_B + KV_B + t_B
    slope_warm_B = KV_B + t_B

    # N* if the FREE fit (cD, cN fitted separately) is used instead of the analytic-taps model -> sensitivity
    def pred_free(N: int, D: int, cap: int) -> float:
        return A3 + cD * D / 1e6 + cN * N / 1e6 + KV_B * 1e-9 * cap
    lo_, hi_ = 1_000.0, 6_000_000.0
    for _ in range(70):
        m_ = (lo_ + hi_) / 2
        if pred_free(int(m_), int(m_), cap_cold(int(m_))) >= W:
            hi_ = m_
        else:
            lo_ = m_
    N_free = hi_

    # leave-one-out
    loo = []
    for i in range(len(fit)):
        (Ai, ti), _, _, _ = ols([r for k, r in enumerate(X) if k != i], [v for k, v in enumerate(yv) if k != i])
        loo.append(solve(W, *cold, A_=Ai, t_=ti))

    # validation residuals (measured - predicted) -> model-error band (EMPIRICAL, from unfitted points)
    vres = []
    for v in val:
        p = pred(v["N"], v["D"], v["cap"])
        vres.append(dict(name=v["name"], N=v["N"], measured=round(v["y"], 3), predicted=round(p, 3), err=round(v["y"] - p, 3), note=v["note"]))
    e_lo, e_hi = min(r["err"] for r in vres), max(r["err"] for r in vres)
    N_a_band = (solve(W, *cold, extra=e_hi), solve(W, *cold, extra=e_lo))      # measured can be higher (e_hi) -> earlier crossing
    two_anchor = []
    anchor = fit[4]["y"]
    for v in val:
        if v["N"] == 350_124:
            s = (v["y"] - anchor) / (v["N"] - 100_497)
            two_anchor.append(dict(name=v["name"], slope_B_per_row=round(s * 1e9, 0), N_star=round(100_497 + (W - anchor) / s)))

    # ---- resident (OS phys_footprint) domain: footprint = active + cache + host(H);  cache -> 0 once active > gc_limit
    H_lo, H_hi = fp["host"], fp["peak"] - GC                       # idle non-GPU part ; lifetime-max minus gc_limit
    N_gc = solve(GC, *cold)
    N_phys = solve(PHYS, *cold)
    N_r = (solve(W, *cold, extra=H_hi), solve(W, *cold, extra=H_lo))                       # (earlier, later)
    N_r_band = (solve(W, *cold, extra=H_hi + e_hi), solve(W, *cold, extra=H_lo + e_lo))

    # ---- scenarios (allocator domain, GB) ---------------------------------------------------------------
    Ns = [20_076, 100_497, 131_072, 200_000, 262_144, 300_000, 328_000, 400_000, 500_000, HERMES_COMPRESS_AT, 900_000, CAP]
    def curve(D_of_N, cap_of_N, second_cap: int = 0, taps_fixed: bool = False):
        out = []
        for N in Ns:
            D = 0 if taps_fixed else D_of_N(N)
            a = pred(N, D, cap_of_N(N)) + KV_B * 1e-9 * second_cap
            out.append(dict(N=N, alloc=a, vsW=a - W, fp_lo=max(a + H_lo, fp["peak"]) if a < GC else a + H_lo, fp_hi=max(a + H_hi, fp["peak"]) if a < GC else a + H_hi))
        return out
    dbl = (lambda N: max(cap_cold(N), min(CAP, 2 * N)))                        # capacity rounded UP 2x (upper bound; turn-by-turn growth gives 65536*2^k <= 2x)
    sc = {
        "cold prefill (as shipped)": curve(*cold),
        "warm turn, D=2,048 rows": curve(*warm(2048)),
        "warm turn D=2,048, capacity rounded up 2x (upper bound)": curve(lambda N: 2048, dbl),
        "warm turn, D=25,662 (largest state.db growth)": curve(*warm(25_662)),
        "warm turn D=2,048 + 2nd resident session at 400K": curve(*warm(2048), second_cap=400_000),
        "warm turn D=2,048 + 2nd resident session at cap": curve(*warm(2048), second_cap=CAP),
        "cold prefill IF taps retention were fixed (projection)": curve(*cold, taps_fixed=True),
    }
    # largest NEW rows a single warm turn may prefill at context N before the allocator peak crosses W (exact / doubled capacity)
    dstar = []
    for N in (200_000, 328_000, 400_000, 500_000, HERMES_COMPRESS_AT, 900_000, CAP):
        row_ = {"N": N}
        for lab, cap_ in (("cap_exact", max(cap_cold(N), N)), ("cap_doubled", dbl(N))):
            room = W - (A + t_gbm * N / 1e6 + KV_B * 1e-9 * min(cap_, CAP))
            row_[lab] = max(0, min(N, int(room / (TAPS_B * 1e-9))))
        dstar.append(row_)
    N_cold_fixed = solve(W, lambda N: 0, cap_cold)
    N_warm_2k = solve(W, *warm(2048), hi=4_000_000)
    N_warm_25k = solve(W, *warm(25_662), hi=4_000_000)
    N_prod_marker = {n: pred(n, n, cap_cold(n)) for n in (MAX_SEEN_PROMPT, 250_000)}

    # ---- A2 corrections ----------------------------------------------------------------------------------
    p_alloc = fit[4]["y"]
    p_res = fp["peak"]
    tr_alloc, tr_res = p_alloc - WEIGHTS, p_res - WEIGHTS
    head_q3 = W - 110.42
    q1e = {nm: dict(non_weight=round(nw, 2), demand=round(6.51 + nw, 2), headroom=round(head_q3, 2), over=round(6.51 + nw - head_q3, 2), over_with_margin=round(6.51 + nw + 2 - head_q3, 2))
           for nm, nw in (("allocator_corrected", tr_alloc), ("resident_corrected", tr_res), ("allocator_A2_reported", 20.133), ("resident_A2_reported", 9.3))}
    q1e["allocator_if_taps_fixed_projection"] = dict(non_weight=round(tr_alloc - TAPS_B * 1e-9 * 100_497, 2),
                                                     over=round(6.51 + tr_alloc - TAPS_B * 1e-9 * 100_497 - head_q3, 2))
    ram = ram_windows()

    def kfmt(n):
        return "n/a" if n is None else f"{n / 1000:.0f}K"

    sec_cold = sc["cold prefill (as shipped)"]
    at_cap_cold = sec_cold[-1]["alloc"]
    sec_warm = sc["warm turn, D=2,048 rows"]
    sec_warm2 = sc["warm turn D=2,048 + 2nd resident session at 400K"]
    sec_warm3 = sc["warm turn D=2,048 + 2nd resident session at cap"]

    breach_alloc = (f"~{kfmt(N_a)} tokens (cold single-request prefill; band {kfmt(N_a_band[0])}-{kfmt(N_a_band[1])} from unfitted-validation error "
                    f"{e_lo:+.1f}..{e_hi:+.1f} GB, LOO {kfmt(min(loo))}-{kfmt(max(loo))}); 2 of 3 measured cold-350K peaks already exceed W. "
                    f"Warm turns: no crossing inside the {CAP:,} cap ({sec_warm[-1]['alloc']:.1f} GB at cap, {sec_warm[-1]['vsW']:+.1f} GB vs W, inside the model error)")
    breach_res = (f"~{kfmt(N_r[0])}-{kfmt(N_r[1])} tokens (cold prefill; model band {kfmt(N_r_band[0])}-{kfmt(N_r_band[1])}; LOW confidence: "
                  f"one footprint data point, plateau pinned at {fp['peak']:.1f} GB until active passes gc_limit at ~{kfmt(N_gc)})")
    breaches = (f"YES, conditionally. The shipped cap ({CAP:,}) admits it: a single COLD prefill crosses W at ~{kfmt(N_a)} (allocator) / "
                f"~{kfmt(N_r[0])}-{kfmt(N_r[1])} (OS footprint) and extrapolates to {at_cap_cold:.0f} GB at the cap (> {PHYS:.0f} GB RAM; deep-cold UNVERIFIED). "
                f"Warm turns do not cross inside the cap but leave only {-sec_warm[-1]['vsW']:.1f} GB (less than the model error) and a second deep resident session removes it. "
                f"Not a demonstrated crash: past-W loads completed in soaks with OS compression/swap absorbing. Largest real prompt seen: {MAX_SEEN_PROMPT:,} (state.db, 96 calls).")

    xchk = []
    base_other = 119.371 - fp["now"]                     # non-runner system RAM on studio1: idle RAM-used (01:30-02:00) minus runner footprint-now
    for (nm, g_true), (_, rs1, rs2) in zip((("r500 delta", fit[7]["y"]), ("r750 delta", fit[8]["y"]), ("r1m delta", fit[9]["y"])), ram["soak13_ram_plus_swap"][2:5]):
        xchk.append(dict(rung=nm, gauge_true=g_true, gauge_labelled=g_true * K, ram_plus_swap_m4_1=rs1, ram_plus_swap_m4_2=rs2))
    out = dict(
        schema="q2b-memory-headroom/2", generated=dt.datetime.now(CDT).isoformat(timespec="seconds"),
        W_gb=round(W, 5), W_mib=120000, W_gib=round(W * 1e9 / 2**30, 3), phys_ram_gb=round(PHYS, 3), gc_limit_gb=round(GC, 3),
        weights_gb=WEIGHTS,
        peak_102k_resident_gb=round(p_res, 3), peak_102k_allocator_gb=round(p_alloc, 3),
        peak_102k_resident_gb_as_reported_A2=114.0, peak_102k_allocator_gb_as_reported_A2=124.833,
        units_correction=dict(
            allocator=f"VM gauge exo_peak_memory_bytes is x{K:.9f} too high (Memory.from_gb(P/1e9)); 124.833 -> {p_alloc:.3f} e9 B",
            resident=f"`footprint` prints binary units: '114 GB' = 113.74 GiB = {p_res:.3f} e9 B (lifetime max, libproc ri_lifetime_max_phys_footprint)"),
        peak_102k_resident_note="phys_footprint LIFETIME max of the runner process (load+warmup+serving); during serving it is most likely a plateau set by MLX gc_limit trimming (see md s2), not attributable to the 100K request alone",
        peak_102k_request="the 100,497-row COLD prefill that finished 2026-10-10 00:40:23 CDT (not the 102,411-token follow-up, a 2,059-row delta)",
        transient_gb=dict(allocator=round(tr_alloc, 3), resident=round(tr_res, 3), as_reported_A2=dict(allocator=20.133, resident=9.3)),
        slope_gb_per_token=round(slope_cold_B * 1e-9, 9),
        slope_scope="cold single-request prefill (D=N), allocator domain, GB per token of context",
        slope_gb_per_token_warm_turn=round(slope_warm_B * 1e-9, 9),
        slope_components_b_per_token=dict(retained_dspark_taps_per_prefilled_row=TAPS_B, kv_comp_kv_plus_index_k=KV_B,
                                          unattributed_O_N_fitted=round(t_B, 0), rope_tables=0,
                                          cold_total=round(slope_cold_B, 0), warm_total=round(slope_warm_B, 0)),
        breach_ctx_allocator=f"~{kfmt(N_a)} tokens, cold single-request prefill (band {kfmt(N_a_band[0])}-{kfmt(N_a_band[1])})",
        breach_ctx_resident=f"~{kfmt(N_r[0])}-{kfmt(N_r[1])} tokens, cold prefill (UNKNOWN-grade: one footprint reading + plateau assumption)",
        breaches_shipped=f"YES, conditionally: cap {CAP:,} admits cold prefills past W (>~{kfmt(N_a)}); warm turns not predicted to cross but margin {-sec_warm[-1]['vsW']:.1f} GB < model error; budget breach, not a demonstrated crash",
        breach_ctx_allocator_detail=breach_alloc, breach_ctx_resident_detail=breach_res, breaches_shipped_detail=breaches,
        breach_ctx_allocator_tokens=dict(central=round(N_a), band=[round(N_a_band[0]), round(N_a_band[1])], loo=[round(min(loo)), round(max(loo))]),
        breach_ctx_resident_tokens=dict(central_band=[round(N_r[0]), round(N_r[1])], model_band=[round(N_r_band[0]), round(N_r_band[1])]),
        breach_ctx_warm_allocator_tokens=dict(D2048=None if N_warm_2k is None else round(N_warm_2k), D25662=None if N_warm_25k is None else round(N_warm_25k),
                                              note="beyond the 1,048,576 cap -> no breach inside the admitted window"),
        admitted_cap_ctx=CAP, admitted_cap_source="GET http://192.168.86.48:52415/state instances.*.maxKvTokens (read-only); Hermes pin context_length 1,048,576",
        hermes_compress_at_tokens=HERMES_COMPRESS_AT,
        fit=dict(A_gb=round(A, 4), t_b_per_token=round(t_B, 1), sigma_gb=round(sigma, 3), n=len(fit),
                 free_fit=dict(A=round(A3, 3), cD_b_per_row=round(cD_B, 0), se_cD=round(seD_B, 0), cN_b_per_token=round(cN_B, 0), se_cN=round(seN_B, 0), sigma=round(sig3, 3), analytic_taps_b_per_row=TAPS_B),
                 residuals_gb={o["id"]: round(r, 3) for o, r in zip(fit, res)}),
        validation=vres, validation_error_band_gb=[e_lo, e_hi], upper_envelope_old_builds=[dict(name=e["name"], N=e["N"], measured=round(e["y"], 2)) for e in env],
        two_anchor_cold350K=two_anchor, active_crosses_gc_limit_cold_tokens=round(N_gc), active_crosses_physical_ram_cold_tokens=round(N_phys),
        cold_N_star_if_taps_fixed_projection_tokens=None if N_cold_fixed is None else round(N_cold_fixed),
        transient_gb_range=[round(tr_alloc, 3), round(tr_res, 3)],
        N_star_free_fit_cold_allocator_tokens=round(N_free),
        N_star_cold_if_capacity_doubled_tokens=None if N_dbl is None else round(N_dbl),
        warm_turn_max_new_rows_before_W=dstar,
        host_H_bounds_gb=[round(H_lo, 3), round(H_hi, 3)], footprint_runner_studio1=dict(now_gb=round(fp["now"], 3), peak_gb=round(fp["peak"], 3), ioaccelerator_gb=round(fp["ioaccel"], 3)),
        scenarios={k: [{kk: round(vv, 3) if isinstance(vv, float) else vv for kk, vv in r.items()} for r in v] for k, v in sc.items()},
        q1e_rederivation=q1e, system_ram_used_max_gb=ram, unit_crosscheck_soak13=xchk, non_runner_system_ram_gb=round(base_other, 2),
    )
    (HERE / "q2b-memory-headroom.json").write_text(json.dumps(out, indent=1))

    # ======================================================================================= console report
    P = print
    P(f"W={W:.5f} GB (={W*1e9/2**30:.2f} GiB)  gc_limit={GC:.3f}  phys={PHYS:.3f}  K={K:.9f}  taps={TAPS_B} B/row  KV={KV_B} B/tok  live cap={live_cap:,}")
    P("\n== FIT SET (true-unit allocator peaks = up-step requests) ==")
    P(f"{'id':10s} {'N':>9s} {'D':>8s} {'cap':>9s} {'labelled':>9s} {'true':>8s} | {'pred':>8s} {'resid':>7s}")
    for o, r in zip(fit, res):
        P(f"{o['id']:10s} {o['N']:9d} {o['D']:8d} {o['cap']:9d} {o['labelled']:9.3f} {o['y']:8.3f} | {pred(o['N'], o['D'], o['cap']):8.3f} {r:7.3f}   {o['note']}")
    P(f"\n2-param fit: A = {A:.3f} GB ; t = {t_B:.0f} B/token (se {se[1] * 1e3:.0f}) ; sigma = {sigma:.3f} GB ; n={len(fit)}")
    P(f"free 3-param fit: A={A3:.3f}  cD={cD_B:.0f} +- {seD_B:.0f} B/row (analytic retained taps = {TAPS_B}; fp32 taps would be {2 * TAPS_B})  cN={cN_B:.0f} +- {seN_B:.0f} B/token (excl. KV) ; sigma={sig3:.3f}")
    P("\n== VALIDATION (unfitted; measured - predicted, GB) ==")
    for v in vres:
        P(f"  {v['name']:20s} N={v['N']:>8d} measured {v['measured']:8.3f} predicted {v['predicted']:8.3f} err {v['err']:+6.3f}   {v['note']}")
    for e in env:
        P(f"  [envelope] {e['name']:24s} N={e['N']:>8d} measured {e['y']:8.2f}  (pre-bf16-row/M2 build; model gives {pred(e['N'], e['N'], cap_cold(e['N'])):.2f})")
    P(f"  error band = [{e_lo:+.2f}, {e_hi:+.2f}] GB")
    P("\n== N* (tokens) ==")
    P(f"  allocator cold: central {N_a:,.0f}; band {N_a_band[0]:,.0f}..{N_a_band[1]:,.0f}; LOO {min(loo):,.0f}..{max(loo):,.0f}; two-anchor {[t['N_star'] for t in two_anchor]}")
    P(f"  resident cold:  {N_r[0]:,.0f}..{N_r[1]:,.0f}; model band {N_r_band[0]:,.0f}..{N_r_band[1]:,.0f}; active>gc_limit at {N_gc:,.0f}; active>PHYS at {N_phys:,.0f}")
    P(f"  warm D=2048: {N_warm_2k}; warm D=25,662: {N_warm_25k}; cold if taps fixed: {N_cold_fixed}; cold with FREE fit: {N_free:,.0f}")
    P("  warm-turn max new rows before W (N: exact-cap / doubled-cap): " + "; ".join(f"{d['N']:,}: {d['cap_exact']:,}/{d['cap_doubled']:,}" for d in dstar))
    P(f"  H bounds {H_lo:.3f}..{H_hi:.3f}")
    for name, tab in sc.items():
        P(f"\n== {name} ==")
        for r in tab:
            P(f"  N={r['N']:>9d}  alloc {r['alloc']:7.2f} ({r['vsW']:+6.2f} vs W)   footprint {r['fp_lo']:7.2f}..{r['fp_hi']:7.2f}")
    P("\n== Q1E re-derivation ==")
    for k_, v in q1e.items():
        P(f"  {k_}: {v}")
    P("\n== system RAM-used maxima (GB; m4-1, m4-2) ==")
    for n, a_, b_ in ram["soak13_ram_plus_swap"]:
        P(f"  soak13 concurrent ram+swap max  {n:26s} {a_:8.3f} {b_:8.3f}")
    for grp, rows in (("current", ram["current"]), ("soak13", ram["soak13"])):
        for n, a_, b_, sa_, sb_ in rows:
            P(f"  {grp:8s} {n:26s} ram {a_:8.3f} {b_:8.3f}   swap {sa_:6.2f} {sb_:6.2f}")
    P(f"\nwrote {HERE / 'q2b-memory-headroom.json'}")

    # ======================================================================================= generated markdown
    def tbl(rows, hdr):
        s = "| " + " | ".join(hdr) + " |\n|" + "|".join("---:" if i else "---" for i in range(len(hdr))) + "|\n"
        for r in rows:
            s += "| " + " | ".join(str(c) for c in r) + " |\n"
        return s

    def curve_tbl(name):
        return tbl([(f"{r['N']:,}", f"{r['alloc']:.2f}", f"{r['vsW']:+.2f}", f"{r['fp_lo']:.1f}-{r['fp_hi']:.1f}") for r in sc[name]],
                   ["context N (tokens)", "allocator peak (GB)", "vs W", "OS footprint (GB)"])

    md = f"""# q2b — memory headroom of the SHIPPED dsv41 serve at MAX context (desk-only)

Author: Phase-20 subagent (Q2-gamma-mem). Branch `p20/q2-gamma-mem`. Desk analysis only: **0 boots, 0 generation requests,
0 writes to any node.** Read-only: `GET /state`, VictoriaMetrics GET/export, `ssh` (`ps`/`footprint`/`vmmap`/`vm_stat`/`sysctl`/`cat`/`grep`/`scp` from node),
plus local laptop probes (never on a cluster node). Independent of, and not blocking, the gamma round.

## FINDING (stated first)

**BREACH — conditional.** The shipped configuration admits contexts far past the point where its own peak memory crosses the wired limit
**W = 120000 MiB = 125.829 GB** (`sysctl iogpu.wired_limit_mb` = 120000 on both nodes, re-read live).

| | result |
|---|---|
| Admitted cap (live `/state` `maxKvTokens`) | **{CAP:,}** tokens (Hermes pin `context_length` {CAP:,}; its config `compression.threshold` 0.7 nominally compacts at ~{HERMES_COMPRESS_AT:,} — application not verified) |
| **Allocator domain** (`mx.get_peak_memory`, the quantity MLX compares with `gc_limit = 0.95·W`) — cold single-request prefill | crosses W at **N\\* ≈ {kfmt(N_a)}** tokens (band **{kfmt(N_a_band[0])}–{kfmt(N_a_band[1])}** from validation error; fit-only LOO {kfmt(min(loo))}–{kfmt(max(loo))}) |
| **Resident domain** (OS `phys_footprint`, adds ~{H_lo:.1f}–{H_hi:.1f} GB host memory) — cold prefill | would cross W at **N\\* ≈ {kfmt(N_r[0])}–{kfmt(N_r[1])}** tokens (model band {kfmt(N_r_band[0])}–{kfmt(N_r_band[1])}) — **UNKNOWN-grade**: one footprint reading plus the gc_limit-plateau assumption, never behaviour-tested on the nodes |
| Warm turns on a resident session (small delta) | **not predicted to cross inside the cap** ({sec_warm[-1]['alloc']:.1f} GB at {CAP:,}, {sec_warm[-1]['vsW']:+.1f} GB vs W) — but the margin is **inside the model error band** (+{e_hi:.1f} GB would reach {sec_warm[-1]['alloc'] + e_hi:.1f}), so a breach near the cap is **not excluded**; one extra cap-sized resident session ({sec_warm3[-1]['alloc']:.1f} GB) does cross it |
| Cold prefill at the cap | extrapolates to **{at_cap_cold:.0f} GB** (> {PHYS:.0f} GB physical RAM) — **UNVERIFIED** (no cold prefill > 350K has ever been measured on a near-shipped build) |
| Largest real prompt seen | {MAX_SEEN_PROMPT:,} tokens (`state.db`, 96 Hermes calls to this model) — **below N\\* in both domains** |

This is **measured, not only extrapolated — but on older builds**: three independent cold-350K requests (instances d301ea85 / 5689f0ba / 30b396d2, builds of 2026-10-06/07, i.e. *before* next13, same engine path)
peaked at {", ".join(f"{v['y']:.1f}" for v in val if v['N'] == 350_124)} GB in **true units** (W = {W:.1f}; two of three above W). The **next13** soak ladder (build `f0840af1c`/mlx-lm `6cc9c1e`; prod is `99e2966ee`/`689e4ea`, see §7) peaked at
{fit[7]['y']:.1f} / {fit[8]['y']:.1f} / {fit[9]['y']:.1f} GB after the 500K / 750K / 1.04M deltas (W{fit[7]['y'] - W:+.1f} / {fit[8]['y'] - W:+.1f} / {fit[9]['y'] - W:+.1f}). System RAM-used reached
{min(min(r[1], r[2]) for r in ram['soak13'][2:]):.1f}–{max(max(r[1], r[2]) for r in ram['soak13'][2:]):.1f} GB on both nodes during those deltas (idle: ~{ram['current'][-1][1]:.0f} GB) and swap rose from 0.1 to **{ram['soak13'][4][3]:.1f} / {ram['soak13'][4][4]:.1f} GB** (m4-1 / m4-2) at the deepest delta.
**No cold prefill > 350K has ever been run on the current (next19) build.**

**What "breach" means (which limit actually fails).** `W` is the kernel's GPU-wired ceiling (`iogpu.wired_limit_mb`), not an allocator limit. In the pinned MLX allocator source (`603f16eb7`) `malloc()` never throws at W: it only trims its
buffer cache at `gc_limit = 0.95·W`, and this fork keeps MLX residency sets disabled (`resident.cpp`; `MLX_RESIDENCY_SETS` is not set on the runner), so exo's `mx.set_wired_limit(...)` call does not itself pin anything (source reading, not behaviour-tested).
Above W the kernel cannot wire more GPU pages; the observed signature is compression/swap — the swap rise above, and studio2's runner now holds **18.7 GiB of its IOAccelerator swapped out** (studio1 1.9 GiB) — though attributing
that swap to W-exceedance (rather than other pressure) is an inference. So this is a **budget breach with an observed pressure signature, not a demonstrated crash**: those loads completed (docs: "compression absorbs, but margin is zero"). Risks: swap/compressor slowdown of the next request, the hang-watchdog false-positive class (SKILL.md "A runner killed mid-long-op"),
and the 124000-MiB Metal-allocator wedge history if anyone raises W without re-validating.

**Two inputs handed to this round are mis-scaled (details below), and one cause of the steep slope is fixable:**
1. The VM gauge `exo_peak_memory_bytes` is **×{K:.4f} too high** → A2's "124.833 GB allocator peak at ~102K" is really **{p_alloc:.3f} GB**.
2. `footprint` prints **binary** units → A2's "114.0 GB resident peak" is **{p_res:.3f} × 10⁹ B** (113.74 GiB). The two domains swap order once corrected.
3. `Conversation.prefill` retains every chunk's DSpark taps (**{TAPS_B:,} B per prefilled row**, {TAPS_B * 100_497 / 1e9:.1f} GB at 100K rows) until the prefill returns; it is ~{TAPS_B / slope_cold_B * 100:.0f}% of the cold-prefill slope. Flagged only (out of scope to fix); a projection with it fixed is in §5.

## 1. Corrected inputs (verified at entry)

| quantity | as handed over (A2 / Q1E) | verified | how |
|---|---|---|---|
| W | 125.829 GB | **{W:.3f} GB** = 120000 MiB = 117.19 GiB (`hw.memsize` {PHYS:.2f} GB) | `sysctl` on both nodes; runner log `Wired limit set to 117.19 GiB` |
| weights / rank | 104.7 GB | **104.7 GB** (decimal) | `load.py:190` prints `mx.get_active_memory() / 1e9`; both ranks |
| admitted cap | – | **{CAP:,}** | `GET /state` → `maxKvTokens` (also `kvCacheBits` 0) |
| allocator peak at ~102K | 124.833 GB | **{p_alloc:.3f} GB** | gauge is `Memory.from_gb(mx.get_peak_memory()/1e9)` and `from_gb(v)=round(v·1024³)` → ×{K:.9f}; executed on a known 4.0 GB allocation (`q2b_memory_unit_probe.out.txt`). Independent cross-check: 75 of 104 EXL3 instances' first reading clusters at 118.651 → /{K:.4f} = 110.50 = the runner's own `warmup: peak=110.5 GB` line |
| resident peak at ~102K | 114.0 GB | **{p_res:.3f} GB** (113.74 GiB) | `footprint` default output is binary: 4,000,006,400 B allocated → `-f bytes` reports that region as 4,000,956,416 B, which the default output prints as "3816 MB" (= 3815.6 MiB; a decimal reading would be 4001 MB) (`q2b_footprint_unit_probe.out.txt`); runner pid 49330 `-f bytes` peak 122,126,271,536 B = libproc `ri_lifetime_max_phys_footprint` = `vmmap` 113.7G; studio2 runner 122.104 GB |
| non-weight transient | 20.13 alloc / 9.3 resident | **{tr_alloc:.2f} alloc / {tr_res:.2f} resident** | peak − 104.7 |
| "peak at the 102,411-token prompt" | – | the peak was set by the **100,497-row cold prefill** (finished 00:40:23); the 102,411-token request was a 2,059-row delta on the resident session | exo.log `prefill controls … rows=100497` / `turn reuse: prompt=102411 prefill=2059`; VM prompt-token counter |

Caveat on the resident number: `ri_lifetime_max_phys_footprint` is a **process-lifetime** max. System RAM-used was already within ~1 GB of its 100K value during 18K–20K-row cold prefills
({ram['current'][2][1]:.2f} GB at 18K vs {ram['current'][4][1]:.2f} GB at 100K), so the 122.1 GB is **most likely a plateau set by the allocator's gc_limit trimming** (§2; inference from allocator source + RAM-used series, not behaviour-tested on the nodes), not a measure of the 100K request specifically. It cannot be extrapolated linearly from one point.

**Third, independent confirmation of the gauge correction (physical consistency).** Active allocator bytes must fit inside what the OS says is in use (RAM-used + swap-used). On the next13 soak's deepest rungs (concurrent
15 s samples, both nodes, raw files in this directory; non-runner system RAM ≈ {base_other:.1f} GB on studio1):

{tbl([(x['rung'], f"{x['gauge_labelled']:.1f}", f"{x['gauge_true']:.1f}", f"{x['ram_plus_swap_m4_1']:.1f}", f"{x['ram_plus_swap_m4_2']:.1f}", f"{x['ram_plus_swap_m4_1'] - x['gauge_true']:+.1f} / {x['ram_plus_swap_m4_2'] - x['gauge_true']:+.1f}", f"{x['ram_plus_swap_m4_1'] - x['gauge_labelled']:+.1f} / {x['ram_plus_swap_m4_2'] - x['gauge_labelled']:+.1f}") for x in xchk], ["rung", "gauge as labelled", "gauge corrected (÷1.0737)", "RAM+swap m4-1", "RAM+swap m4-2", "(RAM+swap) − corrected", "(RAM+swap) − labelled"])}
With the corrected gauge the residual is **{min(min(x['ram_plus_swap_m4_1'], x['ram_plus_swap_m4_2']) - x['gauge_true'] for x in xchk):+.1f} … {max(max(x['ram_plus_swap_m4_1'], x['ram_plus_swap_m4_2']) - x['gauge_true'] for x in xchk):+.1f} GB**: the same order as the ≈{base_other + fp['host']:.1f} GB of non-allocator memory seen at idle
(runner host {fp['host']:.1f} GB + other processes ≈{base_other:.1f} GB); it sits somewhat under that because 15 s samples can miss the peak. With the labelled gauge the allocator's active bytes would **exceed** everything the OS reports in use at every rung
(negative residual in the last column) — impossible. (A consistency check, not proof: macmon's `ram_usage` definition is not independently verified.)

## 2. How the peak is built (shipped code, measured on the real classes)

Allocator peak = `mx.get_peak_memory()` = high-water of ACTIVE bytes (cache pool is a separate counter; A2 micro-test). Components of a cold prefill of `N` rows:

* **weights** 104.7 GB (+ DSpark draft head, attached after the 104.7 reading; UNMEASURED, ≈3.6 GB/rank if TP-sharded — checkpoint `mtp.*` = 7.243 GB; warmup peak 110.5 GB bounds it ≤ 5.8) → intercept **A = {A:.2f} GB** includes these and the base transient.
* **KV (resident session)** = `comp_kv` 2,560 B + `index_k` 640 B = **{KV_B:,} B per capacity-row** (real `ModelCache`, `q2b_kv_per_token_probe.out.txt`); capacity grows `min(max(req, 2·cap), {CAP:,})` from 65,536 (`cache.py:188`) → {KV_B * CAP / 1e9:.2f} GB at the cap per session (`max_sessions=2`). `win_kv` 5.2 MB fixed. **RoPE tables: none** (per-call since 2026-10-04: `attention.py:127`, `mtp.py:119`).
* **retained DSpark taps** = 3 layers × 5,120 × 2 B (bf16) = **{TAPS_B:,} B per PREFILLED row** (stub-model probe on the real `engine_prefill`: `sum(nbytes) = rows·30,720` exactly, `q2b_taps_retention_probe.out.full.txt`). The free fit below recovers **{cD_B:,.0f} ± {seD_B:,.0f} B/row** (fp32 taps would be {2 * TAPS_B:,}).
* **unattributed O(N)** = **{t_B:,.0f} B/token** (fitted). Candidates, UNVERIFIED: per-chunk offset-scaling transients (indexer block-maxima `step·offset·1 B` = 2,048 B/token at the shipped chunk 2,048; compressor `kv_all` copies; sparse-attn gathers).
* **MLX allocator** (`mlx/backend/metal/allocator.cpp`, the exact submodule `603f16eb7` the nodes run): `gc_limit = 0.95·W = {GC:.2f} GB`; on a cache miss `if active+cache+size ≥ gc_limit → release cached buffers`; **no hard throw at W**. So while active < gc_limit the process's GPU total sits pinned near gc_limit (cache fills the gap) and OS footprint ≈ gc_limit + host ≈ {fp['peak']:.1f} GB; only when **active** passes gc_limit (cold N ≈ {N_gc:,.0f}) does footprint track active (+ host {H_lo:.2f}–{H_hi:.2f} GB).

Model: `peak(N) = A + t·N + {KV_B}·cap(N) + {TAPS_B:,}·D`, `D` = rows prefilled by the request (cold: D = N; warm turn: D ≈ 2K).
A and t are **fitted**; KV and taps are **analytic**. Fit set = every request whose finish raised the (monotone, process-lifetime) gauge, in true units, with the VM prompt-token delta checked equal to `N` and, for the current boot, the exo.log `rows=` line checked equal to `D`:

{tbl([(o['id'], f"{o['N']:,}", f"{o['D']:,}", f"{o['cap']:,}", f"{o['y']:.3f}", f"{pred(o['N'], o['D'], o['cap']):.3f}", f"{r:+.3f}", o['note']) for o, r in zip(fit, res)], ["id", "N", "D", "capacity", "measured (true GB)", "model", "resid", "note"])}
Fit: **A = {A:.3f} GB, t = {t_B:,.0f} B/token, σ = {sigma:.3f} GB (n = {len(fit)}).** The 3 delta rungs (D ≠ N) make `D` and `N` separately identifiable: free fit `cD = {cD_B:,.0f} ± {seD_B:,.0f} B/row` (analytic {TAPS_B:,}), `cN = {cN_B:,.0f} ± {seN_B:,.0f} B/token` (excluding KV). Standard errors are optimistic (n = 10, pooled builds); builds differ ≤ 0.15 GB at the matched ~100K point (next19 {fit[4]['y']:.2f} vs next13 {fit[5]['y']:.2f}).

### Unfitted validation (cold, VM hit-kind `none`) — this sets the model-error band

{tbl([(v['name'], f"{v['N']:,}", f"{v['measured']:.3f}", f"{v['predicted']:.3f}", f"{v['err']:+.3f}", v['note']) for v in vres], ["request", "N", "measured", "model", "meas − model", "build"])}
Error band **[{e_lo:+.2f}, {e_hi:+.2f}] GB** (measured higher = earlier crossing). Old pre-bf16-row/pre-M2 builds (upper envelope only, not used): {"; ".join(f"{e['name']} {e['y']:.1f} GB" for e in env)}.

## 3. N\\* (tokens) — both domains

{tbl([
 ("allocator, cold prefill, central (fit)", f"{N_a:,.0f}", f"slope {slope_cold_B:,.0f} B/token = **{slope_cold_B * 1e-9:.3e} GB/token**"),
 ("allocator, cold, band from validation error", f"{N_a_band[0]:,.0f} – {N_a_band[1]:,.0f}", f"e = {e_hi:+.2f} / {e_lo:+.2f} GB"),
 ("allocator, cold, leave-one-out", f"{min(loo):,.0f} – {max(loo):,.0f}", "fit stability only"),
 (f"allocator, cold, FREE fit (cD, cN fitted separately; taps coefficient {cD_B / 1e3:.1f} vs analytic {TAPS_B / 1e3:.1f} KB/row)", f"{N_free:,.0f}", "sensitivity to the ~2.4σ taps-coefficient gap (something is mildly unmodeled)"),
 ("allocator, cold, IF capacity were rounded up 2× (worst case)", "n/a" if N_dbl is None else f"{N_dbl:,.0f}", "shipped cold path uses ONE up-front ensure_capacity ⇒ cap(N)=N for N ≥ 131,072 (verified in q2b_kv_per_token_probe.out.txt); at N* the cap is ≈ N* itself"),
 ("allocator, cold, empirical two-anchor (100K anchor → each measured cold-350K)", " / ".join(f"{t['N_star']:,}" for t in two_anchor), "no model: straight lines through measured points"),
 ("resident (OS footprint), cold", f"{N_r[0]:,.0f} – {N_r[1]:,.0f}", f"active + host {H_lo:.2f}…{H_hi:.2f} GB; model band {N_r_band[0]:,.0f}–{N_r_band[1]:,.0f}"),
 (f"active passes gc_limit ({GC:.1f} GB) — end of the footprint plateau", f"{N_gc:,.0f}", "cold"),
 (f"active passes physical RAM ({PHYS:.1f} GB)", f"{N_phys:,.0f}", "cold (needs OS swap)"),
 ("warm turn D=2,048 / D=25,662, allocator", f"{'beyond cap' if N_warm_2k is None or N_warm_2k > CAP else f'{N_warm_2k:,.0f}'} / {'beyond cap' if N_warm_25k is None or N_warm_25k > CAP else f'{N_warm_25k:,.0f}'}", f"model crossing {'' if N_warm_2k is None else f'{N_warm_2k:,.0f}'} / {'' if N_warm_25k is None else f'{N_warm_25k:,.0f}'} — outside the admitted window"),
 ("naive 'as asked': A2 numbers (mis-scaled) + KV-only slope, allocator / resident", f"{102_411 + (W - 124.832998) / (KV_B * 1e-9):,.0f} / {102_411 + (W - 114.0) / (KV_B * 1e-9):,.0f}", "right order for the wrong reasons (resident 3.8M is wrong; KV-only slope is wrong)"),
 ], ["quantity", "tokens", "notes"])}
Slopes: cold prefill = {TAPS_B:,} (taps) + {KV_B:,} (KV) + {t_B:,.0f} (fitted O(N)) = **{slope_cold_B:,.0f} B/token**; warm turn = {KV_B:,} + {t_B:,.0f} = **{slope_warm_B:,.0f} B/token** (taps scale with the rows PREFILLED, not the context). The KV-only slope ({KV_B:,} B/token) alone would put the crossing at {100_497 + (W - anchor) / (KV_B * 1e-9):,.0f} tokens — the measured deep points rule that out.

## 4. peak(N) curves (allocator domain; true GB; "vs W" = peak − {W:.2f})

### 4a. Cold prefill (a single request with no resident/parked prefix) — AS SHIPPED
{curve_tbl("cold prefill (as shipped)")}
### 4b. Warm turn (≈2K new rows on a resident conversation; exact capacity)
{curve_tbl("warm turn, D=2,048 rows")}
### 4c. Warm turn + a second resident session (`max_sessions=2`) — sensitivity (analytic)
Warm turn at the {CAP:,}-row cap PLUS a second resident session: second session at 400K rows → {sec_warm2[-1]['alloc']:.2f} GB ({sec_warm2[-1]['vsW']:+.2f} vs W); second session also at the cap → **{sec_warm3[-1]['alloc']:.2f} GB ({sec_warm3[-1]['vsW']:+.2f} vs W)**. (The engine keeps `max_sessions=2`; whether Hermes ever holds two deep conversations is UNKNOWN.)

### 4d. Warm turn with capacity rounded UP to 2× the context (upper bound; a session grown turn-by-turn has capacity 65,536·2^k, i.e. ≤ 2×)
{curve_tbl("warm turn D=2,048, capacity rounded up 2x (upper bound)")}
### 4e. Largest single warm turn (new rows) that keeps the allocator peak ≤ W (model central; error band ±1–2 GB ≈ ±35–70K rows)
{tbl([(f"{d['N']:,}", f"{d['cap_exact']:,}", f"{d['cap_doubled']:,}") for d in dstar], ["context N", "max new rows (exact capacity)", "max new rows (2× capacity)"])}
## 5. Adjacent findings (flagged, NOT fixed — outside this task's scope)

1. **Retained DSpark taps (memory bug in the shipped prefill path).** `Conversation.prefill` (`dsv41/session.py:644`) passes a local list as `taps_out=` into `SessionCache.append_turn` → `engine_prefill`, which appends *every* chunk's taps although `taps_cb` (`_on_chunk_taps`) already consumed them; the list lives until `prefill()` returns. {TAPS_B:,} B per prefilled row = {TAPS_B * 100_497 / 1e9:.1f} GB at 100K, {TAPS_B * 340_248 / 1e9:.1f} GB for the r500 delta, {TAPS_B * CAP / 1e9:.1f} GB for a cold 1M. Steady-state memory is unaffected (peak only). **Projection, untested:** without it the cold curve collapses onto the warm curve (cold N\\* {'beyond the cap' if N_cold_fixed is None or N_cold_fixed > CAP else f'{N_cold_fixed:,.0f}'}; {sc['cold prefill IF taps retention were fixed (projection)'][-1]['alloc']:.1f} GB at the cap). Suggested minimal change (UNTESTED; needs the dsv41 session/draft-lockstep tests and a live A/B): do not pass `taps_out` when `taps_cb` is installed (the list is only the fallback for `_feed_taps`, `session.py:677`).
2. **Gauge unit bug.** `exo_peak_memory_bytes` for dsv41 is ×{K:.4f} high (`engine.py:284`, `rounds.py:227`; the batched generator uses `/1024**3` correctly). Also it is a **process-lifetime** high-water (no `reset_peak_memory` in the dsv41 path), not "peak of the most recent request". Any alert or doc comparing it to W is wrong by 7.4%.
3. **`footprint` units.** Q1E-A1 states `footprint` "reports decimal GB (verified)". It is binary; the "verification" matched IOAccelerator 105 *GiB* (= 112.6 × 10⁹ B, which must also hold the draft head, resident KV and the MLX cache pool) against weights 104.7 × 10⁹ B. Prior docs quoting "113 GB peaks" (PERFORMANCE_HISTORY soak entries) are, **if** they came from `footprint` default output, GiB (113 GiB = 121.3 × 10⁹ B) — their source tool is not identified (`bench/soak_next13.sh` has no footprint sampling), so see the UNRECONCILED item in §8.
4. **studio2 anomaly (cause UNKNOWN).** 00:57 CDT, 16 min after the last request, studio2 RAM-used jumped 120.4 → 131.3–132.7 GB and swap 0.08 → 19.5 GB (then settled at 9.65 GB used; no request in flight). The studio2 runner now shows **18.7 G of IOAccelerator swapped out** (studio1: 1.9 G) — the next studio2 request pays page-in latency. Raw: `q2b_raw_node_studio*_vmmap_footprint.txt`, `q2b_raw_vm_sys_series.json`.

## 6. Q1E consequence (CLOSE verdict unchanged, but the margins were mis-stated)

Rule `B_arm + rest(6.51) + transient + 2 ≤ W` for q3g128 (B_arm 110.42 → headroom {head_q3:.2f} GB):

{tbl([(k_.replace('_', ' '), v['non_weight'], v['demand'], v['headroom'], f"{v['over']:+.2f}", f"{v['over_with_margin']:+.2f}") for k_, v in q1e.items() if 'demand' in v], ["reading", "non-weight", "rest+non-weight", "headroom", "over by", "over by (+2 GB margin)"])}
Both corrected readings still FAIL, so CLOSE stands. The "razor-thin 0.4 GB on the resident reading" was an artifact of mixing a binary `footprint` number with a decimal weight; the favourable reading is now the *allocator* one (+{q1e['allocator_corrected']['over']:.2f}). If retained taps were fixed the allocator reading would be {q1e['allocator_if_taps_fixed_projection']['over']:+.2f} GB before the pre-registered 2 GB margin and {q1e['allocator_if_taps_fixed_projection']['over'] + 2:+.2f} GB with it (still FAILS) — a new hypothesis for a separate pre-registered round (Q1E reopen condition (b)), not a change to this verdict.

## 7. Assumptions

* Gauge = rank-0 runner's `mx.get_peak_memory()` (only rank 0 emits stats; `runner.py:875`); footprint read on rank 1; both ranks' lifetime max agree to 0.02 GB.
* Builds pooled: the fit mixes next19 (current boot) and next13 (soak13). `git diff 6cc9c1e 689e4ea` (mlx-lm) touches indexer small-n fp32 row (n ≤ 16), sparse-attention gates, `exl3_build.py` dense policy; exo `dsv41/{{session,rounds,park}}.py` identical, `engine.py` +56 lines. Matched-context check: ≤ 0.15 GB.
* The gauge is a process-lifetime high-water that never resets in the dsv41 path (no `reset_peak_memory`). Checked on the raw series: 0 downward steps in the two fit instances (7aa2dbd3, 38ed8ddb); 8 of 47,161 consecutive pairs fleet-wide step down — 3 are 0.00 GB float noise, 4 are on 10-01…10-03/10-09 instances outside the fit, and the 44.7 GB drop (instance 867ef937, 2026-10-05 05:34, 163.3 → 118.6 labelled = back to the warmup level) is consistent with a runner restart under the same instance id after the r1m watchdog kill (inference). So an up-step identifies the request that set the process maximum.
* `ensure_capacity` is called once up front with `offset + delta` (`session_cache.py:617`); the r750 rung therefore holds 2× capacity (999,984).
* Taps are bf16 (`EXO_COMPUTE_DTYPE=bf16`; the free fit gives {cD_B:,.0f} ± {seD_B:,.0f} B/row vs {TAPS_B:,} analytic — fp32 would be {2 * TAPS_B:,}, excluded; the ~2.4σ shortfall is unmodeled, hence the FREE-fit row in §3).
* Only ONE long conversation resident (other session small, 65,536-row capacity ≈ 0.21 GB — inside the residual).
* Extrapolating the cold model from D ≤ 340K to D = {CAP:,} assumes linearity; the deltas bracket the cold-N\\* region (500K-rung predicted from 160K-cold within 0.4 GB), but nothing deeper-cold than 350K exists on a current build.
* "Breach" = peak allocator active memory (or OS footprint) above W. The allocator does not throw there; consequences are OS compression/swap, GPU stalls, watchdog false positives.

## 8. UNKNOWN / not verified

* Behaviour of a **cold prefill > 350K on the shipped build** (never run); the {at_cap_cold:.0f} GB at the cap is an extrapolation. A live test is the PM's call (needs boots/cluster time — not done here).
* Draft-head weight size per rank (inside A); mechanism of the fitted {t_B:,.0f} B/token O(N) term (candidates only).
* The allocator-plateau model was **not behaviour-tested on the nodes** (read-only); it rests on the allocator source, the system RAM-used series (plateau 125.8–126.6 GB at every cold prefill 18K–160K; 133.1–133.7 GB on the deep deltas) and one footprint reading. Resident N\\* is the weakest number here.
* Resident-domain N\\* (and the ~122.1 GB plateau) is **UNKNOWN-grade**: it chains one `footprint` reading with the allocator-trim reading of the source.
* Warm-turn margin at the cap ({-sec_warm[-1]['vsW']:.1f} GB) is smaller than the unfitted-validation error ({e_lo:+.1f}..{e_hi:+.1f} GB): "no crossing" is a model prediction, not a guarantee.
* The exo swap gauge peaked at 19.5 GB on studio2 at 00:57 CDT while `vm.swapusage` reports a 10,240 MiB swap file now — the gauge and the OS number are not reconciled.
* Whether memory pressure at W+ would hang/kill a runner on the *current* build (soaks completed on older builds; docs record a watchdog false-positive kill and a 124000-MiB wedge).
* Cause of the studio2 swap/RAM event at 00:57 CDT.
* `state.db` covers only 96 calls through 2026-10-07; real traffic since then is unknown.
* **UNRECONCILED:** PERFORMANCE_HISTORY's soak13 entry says "Memory stayed 113 GB both nodes through the deep delta (well under the wired limit)" while the VM gauge (true units) says the allocator peaked at {fit[9]['y']:.1f} GB and system RAM-used+swap says ~137 GB on those rungs. The doc's source tool/sampling is not identified; if it was a `footprint` snapshot it is a steady-state reading, not a peak. This report relies on the gauge and the system series, which agree with each other (§1 table), not on that sentence.

## 9. Raw readings and reproduction

All numbers above come from files in this directory; `python3 q2b_headroom_model.py` regenerates this document and `q2b-memory-headroom.json` from them (stdlib, no network).

| file | content |
|---|---|
| `q2b_raw_vm_instances_dsv41.json` | VM `exo_peak_memory_bytes` for 104 EXL3 instances: every up-step + prompt-token & hit-kind deltas (from `q2b_vm_history.py`) |
| `q2b_raw_vm_current_peak.json`, `q2b_raw_vm_sys_series.json`, `q2b_raw_vm_ram_soak13_window.json` | raw 15 s samples: current-instance gauge; system RAM-used/swap (both nodes) |
| `q2b_raw_node_studio{{1,2}}_vmmap_footprint.txt` | `sysctl`, `vmmap -summary`, `footprint -f bytes`, `vm_stat` on both runners (2026-10-10 04:27 CDT) |
| `q2b_raw_studio1_exo_log_excerpts.txt` | studio1 `exo.log` (current boot): load, warmup, prefill/turn-reuse/park lines |
| `q2b_raw_state_capacity.json`, `q2b_raw_checkpoint_config.json` | live `/state` capacity fields; checkpoint `config.json` |
| `q2b_memory_unit_probe.py/.out.txt`, `q2b_footprint_unit_probe.py/.out.txt`, `q2b_footprint_calibration.py/.out.txt` | unit proofs (laptop, known allocations) |
| `q2b_kv_per_token_probe.py/.out.txt`, `q2b_taps_retention_probe.py/.out.full.txt` | KV bytes/token + capacity rule; taps retention (real `ModelCache`/`engine_prefill`, stub model) |
| `q2b_raw_node_wired_swapped.txt`, `q2b_capture_node_readonly.sh` | `footprint --wired --swapped`/`--sysFootprint`/`vmmap`/`vm_stat` on both runners (06:57 CDT) and the read-only script that makes them |
| `q2b_vm_history.py`, `q2b_fetch_sys_series.py`, `q2b_headroom_model.py/.out.txt` | VM extraction; the model |
"""
    (HERE / "q2b-memory-headroom.md").write_text(md)
    P(f"wrote {HERE / 'q2b-memory-headroom.md'}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
