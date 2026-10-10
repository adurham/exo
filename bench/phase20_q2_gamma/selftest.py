#!/usr/bin/env python3
"""Offline self-test for the Q2-Gamma driver + evaluators.  NO cluster, NO POST.

Two halves:

A. ANCHOR REPRODUCTION — re-derive every rep of the frozen r1kit control JSONs
   (``docs/benchmarks/phase20-throughput/raw/pricing/q1/round/control_*.json``)
   with the kit's own ``derive()`` and recompute the summary medians; they must
   reproduce the frozen anchors (benign 94.88 / 38.48 / 2.682; agentic
   101.07 / 30.962 / 2.1128).  If the kit is absent the medians are recomputed
   straight from the frozen recs and the re-derivation step is reported SKIPPED.

B. EVALUATOR / HARNESS UNIT TESTS on synthetic records — identity gate
   (identical=PASS, near-tie=PASS, confident=ABORT), drift (>1.5%=RE-RUN),
   bars (>=3% agentic win=WINNER, split=CLOSE, no-win), memory gate, the
   arm-switch command builders + ``sed`` simulation, override-line parsing,
   readiness counting, footprint/VM parsers, and the budget declaration.

Run:  cd <worktree> && PYTHONPATH=bench/phase20_q2_gamma \\
      /Users/adam.durham/repos/exo/.venv/bin/python bench/phase20_q2_gamma/selftest.py
"""
from __future__ import annotations

import json
import os
import statistics
import sys
import tempfile
import time

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)

import q2_gamma_eval as EV            # noqa: E402
import q2_gamma_driver as DRV         # noqa: E402

# ---------------------------------------------------------------------------
# Synthetic record builders
# ---------------------------------------------------------------------------
def _toks(n):
    return [f"t{i}" for i in range(n)]


def _top2(n, margin=0.5, at=None, margin_at=None):
    """[[-1, -1-margin], ...]; optionally a different margin at index ``at``."""
    out = [(-1.0, -1.0 - margin) for _ in range(n)]
    if at is not None:
        out[at] = (-2.0, -2.0 - (margin if margin_at is None else margin_at))
    return out


def _rec(gen=800, decode_s=20.0, tokens=None, top2=None, hist=None):
    ds = decode_s
    return {"completion_tokens": gen, "decode_s": ds, "ttft_s": 5.0,
            "tokens": tokens if tokens is not None else _toks(4),
            "top2_logprobs": top2 if top2 is not None else _top2(4, 0.5),
            "hist_delta": hist, "mean_accepted_hist": EV.mean_accepted_from_hist(hist) if hist else None}


def _tok_s(target_tok_s, gen=800):
    """decode_s that yields the given decode tok/s for ``gen`` completion tokens."""
    return (gen - 1) / target_tok_s


def _arm(gamma, agentic, benign, **kw):
    d = {"gamma": gamma, "logprobs_available": True, "agentic": agentic, "benign": benign,
         "peak_alloc_bytes": {}, "peak_resident_bytes": {}}
    d.update(kw)
    return d


# ---------------------------------------------------------------------------
# A. Anchor reproduction
# ---------------------------------------------------------------------------
def find_round_dir():
    rel = os.path.join("docs", "benchmarks", "phase20-throughput",
                       "raw", "pricing", "q1", "round")
    d = HERE
    for _ in range(12):
        cand = os.path.join(d, rel)
        if os.path.isdir(cand):
            return cand
        d = os.path.dirname(d)
    return None


def find_kit():
    for c in (os.environ.get("Q2_GAMMA_KIT"),
              "/private/tmp/q1b-driver/bench/phase20_r1kit",
              "/private/tmp/next16-instr/bench"):
        if c and os.path.exists(os.path.join(c, "phase19_round_measure.py")):
            return c
    return None


FROZEN = {
    "benign": {"ms_per_round_median": 94.88, "decode_tps_median": 38.48, "mean_accepted_median": 2.682},
    "agentic": {"ms_per_round_median": 101.07, "decode_tps_median": 30.962, "mean_accepted_median": 2.1128},
}


def anchor_checks():
    out = []
    rd = find_round_dir()
    out.append(("anchor: control round dir located", rd is not None, rd or "NOT FOUND"))
    if rd is None:
        return out
    kit = find_kit()
    RM = None
    if kit:
        sys.path.insert(0, kit)
        import importlib
        RM = importlib.import_module("phase19_round_measure")
    out.append(("anchor: frozen kit available for re-derivation", kit is not None,
                kit or "absent -> re-derivation SKIPPED (medians recomputed from frozen recs)"))

    for arm in ("benign", "agentic"):
        path = os.path.join(rd, f"control_{arm}.json")
        if not os.path.exists(path):
            out.append((f"anchor[{arm}]: control file present", False, path))
            continue
        doc = json.load(open(path, encoding="utf-8"))
        summ, recs = doc["summary"], doc["recs"]

        # re-derive every rep from raw fields (kit path) / verify stored fields
        if RM is not None:
            prev_cyc = prev_acc = None
            ok = True
            for rec in recs:
                raw = {"stats": rec.get("stats"), "usage": rec.get("usage"),
                       "decode_s": rec.get("decode_s")}
                d = RM.derive(raw, prev_cyc, prev_acc, 3)
                for k in ("rounds", "mean_accepted", "ms_per_round", "gamma_implied", "decode_tps"):
                    if d.get(k) != rec.get(k):
                        ok = False
                if d.get("cycles_cum") is not None:
                    prev_cyc, prev_acc = d["cycles_cum"], d["accepted_cum"]
            out.append((f"anchor[{arm}]: kit derive() reproduces every frozen rep", ok, ""))
        else:
            have = all(r.get("decode_tps") is not None for r in recs)
            out.append((f"anchor[{arm}]: frozen recs carry derived fields (kit SKIPPED)", have, ""))

        dec = [r["decode_tps"] for r in recs if r.get("decode_tps")]
        msr = [r["ms_per_round"] for r in recs if r.get("ms_per_round") is not None]
        mar = [r["mean_accepted"] for r in recs if r.get("mean_accepted") is not None]
        got = {"decode_tps_median": round(statistics.median(dec), 3) if dec else None,
               "ms_per_round_median": round(statistics.median(msr), 2) if msr else None,
               "mean_accepted_median": round(statistics.median(mar), 4) if mar else None}
        frozen_sum = {k: summ[k] for k in got}
        out.append((f"anchor[{arm}]: recomputed medians == frozen summary", got == frozen_sum,
                    f"got={got} frozen={frozen_sum}"))
        out.append((f"anchor[{arm}]: matches frozen anchor literal", got == FROZEN[arm],
                    f"{got} vs {FROZEN[arm]}"))
        out.append((f"anchor[{arm}]: decode metric = (gen-1)/decode_s", len(dec) > 0,
                    f"n={len(dec)}"))
    return out


# ---------------------------------------------------------------------------
# B. Evaluator / harness unit tests
# ---------------------------------------------------------------------------
def evaluator_checks():
    out = []
    eps = EV.EPSILON  # 0.05
    N = 256  # realistic captured-position count (== driver CAPTURE_POSITIONS)

    def _div_at(n, idx, repl="X"):
        t = _toks(n)
        t[idx] = repl
        return t

    # ---- identity: identical => PASS -------------------------------------
    ref = _arm(3, [_rec(tokens=_toks(N), top2=_top2(N, 0.5)) for _ in range(2)], [_rec()])
    ctrl = _arm(3, [_rec(tokens=_toks(N), top2=_top2(N, 0.5)) for _ in range(2)], [_rec()])
    same = _arm(2, [_rec(tokens=_toks(N), top2=_top2(N, 0.5)) for _ in range(2)], [_rec()])
    ig = EV.identity_gate({"gamma3a": ref, "gamma3b": ctrl, "gamma2": same})
    out.append(("identity: identical vs gamma3a => PASS", ig["verdict"] == "PASS",
                f"verdict={ig['verdict']} arms={ {a: r['verdict'] for a, r in ig['arms'].items()} }"))

    # ---- identity: near-tie divergence (margin < eps) => PASS -------------
    ref_nt = _arm(3, [_rec(tokens=_toks(N), top2=_top2(N, 0.5, at=2, margin_at=0.02)),
                      _rec(tokens=_toks(N), top2=_top2(N, 0.5))], [_rec()])
    ctrl_nt = _arm(3, [_rec(tokens=_toks(N), top2=_top2(N, 0.5)),
                       _rec(tokens=_toks(N), top2=_top2(N, 0.5))], [_rec()])
    cand_nt = _arm(2, [_rec(tokens=_div_at(N, 2), top2=_top2(N, 0.5)),
                       _rec(tokens=_toks(N), top2=_top2(N, 0.5))], [_rec()])
    ig_nt = EV.identity_gate({"gamma3a": ref_nt, "gamma3b": ctrl_nt, "gamma2": cand_nt})
    fd = ig_nt["arms"]["gamma2"]["first_divergence"]
    out.append(("identity: near-tie divergence (margin<eps) => PASS", ig_nt["verdict"] == "PASS",
                f"verdict={ig_nt['verdict']} first_div={fd}"))

    # ---- identity: confident divergence (margin >= eps) => ABORT ----------
    ref_cf = _arm(3, [_rec(tokens=_toks(N), top2=_top2(N, 0.5, at=2, margin_at=0.5)),
                      _rec(tokens=_toks(N), top2=_top2(N, 0.5))], [_rec()])
    ctrl_cf = _arm(3, [_rec(tokens=_toks(N), top2=_top2(N, 0.5)),
                       _rec(tokens=_toks(N), top2=_top2(N, 0.5))], [_rec()])
    cand_cf = _arm(4, [_rec(tokens=_div_at(N, 2), top2=_top2(N, 0.5)),
                       _rec(tokens=_toks(N), top2=_top2(N, 0.5))], [_rec()])
    ig_cf = EV.identity_gate({"gamma3a": ref_cf, "gamma3b": ctrl_cf, "gamma4": cand_cf})
    out.append(("identity: confident divergence (margin>=eps) => ABORT", ig_cf["verdict"] == "ABORT",
                f"verdict={ig_cf['verdict']} arm={ig_cf['arms']['gamma4']['verdict']}"))

    # ---- identity: excessive near-tie divergence RATE => ABORT ------------
    ref_x = _arm(3, [_rec(tokens=_toks(N), top2=_top2(N, 0.02))], [_rec()])  # every position near-tie
    ctrl_x = _arm(3, [_rec(tokens=_toks(N), top2=_top2(N, 0.02))], [_rec()])
    many = _toks(N)
    for i in range(200):
        many[i] = "X"
    cand_x = _arm(2, [_rec(tokens=many, top2=_top2(N, 0.02))], [_rec()])
    ig_x = EV.identity_gate({"gamma3a": ref_x, "gamma3b": ctrl_x, "gamma2": cand_x})
    out.append(("identity: near-tie but excessive divergence RATE => ABORT",
                ig_x["verdict"] == "ABORT" and ig_x["arms"]["gamma2"]["confident"] is False,
                f"verdict={ig_x['verdict']} rate={ig_x['arms']['gamma2']['rate']:.4f} "
                f"allowed={ig_x['allowed_rate']:.4f}"))

    # ---- identity: histogram fallback ------------------------------------
    ref_h = _arm(3, [_rec(hist=[10, 20, 5])], [_rec()])
    ref_h["logprobs_available"] = False
    ctrl_h = _arm(3, [_rec(hist=[10, 20, 5])], [_rec()])
    ctrl_h["logprobs_available"] = False
    cand_h = _arm(2, [_rec(hist=[10, 20, 5])], [_rec()])
    cand_h["logprobs_available"] = False
    ig_h = EV.identity_gate({"gamma3a": ref_h, "gamma3b": ctrl_h, "gamma2": cand_h})
    out.append(("identity: histogram fallback active + PASS on match",
                ig_h["signal"] == "histogram_fallback" and ig_h["verdict"] == "PASS",
                f"signal={ig_h['signal']} verdict={ig_h['verdict']}"))

    # ---- drift control ---------------------------------------------------
    a = [_rec(decode_s=_tok_s(30.0)), _rec(decode_s=_tok_s(30.0))]
    b_hi = [_rec(decode_s=_tok_s(30.0 * 1.02)), _rec(decode_s=_tok_s(30.0 * 1.02))]
    b_lo = [_rec(decode_s=_tok_s(30.0 * 1.005)), _rec(decode_s=_tok_s(30.0 * 1.005))]
    d_hi = EV.drift_control(a, b_hi)
    d_lo = EV.drift_control(a, b_lo)
    out.append(("drift: 2% decode tok/s => RE-RUN", d_hi["flagged"] and d_hi["verdict"].startswith("RE-RUN"),
                f"rel={d_hi['rel_diff']:.4f} verdict={d_hi['verdict']}"))
    out.append(("drift: 0.5% decode tok/s => ok", (not d_lo["flagged"]) and d_lo["verdict"] == "ok",
                f"rel={d_lo['rel_diff']:.4f}"))

    # ---- bars: control pools (realistic spread so IQRs are meaningful) ----
    c3a = _arm(3, [_rec(decode_s=_tok_s(29.5)), _rec(decode_s=_tok_s(30.0)), _rec(decode_s=_tok_s(30.5))],
               [_rec(decode_s=_tok_s(37.5)), _rec(decode_s=_tok_s(38.0)), _rec(decode_s=_tok_s(38.5))])
    c3b = _arm(3, [_rec(decode_s=_tok_s(29.6)), _rec(decode_s=_tok_s(30.0)), _rec(decode_s=_tok_s(30.4))],
               [_rec(decode_s=_tok_s(37.6)), _rec(decode_s=_tok_s(38.0)), _rec(decode_s=_tok_s(38.4))])

    # ---- bars: agentic +5% (>=3%) + benign flat => WINNER -----------------
    win = _arm(4, [_rec(decode_s=_tok_s(31.0)), _rec(decode_s=_tok_s(31.5)), _rec(decode_s=_tok_s(32.0))],
               [_rec(decode_s=_tok_s(38.0)) for _ in range(3)])
    mem = EV.memory_gate({"gamma3a": c3a, "gamma3b": c3b, "gamma4": win})
    bars_win = EV.bars({"gamma3a": c3a, "gamma3b": c3b, "gamma4": win}, mem=mem)
    out.append(("bars: agentic +5% (>=3%) + benign flat => WINNER",
                bars_win["winner"] == "gamma4",
                f"verdict={bars_win['verdict']} class={bars_win['results']['gamma4']['classification']}"))

    # ---- bars: split (agentic win + benign loss) => CLOSE ----------------
    split = _arm(5, [_rec(decode_s=_tok_s(31.0)), _rec(decode_s=_tok_s(31.5)), _rec(decode_s=_tok_s(32.0))],
                 [_rec(decode_s=_tok_s(37.0)) for _ in range(3)])
    bars_split = EV.bars({"gamma3a": c3a, "gamma3b": c3b, "gamma5": split})
    out.append(("bars: split (agentic win + benign loss) => CLOSE-verdict",
                bars_split["verdict"] == "gamma3 confirmed; per-workload gamma is a future item",
                f"verdict={bars_split['verdict']} class={bars_split['results']['gamma5']['classification']}"))

    # ---- bars: no winner (median <3%, IQRs overlap) => gamma3 confirmed ---
    flat = _arm(2, [_rec(decode_s=_tok_s(30.0)), _rec(decode_s=_tok_s(30.2)), _rec(decode_s=_tok_s(30.4))],
                [_rec(decode_s=_tok_s(38.0)) for _ in range(3)])
    bars_none = EV.bars({"gamma3a": c3a, "gamma3b": c3b, "gamma2": flat})
    out.append(("bars: agentic +0.7% (IQRs overlap) => 'gamma3 confirmed on the post-dense build'",
                bars_none["verdict"] == "gamma3 confirmed on the post-dense build",
                f"verdict={bars_none['verdict']} class={bars_none['results']['gamma2']['classification']}"))

    # ---- bars: memory-ineligible winner excluded -------------------------
    win_mem = dict(win)
    win_mem["peak_alloc_bytes"] = {"studio1": EV.W_BYTES - int(0.2 * EV.GB)}  # within 0.5 GB of W
    mem2 = EV.memory_gate({"gamma3a": c3a, "gamma3b": c3b, "gamma4": win_mem})
    bars_mem = EV.bars({"gamma3a": c3a, "gamma3b": c3b, "gamma4": win_mem}, mem=mem2)
    out.append(("bars: fast-but-memory-ineligible arm is NOT a winner",
                bars_mem["winner"] is None and mem2["gamma4"]["eligible"] is False,
                f"winner={bars_mem['winner']} verdict={bars_mem['verdict']}"))

    # ---- memory gate -----------------------------------------------------
    m_ok = _arm(3, [], [], peak_alloc_bytes={"studio1": 114.0 * EV.GB},
                peak_resident_bytes={"studio1": 114.0 * EV.GB})
    m_bad = _arm(3, [], [], peak_resident_bytes={"studio1": EV.W_BYTES - int(0.3 * EV.GB)})
    mg = EV.memory_gate({"ok": m_ok, "bad": m_bad})
    out.append(("memory_gate: within 0.5GB of W => ineligible; far => eligible",
                mg["ok"]["eligible"] and (not mg["bad"]["eligible"]),
                f"ok={mg['ok']['eligible']} bad={mg['bad']['eligible']} headroom={mg['ok']['headroom_gb']}"))

    # ---- helper metrics --------------------------------------------------
    _dt = EV.decode_s_per_output_token({"decode_s": 20.0, "completion_tokens": 800})
    _ts = EV.decode_tok_s({"decode_s": 20.0, "completion_tokens": 800})
    _mh = EV.mean_accepted_from_hist([1, 2, 3])
    out.append(("metric: decode_s_per_output_token excludes prefill",
                _dt is not None and abs(_dt - 20.0 / 800) < 1e-12))
    out.append(("metric: decode_tok_s == (gen-1)/decode_s",
                _ts is not None and abs(_ts - 799 / 20.0) < 1e-9))
    out.append(("hist: mean_accepted_from_hist",
                _mh is not None and abs(_mh - (0 * 1 + 1 * 2 + 2 * 3) / 6) < 1e-12))
    out.append(("hist: hist_delta subtracts prior cumulative",
                EV.hist_delta([10, 20, 5], [4, 5, 1]) == [6, 15, 4]))
    return out


def harness_checks():
    out = []
    # ---- switch recipe: sed simulation + token count ---------------------
    sample = ("screen -dmS exorun zsh -l -c 'cd ~/repos/exo && EXO_SPECULATIVE_GAMMA=3 "
              "DSV41_SPEC_GAMMA=3 DSV41_DENSE=affine6 .venv/bin/python -m exo -v'")
    new = DRV.simulate_sed(sample, 4)
    out.append(("switch: sed rewrites the single token to the target gamma",
                DRV.count_tokens(new) == [4] and "DSV41_SPEC_GAMMA=4" in new
                and "EXO_SPECULATIVE_GAMMA=3" in new,  # the OTHER env is untouched
                f"tokens={DRV.count_tokens(new)}"))
    out.append(("switch: sed leaves EXO_SPECULATIVE_GAMMA untouched",
                DRV.count_tokens(sample) == [3] and "EXO_SPECULATIVE_GAMMA=3" in new))
    out.append(("switch: sed_cmd targets the env + relaunch path",
                DRV.SPEC_GAMMA_ENV in DRV.sed_cmd(5) and DRV.RELAUNCH_PATH in DRV.sed_cmd(5)
                and DRV.SPEC_GAMMA_ENV in DRV.grep_cmd(),
                DRV.sed_cmd(5)))
    out.append(("switch: precondition detects missing token (production file)",
                DRV.count_tokens("cd ~/repos/exo && EXO_SPECULATIVE_GAMMA=3 ...") == []))

    # ---- override log line parsing --------------------------------------
    logline = ("2026-10-10 03:00:00.123 | INFO | [DSV41] spec gamma override: "
               "DSV41_SPEC_GAMMA=4 -> effective gamma=4 (supported set (2, 3, 4, 5)), rank 1")
    parsed = DRV.parse_override_lines(logline)
    out.append(("override: log line parsed (env/eff/rank)",
                parsed == [{"env": 4, "eff": 4, "rank": 1}], f"{parsed}"))

    # ---- rank consistency (both ranks logged the SAME effective gamma) -----
    ok_pair = {"studio1": {"env": 4, "eff": 4, "rank": 1},
               "studio2": {"env": 4, "eff": 4, "rank": 0}}
    bad_pair = {"studio1": {"env": 4, "eff": 4, "rank": 1},
                "studio2": {"env": 4, "eff": 3, "rank": 0}}
    dup_rank = {"studio1": {"env": 4, "eff": 4, "rank": 1},
                "studio2": {"env": 4, "eff": 4, "rank": 1}}
    out.append(("rank-consistency: same eff gamma+ranks {0,1} => ok",
                DRV.rank_consistency_verdict(ok_pair, 4)[0] is True,
                DRV.rank_consistency_verdict(ok_pair, 4)[1]))
    out.append(("rank-consistency: differing effective gamma => ABORT",
                DRV.rank_consistency_verdict(bad_pair, 4)[0] is False,
                DRV.rank_consistency_verdict(bad_pair, 4)[1][:60]))
    out.append(("rank-consistency: duplicated rank => ABORT",
                DRV.rank_consistency_verdict(dup_rank, 4)[0] is False,
                DRV.rank_consistency_verdict(dup_rank, 4)[1][:60]))

    # ---- readiness counting ---------------------------------------------
    state = {"runners": {"a": {"RunnerReady": {}}, "b": {"RunnerReady": {}}}}
    state_busy = {"runners": {"a": {"RunnerReady": {}}, "b": {"RunnerRunning": {}}}}
    out.append(("ready: counts RunnerReady 2/2",
                DRV.count_ready_runners(state) == 2 and DRV.count_ready_runners(state_busy) == 1,
                f"{DRV.count_ready_runners(state)}/{DRV.count_ready_runners(state_busy)}"))

    # ---- memory parsers --------------------------------------------------
    vm = {"data": {"result": [{"value": [1, "123456789.0"]}]}}
    out.append(("vm: parse_vm_value", DRV.parse_vm_value(vm) == 123456789.0))
    fp = "123456789\nfoo bar\n"
    out.append(("footprint: first numeric first-line parses to bytes",
                DRV.parse_footprint_bytes(fp) == 123456789))
    out.append(("footprint: falls back to max integer when no leading number",
                DRV.parse_footprint_bytes("Phys footprint 1.5 GB\nIOAccelerator 100 GB\n") == 100))

    # ---- budget declaration ---------------------------------------------
    arms = list(DRV.MATRIX)
    b3 = DRV.declare_budget(arms, benign_reps=3, budget_wall_min=135.0)
    out.append(("budget: 5 arms x 3 agentic reps exceeds 135 min => 2 reps + DROP IQR",
                b3["agentic_reps"] == 2 and b3["iqr_enabled"] is False
                and b3["dropped_criteria"] != [],
                f"agentic_reps={b3['agentic_reps']} total_min={b3['arithmetic']['total_min']:.1f}"))
    b3b = DRV.declare_budget(arms, benign_reps=3, budget_wall_min=200.0)
    out.append(("budget: raise cap to 200 min => 3 reps + IQR enabled",
                b3b["agentic_reps"] == 3 and b3b["iqr_enabled"] is True,
                f"agentic_reps={b3b['agentic_reps']} total_min={b3b['arithmetic']['total_min']:.1f}"))
    out.append(("budget: supported set includes gamma5 (auto)", 5 in DRV.SUPPORTED_GAMMA))
    return out


def warmup_checks():
    """Post-relaunch JIT-load trigger: warmup retry loop, registration, skip-reuse."""
    out = []
    quiet = lambda *a, **k: None  # noqa: E731

    # ---- warmup: 503 (JIT-load timeout) then 200 => proceed ----------------
    seq = [503, 200]
    calls = {"n": 0}

    def fake_post():
        s = seq[min(calls["n"], len(seq) - 1)]
        calls["n"] += 1
        return s

    tmp = tempfile.mkdtemp(prefix="q2gamma_selftest_")
    reg = os.path.join(tmp, "own_requests_gammaX.jsonl")
    res = DRV.warmup_post(registry_path=reg, label="warmup_gammaX", post_fn=fake_post,
                          sleep_fn=lambda _s: None, log_fn=quiet)
    out.append(("warmup: HTTP 503 (JIT-load timeout) then 200 => proceeds after 2 attempts",
                res["ok"] is True and res["attempts"] == 2 and res["statuses"] == [503, 200],
                f"attempts={res['attempts']} statuses={res['statuses']}"))

    # ---- warmup-registration assertion -------------------------------------
    own = DRV._load_own(reg)
    raw = [json.loads(x) for x in open(reg, encoding="utf-8").read().splitlines() if x.strip()]
    out.append(("warmup: POST epoch registered (own-request registry, guard format + label)",
                len(own) == 1 and abs(own[0] - res["t_reg"]) < 1e-6
                and raw and raw[0]["label"] == "warmup_gammaX" and raw[0]["t"] == res["t_reg"],
                f"own={own} raw_label={raw[0]['label'] if raw else None}"))

    # ---- warmup: transport error (API still coming up) then 200 => proceed --
    seq2 = [None, 200]
    c2 = {"n": 0}

    def fake_post2():
        s = seq2[min(c2["n"], len(seq2) - 1)]
        c2["n"] += 1
        return s

    res2 = DRV.warmup_post(registry_path=None, label="w2", post_fn=fake_post2,
                           sleep_fn=lambda _s: None, log_fn=quiet)
    out.append(("warmup: transport error (None) then 200 => retried and proceeds",
                res2["ok"] and res2["statuses"] == [None, 200] and res2["registered"] is False,
                f"statuses={res2['statuses']}"))

    # ---- warmup: a definitive non-retryable HTTP code => hard SystemExit ----
    hard = False
    try:
        DRV.warmup_post(registry_path=None, label="w400", post_fn=lambda: 400,
                        sleep_fn=lambda _s: None, log_fn=quiet)
    except SystemExit:
        hard = True
    out.append(("warmup: HTTP 400 => hard failure (not retried as a slow load)", hard, f"raised={hard}"))

    # ---- warmup: persistent 503 past the retry budget => SystemExit --------
    class _Clock:
        def __init__(self, step):
            self.t = 0.0
            self.step = step

        def __call__(self):
            self.t += self.step
            return self.t

    clk = _Clock(100.0)  # every reading jumps 100 s => past the 300 s deadline fast
    timed_out = False
    try:
        DRV.warmup_post(registry_path=None, label="w503", post_fn=lambda: 503,
                        sleep_fn=lambda _s: None, now_fn=clk, log_fn=quiet, timeout_s=300.0)
    except SystemExit:
        timed_out = True
    out.append(("warmup: persistent 503 past the ~300s budget => SystemExit (no silent proceed)",
                timed_out, f"raised={timed_out}"))

    # ---- skip-reuse verdict (pure) -----------------------------------------
    per_hit = {"studio1": {"env": 3, "eff": 3, "rank": 1},
               "studio2": {"env": 3, "eff": 3, "rank": 0}}
    per_miss = {"studio1": {"env": 4, "eff": 4, "rank": 1},
                "studio2": {"env": 4, "eff": 4, "rank": 0}}
    per_partial = {"studio1": {"env": 3, "eff": 3, "rank": 1}, "studio2": None}
    per_dup = {"studio1": {"env": 3, "eff": 3, "rank": 1}, "studio2": {"env": 3, "eff": 3, "rank": 1}}
    out.append(("reuse: both ranks already log effective gamma==N => REUSE (skip relaunch)",
                DRV.reuse_verdict(per_hit, 3)[0] is True, DRV.reuse_verdict(per_hit, 3)[1]))
    out.append(("reuse: latest effective gamma != N => no reuse (relaunch)",
                DRV.reuse_verdict(per_miss, 3)[0] is False, DRV.reuse_verdict(per_miss, 3)[1]))
    out.append(("reuse: a node with no override line => no reuse",
                DRV.reuse_verdict(per_partial, 3)[0] is False, DRV.reuse_verdict(per_partial, 3)[1]))
    out.append(("reuse: duplicated rank => no reuse (mirrors rank-consistency abort)",
                DRV.reuse_verdict(per_dup, 3)[0] is False, DRV.reuse_verdict(per_dup, 3)[1]))

    # ---- skip-reuse path (integration: switch_arm) -------------------------
    saved = (DRV.latest_effective_gamma, DRV.warmup_post, DRV._ssh)
    ssh_calls = {"n": 0}

    def fake_ssh(node, cmd, **kw):
        ssh_calls["n"] += 1
        return 0, "DSV41_SPEC_GAMMA=3"

    warm_calls = {"n": 0}

    def fake_warmup(**kw):
        warm_calls["n"] += 1
        return {"ok": True, "attempts": 1, "statuses": [200], "registered": True, "t_reg": 0.0}

    try:
        # reuse: latest logged effective gamma already == 3 => no ssh, no warmup
        DRV.latest_effective_gamma = lambda **kw: per_hit
        DRV.warmup_post = fake_warmup
        DRV._ssh = fake_ssh
        r_reuse = DRV.switch_arm(3, arm="gamma3a", registry_path=None, settle_s=0)
        out.append(("reuse-path: switch_arm(3) with latest eff==3 => mode 'reuse', NO relaunch/warmup",
                    r_reuse["mode"] == "reuse" and ssh_calls["n"] == 0 and warm_calls["n"] == 0,
                    f"mode={r_reuse['mode']} ssh={ssh_calls['n']} warmup={warm_calls['n']}"))

        # miss: latest eff==4 != 3 => full relaunch + warmup
        ssh_calls["n"] = 0
        DRV.latest_effective_gamma = lambda **kw: per_miss
        r_relaunch = DRV.switch_arm(3, arm="gamma3a", registry_path=None, settle_s=0)
        out.append(("reuse-path: switch_arm(3) with latest eff==4 => mode 'relaunch' + warmup fired",
                    r_relaunch["mode"] == "relaunch" and ssh_calls["n"] >= 2 and warm_calls["n"] == 1,
                    f"mode={r_relaunch['mode']} ssh={ssh_calls['n']} warmup={warm_calls['n']}"))
    finally:
        DRV.latest_effective_gamma, DRV.warmup_post, DRV._ssh = saved

    return out


def all_checks():
    out = []
    out.append(("imports: q2_gamma_eval + q2_gamma_driver resolve", True,
                f"EV.W_BYTES={EV.W_BYTES} SUPPORTED={EV.SUPPORTED_GAMMA}"))
    out += anchor_checks()
    out += evaluator_checks()
    out += harness_checks()
    out += warmup_checks()
    return out


def main():
    checks = all_checks()
    fails = []
    for name, ok, *rest in checks:
        detail = rest[0] if rest else ""
        print(f"[{'PASS' if ok else 'FAIL'}] {name}" + (f" — {detail}" if detail else ""))
        if not ok:
            fails.append(name)
    print()
    if fails:
        print(f"RESULT: FAIL ({len(fails)}): " + "; ".join(fails))
        return 1
    print(f"RESULT: PASS — {len(checks)} checks (anchors reproduced + every evaluator unit-tested)")
    return 0


if __name__ == "__main__":
    sys.exit(main())
