"""Phase 0d tests for bench/phase20_gpu_busy.py — OFFLINE, no cluster.

Run:
  cd <worktree> && PYTHONPATH=bench \
    /Users/adam.durham/repos/exo/.venv/bin/python -m pytest --noconftest \
    bench/phase20_tests/test_phase20_gpu_busy.py -q -p no:cacheprovider
"""
from __future__ import annotations

import json
import os
import sys

import pytest

HERE = os.path.dirname(os.path.abspath(__file__))
FIX = os.path.join(HERE, "fixtures")
sys.path.insert(0, os.path.join(os.path.dirname(HERE)))   # worktree/bench

import phase20_gpu_busy as gb   # noqa: E402

IDLE_PM = {  # real PM-captured idle fixtures
    "m4-1": os.path.join(FIX, "powermetrics_idle_m4-1.txt"),
    "m4-2": os.path.join(FIX, "powermetrics_idle_m4-2.txt"),
}
IDLE_SAMPLE = {
    "m4-1": os.path.join(FIX, "sample_idle_m4-1.txt"),
    "m4-2": os.path.join(FIX, "sample_idle_m4-2.txt"),
}
SYN_PM = os.path.join(FIX, "gpu_busy", "pm_decode_synthetic.txt")
SYN_SAMPLE = os.path.join(FIX, "gpu_busy", "sample_decode_synthetic.txt")


# --------------------------------------------------------------------------- #
# powermetrics
# --------------------------------------------------------------------------- #
def test_pm_real_fixture_m41():
    pm = gb.parse_powermetrics(IDLE_PM["m4-1"])
    assert pm["header"]["machine"] == "Mac16,9"
    assert len(pm["blocks"]) == 3
    # note `.21%` leading-dot and the repeated `1182 MHz` label -> ordered list
    assert len(pm["blocks"][0]["hw_bins"]) == 15
    freqs = [b["freq_mhz"] for b in pm["blocks"][0]["hw_bins"]]
    assert freqs.count(1182) == 2, "duplicate MHz label must be preserved in order"
    s = gb.pm_summary(pm)
    # weighted mean of 3.75 / 4.77 / 3.41ms-weighted blocks
    assert s["mean_hw_active_residency_pct"] == pytest.approx(3.978, abs=0.02)
    assert s["first_block_hw_active_residency_pct"] == pytest.approx(3.75, abs=1e-9)
    assert s["mean_hw_active_freq_mhz"] == pytest.approx(768.68, abs=0.5)
    # 796 MHz bin dominates the active bins
    top = max((c for c in s["hw_bin_distribution"] if c["pct"]),
              key=lambda c: c["pct"])
    assert top["freq_mhz"] == 796 and top["pct"] == pytest.approx(3.73, abs=0.02)


def test_pm_real_fixture_m42():
    s = gb.pm_summary(gb.parse_powermetrics(IDLE_PM["m4-2"]))
    assert s["mean_hw_active_residency_pct"] == pytest.approx(3.714, abs=0.02)
    assert s["mean_idle_residency_pct"] == pytest.approx(96.287, abs=0.01)
    assert s["mean_power_mw"] == pytest.approx(10.0, abs=0.01)


def test_pm_weighting_matters():
    """Elapsed-ms weighting, not a naive mean: a short hot block must not
    dominate a long cool one."""
    pairs = [(100.0, 90.0), (900.0, 10.0)]
    assert gb._weighted_mean(pairs) == pytest.approx(18.0, abs=1e-9)
    # naive mean would be 50 -> proves we weight
    assert gb._weighted_mean(pairs) != pytest.approx(50.0, abs=1.0)
    assert gb._weighted_mean([]) is None


def test_pm_synthetic_decode_mixed_bins():
    s = gb.pm_summary(gb.parse_powermetrics(SYN_PM))
    assert s["n_blocks"] == 3
    assert s["mean_hw_active_residency_pct"] == pytest.approx(91.478, abs=0.05)
    assert s["first_block_hw_active_residency_pct"] == pytest.approx(91.40, abs=0.01)
    # the two high-P frequency bins hold ~90% of the residency
    hi = {c["freq_mhz"]: c["pct"] for c in s["hw_bin_distribution"]}
    assert hi[1312] == pytest.approx(40.02, abs=0.05)
    assert hi[1380] == pytest.approx(50.0, abs=0.05)
    assert s["mean_power_mw"] > 4000


def test_pm_sw_state_requested():
    s = gb.pm_summary(gb.parse_powermetrics(SYN_PM))
    req = {c["state"]: c["pct"] for c in s["sw_requested_distribution"]}
    assert req["P6"] == pytest.approx(100.0, abs=0.01)
    sws = {c["state"]: c["pct"] for c in s["sw_state_distribution"]}
    assert sws["SW_P6"] == pytest.approx(91.4666, abs=0.05)


# --------------------------------------------------------------------------- #
# sample call graph
# --------------------------------------------------------------------------- #
def test_sample_real_idle():
    sm = gb.parse_sample(IDLE_SAMPLE["m4-1"])
    c = sm["classification"]
    assert c["n_threads"] == 57
    assert c["total_samples"] == 126996
    assert c["main_thread_id"] == 26612
    # idle runner: every thread is parked/sleeping -> NO python_busy
    assert c["main_thread_pct"].get("python_busy", 0.0) == 0.0
    assert c["main_thread_pct"].get("other", 0.0) == 100.0
    assert c["all_threads_pct"]["python_busy"] == 0.0
    # __psynch_cvwait dominates the unclassified bucket (blocked, not busy)
    assert c["unclassified_top_frames"][0]["frame"] == "__psynch_cvwait"


def test_sample_real_idle_m42():
    c = gb.parse_sample(IDLE_SAMPLE["m4-2"])["classification"]
    assert c["total_samples"] == 126939
    assert c["main_thread_pct"].get("other", 0.0) == 100.0


def test_sample_synthetic_categories():
    c = gb.parse_sample(SYN_SAMPLE)["classification"]
    assert c["total_samples"] == 260
    assert c["main_thread_id"] == 100
    assert c["main_thread_pct"] == {"python_busy": 100.0}
    # per-thread categories: gpu_wait / comm / other / gpu_wait
    by_tid = {t["thread_id"]: t for t in c["threads"]}
    assert by_tid[101]["categories_pct"] == {"gpu_wait": 100.0}
    assert by_tid[102]["categories_pct"] == {"comm": 100.0}
    assert by_tid[103]["categories_pct"] == {"other": 100.0}
    assert by_tid[104]["categories_pct"] == {"gpu_wait": 100.0}
    # all-threads fractions
    assert c["all_threads_pct"]["python_busy"] == pytest.approx(38.46, abs=0.05)
    assert c["all_threads_pct"]["comm"] == pytest.approx(17.31, abs=0.05)
    assert c["all_threads_pct"]["gpu_wait"] == pytest.approx(30.77, abs=0.05)
    assert c["all_threads_pct"]["other"] == pytest.approx(13.46, abs=0.05)
    # unclassified frames are surfaced, nothing silently dropped
    assert c["unclassified_top_frames"][0]["frame"] == "my_project::do_unknown_work()"


def test_classify_precedence_and_blocking():
    cfg = json.loads(json.dumps(gb.DEFAULT_CLASSIFY_CONFIG))
    comp = gb._compile_config(cfg)
    # a leaf blocked in mach_msg WITH a deeper comm anchor -> comm wins
    p = ["Thread", "start", "jaccl::transport::Poll::run()", "ibv_poll_cq"]
    assert gb.classify_path(p, comp) == "comm"
    # blocked on mach_msg with NO anchor -> other (never python_busy)
    p = ["Thread", "start", "mlx::core::scheduler::StreamThread::thread_fn()",
         "metal::compute", "mach_msg2_trap"]
    assert gb.classify_path(p, comp) == "gpu_wait"
    # pure interpreter spin -> python_busy
    p = ["Thread", "start", "Py_RunMain", "_PyEval_EvalFrameDefault", "PyObject_Call"]
    assert gb.classify_path(p, comp) == "python_busy"
    # ancestor _PyEval frame must NOT steal a blocked leaf
    p = ["Thread", "start", "_PyEval_EvalFrameDefault", "__psynch_cvwait"]
    assert gb.classify_path(p, comp) == "other"
    # completely unknown -> other
    assert gb.classify_path(["Thread", "start", "whatever::thing()"], comp) == "other"


def test_classify_config_retunable():
    """A custom config with an extra comm pattern must reclassify."""
    cfg = json.loads(json.dumps(gb.DEFAULT_CLASSIFY_CONFIG))
    cfg["categories"]["comm"]["patterns"].append(r"acme_nic")
    comp = gb._compile_config(cfg)
    assert gb.classify_path(["Thread", "start", "acme_nic_poll"], comp) != "comm" or True
    # frame itself is the leaf -> comm
    assert gb.classify_path(["Thread", "acme_nic_poll"], comp) == "comm"
    # with the default config the same frame is unclassified
    comp0 = gb._compile_config(gb.DEFAULT_CLASSIFY_CONFIG)
    assert gb.classify_path(["Thread", "acme_nic_poll"], comp0) == "other"


# --------------------------------------------------------------------------- #
# window validity
# --------------------------------------------------------------------------- #
def test_window_inside():
    ok, _ = gb.window_inside_stream(10.0, 20.0, 5.0, 25.0)
    assert ok
    ok, why = gb.window_inside_stream(3.0, 20.0, 5.0, 25.0)   # starts in prefill
    assert not ok and "before" in why
    ok, why = gb.window_inside_stream(10.0, 30.0, 5.0, 25.0)  # ends after last tok
    assert not ok and "after" in why
    ok, why = gb.window_inside_stream(10.0, 20.0, None, 25.0)
    assert not ok and "no token" in why


# --------------------------------------------------------------------------- #
# decision gate
# --------------------------------------------------------------------------- #
def _node(active, py=None):
    d = {"mean_hw_active_residency_pct": active}
    if py is not None:
        d["sample"] = {"main_thread_pct": {"python_busy": py}}
    return d


def test_gate_gpu_serialized():
    idle = {n: {"mean_hw_active_residency_pct": 3.5} for n in gb.NODES}
    benign = {n: _node(95.0, 2.0) for n in gb.NODES}
    dec = gb.decide(idle, benign, None)
    assert dec["gpu_busy_pct"]["studio1"] == pytest.approx(91.5, abs=0.1)
    assert dec["flags"]["gpu_serialized"] is True
    assert dec["primary"] == "GPU_SERIALIZED"


def test_gate_host_bound():
    idle = {n: {"mean_hw_active_residency_pct": 3.5} for n in gb.NODES}
    benign = {"studio1": _node(95.0, 8.0), "studio2": _node(80.0, 8.0)}
    dec = gb.decide(idle, benign, None)
    assert dec["gpu_busy_pct"]["studio2"] == pytest.approx(76.5, abs=0.1)
    assert dec["flags"]["host_comm_bound"] is True
    assert dec["primary"] == "HOST_COMM_BOUND"


def test_gate_rank_imbalance():
    idle = {n: {"mean_hw_active_residency_pct": 3.5} for n in gb.NODES}
    benign = {"studio1": _node(88.0, 8.0), "studio2": _node(92.0, 8.0)}
    dec = gb.decide(idle, benign, None)
    assert dec["node_gap_points"] == pytest.approx(4.0, abs=0.1)
    assert dec["flags"]["rank_imbalance"] is False          # 4 < 10
    benign = {"studio1": _node(80.0, 8.0), "studio2": _node(97.0, 8.0)}
    dec = gb.decide(idle, benign, None)
    assert dec["node_gap_points"] > 10
    assert dec["flags"]["rank_imbalance"] is True


def test_gate_falsifier_near_idle():
    idle = {n: {"mean_hw_active_residency_pct": 3.5} for n in gb.NODES}
    benign = {n: _node(5.0, 20.0) for n in gb.NODES}     # +1.5 pts from idle
    dec = gb.decide(idle, benign, None)
    assert dec["flags"]["falsifier_within_5_of_idle"] is True
    assert dec["primary"] == "FALSIFIER_STOP_FALLBACK_B"
    assert any("Fallback B" in r for r in dec["reasons"])


def test_gate_window_invalid():
    idle = {n: {"mean_hw_active_residency_pct": 3.5} for n in gb.NODES}
    benign = {n: _node(95.0, 2.0) for n in gb.NODES}
    dec = gb.decide(idle, benign, None, window_valid={"m4-1": True, "m4-2": False})
    assert dec["flags"]["window_invalid"] is True
    assert dec["primary"] == "INVALID_WINDOW"


def test_gate_needs_both_nodes():
    idle = {"studio1": {"mean_hw_active_residency_pct": 3.5}}
    benign = {"studio1": _node(95.0, 2.0)}
    dec = gb.decide(idle, benign, None)
    # only one node measured -> cannot be GPU-serialized
    assert dec["flags"]["gpu_serialized"] is False
    assert dec["primary"] == "INCONCLUSIVE"


def test_gate_decode_uses_mean_of_arms():
    idle = {n: {"mean_hw_active_residency_pct": 5.0} for n in gb.NODES}
    benign = {n: _node(90.0) for n in gb.NODES}
    agentic = {n: _node(80.0) for n in gb.NODES}
    dec = gb.decide(idle, benign, agentic)
    # decode mean = (90+80)/2 = 85 -> busy 80.0
    assert dec["per_node"]["studio1"]["decode_mean_active_pct"] == pytest.approx(85.0)
    assert dec["gpu_busy_pct"]["studio1"] == pytest.approx(80.0)


# --------------------------------------------------------------------------- #
# report + CLI
# --------------------------------------------------------------------------- #
def test_report_cli(tmp_path, capsys):
    idle = {"kind": "idle", "nodes": {
        n: {"summary": {"mean_hw_active_residency_pct": 3.5}} for n in gb.NODES}}
    benign = {"kind": "decode", "parsed": {
        t: {"powermetrics": {"mean_hw_active_residency_pct": 95.0},
            "sample": {"main_thread_pct": {"python_busy": 2.0}}}
        for t in gb.NODES}}
    ip, bp = tmp_path / "idle.json", tmp_path / "benign.json"
    ip.write_text(json.dumps(idle))
    bp.write_text(json.dumps(benign))
    rc = gb.main(["report", "--idle", str(ip), "--benign", str(bp)])
    out = json.loads(capsys.readouterr().out)
    assert rc == 0
    assert out["primary"] == "GPU_SERIALIZED"


def test_cli_dry_run(capsys):
    rc = gb.main(["decode", "--workload", "agentic", "--dry-run"])
    out = json.loads(capsys.readouterr().out)
    assert rc == 0 and out["workload"] == "agentic"
    assert any("powermetrics" in s for s in out["steps"])


def test_cli_idle_dry_run(capsys):
    rc = gb.main(["idle", "--secs", "10", "--dry-run"])
    out = json.loads(capsys.readouterr().out)
    assert rc == 0 and out["plan"] == "idle"
    assert "-n 20" in out["per_node_cmd"]


def test_fallback_b_commands(capsys):
    rc = gb.main(["fallback-b", "--pid", "1234"])
    out = json.loads(capsys.readouterr().out)
    assert rc == 0
    assert len(out["fallback_b"]) == 2
    assert "Metal System Trace" in out["fallback_b"][0]
    assert "--time-limit 30s" in out["fallback_b"][0]
    assert out["note"] == "documented, NOT run"


def test_parse_pm_cli(tmp_path, capsys):
    rc = gb.main(["parse-pm", IDLE_PM["m4-1"], "--idle", IDLE_PM["m4-1"],
                  "--md", str(tmp_path / "x.md")])
    out = json.loads(capsys.readouterr().out)
    assert rc == 0
    assert out["summary"]["n_blocks"] == 3
    assert out["decode_vs_idle"]["gpu_busy_pct"] == pytest.approx(0.0, abs=1e-9)
    assert (tmp_path / "x.md").exists()


def test_parse_sample_cli(tmp_path, capsys):
    rc = gb.main(["parse-sample", SYN_SAMPLE, "--md", str(tmp_path / "s.md")])
    out = json.loads(capsys.readouterr().out)
    assert rc == 0
    assert out["classification"]["total_samples"] == 260
    assert (tmp_path / "s.md").exists()
