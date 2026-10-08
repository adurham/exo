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
import threading
import time
import types

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
    # the frame itself is the leaf -> comm
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


# --------------------------------------------------------------------------- #
# FIX 2 — reasoning_content stream (reasoning model: delta.reasoning_content)
# --------------------------------------------------------------------------- #
class _FakeSSEResponse:
    """Context-manager, line-iterable stand-in for an urllib response."""

    def __init__(self, lines):
        self._lines = lines

    def __enter__(self):
        return self

    def __exit__(self, *exc):
        return False

    def __iter__(self):
        for ln in self._lines:
            yield (ln + "\n").encode("utf-8")


def _sse(*frames):
    return list(frames)


def test_stream_reasoning_content_only_fires_first_token(monkeypatch):
    """The real model streams ONLY delta.reasoning_content (content=''); the
    stream client must accumulate it and fire on_first_token / set ttft on it."""
    seen = {}
    lines = _sse(
        'data: {"choices":[{"delta":{"role":"assistant","content":""}}]}',
        'data: {"choices":[{"delta":{"reasoning_content":"We"}}]}',
        'data: {"choices":[{"delta":{"reasoning_content":" need"}}]}',
        ': generation_stats {"prompt_tps": 123.4, "prompt_tokens": 100, '
        '"generation_tokens": 2, "prefix_cache_hit": true}',
        'data: [DONE]',
    )
    monkeypatch.setattr(gb.urllib.request, "urlopen",
                        lambda req, timeout=None: _FakeSSEResponse(lines))

    def on_first(epoch, resp):
        seen["epoch"] = epoch

    out = gb.stream_once("hi", 8, on_first_token=on_first)
    assert "epoch" in seen, "on_first_token must fire on a reasoning_content-only stream"
    assert out["first_token_epoch"] is not None
    assert out["ttft_s"] is not None and out["ttft_s"] >= 0.0
    assert out["last_token_epoch"] is not None
    # both reasoning fragments counted as output text
    assert out["content_chars"] == len("We") + len(" need")
    assert out["stats"]["prefix_cache_hit"] is True
    assert out["stats"]["prompt_tps"] == pytest.approx(123.4)


def test_stream_content_and_reasoning_both_accumulate(monkeypatch):
    """A stream carrying both kinds fires once, on the first non-empty delta."""
    calls = []
    lines = _sse(
        'data: {"choices":[{"delta":{"content":"Hello"}}]}',
        'data: {"choices":[{"delta":{"reasoning_content":"thinking"}}]}',
        'data: [DONE]',
    )
    monkeypatch.setattr(gb.urllib.request, "urlopen",
                        lambda req, timeout=None: _FakeSSEResponse(lines))
    out = gb.stream_once("hi", 8, on_first_token=lambda e, r: calls.append(e))
    assert len(calls) == 1
    assert out["content_chars"] == len("Hello") + len("thinking")


# --------------------------------------------------------------------------- #
# FIX 1 — cmd_decode wires the persistent own-request registry into ChunkGuard
# --------------------------------------------------------------------------- #
class _FakeGuard:
    """Records the ChunkGuard kwargs cmd_decode is (not) passing."""

    last_kwargs: dict | None = None

    def __init__(self, label, **kwargs):
        type(self).last_kwargs = {"label": label, **kwargs}
        self.cancel_event = threading.Event()
        self.registered = 0

    def __enter__(self):
        return self

    def __exit__(self, *exc):
        return False

    def register_own_request(self, t=None):
        self.registered += 1


def _fake_guard_module():
    m = types.ModuleType("phase20_guard")
    m.ChunkGuard = _FakeGuard
    return m


def test_cmd_decode_loads_registry_and_passes_to_guard(tmp_path, monkeypatch,
                                                       capsys):
    """cmd_decode must load the persistent registry and pass BOTH own_requests
    (non-empty, from the file) and registry_path to ChunkGuard -- otherwise the
    guard's entry idle-check stalls on this tool's own prior POSTs (FIX 1)."""
    outdir = tmp_path / "out"
    raw = outdir / "raw"
    raw.mkdir(parents=True)
    reg = raw / gb.DEFAULT_REGISTRY_NAME
    reg.write_text(
        '{"label": "phase20-gpu-busy-benign", "t": 1000.0}\n'
        '{"label": "phase20-delta-ladder", "t": 1234.5}\n'
        "not-json-tolerated\n"
        "\n")
    # no cluster, and do NOT write into the repo's real RAW_DIR
    monkeypatch.setattr(gb, "stream_once", lambda *a, **k: {"usage": None})
    monkeypatch.setattr(gb, "RAW_DIR", str(tmp_path / "raw_out"))
    import sys as _sys
    monkeypatch.setitem(_sys.modules, "phase20_guard", _fake_guard_module())

    rc = gb.main(["decode", "--workload", "benign", "--depth", "1000",
                  "--outdir", str(outdir)])
    capsys.readouterr()
    assert rc == 0
    kw = _FakeGuard.last_kwargs
    assert kw is not None, "ChunkGuard was never constructed"
    assert kw.get("own_requests") == [1000.0, 1234.5], \
        "own_requests must be the registry epochs, loaded from the file"
    assert kw.get("registry_path") == str(reg), \
        "registry_path must point at the default <outdir>/raw/ registry"


def test_cmd_decode_own_registry_override(tmp_path, monkeypatch, capsys):
    """--own-registry PATH overrides the default registry location."""
    outdir = tmp_path / "out"
    alt = tmp_path / "shared_registry.jsonl"
    alt.write_text('{"label": "x", "t": 42.0}\n')
    monkeypatch.setattr(gb, "stream_once", lambda *a, **k: {"usage": None})
    monkeypatch.setattr(gb, "RAW_DIR", str(tmp_path / "raw_out"))
    import sys as _sys
    monkeypatch.setitem(_sys.modules, "phase20_guard", _fake_guard_module())

    rc = gb.main(["decode", "--workload", "benign", "--depth", "1000",
                  "--outdir", str(outdir), "--own-registry", str(alt)])
    capsys.readouterr()
    assert rc == 0
    kw = _FakeGuard.last_kwargs
    assert kw["registry_path"] == str(alt)
    assert kw["own_requests"] == [42.0]


# --------------------------------------------------------------------------- #
# FIX 2c — a no-first-token stream must not crash window validity
# --------------------------------------------------------------------------- #
def test_window_validity_survives_none_ttft():
    """A stream that yields no token at all -> ttft/first/last None; the window
    check must return (False, ...) not raise, and the record stays writable."""
    first = last = None
    ok, why = gb.window_inside_stream(10.0, 20.0, first, last)
    assert ok is False
    assert "no token" in why
    # the shape cmd_decode feeds JSON is still serializable
    rec = {"decode": {"ttft_s": None, "first_token_epoch": first,
                      "last_token_epoch": last, "decode_s": None},
           "window_valid": {"m4-1": ok}}
    assert json.loads(json.dumps(rec))["window_valid"]["m4-1"] is False


def test_load_own_requests_tolerates_missing_and_garbage(tmp_path):
    """The loader must never fail a run: missing file -> [], garbage skipped."""
    assert gb.load_own_requests(str(tmp_path / "nope.jsonl")) == []
    assert gb.load_own_requests(None) == []
    p = tmp_path / "r.jsonl"
    p.write_text('\n{"t": 1.5}\n{bad\n{"label": "no-t"}\n{"t": "2.5"}\n')
    assert gb.load_own_requests(str(p)) == [1.5, 2.5]


# --------------------------------------------------------------------------- #
# FIX 2 — non-blocking samplers + consistent first-token callback signature
# --------------------------------------------------------------------------- #
def _install_fake_guard(monkeypatch):
    import sys as _sys
    monkeypatch.setitem(_sys.modules, "phase20_guard", _fake_guard_module())


def _run_decode_with_fakes(tmp_path, monkeypatch, fake_stream, *,
                           ssh_sleep=0.5,
                           pm_windows=None):
    """Drive the real cmd_decode with a fake guard + fake ssh captures.

    ``fake_stream`` stands in for stream_once; it calls the callback exactly
    as the real one does (``on_first_token(now, resp)`` -- TWO args).
    Returns ``(rc, rec, capture_calls)`` where ``rec`` is the written JSON.
    """
    pm_windows = pm_windows or {"studio1": (1234.6, 1235.4),
                                "studio2": (1234.6, 1235.4)}
    syn_pm = open(SYN_PM).read()
    syn_sample = open(SYN_SAMPLE).read()
    capture_calls: list[str] = []

    def fake_ssh(node, remote_cmd, out_path, timeout):
        capture_calls.append(node)
        time.sleep(ssh_sleep)                     # a slow window capture
        content = syn_pm if out_path.endswith("powermetrics.txt") else syn_sample
        open(out_path, "w").write(content)
        start, end = pm_windows.get(node, (1234.6, 1235.4))
        return {"node": node, "cmd": remote_cmd, "out": out_path,
                "start_epoch": start, "end_epoch": end,
                "returncode": 0, "stderr": ""}

    monkeypatch.setattr(gb, "stream_once", fake_stream)
    monkeypatch.setattr(gb, "_ssh_capture", fake_ssh)
    monkeypatch.setattr(gb, "_runner_pid", lambda node, timeout=30: 4321)
    monkeypatch.setattr(gb, "RAW_DIR", str(tmp_path / "raw_out"))
    _install_fake_guard(monkeypatch)

    outdir = tmp_path / "out"
    rc = gb.main(["decode", "--workload", "benign", "--depth", "1000",
                  "--secs", "50", "--outdir", str(outdir)])
    jpath = tmp_path / "raw_out" / "gpu_busy.benign.json"
    rec = json.loads(jpath.read_text()) if jpath.exists() else None
    return rc, rec, capture_calls


def test_decode_first_token_callback_two_args_nonblocking(tmp_path, monkeypatch,
                                                           capsys):
    """Pre-fix, stream_once calls ``on_first_token(now, resp)`` (2 args) but the
    callback took 1 -> TypeError.  Post-fix the callback accepts
    ``(first_epoch, resp)``, records the epoch, spawns the capture thread(s) and
    RETURNS PROMPTLY -- it does not join the slow ssh capture inline."""
    cb_elapsed = {}

    def fake_stream(prompt, max_tokens, on_first_token=None, stop_event=None,
                    **kw):
        if on_first_token is not None:            # None == the warmup feed
            t0 = time.perf_counter()
            on_first_token(1234.5, object())      # exactly what stream_once does
            cb_elapsed["s"] = time.perf_counter() - t0
        return {"wall_s": 1.0, "ttft_s": 0.1, "first_token_epoch": 1234.5,
                "last_token_epoch": 1235.5, "decode_s": 1.0,
                "content_chars": 10, "usage": None, "stats": None}

    rc, rec, calls = _run_decode_with_fakes(tmp_path, monkeypatch, fake_stream,
                                            ssh_sleep=0.5)
    capsys.readouterr()
    assert rc == 0
    assert "s" in cb_elapsed, "callback was never invoked"
    # the slow capture takes 0.5 s; a non-blocking callback must be far under it
    assert cb_elapsed["s"] < 0.2, \
        f"first-token callback blocked the read loop for {cb_elapsed['s']:.3f}s"
    # BOTH node samplers were spawned from the single inline callback
    # (two captures per node: powermetrics + sample)
    assert sorted(set(calls)) == ["studio1", "studio2"]
    assert len(calls) == 4
    assert rec is not None


def test_decode_background_samplers_collected_after_stream(tmp_path, monkeypatch,
                                                            capsys):
    """The ssh captures run in the BACKGROUND; their results are joined and
    attached to the record only after the stream ends, before parsing."""
    released = {"stream_done": False}

    def fake_stream(prompt, max_tokens, on_first_token=None, stop_event=None,
                    **kw):
        if on_first_token is not None:            # None == the warmup feed
            assert not released["stream_done"]
            on_first_token(1234.5, object())
            released["stream_done"] = True        # stream returns -> join happens next
        return {"wall_s": 1.0, "ttft_s": 0.1, "first_token_epoch": 1234.5,
                "last_token_epoch": 1235.5, "decode_s": 1.0,
                "content_chars": 10, "usage": None, "stats": None}

    rc, rec, calls = _run_decode_with_fakes(tmp_path, monkeypatch, fake_stream,
                                            ssh_sleep=0.3)
    capsys.readouterr()
    assert rc == 0
    # background captures were collected and parsed into the record
    # (node_out / parsed are keyed by the node TAG: m4-1, m4-2)
    for tag in ("m4-1", "m4-2"):
        entry = rec["parsed"][tag]
        assert entry["powermetrics"]["n_blocks"] == 3
        assert entry["sample"]["total_samples"] == 260
    # recorded JSON schema keys stay stable
    assert set(rec["decode"]) == {"wall_s", "ttft_s", "first_token_epoch",
                                  "last_token_epoch", "decode_s",
                                  "content_chars", "usage", "stats"}


def test_decode_window_validity_end_to_end(tmp_path, monkeypatch, capsys):
    """Window-validity is still computed from each node's powermetrics
    [start,end] against [first_token,last_token]: inside -> True, a window that
    starts before the first token -> False ('before')."""
    def fake_stream(prompt, max_tokens, on_first_token=None, stop_event=None,
                    **kw):
        if on_first_token is not None:            # None == the warmup feed
            on_first_token(1234.5, object())
        return {"wall_s": 1.0, "ttft_s": 0.1, "first_token_epoch": 1234.5,
                "last_token_epoch": 1235.5, "decode_s": 1.0,
                "content_chars": 10, "usage": None, "stats": None}

    rc, rec, _ = _run_decode_with_fakes(
        tmp_path, monkeypatch, fake_stream, ssh_sleep=0.05,
        pm_windows={"studio1": (1234.6, 1235.4),   # fully inside
                    "studio2": (1230.0, 1235.4)})  # starts in prefill
    capsys.readouterr()
    assert rc == 0
    assert rec["window_valid"] == {"m4-1": True, "m4-2": False}
    valid, why = gb.window_inside_stream(1230.0, 1235.4, 1234.5, 1235.5)
    assert valid is False and "before" in why


# --------------------------------------------------------------------------- #
# FIX 3 — window-validity epoch UNITS: decode epochs must be WALL CLOCK
# (time.time()), matching the sampler capture start_epoch/end_epoch.
# Pre-fix, stream_once stamped first/last_token_epoch with time.perf_counter()
# (monotonic, ~seconds-since-boot) while _ssh_capture used time.time()
# (~1.79e9 = 2026 wall seconds); window_inside_stream then compared
# incomparable magnitudes and could never return True.
# --------------------------------------------------------------------------- #
def test_stream_once_epochs_are_wall_clock(monkeypatch):
    """The units test: stream_once must stamp first/last_token_epoch in the
    SAME units as the sampler side -- wall clock (time.time()).  A monotonic
    perf_counter value (seconds since boot) is orders of magnitude below the
    1.79e9 wall epoch, so this assertion fails pre-fix."""
    lines = _sse(
        'data: {"choices":[{"delta":{"content":"Hello"}}]}',
        'data: {"choices":[{"delta":{"content":" world"}}]}',
        'data: [DONE]',
    )
    monkeypatch.setattr(gb.urllib.request, "urlopen",
                        lambda req, timeout=None: _FakeSSEResponse(lines))
    t_before = time.time()
    out = gb.stream_once("hi", 8)
    t_after = time.time()

    first, last = out["first_token_epoch"], out["last_token_epoch"]
    assert first is not None and last is not None
    # same clock as _ssh_capture's start_epoch/end_epoch (time.time()):
    assert t_before <= first <= t_after, (
        f"first_token_epoch {first!r} is not a wall-clock epoch in "
        f"[{t_before}, {t_after}] (perf_counter/monotonic leak?)")
    assert t_before <= last <= t_after, (
        f"last_token_epoch {last!r} is not a wall-clock epoch in "
        f"[{t_before}, {t_after}] (perf_counter/monotonic leak?)")
    # and within the same magnitude as a real sampler capture epoch
    assert abs(first - time.time()) < 300.0 and abs(last - time.time()) < 300.0
    # ttft/decode_s stay small durations (they are differences, not epochs)
    assert 0.0 <= out["ttft_s"] < 60.0
    assert 0.0 <= out["decode_s"] < 60.0


def test_window_inside_same_unit_wall_clock_magnitudes():
    """window_inside_stream must be meaningful with REAL wall-clock epochs
    (~1.79e9).  A sampler window strictly inside the decode -> True; a window
    overhanging the decode end -> False."""
    A, B = 1.79e9, 1.79e9 + 21.8           # decode [first token, last token]
    a, b = A + 0.7, A + 20.4              # sampler window strictly inside
    ok, why = gb.window_inside_stream(a, b, A, B)
    assert ok is True, why
    # overhanging the decode END -> False
    ok, why = gb.window_inside_stream(a, B + 3.0, A, B)
    assert ok is False and "after" in why
    # overhanging the decode START (prefill contamination) -> False
    ok, why = gb.window_inside_stream(A - 1.0, b, A, B)
    assert ok is False and "before" in why


def test_decode_window_validity_wall_clock_end_to_end(tmp_path, monkeypatch,
                                                       capsys):
    """End-to-end through cmd_decode with wall-clock epochs on BOTH sides (as
    now recorded post-fix): a sampler window inside the decode -> True, one
    overhanging the decode end -> False."""
    A, B = 1.79e9, 1.79e9 + 21.8

    def fake_stream(prompt, max_tokens, on_first_token=None, stop_event=None,
                    **kw):
        if on_first_token is not None:            # None == the warmup feed
            on_first_token(A, object())
        return {"wall_s": 22.0, "ttft_s": 0.7, "first_token_epoch": A,
                "last_token_epoch": B, "decode_s": round(B - A, 3),
                "content_chars": 10, "usage": None, "stats": None}

    rc, rec, _ = _run_decode_with_fakes(
        tmp_path, monkeypatch, fake_stream, ssh_sleep=0.02,
        pm_windows={"studio1": (A + 0.7, A + 20.4),   # inside [A, B]
                    "studio2": (A + 0.7, B + 3.0)})   # overhangs the end
    capsys.readouterr()
    assert rc == 0
    assert rec["window_valid"] == {"m4-1": True, "m4-2": False}
