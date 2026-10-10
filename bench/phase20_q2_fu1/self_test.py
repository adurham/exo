#!/usr/bin/env python3
"""Offline self-test for the Q2-FU1 capture hook + engine bootstrap + dumps.

NO GPU, NO checkpoint, NO cluster: a tiny REAL ``deepseek_v41`` Model (CPU,
6 layers, 256-token vocab, real ModelCache / compressor / indexer / sparse
attention) is driven through the REAL ``Dsv41Engine`` request path, with the
hook installed exactly as the engine bootstrap installs it in production.

Proves:
  1. ``_maybe_install_q2fu1_capture`` installs when the env names a dir and is
     a NO-OP when unset (module never imported twice, no cost).
  2. The class-level monkeypatches fire: Model / Block / Gate / ShardedHead
     wrappers all capture (every capture record carries idx_L / w_L / hid_L /
     margin_L for every body layer, hidden_in/out, logits).
  3. A dump npz has the exact pinned keys + shapes.
  4. Faithfulness: an ARMED request (sweep ran: ~29 diagnostic forwards +
     full restores) emits the byte-identical token stream + logprobs of an
     UNARMED request on the same model.
  5. Integrity: the sweep's own cache digest before/after the sweep matches
     and the post-sweep M=1 forward reproduces chain step 0 bitwise.
  6. Cross-shape read-out works: row i of the M-row forward equals the M=1
     sequential step i (CPU float-noise tolerance).
  7. The rollback A/B arm discriminates: with accepted=1, M=3 (rejected slot
     aliases a dead position) A == B; M=4/5/6 (a rejected slot aliases a live
     read) A != B -- the window-ring clobber, offline.

Run:
  cd <worktree> && PYTHONPATH=bench/phase20_q2_fu1 \\
    /Users/adam.durham/repos/exo/.venv/bin/python bench/phase20_q2_fu1/self_test.py
"""
from __future__ import annotations

import importlib.util
import json
import os
import sys
import tempfile
from pathlib import Path

HERE = Path(__file__).resolve().parent
REPO = HERE.parent.parent
sys.path.insert(0, str(REPO / "src"))
sys.path.insert(0, str(REPO / "mlx-lm"))   # the pinned submodule (689e4ea)
sys.path.insert(0, str(HERE))

import numpy as np  # noqa: E402

import mlx.core as mx  # noqa: E402

mx.set_default_device(mx.cpu)  # before exo/mlx_lm import anything GPU-bound

# exo.shared.constants resolves the dashboard dir at import; point it at a
# harmless placeholder path (it is never served in this offline harness).
os.environ.setdefault("EXO_DASHBOARD_DIR", "nonexistent-dashboard-assets")

import mlx.nn as nn  # noqa: E402

from mlx_lm.models.deepseek_v41 import exl3_build as M_BUILD  # noqa: E402
from mlx_lm.models.deepseek_v41.config import ModelArgs  # noqa: E402
from mlx_lm.models.deepseek_v41.model import Model  # noqa: E402

PROMPT_LEN = 40
N_CHECKS = [0]


def check(cond: bool, msg: str) -> None:
    N_CHECKS[0] += 1
    if not cond:
        raise AssertionError(f"SELF-TEST FAILED: {msg}")
    print(f"  ok  {msg}")


def _load_conftest():
    path = REPO / "src" / "exo" / "worker" / "engines" / "mlx" / "dsv41" / "tests" / "conftest.py"
    spec = importlib.util.spec_from_file_location("dsv41_tests_conftest", path)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


# ---------------------------------------------------------------------------
# tiny real model + a world=1 ShardedHead lookalike (the hook patches the
# class, so substituting the class exercises the real monkeypatch machinery)
# ---------------------------------------------------------------------------
class FakeShardedHead(nn.Module):
    def __init__(self, dim: int, vocab: int, rank: int = 0, world: int = 1):
        super().__init__()
        self._proj = nn.Linear(dim, vocab // world, bias=False)
        self._lo = rank * (vocab // world)
        self._world = world
        self._vocab = vocab

    def local(self, h):
        return self._proj(h).astype(mx.float32)

    def combine_argmax(self, y):
        return mx.argmax(y, axis=-1).astype(mx.int32)

    def argmax(self, h):
        return self.combine_argmax(self.local(h))

    def topk_logprobs(self, h, k: int):
        from mlx_lm.models.deepseek_v41 import logprobs as _lp
        y = self.local(h)
        shp = y.shape[:-1]
        ids, sel, tid, tlp = _lp.from_logits(y.reshape(-1, y.shape[-1]), k)
        return ids.reshape(shp), {"selected": sel.reshape(shp),
                                  "top_ids": tid.reshape(*shp, tid.shape[-1]),
                                  "top_logprobs": tlp.reshape(*shp, tlp.shape[-1])}

    def __call__(self, h):
        return self.local(h)          # world == 1: the full row


def toy_args() -> ModelArgs:
    return ModelArgs(
        vocab_size=256, dim=64, n_layers=6, n_heads=2, head_dim=128,
        rope_head_dim=16, q_lora_rank=32, o_lora_rank=32, o_groups=2,
        window_size=8, moe_inter_dim=32, n_routed_experts=8,
        n_shared_experts=1, n_activated_experts=2,
        compress_ratios=(0, 2, 2, 1, 1, 1), kv_source_layers=(1, 3),
        index_source_layers=(1, 3, 5), index_n_heads=2, index_head_dim=32,
        index_topk=8, hc_mult=2, max_seq_len=256,
        dspark_target_layer_ids=(0, 1, 2), dspark_noise_token_id=250)


def build_model() -> Model:
    mx.random.seed(7)
    m = Model(toy_args())
    m.head = FakeShardedHead(toy_args().dim, toy_args().vocab_size)
    mx.eval(m.parameters())
    return m


# ---------------------------------------------------------------------------
# engine driver helpers
# ---------------------------------------------------------------------------
class _Sender:
    def __init__(self):
        self.events = []

    def send(self, item):
        self.events.append(item)


class _Receiver:
    def collect(self):
        return []


def build_engine(model, tokenizer):
    from exo.shared.types.common import ModelId
    from exo.worker.engines.mlx.dsv41.engine import Dsv41Engine
    from exo.worker.engines.mlx.dsv41.load import Dsv41Loaded

    loaded = Dsv41Loaded(
        model=model, tokenizer=tokenizer, args=model.args,
        model_path=Path("/nonexistent-checkpoint"), built_layers=list(range(6)),
        full_stack=True, rank=0, world=1, load_seconds=0.0, head=None)
    return Dsv41Engine(
        loaded=loaded, model_id=ModelId("deeps/q2fu1-selftest"), group=None,
        cancel_receiver=_Receiver(), event_sender=_Sender(),  # type: ignore[arg-type]
        device_rank=0, speculative=False, prefill_chunk_size=8,
        max_kv_tokens=256)


def run_request(engine, correlation_id: str, max_tokens: int = 4):
    from exo.shared.models.model_cards import ModelId
    from exo.shared.types.common import CommandId
    from exo.shared.types.tasks import TextGeneration, TaskId
    from exo.shared.types.worker.instances import InstanceId
    from exo.shared.types.text_generation import (
        InputMessage, InputMessageContent, TextGenerationTaskParams)
    from exo.shared.types.worker.runner_response import FinishedResponse

    params = TextGenerationTaskParams(
        model=ModelId("deeps/q2fu1-selftest"),
        input=[InputMessage(role="user", content=InputMessageContent("hello"))],
        max_output_tokens=max_tokens, temperature=0.0,
        logprobs=True, top_logprobs=2, correlation_id=correlation_id)
    engine.submit(TextGeneration(
        task_id=TaskId(f"task-{correlation_id[:12]}"),
        command_id=CommandId(f"cmd-{correlation_id[:12]}"),
        task_params=params, instance_id=InstanceId("inst-1")))
    chunks, finished = [], False
    for _ in range(4000):
        step = list(engine.step())
        if not step:
            break
        for _tid, item in step:
            if isinstance(item, FinishedResponse):
                finished = True
            else:
                chunks.append(item)
    assert finished, "engine never reported FinishedResponse"
    import gc
    import time as _t
    from pathlib import Path as _P
    # The engine lets the request generator be collected (it is not pulled to
    # StopIteration); its `finally` writes the capture summary.  Force it.
    summary = _P(os.environ.get("DSV41_Q2FU1_CAPTURE", ".")) / f"summary_{correlation_id.split(';')[1].split('=')[1]}_rank0.json"
    for _ in range(100):
        if summary.exists():
            break
        gc.collect()
        _t.sleep(0.05)
    ids = [int(c.token_id) for c in chunks if hasattr(c, "token_id")]
    lps = [float(c.logprob) for c in chunks if getattr(c, "logprob", None) is not None]
    return ids, lps


def load_npz(path: Path) -> dict:
    with np.load(path) as z:
        return {k: z[k] for k in z.files}


# ---------------------------------------------------------------------------
# main
# ---------------------------------------------------------------------------
def main() -> int:
    outdir = Path(tempfile.mkdtemp(prefix="q2fu1-selftest-"))
    print(f"outdir={outdir}")
    conftest = _load_conftest()

    # ------------------------------------------------------------------ 0 ---
    print("\n[0] bootstrap: env-gated install")
    import q2fu1_capture as cap_mod
    from exo.worker.engines.mlx.dsv41 import engine as ENG

    os.environ.pop("DSV41_Q2FU1_CAPTURE", None)
    cap_mod._CAP = None
    ENG._maybe_install_q2fu1_capture()
    check(cap_mod._CAP is None, "unset env => no install (production inert)")

    # substitute the head class BEFORE install so the head patch is exercised
    M_BUILD.ShardedHead = FakeShardedHead
    os.environ["DSV41_Q2FU1_CAPTURE"] = str(outdir)
    ENG._maybe_install_q2fu1_capture()
    check(cap_mod._CAP is not None, "set env => hook installed via engine bootstrap")
    cap = cap_mod._CAP
    check(cap_mod._CAP.outdir == str(outdir), "install honours the env outdir")

    # ------------------------------------------------------------------ 1 ---
    print("\n[1] armed request: sweep over the real engine path")
    tok_cls = conftest.FakeTokenizer

    class LongFakeTokenizer(tok_cls):  # type: ignore[misc,valid-type]
        def encode(self, text, add_special_tokens=False):
            return list(range(10, 10 + PROMPT_LEN))

    model = build_model()
    engine = build_engine(model, LongFakeTokenizer())
    ids_armed, lp_armed = run_request(engine, "q2fu1;arm=qa;sweep=1")
    check(len(ids_armed) == 4, f"armed request streamed 4 tokens ({ids_armed})")

    summary_path = outdir / "summary_qa_rank0.json"
    check(summary_path.exists(), "sweep summary written")
    summary = json.loads(summary_path.read_text())
    integ = summary["integrity"][0] if summary["integrity"] else {}
    check(integ.get("cache_restore_ok") is True,
          f"integrity: cache digest restored exactly ({integ.get('digest_pre')})")
    check(integ.get("m1_repro_ok") is True,
          "integrity: post-sweep M=1 reproduces chain step 0 bitwise")
    chain = summary["outids"]["s0"]  # placeholder, replaced below
    chain = integ["chain"]
    check(len(chain) == 7, f"chain has anchor + 6 greedy steps ({chain})")

    # ------------------------------------------------------------------ 2 ---
    print("\n[2] dump format: pinned keys + shapes")
    f4 = load_npz(outdir / "qa_M4_M4.npz")
    check(int(f4["M"]) == 4, "M=4")
    check(int(f4["start_pos"]) == PROMPT_LEN, f"start_pos={PROMPT_LEN}")
    check(int(f4["vocab_lo"]) == 0 and int(f4["vocab_hi"]) == 256,
          "vocab slice [0,256)")
    check(f4["logits"].shape == (4, 256), "logits float32[4, 256]")
    check(f4["logits"].dtype == np.float32, "logits dtype fp32")
    check(f4["rows_ids"].shape == (4,) and f4["rows_ids"].dtype == np.int32,
          "rows_ids int32[4]")
    check(np.array_equal(f4["rows_ids"].astype(int), np.array(chain[:4])),
          "rows_ids == the fixed chain prefix")
    check(f4["hidden_in"].shape == (4, 64) and f4["hidden_out"].shape == (4, 64),
          "hidden_in/out [4, 64]")
    check(int(f4["n_captured_layers"]) == 6, "all 6 body layers captured")
    for L in range(6):
        check(f4[f"idx_{L}"].shape == (4, 2) and f4[f"idx_{L}"].dtype == np.int32,
              f"idx_{L} int32[4, topk=2]")
        check(f4[f"w_{L}"].shape == (4, 2), f"w_{L} [4, 2]")
        check(f4[f"hid_{L}"].shape == (4, 64), f"hid_{L} [4, 64]")
        check(f4[f"margin_{L}"].shape == (4,), f"margin_{L} [4]")
    check(all(int(x) in range(8) for x in f4["idx_0"].reshape(-1)),
          "router ids in range (real Gate capture)")
    check(bool(np.isfinite(f4["margin_0"]).all()), "router margins finite")
    tag_M = {"s0": 1, "s1": 1, "s2": 1, "s3": 1, "s4": 1, "s5": 1,
             "M1": 1, "M3": 3, "M4": 4, "M5": 5, "M6": 6,
             "M4rep2": 4, "M1fp32": 1, "M4fp32": 4,
             "M1fence0": 1, "M4fence0": 4,
             "g2_fwd": 3, "g2_A_m1": 1, "g2_B_m1": 1,
             "g3_fwd": 4, "g3_A_m1": 1, "g3_B_m1": 1,
             "g4_fwd": 5, "g4_A_m1": 1, "g4_B_m1": 1,
             "g5_fwd": 6, "g5_A_m1": 1, "g5_B_m1": 1,
             "M1final": 1}
    for tag, k in tag_M.items():
        stem = f"qa_M{k}_{tag}.npz"
        check((outdir / stem).exists(), f"dump present: {stem}")

    # ------------------------------------------------------------------ 3 ---
    print("\n[3] cross-shape read-out (row i of M-shape vs M=1 seq step i)")
    seq = [load_npz(outdir / f"qa_M1_s{i}.npz")["logits"][0] for i in range(6)]
    for M in (3, 4, 5, 6):
        fm = load_npz(outdir / f"qa_M{M}_M{M}.npz")["logits"]
        d = max(float(np.max(np.abs(fm[i] - seq[i]))) for i in range(M))
        check(d < 1e-3, f"M={M}: max|row_i(M) - seq_i| = {d:.2e} (< 1e-3 CPU)")
    f4b = load_npz(outdir / "qa_M4_M4rep2.npz")["logits"]
    d = float(np.max(np.abs(f4["logits"] - f4b)))
    check(d == 0.0, f"noise floor: M=4 repeat max|delta| = {d:.2e} (exact)")

    # ------------------------------------------------------------------ 4 ---
    print("\n[4] rollback A/B (accepted=1): ring-clobber discrimination")
    for gm, expect_diff in ((2, False), (3, True), (4, True), (5, True)):
        fa = load_npz(outdir / f"qa_M1_g{gm}_A_m1.npz")["logits"][0]
        fb = load_npz(outdir / f"qa_M1_g{gm}_B_m1.npz")["logits"][0]
        d = float(np.max(np.abs(fa - fb)))
        if expect_diff:
            check(d > 1e-3, f"gamma={gm}: A(+ring clobber) vs B(ring restore) "
                            f"max|delta| = {d:.3f} > 1e-3 (contamination)")
        else:
            check(d < 1e-6, f"gamma={gm}: A == B (max|delta| = {d:.2e}; the "
                            f"rejected slot aliases a dead position)")

    # ------------------------------------------------------------------ 5 ---
    print("\n[5] faithfulness: armed vs unarmed token stream")
    engine2 = build_engine(build_model(), LongFakeTokenizer())
    ids_plain, lp_plain = run_request(engine2, "q2fu1;arm=qb;sweep=0")
    check(ids_plain == ids_armed,
          f"armed stream == unarmed stream ({ids_armed})")
    check(lp_armed == lp_plain, "logprobs identical (bitwise floats)")

    # ------------------------------------------------------------------ 6 ---
    print("\n[6] bootstrap no-op when unset, again after use")
    cap_mod.uninstall()
    check(cap_mod._CAP is None, "uninstall clears the singleton")
    os.environ.pop("DSV41_Q2FU1_CAPTURE", None)
    ENG._maybe_install_q2fu1_capture()
    check(cap_mod._CAP is None, "env unset => still inert after install/uninstall")

    # ------------------------------------------------------------------ 7 ---
    print("\n[7] driver offline: merge + analyze + faithfulness gate")
    import importlib
    drv = importlib.import_module("q2fu1_driver")
    capdir = outdir / "merged"
    r0, r1 = capdir / "rank_studio1", capdir / "rank_studio2"
    r0.mkdir(parents=True), r1.mkdir(parents=True)
    for f in sorted(outdir.glob("qa_*.npz")):
        d = load_npz(f)
        half = d["logits"].shape[1] // 2
        d0 = dict(d)
        d0["logits"] = np.where(np.arange(d["logits"].shape[1])[None, :] < half,
                                d["logits"], np.nan).astype(np.float32)
        d1 = dict(d)
        d1["logits"] = np.where(np.arange(d["logits"].shape[1])[None, :] >= half,
                                d["logits"], np.nan).astype(np.float32)
        with open(r0 / f.name, "wb") as fh:
            np.savez(fh, **d0)
        with open(r1 / f.name, "wb") as fh:
            np.savez(fh, **d1)
    capdir.mkdir(exist_ok=True)
    mres = drv.merge_all(str(capdir))
    check(not mres["problems"], "merge_all: no rank-mismatch problems")
    orig = load_npz(outdir / "qa_M4_M4.npz")["logits"]
    merged = load_npz(capdir / "qa_M4_M4.merged.npz")["logits"]
    check(np.array_equal(orig, merged, equal_nan=True),
          "merged logits == original (bit-exact)")
    bad = np.array(merged, copy=True)
    bad[0, 0] += 1.0
    d3 = load_npz(capdir / "qa_M3_M3.merged.npz")
    d3["logits"] = bad[:3]
    with open(capdir / "qa_M3_M3.merged.npz", "wb") as fh:
        np.savez(fh, **d3)
    an = drv.analyze_captures(str(capdir))
    check(an["noise_floor"]["max_abs_delta"] == 0.0, "analyzer: noise floor 0.0")
    check(an["shape_readout"]["M4"]["routing_identical_row0"] is True,
          "analyzer: M=4 row0 routing identical to M=1 (toy)")
    check(an["shape_readout"]["M4"]["max_abs_delta_logits_row0_vs_M1"] < 1e-3,
          "analyzer: M=4 row0 delta small (toy)")
    check(an["rollback_ab"]["g3"]["max_abs_delta_A_vs_B"] > 1e-3,
          "analyzer: gamma3 rollback A/B contamination present")
    check(an["rollback_ab"]["g2"]["max_abs_delta_A_vs_B"] < 1e-6,
          "analyzer: gamma2 rollback A/B clean")
    rec_dir = outdir / "recorded"
    rec_dir.mkdir()
    for rec_arm, arm_key in (("gamma3a", "g3"), ("gamma4", "g4")):
        with open(rec_dir / f"arm_{rec_arm}.json", "w", encoding="utf-8") as fh:
            json.dump({"agentic": [{"tokens": [11, 22, 33],
                                    "top2_logprobs": [[-0.1, -1.2],
                                                      [-0.2, -1.3],
                                                      [-0.3, -1.4]]}]}, fh)
    gate = drv.faithfulness_gate(
        {"g3": {"tokens": [11, 22, 33],
                "top2_logprobs": [[-0.1, -1.2], [-0.2, -1.3], [-0.3, -1.4]]},
         "g4": {"tokens": [11, 22, 44],
                "top2_logprobs": [[-0.1, -1.2], [-0.2, -1.3], [-0.3, -1.5]]}},
        str(rec_dir))
    check("MISMATCH" in gate["verdict"],
          "faithfulness gate: catches a divergent stream (gamma4)")
    gate2 = drv.faithfulness_gate(
        {"g3": {"tokens": [11, 22, 33],
                "top2_logprobs": [[-0.1, -1.2], [-0.2, -1.3], [-0.3, -1.4]]},
         "g4": {"tokens": [11, 22, 33],
                "top2_logprobs": [[-0.1, -1.2], [-0.2, -1.3], [-0.3, -1.4]]}},
        str(rec_dir))
    check("EXACT" in gate2["verdict"],
          "faithfulness gate: EXACT when both streams reproduce")

    # ------------------------------------------------------------------ 8 ---
    print("\n[8] speculative round path: ms passive capture + rollback probes")
    from mlx_lm.models.deepseek_v41.mtp import DSparkHead

    # reinstall after the section-[6] uninstall (also proves reinstall works)
    os.environ["DSV41_Q2FU1_CAPTURE"] = str(outdir)
    ENG._maybe_install_q2fu1_capture()
    check(cap_mod._CAP is not None and cap_mod._CAP._installed,
          "reinstall after uninstall works")

    def build_spec_engine():
        m = build_model()
        mx.random.seed(11)
        dsh = DSparkHead(toy_args())
        mx.eval(dsh.parameters())
        eng = build_engine(m, LongFakeTokenizer())
        eng.loaded.head = dsh
        eng.speculative = True
        return eng

    e1 = build_spec_engine()
    ids_ms, _ = run_request(e1, "q2fu1;arm=qm;sweep=0;ms=3", max_tokens=12)
    check((outdir / "qm_M4_r1.npz").exists(),
          "ms arm: round 1 captured as a real M=4 verify forward")
    check((outdir / "qm_M1_r0.npz").exists(),
          "ms arm: the priming round captured (M=1 plain step)")
    spec_stats = json.loads((outdir / "summary_qm_rank0.json").read_text())
    check(len(spec_stats["outids"].get("r1", [])) == 4,
          "ms arm: r1 out_ids has one argmax per verify row")

    e2 = build_spec_engine()
    ids_rb, _ = run_request(e2, "q2fu1;arm=qr;sweep=0;ms=3;rollback=1;rounds=3",
                            max_tokens=12)
    check((outdir / "qr_M1_g3_A_m1_r1.npz").exists()
          and (outdir / "qr_M1_g3_B_m1_r1.npz").exists(),
          "rollback arm: A/B probe dumps written for real round 1")
    check(ids_rb == ids_ms,
          f"rollback probes are stream-neutral: {ids_rb} == {ids_ms}")

    print(f"\nSELF-TEST OK ({N_CHECKS[0]} checks) -- outdir {outdir}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
