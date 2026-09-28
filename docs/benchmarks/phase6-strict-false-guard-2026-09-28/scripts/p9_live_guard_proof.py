"""LIVE proof for the rename-aware STRICT-FALSE-LOAD-GUARD (p9-guard).

Runs on a Mac Studio node with real mlx. Builds a TINY DeepSeek-V4 -- the
architecture class that defines ``sanitize()`` and that the guard used to skip
ENTIRELY, including the production model this cluster serves -- saves it,
loads it through the real mlx-lm ``load_model(..., strict=False)`` path wrapped
in exo's ``_RecordLoadedWeightKeys``, and checks three contracts:

  (A) healthy load                     -> NO guard ERROR (was: guard skipped)
  (B) a key that sanitize() cannot     -> ERROR naming the key (the entire
      reconstruct (dropped)               point: impossible before this fix)
  (C) PRE-sanitize key names on disk   -> raw diff WOULD false-alarm, captured
      that sanitize() fixes               diff is correctly silent. Numbers for
                                          both are printed side by side.

Each case prints the guard's actual log lines. Exit code 0 == all pass.

Run on a node:
    PYTHONPATH=~/repos/exo/src:~/repos/exo/.venv/bin/python p9_live_guard_proof.py
"""
import json
import os
import struct
import sys
import tempfile
from pathlib import Path

REPO = Path(os.path.expanduser("~/repos/exo"))
sys.path.insert(0, str(REPO / "mlx-lm"))
sys.path.insert(0, str(REPO / "src"))

import mlx.core as mx  # noqa: E402
from mlx_lm.models import deepseek_v4 as dsv4  # noqa: E402
from mlx_lm.utils import tree_flatten  # noqa: E402

from exo.worker.engines.mlx import utils_mlx  # noqa: E402
from exo.worker.engines.mlx.utils_mlx import (  # noqa: E402
    _load_checkpoint_key_set,
    _RecordLoadedWeightKeys,
    _run_strict_false_load_guard,
)

# ---------------------------------------------------------------- logger tap
# utils_mlx binds `logger` at module import (loguru, via bootstrap). Replace the
# module-global with a recorder so the guard's real call path is exercised, not
# a reimplementation of it.
class _Recorder:
    def __init__(self) -> None:
        self.records: list[tuple[str, str]] = []

    def _mk(self, lvl: str):
        def _f(msg, *a, **kw):
            self.records.append((lvl, str(msg)))
            print(f"   [{lvl}] {msg}")
        return _f

    def info(self, msg, *a, **kw):    # noqa: ANN001, ANN201
        self._mk("INFO")(msg)

    def error(self, msg, *a, **kw):   # noqa: ANN001, ANN201
        self._mk("ERROR")(msg)

    def warning(self, msg, *a, **kw):  # noqa: ANN001, ANN201
        self._mk("WARNING")(msg)

    def opt(self, **kw):  # loguru chaining used by other code paths
        return self


LOG = _Recorder()

# ---------------------------------------------------------------- tiny model
TINY = dict(
    model_type="deepseek_v4",
    vocab_size=64,
    hidden_size=16,
    num_hidden_layers=1,
    num_attention_heads=2,
    num_key_value_heads=1,
    q_lora_rank=8,
    o_lora_rank=4,
    o_groups=1,
    head_dim=8,
    qk_rope_head_dim=4,
    sliding_window=8,
    compress_ratios=[0],
    index_n_heads=2,
    index_head_dim=4,
    index_topk=2,
    moe_intermediate_size=8,
    n_routed_experts=2,
    n_shared_experts=1,
    num_experts_per_tok=1,
    num_hash_layers=1,
    hc_mult=1,
    hc_sinkhorn_iters=1,
    num_nextn_predict_layers=0,
)


def _build(dirpath: Path) -> tuple[object, dict]:
    m = dsv4.Model(dsv4.ModelArgs(**TINY))
    mx.eval(m.parameters())
    flat = dict(tree_flatten(m.parameters()))
    mx.save_safetensors(str(dirpath / "model.safetensors"), flat)
    (dirpath / "config.json").write_text(json.dumps(TINY))
    return m, flat


def _write_index(dirpath: Path) -> None:
    """Add a weight_map index so `_load_checkpoint_key_set` (the guard's raw
    fallback) can actually read this fixture's keys -- without it the raw path
    returns None and any raw-vs-tree comparison is vacuous."""
    keys = _shard_keys(dirpath)
    (dirpath / "model.safetensors.index.json").write_text(
        json.dumps({"weight_map": {k: "model.safetensors" for k in keys}})
    )


def _shard_keys(dirpath: Path) -> list[str]:
    """Read every key from the shard header directly.

    Deliberately NOT via ``_load_checkpoint_key_set``: that helper returns None
    for a checkpoint with no ``model.safetensors.index.json`` (single-shard),
    which is exactly what this tiny fixture is. The guard's raw-on-disk
    fallback therefore cannot check this checkpoint at all -- while the
    load-time capture still covers it, which is a nice illustration of why the
    capture is the better primary signal.
    """
    src = dirpath / "model.safetensors"
    with open(src, "rb") as f:
        n = struct.unpack("<Q", f.read(8))[0]
        header = json.loads(f.read(n))
    return sorted(k for k in header if k != "__metadata__")


def _rewrite(dirpath: Path, transform) -> None:
    """Rewrite model.safetensors applying transform(key)->key|None."""
    src = dirpath / "model.safetensors"
    with open(src, "rb") as f:
        n = struct.unpack("<Q", f.read(8))[0]
        header = json.loads(f.read(n))
        raw = f.read()
    new_header, chunks, off = {}, [], 0
    for k, meta in header.items():
        if k == "__metadata__":
            continue
        start, end = meta["data_offsets"]
        blob = raw[start:end]
        nk = transform(k)
        if nk is None:
            continue
        new_header[nk] = {
            "dtype": meta["dtype"],
            "shape": meta["shape"],
            "data_offsets": [off, off + len(blob)],
        }
        chunks.append(blob)
        off += len(blob)
    hb = json.dumps(new_header).encode()
    hb += b" " * ((8 - (len(hb) % 8)) % 8)
    with open(src, "wb") as f:
        f.write(struct.pack("<Q", len(hb)))
        f.write(hb)
        f.write(b"".join(chunks))


def _load_and_guard(d: Path, model_path_label=None):
    from mlx_lm.utils import load_model as lm  # noqa: N813

    with _RecordLoadedWeightKeys() as cap:
        model, _cfg = lm(d, lazy=True, strict=False)
    _run_strict_false_load_guard(model, d, cap)
    return model, cap


def _guard_lines() -> list[tuple[str, str]]:
    return [r for r in LOG.records if "STRICT-FALSE-LOAD-GUARD" in r[1]]


def main() -> int:
    utils_mlx.logger = LOG  # swap in the recorder
    results: dict[str, bool] = {}

    with tempfile.TemporaryDirectory() as td:
        d = Path(td)
        print("=== (0) build tiny DSv4 + export its weights ===")
        model, flat = _build(d)
        print(f"    {len(flat)} tensors, e.g. {sorted(flat)[:3]}")

        # ---------- (A) healthy load ----------
        print("\n=== (A) HEALTHY load: expect ZERO guard lines ===")
        LOG.records.clear()
        _m, cap = _load_and_guard(d)
        lines = _guard_lines()
        results["A_healthy_silent"] = not lines
        print(f"    captured={len(cap.get('keys') or [])} keys; "
              f"guard lines={len(lines)} -> "
              f"{'PASS' if not lines else 'FAIL'}")

        # ---------- (B) genuinely broken load ----------
        print("\n=== (B) DROPPED key: expect ERROR naming it ===")
        keys = _shard_keys(d)
        victim = next(
            (k for k in keys if "layers.0" in k and k.endswith(".weight")),
            keys[-1],
        )
        print(f"    dropping {victim!r}")
        _rewrite(d, lambda k: None if k == victim else k)
        LOG.records.clear()
        _m, _cap = _load_and_guard(d)
        lines = _guard_lines()
        errs = [m for lvl, m in lines if lvl == "ERROR"]
        results["B_dropped_caught"] = bool(errs) and any(victim in m for m in errs)
        print(f"    ERROR lines={len(errs)}; names victim="
              f"{any(victim in m for m in errs)} -> "
              f"{'PASS' if results['B_dropped_caught'] else 'FAIL'}")

        # ---------- (C) pre-sanitize names: raw diff would lie ----------
        print("\n=== (C) PRE-sanitize names on disk: raw diff lies, "
              "captured diff is right ===")
        _build(d)  # restore a healthy shard
        # DSv4 sanitize() maps `hc_attn_fn` -> `attn_hc.fn`. Write the file the
        # way the upstream checkpoint does (pre-sanitize) and let sanitize()
        # fix it. A raw on-disk-vs-param diff MUST report a mismatch here; the
        # captured set must not.
        swapped = 0

        def _to_presanitize(k: str):
            nonlocal swapped
            if ".attn_hc.fn" in k:
                swapped += 1
                return k.replace(".attn_hc.fn", ".hc_attn_fn")
            return k

        _rewrite(d, _to_presanitize)
        _write_index(d)  # so the RAW fallback path can actually see these keys
        raw_keys = _load_checkpoint_key_set(d)
        print(f"    rewrote {swapped} key(s) to pre-sanitize names")
        LOG.records.clear()
        m3, cap3 = _load_and_guard(d)
        lines = _guard_lines()
        captured = cap3.get("keys") or set()
        params = {k for k, _ in tree_flatten(m3.parameters())}
        raw_missing = len(params - (raw_keys or set()))
        raw_extra = len((raw_keys or set()) - params)
        cap_missing = len(params - captured)
        cap_extra = len(captured - params)
        print(f"    raw on-disk  diff: missing={raw_missing} extra={raw_extra}")
        print(f"    captured set diff: missing={cap_missing} extra={cap_extra}")
        results["C_capture_clean"] = not lines
        results["C_raw_would_lie"] = (raw_missing + raw_extra) > 0
        print(f"    guard lines={len(lines)} -> "
              f"{'PASS (silent)' if not lines else 'FAIL'}")

    print("\n=== SUMMARY ===")
    for k, v in results.items():
        print(f"  {k:24s} {'PASS' if v else 'FAIL'}")
    ok = all(results.values())
    print(f"OVERALL: {'PASS' if ok else 'FAIL'}")
    return 0 if ok else 1


if __name__ == "__main__":
    sys.exit(main())
