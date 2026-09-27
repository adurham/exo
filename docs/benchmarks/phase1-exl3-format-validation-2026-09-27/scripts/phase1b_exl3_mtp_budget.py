#!/usr/bin/env python3
"""Phase 1 addendum for the dealignai DSv4.1-Flash EXL3 2.9bpw repo.

Part B (fast, runs first): the plan's per-rank byte table -- "walk every
shard header and tabulate bits per tensor by component: routed experts,
shared expert, attention, dense MLP, embeddings, head, MTP. Compute
per-rank resident bytes under the V4 sharding rule (experts split,
everything else replicated). This is the real memory budget."
Implemented by reading ONLY safetensors headers (data_offsets), so it is
exact bytes as stored, not an estimate.

Part A: the plan's remaining bit-exact check -- "one 4-bit MTP tensor".
quantization_config's tensor_storage carries zero mtp.* entries, so the
normal loader path (layer_meta_from_config) cannot build it; this part
reads the raw tensors from the shard via safe_open and constructs the
EXL3Layer by hand, then runs the same ref-vs-mlx comparison as the main
phase-1 script.

Run inside ~/phase1-exl3/.venv on macstudio-m4-1.
"""

from __future__ import annotations

import collections
import json
import os
import re
import struct
import sys
import traceback

import numpy as np

MODEL_DIR = os.path.expanduser(
    "~/.exo/models/dealignai--DeepSeek-V4.1-Flash-UNCENSORED-EXL3-2.9bpw"
)
TWO_RANKS = 2
WIRED_LIMIT_GB = 115.0
TRIGGER_GB = 98.0

RESULTS: list[dict] = []


def record(name: str, ok: bool, detail: str) -> None:
    RESULTS.append({"check": name, "ok": bool(ok), "detail": detail})
    print(f"[{'PASS' if ok else 'FAIL'}] {name}: {detail}", flush=True)


def gb(n: float) -> float:
    return n / 1e9


def read_shard_header(path: str) -> dict:
    with open(path, "rb") as f:
        n = struct.unpack("<Q", f.read(8))[0]
        return json.loads(f.read(n))


# --------------------------------------------------------------------------
# Part B: per-rank byte table from shard headers
# --------------------------------------------------------------------------
def byte_table() -> None:
    idx = json.load(
        open(os.path.join(MODEL_DIR, "model.safetensors.index.json"), encoding="utf-8")
    )["weight_map"]
    shards = sorted(set(idx.values()))

    per_tensor: dict[str, int] = {}
    for shard in shards:
        hdr = read_shard_header(os.path.join(MODEL_DIR, shard))
        for k, meta in hdr.items():
            if k == "__metadata__":
                continue
            start, end = meta["data_offsets"]
            per_tensor[k] = int(end) - int(start)
    print(f"shards walked: {len(shards)}   tensors: {len(per_tensor)}", flush=True)

    groups = collections.OrderedDict(
        [
            ("routed_experts", re.compile(r"^layers\.\d+\.ffn\.experts\.\d+\.")),
            ("shared_experts", re.compile(r"^layers\.\d+\.ffn\.shared_experts\.")),
            ("attention", re.compile(r"^layers\.\d+\.attn\.")),
            ("router+glue", re.compile(r"^layers\.\d+\.(ffn\.gate\.|ffn_norm\.|attn_norm\.|hc_)")),
            ("embed", re.compile(r"^(embed\.|hc_expand|hc_collapse)")),
            ("head", re.compile(r"^head\.")),
            ("norm", re.compile(r"^norm\.")),
            ("mtp", re.compile(r"^mtp\.")),
            ("vision", re.compile(r"^(vision\.|aligner\.|image_)")),
        ]
    )
    comp: dict[str, int] = collections.OrderedDict((g, 0) for g in groups)
    comp["other"] = 0
    counts: collections.Counter = collections.Counter()
    for k, nb in per_tensor.items():
        for g, rx in groups.items():
            if rx.match(k):
                comp[g] += nb
                counts[g] += 1
                break
        else:
            comp["other"] += nb
            counts["other"] += 1

    total = sum(comp.values())
    print()
    print(f"{'component':16s} {'tensors':>8s} {'bytes':>15s} {'GB':>9s} {'share':>7s}")
    for g, nb in comp.items():
        print(f"{g:16s} {counts[g]:8d} {nb:15d} {gb(nb):9.2f} {100 * nb / total:6.1f}%")
    print(f"{'TOTAL':16s} {sum(counts.values()):8d} {total:15d} {gb(total):9.2f}")

    # experts-per-layer sanity
    layers: dict[int, set] = collections.defaultdict(set)
    mats: collections.Counter = collections.Counter()
    for k in per_tensor:
        m = re.match(r"^layers\.(\d+)\.ffn\.experts\.(\d+)\.(w[123])\.", k)
        if m:
            layers[int(m.group(1))].add(int(m.group(2)))
            mats[m.group(3)] += 1
    n_layers = len(layers)
    print(
        f"\nexpert sanity: {n_layers} MoE layers, "
        f"experts/layer min={min(len(v) for v in layers.values())} "
        f"max={max(len(v) for v in layers.values())}, "
        f"w1/w2/w3 matrix counts={dict(mats)}"
    )

    routed = comp["routed_experts"]
    replicated = total - routed
    rank_experts = routed / TWO_RANKS
    r0 = rank_experts + replicated
    r1 = r0 - comp["vision"]
    r2 = r1 - comp["mtp"]
    r3 = r2 - comp["head"] / 2 - comp["embed"] / 2
    r4 = r3 - 0.10 * rank_experts

    scenarios = [
        ("R0 raw (experts/2 + all replicated)", r0),
        ("R1 R0 - vision tower", r1),
        ("R2 R1 - MTP tensors", r2),
        ("R3 R2 - head/2 - embed/2 (shard head+embed)", r3),
        ("R4 R3 - coldest 10% of rank experts (SSD stream)", r4),
    ]
    print()
    print(f"per-rank resident weight bytes (V4 rule: experts split, everything else replicated):")
    print(f"{'scenario':58s} {'GB':>9s}  {'>98GB?':>7s}")
    for name, nb in scenarios:
        over = "OVER" if gb(nb) > TRIGGER_GB else "ok"
        print(f"{name:58s} {gb(nb):9.2f}  {over:>7s}")
    print()
    print(
        f"reference: wired limit {WIRED_LIMIT_GB:.0f} GB; V4-Flash precedent "
        f"72 GB weights/rank + ~16 GB overhead = 88 GB peak"
    )
    print(
        f"R0 + 16 GB overhead = {gb(r0) + 16:.1f} GB vs {WIRED_LIMIT_GB:.0f} GB wired; "
        f"R4 + 16 GB overhead = {gb(r4) + 16:.1f} GB"
    )

    record(
        "budget/per-rank byte table",
        True,
        f"R0={gb(r0):.1f}GB R2={gb(r2):.1f}GB R3={gb(r3):.1f}GB R4={gb(r4):.1f}GB "
        f"(trigger {TRIGGER_GB}GB)",
    )
    record(
        "budget/trigger check (R0 > 98GB means mitigations mandatory)",
        gb(r0) > TRIGGER_GB,
        f"R0={gb(r0):.2f} GB vs trigger {TRIGGER_GB} GB; "
        f"R4 (all listed mitigations)={gb(r4):.2f} GB",
    )


# --------------------------------------------------------------------------
# Part A: one 4-bit MTP tensor group, raw-shard path
# --------------------------------------------------------------------------
def find_module(prefix: str) -> str | None:
    idx = json.load(
        open(os.path.join(MODEL_DIR, "model.safetensors.index.json"), encoding="utf-8")
    )["weight_map"]
    for k in idx:
        if k.startswith(prefix + ".") and k.endswith(".trellis"):
            return k[: -len(".trellis")]
    return None


def mtp_check() -> None:
    import mlx.core as mx  # noqa: F401  (import check)
    from safetensors import safe_open

    from ponyexl3.ref.layer import EXL3Layer
    from ponyexl3.ref.decode import decode_packed_trellis
    from ponyexl3.ref.reconstruct import reconstruct_public_weights
    from ponyexl3.mlx.reconstruct import reconstruct_inner_mlx, reconstruct_public_mlx

    idx = json.load(
        open(os.path.join(MODEL_DIR, "model.safetensors.index.json"), encoding="utf-8")
    )["weight_map"]

    candidates = ["mtp.0.ffn.experts.10.w1", "mtp.0.attn", "mtp.0.main_proj"]
    tested = 0
    for prefix in candidates:
        module = find_module(prefix)
        if module is None and prefix in ("mtp.0.attn", "mtp.0.main_proj"):
            # prefix itself may be a container; search any first trellis under it
            for k in idx:
                if k.startswith(prefix + ".") and k.endswith(".trellis"):
                    module = k[: -len(".trellis")]
                    break
        if module is None:
            print(f"\n--- {prefix}: not found, skipping", flush=True)
            continue

        shard = os.path.join(MODEL_DIR, idx[module + ".trellis"])
        with safe_open(shard, framework="np") as st:
            keys = set(st.keys())
            trellis = np.array(st.get_tensor(module + ".trellis"))
            suh = np.array(st.get_tensor(module + ".suh")) if module + ".suh" in keys else None
            svh = np.array(st.get_tensor(module + ".svh")) if module + ".svh" in keys else None
            mul1_raw = (
                np.array(st.get_tensor(module + ".mul1")) if module + ".mul1" in keys else None
            )

        in_tiles, out_tiles, packed_size = trellis.shape
        k = packed_size * 16 // 256
        mul1 = bool(int(mul1_raw)) if mul1_raw is not None else False

        layer = EXL3Layer(
            key=module,
            in_features=in_tiles * 16,
            out_features=out_tiles * 16,
            k=k,
            trellis=trellis,
            suh=suh,
            svh=svh,
            mcg=False,
            mul1=mul1,
        )
        layer.validate()

        print(
            f"\n--- {module}\n"
            f"    in={layer.in_features} out={layer.out_features} k={k} "
            f"bits={k} mul1={mul1} trellis={trellis.shape}",
            flush=True,
        )

        ref = np.asarray(decode_packed_trellis(layer.trellis, k, layer.codebook_mode), dtype=np.float32)
        got = np.asarray(
            np.array(reconstruct_inner_mlx(layer.trellis, k, mcg=False, mul1=mul1)), dtype=np.float32
        )
        inner_ok = ref.shape == got.shape and np.array_equal(ref, got)
        record(
            f"{module}/inner_decode ref-vs-mlx",
            inner_ok,
            f"shapes ref{ref.shape} mlx{got.shape} equal={np.array_equal(ref, got) if ref.shape == got.shape else 'n/a'} "
            f"max_abs_diff={float(np.max(np.abs(ref - got))) if ref.shape == got.shape else float('nan')}",
        )

        ref_w = np.asarray(
            reconstruct_public_weights(layer.trellis, layer.suh, layer.svh, k, mcg=False, mul1=mul1),
            dtype=np.float16,
        )
        got_w = np.asarray(np.array(reconstruct_public_mlx(layer)), dtype=np.float16)
        full_ok = ref_w.shape == got_w.shape and np.array_equal(ref_w, got_w)
        record(
            f"{module}/full_layer ref-vs-mlx",
            full_ok,
            f"shapes ref{ref_w.shape} mlx{got_w.shape} equal={np.array_equal(ref_w, got_w) if ref_w.shape == got_w.shape else 'n/a'} "
            f"max_abs_diff={float(np.max(np.abs(ref_w.astype(np.float32) - got_w.astype(np.float32)))) if ref_w.shape == got_w.shape else float('nan')}",
        )
        tested += 1

    record("mtp/4-bit group tested", tested > 0, f"{tested} raw-path MTP group(s) compared")


def main() -> int:
    print(f"model: {MODEL_DIR}", flush=True)
    print("=" * 72)
    print("PART B: per-rank byte table (from shard headers)")
    print("=" * 72, flush=True)
    try:
        byte_table()
    except Exception as exc:  # noqa: BLE001
        record("budget/byte table", False, f"{type(exc).__name__}: {exc}")
        traceback.print_exc()

    print()
    print("=" * 72)
    print("PART A: 4-bit MTP tensor group (raw-shard path)")
    print("=" * 72, flush=True)
    try:
        mtp_check()
    except Exception as exc:  # noqa: BLE001
        record("mtp/check", False, f"{type(exc).__name__}: {exc}")
        traceback.print_exc()

    total = len(RESULTS)
    passed = sum(1 for r in RESULTS if r["ok"])
    print("\n" + "=" * 72)
    print(f"PHASE 1B VERDICT: {passed}/{total} checks passed, {total - passed} failed")
    for r in RESULTS:
        if not r["ok"]:
            print(f"  FAILED: {r['check']} -- {r['detail']}")
    print("=" * 72)
    with open(os.path.expanduser("~/phase1b-results.json"), "w", encoding="utf-8") as f:
        json.dump({"checks": RESULTS, "passed": passed, "total": total}, f, indent=1)
    return 0 if passed == total else 1


if __name__ == "__main__":
    sys.exit(main())
