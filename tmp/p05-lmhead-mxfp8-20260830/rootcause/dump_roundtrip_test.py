"""Verify the PRENORM_H_DUMP fix actually round-trips bf16 correctly.

The original dump line was:
    np.asarray(h.astype(mx.bfloat16), stream=mx.cpu)
which raises TypeError unconditionally (numpy asarray has no `stream`
kwarg; numpy also has no bfloat16 dtype). The enclosing bare
`except Exception: pass` swallowed it, so /tmp/p05_hdump stayed EMPTY on
both nodes across the entire arm-A window.

This test proves the replacement (uint16 bit-view) is lossless.
"""

import mlx.core as mx
import numpy as np


def dump_bytes(h: mx.array) -> bytes:
    """Exactly the fixed dump path from deepseek_v4.py."""
    hb16 = h.astype(mx.bfloat16)
    mx.eval(hb16)
    return np.asarray(hb16.view(mx.uint16), dtype=np.uint16).tobytes()


def load_bf16(raw: bytes, shape) -> mx.array:
    u16 = np.frombuffer(raw, dtype=np.uint16)
    return mx.array(u16).view(mx.bfloat16).reshape(shape)


def main() -> int:
    print("=" * 62)
    print("PRENORM_H_DUMP round-trip verification")
    print("=" * 62)

    # 1. The original line must actually raise (proving it was a silent no-op)
    h = mx.random.normal((1, 3, 4096)).astype(mx.bfloat16)
    try:
        np.asarray(h.astype(mx.bfloat16), stream=mx.cpu)
        print("UNEXPECTED: original line did NOT raise")
        return 1
    except TypeError as exc:
        print(f"[OK] original dump line raises TypeError: {exc}")

    # 2. Fixed path round-trips bit-exactly at every production shape
    ok = True
    for shape in [(1, 1, 4096), (1, 3, 4096), (1, 4, 4096), (1, 128, 4096)]:
        src = mx.random.normal(shape).astype(mx.bfloat16)
        raw = dump_bytes(src)
        back = load_bf16(raw, shape)
        maxdiff = float(mx.max(mx.abs(back.astype(mx.float32) - src.astype(mx.float32))))
        nbytes_ok = len(raw) == int(np.prod(shape)) * 2
        status = "OK" if (maxdiff == 0.0 and nbytes_ok) else "FAIL"
        if status == "FAIL":
            ok = False
        print(f"[{status}] shape={str(shape):16s} bytes={len(raw):8d} maxdiff={maxdiff}")

    print("=" * 62)
    print("RESULT:", "PASS — dump fix is lossless" if ok else "FAIL")
    return 0 if ok else 1


if __name__ == "__main__":
    raise SystemExit(main())
