#!/usr/bin/env python3
"""Additive clamp patch for PonyExl3 exl3_moe.py -- new silu_clamp activation.

DeepSeek V4.1 routed experts use a clamped SwiGLU (swiglu_limit=10.0): up
clipped two-sided, gate upper-only. The library only had silu/gelu, so the
EXL3 expert path ran unclamped. This adds an opt-in mode; stock modes and the
default path are untouched (bit-identical). Limit from EXL3_MOE_CLAMP (10.0).

Usage: patch_exl3_clamp.py [path-to-exl3_moe.py]
"""
import hashlib
import os
import sys

P = sys.argv[1] if len(sys.argv) > 1 else os.path.expanduser(
    "~/repos/ref/PonyExl3/ponyexl3/mlx/exl3_moe.py")

src = open(P).read()
md5_before = hashlib.md5(src.encode()).hexdigest()
if chr(34) + "silu_clamp" + chr(34) in src:
    print("ALREADY_PATCHED md5=%s" % md5_before)
    sys.exit(0)

OLD1 = "\n".join([
    "# Gate activation for stacked experts: Qwen MoE uses SwiGLU (silu*up);",
    "# Gemma4 routed experts use GeGLU (gelu_approx(gate)*up), matching exllamav3.",
    chr(95) + "MOE_ACTIVATIONS = frozenset({\"silu\", \"gelu\"})",
    "",
]) + "\n"
NEW1 = "\n".join([
    "# Gate activation for stacked experts: Qwen MoE uses SwiGLU (silu*up);",
    "# Gemma4 routed experts use GeGLU (gelu_approx(gate)*up), matching exllamav3.",
    "# \"silu_clamp\" is DeepSeek V4.1 clamped SwiGLU: up two-sided, gate",
    "# upper-only, both at EXL3_MOE_CLAMP (default 10.0). Additive mode --",
    "# the stock modes are unchanged and remain the default.",
    chr(95) + "MOE_ACTIVATIONS = frozenset({\"silu\", \"gelu\", \"silu_clamp\"})",
    "",
    chr(95) + "CLAMP_LIMIT = float(os.environ.get(\"EXL3_MOE_CLAMP\", \"10.0\") or 10.0)",
    "",
]) + "\n"

OLD2 = "\n".join([
    "    if act == \"gelu\":",
    "        return (",
    "            \"float tanh_arg = 0.797884560803f * (g + 0.044715f * g * g * g);\\n\"",
    "            \"        float h = 0.5f * g * (1.0f + tanh(tanh_arg)) * u;\"",
    "        )",
    "    return \"float h = (g / (1.0f + exp(-g))) * u;\"",
    "",
]) + "\n"
NEW2 = "\n".join([
    "    if act == \"gelu\":",
    "        return (",
    "            \"float tanh_arg = 0.797884560803f * (g + 0.044715f * g * g * g);\\n\"",
    "            \"        float h = 0.5f * g * (1.0f + tanh(tanh_arg)) * u;\"",
    "        )",
    "    if act == \"silu_clamp\":",
    "        return (",
    "            \"g = fmin(g, %.1ff);\\n\" % _CLAMP_LIMIT",
    "            + \"        u = fmin(fmax(u, -%.1ff), %.1ff);\\n\" % (_CLAMP_LIMIT, _CLAMP_LIMIT)",
    "            + \"        float h = (g / (1.0f + exp(-g))) * u;\"",
    "        )",
    "    return \"float h = (g / (1.0f + exp(-g))) * u;\"",
    "",
]) + "\n"

OLD3 = "\n".join([
    "def _moe_gate_activation(g: mx.array, u: mx.array, act: str) -> mx.array:",
    "    if act == \"gelu\":",
    "        return nn.gelu_approx(g) * u",
    "    return nn.silu(g) * u",
    "",
]) + "\n"
NEW3 = "\n".join([
    "def _moe_gate_activation(g: mx.array, u: mx.array, act: str) -> mx.array:",
    "    if act == \"gelu\":",
    "        return nn.gelu_approx(g) * u",
    "    if act == \"silu_clamp\":",
    "        g = mx.minimum(g, _CLAMP_LIMIT)",
    "        u = mx.clip(u, -_CLAMP_LIMIT, _CLAMP_LIMIT)",
    "        return nn.silu(g) * u",
    "    return nn.silu(g) * u",
    "",
]) + "\n"

orig = src
for old, new, label in ((OLD1, NEW1, "activations"), (OLD2, NEW2, "glu_expr"),
                        (OLD3, NEW3, "host_act")):
    n = src.count(old)
    assert n == 1, "%s: expected exactly 1 occurrence, found %d" % (label, n)
    src = src.replace(old, new, 1)

open(P + ".preclamp.bak", "w").write(orig)
open(P, "w").write(src)
import py_compile
py_compile.compile(P, doraise=True)
print("PATCHED md5_before=%s md5_after=%s" % (md5_before, hashlib.md5(src.encode()).hexdigest()))
