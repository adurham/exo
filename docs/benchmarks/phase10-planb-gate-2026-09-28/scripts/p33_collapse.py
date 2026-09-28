#!/usr/bin/env python3
"""p33 -- head collapse check on states saved by a trace run.

For L in (38, 39): collapse = hc_pre(h_L, pm_L); out = norm(collapse);
logits = out @ embed.T (the checkpoint ties embeddings); prints rms + top-5 +
the L38-vs-L39 logit delta. TRACE_RDIR points at the run's records dir.
"""
import importlib.util
import os

import numpy as np
import mlx.core as mx

HOME = os.path.expanduser("~")
spec = importlib.util.spec_from_file_location("p30", HOME + "/p30_exl3_trace.py")
p30 = importlib.util.module_from_spec(spec)
spec.loader.exec_module(p30)
from deepseek_v41_mlx.hyper_connections import hc_pre          # noqa: E402
from deepseek_v41_mlx.layers import RMSNorm                     # noqa: E402

RDIR = os.path.expanduser(os.environ.get("TRACE_RDIR", "~/p30-records"))


def _rs(a):
    return float(mx.sqrt(mx.mean(mx.square(a.astype(mx.float32)))))


print("[p33] rdir=%s act=%s" % (RDIR, p30.ENV_ACT), flush=True)
hw = p30.mx_t("embed.weight")
nrm = RMSNorm(p30.ARGS.dim, p30.ARGS.norm_eps)
nrm.load_weights([("weight", p30.mx_t("norm.weight", mx.float32))], strict=False)

res = {}
for L in (38, 39):
    h = mx.load(os.path.join(RDIR, "h_%02d.npy" % L))
    pm = mx.load(os.path.join(RDIR, "pm_%02d.npy" % L))
    col = hc_pre(h, pm)
    nout = nrm(col)
    lg = (nout.astype(mx.float32) @ hw.astype(mx.float32).T)[0, -1]
    mx.eval(lg)
    top = mx.argsort(-lg)[:5]
    mx.eval(top)
    print("[p33] L%d collapse_rms=%.3f norm_out_rms=%.3f top5=%s logits_absmax=%.3f" % (
        L, _rs(col), _rs(nout), [int(i) for i in top], float(mx.max(mx.abs(lg)))), flush=True)
    res[L] = np.array(lg)
d = np.abs(res[38] - res[39])
print("[p33] |logits38-logits39| mean=%.4f max=%.3f  argmax38=%d argmax39=%d" % (
    float(d.mean()), float(d.max()), int(res[38].argmax()), int(res[39].argmax())), flush=True)
print("[p33] DONE")
