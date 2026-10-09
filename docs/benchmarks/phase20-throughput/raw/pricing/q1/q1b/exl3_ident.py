# exl3 byte-identity: prod 16830e1 inline TP-slice construction vs cb163da _dense_slice (exl3 mode),
# real Exl3Proj forward on a synthetic EXL3 group; prints a sha256 of the outputs.
import sys, hashlib, numpy as np, mlx.core as mx
from unittest import mock
from mlx_lm.models.exl3.ref.layer import EXL3Layer
from mlx_lm.models.exl3 import EXL3Linear
from mlx_lm.models.deepseek_v41 import exl3_build as eb
r = np.random.default_rng(3); K = 4
def synth(fin, fout):
    return EXL3Layer(key="g", in_features=fin, out_features=fout, k=K,
        trellis=r.integers(0, 2**16, size=(fin//16, fout//16, 256*K//16), dtype=np.uint16),
        suh=r.choice(np.array([-1.,1.],np.float16), size=fin), svh=r.choice(np.array([-1.,1.],np.float16), size=fout), mul1=True)
lays = {"wq_b": synth(256, 512), "wo_b": synth(512, 256)}
h = hashlib.sha256()
for rank in (0, 1):
    for name, ax in (("wq_b", "out"), ("wo_b", "in")):
        if hasattr(eb, "_dense_slice"):
            with mock.patch.object(eb, "DENSE_MODE", "exl3"), \
                 mock.patch("mlx_lm.models.exl3.loader.load_dense_layer", lambda ck, n: lays[n]):
                p = eb._dense_slice(None, name, axis=ax, rank=rank, world=2)
        else:   # production 16830e1 build_block inline expression
            p = eb.Exl3Proj(EXL3Linear(eb._slice_dense(lays[name], axis=ax, rank=rank, world=2)))
        fin = lays[name].in_features // (2 if ax == "in" else 1)
        for M in (1, 4, 512):
            x = mx.array(np.random.default_rng(M + rank).standard_normal((1, M, fin)).astype(np.float32)).astype(mx.bfloat16)
            y = p(x); mx.eval(y)
            h.update(np.array(y.astype(mx.float32)).tobytes())
print(sys.argv[1], h.hexdigest())
