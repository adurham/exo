# MLX buffer-cache residue after the dense build of N layers (real DSv4.1-Flash shapes, rank0/world2),
# EXL3 k varied per tensor like the 2.9bpw checkpoint. Compares DENSE modes.
import sys, numpy as np, mlx.core as mx
from unittest import mock
from mlx_lm.models.exl3.ref.layer import EXL3Layer
from mlx_lm.models.exl3 import EXL3Linear
from mlx_lm.models.deepseek_v41 import exl3_build as eb
D, QL, H, HD, OG, OL, I = 5120, 1280, 64, 512, 8, 1024, 2304
rep = {"attn.wq_a":(D,QL),"attn.wkv":(D,HD),"attn.compressor.wkv":(D,HD),"attn.indexer.wq_b":(QL,32*128),"attn.indexer.wk":(HD,128)}
rep.update({f"attn.wo_a.slice.{g}":(H*HD//OG, OL) for g in range(4)})
sh = {"attn.wq_b":(QL,H*HD,"out"),"attn.wo_b":(OG*OL,D,"in"),"ffn.shared_experts.w1":(D,I,"out"),"ffn.shared_experts.w3":(D,I,"out"),"ffn.shared_experts.w2":(I,D,"in")}
def synth(name, fin, fout, seed):
    r = np.random.default_rng(seed); K = 2 + seed % 5           # k in 2..6
    return EXL3Layer(key=name, in_features=fin, out_features=fout, k=K,
        trellis=r.integers(0, 2**16, size=(fin//16, fout//16, 256*K//16), dtype=np.uint16),
        suh=r.choice(np.array([-1.,1.],np.float16), size=fin), svh=r.choice(np.array([-1.,1.],np.float16), size=fout), mul1=True)
mode, N = sys.argv[1], int(sys.argv[2])
cache = {}
def ld(ck, n):
    L, g = n.split(".", 2)[1], n.split(".", 2)[2]
    s = (rep.get(g) or sh[g][:2])
    key = (g, int(L) % 5)
    if key not in cache: cache[key] = synth(n, *s, hash(key) % 1000)
    return cache[key]
keep = []; mx.clear_cache(); a0 = mx.get_active_memory()
with mock.patch.object(eb,"DENSE_MODE",mode), mock.patch("mlx_lm.models.exl3.loader.load_dense_layer", ld), \
     mock.patch.object(eb, "load_dense_linear", lambda ck, n: EXL3Linear(ld(ck, n))):
    for L in range(N):
        for g in rep: keep.append(eb._dense(None, f"layers.{L}.{g}"))
        for g,(fi,fo,ax) in sh.items(): keep.append(eb._dense_slice(None, f"layers.{L}.{g}", axis=ax, rank=0, world=2))
        if L in (0, N//2, N-1):
            print(f"{mode} after layer {L}: weights={(mx.get_active_memory()-a0)/2**30:.2f}GiB cache_residue={mx.get_cache_memory()/2**30:.2f}GiB", flush=True)
