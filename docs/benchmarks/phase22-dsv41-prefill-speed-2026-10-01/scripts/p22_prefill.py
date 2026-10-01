#!/usr/bin/env python3
"""p22_prefill -- DSv4.1 prefill baseline + attribution + config-only levers.

TWO-NODE (TP=2 over JACCL). Mirrors PRODUCTION by calling the REAL
``engine_prefill``: its source is extracted verbatim (ast) from the DEPLOYED
``exo/worker/engines/mlx/dsv41/session.py`` and compiled in place, so this is
production's driver and not a reimplementation of its semantics. The file's md5
and the extracted source are written into the run directory for the reviewer.

Arms (one process unless MLX_MAX_OPS_PER_BUFFER forces a relaunch):
  base       engine_prefill, taps ON, chunk 2048  -- production's shape
  notaps     engine_prefill, taps OFF             -- isolates the DSpark head's
                                                     per-chunk append_ctx cost
  pf_fenced  PF.prefill (fenced every 2 layers, no taps) -- the shape the older
                                                     245-254 tok/s numbers used
  msl16384   engine_prefill, cache sized like the served instance (16384)
  chunkN     engine_prefill, chunk N (2048/4096/8192)

Per arm: per-chunk wall ms, cumulative GPU time, dispatch count, peak + cache
memory, tokens-per-expert; plus a 64-token greedy sha256 oracle and the
first-token fp32 logits saved to .npy (the fallback oracle when a shape change
legitimately moves a near-tie token: chunk-size changes are NOT expected
bit-exact, MLX_MAX_OPS_PER_BUFFER changes ARE).

P22_ATTR=1 wraps every module class + sparse_attn + all_sum with eval-fenced
exclusive-time instrumentation for SHARES only (eval-fencing inflates absolutes
~3x, measured in a prior phase; take absolutes from the unfenced base arm).

Env: P22_TAG=run1 P22_LENS=2048,8192,16384 P22_REPS=2 P22_CHUNKS=2048
     P22_ARMS=base,notaps,pf_fenced,msl16384  P22_CHUNK=2048
     P22_ATTR=0|1 P22_ATTR_LEN=16384 P22_ORACLE=1 P22_ORACLE_TOKENS=64
     P22_PROMPTS=~/p22_prompts P22_MAX_SEQ_PAD=1024 P22_SESSION_PY=<deployed session.py>
"""
import ast
import hashlib
import json
import os
import time
from collections import defaultdict

import numpy as np

HOME = os.path.expanduser("~")
import sys
sys.path.insert(0, os.environ.get("P22_PKG", HOME + "/dsv41-test"))
import mlx.core as mx  # noqa: E402

MODEL = HOME + "/.exo/models/dealignai--DeepSeek-V4.1-Flash-UNCENSORED-EXL3-2.9bpw"
NATIVE = HOME + "/.exo/models/deepseek-ai--DeepSeek-V4.1-Flash-engram"
PROMPTS = os.environ.get("P22_PROMPTS", HOME + "/p22_prompts")
TAG = os.environ.get("P22_TAG", "run")
LENS = [int(x) for x in os.environ.get("P22_LENS", "2048,8192,16384").split(",")]
REPS = int(os.environ.get("P22_REPS", "2"))
BASE_CHUNK = int(os.environ.get("P22_CHUNK", "2048"))
CHUNKS = [int(x) for x in os.environ.get("P22_CHUNKS", "").split(",") if x]
ARMS = [a for a in os.environ.get("P22_ARMS", "base,notaps,pf_fenced,msl16384").split(",") if a]
ATTR = os.environ.get("P22_ATTR", "0") == "1"
ATTR_LEN = int(os.environ.get("P22_ATTR_LEN", "16384"))
ORACLE = os.environ.get("P22_ORACLE", "1") == "1"
ORACLE_TOKENS = int(os.environ.get("P22_ORACLE_TOKENS", "64"))
PAD = int(os.environ.get("P22_MAX_SEQ_PAD", "1024"))
SESSION_PY = os.environ.get(
    "P22_SESSION_PY",
    HOME + "/repos/exo/src/exo/worker/engines/mlx/dsv41/session.py")
RUN_DIR = HOME + f"/p22_out/{TAG}"
os.makedirs(RUN_DIR, exist_ok=True)

group = mx.distributed.init(backend="jaccl", strict=True)
RANK = group.rank()

# Production's own load path raises the wired limit BEFORE building
# (``load_dsv41`` -> ``set_wired_limit_for_model(model_card.storage_size)``).
# Mirror it, and do it AFTER the build too, exactly as the skill's rule says
# (the call only re-partitions buffers that already exist).
WIRED_BYTES = 210713013432  # the card's storage_size, from its own toml
mx.set_wired_limit(int(mx.device_info()["max_recommended_working_set_size"]))


def log(what, **kw):
    """Progress line. Rank 0 also mirrors into `progress_<tag>.jsonl` so the run
    survives a lost console; non-zero ranks print too (their stdout goes to their
    own log and is how a peer-side failure is seen -- a rank-1 crash otherwise
    reaches rank 0 only as a generic peer/connection error)."""
    line = f"[p22 r{RANK}] {what} " + json.dumps(kw, default=str)
    print(line, flush=True)
    if RANK == 0:
        with open(os.path.join(RUN_DIR, "progress.jsonl"), "a") as f:
            f.write(line + "\n")


# --------------------------------------- production's own prefill, verbatim
SESSION_MD5 = hashlib.md5(open(SESSION_PY, "rb").read()).hexdigest()
_src = open(SESSION_PY).read()
_fn_src = None
for _node in ast.parse(_src).body:
    if isinstance(_node, ast.FunctionDef) and _node.name == "engine_prefill":
        _fn_src = ast.get_source_segment(_src, _node)
        break
if _fn_src is None:
    raise SystemExit(f"engine_prefill not found in {SESSION_PY}")
_ns = {"mx": mx, "np": np, "Any": __import__("typing").Any, "time": time}
# prepend the future import: the extracted segment is normally compiled inside a
# module that has `from __future__ import annotations`, so its annotations are
# strings. Without it, `model: Any` evaluates at def time and NameErrors.
exec(compile("from __future__ import annotations\n" + _fn_src, SESSION_PY, "exec"), _ns)  # noqa: S102
engine_prefill = _ns["engine_prefill"]
open(os.path.join(RUN_DIR, "engine_prefill_extracted.py"), "w").write(_fn_src)


from mlx_lm.models.deepseek_v41 import exl3_build as eb  # noqa: E402
from mlx_lm.models.deepseek_v41 import prefill as PF  # noqa: E402
from mlx_lm.models.deepseek_v41 import collective as CO  # noqa: E402
from mlx_lm.models.deepseek_v41 import attention as ATT  # noqa: E402

log("env", tag=TAG, base_chunk=BASE_CHUNK, chunks=CHUNKS, lens=LENS, reps=REPS,
    arms=ARMS, attr=ATTR, oracle=ORACLE, session_py=SESSION_PY,
    session_md5=SESSION_MD5,
    ops_per_buffer=os.environ.get("MLX_MAX_OPS_PER_BUFFER"),
    max_mb_per_buffer=os.environ.get("MLX_MAX_MB_PER_BUFFER"))

# -- import-resolution audit: the shared venv's .pth files can silently resolve
#    `mlx_lm` to the production checkout instead of this tree (repos/exo/AGENTS.md).
#    Record what actually got imported so a wrong-tree run is visible, not silent.
import mlx_lm as _pkg  # noqa: E402
from mlx_lm.models.deepseek_v41 import model as _m  # noqa: E402
log("imports", mlx_lm=_pkg.__file__, deepseek_v41_model=_m.__file__,
    sys_path0=sys.path[0], mlx=mx.__file__ if hasattr(mx, "__file__") else "n/a")

t0 = time.perf_counter()
model, report = eb.build_model(MODEL, native_dir=NATIVE, rank=RANK, world=2,
                               group=group)
model.set_token_map(json.load(open(HOME + "/dsv41-test/engram_token_map.json")))
mx.eval(mx.distributed.all_sum(mx.ones(1), group=group))
build_s = time.perf_counter() - t0
head = None
try:
    from mlx_lm.models.exl3.loader import Exl3Checkpoint
    head = eb.build_mtp(Exl3Checkpoint(MODEL), model.args, rank=RANK, world=2,
                        group=group)
    log("head_attached", stages=len(head.stages), block=head.block_size)
except Exception as e:  # noqa: BLE001
    log("head_attach_failed", err=f"{type(e).__name__}: {e}")
t0 = time.perf_counter()
warm = PF.load_warmup(model, head)
# re-apply after the weights exist (the call only re-partitions existing buffers)
mx.set_wired_limit(int(mx.device_info()["max_recommended_working_set_size"]))
t_wired = time.perf_counter() - t0
log("built", build_s=round(build_s, 1), warmup_s=round(t_wired, 1),
    warmup=warm, active_gb=round(mx.get_active_memory() / 1e9, 2),
    peak_gb=round(mx.get_peak_memory() / 1e9, 2),
    cache_gb=round(mx.metal.get_cache_memory() / 1e9, 2))

# ---------------------------------------------------------------- attribution
ACC = defaultdict(float)
CNT = defaultdict(int)
STACK = []
ON = [False]


def _ev(o):
    if isinstance(o, mx.array):
        mx.eval(o)
    elif isinstance(o, (tuple, list)):
        for v in o:
            _ev(v)
    elif isinstance(o, dict):
        for v in o.values():
            _ev(v)


def timed(name, fn):
    def w(*a, **k):
        if not ON[0]:
            return fn(*a, **k)
        STACK.append(0.0)
        t = time.perf_counter()
        o = fn(*a, **k)
        _ev(o)
        dt = (time.perf_counter() - t) * 1e3
        child = STACK.pop()
        ACC[name] += dt - child
        CNT[name] += 1
        if STACK:
            STACK[-1] += dt
        return o
    return w


if ATTR:
    classes = {}
    for m in list(model.modules()) + (list(head.modules()) if head else []):
        c = type(m)
        if c.__name__ == "Model" or c in classes:
            continue
        classes[c] = c.__call__
    for c, orig in classes.items():
        c.__call__ = (lambda name, f: (
            lambda self, *a, **k: timed(name, lambda *aa, **kk: f(self, *aa, **kk))(*a, **k)
        ))(c.__name__, orig)
    ATT.sparse_attn = timed("fn:sparse_attn", ATT.sparse_attn)
    CO.all_sum = timed("fn:all_sum", CO.all_sum)
    log("attr_wrapped", n=len(classes),
        classes=sorted(c.__name__ for c in classes))


def bucket_report(tag, total_ms):
    top = sorted(ACC.items(), key=lambda kv: -kv[1])
    log(f"attr:{tag}", total_ms=round(total_ms, 1),
        buckets={k: [round(v, 1), CNT[k]] for k, v in top[:16]})
    ACC.clear()
    CNT.clear()


# ---------------------------------------------------------------------- prompt
def prompt_ids(L):
    return json.load(open(f"{PROMPTS}/prompt_{L // 1024}k_ids.json"))


TAPS_IDS = list(model.args.dspark_target_layer_ids)


def tapcat(t):
    return mx.concatenate([t[x] for x in TAPS_IDS], axis=-1)


# ------------------------------------------------------------------- one arm
def run_arm(ids, *, chunk, driver="engine", taps=True, max_seq_len=None,
            attr_chunk_idx=None, label="", want_oracle=ORACLE):
    """One prefill arm. Returns (anchor_logits, cache, summary_dict)."""
    n = len(ids)
    msl = max_seq_len if max_seq_len is not None else n + PAD
    cache = model.make_cache(1, max_seq_len=msl)
    dsc = head.make_cache(1) if (taps and driver == "engine" and head is not None) else None
    mx.reset_peak_memory()
    mx.metal.reset_gpu_time()
    mx.metal.reset_dispatch_count()
    per_chunk = []
    t_all = time.perf_counter()
    gpu_start = mx.metal.gpu_time_ns()

    if driver == "engine":
        taps_out = [] if dsc is not None else None

        # progress() is the driver's own hook: per-chunk walls with NO change to
        # the driver's behaviour (the engine uses it for the same purpose).
        def progress(nchunks, rows_done, elapsed):
            per_chunk.append({
                "idx": nchunks - 1, "rows_done": rows_done,
                "wall_ms": round(elapsed * 1e3, 1),
                "gpu_ms_cum": round((mx.metal.gpu_time_ns() - gpu_start) / 1e6, 1),
                "dispatches_cum": mx.metal.dispatch_count(),
                "peak_gb": round(mx.get_peak_memory() / 1e9, 2),
                "cache_mb": round(mx.metal.get_cache_memory() / 1e6, 1),
            })

        ON[0] = attr_chunk_idx is not None
        out = engine_prefill(model, ids, cache, chunk=chunk, long_chunk=chunk,
                             return_taps=(dsc is not None),
                             last_logit_only=True, argmax=False,
                             taps_out=taps_out, progress=progress)
        ON[0] = False
        for ct in (taps_out or []):
            head.append_ctx(tapcat(ct), dsc)
        if dsc is not None:
            mx.eval(*[w.win_kv for w in dsc])
    else:  # the older fenced driver, same call shape the 245-254 numbers used
        ON[0] = attr_chunk_idx is not None
        t0 = time.perf_counter()
        out = PF.prefill(model, ids, cache, chunk=chunk, long_chunk=chunk,
                         return_taps=False, last_logit_only=True, argmax=False)
        dt = (time.perf_counter() - t0) * 1e3
        ON[0] = False
        per_chunk.append({"idx": 0, "rows_done": n, "wall_ms": round(dt, 1),
                          "gpu_ms_cum": round((mx.metal.gpu_time_ns() - gpu_start) / 1e6, 1),
                          "dispatches_cum": mx.metal.dispatch_count(),
                          "peak_gb": round(mx.get_peak_memory() / 1e9, 2),
                          "cache_mb": round(mx.metal.get_cache_memory() / 1e6, 1)})

    total = time.perf_counter() - t_all
    if attr_chunk_idx is not None:
        bucket_report(label, total * 1e3)
    res = {
        "label": label, "driver": driver, "taps": bool(dsc is not None),
        "rows": n, "chunk": chunk,
        "n_chunks": max(1, -(-n // chunk)), "max_seq_len": msl,
        "total_s": round(total, 2), "tok_s": round(n / total, 1),
        "peak_gb": round(mx.get_peak_memory() / 1e9, 2),
        "gpu_ms_total": round((mx.metal.gpu_time_ns() - gpu_start) / 1e6, 1),
        "dispatches_total": mx.metal.dispatch_count(),
        "tokens_per_expert_at_chunk": round(chunk * 6 / 192, 1),
        "per_chunk": per_chunk,
    }
    if want_oracle:
        # engine_prefill returns (out, taps) when return_taps is on
        handle = out[0] if isinstance(out, tuple) else out
        a = handle.reshape(-1).astype(mx.float32)
        mx.eval(a)
        first = int(mx.argmax(a).item())
        toks = [first]
        nxt = mx.array([[first]], dtype=mx.int32)
        dts = []
        for _ in range(max(0, ORACLE_TOKENS - 1)):
            t0 = time.perf_counter()
            nxt = model(nxt, cache, last_logit_only=True, argmax=True)
            mx.eval(nxt)
            dts.append((time.perf_counter() - t0) * 1e3)
            toks.append(int(nxt.reshape(-1)[-1].item()))
        np.save(os.path.join(RUN_DIR, f"logits_{label}.npy"), np.asarray(a))
        med = round(float(np.median(dts)), 1) if dts else None
        res.update({
            "oracle_sha256": hashlib.sha256(
                np.asarray(toks, dtype=np.int64).tobytes()).hexdigest(),
            "oracle_tokens": toks,
            "decode_ms_median": med,
            "decode_tok_s": round(1000 / med, 1) if med else None,
        })
    return out, cache, res


# ------------------------------------------------------------------- main loop
results = []
for rep in range(REPS):
    for L in LENS:
        ids = prompt_ids(L)
        plan = []
        if "base" in ARMS:
            plan.append(("base", dict(chunk=BASE_CHUNK, driver="engine", taps=True)))
        if "notaps" in ARMS and L == 8192:
            plan.append(("notaps", dict(chunk=BASE_CHUNK, driver="engine", taps=False)))
        if "pf_fenced" in ARMS and L == 8192:
            plan.append(("pf_fenced", dict(chunk=BASE_CHUNK, driver="pf", taps=False)))
        if "msl16384" in ARMS and L == 8192:
            plan.append(("msl16384", dict(chunk=BASE_CHUNK, driver="engine",
                                         taps=True, max_seq_len=16384)))
        for c in CHUNKS:
            if c != BASE_CHUNK:
                plan.append((f"chunk{c}", dict(chunk=c, driver="engine", taps=True)))
        for label, kw in plan:
            lbl = f"rep{rep}_L{L}_{label}"
            attr_idx = 0 if (ATTR and L == ATTR_LEN and label == "base") else None
            anchor, cache, res = run_arm(ids, attr_chunk_idx=attr_idx, label=lbl, **kw)
            res["rep"] = rep
            res["L"] = L
            log("arm", **{k: v for k, v in res.items()
                          if k not in ("per_chunk", "oracle_tokens")})
            log("per_chunk", label=lbl,
                v=[(c.get("idx"), c.get("rows_done"), c.get("wall_ms"),
                    c.get("gpu_ms_cum"), c.get("peak_gb"), c.get("cache_mb"))
                   for c in res["per_chunk"]])
            if ORACLE:
                log("oracle", label=lbl, sha=res["oracle_sha256"],
                    dec_ms=res["decode_ms_median"], tok_s=res["decode_tok_s"],
                    toks=res["oracle_tokens"][:12])
            results.append(res)
            del cache, anchor
            mx.clear_cache()

payload = {"tag": TAG, "rank": RANK, "results": results,
           "env": {"base_chunk": BASE_CHUNK, "chunks": CHUNKS, "lens": LENS,
                   "arms": ARMS, "session_md5": SESSION_MD5,
                   "MLX_MAX_OPS_PER_BUFFER": os.environ.get("MLX_MAX_OPS_PER_BUFFER"),
                   "MLX_MAX_MB_PER_BUFFER": os.environ.get("MLX_MAX_MB_PER_BUFFER")}}
with open(os.path.join(RUN_DIR, "results.json"), "w") as f:
    json.dump(payload, f, indent=1)
log("WROTE", dir=RUN_DIR)
if RANK == 0:
    print("P22_DONE", flush=True)
