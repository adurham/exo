#!/usr/bin/env python3
"""p47 -- first real end-to-end V4.1 decode across both nodes (TP=2, JACCL).

Each rank builds the full model with its half-width expert slice
(exl3_build.build_model world=2), attention/dense replicated, routed partials
all_summed. Greedy decode, no MTP. Reports prefill and decode tok/s + text.
Env: P47_RANK (0/1), P47_TOKENS (decode length, default 200), P47_PROMPT.
"""
import json, os, sys, time
import numpy as np

HOME = os.path.expanduser("~")
sys.path.insert(0, HOME + "/dsv41-test")
import mlx.core as mx

MODEL = HOME + "/.exo/models/dealignai--DeepSeek-V4.1-Flash-UNCENSORED-EXL3-2.9bpw"
NATIVE = HOME + "/.exo/models/deepseek-ai--DeepSeek-V4.1-Flash-engram"
TOKMAP = HOME + "/dsv41-test/engram_token_map.json"


def log(*a):
    print(f"[p47 r{RANK}]", *a, flush=True)


group = mx.distributed.init(backend="jaccl", strict=True)
RANK, WORLD = group.rank(), group.size()
log(f"jaccl up world={WORLD}")
mx.eval(mx.distributed.all_sum(mx.ones(4), group=group))
log("all_sum ok")

from mlx_lm.models.deepseek_v41 import exl3_build as eb  # noqa: E402

t0 = time.time()
model, rep = eb.build_model(MODEL, native_dir=NATIVE, rank=RANK, world=WORLD, group=group)
model.set_token_map(json.load(open(TOKMAP)))
mx.eval(mx.distributed.all_sum(mx.ones(1), group=group))
log(f"built in {time.time()-t0:.1f}s active={mx.get_active_memory()/1e9:.1f}GB "
    f"peak={mx.get_peak_memory()/1e9:.1f}GB")

EVAL_EVERY = int(os.environ.get("P47_EVAL_EVERY", "4"))
if EVAL_EVERY:
    from mlx_lm.models.deepseek_v41.model import Block
    _orig_block_call = Block.__call__

    def _fenced(self, x, pre_mix, start_pos, cache, shared):
        h, pm = _orig_block_call(self, x, pre_mix, start_pos, cache, shared)
        if self.layer_id % EVAL_EVERY == EVAL_EVERY - 1:
            mx.eval(h, pm)
        return h, pm

    Block.__call__ = _fenced
log(f"eval every {EVAL_EVERY} layers")

from tokenizers import Tokenizer  # noqa: E402
import jinja2  # noqa: E402

tok = Tokenizer.from_file(MODEL + "/tokenizer.json")
tmpl = jinja2.Environment().from_string(open(MODEL + "/chat_template.jinja").read())
question = os.environ.get("P47_PROMPT",
    "Explain in a few paragraphs why the sky is blue, then give one surprising fact about light.")
text = tmpl.render(messages=[{"role": "user", "content": question}],
                   add_generation_prompt=True, enable_thinking=False)
ids = tok.encode(text, add_special_tokens=False).ids
EOS = 1
n_new = int(os.environ.get("P47_TOKENS", "200"))
log(f"prompt {len(ids)} tokens")


def pick(logits):
    t = mx.argmax(logits[:, -1, :], axis=-1).astype(mx.int32)
    # rank 0 decides; identical on both ranks by construction, enforced anyway
    t = mx.distributed.all_sum(mx.where(RANK == 0, t, 0), group=group)
    return t


def run(seed_ids, n):
    cache = model.make_cache(1, max_seq_len=len(seed_ids) + n + 16)
    x = mx.array([seed_ids])
    t0 = time.time()
    nxt = pick(model(x, cache, last_logit_only=True))
    mx.eval(nxt)
    t_pf = time.time() - t0
    out = [int(nxt.item())]
    step_t = []
    for _ in range(n - 1):
        if out[-1] == EOS:
            break
        s = time.time()
        nxt = pick(model(nxt[None], cache, last_logit_only=True))
        mx.eval(nxt)
        step_t.append(time.time() - s)
        out.append(int(nxt.item()))
    return t_pf, step_t, out


# warmup (kernel JIT) then measured run
run(ids[:16], 4)
t_pf, st, out = run(ids, n_new)
st = np.array(st[2:]) if len(st) > 4 else np.array(st)
log(f"prefill {len(ids)} tok in {t_pf:.2f}s = {len(ids)/t_pf:.1f} tok/s")
log(f"decode {len(out)} tok: median {1/np.median(st):.2f} tok/s "
    f"(mean step {st.mean()*1e3:.1f} ms, p10 {np.percentile(st,10)*1e3:.1f} p90 {np.percentile(st,90)*1e3:.1f})")
if RANK == 0:
    print("----- TEXT -----\n" + tok.decode(out) + "\n----- END -----", flush=True)
if os.environ.get("P47_PROFILE") == "1":
    import collections
    from mlx_lm.models.deepseek_v41 import model as M, moe as MO, attention as A, engram as E
    from mlx_lm.models.deepseek_v41 import exl3_build as EB
    acc = collections.defaultdict(float)

    def timed(name, fn):
        def w(*a, **k):
            mx.eval([t for t in a if isinstance(t, mx.array)])
            s = time.perf_counter(); r = fn(*a, **k)
            mx.eval(r if isinstance(r, (mx.array, tuple, list)) else [])
            acc[name] += time.perf_counter() - s
            return r
        return w
    A.Attention.__call__ = timed("attention", A.Attention.__call__)
    MO.Gate.__call__ = timed("gate", MO.Gate.__call__)
    EB.Exl3Experts.__call__ = timed("experts", EB.Exl3Experts.__call__)
    MO.SharedExpert.__call__ = timed("shared_expert", MO.SharedExpert.__call__)
    E.Engram.__call__ = timed("engram", E.Engram.__call__)
    _ag = mx.distributed.all_sum
    def _as(x, group=None):
        mx.eval(x); s = time.perf_counter(); r = _ag(x, group=group); mx.eval(r)
        acc["all_sum"] += time.perf_counter() - s; return r
    MO.mx.distributed.all_sum = _as
    _hc = M.hc_mixes
    M.hc_mixes = timed("hc_mixes", _hc)
    M.hc_post = timed("hc_post", M.hc_post)
    EB.Exl3Proj.__call__ = timed("head_or_dense_proj_toplevel", EB.Exl3Proj.__call__)
    t_pf, st2, out2 = run(ids, 22)
    n = len(st2)
    tot = sum(st2)
    log(f"PROFILE over {n} fenced decode steps, {tot/n*1e3:.1f} ms/step (fenced):")
    for k, v in sorted(acc.items(), key=lambda kv: -kv[1]):
        log(f"   {k:32s} {v/(n+1)*1e3:7.2f} ms/step")
log("P47_DONE")
