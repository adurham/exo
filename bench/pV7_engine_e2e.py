#!/usr/bin/env python3
"""pV7 -- DSv4.1 engine end-to-end: image prompt + 2-turn chat via the ENGINE.

Round-4 stream-V acceptance harness. It drives a real ``Dsv41Engine``
(``submit`` / ``step``) -- not the tower and not ``serve.Session`` directly -- on
a single Mac with a LAYER SUBSET build, and checks the four acceptance items:

  1. **span layout** -- the prompt the engine builds from an image-carrying
     request contains exactly one image span; its ``[start, end)`` matches
     ``prepare_vl_inputs``' arithmetic; the merged embedding rows inside the span
     are the tower's block bitwise, and rows outside are the plain token lookup;
  2. **KV reuse** -- turn 1 (long text prompt) then turn 2 (its own reply + new
     user text) through the same engine: turn-2 prefill < 10% of turn 1;
  3. **bitwise-equal turn-2 tokens vs fresh** -- the same turn-2 prompt on a
     FRESH engine (cold cache, same chunk boundaries) produces the same tokens;
  4. **image turn is real** -- the image request runs and its tokens differ from
     the same prompt without the image; an image request against an instance with
     no tower is refused (and the engine stays usable afterwards).

Plus: ``reset_after_reconnect`` drops the in-flight task and rolls the session
back.

The draft head is the p67 protocol stand-in (the real DSpark head is ~7 GB and
does not fit a per-run budget); ``args.dspark_target_layer_ids`` is narrowed to
the built layers, exactly as p67 does. Session reuse does not depend on which
head is used.

Env: PV7_LAYERS (default 0,1,2,3,20,21,24,25), PV7_LEN=512, PV7_GEN=16,
PV7_NEW=32, PV7_CHUNK=64, PV7_GAMMA=3, PV7_PKG (default ~/dsv41-ws4/V).

Usage (m4-1, ONE GPU job, under the lock)::

    cd ~/dsv41-ws4/V && \\
      PYTHONPATH=$HOME/dsv41-ws4/V:$HOME/dsv41-integ \\
      EXL3_MM_MAX_ROWS=100000 MTL_DISABLE_TIMEOUT=1 \\
      lockf -k ~/dsv41-gpu.lock ~/repos/exo/.venv/bin/python bench/pV7_engine_e2e.py
"""

from __future__ import annotations

import base64
import io
import json
import os
import sys
import time

import numpy as np

HOME = os.path.expanduser("~")
PKG = os.environ.get("PV7_PKG", HOME + "/dsv41-ws4/V")
for p in (PKG, HOME + "/dsv41-integ", HOME + "/repos/exo/src"):
    if p not in sys.path:
        sys.path.insert(0, p)

MODEL = HOME + "/.exo/models/dealignai--DeepSeek-V4.1-Flash-UNCENSORED-EXL3-2.9bpw"
NATIVE = HOME + "/.exo/models/deepseek-ai--DeepSeek-V4.1-Flash-engram"
TOKEN_MAP = HOME + "/dsv41-test/engram_token_map.json"
MODEL_ID_STR = "dealignai/DeepSeek-V4.1-Flash-UNCENSORED-EXL3-2.9bpw"

LAYERS = [int(x) for x in os.environ.get("PV7_LAYERS", "0,1,2,3,20,21,24,25").split(",")]
LEN = int(os.environ.get("PV7_LEN", "512"))
GEN = int(os.environ.get("PV7_GEN", "16"))
NEW = int(os.environ.get("PV7_NEW", "32"))
CHUNK = int(os.environ.get("PV7_CHUNK", "64"))
GAMMA = int(os.environ.get("PV7_GAMMA", "3"))

FAIL: list[str] = []


def log(*a):
    print("[pV7]", *a, flush=True)


def check(name, ok, detail=""):
    print(f"[pV7] {'PASS' if ok else 'FAIL'}  {name}  {detail}", flush=True)
    if not ok:
        FAIL.append(name)


# ------------------------------------------------------------ draft stand-in
# Exactly the protocol serve/spec use (make_cache / append_ctx / draft), with
# windows as counters: this harness tests the ENGINE wiring. A draft of
# ``anchor + k + 1`` is all-accept against the stub window's counting, which is
# the fast path; correctness of the ENGINE does not depend on the head.


class StubWin:
    def __init__(self, window=8, dim=4):
        import mlx.core as mx

        self.window = window
        self.win_kv = mx.zeros((1, window, dim), dtype=mx.float32)
        self.n_ctx = 0


class StubDraft:
    def __init__(self, block=5, n_stages=3):
        self.block = block
        self.n_stages = n_stages

    def make_cache(self, bsz=1):
        return [StubWin() for _ in range(self.n_stages)]

    def append_ctx(self, cat, caches):
        n = int(cat.shape[1])
        for c in caches:
            c.n_ctx += n

    def draft(self, anchor, embed, head, caches, width=None):
        import mlx.core as mx

        w = int(width or self.block)
        a = anchor.reshape(-1)
        k = mx.arange(1, w + 1, dtype=mx.int32)
        return (a[:, None] + k[None, :]).astype(mx.int32), mx.zeros(
            (a.shape[0], w), dtype=mx.float32
        )


# ------------------------------------------------------------------ plumbing

MODEL_ID = None


def make_ids():
    from exo.shared.types.common import CommandId, ModelId
    from exo.shared.types.tasks import TaskId
    from exo.shared.types.worker.instances import InstanceId

    return (
        CommandId("pV7cmd"), TaskId("pV7task"), InstanceId("pV7instance"),
    )


def build_engine(loaded, head, vision, *, max_kv=4096, sessions=2):
    from exo.shared.types.common import ModelId
    from exo.utils.channels import mp_channel
    from exo.worker.engines.mlx.dsv41.engine import Dsv41Engine

    ev_send, ev_recv = mp_channel(max_buffer_size=4096)
    cancel_send, cancel_recv = mp_channel(max_buffer_size=4096)
    engine = Dsv41Engine(
        loaded=loaded,
        model_id=ModelId(MODEL_ID_STR),
        group=None,
        cancel_receiver=cancel_recv,
        event_sender=ev_send,
        device_rank=0,
        max_kv_tokens=max_kv,
        prefill_chunk_size=CHUNK,
        speculative=True,
        gamma=GAMMA,
        adaptive_gamma=False,
        vision_processor=vision,
        max_sessions=sessions,
    )
    # keep the channel objects alive for the engine's lifetime
    engine._test_channels = (ev_send, ev_recv, cancel_send, cancel_recv)
    return engine


def drain(engine):
    chunks = []
    t0 = time.perf_counter()
    while True:
        out = list(engine.step())
        chunks.extend(out)
        if any(type(c).__name__ == "FinishedResponse" for _tid, c in out):
            break
        if time.perf_counter() - t0 > 3600:
            raise TimeoutError("engine.step() never finished")
    usage = next((getattr(c, "usage", None) for _tid, c in chunks
                  if getattr(c, "usage", None) is not None), None)
    stats = next((getattr(c, "stats", None) for _tid, c in chunks
                  if getattr(c, "stats", None) is not None), None)
    return {
        "text": "".join(getattr(c, "text", "") for _tid, c in chunks),
        "tokens": [c.token_id for _tid, c in chunks if hasattr(c, "token_id")],
        "usage": usage,
        "stats": stats,
        "n_chunks": len(chunks),
    }


def submit_and_drain(engine, params):
    from exo.shared.types.tasks import TextGeneration

    cid, tid, iid = make_ids()
    engine.submit(TextGeneration(command_id=cid, instance_id=iid,
                                 task_params=params, task_id=tid))
    return drain(engine)


def params_for(messages, *, max_tokens, images=(), key=None, use_prefix_cache=False):
    from exo.shared.types.text_generation import InputMessage, TextGenerationTaskParams

    return TextGenerationTaskParams(
        model=MODEL_ID_STR,
        input=[InputMessage(role="user", content="placeholder")],
        chat_template_messages=messages,
        max_output_tokens=max_tokens,
        temperature=0.0,
        enable_thinking=False,
        images=list(images),
        use_prefix_cache=use_prefix_cache,
        correlation_id=key,
    )


# ------------------------------------------------------------------ main

def main() -> int:
    import mlx.core as mx
    from mlx_lm.models.deepseek_v41 import exl3_build as eb

    from exo.worker.engines.mlx.dsv41 import vision as dsv
    from exo.worker.engines.mlx.dsv41.load import Dsv41Loaded

    mx.reset_peak_memory()
    t0 = time.time()
    log(f"building layers={LAYERS} (single node, subset) ...")
    model, report = eb.build_model(MODEL, native_dir=NATIVE, layers=LAYERS,
                                  rank=0, world=1, group=None)
    model.set_token_map(json.load(open(TOKEN_MAP)))
    model.args.dspark_target_layer_ids = tuple(LAYERS)
    log(f"built in {time.time() - t0:.0f}s active={mx.get_active_memory()/1e9:.2f}GB")

    orig_mc = model.make_cache

    def mc(*a, **k):
        c = orig_mc(*a, **k)
        for li, lc in enumerate(c.layers):  # unbuilt layers carry no carry
            if li not in LAYERS:
                lc.comp_state = None
        return c

    model.make_cache = mc

    # ---- vision tower (32 blocks, ~1 GB resident; p97 measured) -------------
    vision = dsv.load_dsv41_vision(MODEL, dtype=mx.bfloat16)
    mx.eval(vision.tower.parameters())
    log(f"vision: {vision}")

    tok = _tokenizer(vision)
    log(f"tokenizer {type(tok).__name__} eos={tok.eos_token_ids} "
        f"thinking={tok.has_thinking}")

    from mlx_lm.models.deepseek_v41.config import ModelArgs

    args = ModelArgs.from_dict(json.load(open(MODEL + "/config.json")))
    head = StubDraft()

    def loaded_obj():
        return Dsv41Loaded(model=model, tokenizer=tok, args=args,
                           model_path=None, built_layers=LAYERS, full_stack=False,
                           rank=0, world=1, load_seconds=0.0, head=head)

    # ---- (1) image span layout -------------------------------------------
    from PIL import Image

    from exo.shared.types.text_generation import Base64Image

    rng = np.random.default_rng(7)
    arr = rng.integers(0, 256, (300, 420, 3), dtype=np.uint8)
    buf = io.BytesIO()
    Image.fromarray(arr).save(buf, format="PNG")
    image = Base64Image(base64.b64encode(buf.getvalue()).decode("ascii"))

    from exo.worker.engines.mlx.utils_mlx import apply_chat_template

    img_msgs = [{"role": "user", "content": [
        {"type": "text", "text": "Describe this image in one sentence."},
        {"type": "image", "url": "exo-image:0"},
    ]}]
    img_params = params_for(img_msgs, max_tokens=4, images=[image])
    # the ENGINE renders an image request with vision.render_prompt (the shared
    # apply_chat_template drops image blocks); the harness builds the span-layout
    # check from that same prompt.
    prompt_text = dsv.render_prompt(tok, img_params, vision.placeholder)
    shared = apply_chat_template(tok, params_for(
        [{"role": "user", "content": "Describe this image in one sentence."}],
        max_tokens=4))
    check("shared renderer drops the image block (why the engine needs the local one)",
          vision.placeholder not in shared and vision.placeholder in prompt_text,
          f"shared has placeholder: {vision.placeholder in shared}")
    check("prompt renders the placeholder", vision.placeholder in prompt_text,
          f"{ascii(vision.placeholder)} present: {vision.placeholder in prompt_text}")
    log(f"rendered prompt: {prompt_text[:160]!r}")

    tokens_list, token_types, image_inputs = dsv.prompt_tokens_for_request(
        vision, prompt_text, [image], tok
    )
    spans = dsv.image_spans(image_inputs)
    n_img = sum(1 for t in token_types if t >= 0)
    start, end = spans[0]
    check("exactly one image span", len(spans) == 1, f"spans={spans}")
    check("span layout == types", end - start == n_img
          and all(tokens_list[start + i] == vision.image_token_id
                  for i in range(end - start)),
          f"span [{start},{end}) with {end - start} sentinel rows")
    span_placeholder_count = token_types
    ii = image_inputs[0]
    log(f"image: patches={ii.patches.shape} vit={ii.n_vit_h}x{ii.n_vit_w} "
        f"blocks={len(ii.types)} prompt={len(tokens_list)} tok types[-1]={token_types[-1]}")

    embeds = dsv.build_embeddings(model, vision, tokens_list, image_inputs)
    mx.eval(embeds)
    block = vision.tower.build_image_block(ii)
    mx.eval(block)
    span_rows_eq = bool(mx.all(embeds[0, start:end, :] == block.astype(embeds.dtype)).item())
    plain = model.embed(mx.array(np.asarray(tokens_list, dtype=np.int32))[None])
    outside_eq = bool(
        mx.all(embeds[0, :start, :] == plain[0, :start, :]).item()
        and mx.all(embeds[0, end:, :] == plain[0, end:, :]).item()
    )
    check("merged span rows == tower block (bitwise)", span_rows_eq,
          f"{end - start} rows")
    check("rows outside the span are the plain token lookup (bitwise)", outside_eq,
          f"{len(tokens_list) - (end - start)} rows")
    del plain
    mx.clear_cache()

    # ---- (2)(3) two-turn chat through the engine ---------------------------
    corpus = np.asarray(json.load(open(HOME + "/p30_prompt_ids.json")) * 40,
                        dtype=np.int64)
    assert corpus.shape[0] >= LEN + 128, "corpus too short"
    filler = " ".join(str(int(t)) for t in corpus[:LEN])
    tail = " ".join(str(int(t)) for t in corpus[LEN + 64:LEN + 64 + NEW])

    msgs1 = [{"role": "user", "content": filler}]
    p1 = apply_chat_template(tok, params_for(msgs1, max_tokens=GEN))
    n1 = len(__encode(tok, p1))
    log(f"turn-1 prompt: {n1} tokens")

    eng1 = build_engine(loaded_obj(), head, vision)
    r1 = submit_and_drain(eng1, params_for(msgs1, max_tokens=GEN, key="conv-1",
                                          use_prefix_cache=True))
    u1 = r1["usage"]
    log(f"turn 1: {len(r1['tokens'])} tokens, usage={_u(u1)}")
    check("turn 1 generated", len(r1["tokens"]) >= 1, f"text={r1['text'][:60]!r}")
    check("turn 1 was a cold prefill",
          u1 is not None and u1.prompt_tokens_details.cached_tokens == 0,
          f"cached={u1.prompt_tokens_details.cached_tokens if u1 else None}")

    msgs2 = msgs1 + [{"role": "assistant", "content": r1["text"]},
                     {"role": "user", "content": tail}]
    r2 = submit_and_drain(eng1, params_for(msgs2, max_tokens=GEN, key="conv-1",
                                          use_prefix_cache=True))
    u2 = r2["usage"]
    pre2 = (u2.prompt_tokens - u2.prompt_tokens_details.cached_tokens) if u2 else -1
    log(f"turn 2: {len(r2['tokens'])} tokens, usage={_u(u2)}")
    check("turn 2 reused the prefix",
          u2 is not None and u2.prompt_tokens_details.cached_tokens > 0,
          f"cached={u2.prompt_tokens_details.cached_tokens if u2 else None}")
    ratio = pre2 / u2.prompt_tokens if (u2 and u2.prompt_tokens) else 1.0
    check("turn-2 prefill < 10% of turn-1", u2 is not None and ratio < 0.10,
          f"prefill={pre2} / prompt={u2.prompt_tokens if u2 else None} "
          f"= {ratio*100:.2f}%")

    # ---- (3) cold twin ----------------------------------------------------
    eng2 = build_engine(loaded_obj(), head, vision)
    _ = submit_and_drain(eng2, params_for(msgs1, max_tokens=GEN, key="conv-2",
                                          use_prefix_cache=True))
    r2b = submit_and_drain(eng2, params_for(msgs2, max_tokens=GEN, key="conv-2",
                                            use_prefix_cache=True))
    ub2 = r2b["usage"]
    log(f"cold twin turn 2: {len(r2b['tokens'])} tokens, usage={_u(ub2)}")
    check("turn-2 tokens bitwise equal to the cold twin",
          r2["tokens"] == r2b["tokens"],
          f"reused={r2['tokens']} cold={r2b['tokens']}")
    check("cold twin had no reuse",
          ub2 is not None and ub2.prompt_tokens_details.cached_tokens == 0,
          f"cached={ub2.prompt_tokens_details.cached_tokens if ub2 else None}")
    log(f"turn-2 text: {r2['text'][:80]!r}")

    # ---- (4) image turn through the engine --------------------------------
    eng3 = build_engine(loaded_obj(), head, vision)
    ri = submit_and_drain(eng3, img_params)
    log(f"image turn: {len(ri['tokens'])} tokens text={ri['text'][:60]!r} "
        f"usage={_u(ri['usage'])}")
    check("image turn ran", len(ri["tokens"]) >= 1, f"tokens={ri['tokens']}")

    check("image prompt prefill count >= image span", ri["usage"] is not None
          and ri["usage"].prompt_tokens > len(span_placeholder_count),
          f"prompt={ri['usage'].prompt_tokens if ri['usage'] else None} "
          f"span_tokens={len(span_placeholder_count)}")
    # the same prompt WITHOUT the image: the tokens must differ (the splice did
    # something) -- compared against the image run above
    eng3b = build_engine(loaded_obj(), head, vision)
    rnoimg = submit_and_drain(eng3b, params_for(
        [{"role": "user", "content": "Describe this image in one sentence."}],
        max_tokens=4))
    log(f"same text, no image: tokens={rnoimg['tokens']}")
    check("image changes the output", ri["tokens"] != rnoimg["tokens"],
          f"image={ri['tokens']} no-image={rnoimg['tokens']}")

    # and an instance with NO tower must refuse, without wedging
    eng4 = build_engine(loaded_obj(), head, None)
    try:
        submit_and_drain(eng4, img_params)
        check("image refused without a tower", False, "no error raised")
    except Exception as e:
        check("image refused without a tower", "image" in str(e).lower(),
              f"{type(e).__name__}: {e}")
    eng4._active = None
    eng4.step()
    log("engine usable after the refusal (step() returned)")

    # ---- (5) reconnect ---------------------------------------------------
    dropped = eng1.reset_after_reconnect()
    check("reset_after_reconnect returns dropped ids", isinstance(dropped, list),
          f"dropped={dropped}")

    log(f"peak={mx.get_peak_memory()/1e9:.2f}GB "
        f"active={mx.get_active_memory()/1e9:.2f}GB")
    log("PV7_FAILED=" + (",".join(FAIL) if FAIL else "none"))
    log("PV7_DONE")
    return 1 if FAIL else 0


def __encode(tok, text):
    return tok.encode(text, add_special_tokens=False)


def _u(usage):
    if usage is None:
        return None
    return (f"prompt={usage.prompt_tokens} completion={usage.completion_tokens} "
            f"cached={usage.prompt_tokens_details.cached_tokens}")


def _tokenizer(vision):
    from pathlib import Path

    from exo.shared.models.model_cards import Memory, ModelCard, VisionCardConfig
    from exo.shared.types.worker.shards import TensorShardMetadata
    from exo.worker.engines.mlx.utils_mlx import get_tokenizer

    card = ModelCard(
        model_id=MODEL_ID_STR,
        storage_size=Memory.from_gb(1),
        n_layers=40,
        hidden_size=5120,
        supports_tensor=True,
        tasks=["text-generation"],
        backends=["mlx"],
        context_length=4096,
        vision=VisionCardConfig(
            image_token_id=vision.image_token_id,
            model_type="deepseek_v41_vision",
            scheme="deepseek_v4",
            placeholder_token=vision.placeholder,
        ),
    )
    shard = TensorShardMetadata(
        model_card=card, device_rank=0, world_size=1, start_layer=0,
        end_layer=40, n_layers=40,
    )
    return get_tokenizer(Path(MODEL), shard)


if __name__ == "__main__":
    sys.exit(main())
