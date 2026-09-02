"""Offline exact render through the EXACT live request path:
raw JSON body -> ChatCompletionRequest.model_validate -> the adapter's
request->TextGenerationTaskParams conversion -> utils_mlx.apply_chat_template
-> deepseek_v4_encoding.encode_messages -> HF tokenizer ids.

Mirrors what the live /v1/chat/completions handler does, offline.
"""
import inspect
import json
import sys

sys.path.insert(0, "/Users/adam.durham/repos/exo/src")

from exo.api.types.api import ChatCompletionRequest  # noqa: E402

TOOL_DEF = {
    "type": "function",
    "function": {
        "name": "get_weather",
        "description": "Get current weather for a city.",
        "parameters": {
            "type": "object",
            "properties": {"city": {"type": "string"}},
            "required": ["city"],
        },
    },
}

ASSISTANT_MSG = {
    "role": "assistant",
    "content": None,
    "tool_calls": [
        {
            "id": "call_1",
            "type": "function",
            "function": {"name": "get_weather", "arguments": '{"city": "Hangzhou"}'},
        }
    ],
}

BODY_BASE = {
    "model": "deepseek-ai/DeepSeek-V4-Flash-0731",
    "max_tokens": 1,
    "temperature": 0,
    "tools": [TOOL_DEF],
    "messages": [
        {"role": "system", "content": "You are a helpful assistant. Answer briefly."},
        {"role": "user", "content": "What is the weather in Hangzhou? Use the tool."},
        dict(ASSISTANT_MSG),
        {"role": "user", "content": "Now summarize the result in one sentence."},
    ],
}

VARIANTS = {
    "a_absent": None,
    "b_empty": "",
    "c_space": " ",
}

bodies = {}
for name, rc in VARIANTS.items():
    body = json.loads(json.dumps(BODY_BASE))
    if rc is not None:
        body["messages"][2]["reasoning_content"] = rc
    bodies[name] = ChatCompletionRequest.model_validate(body)

# Find the adapter's request->TaskParams conversion function.
from exo.api.adapters import chat_completions as cc  # noqa: E402

convert = None
for fn_name, fn in inspect.getmembers(cc, inspect.isfunction):
    params = list(inspect.signature(fn).parameters)
    if params and params[0] == "request" and "request" in inspect.signature(fn).parameters:
        convert = fn
        print(f"candidate converter: {fn_name}{inspect.signature(fn)}", file=sys.stderr)
if convert is None:
    raise SystemExit("no converter found; adapter function list needed")

import asyncio

task_params_by_variant = {
    name: asyncio.run(convert(req)) for name, req in bodies.items()
}

import transformers  # noqa: E402

TOK_PATH = (
    "/Users/adam.durham/.cache/huggingface/hub/"
    "models--deepseek-ai--DeepSeek-V4-Flash-0731/snapshots/"
    "7872f01b1d1fe23eabc4c98b48bffcef5a386062"
)
tok = transformers.AutoTokenizer.from_pretrained(TOK_PATH, trust_remote_code=True)

from exo.worker.engines.mlx.utils_mlx import apply_chat_template as exo_apply  # noqa: E402


class FakeTokenizer:
    """exo's v4 branch never calls tokenizer.apply_chat_template; guard it."""

    chat_template = None

    def apply_chat_template(self, *a, **kw):
        raise RuntimeError("v4 branch must not call tokenizer.apply_chat_template")


results = {}
for name, tp in task_params_by_variant.items():
    prompt = exo_apply(FakeTokenizer(), tp)  # type: ignore[arg-type]
    token_ids = tok.encode(prompt, add_special_tokens=False)
    results[name] = {"prompt": prompt, "token_ids": token_ids, "n_tokens": len(token_ids)}

diff = {}
for pair in [("a_absent", "b_empty"), ("a_absent", "c_space"), ("b_empty", "c_space")]:
    x, y = pair
    tx, ty = results[x]["token_ids"], results[y]["token_ids"]
    n = max(len(tx), len(ty))
    differing = []
    for i in range(n):
        vx = tx[i] if i < len(tx) else None
        vy = ty[i] if i < len(ty) else None
        if vx != vy:
            differing.append(
                {
                    "index": i,
                    "x": vx,
                    "y": vy,
                    "x_tok": tok.decode([vx]) if vx is not None else None,
                    "y_tok": tok.decode([vy]) if vy is not None else None,
                }
            )
    diff[f"{x}_vs_{y}"] = differing

out = {
    "encoder_path": (
        "ChatCompletionRequest.model_validate -> chat_completions adapter -> "
        "TextGenerationTaskParams(chat_template_messages) -> utils_mlx.apply_chat_template -> "
        "deepseek_v4_encoding.encode_messages (model contains 'deepseek-v4'); "
        "tools reach encoder via system msg['tools'] (utils_mlx.py:1493-1501) which "
        "forces drop_thinking=False (deepseek_v4_encoding.py:639-641); "
        "tokenizer: deepseek-ai/DeepSeek-V4-Flash-0731 snapshot 7872f01b"
    ),
    "request_bodies": {name: req.model_dump(exclude_none=True) for name, req in bodies.items()},
    "results": results,
    "diffs": diff,
}

with open(
    "/Users/adam.durham/repos/exo/tmp/prefill-round3-20260902/artifacts/a1_offline_render.json", "w"
) as f:
    json.dump(out, f, indent=2, ensure_ascii=False)

with open(
    "/Users/adam.durham/repos/exo/tmp/prefill-round3-20260902/artifacts/a1_offline_diff.md", "w"
) as f:
    f.write("# a1 offline render diff (exo DSv4 encoder, exact live path)\n\n")
    for name in VARIANTS:
        r = results[name]
        f.write(f"## variant {name} — n_tokens={r['n_tokens']}\n\n```\n{r['prompt']}\n```\n\n")
    for pname, p in diff.items():
        f.write(f"## diff {pname}: {len(p)} differing positions\n\n")
        if not p:
            f.write("Identical.\n\n")
        for d in p:
            f.write(f"- idx {d['index']}: {d['x']}({d['x_tok']!r}) -> {d['y']}({d['y_tok']!r})\n")
        f.write("\n")

print("== prompt lengths ==")
for name in VARIANTS:
    print(f"{name}: {len(results[name]['prompt'])} chars, {results[name]['n_tokens']} tokens")
print("== diffs ==")
for pname, p in diff.items():
    print(f"{pname}: {len(p)} differing positions")
    for d in p[:12]:
        print(f"  idx {d['index']}: {d['x']} {d['x_tok']!r} -> {d['y']} {d['y_tok']!r}")
print()
print("== variant a prompt ==")
print(results["a_absent"]["prompt"])
print()
print("== variant c prompt ==")
print(results["c_space"]["prompt"])