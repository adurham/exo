# type: ignore
#!/usr/bin/env python3
"""Eagle soft-embedding step-1 P(top-1) microbench (Phase 14 Plan B.2).

Single-node measurement: does replacing the chained-MTP step-1 hard-argmax
embedding with a probability-weighted top-K mixture lift step-1 P(top-1)
(= mean(MTP step-1 argmax == main-model verify-argmax)) by at least the
+3pp gate?

Loads DSv4-Flash-8bit on a single GPU (no TP / pipeline collectives,
since step-1 acceptance is a single-node property). For each of N
fixed-context samples, runs MTP step 0 once, then MTP step 1 TWICE
(hard-embed then soft-embed) from the same post-step-0 state, and
compares both against the main-model's verify-argmax at the same
position.

Usage (must run on a node with enough memory to hold DSv4-Flash-8bit
unsharded — see start_cluster.sh for the cluster default model):

    EXO_DSV4_MTP=1 \\
      uv run python bench/mtp_eagle_microbench.py \\
        --model mlx-community/DeepSeek-V4-Flash-8bit \\
        --n-samples 200 \\
        --eagle-k 8

The script ignores ``EXO_DSV4_MTP_EAGLE_K`` from the env (it toggles the
predictor's ``eagle_k`` attribute explicitly between hard and soft
passes). Other ``EXO_DSV4_*`` envs (FUSED_MOE, COMPILE_FFN, etc.) are
forced off so each cycle's measurement is comparable. KV cache stays
at bf16 (EXO_KV_CACHE_BITS=0).
"""
from __future__ import annotations

import argparse
import os
import time
from statistics import mean
from typing import Any

# Set required envs BEFORE importing mlx_lm so the deepseek_v4 module
# constructs MTP heads at load time. Other envs are forced off to keep
# the path comparable and to avoid compile-graph reuse issues across
# the back-to-back step-1 calls.
os.environ.setdefault("EXO_DSV4_MTP", "1")
os.environ["EXO_DSV4_MTP_EAGLE_K"] = "0"  # we toggle on the predictor explicitly
os.environ.setdefault("EXO_KV_CACHE_BITS", "0")
os.environ.setdefault("EXO_DSV4_FUSED_MOE", "0")
os.environ.setdefault("EXO_DSV4_COMPILE_FFN", "0")
os.environ.setdefault("EXO_DSV4_COMPILE_LAYER", "0")
# Disable per-rank fence — single-node bench, no collectives.
os.environ.setdefault("EXO_DSV4_FENCE_EVERY_N_LAYERS", "0")

import mlx.core as mx  # noqa: E402
from mlx_lm import load  # noqa: E402

from exo.worker.engines.mlx.speculative.dsv4_mtp import DSv4MTPPredictor  # noqa: E402
from exo.worker.engines.mlx.speculative.mtp_module import (  # noqa: E402
    _compute_eagle_soft_emb,
)


# Diverse short / medium prompts. Repeat-with-truncation lifts the count
# to N_SAMPLES (default 200) while still varying the context per cycle.
# Mix domains so step-1 P(top-1) reflects general decode rather than a
# single style.
_BASE_PROMPTS: list[str] = [
    "The capital of France is Paris, and its most famous landmark is the Eiffel Tower, which was completed in 1889 for the World's Fair.",
    "Photosynthesis is the process by which green plants use sunlight to synthesize foods with the help of chlorophyll, water, and carbon dioxide.",
    "Albert Einstein developed the theory of general relativity, one of the two pillars of modern physics alongside quantum mechanics.",
    "The Pacific Ocean is the largest and deepest of the world's oceanic divisions, covering more than 60 million square miles.",
    "A binary search tree is a data structure where each node has at most two children, with the left child containing a smaller value than the parent.",
    "Mount Everest, located in the Himalayas on the border between Nepal and Tibet, is the highest mountain in the world above sea level.",
    "Shakespeare wrote thirty-seven plays and over one hundred fifty sonnets that continue to be performed and studied around the world.",
    "Python is a high-level interpreted programming language known for its readability and extensive standard library, widely used in scientific computing.",
    "The Roman Empire reached its greatest territorial extent under the reign of Emperor Trajan in the early second century of the common era.",
    "DNA, or deoxyribonucleic acid, is a molecule that carries genetic instructions used in the growth, development, and reproduction of all known living organisms.",
    "The Industrial Revolution began in Great Britain during the eighteenth century, transforming agrarian economies into industrial and urban ones.",
    "Quantum entanglement is a physical phenomenon that occurs when pairs of particles interact in ways such that their quantum states cannot be described independently.",
    "Beethoven composed nine symphonies, the ninth of which features a choral finale based on Schiller's Ode to Joy.",
    "The Amazon rainforest, often referred to as the lungs of the planet, spans across nine countries in South America and contains millions of species.",
    "Machine learning is a subset of artificial intelligence that focuses on the development of algorithms that can learn from and make predictions on data.",
    "The Great Wall of China was built over many centuries by different dynasties to protect Chinese states from nomadic incursions from the north.",
    "Climate change is a long-term shift in global or regional climate patterns, often attributed to increased levels of atmospheric carbon dioxide.",
    "The human brain contains approximately eighty-six billion neurons connected by trillions of synapses, forming the basis of conscious experience.",
    "Marie Curie was the first woman to win a Nobel Prize and the only person to win the prize in two different scientific fields, physics and chemistry.",
    "The French Revolution, which began in 1789, fundamentally transformed the political and social landscape of France and inspired movements across Europe.",
    "Black holes are regions of spacetime where gravity is so strong that nothing, not even light, can escape once it crosses the event horizon.",
    "The Silk Road was an ancient network of trade routes that connected the East and West, facilitating not only commerce but also cultural exchange.",
    "Linnaean taxonomy is the system of biological classification developed by Carl Linnaeus, organizing organisms into a hierarchy of kingdom, phylum, class, order, family, genus, and species.",
    "The first manned mission to land on the Moon was Apollo 11 in July 1969, with Neil Armstrong becoming the first human to set foot on its surface.",
    "Functional programming emphasizes the use of pure functions, immutable data, and the avoidance of side effects, with languages such as Haskell and Lisp.",
    "The Renaissance was a period of cultural, artistic, political, and economic rebirth following the Middle Ages, beginning in Italy in the late fourteenth century.",
    "Photons, the fundamental particles of light, exhibit both wave-like and particle-like properties, a duality central to quantum mechanics.",
    "The Sahara is the largest hot desert in the world, covering most of northern Africa with an area comparable to that of the United States.",
    "Linear algebra studies vectors, vector spaces, and linear transformations, providing the mathematical foundation for many engineering and computer-science applications.",
    "World War II, which lasted from 1939 to 1945, was the deadliest conflict in human history, involving more than thirty countries and resulting in tens of millions of casualties.",
    "Tectonic plates are massive sections of the Earth's lithosphere that slowly move and interact, causing earthquakes, volcanic activity, and mountain formation at their boundaries.",
    "The novel Pride and Prejudice, written by Jane Austen and published in 1813, remains one of the most popular works of English literature.",
    "Vaccines stimulate the body's immune system to recognize and combat specific pathogens, reducing the spread of infectious diseases across populations.",
    "The transistor, invented at Bell Labs in 1947, revolutionized electronics and laid the foundation for modern digital computing and telecommunications.",
    "Cryptography is the practice and study of techniques for secure communication in the presence of adversaries, encompassing both symmetric and asymmetric methods.",
    "Volcanic eruptions can dramatically alter landscapes, climate, and ecosystems, with major historical examples including the eruption of Mount Vesuvius in 79 AD that destroyed Pompeii.",
    "The Theory of Evolution by natural selection, proposed by Charles Darwin in his book On the Origin of Species, explains the diversity of life on Earth.",
    "Optical fibers transmit data as pulses of light through long strands of glass or plastic, enabling high-speed internet and long-distance telecommunications.",
    "The Mona Lisa, painted by Leonardo da Vinci in the early sixteenth century, is one of the most famous works of art in the world, displayed at the Louvre in Paris.",
    "Photolithography is a process used in semiconductor manufacturing to transfer geometric patterns from a photomask to a light-sensitive resist on a silicon substrate.",
]


def _build_samples(
    tokenizer: Any,
    base_prompts: list[str],
    n_samples: int,
    min_ctx: int,
    max_ctx: int,
) -> list[mx.array]:
    """Build N (1, L) int32 token arrays. We tokenise each base prompt and
    truncate to a sequence of context lengths spanning ``[min_ctx, max_ctx]``
    to produce diverse contexts while keeping the prompt count finite.
    """
    samples: list[mx.array] = []
    tokenised: list[list[int]] = []
    for p in base_prompts:
        ids = tokenizer.encode(p)
        if len(ids) >= min_ctx:
            tokenised.append(ids)
    if not tokenised:
        raise RuntimeError("no base prompt tokenised to >= min_ctx — widen the prompt list")

    # Spread context lengths uniformly in [min_ctx, min(max_ctx, len)] per
    # prompt; cycle through prompts until we hit n_samples.
    i = 0
    while len(samples) < n_samples:
        ids = tokenised[i % len(tokenised)]
        # 5 lengths per prompt: min_ctx, +25%, +50%, +75%, full (capped)
        upper = min(max_ctx, len(ids))
        if upper <= min_ctx:
            cuts = [upper]
        else:
            step = max(1, (upper - min_ctx) // 4)
            cuts = [min(min_ctx + k * step, upper) for k in range(5)]
        for cut in cuts:
            samples.append(mx.array([ids[:cut]], dtype=mx.int32))
            if len(samples) >= n_samples:
                break
        i += 1
    return samples


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument(
        "--model", default="mlx-community/DeepSeek-V4-Flash-8bit",
        help="HF repo id for the model (default: cluster's DSv4-Flash-8bit).",
    )
    ap.add_argument("--n-samples", type=int, default=200)
    ap.add_argument("--eagle-k", type=int, default=8, help="K for the soft-embedding top-K mixture")
    ap.add_argument("--min-ctx", type=int, default=24)
    ap.add_argument("--max-ctx", type=int, default=64)
    ap.add_argument("--warmup", type=int, default=3, help="warmup samples before timing")
    args = ap.parse_args()

    print(f"[eagle-bench] loading model: {args.model}", flush=True)
    t_load_0 = time.perf_counter()
    model, tokenizer = load(args.model)
    t_load_1 = time.perf_counter()
    print(f"[eagle-bench] model loaded in {t_load_1 - t_load_0:.1f}s", flush=True)

    inner = getattr(model, "model", None) or model.language_model.model
    if not hasattr(inner, "mtp") or not inner.mtp:
        raise RuntimeError(
            "model has no MTP module — make sure the checkpoint contains "
            "mtp.* weights and EXO_DSV4_MTP=1 is set BEFORE the load() call."
        )

    # Wire up pre_norm capture exactly like MTPBatchGenerator._setup_hidden_capture
    captured: dict[str, mx.array] = {}
    original_norm = inner.norm

    class _CapturingNorm:
        def __init__(self, orig: Any) -> None:
            self._orig = orig
            self.weight = orig.weight

        def __call__(self, x: mx.array) -> mx.array:
            captured["pre_norm"] = x
            return self._orig(x)

        def __getattr__(self, name: str) -> Any:
            return getattr(self._orig, name)

    inner.norm = _CapturingNorm(original_norm)  # type: ignore[assignment]

    mtp_pred = DSv4MTPPredictor(model, mtp_idx=0)

    samples = _build_samples(
        tokenizer, _BASE_PROMPTS, args.n_samples, args.min_ctx, args.max_ctx
    )
    print(
        f"[eagle-bench] built {len(samples)} samples "
        f"(ctx in [{args.min_ctx}, {args.max_ctx}])",
        flush=True,
    )

    hard_match = 0
    soft_match = 0
    timed_n = 0
    hard_walls_ms: list[float] = []
    soft_walls_ms: list[float] = []
    total_walls_ms: list[float] = []

    for s_idx, tok_arr in enumerate(samples):
        # Fresh caches each cycle — keeps measurements independent.
        main_cache = inner.make_cache()
        mtp_pred.reset_cache(batch_size=1)
        captured.clear()
        mtp_pred.set_eagle_soft_emb(None)

        # 1) Prefill
        _ = model(tok_arr, cache=main_cache)
        pre_norm_full = captured.get("pre_norm")
        if pre_norm_full is None:
            raise RuntimeError("pre_norm capture missed — wrapped norm not invoked")
        pre_norm_last = pre_norm_full[:, -1:, :]
        last_tok = tok_arr[:, -1:]
        mx.eval(pre_norm_last)

        # 2) MTP step 0 (hard embed — predictor.eagle_k semantics don't
        #    matter here since we call predict() directly, not draft_tokens).
        logits_0, h_1 = mtp_pred.predict(
            pre_norm_last, last_tok, return_hidden=True
        )
        tok_0 = mx.argmax(logits_0, axis=-1).reshape(1, 1)
        mx.eval(tok_0, logits_0, h_1)

        # MTP cache is now at offset = L + 1.
        post_step0_offset = mtp_pred._cache.offset

        # 3) Step 1 HARD — no soft emb installed.
        t_h0 = time.perf_counter()
        mtp_pred.set_eagle_soft_emb(None)
        logits_1_hard, _ = mtp_pred.predict(
            h_1, tok_0, return_hidden=True
        )
        tok_1_hard = mx.argmax(logits_1_hard, axis=-1).reshape(-1)
        mx.eval(tok_1_hard)
        t_h1 = time.perf_counter()

        # Trim MTP cache back to post-step-0 so SOFT runs from the same state.
        delta = mtp_pred._cache.offset - post_step0_offset
        if delta > 0:
            mtp_pred._cache.trim(delta)

        # 4) Step 1 SOFT — install soft_emb from step-0 logits.
        t_s0 = time.perf_counter()
        soft_emb = _compute_eagle_soft_emb(
            logits_0, mtp_pred.embed_tokens, args.eagle_k
        )
        mtp_pred.set_eagle_soft_emb(soft_emb)
        try:
            logits_1_soft, _ = mtp_pred.predict(
                h_1, tok_0, return_hidden=True
            )
        finally:
            mtp_pred.set_eagle_soft_emb(None)
        tok_1_soft = mx.argmax(logits_1_soft, axis=-1).reshape(-1)
        mx.eval(tok_1_soft)
        t_s1 = time.perf_counter()

        # Trim MTP cache back again so the next sample starts fresh
        # (technically redundant since we reset_cache next iter, but keeps
        # the cache offset bookkeeping straight if anyone reuses _cache).
        delta = mtp_pred._cache.offset - post_step0_offset
        if delta > 0:
            mtp_pred._cache.trim(delta)

        # 5) Verify target: main model with tok_0 appended.
        captured.clear()
        verify_logits = model(tok_0, cache=main_cache)
        verify_target = mx.argmax(verify_logits[:, -1, :], axis=-1).reshape(-1)
        mx.eval(verify_target)

        # 6) Compare
        v = int(verify_target.item())
        h = int(tok_1_hard.item())
        s = int(tok_1_soft.item())

        if s_idx >= args.warmup:
            if h == v:
                hard_match += 1
            if s == v:
                soft_match += 1
            timed_n += 1
            hard_walls_ms.append((t_h1 - t_h0) * 1000)
            soft_walls_ms.append((t_s1 - t_s0) * 1000)
            total_walls_ms.append((t_s1 - t_h0) * 1000)

        if (s_idx + 1) % 20 == 0:
            running_hard = (hard_match / max(timed_n, 1)) if timed_n else 0.0
            running_soft = (soft_match / max(timed_n, 1)) if timed_n else 0.0
            print(
                f"[eagle-bench] {s_idx + 1}/{len(samples)} timed={timed_n} "
                f"hard={running_hard:.4f} soft={running_soft:.4f} "
                f"delta={(running_soft - running_hard) * 100:+.2f}pp",
                flush=True,
            )

    if timed_n == 0:
        raise RuntimeError("zero timed samples — increase --n-samples above --warmup")

    p_hard = hard_match / timed_n
    p_soft = soft_match / timed_n
    delta_pp = (p_soft - p_hard) * 100
    hard_mean_ms = mean(hard_walls_ms)
    soft_mean_ms = mean(soft_walls_ms)
    overhead_pct = (soft_mean_ms - hard_mean_ms) / hard_mean_ms * 100

    print("")
    print("=" * 72)
    print(f"[eagle-bench] step-1 P(top-1) over {timed_n} samples")
    print(f"  baseline (K=0):  {p_hard:.4f}")
    print(f"  eagle (K={args.eagle_k}):     {p_soft:.4f}")
    print(f"  delta:           {delta_pp:+.2f}pp")
    print("")
    print(f"  hard step-1 wall: {hard_mean_ms:.2f} ms")
    print(f"  soft step-1 wall: {soft_mean_ms:.2f} ms ({overhead_pct:+.1f}%)")
    print("")
    gate_pp = 3.0
    if delta_pp >= gate_pp:
        print(f"  GATE PASS (>= +{gate_pp:.1f}pp)")
    else:
        print(f"  GATE FAIL  (< +{gate_pp:.1f}pp)")
    print("=" * 72)


if __name__ == "__main__":
    main()
