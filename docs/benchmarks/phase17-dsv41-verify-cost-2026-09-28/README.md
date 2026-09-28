# Phase 17 -- where the verify row cost goes; spec decode 22.7 -> 23.5 tok/s

Date: 2026-09-28. Fable consult, then executed. Production stopped cleanly for
the two-node run and restored (fresh pids, env verified, real completion OK).

## Bottom line

Mean spec decode at gamma=3 over 3 prompts: **23.5 tok/s** (was 22.7); plain
16.9 (was 16.5). Per prompt: 22.4 / **26.5** / 21.6. Still short of 25 on
average. The extra cost of checking more tokens per pass is mostly the MoE
experts reading more distinct experts -- real work, not overhead -- so the
remaining gains are smaller than the consult expected.

## 1. Verify row cost, measured (8-layer harness, unfenced stubs, ms/step)

| removed | R=1 | R=4 | its share of the R1->R4 increase (9.3 ms) |
|---|---:|---:|---:|
| nothing | 13.44 | 22.78 | -- |
| routed experts | 11.17 | 15.95 | **4.6 ms** |
| all dense EXL3 projections | 6.75 | 12.49 | **3.6 ms** |
| head | 11.33 | 19.64 | 1.0 ms |
| sparse attention | 12.20 | 21.21 | 0.3 ms |
| indexer, fake-quant, hc, rope, rms, gate, engram | | | <= 0.3 ms each |

The consult's guess (attention/indexer overhead, ~5 ms/row addressable) is
NOT what the measurement shows. Experts: a 4-row window touches 59-87% unique
experts per layer (p30 trace), so R=4 reads ~3x the expert bytes of R=1 --
that is bandwidth, not waste. Dense EXL3 projections: the R<=16 small-batch
GEMM costs more per extra row than expected; that is the one real lever left
here (kernel work).

## 2. Draft round 13.1 -> 11.1 ms

Ablation (width 3): lm head 3.1 ms, attention 2.9, markov head 1.9, MoE 2.2.
Fix: the draft now uses the vocab-sharded head plus a vocab-sharded markov
head, and a new exact cross-rank argmax (each rank sends its (max, index)
pair; ties go to the lowest index, same as argmax over the full row). The
body uses the same argmax path, so full logits are never gathered.

## 3. Compiled rms / rope / fake-quant

Bit-identical to uncompiled (logits exact over the harness run), -0.45 ms at
8 layers. Now default (`DSV41_COMPILE_OPS=1`).

## 4. Output quality (the consult's flag)

All three prompts' spec text read (gamma=3). All fluent and on-task:
- sky: identical to greedy for 77 tokens, then an equally correct wording.
- ISO-8601 parser: diverges at token 41 -- greedy writes a hand-rolled
  function, spec opens with a regex-based one. Both valid starts.
- sourdough tips: diverges at token 14 with different but sensible tips.
Divergence is the known chunk-verify property (not bit-identical to greedy),
same tradeoff production DSv4 takes at short context. No garbling, no
cross-script fragments seen.

## Numbers (run 3)

plain 59.1-59.2 ms; draft 11.1 ms; verify R1..R6 = 58.5 74.9 87.9 97.7 111.7
120.1 ms; gamma 2/3/4 mean = 23.3 / 23.5 / 21.2 tok/s.

## Next

1. EXL3 small-batch dense GEMM at R=2..6 (3.6 ms of the 9.3 ms marginal on
   8 layers) -- kernel work.
2. Adaptive gamma per round from draft confidence (prompt 2 prefers 2-3,
   prompt 1 prefers 3-4).
3. Draft attention (2.9 ms) and markov loop (1.9 ms).
One transient: a relaunch right after shutdown hit a JACCL "Cannot allocate
memory" queue-pair error on m4-1; an immediate retry worked.

## Artifacts

`raw/p56-run3.log`, `scripts/p48_bench.py`, `scripts/p62_draft_ablate.py`,
`scripts/p61_overlap.py`, `scripts/p56_spec.py`.
