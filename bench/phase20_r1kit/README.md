# phase20_r1kit — fixed-replay A/B throughput driver (recovered)

**Status:** build-only artifact. **DO NOT RUN against the cluster** unless a boot is
declared and a GO lands. This branch is a durable home for the driver that produced the
Phase-20 Q1 **control** JSONs (`control_benign.json`, `control_agentic.json`); it is not
merged into any round branch.

## Provenance

The driver was authored by the Phase-5 R1-kit subagent as
`~/.hermes/cache/scratch/p5/r1kit/r1_driver.py` (a thin extension of the ship-day
`~/.hermes/cache/scratch/p3b/p3b_driver.py`). The prior round's child ran it with
`--salt q1eval` and label `q1_ctl_benign` / `q1_ctl_agentic` to produce the Q1 control
anchors. It was **found intact on disk** and copied here **verbatim**:

| file | role | source |
|---|---|---|
| `r1_driver.py` | the recovered driver (fixed-salt + persisted own-request registry) | `scratch/p5/r1kit/r1_driver.py` (byte-identical) |
| `p3b_driver.py` | base driver: `stream_once` (SSE parser), `RM`/`AM` re-export | `scratch/p3b/p3b_driver.py` |
| `phase19_round_measure.py` | benign prompt builder (`build_prompt`), `derive()` metrics, `TASKS`, `MODEL` | `/private/tmp/next16-instr/bench/` |
| `phase19_agentic_measure.py` | agentic prompt builder (`build_agentic_prompt`), reads real session from `state.db` | `/private/tmp/next16-instr/bench/` |
| `phase20_guard.py` | `wait_for_idle` + `ChunkGuard` (idle gate, own-request registry, abort/wall-cap) | `bench/phase20_guard.py` |

The five files are vendored here so the kit is self-contained: `r1_driver.py` resolves its
dependencies either from the original absolute paths (when present) **or** from this
directory (its own `sys.path[0]` when run as a script, or via `PYTHONPATH`).

The vendored copies are byte-identical to their sources; `r1_driver.py` itself is
**unmodified** — the only change in the kit is *where it lives*.

## Salt parameterisation

`r1_driver.py` already supports an arbitrary base salt via `--salt` (default `r1fix-a`).
Per rep `n` the prompt salt is `f"{base}-{n}"`, so the same bytes replay on both arms and
across reps within an arm. The Q1 **new round** (ROUND-Q1B-STALL §0, boot #1) uses
**`--salt q1b`** → salts `q1b-0`, `q1b-1`, … and `summary.salt_base == "q1b"`.

## Invocation (fixed-replay A/B, one arm per invocation)

```bash
PY=/Users/adam.durham/repos/exo/.venv/bin/python
KIT=bench/phase20_r1kit/r1_driver.py
O=<out-dir>            # e.g. docs/benchmarks/phase20-throughput/raw/pricing/q1b/round
SALT=q1b

# control arm (benign 20K + agentic 91K), 4 reps/arm, same content both arms
$PY $KIT --arm benign  --total-reps 4 --reps-per-chunk 4 --depth 20000 \
         --max-tokens 800 --gamma 3 --salt $SALT \
         --label q1b_ctl_benign  --out $O/control_benign.json
$PY $KIT --arm agentic --total-reps 4 --reps-per-chunk 2 \
         --max-tokens 800 --gamma 3 --salt $SALT \
         --label q1b_ctl_agentic --out $O/control_agentic.json
```

The driver POSTs a **fixed** replay to the exo OpenAI-compatible endpoint
(`$PHASE19_API`, default `http://192.168.86.48:52415/v1/chat/completions`) with
`model=dealignai/DeepSeek-V4.1-Flash-UNCENSORED-EXL3-2.9bpw`, `temperature=0`,
`stream=true`; benign prompts are ~20 K tokens (`depth=20000`), agentic prompts are the
91 K real-session replay. Between chunks it waits for the cluster to be idle via
`phase20_guard.wait_for_idle` (own-request registry subtracts the bench's own traffic).

## Output schema (`<out>.json` = `{"summary": {...}, "recs": [...]}`)

`summary` keys: `label, arm, salt_base, round_prof, reps, depth, max_tokens,
decode_tps_median, decode_tps_all, ms_per_round_median, ms_per_round_all,
mean_accepted_median, aborted, abort_reason`.

Per-rep **derived metrics** (from `phase19_round_measure.derive`, gamma=3):

```
rounds        = d_cycles / gamma
mean_accepted = gamma * d_accepted / d_cycles
ms_per_round  = decode_s * 1000 / rounds
rounds_est    = completion_tokens / (1 + d_accepted/d_cycles)
gamma_implied = d_cycles / rounds_est
decode_tps    = (completion_tokens - 1) / decode_s
```

The **first rep of each arm** has `rounds=null` (warm-up: `stats` not read against a prior
baseline, `prev_cyc=None`).

## Static validation

`selftest.py` runs fully offline (no cluster, no POST): it imports the kit, parses the CLI,
checks the salt sequence, **re-derives every control rep** from the frozen
`control_*.json` raw fields and asserts the metrics match, recomputes the summary medians,
asserts the summary key set matches the frozen schema, and builds both prompt corpora to
confirm the meta shapes. See its console output.

```bash
cd <this worktree>
/Users/adam.durham/repos/exo/.venv/bin/python bench/phase20_r1kit/selftest.py
```

## Frozen control anchors (sanity target)

| arm | ms_per_round_median | decode_tps_median | mean_accepted_median |
|---|---:|---:|---:|
| benign 20K | 94.88 | 38.48 | 2.682 |
| agentic 91K | 101.07 | 30.962 | 2.1128 |

`selftest.py` reproduces all six from the raw control recs.
