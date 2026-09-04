# Round 5 — Operational Harness Map (READ-ONLY reconnaissance)

**Generated:** 2026-09-03, read-only recon only. No relaunch, no benchmark run, no
git push performed by this dispatch. HEAD `ccc692ff3`, branch `main`.

---

## 1) Relaunch with a gamma override

### Where `EXO_SPECULATIVE_GAMMA` lives in `start_cluster.sh`

```
176: : "${EXO_SPECULATIVE_GAMMA:=3}"
```
Surrounding block (lines ~174-183):
```bash
: "${MLX_SDPA_BLOCKS:=}"
: "${EXO_SPECULATIVE_GAMMA:=3}"
# Per-model gamma override for the Qwen3.5-style MTP path (Qwen3.6). Its
# dedicated head is trained with block_size=3, so it sustains a deeper draft
# chain than DSv4's depth-1 head — default γ=3, independent of the DSv4
# EXO_SPECULATIVE_GAMMA above.
# NOTE (2026-08-27): DSv4 default raised 2→3 to match the promoted DSpark MTP
# production baseline (24-run verdict @100K was measured at γ=3; see
# docs/dspark-mtp-production-baseline-2026-08-27.md).
: "${EXO_QWEN_SPECULATIVE_GAMMA:=3}"
```
This is bash parameter-expansion default assignment (`:=`) — **it ONLY sets the
var if unset/empty in the script's own environment**. It does NOT hardcode
gamma. Consumption/pass-through to the runner spawn line:
```
1721:  EXO_ENV="$EXO_ENV EXO_SPECULATIVE_GAMMA=$EXO_SPECULATIVE_GAMMA"
```
`EXO_ENV` is later baked verbatim into the remote `ssh ... screen -dmS exorun
zsh -l -c '... $EXO_ENV ... .venv/bin/python -m exo -v ...'` launch line (see
§2 for the exact spawn form, confirmed live at line 2851-ish region).

### EXACT invocation to relaunch with gamma=N

**Prefixing the env var to `./start_cluster.sh` DOES override it** (standard
shell `VAR=val cmd` semantics beat the script's own `:=` default):
```bash
cd /Users/adam.durham/repos/exo
EXO_SPECULATIVE_GAMMA=4 ./start_cluster.sh
```
No script edit needed for this lever — it is a genuine env-var pass-through,
not a hardcoded value (unlike some other knobs flagged elsewhere in this
campaign's I2-C2-TAX-AUDIT.md).

### Project's normal invocation form (from real shell history + prior rounds)

Zsh history (`~/.zsh_history`) shows exactly two real invocations on this
machine:
```
./start_cluster.sh
EXO_DSV4_MTP_PROFILE=50 EXO_DSV4_RB_PROFILE=1 ./start_cluster.sh
```
i.e. the project's convention is a **bare `./start_cluster.sh`** for normal
relaunches, and **`VAR=val ./start_cluster.sh`** (space-separated, no `export`)
for one-off diagnostic/tuning overrides — exactly the form to use for gamma:
```bash
EXO_SPECULATIVE_GAMMA=3 ./start_cluster.sh   # etc. for 3,4,2,5,3
```
Round 1's `I1-PATCH-NOTES.md:230` independently documents the same pattern for
a different var: `cd ~/repos/exo && EXO_DSV4_COLL_PROFILE=20 ./start_cluster.sh`.

### Relaunch wall-time — NOT STATED in any prior round report

**Searched:** `grep -rn "relaunch.*took|~[0-9]* min|boot took|READY (2/2)" tmp/perf-campaign-2/round*/REPORT.md`
— zero matches. **DOES NOT EXIST as a documented number in this campaign's
round1-4 reports.** Round 4's REPORT.md mentions 3 boots occurred plus a
"post-restore decode smoke test at 2K ctx" but never quotes a wall-clock
duration for the relaunch step itself. Do not assume a number for the 3-hour
time-box; budget conservatively (a full `start_cluster.sh` run rebuilds the
venv from clean submodules per the `exo-cluster-deployment` skill's "Cluster
Restart" section — historically these have run several minutes, but no exact
figure is on record for THIS campaign).

### Route-clear sudo gate — VERIFIED FIXED on both nodes right now

Live `sudo -n -l` output, both nodes (2026-09-03, this session):

**macstudio-m4-1:**
```
User adam.durham may run the following commands on Adams-Mac-Studio-M4-1:
    (ALL) ALL
    (root) NOPASSWD: /usr/sbin/sysctl iogpu.wired_limit_mb\=*
    (root) NOPASSWD: /sbin/route delete -net *
    (root) NOPASSWD: /usr/bin/fdesetup authrestart*
    (ALL) NOPASSWD: /usr/bin/ktrace
    (ALL) NOPASSWD: /usr/bin/powermetrics
```

**macstudio-m4-2:**
```
User adam.durham may run the following commands on Adams-Mac-Studio-M4-2:
    (ALL) ALL
    (root) NOPASSWD: /usr/sbin/sysctl iogpu.wired_limit_mb\=*
    (root) NOPASSWD: /sbin/route delete -net *
    (root) NOPASSWD: /usr/bin/fdesetup authrestart*
    (ALL) NOPASSWD: /usr/bin/ktrace
    (ALL) NOPASSWD: /usr/bin/powermetrics
```

**Both nodes have `NOPASSWD: /sbin/route delete -net *`.** The known failure
mode (script running `sudo route delete -net` over non-interactive ssh WITHOUT
`-n`, hanging silently at "Testing direct-link connectivity...") is CURRENTLY
COVERED by this scoped sudoers wildcard rule on both nodes — this matches the
fix documented in the `exo-cluster-deployment` skill's
`references/route-clear-sudo-gate-2026-08-31.md` (the incident writeup for
this exact hang). **No blocking risk from this specific gate today**, but
re-verify with a fresh `ssh <node> "sudo -n -l"` immediately before each of
the 5 relaunches in the sweep — sudoers files can be edited independently of
this repo.

---

## 2) Verify the env is actually live

Exact `ps eww` pattern used in rounds 1-4 (quoted from
`tmp/perf-campaign-2/round1/I1-PATCH-NOTES.md:219`):
```bash
ssh $N "ps eww \$(pgrep -f 'python -m exo' | head -1) | tr ' ' '\n' | grep EXO_DSV4_COLL_PROFILE"
```
Generalized for gamma, run on BOTH nodes after each relaunch:
```bash
for N in macstudio-m4-1 macstudio-m4-2; do
  echo "=== $N ==="
  ssh "$N" "ps eww \$(pgrep -f '.venv/bin/python -m exo -v' | tail -1) | tr ' ' '\n' | grep -E '^EXO_SPECULATIVE_GAMMA='"
done
```
Note: `pgrep -f 'python -m exo'` can match multiple lines (screen wrapper,
login shell, zsh -l, and the real `.venv/bin/python -m exo -v` child) — round1
used `| head -1`, but the REAL runner env is the LAST matching PID in the
process tree (the actual python process), confirmed live this session as
`.venv/bin/python -m exo -v` (PID 79266 on m4-1, 86069 on m4-2) — **prefer
`grep '.venv/bin/python -m exo -v' | tail -1` over `| head -1`** to avoid
reading the wrapper shell's copy of the command line instead of the real
child's resolved env (VERIFICATION.md:219 makes the same distinction: "runner
= the multiprocessing-fork child, not the -m exo parent").

---

## 3) 89K-depth measurement harness

### The actual generator: `bench/quality_probe_dsv4.py`
- Round 4's needle driver imports it (`tmp/perf-campaign-2/round3/r3_needle_capture.py:19-22`
  does `importlib.util.spec_from_file_location("qp", ".../bench/quality_probe_dsv4.py")`
  then calls `qp.build_prompt(TARGET)`).
- Round 4's own PLAN.md explicitly cites it: `tmp/perf-campaign-2/round4/PLAN.md:10-11`
  — "Workload: real 89K-depth needle request (bench/quality_probe_dsv4.py
  build_prompt at ~89K target) + decode_probe runs."
- Standalone invocation form (module has its own `main`/argparse — run directly
  against the live API, `httpx`-based, async, supports `--concurrency`):
  ```bash
  cd /Users/adam.durham/repos/exo
  .venv/bin/python bench/quality_probe_dsv4.py <target_tokens=89000> [--concurrency N] [--out path.json]
  ```
  Prints/writes a JSON blob per the schema seen in
  `tmp/perf-campaign-2/round4/results/needle_89k_boot2.json` (schema_version 2):
  top-level `iters[]` each with `iter_wall_s`, `aggregate_decode_tps`,
  `all_needles_found`, `any_special_tokens_leaked`; per-stream
  `ttft_s`, `total_s`, `prompt_tokens`, `generation_tokens`,
  `prefill_tps_apparent`, `decode_tps_apparent`, `needle_found`, `quality.*`.

### A lighter-weight alternative: `bench/ab_probe_tier1.py`
- Single-request A/B probe, prints one JSON result (`prefill_s`, `prefill_tps`,
  `decode_s`, `decode_tps`, `needle_hit`, `bos_spam`, `output_head/tail`) —
  computed CLIENT-SIDE from `stream_options.include_usage` (usage arrives on
  the terminal SSE chunk) and wall-clock `t_first`/`t_end` around the stream,
  NOT from the server log. This is the fixed-prompt battery referenced in Q5(i).
  Invocation:
  ```bash
  .venv/bin/python bench/ab_probe_tier1.py 89000 --max-tokens <N> --tag <label> [--out path.json]
  ```
  Default `API = "http://192.168.86.201:52415"`, hardcoded in the script
  (edit or wrap if the API host differs — it should match `macstudio-m4-1`'s
  IP per §6).

### Candidates that do NOT exist / are not the 89K harness
- `scripts/dspark_context_sweep.py` — **DOES NOT EXIST.** `scripts/` contains
  only `fetch_kv_heads.py`, `download_model_to_cluster.py`,
  `context_stress_sweep.sh`, `prefill_overlap_ab.py`, `convert_dsv4_mtp.sh`.
  Search run: `find scripts -maxdepth 2`.
- `r3_needle_capture.py` (round 3) exists at
  `tmp/perf-campaign-2/round3/r3_needle_capture.py` but targets 200K by
  default (`TARGET = 200000` unless overridden by argv[4]) and is a thin
  concurrent-stream wrapper around `quality_probe_dsv4.build_prompt` — usable
  for 89K by passing `89000` as the 4th positional arg, but the round-4 result
  files show round 4 used `quality_probe_dsv4.py` directly/via its own driver,
  not this script.

### The methodologically correct decode-vs-prefill split

The task's proposed method (SSH-grep the server's own prefill-complete log
line as the boundary, then `decode_tok_s = completion_tokens / (http_response_time − prefill_complete_time)`)
matches how this repo's own log line is framed. **Confirmed line exists:**
```
src/exo/worker/engines/mlx/generator/generate.py:978
    logger.debug(
        f"Prefill complete: {num_tokens} tokens in {elapsed:.2f}s "
        f"({tokens_per_sec:.1f} tok/s)"
    )
```
This is a `logger.debug` call — the runner's default `LOG_LEVEL` on the live
config is `INFO` (confirmed in the live `ps eww` dump, §6), so **this line will
NOT appear in `~/exo.log` at the current log level.** To use this method for
the sweep, either (a) grep the log anyway in case debug logging is enabled
elsewhere, or (b) rely on the harness's own `iter_wall_s` /
`ttft_s` / `prefill_tps_apparent` fields (quality_probe_dsv4.py's own
computed split) which do NOT depend on this debug line. **Flag to the sweep
operator: verify `LOG_LEVEL=DEBUG` is set (or temporarily override it) before
depending on grepping this exact line — at the current shipped `LOG_LEVEL=INFO`
it will not be emitted.**

Grep to extract it once emitted:
```bash
ssh macstudio-m4-1 "grep 'Prefill complete' ~/exo.log | tail -5"
```

---

## 4) MTP-PROF cycle-time brackets

**Location note:** the task description says "mlx-lm submodule's dsv4_mtp.py"
— this is NOT where it lives. `dsv4_mtp.py` is at
`src/exo/worker/engines/mlx/speculative/dsv4_mtp.py` (part of the exo
superproject, not the `mlx-lm` git submodule). Confirmed via
`find . -name "dsv4_mtp.py"` → single hit at that path.

### (a) Enabling env var — CONFIRMED: value is a CADENCE, not a boolean

```
src/exo/worker/engines/mlx/speculative/dsv4_mtp.py:244
    _PROFILE_INTERVAL = int(os.environ.get("EXO_DSV4_MTP_PROFILE", "0"))
```
Surrounding comment (lines 240-243):
```
# Per-cycle phase timing. When EXO_DSV4_MTP_PROFILE > 0, brackets the
# draft / verify / accept phases with mx.eval + perf_counter, summarising
# every N cycles. Inserts evals at phase boundaries which serialises
# pipelining — measurements are upper bounds on real production walls.
```
`_phase_timer.end_cycle()` (line ~816): `if _PROFILE_INTERVAL > 0 and self.cycles % _PROFILE_INTERVAL == 0: self.dump()`.
**Confirmed: the value IS a dump cadence (dump every N cycles), not an
on/off boolean.** Setting `EXO_DSV4_MTP_PROFILE=1` dumps EVERY cycle — this
inserts `mx.eval`/sync phase-boundary calls (per the comment above, "serialises
pipelining... measurements are upper bounds") on every single cycle, which
will pollute per-cycle wall time for the whole run. **`EXO_DSV4_MTP_PROFILE=50`
is the value actually used historically** — confirmed in `~/.zsh_history`:
`EXO_DSV4_MTP_PROFILE=50 EXO_DSV4_RB_PROFILE=1 ./start_cluster.sh`. Use 50 for
the sweep's diagnostic boots, never 1, unless a single-cycle dump is
specifically wanted for a short debug session.

Companion var `EXO_DSV4_RB_PROFILE=1` (rollback sub-phase attribution,
line 254 `_RB_PROFILE = os.environ.get("EXO_DSV4_RB_PROFILE", "0") == "1"`)
— this one genuinely IS a boolean and REQUIRES `EXO_DSV4_MTP_PROFILE>0` to have
any effect (per its own comment block at lines 244-253).

### (b) Exact phase series names emitted

From all `prof.record(...)` call sites (`grep -n '\.record(' dsv4_mtp.py`):
`draft`, `verify`, `accept`, `commit`, `rollback`, `total` (the "known" set
consumed by `_PhaseTimer.dump()`'s `known = ("draft", "verify", "accept",
"commit", "rollback", "total")` tuple, line ~817), plus extras emitted only
under `EXO_DSV4_RB_PROFILE=1`: `rb_snap`, `rb_gate`, `rb_drain`, `rb_ring`,
`rb_pool`, `rb_pool_restores`, `rb_commitfwd`, `rb_tail`.

### (c) Per-series unit (TIME vs COUNT) — verified against `_ProfUnit` metadata and `record()` call sites

Fix commit confirmed real and on `main`:
```
commit bbb0e93418f822257e5e0b045f56b92637ed8a36
fix(profiler,telemetry): per-series unit metadata, runner comm filter, stale comments
```
Doc commit `a30d13af3` references it as "SIDE-FIXES entry — profiler units,
telemetry filter, stale comments (bbb0e934)". Commit message states: "the
integer counter rb_pool_restores rendered as 'mean=18.91ms' — a phantom 25%-of-cycle
hotspot". **`_ProfUnit = Literal["ms", "count"]` at dsv4_mtp.py:782.**

All 23 `record()` call sites pass no explicit `unit` (defaulting to `"ms"`)
EXCEPT one:
```
dsv4_mtp.py:5024-5026
    prof.record(
        "rb_pool_restores", float(_rb_pool_restores), unit="count"
    )
```
**`rb_pool_restores` is the ONLY count series; every other series (`draft`,
`verify`, `accept`, `commit`, `rollback`, `total`, `rb_snap`, `rb_gate`,
`rb_drain`, `rb_ring`, `rb_pool`, `rb_commitfwd`, `rb_tail`) is a genuine
time-in-ms series.** The formatter (`_PhaseTimer.dump()`, lines ~817-841)
branches on `unit` per-series and prints the correct suffix (`ms` vs bare
number). This confirms the fix — a fresh MTP-PROF dump today will label
`rb_pool_restores` without the `ms` suffix, not manufacture the phantom
hotspot the commit fixed.

### (d) Grep to extract cycle-time bracket + phase ratios from a runner log

```bash
ssh macstudio-m4-1 "grep '\[MTP-PROF\]' ~/exo.log | tail -100"
```
Sample line shapes emitted by `dump()`:
```
[MTP-PROF] cycles=<N> B=<b>:<count>,...
[MTP-PROF]   B=<b> draft      mean=  X.XXms min=  X.XXms max=  X.XXms n=<n>
[MTP-PROF]   B=<b> rb_pool_restores mean=  X.XX min=  X.XX max=  X.XX n=<n>   # NOTE: no "ms" suffix — count series
```
To compute phase ratios (e.g. draft/total, verify/total) from a raw dump,
pull the `mean=` values per phase per `B=` bucket and divide — no script for
this exists in the repo currently (grep found none); would need to be
written ad hoc if the sweep wants automated ratio extraction.

---

## 5) Quality gate inventory

### (i) `bench/ab_probe_tier1.py` — fixed-prompt battery

```bash
.venv/bin/python bench/ab_probe_tier1.py [target_tokens] [--max-tokens N] [--tag LABEL] [--out path]
```
Default `API = "http://192.168.86.201:52415"`, `MODEL = "mlx-community/DeepSeek-V4-Flash"`
(hardcoded constants at top of file — confirm these match the live model id
from §6 before running, since the live `/v1/models` id may differ). Sends ONE
request, prints/writes a single JSON result (not a 7/7 battery internally —
"7/7" implies running it 7× with 7 different prompt configs / labels; **the
script itself has no built-in 7-prompt loop** — confirmed by reading the full
file: it's a single-shot probe with `needle_hit` (bool) and `bos_spam` (bool)
fields). **A PASS for one invocation = `needle_hit: true` AND `bos_spam:
false`** in the printed/saved JSON. The "7/7" framing implies the sweep
operator runs it 7 times (e.g. across the 7-prompt degen set used elsewhere in
this campaign, see (iii) below) and tallies PASS/FAIL externally — no
orchestration script for that loop exists in the repo; would need to be
written for the sweep.

### (ii) Needle exact-match at 89K — round 4's harness

Found: `tmp/perf-campaign-2/round4/results/needle_89k_boot2.json` (and
`_boot1.json`, `_identity.json`) — generated by `bench/quality_probe_dsv4.py`
(see §3). **PASS shape** (from the real boot2 result, schema_version 2):
```json
"summary": {
  "all_iters_ok": true,
  "iters_with_special_token_leak": 0,
  "iters_with_bistability": 0,
  "iters_with_all_needles": 1,     // == iters_n for a clean pass
  ...
}
```
i.e. PASS = `all_iters_ok: true`, `iters_with_special_token_leak: 0`,
`iters_with_bistability: 0`, and `iters_with_all_needles == iters_n`. Per-stream
`response_text` should equal the expected needle string exactly
(`"FALCON-MERCURY-7749"` in the captured example — this is
`quality_probe_dsv4.py`'s `expected_needle` field, itself derived from the
`NEEDLE` constant it shares with `ab_probe_tier1.py`).

### (iii) Temp=0 byte-identity across two configs on 3 prompts

**No standalone byte-identity SCRIPT with that exact name exists** (search:
`find . -iname "*byte*ident*"` and `find . -iname "*digest*.py"` both came up
empty for scripts — only doc files matched:
`docs/dspark-tier1-byte-identity-2026-08-26.md`). The closest real artifact is
`bench/spec_degen_capture.py` — a capture harness (not a comparator) that
drives a fixed `PROMPTS` list (7 entries: `sys_primary_colors`,
`sys_capital_france`, `sys_count_to_five`, plus 4 more incl. one control) at
`temperature=0.0` and saves per-prompt `content`/`reasoning_content`/
`finish_reason`/`token_ids` to JSON:
```bash
uv run python3 bench/spec_degen_capture.py \
  --base-url http://localhost:52415 \
  --model <model-id> \
  --max-tokens 200 --out ~/spec_degen_groundtruth.json
```
Run it once per config (config A output, config B output), then DIFF the two
JSON files' `content`+`reasoning_content` fields yourself — **no diff/compare
script exists in the repo** (this is what `docs/dspark-tier1-byte-identity-2026-08-26.md`
did manually: "2/3 short prompts byte-identical... 1/3 differs" was a
hand-compiled table, not an automated tool's output). For the sweep's
"3 prompts" ask, use 3 of the 7 `PROMPTS` entries in that script (the doc
itself only used the 3 SHORT prompts that reach `finish=stop` without
truncation: `sys_capital_france`, `sys_count_to_five`, `sys_primary_colors`).

**Tool-call `id` field stripping — CONFIRMED necessary, not yet automated.**
No existing script in this repo strips `tool_call.id` before hashing/comparing
— this must be done by hand/ad hoc when comparing tool-call-bearing responses:
```python
import json
def strip_tool_call_ids(obj):
    if isinstance(obj, dict):
        return {k: (None if k == "id" and "type" in obj and obj.get("type") == "function" else strip_tool_call_ids(v))
                for k, v in obj.items()}
    if isinstance(obj, list):
        return [strip_tool_call_ids(x) for x in obj]
    return obj
```
(Illustrative only — not a repo file; write/adapt before use in the actual
sweep run.)

### (iv) DSML tool-call correctness on 2 tool-call prompts

**Found: `bench/dsv4_dsml_battery.py`.** Purpose per its own docstring:
regression gate for DSML/tool-call markup leakage, malformed tool calls, or
new DSML parse failures in the server log during the run.
```bash
.venv/bin/python bench/dsv4_dsml_battery.py \
  [--api http://192.168.86.201:52415/v1] \
  [--ctx 4096 65536 122880] [--turns 6] [--log ~/exo.log]
```
Exit code: **0 = battery clean; 1 = corruption detected** (details printed to
stdout). It has its own `TOOLS` list (starts with a `terminal` function tool,
confirmed from file head) and checks responses against `LEAK_PATTERNS`
(regex for `<｜DSML｜`, `</tool_call`, `</parameter>`, `</invoke>`, `<|im_`
leakage into content) plus `LOG_FAIL_PATTERNS` grepped from the exo log
(`"DSML tool call parsing failed"`, `"unterminated invoke"`, `"invoke body
was corrupt"`). Defaults to a `--ctx` LADDER (3 context sizes, not literally
"2 tool-call prompts") — pass `--ctx <one size>` or trim `--turns` to shape it
to exactly 2 tool-call-eliciting turns if the sweep wants a narrower gate; the
script's own default is broader than the task's "2 prompts" framing.
Secondary candidate `bench/eval_tool_calls.py` also exists but was not
inspected in depth for this map — `dsv4_dsml_battery.py` is the one purpose-built
as a regression GATE (exit-code contract); prefer it.

---

## 6) Cluster state right now (as of this recon, 2026-09-03)

- **Both runners UP.** `pgrep -fl 'python -m exo'` on both nodes shows a live
  process tree (screen wrapper → login shell → zsh -l → real
  `.venv/bin/python -m exo -v`), PIDs 79266 (m4-1) / 86069 (m4-2) for the real
  runner processes.
- **API endpoint:** `http://192.168.86.201:52415` (m4-1's IP; matches the
  hardcoded default in both `bench/ab_probe_tier1.py` and
  `bench/quality_probe_dsv4.py`'s convention). `curl -s -o /dev/null -w '%{http_code}' http://192.168.86.201:52415/v1/models`
  → **200**. Response body lists multiple models incl.
  `mlx-community/MiniMax-M2.7-4bit`, `mlx-community/Qwen3-Coder-480B-A35B-Instruct-4bit`
  (this is the general model catalog, not necessarily what's currently
  PLACED/loaded — did not query `/state` for current placement as that's
  beyond this recon's live-env-dump scope in the task).
- **Current live `EXO_SPECULATIVE_GAMMA`: `3`** (confirmed via `ps eww` on
  m4-1's real runner PID).
- **Full speculative-related env from the live runner (m4-1, PID 79266),
  filtered to `EXO_SPEC`/`EXO_DSV4`/`MLX_`:**
```
MLX_JACCL_RELIABLE_INFLIGHT=8
MLX_MAX_OPS_PER_BUFFER=200
MLX_MAX_MB_PER_BUFFER=200
EXO_DSV4_SEQSPLIT_BALANCED=1
EXO_DSV4_SPARSE_SDPA_TILE=128
EXO_DSV4_BATCHED_PREFILL=1
EXO_DSV4_PREFILL_ARGPARTITION=1
EXO_DSV4_ARGPARTITION_MIN_P=8192
EXO_DSV4_LMHEAD_LASTROW=1
EXO_DSV4_LMHEAD_MXFP8=1
EXO_DSV4_SEQ_SPLIT=1
EXO_SPECULATIVE=1
EXO_SPECULATIVE_GAMMA=3
EXO_DSV4_FUSED_MOE=0
EXO_DSV4_COMPILE_FFN=0
EXO_DSV4_COMPILE_LAYER=0
EXO_DSV4_FENCE_EVERY_N_LAYERS=4
EXO_DSV4_FENCE_ASYNC=1
EXO_DSV4_FENCE_ASYNC_C2=0
EXO_DSV4_BS_MIN_ACCEPT=1
EXO_DSV4_INDEX_TOPK=512
EXO_DSV4_EXACT_TOPK_PREFILL=1
EXO_DSV4_QUERY_TILED_SDPA=1
EXO_DSV4_QUERY_TILED_B=64
EXO_DSV4_MTP=1
EXO_DSV4_DSPARK=1
EXO_DSV4_VERIFY_ROWSEQ_VEC=1
EXO_DSV4_VERIFY_ROWSEQ_VEC_ROWSDPA=3
EXO_DSV4_ATTN_ALLSUM=0
EXO_DSV4_MTP_C2_MAX_CTX=1
EXO_DSV4_MTP_DEDICATED=0
EXO_DSV4_MTP_EAGLE_K=8
EXO_DSV4_MTP_TIEBREAK_FIX=0
EXO_DSV4_MTP_TIEBREAK_EPS=0.5
EXO_DSV4_MTP_ACCEPT_LOGPROBS=1
EXO_DSV4_POOL_SNAPSHOT_BATCH=1
EXO_DSV4_POOL_RESTORE_AFTER_TRIM=1
EXO_DSV4_ROWSEQ_ROWMASK=1
EXO_DSV4_SPEC_STATE_RESTORE=1
EXO_DSV4_SPEC_CACHE_ROLLBACK=1
EXO_DSV4_SPEC_CACHE_ROLLBACK_C2=1
MLX_GEMV_BATCH_INVARIANT=1
MLX_STEEL_BATCH_INVARIANT=1
EXO_DSV4_ROWSEQ_FULLBLOCK=1
EXO_DSV4_ROWSEQ_FULLBLOCK_MOE=0
EXO_DSV4_MOE_PARTS_ROWSEQ=shared
EXO_DSV4_MTP_MAX_CTX=0
EXO_DSV4_MTP_TIE_REVERIFY=0
EXO_DSV4_VERIFY_ROWSEQ=1
EXO_DSV4_VERIFY_ROWSEQ_MIN_CTX=0
EXO_DSV4_VERIFY_BATCH=1
EXO_DSV4_VERIFY_BATCH_MIN_CTX=8192
MLX_JACCL_ACK_SYNC_PRE=1
MLX_JACCL_ACK_RETRANSMIT_US=500000
MLX_JACCL_RECONNECT_FRESH=1
MLX_JACCL_RELIABLE_OPTIMISTIC=1
EXO_DSV4_POOL_DEFER_COPY_MAX_BYTES=8388608
EXO_DSV4_HC_EXPAND_KERNEL=1
EXO_DSV4_HC_COLLAPSE_KERNEL=1
MLX_JACCL_SHARDING_MODE=Tensor
MLX_JACCL_RELIABLE_DATA=1
MLX_JACCL_RELIABLE_MAX_SZ=2
MLX_EVENT_WAIT_TIMEOUT_MS=20000
```
- **`EXO_DSV4_MTP_PROFILE` and `EXO_PROFILER` are ABSENT from the live env**
  (neither appears in the full unfiltered dump either) — confirms the round-4
  finding is CURRENTLY STILL TRUE: production runs with `EXO_PROFILER_LEVEL=1`
  (visible in the earlier full `ps eww` dump) but WITHOUT `EXO_PROFILER`, so no
  profiler hook is registered and the async fence is intact for any future
  fence/timing measurement taken against this exact live config. No regression
  to flag.

---

## Constraints honored
- Read-only: no `start_cluster.sh` invocation, no runner restart, no
  benchmark request fired against the cluster, no git push. Only `ssh ...
  sudo -n -l`, `ssh ... pgrep/ps eww`, and one `curl` GET against
  `/v1/models` were run against the live cluster — all non-mutating.
- No writes outside this file's directory.
