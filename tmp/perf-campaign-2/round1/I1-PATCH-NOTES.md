# I1 — EXO_DSV4_COLL_PROFILE instrumentation patch (IMPLEMENT-ONLY)

Campaign 2 / round 1 / I1. Written 2026-09-03. **Nothing was deployed, relaunched,
measured, or committed.** The cluster was not touched. All edits are uncommitted
working-tree changes; the PM handles commits.

Goal: measure what SHARE of a decode VERIFY forward is spent inside the per-layer
post-MoE cross-rank `all_sum`, expressed as a WITHIN-BOOT ratio (absolute ms
across boots is inadmissible — between-boot variance is ~6 tok/s).

---

## 1. FILES TOUCHED (complete list)

| repo | file | change |
|---|---|---|
| `~/repos/exo/mlx-lm` | `mlx_lm/models/deepseek_v4.py` | +183 lines net (see A6 note) |
| `~/repos/exo` | `src/exo/worker/engines/mlx/speculative/dsv4_mtp.py` | +130 / −6 |
| `~/repos/exo` | `start_cluster.sh` | +13 (allowlist + comment) |
| `~/repos/exo` | `tmp/perf-campaign-2/round1/jaccl_allsum_probe.py` | NEW file (untracked) |
| `~/repos/exo` | `tmp/perf-campaign-2/round1/I1-PATCH-NOTES.md` | NEW file (this doc) |

`git -C ~/repos/exo diff --stat` also shows a ` mlx-lm | 0` row: that is the
submodule gitlink registering the submodule's dirty working tree. No other file
is modified. `git status --short` in `~/repos/exo` shows only these three `M`
entries plus one ` m mlx-lm` plus untracked `tmp/` dirs.

### A6 note on the raw diffstat
`git -C ~/repos/exo/mlx-lm diff --stat` reports `258 insertions(+), 75 deletions(-)`.
Those 75 "deletions" are **pure re-indentation** of the pre-existing Phase-H
fence block (it moved one level deeper into a new `else:`). Proof:

```
git -C ~/repos/exo/mlx-lm diff --stat -w
  mlx_lm/models/deepseek_v4.py | 183 +++++++++++++++++++++++++++++++++
  1 file changed, 183 insertions(+)          <-- ZERO deletions ignoring whitespace
```

No pre-existing logic was removed or altered.

---

## 2. EXACT FILE:LINE MAP

### `mlx-lm/mlx_lm/models/deepseek_v4.py`

| lines | what |
|---|---|
| `84–144` | module header: rationale, gating contract, lazy-eval discipline, observer-effect caveat |
| `145` | `_COLL_PROFILE_INTERVAL = int(os.environ.get("EXO_DSV4_COLL_PROFILE", "0") or "0")` |
| `146` | `_COLL_PROFILE = _COLL_PROFILE_INTERVAL > 0` — the gate |
| `147` | `_COLL_PROF_PERF = _bp_time.perf_counter` |
| `149` | `_COLL_PROF_SAMPLES: List[float]` — per-forward per-layer sample buffer |
| `153` | `_COLL_PROF_DEPTH = [0]` — re-entrancy guard |
| `155` | `_COLL_PROF_FWD = [0]` — forward counter (drives the emit interval) |
| `158–182` | `_coll_profile_forward()` — the whole-forward (verify) bracket |
| `185–219` | `_coll_profile_emit()` — the `[COLL-PROF]` line |
| `3211` | `if self.sharding_group is not None:` (unchanged) |
| `3212` | `if _COLL_PROFILE and _COLL_PROF_DEPTH[0]:` — the per-layer gate |
| `3213–3246` | profiled per-layer branch |
| `3247–3248` | `else:` + `with span("moe.all_sum"):` — **production path, unchanged** |
| `3249–3324` | the original Phase-H fence block, re-indented only |
| `7395–7404` | `Model.__call__` entry gate → `_coll_profile_forward` |

### `src/exo/worker/engines/mlx/speculative/dsv4_mtp.py`

| lines | what |
|---|---|
| `845–888` | COORD-profiler header incl. the explicit "what is NOT bracketed and why" |
| `889–890` | `_COLL_PROFILE_INTERVAL` / `_COLL_PROFILE` (same env var) |
| `891–893` | `_COLL_PROF_COORD_CYCLES`, `_COLL_PROF_COORD_ACC` |
| `896–898` | `_coll_prof_coord_record()` |
| `901–926` | `_coll_prof_coord_emit()` |
| `2338–2355` | bracket around `all_sum(presence_arr, coord_group)` (+ untouched `else`) |
| `2392–2410` | bracket around `all_max(_num_tokens, coord_group)` (+ untouched `else`) |
| `2412–2414` | `if _COLL_PROFILE: _coll_prof_coord_emit()` — one coord "cycle" per `_next()` |

### `start_cluster.sh`

| line | what |
|---|---|
| `2037` | the mirrored `EXO_DSV4_MTP_PROFILE` line (unchanged) |
| `2038–2049` | comment block |
| **`2050`** | **`[ -n "${EXO_DSV4_COLL_PROFILE:-}" ] && EXO_ENV="$EXO_ENV EXO_DSV4_COLL_PROFILE=$EXO_DSV4_COLL_PROFILE"`** |

Placement verified by reading the surrounding code, not by pattern-matching:
`EXO_ENV` is initialised at `start_cluster.sh:1545` and consumed at `:2791`
(`LAUNCH_CMD=`) and `:2845` (the `screen -dmS exorun` invocation). Line 2050 sits
between them, inside the same `for NODE` body, so it genuinely reaches the runner
process.

---

## 3. A5 — EVERY END-TIMESTAMP AND ITS PRECEDING FORCED MATERIALIZATION

Four timed regions exist. Each row is `eval-before-t1` (mandatory) and
`eval-before-t0` (input, to prevent upstream work being billed to the wire).

| # | region | t0 line | input eval (before t0) | collective | **output eval (before t1)** | t1 line |
|---|---|---|---|---|---|---|
| 1 | per-layer MoE all_sum | `deepseek_v4.py:3229` | **`deepseek_v4.py:3228` `mx.eval(y)`** | `:3231` | **`deepseek_v4.py:3236` `mx.eval(y)`** | `:3238` |
| 2 | whole verify forward | `deepseek_v4.py:171` | n/a (see note) | whole model | **`deepseek_v4.py:177` `mx.eval(out)`** | `:178` |
| 3 | coord `all_sum` presence | `dsv4_mtp.py:2343` | **`dsv4_mtp.py:2342` `mx.eval(presence_arr)`** | `:2344` | **`dsv4_mtp.py:2349` `mx.eval(counted)`** | `:2351` |
| 4 | coord `all_max` numtokens | `dsv4_mtp.py:2399` | **`dsv4_mtp.py:2398` `mx.eval(_cp_in)`** | `:2400` | **`dsv4_mtp.py:2401` `mx.eval(synced)`** | `:2404` |

**Region 2 input-eval note.** The verify bracket deliberately has no pre-t0 eval.
Its t0 is taken at the model-forward entry point; there is no "input" to drain
other than the token ids, which are already materialized by the caller. Adding a
sync there would only measure the caller's own graph.

**Why regions 1/3/4 also eval the INPUT (explicit reasoning, per the brief).**
Without it, the post-collective `mx.eval` drains the entire upstream chain that
fed the collective and bills it to the wire. At region 1 that upstream chain is
the whole MoE block (`gather_qmm` over 24 (row,expert) pairs, measured
0.27–0.43 ms/layer) — an order of magnitude larger than the collective itself,
so the input eval is the difference between measuring the wire and measuring the
MoE. Regions 3/4 are cheap CPU-built int32 arrays; the input eval there costs
essentially nothing and keeps array construction out of the bracket.

**Regions 3 and 4 add ZERO net synchronization.** Both collectives were
*already* followed by an unconditional `mx.eval` in production (`dsv4_mtp.py:2355`
and `:2410`, preserved verbatim in the `else` branches) because the results are
immediately `.tolist()`-ed. The profiler only reads a clock either side of a
materialization that already happened.

---

## 4. A4 — PROOF THE OFF PATH IS UNCHANGED

Three gate sites, all short-circuiting on a module-global bool evaluated **once at
import**, before any eval/timing call:

- `deepseek_v4.py:3212` — `if _COLL_PROFILE and _COLL_PROF_DEPTH[0]:` → falls to
  `else:` at `:3247`, which is the byte-identical production `with span(...)` block.
- `deepseek_v4.py:7403` — `if _COLL_PROFILE and not _COLL_PROF_DEPTH[0]:` → falls
  through to the unmodified `Model.__call__` body.
- `dsv4_mtp.py:2338 / 2392 / 2412` — `if _COLL_PROFILE:` → `else` branches hold
  the verbatim original statements.

`_COLL_PROFILE` is `_COLL_PROFILE_INTERVAL > 0`; `int(os.environ.get(..., "0") or "0")`
makes unset, `""`, and `"0"` all evaluate to `0` → `False`. **No `os.environ` read,
no `mx.eval`, no `perf_counter`, no list/dict touch occurs on the hot path when
off.** Residual cost = one module-global bool load per site, identical to the
pre-existing `_ALLSUM_PROBE_ENABLED` convention.

Verified locally (real execution, not inspection):

```
$ .venv/bin/python /tmp/coll_prof_selftest.py
_COLL_PROFILE = False  interval = 0
OFF-PATH OK: gate false, no state touched

$ EXO_DSV4_COLL_PROFILE=1 .venv/bin/python /tmp/coll_prof_selftest.py
[COLL-PROF] pid=4289 fwd=1 B=1 L=4 n=43 coll_sum_ms=0.0788 coll_mean_ms=0.0018 \
  coll_min_ms=0.0010 coll_max_ms=0.0055 fwd_ms=7.8117 share=0.0101
SELFTEST OK
```

(single-rank, so `all_sum` is a local identity — the numbers are meaningless;
what this proves is that the bracket fires 43 times, the depth guard raises and
releases, and the line parses.)

Acceptance checks re-run after the final edit:

```
A1  grep -n 'EXO_DSV4_COLL_PROFILE' start_cluster.sh   -> 2050 (allowlist line, not a comment)
A2  bash -n /Users/adam.durham/repos/exo/start_cluster.sh -> exit 0
A3  ast.parse(deepseek_v4.py)  -> OK
    ast.parse(dsv4_mtp.py)     -> OK
    ast.parse(jaccl_allsum_probe.py) -> OK
A6  see §1
```

---

## 5. DEPLOY TO THE STUDIOS — **editing the fork source IS sufficient** (with one caveat)

**Answer: the venv copy does NOT need a manual sync, PROVIDED the next dispatch
relaunches via `start_cluster.sh`. There is no separate command to run on the
Studios.**

Read from `start_cluster.sh`, per-node, in this order:

1. `:1309` — `rsync -a --delete "$HOME/repos/exo/" "$NODE:~/repos/exo/"`.
   **The LAPTOP working tree is the deploy vehicle.** The Studios' own checkouts
   are overwritten and never consulted. Uncommitted edits deploy fine — no
   commit, push, or `git reset` is required.
2. `:1489` — `ssh "$NODE" "... uv pip install --no-deps --force-reinstall ./mlx-lm"`.
   This rebuilds `~/repos/exo/.venv/.../site-packages/mlx_lm/` **from the
   just-rsynced fork source**, so the byte-identical venv copy is regenerated
   with the patch in it automatically.
3. `src/exo/...` needs nothing at all: `site-packages/exo.pth` points at
   `/Users/adam.durham/repos/exo/src` (editable install), so the rsynced
   `dsv4_mtp.py` is live on import.

### CAVEAT — `tmp/` IS EXCLUDED FROM THE RSYNC

`start_cluster.sh:1315` has `--exclude 'tmp/'`. The microbenchmark at
`tmp/perf-campaign-2/round1/jaccl_allsum_probe.py` will **NOT** reach the Studios
via a relaunch. To run it there, copy it explicitly:

```bash
scp ~/repos/exo/tmp/perf-campaign-2/round1/jaccl_allsum_probe.py \
    macstudio-m4-1:~/jaccl_allsum_probe.py
scp ~/repos/exo/tmp/perf-campaign-2/round1/jaccl_allsum_probe.py \
    macstudio-m4-2:~/jaccl_allsum_probe.py
```

### POST-RELAUNCH VERIFICATION (do this before believing any number)

```bash
# 1. the patch is in the INSTALLED package on BOTH nodes (not just the checkout)
for N in macstudio-m4-1 macstudio-m4-2; do
  ssh $N "grep -c '_COLL_PROFILE' ~/repos/exo/.venv/lib/python3.13/site-packages/mlx_lm/models/deepseek_v4.py"
done   # expect a nonzero count on both

# 2. the env var actually reached the runner (the allowlist trap)
for N in macstudio-m4-1 macstudio-m4-2; do
  ssh $N "ps eww \$(pgrep -f 'python -m exo' | head -1) | tr ' ' '\n' | grep EXO_DSV4_COLL_PROFILE"
done   # expect EXO_DSV4_COLL_PROFILE=<N> on both
```

If check 2 is empty, the allowlist edit did not survive — **do not bench**.

---

## 6. HOW TO ENABLE IT

```bash
cd ~/repos/exo && EXO_DSV4_COLL_PROFILE=20 ./start_cluster.sh
```

Value semantics mirror `EXO_DSV4_MTP_PROFILE`:

- unset / `0` → OFF (production behavior, byte-identical)
- `N > 0` → emit one `[COLL-PROF]` line every Nth model forward, and one
  `[COLL-PROF] kind=coord` block every Nth coord cycle.

`1` emits every forward (~25–60 lines/s at decode — heavy log volume). **`20` is
the recommended setting**; combine with `EXO_DSV4_MTP_PROFILE=50` for cross-check
against the existing `[MTP-PROF]` verify phase.

Harvest with the established byte-window pattern (as in
`tmp/verify-decomposition-20260901/run_scan.sh`):

```bash
ssh macstudio-m4-1 "tail -c +$((b1+1)) ~/exo.log | head -c $((a1-b1)) | grep -a 'COLL-PROF'"
```

---

## 7. `[COLL-PROF]` OUTPUT FORMAT (for the parser)

### Per-forward model-TP line (stderr, `deepseek_v4.py:208–217`)

```
[COLL-PROF] pid=<int> fwd=<int> B=<int> L=<int> n=<int> coll_sum_ms=<float> coll_mean_ms=<float> coll_min_ms=<float> coll_max_ms=<float> fwd_ms=<float> share=<float>
```

Real example (single-rank self-test):

```
[COLL-PROF] pid=4289 fwd=1 B=1 L=4 n=43 coll_sum_ms=0.0788 coll_mean_ms=0.0018 coll_min_ms=0.0010 coll_max_ms=0.0055 fwd_ms=7.8117 share=0.0101
```

| field | meaning |
|---|---|
| `pid` | runner pid (distinguishes the two ranks in a merged log) |
| `fwd` | monotonic forward counter since process start |
| `B`, `L` | input shape. **Filter `L=4` for the verify forward** at the deployed γ=3. `L=1` is plain decode; large `L` is prefill |
| `n` | per-layer collectives observed in THIS forward. Expect **43**. `n=0` ⇒ unsharded/single-rank |
| `coll_sum_ms` | Σ of the 43 per-layer brackets |
| `coll_mean/min/max_ms` | per-call stats within this forward |
| `fwd_ms` | the whole-forward bracket, SAME forward |
| **`share`** | `coll_sum_ms / fwd_ms` — **the deliverable metric**, within-boot, within-forward |

Suggested regex:

```python
re.compile(
    r"\[COLL-PROF\] pid=(?P<pid>\d+) fwd=(?P<fwd>\d+) B=(?P<b>\d+) L=(?P<l>\d+) "
    r"n=(?P<n>\d+) coll_sum_ms=(?P<sum>[\d.]+) coll_mean_ms=(?P<mean>[\d.]+) "
    r"coll_min_ms=(?P<min>[\d.]+) coll_max_ms=(?P<max>[\d.]+) "
    r"fwd_ms=(?P<fwd_ms>[\d.]+) share=(?P<share>[\d.]+)"
)
```

### Per-cycle coord line (via `logger.warning`, `dsv4_mtp.py:920–925`)

```
[COLL-PROF] kind=coord pid=<int> cycles=<int> name=<str> n=<int> sum_ms=<f> mean_ms=<f> min_ms=<f> max_ms=<f>
```

`name` ∈ {`all_sum_presence`, `all_max_numtokens`}. Stats are per WINDOW (the
accumulator clears after each emit) and per CYCLE. **Do not sum these with the
per-forward line** — different process group, different transport (TCP not RDMA),
different cadence.

---

## 8. CORRECTNESS CAVEATS — READ BEFORE INTERPRETING ANY NUMBER

1. **Observer effect (the important one).** Production runs
   `EXO_DSV4_FENCE_ASYNC=1`, so the per-layer fence is normally the *non-blocking*
   `mx.async_eval(y)`. Under `COLL_PROFILE`, the bracket forces a **blocking**
   `mx.eval` per layer, destroying CPU-encode/GPU-execute overlap that production
   genuinely enjoys. Therefore:
   - `fwd_ms` under the profiler is **larger** than the real unprofiled verify wall.
     **Do not report profiled `fwd_ms` as a decode-latency figure.**
   - `share` is an **UPPER BOUND** on the collective's true share.
   - A **low** share is strong evidence (the collective isn't the bottleneck even
     when maximally penalised). A **high** share is weaker and must be
     cross-checked against the standalone microbenchmark.
   This is unavoidable — a lazily-scheduled collective cannot be timed without
   being materialised.

2. **`share` includes cross-rank arrival skew.** A collective's wall time on rank
   R includes however long R waited for its peer. That is real cost, but it is
   *straggler* cost, not *transport* cost. Compare the in-model `coll_mean_ms`
   against the microbenchmark's 57KB `p50` to split the two.

3. **`broadcast_from_canonical` is NOT measured.** Deliberate, not an oversight.
   It has 13+ call sites, has no existing eval (its `all_gather` is consumed
   lazily), and its own docstring records that changing how its collective is
   scheduled previously desynchronised chained-MTP outputs across ranks at temp>0
   (every chained step collapsing to BOS). Instrumenting it means injecting 13+
   new syncs into the draft chain per cycle — that changes draft scheduling rather
   than observing it. **Its per-cycle cost is unmeasured; the coord line does not
   cover it.**

4. **Which forward is "verify" must be inferred from `L`.** The instrumentation
   sits in `Model.__call__`, which serves decode, verify, tree-verify and prefill.
   Filter `L=4` (γ+1 at the deployed γ=3) and sanity-check `n=43`.

5. **Not measured on real hardware.** Everything above was verified by inspection,
   `ast.parse`, `bash -n`, and a single-rank local self-test. The 2-rank behavior
   is unexercised by this dispatch, by instruction.

---

## 9. THE MICROBENCHMARK — `tmp/perf-campaign-2/round1/jaccl_allsum_probe.py`

**Written from scratch.** The `scripts/all_sum_latency_probe.py` mentioned in the
`exo-cluster-operations` skill **does not exist** — checked `scripts/`, `bench/`,
and `git log --all --oneline --diff-filter=A -- '*all_sum_latency_probe*'` (empty).
That skill reference is stale; worth patching. The closest existing relatives are
`bench/phase0a_allsum_boundary_decompose.py` (stream-boundary decomposition, not a
payload sweep) and `bench/phase0b_collective_overlap_probe*.py` (overlap). The new
probe borrows `phase0a`'s launch convention and median-over-reps discipline;
everything else is new.

Measures `all_sum` at **1KB / 57KB / 1MB**, N=1000 each, reporting
mean/min/max/p50/p99 (+ stdev). Same eval discipline: input materialized once
before the loop; `mx.eval(y)` before every end timestamp. A tiny 1-element
`all_sum` barrier between iterations bounds arrival-skew accumulation
(`--no-barrier` to see raw skew).

`PRODUCTION_BYTES = 32768` is recorded in the output header — the real per-layer
payload `(1,4,4096)` bf16 = 32 KiB sits between the 1KB and 57KB points.

**NOT RUN**, per instruction. Verified: `ast.parse` OK, `--help` OK, and full
module exec via `importlib` (imports resolve against the live `.venv`).

```bash
# two genuine ranks over the production transport
.venv/bin/mlx.launch -n 2 --backend jaccl --hostfile <jaccl hostfile> \
    ~/jaccl_allsum_probe.py --backend jaccl --json-out ~/allsum_probe.json

# TCP ring fallback — harness sanity only, NOT the production transport
.venv/bin/mlx.launch -n 2 --backend ring ~/jaccl_allsum_probe.py --backend ring
```

**Warning:** jaccl claims the RDMA devices. Do not run the jaccl arm while the
runners hold them — run it inside the relaunch window (after teardown, before
the runners come up), which is why this was scoped to the next dispatch.

---

## 10. HOW TO REVERT

Nothing is committed, so revert is a clean checkout:

```bash
git -C ~/repos/exo/mlx-lm checkout -- mlx_lm/models/deepseek_v4.py
git -C ~/repos/exo checkout -- src/exo/worker/engines/mlx/speculative/dsv4_mtp.py \
                               start_cluster.sh
rm -rf ~/repos/exo/tmp/perf-campaign-2/round1/jaccl_allsum_probe.py
# (leave this notes file, or rm it too)

# verify
git -C ~/repos/exo/mlx-lm diff --stat   # empty
git -C ~/repos/exo diff --stat          # empty
```

The Studios revert on the next `start_cluster.sh` (the rsync + force-reinstall
re-derives both the checkout and the venv from the reverted laptop tree).

**If already committed**, revert the three files with
`git revert <sha>` in each repo, then relaunch.

**Soft revert without touching code:** simply unset `EXO_DSV4_COLL_PROFILE` (or
set it to `0`) and relaunch. The gate makes the code inert — the patch is safe to
leave in the tree while the cluster serves production traffic. That is the whole
point of the gating contract in §4.

---

## 11. OPEN QUESTIONS / ASSUMPTIONS STATED EXPLICITLY

1. **Assumed** the verify forward reaches `Model.__call__` (`deepseek_v4.py:7392`).
   Verified from `dsv4_mtp.py`'s `self.model(...)` call sites (`:3243`, `:5035`,
   `:5105`, `:5168`, `:5322`, `:5734`) — all go through `Model.__call__`. Not
   verified on live hardware.
2. **Assumed** γ=3 ⇒ verify `L=4`. Consistent with the established facts, but the
   next dispatch should confirm empirically from the emitted `L` distribution
   rather than assuming.
3. **Not attempted:** timing `broadcast_from_canonical` (see caveat 3). If the
   coord tail turns out to matter, that needs its own design.
4. **Not attempted:** distinguishing transport time from straggler-wait inside the
   in-model bracket. The microbenchmark is the intended control for that split.
