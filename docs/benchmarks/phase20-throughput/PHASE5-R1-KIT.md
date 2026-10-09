# PHASE 5 — R1 VALIDATION-SESSION KIT (operator runbook)

Author: Phase-5 R1-kit subagent (depth-2). 2026-10-08. **PREPARED ONLY — DO NOT EXECUTE.**

> **STATUS: HELD.** The plan owner is reviewing the lever-2 gate decision
> (`PHASE5-P1-AMENDMENT.md` §9).  **No R1/R2/R3 relaunch, no deploy, no ssh, no POST may run
> until a GO lands.**  Every command below is marked **DO-NOT-RUN-UNTIL-GO**; none of them were
> executed while preparing this kit (all work offline: source-read + arithmetic only).
>
> **Production during this prep: `deploy/next13 @ 576e9d279` (exo) + mlx-lm `3bf8316`, both
> nodes, gates unset, canary healthy, 2 runners Ready. THIS IS THE RESTORE TARGET.** R1 has
> not spent its relaunch.

---

## 0. What R1 is, in one page

**R1 = relaunch #1 of ≤3 this round.** One fresh boot on the **clean lever build at
DEFAULTS**, then a same-session fixed-replay A/B vs the **live production baseline**, a
quality battery on the lever arm, the 91K real-tensor capture replay (ship precondition A2),
and the never-measured `HIER=0` benign point. Budget: **1/3 relaunches.**

| item | value | source |
|---|---|---|
| production (live) | exo `576e9d279` + mlx-lm `3bf8316`, gates unset | `PHASE5-CAMPAIGN.md:7-9` |
| lever build (R1 deploy) | exo `deploy/next18-identity` (`==576e9d279`, diff-empty) + mlx-lm `deploy/next18-lever2 @ cd68bf4` | `PHASE5-P1-LEVER2.md:120-123,329-334` |
| frozen baseline (same harness) | benign 20K **145.75** / agentic 91K **157.10** ms | `PHASE4-CAMPAIGN.md:13-15` |
| frozen next17-defaults | benign 118.56 / agentic 130.07 ms | same |
| next17 + `HIER=0` | agentic **101.06 ms** ≈30.2 t/s (benign **never measured**) | `PHASE5-P0-UNITS.md:286-288` |
| expected lever-at-DEFAULTS | ≈**101 ms agentic** (guard sends small-n to the fallback; L2-full makes it exact) | `PHASE5-P1-LEVER2.md:238-241` |
| **the A/B** | lever-DEFAULTS (in-session) **minus** production baseline (live, pre-relaunch) | this doc §C |
| gates | **G2** agentic Δ ≥15 ms + DISJOINT ranges + benign no-regression + trigger <1 %; **G3** battery CLEAN; **G4** fresh-boot parity within noise | `PHASE5-CAMPAIGN.md:54-56` |

### 0.1 The (a)/(b) decision — **R1 uses (a): lever at DEFAULTS**

The kit **takes the decision the task asked us to state**, and states it:

> **R1 runs the lever at DEFAULTS (option a).** Gates **unset** → `_HIER` default ON (guarded to
> `n > 16`), `_L2_FULL` default ON, `_FENCE_MIN_ROWS` = 16. This is **exactly the config R2 will
> ship** — R1 must validate the ship config.
>
> **`DSV41_INDEXER_HIER=0` is NOT the R1 A/B arm (option b rejected for R1).** It is import-time
> (read once at module load), so it needs a **separate boot**; making it the arm would test a
> config that cannot ship and would burn the round's only R1 relaunch on it. It is provided below
> as a labelled *ablation* that is **not** part of R1's single boot.

**Why the A/B is still informative at DEFAULTS.** The lever-2 guard routes decode(n=1)/verify(n=4)
to the *fallback* path — the same path `HIER=0` forced unconditionally. So next18-at-DEFAULTS
reproduces the `HIER=0` **small-n behaviour** (the ~101 ms agentic win) while keeping the
large-n hierarchical prefill win, and it does it *exactly* (L2-full). The lever-arm number should
land at **≈101 ± ε ms**, not bitwise 101.06: L2-full stores the small-n row in fp32 whereas
next17+HIER=0 used the bf16 fallback row, and large-n is unchanged — immaterial to a G2 floor of
15 ms under a ~56 ms delta.

### 0.2 The A/B structure (the load-bearing clarification)

**Baseline arm = the LIVE production boot, measured in-session, pre-relaunch. Cost: 0 relaunches.**
It is *not* a second boot and *not* frozen-citation-only: re-measuring it (i) gives G2 a real
same-session control, (ii) drift-checks the frozen numbers, and (iii) supplies a go/no-go gate
**before** spending the boot. Sequence is forced: **baseline battery must complete before
relaunch #1 destroys the prod boot.**

**Lever arm = relaunch #1** (the single R1 boot), measured in-session on the lever build.

So R1's ledger = **1 relaunch** (the lever boot). The baseline is free. This resolves the campaign
doc's vague "same-session A/B" wording: the *build* differs across the two arms (cross-boot, by
necessity — the env/code is set at launch), but the *harness, fixed salt, chunking and cluster
state* are identical, so the A/B is content-fixed and same-harness. Label the delta **cross-boot**
(honest ceiling); the expected margin (±56 ms vs a 15 ms bar) makes boot drift irrelevant.

---

## A. Pre-flight — idle-guard, expected-state assertion, canary  — **DO-NOT-RUN-UNTIL-GO**

Script: `~/.hermes/cache/scratch/p5/r1kit/preflight.sh` (read-only except local `/tmp` logs).

**A1 — idle guard before anything** (skill: idle-guard before EVERY chunk/relaunch; wait windows
out; never kill/interfere with user traffic):

```bash
cd /private/tmp/phase20-campaign
~/repos/exo/.venv/bin/python bench/phase20_guard.py wait-idle --max-wait 900 --poll 30
~/repos/exo/.venv/bin/python bench/phase20_guard.py idle      # -> exit 0 idle / 1 busy
```

**A2 — pre-launch state assertion** (skill: assert the expected rev, ABORT if either node differs
— someone else moved it — diagnose before deploying over them):

```bash
for h in studio1 studio2; do
  ssh -o BatchMode=yes -o ConnectTimeout=8 "$h" \
    'printf "%s HEAD=%s MLX=%s\n" "$(hostname)" \
       "$(git -C ~/repos/exo rev-parse --short HEAD)" \
       "$(git -C ~/repos/exo/mlx-lm rev-parse --short HEAD)"'
done
# REQUIRED: HEAD=576e9d279  MLX=3bf8316  on BOTH nodes.  Else ABORT (do not deploy).
```

**A3 — canary, raw GPU, expect healthy ~14-15 TFLOPS** (skill: canary after EVERY boot/READY
before measuring):

```bash
cd /private/tmp/phase20-campaign
~/repos/exo/.venv/bin/python bench/phase20_guard.py canary   # exit 0 healthy / 1 marginal / 2 degraded
```

**ABORT the whole kit if** idle≠0, either node rev differs, or canary ≠ healthy. `marginal` (5-10
TFLOPS under load on the *other* node is contention, not degradation — re-canary when idle);
`degraded` (<5) → stop and report (a reboot-resistant degraded state is new information).

---

## B. Deploy R1 (relaunch #1) — **DO-NOT-RUN-UNTIL-GO**

Script: `~/.hermes/cache/scratch/p5/r1kit/deploy_lever.sh`.

**B1 — deploy via detach-checkout + move BOTH trees** (skill: the mlx-lm submodule **WORKING
TREE** is what `start_cluster.sh` force-installs — a gitlink bump with an un-moved submodule
looks identical in the log):

```bash
cd ~/repos/exo
git checkout --detach 576e9d279            # exo superproject == deploy/next18-identity tip
git -C mlx-lm checkout cd68bf4             # mlx-lm WT == deploy/next18-lever2 (L2-full + guard)

# prove BOTH lever lines are in the WORKING TREE before spending the boot:
grep -c "and m > _FENCE_MIN_ROWS" mlx-lm/mlx_lm/models/deepseek_v41/sparse_attention.py  # lever-1: expect 1
grep -c "if _HIER and n > _FENCE_MIN_ROWS" mlx-lm/mlx_lm/models/deepseek_v41/indexer.py  # lever-2: expect 1
grep -c "_L2_FULL" mlx-lm/mlx_lm/models/deepseek_v41/indexer.py                           # L2-full: expect ≥1
```

**The `EXO_TARGET_BRANCH` subtlety (important).** `start_cluster.sh` checks that local `HEAD` is
an ancestor of `origin/$EXO_TARGET_BRANCH`. `deploy/next18-identity` is **UNPUSHED** (the gitlink
bump to `cd68bf4` is not on `origin`), so `EXO_TARGET_BRANCH=deploy/next18-identity` would trip —
or silently skip — the gate. Use **`deploy/next13`**, which is **the tip of `576e9d279` and exists
on `origin`**: the gate passes cleanly (`576e9d279` is an ancestor of itself), and because the
branch name only drives the *gate*, not what is rsynced, the nodes still install the working tree
(`exo 576e9d279` + `mlx-lm cd68bf4`). (Alternative: `push` a new `origin/deploy/next18-identity`
branch first — a code-branch push, *outside this kit's push scope*; do not do it inside R1.)

```bash
# DEFAULTS: unset every A/B gate so the module defaults apply
unset DSV41_SPARSE_COLSPLIT DSV41_INDEXER_HIER DSV41_INDEXER_L2_FULL \
      DSV41_INDEXER_SMALLN_ROW_BF16 DSV41_INDEXER_ROW_BF16 DSV41_MOE_ALLSUM_BF16 \
      DSV41_SPARSE_FENCE_MIN_ROWS
export EXO_TARGET_BRANCH=deploy/next13
./start_cluster.sh
```

**B2 — READY wait + post-boot canary** (skill: reboot ONLY via `./reboot-node.sh`; this kit does
not reboot — `start_cluster.sh` is a relaunch. If a node must be power-cycled, `cd ~/repos/exo &&
./reboot-node.sh --check studio1 studio2` then `./reboot-node.sh studio1 studio2`; **never raw
`sudo shutdown -r now`** — FileVault pre-boot screen):

```bash
grep -aE "Nodes synchronized|READY \(2/2\)|HEALTHY" /tmp/p5r1/deploy_lever.log | tail
cd /private/tmp/phase20-campaign && ~/repos/exo/.venv/bin/python bench/phase20_guard.py canary
```

**B3 — installed-module verification on BOTH nodes** (skill: resolve the imported module and grep
the *deployed* file, not the gitlink; read the runner env). Script:
`~/.hermes/cache/scratch/p5/r1kit/verify_installed.sh`:

```bash
for h in studio1 studio2; do
  ssh -o BatchMode=yes "$h" '
    V=~/repos/exo/.venv/bin/python
    printf "exo=%s mlx-lm-wt=%s\n" "$(git -C ~/repos/exo rev-parse --short HEAD)" \
                                   "$(git -C ~/repos/exo/mlx-lm rev-parse --short HEAD)"
    IX=$($V -c "import mlx_lm.models.deepseek_v41.indexer as m; print(m.__file__)"); echo "indexer=$IX"
    echo "lever-2 guard: $(grep -c "if _HIER and n > _FENCE_MIN_ROWS" "$IX")"   # expect 1
    echo "L2-full:       $(grep -c "_L2_FULL" "$IX")"                          # expect >=1
    SA=$($V -c "import mlx_lm.models.deepseek_v41.sparse_attention as m; print(m.__file__)")
    echo "lever-1 guard: $(grep -c "and m > _FENCE_MIN_ROWS" "$SA")"           # expect 1
    PID=$(pgrep -f "python -m exo" | head -1)
    echo "runner env:"; ps eww -o command= -p "$PID" | tr " " "\n" | grep -E "^(DSV41_|EXO_)" | sort
  '
done
```

**Required post-boot facts:** both nodes `exo=576e9d279`, `mlx-lm-wt=cd68bf4`; installed `indexer.py`
has the lever-2 guard **and** `_L2_FULL`; installed `sparse_attention.py` has the lever-1 guard;
the runner env shows **no** `DSV41_INDEXER_L2_FULL` / `DSV41_INDEXER_HIER` /
`DSV41_INDEXER_SMALLN_ROW_BF16` keys (DEFAULTS), and `DSV41_SPARSE_FENCE_MIN_ROWS` unset (→ 16).

### B-alt — the `HIER=0` env-arm (an ablation, NOT R1's boot)

If a decision later wants the literal `DSV41_INDEXER_HIER=0` arm it is **its own boot** (import-time
gate):

```bash
cd ~/repos/exo && git checkout --detach 576e9d279 && git -C mlx-lm checkout cd68bf4
unset DSV41_SPARSE_COLSPLIT DSV41_INDEXER_L2_FULL DSV41_INDEXER_SMALLN_ROW_BF16
export DSV41_INDEXER_HIER=0
export EXO_TARGET_BRANCH=deploy/next13
./start_cluster.sh
```

> **Do NOT spend this boot inside R1.** See **§E** and **§J**: the campaign doc's claim that
> the missing `HIER=0` benign point "rides R1 free" is **only true in purpose** (see §E), **not
> literally** — a literal `HIER=0` boot would be R1's *second* relaunch, which the round cannot
> afford. Charge it, if ever spent, to **R3-reserve** on the **ship build** (`next18+HIER=0`), not
> to revive next17.

---

## C. FIXED-REPLAY ASSETS — freeze the content so timing is the only variable — **DO-NOT-RUN-UNTIL-GO**

Script: `~/.hermes/cache/scratch/p5/r1kit/r1_driver.py` (a thin, reviewable extension of the
ship-day `p3b_driver.py`; see its docstring for the exact diff).

**The problem the base driver has.** `p3b_driver.py:114` salts each rep with
`os.urandom(4).hex()` → the prompt **content differs every rep and across arms** — fatal for a
timing A/B (skill: *FIXED replay so content divergence from the lever cannot pollute timing*).
It also never persists/reloads its own-request registry (a killed chunk leaves its POSTs
unregistered → the next chunk's idle gate aborts on them).

**The fix (what the kit's driver adds, and nothing else):**

1. **FIXED salt.** `--salt r1fix-a`; the per-rep salt is `f"{base}-{rep}"` — fully deterministic.
   **Run the SAME `--salt` on both arms** → byte-identical prompts, so the two arms replay
   identical content.
2. **Persisted + reloaded own-request registry.** `--registry
   .../r1kit/own_requests.jsonl` is written by `ChunkGuard.register_own_request()` and **reloaded
   into every chunk's own list** (skill: re-read the own list *after* the log read; persist it to
   disk so a killed chunk's POSTs don't stall the next chunk).

**Harness shapes** (identical on both arms; matches the frozen baseline so the numbers compare):

| arm | driver flags | reps | prompt source |
|---|---|---|---|
| benign 20K | `--arm benign --depth 20000 --max-tokens 800 --gamma 3` | 4 (1 chunk) | `RM.build_prompt(20000, salt, "count")` |
| agentic 91K | `--arm agentic --max-tokens 800 --gamma 3` | 6 (2/chunk) | `AM.build_agentic_prompt(salt, None)` — real Hermes session `20261007_092009_9a2ed7` reconstructed from state.db, flattened to ONE user message |

**Exact invocations (run each arm on each build):**

```bash
PY=$HOME/repos/exo/.venv/bin/python
KIT=~/.hermes/cache/scratch/p5/r1kit/r1_driver.py
O=~/.hermes/cache/scratch/p5/r1kit
SALT=r1fix-a

# --- BASELINE arm: on the LIVE next13/576e9d279 boot, BEFORE relaunch #1 (0 budget) ---
$PY $KIT --arm agentic --total-reps 6 --reps-per-chunk 2 --max-tokens 800 --salt $SALT \
    --label r1_base_agentic --out $O/baseline_agentic.json
$PY $KIT --arm benign  --total-reps 4 --reps-per-chunk 4 --depth 20000 --max-tokens 800 --salt $SALT \
    --label r1_base_benign  --out $O/baseline_benign.json

# --- LEVER arm: on the R1 build, DEFAULTS (post §B) ---
$PY $KIT --arm agentic --total-reps 6 --reps-per-chunk 2 --max-tokens 800 --salt $SALT \
    --label r1_lever_agentic --out $O/r1_lever_agentic.json
$PY $KIT --arm benign  --total-reps 4 --reps-per-chunk 4 --depth 20000 --max-tokens 800 --salt $SALT \
    --label r1_lever_benign  --out $O/r1_lever_benign.json
```

**Interleaving (skill: ≥3 reps, interleaved if possible).** The arms are necessarily cross-boot
(gates are import-time), so full interleaving is impossible; approximate it by measuring the
**baseline on the live boot immediately before the relaunch** and the **lever immediately after
READY+canary**, minimizing the wall-clock gap. If a future boot *can* flip a lever at runtime,
interleave then; here, document the cross-boot label.

**Registration + where outputs land.**

- **Owning the traffic:** the driver calls `guard.register_own_request(time.time())` **immediately
  before every request**; `ChunkGuard(registry_path=...)` appends `{"label","t"}` to
  `~/.hermes/cache/scratch/p5/r1kit/own_requests.jsonl`. Unique salt per *feed* = the fixed
  `r1fix-a-N` sequence (the registry is the "unique salt" that lets the idle gate subtract the
  bench's own traffic).
- **Per-chunk guard JSON:** `~/.hermes/cache/scratch/p5/r1kit/guard/<label>.guard.json`
  (`aborted`, `reason`, `wall_cap_hit`, `own_requests`, `signals_seen`).
- **Results:** `baseline_{benign,agentic}.json(.jsonl)` and `r1_lever_{benign,agentic}.json(.jsonl)`;
  each `.json` carries `summary` (`ms_per_round_median`, `ms_per_round_all`, `decode_tps_median`,
  `mean_accepted_median`).
- **Acceptance sanity (skill):** if `mean_accepted_median` moved materially between arms, the
  comparison is **confounded** (acceptance follows output content) — stop and re-examine, do not
  quote the Δ.

**Chunk discipline:** every chunk ≤15 min, behind `ChunkGuard` (idle-gate on entry, abort-on-
arrival + wall-cap watcher); SIGINT on foreign traffic; log `ABORTED_USER_ARRIVED` /
`ABORTED_WALL_CAP` and discard the chunk. **Never kill/interfere with user traffic — wait it out.**
Agentic 91K ≈ 394 s/rep → 2 reps/chunk.

**G2 analysis (offline, after both arms):**

```bash
python3 ~/.hermes/cache/scratch/p5/r1kit/ab_analyze.py
# prints: AGENTIC delta (baseline-lever) vs 15 ms bar; range-disjointness; benign no-regression;
# mean_accepted sanity.  Cross-check against the frozen anchor: baseline agentic ~157.10,
# lever ~101 -> delta ~56 ms >> 15 ms.
```

---

## D. 91K REAL-TENSOR CAPTURE RUN — **ship precondition A2** — **DO-NOT-RUN-UNTIL-GO**

Script: `~/.hermes/cache/scratch/p5/r1kit/capture_run.sh`.

**What the harness does.** `/private/tmp/next18-lever2/bench/next18_capture.py` monkeypatches
`Indexer.__call__` **inside the model-server process**; for calls with `n ∈ NS` (`1,4`) it records
the real derived inputs (`q`,`index_k`,`w`,`lens`) **and the real `[b,n,k]` output** at decode
(n=1) and verify (n=4), plus an elementwise A/B of HIER-vs-fallback on those **same** args. It
**auto-installs on import when `DSV41_NEXT18_CAPTURE` is set**; the file writes itself atomically
into a bounded ring (`.npz` + `.meta.jsonl`).

**The node invocation (env + bootstrap import):**

```bash
# env (must be set BEFORE the server imports mlx_lm):
DSV41_NEXT18_CAPTURE=/tmp/next18_91k.npz
DSV41_NEXT18_CAPTURE_NS=1,4
# and the server process must import the hook once:
#     import bench.next18_capture   # noqa: F401     (or sitecustomize)
```

**Expected artifact:** `/tmp/next18_91k.npz` on **each** node (the harness is per-process). Copy
back:

```bash
scp studio1:/tmp/next18_91k.npz /private/tmp/next18_91k.m4-1.npz
scp studio2:/tmp/next18_91k.npz /private/tmp/next18_91k.m4-2.npz
```

**OFFLINE REPLAY — diff L2-full vs HIER on those REAL tensors** (the A2 verdict; run on the
laptop with the lever worktree's mlx-lm):

```bash
cd /private/tmp/next18-lever2
PYTHONPATH=$PWD ~/repos/exo/.venv/bin/python bench/next18_replay_capture.py \
    /private/tmp/next18_91k.m4-1.npz
PYTHONPATH=$PWD ~/repos/exo/.venv/bin/python bench/next18_replay_capture.py \
    /private/tmp/next18_91k.m4-2.npz
```

**PASS = `REPLAY_CAPTURE {..., "L2full_vs_hier_total_ndiff": 0, ...}` on BOTH nodes.** Any nonzero
→ the pre-registered **failure branch** (`PHASE5-P1-AMENDMENT.md` §10): L2-full is behaviour-
changing at production parameters → argue value-identity on real production data, or BLOCK the
lever.

> ### ⚠ CAPTURE HOOK WIRE-UP — the one gap this kit cannot close offline
>
> Verified by source-read (offline):
> * `start_cluster.sh` builds `EXO_ENV` from an **allow-list** and forwards **no**
>   `DSV41_NEXT18_CAPTURE*` variables (`grep -n NEXT18 start_cluster.sh` → **no hits**).
> * `mlx_lm` has **no** auto-import of the hook; nothing in the exo repo or the launcher imports
>   `bench.next18_capture`.
>
> So a variable exported in the operator's shell is **silently dropped**, and even if forwarded,
> the server never imports the module → **the capture will not run** without a change. The kit
> therefore ships `launcher_patch_next18.py` (a **prepared, NOT-applied** diff) that adds the
> allow-list forward lines **and** a `/tmp/sitecustomize.py` bootstrap (precedent: the existing
> `EXO_PYTHONPATH_DIAG` / `EXO_PYSAMPLER` mechanism at `start_cluster.sh:1885`).
>
> **DECISION required before R1 (owner):**
> * **(D1)** apply the launcher patch **in the same R1 boot** → capture rides R1 free, A2 satisfied
>   on R1. Requires editing `start_cluster.sh` on the next18 branch before relaunch #1 (a
>   code-branch commit, separate from this kit).
> * **(D2)** run R1 **without** the capture and carry A2 to **R2** (the ship boot already plans a
>   battery + parity smoke; add the capture there). Costs nothing extra *if* the D1 wire-up is
>   ready by R2.
> * **(D3)** run the capture as a **standalone bench-only session on the live R1 boot** if a
>   runtime import path exists — **checked: none does** (`mlx_lm` has no hook; the server process
>   is already running by READY). Not available.
>
> **Recommendation: D1** (it is the whole point of folding the capture into R1), **else D2**. Do
> **not** claim A2 satisfied without a real `.npz` + a 0-diff replay.

---

## E. The MISSING `DSV41_INDEXER_HIER=0` BENIGN point — **DO-NOT-RUN-UNTIL-GO**

**The gap.** Only the **agentic** `next17 + HIER=0` arm exists (**101.06 ms**,
`PHASE3B-SHIP-VALIDATION.md:161`). The matching **benign** `HIER=0` arm was **never** run in any
session (`PHASE5-P0-UNITS.md:286-288`). Expected **≈95 ms** by the benign:agentic ratio.

**What actually rides R1 free.** The DEFAULTS lever boot routes small-n (decode/verify) to the
**same fallback path** `HIER=0` would take, so the lever benign arm **is** the *fallback-path*
benign measurement the missing cell wanted — at ≈95 ms expected. It is **not** the literal
`next17+HIER=0` cell (different build: L2-full's fp32 row vs next17's bf16 fallback; large-n is
still hierarchical at DEFAULTS). **State it as a labelled proxy, not as the literal cell.**

**Exact invocation (rides the R1 lever boot, 0 extra relaunches):**

```bash
PY=$HOME/repos/exo/.venv/bin/python
KIT=~/.hermes/cache/scratch/p5/r1kit/r1_driver.py
$PY $KIT --arm benign --total-reps 4 --reps-per-chunk 4 --depth 20000 --max-tokens 800 \
    --salt r1fix-a --label r1_lever_benign_hier0proxy \
    --out ~/.hermes/cache/scratch/p5/r1kit/r1_lever_benign_hier0proxy.json
```

(This runs as the third leg of `measure_lever.sh`; it is the same benign arm as §C but explicitly
records the *fallback-path* label and pairs with the agentic lever number to re-derive the
benign lever-2 share.) **If the owner insists on the literal `next17+HIER=0` cell, it is its own
boot — charge it to R3-reserve on `next18+HIER=0` (benign+agentic), never a dedicated pre-R1
relaunch.**

---

## F. Battery (G3) on the LEVER arm — **DO-NOT-RUN-UNTIL-GO**

Script: `~/.hermes/cache/scratch/p5/r1kit/battery_lever.sh`.

```bash
cd /private/tmp/phase20-campaign
~/repos/exo/.venv/bin/python bench/phase20_guard.py wait-idle --max-wait 1800 --poll 30
cd /private/tmp/next16-instr/bench/dsv41_quality_battery
/Users/adam.durham/.venv/bin/python battery.py --label r1lever --depth 40000 --force all
/Users/adam.durham/.venv/bin/python compare.py results/g3 results/r1lever
```

**Where results land:** `…/dsv41_quality_battery/results/r1lever/` (`build.json`, `needles.json`,
`tools.json`, `free_prose/`, `park.json`, `summary.json`); raw bodies under `…/raw/`.

**Pass criteria (G3 = battery CLEAN on the lever arm):** `summary.json` `phase_verdict == "CLEAN"` —
i.e. **needles 6/6**, **tools 10/10**, **free-prose `n_dirty == 0`**, **park `recall_teal == true`**,
no HIGH-confidence detector hits (cross-script glue / U+FFFD / repetition / self-doubt). Compare
vs the frozen `g3` baseline: the ship-day run returned **REVIEW** with exactly one item, so the
expected lever result is **CLEAN or REVIEW-only — never DIRTY**. Any `DIRTY` → **G3 FAILS** → do
not ship; write it up as a speed-vs-output tradeoff (the lever is behaviour-changing at production
parameters) and proceed.

---

## G. Optional MoE top-k-hook bench arm — **DO-NOT-RUN-UNTIL-GO**

**Verdict: NOT AVAILABLE AS A CLEAN ENV-GATED ONE-LINER — SKIPPED.**  (Read this as the honest
answer the task allowed, not a dodge.)

Source-read of `mlx_lm/models/deepseek_v41/moe.py` (offline): the only env-gated switch is
`_MOE_ALLSUM_BF16` (a collective-payload dtype, line 51) — there is **no** routing-histogram /
top-k-capture hook, no env gate for one, and the top-k runs inside `Gate.__call__`
(`mx.argpartition(-biased, topk-1)`): adding a hook there **touches the MoE gate path the
forward spends on**, so it is not a "clean one-liner that does not touch the indexer path" — and
it cannot be added at runtime to the already-booted R1 server.  **Skip it.**  (If ever built, it
belongs to a dedicated MoE bench session, not R1.)

Per the campaign doc (`PHASE4-CAMPAIGN.md:48`) the MoE expert differential gets **no dedicated
relaunch** — consistent with skipping here.

---

## H. Post-R1 decision branch — **DO-NOT-RUN-UNTIL-GO**

**After the R1 measurements + battery + A2 replay are in, apply the pre-registered decision.**
G2 needs **all** of: agentic Δ ≥15 ms (**DISJOINT** per-rep ranges), benign no-regression,
trigger telemetry <1 % of small-n calls (L2-full trigger is 100 % **by design** — it *is* the
small-n path; report it as "the design", and if the amended gate keeps a trigger-rate leg, it is
the L2-guard's leg and N/A to L2-full). G3 needs battery CLEAN; G4 needs a fresh-boot parity smoke
within noise.

### H1 — PASS (G2 Δ≥15 ms disjoint ∧ G3 CLEAN ∧ A2 replay 0-diff)

**R2 = ship (relaunch #2):** the lever **default-on, retire the env var** (`DSV41_INDEXER_L2_FULL`
default stays `1`; the guard predicate stays `n > _FENCE_MIN_ROWS`), fresh boot, then
**canary + battery + parity smoke within noise** (fresh-boot G4), then tag `known-good-*` on both
forks (`adurham/exo` + `adurham/mlx-lm`). Budget after R2: **2/3**.

```bash
# R2 (ship) sketch — DO NOT RUN UNTIL GO
cd ~/repos/exo && git checkout --detach 576e9d279 && git -C mlx-lm checkout cd68bf4
unset DSV41_SPARSE_COLSPLIT DSV41_INDEXER_HIER DSV41_MOE_ALLSUM_BF16 DSV41_SPARSE_FENCE_MIN_ROWS
# (L2-full stays default-ON: no DSV41_INDEXER_L2_FULL in env)
export EXO_TARGET_BRANCH=deploy/next13
./start_cluster.sh
~/repos/exo/.venv/bin/python /private/tmp/phase20-campaign/bench/phase20_guard.py canary
# + battery (label r2lever) + a short benign parity smoke vs the R1 lever median (within noise)
# + git tag known-good-decode-next18-<ts> on BOTH forks + PERFORMANCE_HISTORY on main (same window)
```

### H2 — FAIL (Δ<15 ms, or G3 DIRTY, or A2 replay nonzero)

**R3 = reserve-only (relaunch #3): one pre-named retry.** Declare it in the ledger **before**
spending. The pre-named retry is **not** a new lever: it is either (i) the **`next18+HIER=0`
ablation pair** (if the failure is an ambiguous benign regression — disambiguates whether the
large-n hierarchical path or the small-n fallback caused it), or (ii) **restore** the best SHIPPED
build and close the round with the speed-vs-output statement. **Nothing else.** Budget after R3:
**3/3, round closed.**

### H3 — ABORT (a pre-flight / READY / canary / verify leg fails)

Do **not** proceed; run **ABORT / ROLLBACK** (§I) to return to `deploy/next13 @ 576e9d279` +
mlx-lm `3bf8316`. If the failure was *before* the deploy touched the cluster (a refused checkout,
an abort at the idle gate), it does **not** count as a spent relaunch; if the boot completed, it
**does** — record the distinction in the ledger ("attempted" vs "spent").

---

## I. ABORT / ROLLBACK — restore the RESTORE target — **DO-NOT-RUN-UNTIL-GO**

Script: `~/.hermes/cache/scratch/p5/r1kit/restore_production.sh`.

**Target: `deploy/next13 @ 576e9d279` (exo) + mlx-lm `3bf8316`, BOTH nodes, gates unset, canary
healthy, 2 runners Ready.**

```bash
cd /private/tmp/phase20-campaign
~/repos/exo/.venv/bin/python bench/phase20_guard.py wait-idle --max-wait 1800 --poll 30
~/repos/exo/.venv/bin/python bench/phase20_guard.py idle     # must be ok
cd ~/repos/exo
git checkout --detach 576e9d279
git -C mlx-lm checkout 3bf8316
grep -c "and m > _FENCE_MIN_ROWS" mlx-lm/mlx_lm/models/deepseek_v41/sparse_attention.py  # lever-1 present
grep -c "_L2_FULL"               mlx-lm/mlx_lm/models/deepseek_v41/indexer.py             # EXPECT 0 (absent)
unset DSV41_SPARSE_COLSPLIT DSV41_INDEXER_HIER DSV41_INDEXER_L2_FULL \
      DSV41_INDEXER_SMALLN_ROW_BF16 DSV41_INDEXER_ROW_BF16 DSV41_MOE_ALLSUM_BF16 \
      DSV41_SPARSE_FENCE_MIN_ROWS
export EXO_TARGET_BRANCH=deploy/next13
./start_cluster.sh
grep -aE "Nodes synchronized|READY \(2/2\)|HEALTHY" /tmp/p5r1/restore_production.log | tail
cd /private/tmp/phase20-campaign && ~/repos/exo/.venv/bin/python bench/phase20_guard.py canary
```

**Post-restore assertions (write the RESTORED line + parity smoke in the round doc):**

```bash
for h in studio1 studio2; do
  ssh -o BatchMode=yes "$h" \
    'printf "%s HEAD=%s MLX=%s\n" "$(hostname)" \
       "$(git -C ~/repos/exo rev-parse --short HEAD)" \
       "$(git -C ~/repos/exo/mlx-lm rev-parse --short HEAD)"'
done
# REQUIRED: HEAD=576e9d279 MLX=3bf8316 on BOTH; canary healthy; a short benign parity smoke
# vs the pre-R1 baseline within noise (ship-day reproduced to 0.04 ms).
```

Plus: re-assert **gates unset** (the `ps eww` env read in §B3 shows no `DSV41_*` A/B keys), **no
stray bench processes** on either node, and the moved-aside `docs/benchmarks/phase19-latency/`
restored on the laptop (the scripts do this).

---

## J. Gaps / risks (honest; what this kit could not close offline)

1. **The 91K capture needs a launcher change (the R1 blocker).** No `DSV41_NEXT18_CAPTURE*`
   forward and no bootstrap import exist (`§D`). **Decision D1/D2 required before R1.** Without
   it, A2 is **not** satisfied on R1 and must move to R2.
2. **`EXO_TARGET_BRANCH` for an unpushed branch.** `deploy/next18-identity` is unpushed; the kit
   uses `deploy/next13` (tip `576e9d279`, on origin) to satisfy the push-gate while rsyncing the
   lever working tree. *Alternative (outside this kit's push scope): push the branch first.*
3. **Cross-boot A/B.** The env/code is import-time, so the baseline and lever cannot share one
   boot; the Δ is a **cross-boot** ceiling. Mitigated by fixed content, same harness, adjacent-in-
   time arms, and a ~56 ms margin vs a 15 ms bar.
4. **The `HIER=0` benign point is a proxy, not the literal cell.** "Rides R1 free" is true only in
   *purpose* (the lever's small-n fallback path) — **the literal `next17+HIER=0` benign cell is a
   boot**. Flagged in §E; charge to R3-reserve if ever needed.
5. **`DSV41_INDEXER_L2_FULL` / `_SMALLN_ROW_BF16` are not launcher-forwarded.** Irrelevant to R1
   (DEFAULTS), but the `SMALLN_ROW_BF16=1` ablation cannot be run from the launcher without the
   same patch as gap 1.
6. **No runtime kill-switch.** Every A/B here is a boot; there is no cheap mid-run flip (skill:
   "env flips are foot-guns nobody carries into a launch script" — R2 retires the env).
7. **Never verified live:** a real user arrival mid-chunk (would need a live POST — forbidden);
   covered only by the guard's recorded-fixture test. The R1 chunks are the first production use
   of `ChunkGuard` on the TP2 cluster — watch the guard JSONs for spurious aborts.
8. **MoE hook not available** (`§G`); no bench arm for hot-expert-cache sizing this round.

---

## K. Kit index (all under `~/.hermes/cache/scratch/p5/r1kit/`, PREPARED not run)

| file | section | what it does |
|---|---|---|
| `preflight.sh` | A | idle-guard + rev assertion + canary |
| `baseline_measure.sh` | C | baseline arm on the LIVE prod boot (0 budget) |
| `deploy_lever.sh` | B | R1 lever deploy + working-tree guard check + post-boot canary |
| `verify_installed.sh` | B3 | installed-module + runner-env verify on both nodes |
| `r1_driver.py` | C/D/E | FIXED-salt + persisted-registry arm driver (extends `p3b_driver.py`) |
| `measure_lever.sh` | C/D/E | lever leg: agentic (capture window) + benign + HIER0-proxy |
| `capture_run.sh` | D | 91K capture staging + live-hook assertion + scp-back |
| `launcher_patch_next18.py` | D | prepared (NOT-applied) launcher forward + bootstrap diff |
| `battery_lever.sh` | F | G3 battery on the lever arm + compare vs `g3` |
| `ab_analyze.py` | C | G2 Δ/range-disjoint/mean_accepted analysis (offline) |
| `restore_production.sh` | I | rollback to `deploy/next13 @ 576e9d279` + mlx-lm `3bf8316` |

---

## L. Artifacts / provenance (offline, this kit)

- Campaign anchor + gates: `PHASE5-CAMPAIGN.md` (R1/R2/R3 ledger §2, gates §3, P3 §7).
- Gate evidence: `PHASE5-P1-LEVER2.md`, `PHASE5-P1-AMENDMENT.md`; units: `PHASE5-P0-UNITS.md`.
- Guard contract: `bench/phase20_guard.py`, `GUARD-CONTRACT.md`, `GUARD-NOTES.md`.
- Prior ship-day scripts (models): `~/.hermes/cache/scratch/p3b/{deploy_next17.sh,
  deploy_next17_hier0.sh, battery_next17.sh, restore_next13.sh, ship_next17.sh, p3b_driver.py}`;
  `~/.hermes/cache/scratch/p4/{p2_off_deploy.sh, p2_off_measure.sh, restore_final.sh}`.
- Lever code: mlx-lm `/private/tmp/next18-lever2 @ cd68bf4` (`indexer.py`, `_gates.py`; capture +
  replay harnesses under `bench/`); exo `/private/tmp/next18-exo @ deploy/next18-identity`.
- Battery + compare: `/private/tmp/next16-instr/bench/dsv41_quality_battery/{battery.py,compare.py}`.
- **0 relaunches spent preparing this kit. Production untouched: `576e9d279` / `3bf8316`, gates
  unset.** This doc is the resume anchor for R1; execute only under a GO.
