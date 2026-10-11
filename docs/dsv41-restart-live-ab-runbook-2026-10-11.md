# DSv4.1 restart-survival — LIVE A/B deploy + validation + rollback runbook (2026-10-11)

**STATUS: RUNBOOK ONLY — NOT EXECUTED.** Authored OFFLINE (laptop, doc-only).
The live phase below is to be executed by the operator during the approved live
window; nothing in this document has been run, and writing it does not deploy
anything. Cluster contact is **explicitly out of scope for authoring** (the MoE
lever-#2 spike is benching on studio2; zero interference).

Companion documents (the "build notes" this runbook keys off):
`docs/dsv41-restart-survival-plan-2026-10-10.md` (plan, gaps G1–G7, §Live A/B) and
`docs/dsv41-restart-survival-phase0-audit-2026-10-10.md` (phase-0 read-only audit —
the source of the corrected GRACE/shutdown arithmetic and the park-cost figure).
JIT precedent: `docs/jit-model-lifecycle-handoff-2026-06-30.md` + `docs/api.md`
(§Place Instance).

---

## 0. Provenance — the refs this runbook pins

| Role | Ref | Resolves to | Source |
|---|---|---|---|
| **Deploy ref** (to boot) | `deploy/next21-restart` (built by the sibling deploy-construct task) | production tip + cherry-picked G1/G2/G5/G4 fixes | prior-deploy pattern (below) |
| Production tip it is built off | `deploy/next20-vision` tip | `b5f055185` | verify at run time |
| **Rollback ref** | `b5f055185` (`deploy/next20-vision` tip) | the last good (vision) ship | `git rev-parse` |
| **Rollback tag** | `known-good-vision-20261010-174510` | → commit `b5f055185` (annotated; `rev-parse` gives the tag object `f4a3da7bf`) | see §5 |
| Prior-deploy precedent | `deploy/next20-vision` = `b5f055185` = 99e2966ee (next19 production tip) + cherry-picked vision fix | — | prior ship |

**Deploy pattern (unchanged from the vision ship):** *production tip + cherry-picked
fix commits → a dedicated `deploy/<name>` branch → push → boot the branch → test on
it.* For the vision ship the branch was booted with `EXO_TARGET_BRANCH=dsv4-vision-port`;
for this rollout the branch is **`deploy/next21-restart`**. The sibling task builds and
verifies it; **before booting, confirm it exists and record its tip sha** (see step 1.0).

> **TBD (verify, do not assume):** the exact `deploy/next21-restart` tip sha and the
> precise set of cherry-picked commits are produced by the sibling deploy-construct
> task and are unknown as of authoring. `git ls-remote origin refs/heads/deploy/next21-restart`
> at run time is the source of truth; record the sha in the run log.

---

## (a) DEPLOY PLAN

### 1.0 Pre-flight (read-only, on the control laptop)

```bash
cd ~/repos/exo                                   # production checkout — canonical rsync source
git fetch origin
git ls-remote origin refs/heads/deploy/next21-restart        # <-- RECORD THIS SHA
git log --oneline -8 origin/deploy/next21-restart            # eyeball: production tip + G-fixes
```

**GATE A0:** the ref exists, its history reads as "production tip + cherry-picked
restart-survival fixes", and its tip sha is recorded. If the ref is absent or its
lineage is unclear → **STOP**, do not boot.

Also confirm both nodes are idle and the MoE spike is not running on studio2
(residual wired+compressor memory back under the threshold) before proceeding.

### 1.1 Set the production checkout to the deploy ref

The launcher **rsyncs this laptop's `~/repos/exo/` working tree verbatim** to each
studio (`start_cluster.sh` ~L1642; per-node `git reset --hard` was removed
2026-08-15 — the nodes no longer pull from GitHub, they receive the laptop tree).
So the deploy ref must be **checked out on the laptop first**:

```bash
cd ~/repos/exo
git fetch origin
git checkout deploy/next21-restart        # or: git reset --hard origin/deploy/next21-restart
git rev-parse HEAD                          # must == the sha recorded in step 1.0
```

**GATE A1:** laptop `HEAD` == recorded ref sha.

### 1.2 Relaunch via `start_cluster.sh`

```bash
cd ~/repos/exo
EXO_TARGET_BRANCH=deploy/next21-restart ./start_cluster.sh
```

What the launcher does, in order (verified from `start_cluster.sh`, ~L1335–L1877):

1. **Push check** (`~L1352`): verifies the laptop `HEAD` is an ancestor of
   `origin/${EXO_TARGET_BRANCH:-main}`. With `EXO_TARGET_BRANCH=deploy/next21-restart`
   and step 1.1 done, this passes. Non-interactive invocations **exit 1** unless
   pushed or `EXO_ALLOW_UNPUSHED=1` is set — do **not** set that here.
2. **Per node** (`macstudio-m4-1`, `macstudio-m4-2`):
   - `pkill -TERM -f 'python.*exo'` → wait `GRACE` (**15 s**, `scripts/exo_graceful_shutdown.sh:83`) →
     escalate to `kill -9` only if a process survives (`:154-159`). *This is the
     graceful-shutdown path by construction — see §(c) graceful case.*
   - `screen -wipe`; reclaim-curve wait.
   - `rsync -a --delete` from `~/repos/exo/` (excl. `.venv/ __pycache__/ *.pyc
     dashboard/node_modules/ .pytest_cache/ tmp/`).
   - `uv sync --extra mlx --all-packages --inexact --no-install-package mlx`
     → conditional MLX rebuild (skipped if the submodule stamp matches)
     → `uv pip install --no-deps --force-reinstall ./mlx-lm` → maturin `exo_rs`
     → dashboard `npm run build`.
   - Launch: `cd ~/repos/exo && $EXO_ENV … .venv/bin/python -m exo -v >> ~/exo.log 2>&1 &`
     under `screen -dmS exorun`, then `caffeinate -s … wait` (`:3400`, `:3474`).
3. Commit-consistency check across nodes (`:1554+`) — both studios must report the
   same commit; a mismatch aborts.
4. Regenerates `~/relaunch_exo.sh` on each node from the same `EXO_ENV`.

**Node steps in explicit form** (what the script performs; use to verify, not to
shortcut — the script is the supported path):

```bash
# per N in macstudio-m4-1 macstudio-m4-2  (the launcher does this internally):
#   ssh $N "pkill -TERM -f 'python.*exo'; <wait 15s>; <kill -9 survivors>"
#   ssh $N "screen -wipe"
#   rsync -a --delete --exclude '.venv/' ... ~/repos/exo/ $N:~/repos/exo/   # from THIS laptop
#   ssh $N "... build ... && cd ~/repos/exo && .venv/bin/python -m exo -v"
```

> **Note on ordering / R3:** the phase-0 audit (§A8) confirmed the old
> `git reset --hard`-before-signal hazard (**R3**) **does not occur** — the kill
> happens *before* the rsync, and no per-node reset exists anymore. Keep the
> kill→rsync order; do not reorder.

### 1.3 Post-boot bring-up — **the JIT precedent (do not skip)**

**Precedent (vision ship):** after the relaunch, **DSv4 came up with its instance
JIT-UNLOADED** — no resident DSv4 instance — so the next request 404'd/waited until
the operator **explicitly placed the instance with `jit=False`**. Do the same here.

```bash
API="http://<api-host>:52415"        # launcher sets API="http://$API_HOST:52415" (~L3519)

# 1. Confirm cluster + runners
curl -s "$API/state" | jq '{instances: [.instances[].shardAssignments.modelId],
                            nodeMemory: .nodeMemory, runners: [.runners[].taskStatus]}'
# Expect: 2 nodes HEALTHY, runners RunnerReady 2/2.

# 2. Is a DSv4 instance resident? (model id per start_cluster.sh:439)
DSV41="dealignai/DeepSeek-V4.1-Flash-UNCENSORED-EXL3-2.9bpw"
curl -s "$API/state" | jq --arg m "$DSV41" \
  '[.instances[] | select(.shardAssignments.modelId==$m) | {id:.instanceId, jit:.jit}]'
# Empty  -> DSv4 did NOT auto-load. Place it EXPLICITLY (jit=False = pinned/immune).
```

Explicit placement (HTTP `/place_instance` — body schema `PlaceInstanceParams`
`{model_id, sharding, instance_meta, min_nodes}`; **the HTTP API has no `jit` field —
an explicit API placement is *always* `jit=False`/pinned**, per `api/types/api.py:340`
and `shared/types/commands.py:37`; DSv4 is placed Tensor / MlxJaccl / min_nodes=2):

```bash
curl -sS -X POST "$API/place_instance" -H 'content-type: application/json' -d '{
  "model_id": "dealignai/DeepSeek-V4.1-Flash-UNCENSORED-EXL3-2.9bpw",
  "sharding": "Tensor",
  "instance_meta": "MlxJaccl",
  "min_nodes": 2
}'
# /place_instance returns a command_id; instances are stored under a separate
# instance_id — poll /state (or GET /instance/await) until the instance appears and
# runners are READY 2/2, then confirm .jit == false.
```

(Alternatively the launcher's own dashboard-style auto-place at `~L3747-3769` —
`GET /instance/placement` → `POST /instance`, `DSV4_SHARDING=Tensor`,
`DSV4_INSTANCE_META=MlxJaccl`, `min_nodes=2` — normally covers this at boot; the JIT
precedent is precisely the case where it did **not**, so the explicit
`/place_instance` fallback above is the required net.)

**GATE A2:** DSv4 instance resident with `jit=false`, runners READY 2/2, `temp=0`
smoke returns a token. Only then start §(c).

---

## (b) ENVS FOR THE RUN

### 2.1 Primary knob — arm the after-turn idle checkpoint

```bash
EXO_DSV41_CKPT=1        # after-turn idle checkpoint (G1). DEFAULT: OFF (plan §G1 / rollout step 3)
```

`EXO_DSV41_CKPT=1` is the **core treatment** of the A/B: it makes the last
between-turn state (cache + history) durable on the engine thread whenever a
session is idle, so both a graceful restart **and** a crash can restore by
prefix-match. This is the env that must be passed at launch (add to the `EXO_ENV`
passthrough / the `./start_cluster.sh` invocation). Companion knobs and their
defaults (plan §G1):

```bash
# EXO_DSV41_CKPT_MIN_DELTA=2048   # dirty tokens since last commit that qualifies a checkpoint
# EXO_DSV41_CKPT_IDLE_S=20        # idle seconds before a session is checkpointed
```

### 2.2 Decision — `EXO_DSV41_SHUTDOWN_PARK` (park-on-shutdown)

**Decision: leave it DEFAULT-OFF for this run. Do not set it.**

Rationale, with the corrected GRACE/budget arithmetic from the build notes:

- **The plan's premise was wrong and the audit corrected it.** Plan §G3 sized the
  park budget as "≈ 0.7 × supervisor **ExitTimeOut** minus drain". The phase-0 audit
  (§A8) found **there is no launchd/`ExitTimeOut` for exo** — exo runs under
  `screen -dmS exorun` — so the real graceful budget is the shutdown script's
  **`GRACE = 15 s`** (`scripts/exo_graceful_shutdown.sh:83`), i.e. the same 15 s the
  launcher allows before escalating to `kill -9`. Budget arithmetic off the real
  number, not the phantom one:
  - total graceful window = **GRACE = 15 s**
  - minus the in-flight drain = `EXO_DSV41_SHUTDOWN_DRAIN_S` (**default 10 s**) →
  - **park deadline `EXO_DSV41_SHUTDOWN_PARK_S` ≈ 15 − 10 = ~5 s** (plan's "0.7×ExitTimeOut"
    form is retired).
- **What 5 s buys.** Audit §A9 measured a ~92 K-row park at **≈ 471 MiB/session**
  (dominated by `comp_kv.*.bin`), whole store 11 GiB / 44 dirs. A 5 s window at NVMe
  rates covers roughly **one** qualifying session (sub-second–few-second each) —
  **not** a multi-session sweep. So park-on-shutdown can only ever fast-path the
  single most-recent dirty session.
- **It is redundant with the primary mechanism.** `EXO_DSV41_CKPT=1` already persists
  the last between-turn state for *every* idle session (most-recent-first), so the
  session park-on-shutdown would save is *already on disk*. The marginal benefit is
  near-zero.
- **It adds shutdown-time risk.** A park that does not finish inside the 15 s window
  is SIGKILLed mid-write; the invariant (temp dir discarded, previous generation
  intact) holds, but it is avoidable risk for no coverage gain.

```bash
# EXO_DSV41_SHUTDOWN_DRAIN_S=10   # default (plan §G3)
# EXO_DSV41_SHUTDOWN_PARK_S=...   # default ≈ GRACE − drain ≈ 5 s; PARK-ON-SHUTDOWN DEFAULT-OFF
```

Revisit only if an instrumented A9 shows the multi-GiB/s headroom to sweep several
sessions inside the window — at which point it is a latency fast-path on top of CKPT,
never a substitute for it.

### 2.3 Other knobs the operator should know are armed (defaults)

| Env | Default | Meaning |
|---|---|---|
| `EXO_DSV41_PARK` | **ON** (`session.py:73`, `park.py:100`) | SSD parking of idle conversations. On by default; `=0` disables. |
| `EXO_DSV41_PARK_DIR` | `~/.exo/dsv41_park` | Park store root (scanned at init; survives restarts). |
| `EXO_DSV41_PARK_MAX_GB` | `12.0` | LRU byte budget for the store. |
| `EXO_DSV41_PARK_MIN_TOKENS` | `4096` | Sessions shorter than this are evicted, not parked. |
| `EXO_DSV41_PARK_MAX_COUNT` | `64` | Count cap beside the byte budget. |
| `EXO_DSV41_CKPT` | **OFF** | After-turn idle checkpoint — **set to `1` for this run**. |
| `EXO_DSV41_SHUTDOWN_PARK` | **OFF** | Park-on-shutdown — **left OFF** (§2.2). |
| `EXO_DSV41_SHUTDOWN_DRAIN_S` | `10` | Drain window for an in-flight turn. |
| `EXO_JIT_ENABLED` | `1` (`start_cluster.sh:957`) | JIT auto load/unload. On by default since `23fb58fc`. |
| `DSV4_SHARDING` / `DSV4_INSTANCE_META` | `Tensor` / `MlxJaccl` | DSv4 placement (min_nodes 2). |
| `DSV4_MODEL_ID` | `dealignai/DeepSeek-V4.1-Flash-UNCENSORED-EXL3-2.9bpw` (`:439`) | DSv4 model id. |

> **TBD (verify at boot):** after the sibling's deploy ref is built, confirm the
> G2/G5 fingerprint knobs (and any renamed CKPT/SHUTDOWN knobs) are exactly as the
> plan names them by grepping the **deployed** `start_cluster.sh` env block and the
> runner process env (`ps eww`). Names/defaults not yet visible in `origin/main`
> source at authoring are marked here from the plan/build notes; treat the deployed
> `start_cluster.sh` as authoritative.

**Launch line (treatment):**

```bash
cd ~/repos/exo
EXO_TARGET_BRANCH=deploy/next21-restart EXO_DSV41_CKPT=1 ./start_cluster.sh
# (EXO_DSV41_SHUTDOWN_PARK intentionally NOT set — arming it is a deliberate opt-in only.)
```

---

## (c) A/B PROTOCOL

**Fixture:** one real ~90 K-token DSv4 conversation, `temp 0`, greedy. Needles at
**5 depths: 1K / 20K / 45K / 70K / 88K** (plan §Live A/B). Measure **next-turn TTFT**
after the restart for a **~2 K-token delta** appended to the conversation.

**Control (reference):** the plan's number is **≈ 282 s** for the ~92.6 K cold
re-prefill. **Correction to carry in the A/B:** the phase-0 audit (§A3) could not
isolate that exact event in surviving logs; the **closest logged cold is 91,043 rows
≈ 379 s (~240 t/s)** and the 282 s figure is only magnitude-consistent. So the
control bar is stated as **282 s (plan) / 379 s (closest logged proxy)** — do not
treat 282 s as a measured on-box number.

### 3.1 Two shutdown cases (both required)

- **Graceful case** — clean shutdown: the launcher's SIGTERM → ≤15 s GRACE → SIGKILL
  path (§1.2). Exercises park/restore on an orderly teardown.
- **Crash case** — **`kill -9`** on both runners *after* the idle checkpoint has
  committed (wait ≥ `EXO_DSV41_CKPT_IDLE_S` = 20 s idle so `maybe_checkpoint` fires;
  confirm the checkpoint in `~/exo.log`). Exercises crash atomicity: the last
  generation must restore cleanly.

Then relaunch (§1.2), run bring-up + JIT fallback (§1.3), and measure.

### 3.2 Measurements per case

1. **Next-turn TTFT** on the ~2 K delta → expect `restore` log line + TTFT
   ≈ restore + delta + 5 s; **target < 15 s**.
2. **Needles at 1K/20K/45K/70K/88K** all recovered post-restore.
3. **Top-1 token agreement ≥ 99 %** over **256 greedy tokens** vs the fresh-cold
   control; logit agreement vs never-restarted resident exact (or < 1e-5 kernel
   noise); mean KL < 1e-3; greedy 256 tokens identical to the resident control.
4. **DSpark acceptance within ±2%** of the pre-restart resident baseline.
5. **Exact-repeat prompt** uses the persisted anchor and matches.
6. **No silent staleness:** any cross-build/foreign-fingerprint dir is **refused**
   with `park.skip reason=fp_mismatch`, never served (audit §A5 already observed
   12/15 cross-build restores with no guard — this bar exists to prove that is closed).

### 3.3 Mid-turn case (best-effort, not a hard bar)

Restart **during decode** → restore of the previous generation; retry re-prefills
only the last turn (lose only the in-flight turn).

### 3.4 Pre-registered bars

**SUCCESS — all must hold (both graceful and crash cases):**
1. Next-turn TTFT **≤ 15 s** for the ~2 K delta on the ~92 K conversation, logged as a
   restore (not a cold prefill). *(vs 282 s plan control / 379 s closest-logged.)*
2. Needles recovered at **all 5 depths** (1K/20K/45K/70K/88K).
3. **Top-1 agreement ≥ 99 %** over 256 greedy tokens; greedy-256 identical to
   resident control.
4. **DSpark acceptance within ±2 %** of the resident baseline.
5. Exact-repeat anchor match.
6. Cross-build/foreign dirs **refused** (`fp_mismatch`), nothing stale served.

**ABORT — any one → stop, roll back (§5), report:**
1. Next-turn TTFT **> 60 s** on the treatment (≈ 4× target) — restore did not fire.
2. Needle miss at any depth, **or** top-1 agreement **< 95 %**, **or** DSpark
   acceptance outside **±5 %**.
3. Any **collective hang / rank skew / runner bootstrap re-load** on restart
   (the TP2 half-restore failure mode, plan §G4/audit §A7).
4. `fp_mismatch` **falsely refusing a same-build dir** (allowlist false-positive — R1),
   **or** a foreign dir **served** (stale).
5. A `kill -9` left a half-written generation that then **restored** (corruption).
6. Cluster not HEALTHY / runners not **READY 2/2** within the boot window.

**WARN band** (between success and abort): TTFT 15–60 s, top-1 95–99 %, DSpark ±2–5 % —
treat as a soft fail: do not promote, capture evidence, leave `EXO_DSV41_CKPT=1` for
observation or roll back at operator discretion.

---

## (d) ROLLBACK

**Rollback ref:** `b5f055185` = the last good (vision) ship = `deploy/next20-vision` tip.
**Rollback tag:** `known-good-vision-20261010-174510` (annotated → commit `b5f055185`).

Trigger: any ABORT bar (§3.4), or an unrecoverable post-boot state.

```bash
# 1. Point the production checkout back at the last good ref
cd ~/repos/exo
git fetch origin
git checkout known-good-vision-20261010-174510     # or: git reset --hard b5f055185
git rev-parse HEAD                                  # must print b5f055185

# 2. Relaunch the last good build (same launcher path)
EXO_TARGET_BRANCH=deploy/next20-vision ./start_cluster.sh
#    (the tag resolves to b5f055185, which is origin/deploy/next20-vision's tip)

# 3. Bring-up + JIT fallback exactly as §1.3 (DSv4 did not auto-load on the vision
#    ship either — POST /place_instance jit=False again), then confirm READY 2/2.

# 4. Do NOT delete the park store; leave ~/.exo/dsv41_park intact (foreign-fingerprint
#    dirs are refused, not consumed). Do NOT force-push any branch.
```

**Never force-push.** Roll back by checking out the known-good ref and relaunching —
do not rewrite `main` or `deploy/*`.

> **Note:** `deploy/next21-restart` is left in place (do not delete the branch); it is
> harmless once the production checkout no longer points at it.

---

## (e) BOOT-BUDGET NOTE

**1 of 2 boots used — the reserve is held.** This live phase consumes **one** cluster
boot (the §1.2 relaunch). A **second boot is held in reserve** and is **not** spent
here: it is reserved for the rollback relaunch (§5) if an ABORT bar fires, and beyond
that only on explicit operator go. Do not spend the reserve boot on re-runs, tuning
sweeps, or extra A/B arms. The crash case in §3.1 is a `kill -9` + one relaunch — it
*does* draw down the same window, so sequence the graceful and crash cases into the
single boot budget deliberately: one treatment boot + one rollback boot = 2.

---

## 6. Cross-check table — every number, with its source

| Number | Value used here | Source / status |
|---|---|---|
| Cold re-prefill control | **282 s** (plan) / **379 s** (closest logged, 91,043 rows ≈ 240 t/s) | plan §Live A/B; **audit §A3 correction** |
| Active conversation size | ~92.6 K tokens (plan) | plan; audit: exact event not in surviving logs |
| TTFT target | **≤ 15 s** for ~2 K delta | plan §Live A/B |
| Needle depths | 1K/20K/45K/70K/88K (5) | plan §Live A/B |
| Top-1 agreement bar | **≥ 99 %** over 256 greedy tokens | plan §Live A/B |
| Mean KL | < 1e-3 | plan §Live A/B |
| DSpark acceptance | **±2 %** vs resident | plan §Live A/B |
| GRACE (real shutdown budget) | **15 s** | `exo_graceful_shutdown.sh:83`; audit §A8 |
| `EXO_DSV41_SHUTDOWN_DRAIN_S` | 10 s default | plan §G3 |
| `EXO_DSV41_SHUTDOWN_PARK_S` | ≈ 15 − 10 ≈ 5 s (plan's "0.7×ExitTimeOut" **retired**) | plan §G3 + **audit §A8 correction** |
| Park cost per session | **≈ 471 MiB** (~92 K rows) | audit §A9 (measured) |
| Store size | 11 GiB / 44 dirs | audit §A5/A9 |
| `EXO_DSV41_CKPT_MIN_DELTA` | 2048 default | plan §G1 |
| `EXO_DSV41_CKPT_IDLE_S` | 20 s default | plan §G1 |
| `EXO_DSV41_PARK` / `_DIR` / `_MAX_GB` / `_MIN_TOKENS` / `_MAX_COUNT` | ON / `~/.exo/dsv41_park` / 12.0 / 4096 / 64 | **source** (`session.py:73`, `park.py:100-113`) |
| DSv4 model id / sharding / meta / min_nodes | `dealignai/DeepSeek-V4.1-Flash-UNCENSORED-EXL3-2.9bpw` / Tensor / MlxJaccl / 2 | `start_cluster.sh:439`, `:3758` |
| API base | `http://<api-host>:52415` | `start_cluster.sh:3519` |

**Marked TBD (could not source read-only; verify, do not invent):**
- `deploy/next21-restart` tip sha and exact cherry-picked commits (sibling task output).
- Exact deployed names/defaults of the G2/G5 fingerprint + G1/G3 env knobs (grep the
  deployed `start_cluster.sh` / runner `ps eww` before boot).
- On-box DSpark acceptance baseline (read it from the pre-restart resident `/state`
  or the known-good baseline before the restart; the bar is *relative* ±2 %).
- Exact A9 park/restore throughput (audit marks NEEDS-GO; no timing instrumentation).
