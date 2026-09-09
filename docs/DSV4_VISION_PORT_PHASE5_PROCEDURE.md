# DSv4-Flash-Vision-Exp — Phase 5 Procedure (relaunch + on-hardware smoke test)

**Status:** authored 2026-09-09 by the Phase 5 dispatch. Phases 0–4 complete and
double-verified. Phase 5 = the first time the 167 GB Vision-Exp checkpoint runs
through the real serving path on real hardware.

**Authorization:** the `HARD BOUNDARIES` section of
`docs/DSV4_VISION_PORT_PLAN_PHASE34.md` boundary 1 ("stop after Phase 4, do not
run Phase 5") and — necessarily, since they are the same prohibition — the
relaunch clause of boundary 2 have been explicitly lifted by Adam Durham for
this dispatch. Boundaries 3–8 (no hand-edits on the Studios, don't touch the
192.168.20x RDMA net, branch `dsv4-vision-port` only / never `main` / never the
`ml-explore` remote, leave `tmp/` alone, root-cause fixes only, numbers with
every claim) remain **in force**.

---

## 0. Root cause of the model-catalog visibility gap (resolved before any cluster action)

**Symptom.** `GET /v1/models` on the live cluster returns 125 models and does
not include `deepseek-ai/DeepSeek-V4-Flash-Vision-Exp`, even though the
checkpoint is fully downloaded on both nodes at
`~/.exo/models/deepseek-ai--DeepSeek-V4-Flash-Vision-Exp/` with a valid
`model.safetensors.index.json`.

**Root cause — deployment lag, not a code defect.** The catalog is built by
`_CardCache.refresh()` (`src/exo/shared/models/model_cards.py:89-92`), which
`rglob("*.toml")`s `RESOURCES_DIR/inference_model_cards` **from the running
process's own checkout**. It never consults the model directory, so an
on-disk checkpoint with no card is invisible by design.

The card exists only on the branch:

```
$ git cat-file -e HEAD:resources/inference_model_cards/deepseek-ai--DeepSeek-V4-Flash-Vision-Exp.toml
PRESENT on dsv4-vision-port HEAD
$ git cat-file -e origin/main:resources/inference_model_cards/deepseek-ai--DeepSeek-V4-Flash-Vision-Exp.toml
fatal: Not a valid object name ...          <-- ABSENT on origin/main
```

and both Studios are running `main`, deployed 2026-09-07 18:21, uptime 1d11h:

```
macstudio-m4-1: HEAD 11bf2e29c8135dd386cccdbd41a1a65b48f8d35f  branch=main
macstudio-m4-2: HEAD 11bf2e29c8135dd386cccdbd41a1a65b48f8d35f  branch=main
   mlx     e40a416b20851d118b061b3a57d8cab70f5756de
   mlx-lm  7f146542811dd774d2cb3ab38b13c4b25a8e8063   (branch pins 513d6db)
```

**Resolution: no code fix is required.** Deploying `dsv4-vision-port` to both
nodes ships the card and the catalog picks it up on process start. This is
*informative for how relaunch must work*: the relaunch is not merely a restart,
it is a **branch deployment**, and the pre-flight must prove the card is
present in the deployed tree on both nodes before the instance is placed.

**Consequence for placement:** the placement gate never reads download status
for admit/deny — `filter_cycles_by_memory` (`src/exo/master/placement.py:131`,
`placement_utils.py:253-269`) compares `sum(ram_available)` against
`model_card.storage_size`; download fraction only ranks already-admissible
cycles (`placement.py:264-279`). So "downloaded but uncatalogued" could never
have launched, and "catalogued" is sufficient.

---

## 1. Pre-flight checklist

Every item must be **verified with pasted output**, not assumed.

### 1.1 What is live right now (a relaunch interrupts this)

| item | value |
|---|---|
| instance | `c8a841d9-83bd-492a-8a19-f0982827a49e` (`MlxJacclInstance`) |
| model | `deepseek-ai/DeepSeek-V4-Flash-0731` |
| sharding | Tensor, `world_size=2`, JACCL/RDMA, ranks 0/1 across m4-1/m4-2 |
| runners | both `RunnerReady` |
| exo pids | m4-1 `4000`, m4-2 `3294`, started 2026-09-07 18:21, `screen -S exorun` |
| **real traffic** | **none for ~34 h** — last `POST /v1/chat/completions` was `2026-09-07 20:05:03`; the whole current boot logged 21 requests, all on 09-07 |

The "serving real traffic" risk is therefore **idle-cluster risk**, not
mid-request risk. Confirm this again immediately before the kill; if a request
has arrived in the interim, wait for it to drain.

### 1.2 Catalog visibility (§0)

- `resources/inference_model_cards/deepseek-ai--DeepSeek-V4-Flash-Vision-Exp.toml`
  exists on `dsv4-vision-port`, parses as TOML, and validates as a `ModelCard`.
- After deploy, `GET /v1/models` must list `deepseek-ai/DeepSeek-V4-Flash-Vision-Exp`.
  **If it does not, ABORT before placing anything.**

### 1.3 Checkpoint integrity (both nodes)

`config.json` deltas vs `-0731`, read off the real checkpoint:

```
num_nextn_predict_layers: 3        (-0731: 1)
rms_norm_eps: 1e-20                (-0731: 1e-06)
vision_* : 10 TOP-LEVEL keys, no vision_config sub-dict, no image_token_id
architectures: ['DeepseekV4ForCausalLM']
index.json total_size: 167811372792   (-0731: 166878536440)
57 files present, no .incomplete/.tmp/.part
```

### 1.4 Memory budget — the real risk, quantified

Actual per-tensor byte accounting from the safetensors headers of both
checkpoints (`/tmp/mtpscan.py`, run on m4-1):

| group | Vision-Exp GiB | -0731 GiB |
|---|---:|---:|
| backbone `model.layers.*` | 135.265 | 135.265 |
| `embed` + `head` | 1.972 | 1.972 |
| `mtp.0` | 3.175 | 3.175 |
| `mtp.1` | 3.128 | 3.128 |
| `mtp.2` | 3.252 | 3.252 |
| `vision.*` (259 tensors, BF16) | 0.767 | 0.000 |
| `aligner.*` (4 tensors, BF16) | 0.102 | 0.000 |
| **total on disk** | **147.661** | **146.792** |

Both checkpoints ship 3 MTP stages on disk, but `Model.sanitize()`
(`mlx-lm/mlx_lm/models/deepseek_v4.py:8196-8220`) keeps
`mtp.{idx}` only for `idx < num_nextn_predict_layers`. So `-0731` (nextn=1)
loads `mtp.0` alone, while Vision-Exp (nextn=3) would load **all three**, and
`auto_parallel.py:1082-1199` shards every one of them. Only `mtp[0]` is ever
used at decode (`speculative/dsv4_mtp.py:1050`, callers pass `mtp_idx=0`), so
stages 1–2 would be pure waste.

Loaded footprint per node (backbone/MTP sharded at TP=2; vision tower is
**replicated** per the Phase 4 TP work, so it counts in full on each node):

| config | GiB/node |
|---|---:|
| `-0731` as running today (nextn=1, MTP on) | **70.206** |
| Vision-Exp, production spec env (nextn=3, MTP on) | **74.265** (+4.06) |
| **Vision-Exp, spec-off (`EXO_DSV4_MTP=0`)** | **69.488** (−0.72) |

`iogpu.wired_limit_mb=115000` = 112.3 GiB. The measured `-0731` DSpark+MTP
baseline peaks at ~92–99 GB/node under load
(`docs/dspark-mtp-production-baseline-2026-08-27.md`). +4.06 GiB/node is not a
guaranteed OOM but it narrows an already-tight ceiling on the *first ever* run
of this checkpoint.

**Decision: launch Vision-Exp spec-off** —
`EXO_SPECULATIVE=0 EXO_DSV4_MTP=0 EXO_DSV4_DSPARK=0`. Rationale:

1. It makes the loaded footprint **smaller than the known-good baseline**
   (69.5 vs 70.2 GiB/node), so memory is strictly less risky than today.
2. `EXO_DSV4_MTP=0` prevents the *load*, not just the use
   (`deepseek_v4.py:7467-7469` skips constructing `self.mtp`; `sanitize()` sets
   `n_mtp=0`), so the 6.4 GiB of dead stages never materialize.
3. It removes speculative decoding as a confound. Phase 5 is a
   **vision-correctness** test, not a throughput test.
4. It is a *previously exercised* production configuration — the exact
   spec-off config used as the rollback state on 2026-08-26
   (`docs/PERFORMANCE_HISTORY.md`).

There is no per-checkpoint MTP-stage cap in the codebase; adding one would be a
code change outside Phase 5's scope. Spec-off is the correct minimal lever.

### 1.5 Placement arithmetic

`filter_cycles_by_memory` requires `sum(ram_available) >= storage_size`.
Right now, with `-0731` resident:

```
nodeMemory ramAvailable: m4-1 34,832,793,600   m4-2 34,809,741,312   (sum 69.6 GB)
storage_size(Vision-Exp) = 167,811,372,792
```

69.6 GB < 167.8 GB — placement **would fail today**. It only succeeds after the
old runners are killed and memory is reclaimed (each node has
`ramTotal = 137,438,953,472`, so ~250 GB combined once free). This is why
placement must happen *after* the kill+reclaim, which is exactly what
`start_cluster.sh` does. Do not attempt to place Vision-Exp alongside the
running `-0731` instance.

### 1.6 Disk headroom

`/System/Volumes/Data`: m4-1 93 Gi free, m4-2 127 Gi free. The relaunch needs
**no new download** (checkpoint present) and no large scratch: the only writes
are the rsync delta, a `uv pip install` of two local packages, a maturin
release build, and an `npm run build`. Log rotation truncates `~/exo.log`
(164 MB on m4-1, 125 MB on m4-2) into `exo.log.prev`, a net wash. Acceptable.

### 1.7 Toolchain hazard (must not be tripped)

**Neither Studio has a Metal compiler** — Xcode.app was removed; only
CommandLineTools remains:

```
xcrun -f metal  ->  xcrun: error: unable to find utility "metal"
xcode-select -p ->  /Library/Developer/CommandLineTools
```

`start_cluster.sh:1462-1508` hard-fails the deploy if an MLX rebuild is needed
without a Metal toolchain. The rebuild is skipped only when all of: mlx SHA
matches the venv stamp, the mlx submodule tree is clean, `import mlx` works,
and the installed dist's `direct_url.json` points at the node's own `./mlx`.
Verified on both nodes:

```
stamp .venv/.mlx-installed-sha = e40a416b20851d118b061b3a57d8cab70f5756de
git -C mlx rev-parse HEAD      = e40a416b20851d118b061b3a57d8cab70f5756de
mlx worktree/index             = CLEAN
direct_url.json                = {"url":"file:///Users/adam.durham/repos/exo/mlx","dir_info":{}}
```

`origin/main:mlx` and `dsv4-vision-port:mlx` are the **same commit**
(`e40a416b`), so switching branches does not move the mlx pin and the skip
still fires. **Only `mlx-lm` moves** (`7f14654` → `513d6db`), and mlx-lm is a
pure-Python force-reinstall — no compiler needed.

> **Therefore: the control host for the rsync MUST have a populated
> `mlx/build/` (1.0 GB, 20 entries).** `start_cluster.sh` rsyncs with
> `--delete` and does *not* exclude `mlx/build`. Syncing from a tree with an
> empty `mlx/build` would delete the FetchContent dep cache on both nodes.
> That would not break *this* deploy (the skip still fires) but would leave the
> cluster unable to ever rebuild MLX offline. The sandbox checkout has
> `mlx/build` empty (0 entries) and also lacks `jq`, which `start_cluster.sh`
> requires. **The sandbox must not be the rsync source.**

### 1.8 Control host

`start_cluster.sh` must run from a host that has: `jq`, `rsync`, ssh aliases
`macstudio-m4-1`/`macstudio-m4-2`, and a canonical `$HOME/repos/exo` on the
target branch **with the mlx build cache intact**. The historical control host
(the MacBook) is not on this branch. The only host meeting all conditions is
**macstudio-m4-1 itself** — it has `jq`, `rsync`, `screen`, `uv`, key-auth SSH
to both nodes *including itself*, and the 1.0 GB `mlx/build`. Running on a node
is a designed mode (`start_cluster.sh:851-856` has an explicit "Detected M4-1"
branch); the m4-1→m4-1 leg becomes a self-rsync of identical content, which
transfers nothing and deletes nothing.

Preparing m4-1's checkout is a **git-tracked branch switch**, not a hand-edit,
so boundary 3 is respected: no file content originates on the Studio.

### 1.9 sudo

`sudo -n` fails on both nodes (no cached credential, no TTY). `start_cluster.sh`
uses sudo in exactly two places: `sysctl iogpu.wired_limit_mb` and
`xcode-select -s`. Both are already at their desired values
(`iogpu.wired_limit_mb: 115000`; xcode-select correctly on CommandLineTools),
the script has **no `set -e`**, and neither call's exit status is checked — so
both degrade to a printed sudo error and the deploy proceeds. This is expected
and benign. It must not be "fixed" by trying to obtain sudo.

---

## 2. Relaunch procedure

Read from `start_cluster.sh` directly; behaviour confirmed, not remembered.

**What the script actually does, in order:**

1. Host self-detection by IP (`:830-865`) — on m4-1 it prints `Detected M4-1`.
2. Thunderbolt link discovery + direct-link ping, with a passive route repair
   on failure (`:876-1035`). **Aborts** if the direct link is dead.
3. Per-device RDMA port state check (`:1039-1060`).
4. Push check (`:1119-1166`): compares local HEAD against
   `origin/${EXO_TARGET_BRANCH:-main}`. With **no TTY** it *exits 1* unless
   HEAD is an ancestor of that branch or `EXO_ALLOW_UNPUSHED=1`. → we set
   `EXO_TARGET_BRANCH=dsv4-vision-port`, whose HEAD is already pushed
   (`origin/dsv4-vision-port = 7553b36a`), so this passes cleanly.
5. Per node (`:1214+`): `sudo sysctl` wired limit (fails benignly, §1.9) →
   **`pkill -TERM -f 'python.*exo'`, wait up to 15 s, escalate to `-9` only if
   still alive** → `screen -wipe` → reclaim-curve wait → `rsync -a --delete`
   from `$HOME/repos/exo/` excluding `.venv/ __pycache__/ *.pyc
   dashboard/node_modules/ .pytest_cache/ tmp/` → `uv sync --extra mlx
   --all-packages --inexact --no-install-package mlx` → conditional MLX
   rebuild (**skipped**, §1.7) → `uv pip install --no-deps --force-reinstall
   ./mlx-lm` → maturin release build of `exo_rs` → `npm run build` dashboard.
6. Commit-consistency check across nodes (`:1554+`).
7. Regenerates `~/relaunch_exo.sh` on each node from the same `EXO_ENV`, then
   `screen -dmS exorun zsh -l -c '<EXO_ENV> .venv/bin/python -m exo -v >> ~/exo.log'`.
8. Health gate: polls `/state` for `topology.nodes >= 2` **and**
   `nodeIdentities >= 2`, 90 × 2 s = **180 s**, else dumps the log and exits 1.
9. Instance placement: two-step `GET /instance/placement` then `POST /instance`
   for `$DSV4_MODEL_ID` (30 attempts × 5 s), then waits for 2 runners Ready
   (600 × 2 s = **1200 s**). Then the same for Qwen3.6 unless `QWEN36_ENABLED=0`.

**Answer to "does relaunch alone select a model?"** — No, and yes: relaunching
`exo` alone selects nothing (an exo node with no instance serves no model).
`start_cluster.sh` *additionally* places instances via the REST API, and which
model it places is controlled by `DSV4_MODEL_ID` (default
`deepseek-ai/DeepSeek-V4-Flash-0731`, `:379`). The dashboard is not required;
the operator selects the model by setting that variable. The same two-step
`GET /instance/placement` → `POST /instance` flow is what the dashboard uses.

### 2.1 Exact commands

**Step A — put m4-1's canonical checkout on the branch** (git-tracked only):

```bash
ssh macstudio-m4-1 'cd ~/repos/exo && git fetch origin && \
  git checkout -B dsv4-vision-port origin/dsv4-vision-port && \
  git submodule update --init --recursive && \
  git rev-parse HEAD && git submodule status'
```

Expected: HEAD `7553b36a…`, `mlx e40a416b…` (unchanged), `mlx-lm 513d6db…`.

**Step B — verify the card is in the deployed tree** (pre-flight §1.2):

```bash
ssh macstudio-m4-1 'ls -la ~/repos/exo/resources/inference_model_cards/ | grep -i vision'
```

**Step C — launch.** Run detached on m4-1 so it survives SSH teardown:

```bash
ssh macstudio-m4-1 "cd ~/repos/exo && \
  EXO_TARGET_BRANCH=dsv4-vision-port \
  DSV4_MODEL_ID=deepseek-ai/DeepSeek-V4-Flash-Vision-Exp \
  DSV4_SHARDING=Tensor \
  QWEN36_ENABLED=0 \
  EXO_SPECULATIVE=0 EXO_DSV4_MTP=0 EXO_DSV4_DSPARK=0 \
  nohup ./start_cluster.sh > /tmp/phase5_launch.log 2>&1 &"
```

Then poll `/tmp/phase5_launch.log`.

---

## 3. Smoke test

### 3.1 Request shape (traced from source, not guessed)

`src/exo/api/types/api.py:78-83` defines
`ChatCompletionMessageImageUrl{type:"image_url", image_url: dict[str,str]}` —
OpenAI's standard form. `chat_completions.py:82-103` fetches `http(s)://` URLs
or strips a `data:...;base64,` prefix, producing `Base64Image` on
`TextGenerationTaskParams.images`. The worker builds a `VisionProcessor` when
`model_card.vision is not None` (`utils_mlx.py:308-328`), and
`vision.py:738/760` routes `scheme == "deepseek_v4"` into
`deepseek_v4_vision.process()`.

**The caller must NOT include the literal `<|deepseek_image|>` placeholder.**
`deepseek_v4_encoding.py:770-802` substitutes each image content block with the
placeholder server-side, and `_validate_no_image_sp_tokens`
(`:805-817`) raises if the client text already contains one.

### 3.2 The critical trap: vision failures are silent

`generate.py:2079` and `batch_generate.py:2221` wrap vision processing in
`except Exception: logger.warning("Vision processing failed, falling back to
text-only")`. A broken vision path therefore returns **HTTP 200 with a
plausible text-only answer**, not an error.

**Consequence: "the model responded" is NOT a pass.** The pass criteria below
are built to defeat this failure mode, and the server log must be checked for
that warning on every attempt.

### 3.3 Test image

Generate a synthetic image whose content cannot be guessed from the prompt: a
solid-colour canvas with a large printed word and a digit count. The model must
report **both** facts. Using a generated image (not a famous photo) removes any
chance of a text-only model bluffing correctly from priors.

### 3.4 Request

```bash
curl -sS -X POST http://192.168.86.201:52415/v1/chat/completions \
  -H 'Content-Type: application/json' \
  -d '{
    "model": "deepseek-ai/DeepSeek-V4-Flash-Vision-Exp",
    "max_tokens": 200,
    "temperature": 0,
    "messages": [{"role":"user","content":[
      {"type":"text","text":"Look at this image. What word is written on it, and how many circles are below the word? Answer plainly."},
      {"type":"image_url","image_url":{"url":"data:image/png;base64,<PAYLOAD>"}}
    ]}]
  }'
```

### 3.5 Pass / fail criteria

**PASS requires all of:**

- P1. HTTP 200 with a non-empty `choices[0].message.content`.
- P2. The response states the **exact word** rendered in the image.
- P3. The response states the **correct count** of shapes.
- P4. Server log for the request shows **no** `Vision processing failed,
  falling back to text-only`.
- P5. Server log shows the prompt token count is inflated by the image sentinel
  block (an image expands to hundreds of tokens), i.e. prefill token count is
  far larger than the ~40-token text alone — positive evidence the image
  entered the prompt.
- P6. A **control run** with the identical prompt and **no image** must fail to
  produce the word/count (it cannot know them). This proves P2/P3 came from the
  image and not from the prompt.

**FAIL (→ rollback) on any of:** HTTP 5xx; empty/garbage output (repetition
loops, wrong-language spew); the fallback warning in the log; a runner crash;
or P2/P3 wrong while P6 shows the model *could* have known.

**Ambiguous (→ investigate, do not declare success):** correct word but wrong
count, or a hedged "I can't see an image" — that is a functional failure of the
vision path even at HTTP 200.

---

## 4. Rollback plan

**Trigger:** any FAIL in §3.5, or any abort in §5.

**Target state:** `main` @ `11bf2e29c` deployed on both nodes, one
`MlxJacclInstance` serving `deepseek-ai/DeepSeek-V4-Flash-0731`, Tensor/JACCL,
both runners `RunnerReady` — i.e. byte-for-byte the configuration running
before Phase 5 started, including the production speculative env (the defaults
`EXO_SPECULATIVE=1 EXO_DSV4_MTP=1 EXO_DSV4_DSPARK=1` restore themselves once
the overrides are dropped).

```bash
# R1. restore the canonical checkout on the control host
ssh macstudio-m4-1 'cd ~/repos/exo && git checkout main && \
  git submodule update --init --recursive && git rev-parse HEAD && git submodule status'
# expect 11bf2e29c…, mlx e40a416b…, mlx-lm 7f14654…

# R2. redeploy + replace the known-good instance (default DSV4_MODEL_ID, default spec env)
ssh macstudio-m4-1 "cd ~/repos/exo && EXO_TARGET_BRANCH=main QWEN36_ENABLED=0 \
  nohup ./start_cluster.sh > /tmp/phase5_rollback.log 2>&1 &"

# R3. prove it
curl -s http://192.168.86.201:52415/state   # instance modelId == ...-0731, 2x RunnerReady
curl -s http://192.168.86.201:52415/v1/chat/completions -H 'Content-Type: application/json' \
  -d '{"model":"deepseek-ai/DeepSeek-V4-Flash-0731","max_tokens":16,"temperature":0,
       "messages":[{"role":"user","content":"Say hello."}]}'
```

**Rollback validity is established, not assumed.** R2 is the *same script, same
mechanism* that Phase 5 itself uses; if the script can deploy the branch it can
deploy `main`. The rollback is additionally proven by the fact that the current
live cluster was itself brought up this way. The rollback is considered
**verified only when R3 returns a coherent completion** — a `RunnerReady` pair
alone is not proof of serving.

**Escalation if rollback itself fails:**
- Runners won't reach Ready → check `~/exo.log` on both nodes; if a node has
  stuck wired memory (`vm_stat` wired+compressor > 25 GB with no owning
  process), the documented escape hatch is a node reboot
  (`start_cluster.sh:1200-1210`); `./reboot-node.sh` exists for this.
- Thunderbolt/RDMA wedge after a `-9` escalation → reboot the affected node.
  Never bounce the TB interface MTU (`start_cluster.sh:963-971`).
- If neither works, stop and report with the cluster left with no instance
  placed rather than a half-loaded one.

---

## 5. Timeout / abort criteria

Bounded waits, so nothing hangs indefinitely unattended:

| stage | limit | action on breach |
|---|---|---|
| pre-flight checks | — | any failed check → **abort before touching the cluster** |
| SIGTERM → exo exit | 15 s (script) | script escalates to `-9`; note RDMA-leak risk |
| memory reclaim after kill | 180 s (script) | script warns and proceeds; if a node still shows > 25 GB wired+compressor, **abort and reboot that node** |
| rsync + deps + maturin + dashboard, per node | **20 min/node** | abort → rollback |
| cluster stabilize (2 nodes + 2 identities) | 180 s (script) | script exits 1 → rollback |
| placement HTTP accepted | 150 s (30 × 5 s, script) | script errors → rollback |
| 2 runners `RunnerReady` | 1200 s (script); **hard cap 20 min** | rollback. Reference: `-0731` loads in ~30 s and warms in ~38 s total, so anything past ~5 min already indicates a problem |
| smoke-test HTTP response | **300 s** per attempt | treat as FAIL → rollback |
| whole Phase 5 window | **90 min** from first kill | if not serving *something* known-good by then, execute rollback regardless of progress |

**Standing rule:** the session may not end with the cluster non-serving. If
Vision-Exp fails for any reason, `-0731` must be back up and answering a real
completion before stopping.

---

## 6. Execution log

Filled in during execution — see §7 (appended after the run).
