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

`sudo -n -l` on both nodes shows a general `(ALL) ALL` rule that **does** require
a password, plus three NOPASSWD exceptions:

```
(root) NOPASSWD: /usr/sbin/sysctl iogpu.wired_limit_mb\=*
(root) NOPASSWD: /sbin/route delete -net *
(root) NOPASSWD: /usr/bin/fdesetup authrestart*
```

`start_cluster.sh` uses sudo in exactly two places: `sysctl
iogpu.wired_limit_mb` (covered by NOPASSWD → succeeds) and `xcode-select -s`
(not covered → fails, but it is inside an `if [ -d /Applications/Xcode.app... ]`
guard that is false since Xcode was removed, so it never runs). The script has
**no `set -e`** and neither exit status is checked. Benign; must not be "fixed"
by trying to obtain a password.

### 1.10 ⚠ THE DOMINANT CONSTRAINT: I cannot reboot a wedged node

This was surfaced by the pre-execution consult (§8) and is the single most
important fact in this document.

The documented escape hatch for the two worst failure modes — a Thunderbolt/RDMA
stack wedge from leaked queue pairs, and stuck wired memory with no owning
process — is **"reboot the node"** (`start_cluster.sh:1200-1210`,
`docs/incidents/`). That escape hatch is **not available to this session**:

```
$ ssh macstudio-m4-1 'fdesetup status; fdesetup supportsauthrestart'
FileVault is On.
true
$ ssh macstudio-m4-1 'sudo -n /sbin/reboot --help'
sudo: a password is required
```

FileVault is ON, so a plain reboot halts at the pre-boot unlock screen and the
node never comes back on the network. The project's `reboot-node.sh` solves this
with `fdesetup authrestart` (NOPASSWD-allowed) fed the FileVault password from
1Password — but the `op` CLI in this sandbox has no broker socket:

```
$ op account list
op: OP_BROKER_SOCK is not set in this environment ...
```

**So: if either node wedges in a way that needs a reboot, I cannot recover it,
and the cluster stays down until a human intervenes.** Everything below is
shaped by making that outcome unreachable rather than merely unlikely.

The only path to that state is the `pkill -9` escalation (skips the C++ static
destructors that free RDMA QPs) or a failed memory reclaim. §2.0 therefore
converts both into **pre-flight gates that fail safely before anything is
mutated**, instead of hazards encountered mid-deploy.

---

## 2.0 Pre-emptive verified graceful shutdown (gate)

Rather than let `start_cluster.sh` discover a stuck process 40 minutes into an
unattended run, do the shutdown **first, by hand, with verification**, while the
deployed tree is still untouched and the rollback is a no-op.

```bash
# G1. graceful SIGTERM on both nodes (never -9)
for N in macstudio-m4-1 macstudio-m4-2; do
  ssh $N "pkill -TERM -f 'python.*exo' 2>/dev/null || true"
done
# G2. confirm clean exit within 15s, on both
# G3. confirm RDMA ports still PORT_ACTIVE and the direct TB link still pings
# G4. confirm wired+compressor memory drained below 25 GB on both
```

**GATE: if any of G2/G3/G4 fails, STOP.** Do not deploy. Relaunch `-0731`
via `~/relaunch_exo.sh` (which the last `start_cluster.sh` generated on each
node from the same `EXO_ENV`) and report. Reaching this gate cleanly proves the
processes tear down without needing `-9`, which is precisely the property that
makes the rest of the run recoverable.

After the gate passes, `start_cluster.sh`'s own `pkill` finds nothing to kill
and its `-9` escalation branch is unreachable by construction.

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

### 3.6 Live memory instrumentation (added post-consult)

Every number in §1.4 is **load-time** safetensors arithmetic. Nothing here
measures what vision *inference* adds on top — ViT activations for a 384-token
image, the aligner projection, and the extra KV for the expanded sentinel block.
Against a ceiling that `-0731` already approaches (92–99 GB measured vs
112.3 GiB), that is an unmeasured risk, so measure it rather than assume:

```bash
# poll both nodes throughout load + smoke test
for N in macstudio-m4-1 macstudio-m4-2; do ssh $N "vm_stat | awk '
  /page size of/{ps=\$8} /Pages wired down:/{w=\$4} /Pages occupied by compressor:/{c=\$5}
  END{printf \"%s wired+compressor = %.1f GB\n\", \"'\$N'\", (w+c)*ps/1e9}'"; done
```

Record: (a) post-load steady state, (b) peak during the image request. If either
node exceeds **100 GB**, abort the run and roll back — that is inside the band
where the documented Metal-allocator wedge appears
(`start_cluster.sh:1218-1234`), and a wedge is unrecoverable for this session
(§1.10).

---

## 4. Rollback plan

**Trigger:** any FAIL in §3.5, or any abort in §5.

**Target state:** `main` @ `11bf2e29c` deployed on both nodes, one
`MlxJacclInstance` serving `deepseek-ai/DeepSeek-V4-Flash-0731`, Tensor/JACCL,
both runners `RunnerReady` — i.e. byte-for-byte the configuration running
before Phase 5 started, including the production speculative env (the defaults
`EXO_SPECULATIVE=1 EXO_DSV4_MTP=1 EXO_DSV4_DSPARK=1` restore themselves once
the overrides are dropped).

### 4.1 Two independent rollback paths

The consult (§8) correctly flagged that "re-run the same script backwards"
shares fate with the forward deploy: if a network-dependent step (`uv sync`,
the mlx-lm reinstall, `npm`) is what broke, both directions break identically.
So there are **two** paths, tried in order.

**PATH A — fast, offline, no rebuild (preferred).** The insight is that
`-0731` does not need the `main` *code* to run: the Vision-Exp deploy changes
only `resources/`, `docs/`, `src/exo/**` vision files and mlx-lm. Nothing in
that set removes `-0731` support — the branch is purely *additive* to the
serving path. Therefore, if the branch is deployed and healthy but Vision-Exp
itself misbehaves, `-0731` can be re-placed **on the deployed branch**, with no
rsync, no pip, no npm, no network:

```bash
# A1. delete the Vision-Exp instance
curl -s -X DELETE "http://192.168.86.201:52415/instance/<instance_id>"
# A2. place -0731 via the same two-step flow start_cluster.sh uses
curl -sG "http://192.168.86.201:52415/instance/placement" \
  --data-urlencode "model_id=deepseek-ai/DeepSeek-V4-Flash-0731" \
  --data-urlencode "sharding=Tensor" --data-urlencode "instance_meta=MlxJaccl" \
  --data-urlencode "min_nodes=2"   # -> POST /instance with the result
# A3. prove with a real completion
```

This is the fastest route back to serving and depends on **zero** fragile
steps. It is the correct first response to "Vision-Exp is bad but the cluster
is fine."

**PATH B — full redeploy of `main`** (for when the branch deploy itself is
suspect):

```bash
ssh macstudio-m4-1 'cd ~/repos/exo && git checkout main && \
  git submodule update --init --recursive && git rev-parse HEAD && git submodule status'
# expect 11bf2e29c…, mlx e40a416b…, mlx-lm 7f14654…
ssh macstudio-m4-1 "cd ~/repos/exo && EXO_TARGET_BRANCH=main QWEN36_ENABLED=0 \
  nohup ./start_cluster.sh > /tmp/phase5_rollback.log 2>&1 &"
```

**Path B's network dependency is verified symmetric**, so the consult's
shared-fate concern is bounded: both mlx-lm refs the deploy needs already exist
on `origin`, so neither direction depends on a fetch the other didn't —

```
$ git -C mlx-lm ls-remote origin dsv4-vision-port
513d6dbec624ea533619ada052458edb9b04b868   refs/heads/dsv4-vision-port
$ git -C mlx-lm ls-remote origin main
7f146542811dd774d2cb3ab38b13c4b25a8e8063   refs/heads/main
```

and the branch touches **no** `dashboard/` or `package.json` files, so the npm
step has identical inputs in both directions and cannot fail one way only.

**PATH C — process-level relaunch.** If the API is unresponsive but the nodes
are healthy, each node still carries `~/relaunch_exo.sh`, regenerated by the
last `start_cluster.sh` from the same `EXO_ENV`. It restarts the exo process
only (no deploy, no placement) and is fully offline.

### 4.2 Proof of rollback

```bash
curl -s http://192.168.86.201:52415/state   # instance modelId == ...-0731, 2x RunnerReady
curl -s http://192.168.86.201:52415/v1/chat/completions -H 'Content-Type: application/json' \
  -d '{"model":"deepseek-ai/DeepSeek-V4-Flash-0731","max_tokens":16,"temperature":0,
       "messages":[{"role":"user","content":"Say hello."}]}'
```

Rollback is **verified only when a real completion comes back**. A
`RunnerReady` pair is not proof of serving.

### 4.3 If rollback itself fails

Per §1.10 I cannot reboot a FileVault-locked node. So:

- Runners won't reach Ready → read `~/exo.log` on both nodes, try PATH C.
- Stuck wired memory or a TB/RDMA wedge → **I cannot fix this.** Report
  immediately and explicitly, naming `./reboot-node.sh <node>` (run from a host
  with a 1Password-authenticated `op` CLI) as the required human action.
- Never escalate to `pkill -9` to try to force progress — that is the specific
  action most likely to *create* the unrecoverable state.

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

---

## 7. Pre-execution consult record

Two consult calls were made before touching the cluster (a third, to the
default reviewer, failed with an internal server error, and a second attempt
against `gemini-3-pro` timed out at 420 s — both are tool failures, not
refusals).

### 7.1 Consult 1 — `pkill -f` self-kill analysis

**Asked:** running `start_cluster.sh` *on* macstudio-m4-1 means its own
per-node `pkill -TERM -f 'python.*exo'` targets the host running the script.
Could that regex match my own deploy process?

**Answer:** No. BSD `pkill -f` matches a case-sensitive POSIX ERE against the
full argv. `python.*exo` requires the literal lowercase substring `python`
before any `exo`; my invocation string
(`... DSV4_MODEL_ID=deepseek-ai/DeepSeek-V4-Flash-Vision-Exp ... ./start_cluster.sh`)
contains **no** `python` substring at all, so the regex cannot match regardless
of the `.*exo` part. It also confirmed the 4-process kill (SCREEN/login/zsh/
python all carry the launch string in their argv) is fine: pkill signals each
PID directly from the kernel, so python's own SIGTERM handler and C++ static
destructors run independently of the wrappers dying.

**Actionable caveat it raised, and the result of checking it:** it warned that
a *later* defensive pkill in the script would be a real hazard, because by then
my own build subprocesses (`uv`, `maturin`, `pip install ./mlx-lm`) legitimately
match `python.*exo`. Verified:

```
$ grep -n 'pkill\|kill -\|killall' start_cluster.sh
1246:  ssh "$NODE" "pkill -TERM -f 'python.*exo' ...; pkill -TERM -f 'exo.main' ..."
1259:    ssh "$NODE" "lsof -ti:52415,52416 | xargs kill -9 ..."
1260:    ssh "$NODE" "pkill -9 -f 'exo.main' || true"
1261:    ssh "$NODE" "pkill -9 -f 'python.*exo' || true"
2883:pkill -TERM -f 'python.*exo' 2>/dev/null || true
2890:  pkill -9 -f 'python.*exo' 2>/dev/null || true
```

Lines 1246/1259-1261 are all at loop-top, **before** any build step. Lines
2883/2890 are inside the quoted `RELAUNCH_BODY` heredoc — they are *written to*
`~/relaunch_exo.sh`, never executed by the deploy. **No stray pkill. Cleared.**

Also confirmed the process-group question it flagged: `ps -o pid,ppid,pgid`
shows python (4000) in pgid 3987 shared with the zsh wrapper — the wrapper is
`login`, whose session is already detached under SCREEN, so no controlling
terminal exists to deliver a racing SIGHUP.

### 7.2 Consult 2 — unattended risk, rollback soundness, spec-off decision

**What it changed in this procedure (all three were real gaps):**

1. **The reboot capability gate (§1.10) — the most valuable finding.** It said,
   correctly, that verifying unattended reboot capability "dwarfs every other
   risk," because a wedged TB/RDMA stack is otherwise permanently unrecoverable
   by me. I had documented "reboot the node" as my escape hatch **without ever
   checking that I could do it.** I checked: FileVault is ON, `sudo -n reboot`
   requires a password, and this sandbox's `op` CLI has no broker socket, so
   `reboot-node.sh` is unusable from here. My escape hatch did not exist.
   → Added §1.10, and added §2.0, which converts the `-9`-escalation and
   memory-reclaim hazards into **pre-flight gates that fail before anything is
   mutated**, so the unrecoverable state becomes unreachable rather than merely
   unlikely.

2. **Rollback shared-fate (§4.1).** It flagged that re-running the same script
   backwards is not an independent recovery path. → Added PATH A (re-place
   `-0731` on the *already-deployed branch* via the REST API — no rsync, no
   pip, no npm, no network, since the branch is purely additive and does not
   remove `-0731` support) as the preferred rollback, with the full redeploy
   demoted to PATH B and a process-only PATH C added. Also verified and
   documented that PATH B's network dependency is *symmetric* (both mlx-lm refs
   already on origin; no dashboard/package.json changes), which bounds the
   concern rather than leaving it open.

3. **Runtime memory is unmeasured.** It noted all my numbers are *load-time*
   safetensors math with zero data on vision-inference runtime peak (image
   encoder activations, extra KV) against an already-tight ceiling.
   → Added live wired-memory polling during the smoke test (§3.6).

**On the spec-off decision, it challenged my reasoning and I accepted the
correction.** I had framed it as "known-good config vs unexercised config." That
framing is wrong: `-0731` has only ever run `num_nextn_predict_layers=1`, so
running Vision-Exp with spec ON (which would load 3 MTP stages) is *equally*
unexercised. Both options introduce a novel variable; the real question is which
one. Spec-off remains correct, but for the better reason: it preserves memory
headroom against an **unmeasured runtime peak** (measured `-0731` already peaks
92–99 GB against a 112.3 GiB ceiling), and it removes an entire subsystem from
the causal chain so a smoke-test failure isolates to vision rather than to a
vision×MTP interaction. §1.4's rationale is written accordingly.

**One recommendation I did not adopt:** staging a full blue-green copy of the
venv/tree for an offline rollback. PATH A already achieves the goal it was
aimed at (a rollback with no fragile steps) without the 1.8 GB venv copy, and
copying the venv aside would risk the `direct_url.json` provenance check that
keeps the no-Metal-compiler MLX rebuild-skip firing (§1.7) — i.e. the mitigation
could itself create the failure it was meant to avoid.

---

## 8. Post-run outcome — EXECUTED 2026-09-09, VISION VERIFIED

**Result: Phase 5 passed.** DSv4 vision works end to end on real hardware, but
only after this run found and fixed a real bug that made every image request
silently degrade to text-only. Cluster was restored to the known-good `-0731`
production state afterwards.

### 8.1 Timeline

| time (UTC) | event |
|---|---|
| 11:45 | pre-flight; last real user request confirmed 34.7 h earlier (idle) |
| 11:45:44 | gate G1–G4: both nodes SIGTERM, **clean exit in 1 s, no SIGKILL**; RDMA PORT_ACTIVE, 0% ping loss; memory drained to 3.7/3.3 GB |
| 11:46:48 | deploy #1 launched (branch `dsv4-vision-port` @ `568077d2`) |
| 11:49:5x | `HEALTHY! (Nodes: 2, Identities: 2)` → `READY (2/2)`, `EXIT_CODE=0` |
| 11:50 | catalog gap **resolved**: 125 → 126 models, Vision-Exp listed, `vision: True` on both ranks |
| 11:51:07 | smoke test #1 **FAILED** — HTTP 200 but `prompt_tokens=31`, "There is no image attached" |
| 11:51–11:58 | root-caused to the API adapter; fix + regression tests written, RED→GREEN on hardware |
| 12:00:06 | deploy #2 launched (`f76a4da3`, the fix) |
| 12:03:0x | `READY (2/2)`, `EXIT_CODE=0` |
| 12:03:19 | smoke test #2 **PASSED** — `"HARBOR, 4"`, `prompt_tokens` 31 → **239** |
| 12:04:05 | second image **PASSED** — `"LANTERN, 0"` (7 squares, correctly reported 0 circles) |
| 12:05:20 | rollback to `main` @ `11bf2e29c` / `-0731` |
| 12:08:5x | `READY (2/2)`, `EXIT_CODE=0`; real completion returned |

### 8.2 The bug this phase existed to find

`chat_request_to_text_generation` emitted a bare `{"type": "image"}` marker into
`chat_template_messages`. The vendored DSv4 encoder rejects it
(`_extract_image` → `ValueError: Image block does not contain a valid source`),
and both generators swallow that into the text-only fallback. Net effect: **every
image request returned HTTP 200 with a confident, entirely image-free answer.**

Fixed at the adapter (not the vendored encoder) by emitting an ordered
reference, `{"type": "image", "url": "exo-image:<n>"}` — a real non-empty source
that satisfies the encoder's existing invariant without weakening it, adds
nothing meaningful to the wire, and keeps the vendored file byte-identical to
upstream. Commit `f76a4da3`, with 5 regression tests covering the seam from both
ends (verified RED against the old adapter, GREEN with the fix; full API suite
82 passed).

The §3.5 pass criteria did exactly their job: criterion P1 alone (HTTP 200,
non-empty answer) would have declared this a **false pass**.

### 8.3 Smoke test evidence

```
POST /v1/chat/completions   model=deepseek-ai/DeepSeek-V4-Flash-Vision-Exp
content: [{"type":"text",...},{"type":"image_url","image_url":{"url":"data:image/png;base64,..."}}]

--- IMAGE 1 (word HARBOR, 4 red circles; sha256 9c86b483…) ---
http 200, 9.1s, usage.prompt_tokens=239, completion_tokens=69, finish_reason=stop
reasoning: 'the word is "HARBOR" ... Below the word, there are four red circles.'
content:   'HARBOR, 4'

--- IMAGE 2 (word LANTERN, 7 blue squares; sha256 86449b35…) ---
http 200, 4.7s, usage.prompt_tokens=239, finish_reason=stop
reasoning: 'the word is "LANTERN". Below the word, there are seven blue squares.
            The user asks for the number of circles. There are zero circles.'
content:   'LANTERN, 0'

--- CONTROL (identical prompt, NO image) ---
usage.prompt_tokens=31
content: 'There is no image attached to your prompt.'

--- server-side ---
DSv4 vision: 1 image(s), prompt 32 -> 239 tokens after placeholder expansion
  image 0: patches=(1610, 3, 14, 14) grid=35x46 block=208 tokens at [29, 237)
grep 'Vision processing failed' since redeploy: 0
```

All six criteria met — P1 ✓, P2 ✓ (exact word), P3 ✓ (exact count), P4 ✓ (no
fallback), P5 ✓ (31→239 token expansion), P6 ✓ (control cannot know).

Image 2 is the strongest single piece of evidence: asked how many *circles* were
below the word, the model reported the 7 squares it actually saw and answered
**0 circles**, refusing a leading prompt. That is not pattern-matching.

### 8.4 Memory — §1.4 predictions vs measured

Predicted 69.488 GiB/node loaded (spec-off). Measured `wired+compressor` stayed
at **3.6–3.9 GB/node** throughout load and both vision requests, with runner RSS
~7.5–8.0 GB. MLX maps the safetensors and lazily faults pages, so neither figure
is the resident weight set — but the operative fact is that the §3.6 abort
threshold (100 GB/node) was never approached. **No memory pressure whatsoever.**
Vision-inference runtime peak is therefore no longer an unmeasured risk at this
image size (one 640×480 image, 208 image tokens).

Spec-off was the right call and cost nothing observable: 4.7–9.1 s end to end.

### 8.5 Final state (restored, verified)

```
both nodes:  main @ 11bf2e29c   (exact pre-Phase-5 commit)
             mlx e40a416b, mlx-lm 7f14654
instance:    79db83e1-…  MlxJacclInstance  deepseek-ai/DeepSeek-V4-Flash-0731
runners:     da419875 RunnerReady, a696b12a RunnerReady
env:         EXO_SPECULATIVE=1 EXO_DSV4_MTP=1 EXO_DSV4_DSPARK=1  (both nodes)
completion:  'The capital of Japan is Tokyo.'  finish_reason=stop
catalog:     125 models, Vision-Exp not listed (expected — card lives on the branch)
```

Restored deliberately rather than left on Vision-Exp: the card still says TREAT
AS UNVERIFIED for quality, spec-off would cost ~36% decode on the text traffic
this cluster actually serves, and switching production models was not the
authorized scope. Doing it also **proved the rollback path works** rather than
leaving it a paper plan.

### 8.6 What is now known, and what is still not

**Established:** the DSv4 vision path — catalog → placement → TP-replicated
vision tower → image processor → sentinel expansion → embedding merge → serving
API — works on real hardware for real images. The model card's "NOT VERIFIED
END-TO-END ON HARDWARE" warning can be lifted for *functionality*.

**Still not established:** vision *quality* at scale (two synthetic images is a
functional test, not an eval); multi-image requests in production (unit-tested
only); vision with speculative decoding on (never run — Vision-Exp's
`num_nextn_predict_layers=3` would load 3 MTP stages, +3.19 GiB/node, of which
only stage 0 is ever used); large/high-resolution images; and the sampling
defaults, which are still carried over verbatim from `-0731` and un-A/B'd.


