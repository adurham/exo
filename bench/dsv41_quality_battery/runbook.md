# DSv4.1 LIVE QUALITY BATTERY — two-phase runbook

Purpose: catch the *defect class* that slipped past unit/synthetic gates twice
on this model — real precision errors (hc_expand bf16-comb ~1.08% mean rel err)
and a live text-quality defect (lm_head mxfp8 glued cross-lingual subword
fragments: `angleсь`, `camerauden`, `pavement他身上`) that a 15-task high-margin
eval scored 15/15 byte-identical on. The gate here is **low-margin free-form
prose + degeneration detectors + a two-phase diff**, not task pass/fail.

Read first:
- `~/repos/exo/docs/incidents/lmhead-mxfp8-cross-lingual-glue-defect-2026-09-13.md`
- `~/repos/exo/docs/lmhead-mxfp8-defect-and-fallback-investigations-2026-09-14.md`

This battery lives entirely under
`~/.hermes/cache/scratch/quality_battery/` and never edits the repo.
The change under test is the **bf16 indexer score row** precision change
(`DSV41_INDEXER_ROW_BF16`).

---

## 0. What the two phases are

| | Phase A — **baseline** | Phase B — **candidate** |
|---|---|---|
| build | current production build (no indexer-row code) | branch build with `DSV41_INDEXER_ROW_BF16` default on (bf16 row) |
| mlx-lm checkout | the shipped pin (`git -C mlx-lm rev-parse HEAD` at start) | the branch (e.g. `fix/dsv41-...`) |
| launched env | `DSV4_KV_CACHE_BITS=0` (prod default) | `DSV4_KV_CACHE_BITS=0` (+ `DSV41_INDEXER_ROW_BF16=1`, the branch default) |
| results | `results/prodA/` | `results/bf16B/` |

> If the baseline phase is instead run on the **branch** build with the fp32 row
> (so both phases share one checkout and only the env differs), launch Phase A
> with `DSV41_INDEXER_ROW_BF16=0` — see §3. `DSV41_INDEXER_ROW_BF16` is read at
> **import time** (`mlx-lm/mlx_lm/models/deepseek_v41/indexer.py`), so it MUST be
> set in the **launched process env**, not injected later over the API; changing
> it requires a relaunch. **TBD-VERIFY-ON-RUN**: confirm the branch's
> `start_cluster.sh` forwards `DSV41_INDEXER_ROW_BF16` into the runner environ
> (the launcher builds `EXO_ENV` from an allow-list near `start_cluster.sh:2029`;
> a var not on that list may be dropped before the runner import). If it is not
> forwarded, add it to the launched command's env the same way
> `EXO_DSV4_LMHEAD_MXFP8` is (line ~2033), or export it so `screen -dmS … zsh -l -c`
> inherits it.

---

## 1. Pre-flight (both phases)

```bash
# On the laptop (control host). Nothing here touches the cluster yet.
cd ~/repos/exo

# Record the exact SHAs that will be deployed (this is what makes the A/B honest).
git rev-parse HEAD                     > /tmp/qa_exo_head.txt
git -C mlx-lm rev-parse HEAD           > /tmp/qa_mlxlm_head.txt
git -C mlx-lm rev-parse --abbrev-ref HEAD >> /tmp/qa_mlxlm_head.txt
git submodule status                   > /tmp/qa_submods.txt
cat /tmp/qa_exo_head.txt /tmp/qa_mlxlm_head.txt /tmp/qa_submods.txt

# Confirm python3 + the battery are present and self-test clean (offline):
cd ~/.hermes/cache/scratch/quality_battery && python3 battery.py --selftest
```

**Push/alignment guard (from `start_cluster.sh` §0):** the launcher refuses to
proceed non-interactively unless local `HEAD` is an ancestor of
`origin/${EXO_TARGET_BRANCH:-main}` — *the cluster only ever runs pushed
commits*. If you are deploying a branch build, either push the branch and set
`EXO_TARGET_BRANCH=<branch>` on the launch, or launch with
`EXO_ALLOW_UNPUSHED=1`. **A local-only commit will NOT be what the studios run.**

> Deploy mechanism (read this — it changes what "deploy arm B" means): the
> current launcher **rsyncs the laptop tree verbatim** to both nodes
> (`rsync -a --delete --exclude '.venv/' --exclude 'tmp/' … ~/repos/exo/ $NODE:~/repos/exo/`,
> `start_cluster.sh:1642`) and then **force-reinstalls mlx-lm from `./mlx-lm`**.
> So whatever branch/checkout the `mlx-lm` submodule is at **on the laptop** is
> exactly what both studios run. The submodule pointer being "uncommitted"
> (`M mlx-lm` in the superproject) is the *normal* measurement vehicle, provided
> the mlx-lm commit itself is pushed on its fork (see
> `docs/hc-collapse-kernel-ab-2026-08-25.md` §3.1).

---

## 2. Phase A — baseline

### 2a. Put the checkout at the baseline

Production baseline (current default `main`, no indexer-row code):
```bash
cd ~/repos/exo
git checkout main && git pull --ff-only          # or: git checkout <prod-sha>
# mlx-lm back to its shipped pin:
git submodule update --init --recursive mlx-lm
git -C mlx-lm rev-parse HEAD                      # record in /tmp/qa_mlxlm_head.txt
```

### 2b. Relaunch

```bash
cd ~/repos/exo
DSV4_KV_CACHE_BITS=0 ./start_cluster.sh 2>&1 | tee /tmp/qa_launch_A.log
# (add EXO_TARGET_BRANCH=<branch> if you are on a branch rather than main)
```
Wait for the launcher to print ` READY (2/2)` for the DeepSeek V4 instance
(`start_cluster.sh:3696`). Do not proceed until both runners are Ready.

Optional independent readiness check (bounded poll, no blind sleep):
```bash
API=http://macstudio-m4-1.tail19c543.ts.net:52415
for i in $(seq 1 60); do
  n=$(curl -s --max-time 10 "$API/state" | python3 -c "
import json,sys
d=json.load(sys.stdin)
r=d.get('runners') or {}
print(sum(1 for v in r.values() if 'RunnerReady' in v))")
  echo "ready runners: $n"; [ "$n" = "2" ] && break; sleep 5
done
```

### 2c. Run the battery

```bash
cd ~/.hermes/cache/scratch/quality_battery
python3 battery.py --label prodA --depth 350000 all 2>&1 | tee results/prodA/run.log
# quick mode if you are iterating and want a faster loop:
#   python3 battery.py --label prodA --depth 120000 all
```
`all` = build (deep-context prompt, planted stratified needles, warms the
prefix) then eval (needles + free prose + tools + parked restore) and writes
`results/prodA/summary.json`.

Note the phase verdict printed at the end. A production baseline should be
`CLEAN`; if it is not, STOP — the battery is mis-scoped or the cluster is
already unhealthy (triage before comparing anything).

---

## 3. Swap to the candidate build (git worktree / branch flip — NOT file edits)

The change is in `mlx-lm`. Two supported mechanisms; pick one.

**Option 1 — worktree (preferred; keeps the running main tree untouched):**
```bash
cd ~/repos/exo
# Create a worktree of the exo branch that pins the candidate mlx-lm commit.
git worktree add ../exo-qa-bf16 <exo-branch>          # or a detached SHA
cd ../exo-qa-bf16
git submodule update --init --recursive
git -C mlx-lm fetch origin
git -C mlx-lm checkout <mlx-lm-branch-or-sha>          # the bf16 indexer-row commit
git -C mlx-lm rev-parse HEAD                            # record
```
Then launch **from that worktree** (`cd ~/repos/exo-qa-bf16 && ./start_cluster.sh …`)
— the rsync source is the worktree, so the studios get exactly that checkout.
Launching from a different directory than `~/repos/exo` **TBD-VERIFY-ON-RUN**:
`start_cluster.sh` rsyncs `$HOME/repos/exo/` by *hardcoded path*
(`start_cluster.sh:1649`), **not** its own location — so a worktree launch will
still push `~/repos/exo`. If the worktree path differs from `~/repos/exo`, you
must instead flip the branch in `~/repos/exo` itself (Option 2) or override the
rsync source var if one exists. **Verify this before relying on worktrees.**

**Option 2 — branch flip in place (matches the deploy mechanism exactly):**
```bash
cd ~/repos/exo
git checkout <exo-branch>
git submodule update --init --recursive
git -C mlx-lm fetch origin
git -C mlx-lm checkout <mlx-lm-branch-or-sha>
git -C mlx-lm rev-parse HEAD     # this is what the studios will run (rsync)
```
The superproject will show `M mlx-lm` (a pointer-only modification) — that is
the expected, git-clean measurement state (the mlx-lm commit must itself be
pushed on the fork; see the alignment guard in §1). Do **not** commit the
pointer bump as part of the measurement.

### 3a. Relaunch Phase B with the env difference

```bash
cd ~/repos/exo
# Branch default is bf16; set it explicitly so the run is self-documenting:
DSV4_KV_CACHE_BITS=0 DSV41_INDEXER_ROW_BF16=1 ./start_cluster.sh 2>&1 | tee /tmp/qa_launch_B.log
```
- If Phase A was run on the **branch** with the fp32 row as baseline, Phase A's
  launch was instead `DSV4_KV_CACHE_BITS=0 DSV41_INDEXER_ROW_BF16=0 ./start_cluster.sh`.
- Production Phase A needs no `DSV41_INDEXER_ROW_BF16` (the code does not exist
  there).
- Confirm the var reached the runners (**TBD-VERIFY-ON-RUN**), e.g.:
  `ssh -o BatchMode=yes macstudio-m4-1 'ps eww $(pgrep -f "repos/exo/.venv/bin/python") | tr " " "\n" | grep -E "^DSV41_INDEXER_ROW_BF16="'`

Wait for ` READY (2/2)` (§2b), then:

```bash
cd ~/.hermes/cache/scratch/quality_battery
python3 battery.py --label bf16B --depth 350000 all 2>&1 | tee results/bf16B/run.log
```

---

## 4. Blind compare (the SHIP GATE)

```bash
cd ~/.hermes/cache/scratch/quality_battery
python3 compare.py results/prodA results/bf16B --json
```

Verdict semantics (see `compare.py` header):
- **PASS** — B ≥ A on needles/tools/detectors, no new glued fragments, park
  recall held.
- **REVIEW** — no hard failure, but a delta needs eyes (verdict CLEAN↔REVIEW, a
  needle/tool count that shifted). Dump prose side-by-side and read it.
- **FAIL** — any new high-confidence detector hit (glued cross-lingual fragment,
  U+FFFD, repetition loop), a needle/tool regression, or lost park recall.
  **Do not ship the B build.**

Free-prose text WILL differ between arms at temp=0 (the incident doc: free prose
visibly diverges, 0/5 byte-identical) — that is a printed *note*, not a gate.

**Blind side-by-side dump** for human/model review (labels stripped so the
reviewer does not know which arm is which until after judging):
```bash
cd ~/.hermes/cache/scratch/quality_battery
python3 - <<'PY'
import os, json
A, B = "results/prodA/free_prose", "results/bf16B/free_prose"
out = open("results/blind_compare.txt", "w")
ids = sorted(f[:-5] for f in os.listdir(A) if f.endswith(".json") and f != "index.json")
for pid in ids:
    da = json.load(open(os.path.join(A, pid + ".json")))
    db = json.load(open(os.path.join(B, pid + ".json")))
    # deterministic 50/50 arm shuffle by prompt id so the mapping isn't obvious
    first, second = (("X", da), ("Y", db)) if (sum(map(ord, pid)) % 2 == 0) else (("Y", da), ("X", db))
    out.write("="*78 + "\n[{}]  PROMPT: {}\n\n--- ARM {} ---\n{}\n\n--- ARM {} ---\n{}\n\n".format(
        pid, da["prompt"], first[0], first[1]["content"], second[0], second[1]["content"]))
out.close()
print("wrote results/blind_compare.txt  (arm X/Y mapping is in compare.py's run; "
      "reveal only after judging)")
PY
less results/blind_compare.txt
```
Also dump each detected hit with context:
```bash
python3 - <<'PY'
import json, os
for arm in ("prodA", "bf16B"):
    idx = json.load(open("results/%s/free_prose/index.json" % arm))
    for p in idx["probes"]:
        for h in (p.get("detector") or {}).get("hits", []):
            if h.get("confidence") == "high":
                print(arm, p["id"], h["type"], repr(h.get("token", h.get("ngram", ""))))
PY
```

---

## 5. Rollback / restore

```bash
cd ~/repos/exo
git checkout main
git submodule update --init --recursive mlx-lm      # back to shipped pin
git -C mlx-lm status --porcelain                     # expect clean
DSV4_KV_CACHE_BITS=0 ./start_cluster.sh 2>&1 | tee /tmp/qa_restore.log
# worktree cleanup if used:
git worktree remove ../exo-qa-bf16 --force
```

---

## 6. Assumptions / TBD-VERIFY-ON-RUN

- **TBD**: `DSV41_INDEXER_ROW_BF16` forwarding through the launcher's `EXO_ENV`
  allow-list (verify the runner environ with `ps eww`; see §3a). The var is read
  at import in `mlx-lm/mlx_lm/models/deepseek_v41/indexer.py`; if the launcher
  drops it, every phase runs the default and the A/B is void.
- **TBD**: worktree-launch rsync source — the launcher hardcodes
  `$HOME/repos/exo/`, so a worktree at a different path is likely ignored;
  prefer the in-place branch flip (§3 Option 2) unless verified.
- **TBD**: exact READY endpoint/field — the battery's `plan`/`eval` do not gate
  on readiness; use the launcher's ` READY (2/2)` line or the `/state` runner
  poll in §2b.
- **TBD**: parked-restore session semantics (`battery.py` `park` mode) — depends
  on the engine's session-matching and `Dsv41Engine.max_sessions=2`; confirm
  turn-2 elapsed ≪ cold prefill and, with `--ssh-park-check`, a park/restore log
  line. Follows the `park_live_proof.sh` pattern; SSH is read-only and opt-in.
- The `same_script_glue` detector is a **low-confidence** heuristic (the exact
  signature is the cross-script case); it drives REVIEW, not FAIL.
- Token counts are estimates from the measured live ratio 5.111 chars/token
  (`1m_soak2.sh`); the server's `usage.prompt_tokens` is the ground truth and is
  recorded in each result.

## 7. NOT verified offline (build task)

- No cluster contact was made: `build`, `eval`, `all`, and `park` were **not**
  executed against the live endpoint. Only `--selftest`, `plan`, `compare.py`
  (on synthetic result dirs), and `aggregate` were exercised.
- The exact API response shapes were coded from `1m_soak2.sh`,
  `park_live_proof.sh`, and `1m_live_probe*.sh`; confirm on the first live run
  that `tool_calls`, `reasoning_content`, and `usage` parse as expected.
