# Telemetry Sampler (e) — Runner PID/lstart Restart Detector — Validation Report

**Date:** 2026-09-01 (CDT)
**Validator:** subagent (read-only) | **Nodes:** macstudio-m4-1, macstudio-m4-2
**Verdict: PARTIAL — FAIL on criterion (b).** The genuine runner is matched, the signature is stable across samples (no false restart alarm), and a restart would be detected. **BUT** the matcher falsely matches all 3 launcher-wrapper processes per node (SCREEN, login, zsh) via the embedded literal `-m exo` in their launch string — exactly the false-positive class the sampler's own docstring explicitly warns about. Proposed fix (comm filter) verified to collapse to exactly the real runner.

---

## 1. Sampler source / documented intent

- Source: `tmp/research-v1v2v3-20260901/telemetry/collect_telemetry.py`, section `(e)` lines 279–353.
- Default pattern (`_DEFAULT_RUNNER_PATTERN`, lines 297–299):
  `(-m exo|exo -v|batch_generator|exo\.worker|exo\.main)`
- Matcher logic: `parse_ps_runner()` (lines 302–337) parses `ps -axo pid=,lstart=,comm=,args=` rows, splits each line into `pid + rest`, pulls lstart (5 fixed tokens), then keeps a row iff **`rx.search(args)`** (regex over the **entire args string**, line 335).
- Docstring (lines 289–296) and TELEMETRY_KIT.md §Design guarantees + Known-caveat #1 both state the intent: the runner must be identified **without** matching shell wrappers whose lstart changes on every new shell, because that would corrupt restart detection. The kit's stated mitigation ("anchor on `-m exo` / `exo -v`, not the bare python path") is **insufficient** — see below.

## 2. REAL runner signatures observed on both nodes (live, via ssh, read-only)

`ps -axo pid=,lstart=,comm=,args=` captured from each node (S1 and S2, ~20s apart).

**macstudio-m4-1**
| pid | lstart | comm |
|---|---|---|
| **83029** | `Tue Sep  1 16:19:35 2026` | `.venv/bin/python` (`.venv/bin/python -m exo -v`) ← **genuine runner** |
| 83017 | `Tue Sep  1 16:19:35 2026` | `SCREEN` (wrapper) |
| 83018 | `Tue Sep  1 16:19:35 2026` | `login` (wrapper) |
| 83019 | `Tue Sep  1 16:19:35 2026` | `zsh` (wrapper) |
| 83031 | `Tue Sep  1 16:19:36 2026` | `...python -c from multiprocessing.resource_tracker import main;main(7)` |
| 83189 | `Tue Sep  1 16:19:48 2026` | `...python -c from multiprocessing.spawn import spawn_main; ... --multiprocessing-fork` |

**macstudio-m4-2**
| pid | lstart | comm |
|---|---|---|
| **85554** | `Tue Sep  1 16:19:37 2026` | `.venv/bin/python` (`.venv/bin/python -m exo -v`) ← **genuine runner** |
| 85543 | `Tue Sep  1 16:19:37 2026` | `SCREEN` (wrapper) |
| 85544 | `Tue Sep  1 16:19:37 2026` | `login` (wrapper) |
| 85545 | `Tue Sep  1 16:19:37 2026` | `zsh` (wrapper) |
| 85556 | `Tue Sep  1 16:19:37 2026` | `...python -c from multiprocessing.resource_tracker import main;main(7)` |
| 85676 | `Tue Sep  1 16:19:48 2026` | `...python -c from multiprocessing.spawn import spawn_main; ... --multiprocessing-fork` |

PIDs/lstarts match the task's ground-truth exactly.

## 3. Matching-logic results (sampler's own functions run against REAL captured ps output)

Imported `collect_telemetry.py`; fed each node's captured ps text to `parse_ps_runner(_, _DEFAULT_RUNNER_PATTERN)`.

**(a) Genuine runner matched?** YES on both nodes (83029 lstart 16:19:35; 85554 lstart 16:19:37).

**(b) Wrappers avoided?** **NO.** **3 false positives per node** — each matcher's `rx.search(args)` hit the literal `-m exo` that is textually embedded inside the SCREEN/login/zsh launcher command lines:

| node | pid | comm | why matched |
|---|---|---|---|
| m4-1 | 83017 | SCREEN | args embed `... .venv/bin/python -m exo -v >> ~/exo.log` |
| m4-1 | 83018 | login | args embed the same launch string |
| m4-1 | 83019 | zsh | args embed the same launch string |
| m4-1 | 83029 | `.venv/bin/python` | genuine runner |
| m4-2 | 85543 | SCREEN | args embed `.venv/bin/python -m exo -v` |
| m4-2 | 85544 | login | args embed the same |
| m4-2 | 85545 | zsh | args embed the same |
| m4-2 | 85554 | `.venv/bin/python` | genuine runner |

So `count` for node1 = 4, node2 = 4, not the expected 1 genuine runner each.

**Correctly rejected:** the `multiprocessing.resource_tracker` process (83031 / 85556) and the `multiprocessing.spawn/spawn_main` process (83189 / 85676) — their args contain no marker → not matched. ✔

**Why the documented mitigation fails:** the kit's defense (lines 289–296) is to drop a bare `repos/exo/.venv/bin/python` path match. But the wrappers do **not** match via a bare path — they match via the marker `-m exo` itself, which `screen -dmS exorun zsh -l -c "cd ~/repos/exo && ... .venv/bin/python -m exo -v ..."` embeds as literal text in its command line. Because `parse_ps_runner` regex-searches the **full args string** (line 335) rather than restricting to the process **comm**, every wrapper that merely invokes `python -m exo` looks identical to the real runner.

**Practical severity note:** in the *current* state these wrappers are long-lived children of the same 16:19 launch, so their lstart does not independently change across checkpoints unless the SCREEN session is re-created. That limits today's corruption risk. But the failure class is real: any re-exec of the launcher chain (new shell / `screen -R` reattach, a fresh `zsh -l -c`, etc.) gives the wrapper a **new lstart while the runner python keeps its old one**, and the sampler's `processes` list would then mix a stale runner PID with a fresh wrapper PID — indistinguishable from a genuine restart by the documented pid/lstart comparison. This is exactly the docstring's stated corruption scenario, and it is not prevented under the current pattern.

**(c) Restart detection would work:** demonstrated on the real node1 ps data.
- Stability: S1 vs S2 (~20s apart, same unrestarted runner) → **identical** (`pid,lstart`) signature sets on both nodes (node1: {83017,83018,83019,83029} all 16:19:35; node2: {85543,85544,85545,85554} all 16:19:37). **No false restart alarm.**
- Simulated restart (runner python respawned in place: 83029 → new pid 99999 + new lstart `Tue Sep  1 17:00:00 2026`, wrappers unchanged): full signature set **changed** → restart detectable. With the proposed python-comm filter, the discriminated signature cleanly changes from `{(83029, 16:19:35)}` → `{(99999, 17:00:00)}`.

## 4. Bug report — exact lines and proposed diff

- **Bug:** `parse_ps_runner` (collect_telemetry.py, **lines 302–337**) matches the runner marker against the **entire `args` string** (`if rx.search(args)` at **line 335**), so any process whose command line *contains* `-m exo` / `exo -v` (including SCREEN/login/zsh wrappers that just invoke the runner) is counted as a runner. The pattern alone (lines 297–299) cannot distinguish them.
- **Proposed fix (do not apply — reported only):** restrict to rows whose `comm` is (or ends in) a `python*` interpreter. Verified on both nodes: this collapses the 4 matches to exactly the genuine runner (`[{'pid': 83029, '.venv/bin/python'}]`, `[{'pid': 85554, '.venv/bin/python'}]`).

```diff
@@ collect_telemetry.py @@
     def parse_ps_runner(text: str, pattern: str) -> list[dict]:
         ...
         comm = comm_args.split(None, 1)[0] if comm_args else ""
         args = comm_args
-        if rx.search(args):
+        # Only the real runner (and its multiprocessing spawn children) run under
+        # a python interpreter. A SCREEN/login/zsh wrapper that merely *invokes*
+        # `python -m exo -v` embeds the marker in its args and must be excluded.
+        comm_base = comm.rsplit("/", 1)[-1]
+        if comm_base.startswith("python") and rx.search(args):
             procs.append({...})
     return procs
```

Alternative (equivalent): keep `rx.search(args)` but require `re.search(r"\bpython(?:[0-9.]*)?\b", comm)`. (Exo also launches native runner processes under some versions — those would carry markers `exo.worker`/`exo.main` and `comm` `exo`...`; a comm-filter that additionally accepts those would preserve multi-process coverage. Recommend confirming with the codebase before finalizing.)

## 5. End-to-end sampler run (local, informational)

Ran the kit as documented on the laptop:
`/Users/adam.durham/repos/exo/.venv/bin/python collect_telemetry.py validation-local --out /tmp/telemetry_validation_local.jsonl`
→ exit 0, appended valid JSONL, `runner` = `{"processes": [], "count": 0, "pattern": "...", "error": null}` (correct: laptop is not a cluster node, no runner). Confirms the kit runs cleanly; the meaningful validation is the ssh-captured-node analysis above.

## 6. Constraints honored

Read-only throughout: no process restarted/killed/signaled, cluster untouched, `ps` over ssh only (cheap), nothing CPU/GPU-heavy, nothing committed to git, `collect_telemetry.py` NOT modified. Benchmark left undisturbed.

---

### Bottom line
- Genuine runner detection: **PASS** (both nodes, exact pid/lstart).
- Avoid wrapper false positives: **FAIL** (3 launcher wrappers per node matched via embedded `-m exo`).
- Stability / no false restart: **PASS** (identical signatures S1↔S2, ~20s apart).
- Restart detection mechanics: **PASS** (signature changes on PID+lstart change; demonstrated).
- **Overall: PARTIAL-FAIL on criterion (b)** with a small, verified fix available. If wrappers are accepted as "runner siblings" in this launch chain, the practical risk today is low — but the documented corruption path is not actually prevented, so the pattern is under-specified.
