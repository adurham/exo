# DSv4.1 restart-survival — Phase 0 read-only audit (2026-10-10)

**Scope.** Executes the "Phase 0 — read-only audit (A1–A12)" section of
`docs/dsv41-restart-survival-plan-2026-10-10.md` (authored 0ae8f4f78). Read-only
on both cluster nodes; no model requests; no writes/relaunches on nodes. All node
commands were `ls/cat/head/tail/wc/du/stat/grep/sed -n/od/find read-only/git
log|reflog read-only/launchctl list/ps`. Raw per-node extracts were written
locally (laptop) to `/private/tmp/p20-audit-raw/{studio1,studio2,code}/` (not
committed). This doc is the deliverable.

**Evidence base / caveats.**

- Nodes: studio1 = Adams-Mac-Studio-M4-1 (192.168.86.48), studio2 =
  Adams-Mac-Studio-M4-2 (192.168.86.47). Both up ~3d at audit time (19:37 CDT).
- **Log retention is the binding constraint.** Earliest surviving exo log on
  either node = **2026-10-10 09:07** (`~/exo.log.prev`, 09:07→12:04); current
  `~/exo.log` covers 12:04→19:48; `~/.exo/exo_log/` holds only Oct-10 10:47→19:48
  `.zst` rotations. **No Oct 4–9 exo logs exist on either node.** Any event before
  09:07 Oct 10 is unrecoverable from logs.
- Code source of truth: worktree `/private/tmp/p20-restart-audit` = `origin/main`
  0ae8f4f78. Production = `b5f055185` (deploy/next20-vision). `b5f055185` is **not
  an ancestor of main**, but `git diff b5f055185 HEAD -- .../dsv41/park.py` is
  **empty** → park.py is byte-identical between production and main, so the code
  findings below apply to the running cluster.
- **The surviving-log window is dominated by benchmark/probe traffic** ("Archival
  brief…" and "What single dominant color is this image?" prompts), so the
  user's *specific* ~92.6K conversation is not individually isolable in it.

---

## A1 — When did each relaunch happen?

**Method.** grep startup banners in `~/exo.log(.prev)` + node `git reflog`.

**Evidence.** studio1 startup banners today: 09:07:50, 09:10:36, 09:47:22,
10:07:39, 10:31:10, 10:47:57, 11:05:01, 11:21:56, 11:39:03 (in `.prev`);
**12:04:38** (pid 20277) and **17:33:04** (pid 42211) in `exo.log`. studio2:
**12:04:39.520** (pid 31555) and **17:33:00.126** (pid 53441) in `exo.log`.
Reflog (identical on both nodes): `reset: moving to b5f055185` @ **17:32:45/47**;
`checkout` to `99e2966ee` @ **12:00:46**; plus Oct 8–10 reset/checkout traffic
(09:04:40, 22:40:04, 22:09:47, 20:31:50, 15:18:00, 13:04:58, 12:05:22, 10:28:49,
10:03:51 …). Relaunch follows the reflog reset/checkout by ~13–19 s.

**Finding.** ≥11 exo relaunches on Oct 10 alone (a heavy deploy day); the two
afternoon ones (12:04, 17:33) are the ones whose post-relaunch behavior is
observable. Oct 4–9 relaunch times are recoverable only from reflog, not logs.

**Implication.** Relaunch frequency is high (multiple/day) → cold-start cost is
recurring, not a one-off. Log retention (~12 h rolling) means Phase-0 forensics
must lean on reflog + park-store mtimes for anything older.

---

## A2 — What did the first request after each relaunch do?

**Method.** Classify the first `received chat request` after each relaunch banner
as restore-by-key / prefix-restore / cold, using the preceding `prefill controls
… rows=N` and any `session reuse` / `parked session restored` line.

**Evidence.**

- Post-**12:04** relaunch: first real request 12:07:12 → `prefill controls …
  rows=20077` — **COLD**; second 12:08:33 rows=20077 cold; **zero** `parked
  session restored` lines after 12:04 on either node.
- Post-**17:33** relaunch: warmup 17:41:56 (rows≈11/12) then benchmark+vision
  probes 17:42–18:36; **zero** restore/reuse lines.
- Cold full-prefill calibration: **rows=91,043 × 12 on each node**, 09:54:33 →
  11:49:19, recurring every ~6–9 min *within a single build window* (so NOT
  restart-caused). First one: request 09:54:32.835 → TaskFinished 10:00:51.893 =
  **379 s** (~240 t/s). Each was immediately preceded by `parked an idle
  conversation to SSD (20,877/20,878 rows)` → these are the `SESSION-SALT
  q2gamma-0/1/2` probes: a fresh token-0 salt makes the parked 20.8K conv
  non-matching → full 91,043-row cold every run.
- **Calibration event confirmed:** rows=123,186 @ **16:02:41.056** (request
  16:02:40.804 → TaskFinished 16:11:04.019 ≈ 503 s); template head `Reasoning
  Effort: Absolute maximum…`; a non-parking evict fired at 16:02:41.044
  immediately prior. This is a token-0 effort-preamble change, **not** a relaunch.

**Finding.** Both Oct-10 afternoon relaunches were followed by **cold** first
requests with **zero** restores. The largest cold in the surviving logs is
123,186 rows (the preamble calibration); the ~92.6K/282 s user event is **not in
surviving logs** (max rows present = 123,186). The recurring 91,043-row colds
(~379 s) are the closest logged proxy for a full cold re-prefill.

**Implication.** The ~282 s figure is consistent in magnitude with logged colds
(~379 s at ~240 t/s); a restart that loses the active conversation costs
multiple minutes of TTFT. Confirms the cost side of the hypothesis.

---

## A3 — Was the lost conversation ever parked?  *(go/no-go gate)*

**Method.** (a) enumerate every park trigger in code; (b) classify post-relaunch
requests; (c) park-dir ↔ conversation prefix.

**Evidence.**

- **Code — park fires ONLY on idle eviction.** The sole park call site is
  `_evict(...)` (`session.py:1096`, called at `:1093` from the LRU get path);
  `_evict` → `store.park(session, key=…)` (`session.py:1110`). `drop()`
  (`session.py:1129`) and one-off `close()` never park. `grep -rE
  'atexit|add_signal_handler|SIGTERM|SIGINT|shutting_down|maybe_checkpoint|
  on_shutdown'` over `…/dsv41/` returns **nothing** (the only "checkpoint" hits
  are model-weight docstrings). **There is no after-turn checkpoint and no
  park-on-shutdown.** A resident (active) conversation is therefore never parked.
- **Logs.** Post-12:04 and post-17:33 first requests were cold with zero
  restores (A2). By contrast **15 restores (10:33:35 → 11:42:23) restored the
  idle-evicted 3-conversation set** (keys `p:8a28da3b88f4e3d36aee631241489de2`,
  `p:2039f2fa7fae90af98bd767ce2afc09a`, `p:629c1503e60553aab81b1167cc4c9b61`,
  ~20,877 rows each) → parked conversations **do** survive and restore across
  relaunches; the ones that were never parked do not.
- **Park-dir set (studio1).** 44 dirs, every manifest `"key": null`; token
  distribution ≈ 16 × ~91.8 K (cap 131072), 22 × ~20.8 K (cap 65536), 3 × ~8 K,
  1 × 40,189, 1 × 20,240, 1 × 20,477; mtimes **Oct 8 17:23 → Oct 10 15:10**.

**Finding.** **HYPOTHESIS CONFIRMED — the active conversation was never parked.**
This is unambiguous at the code level (park is idle-eviction-only) and
corroborated behaviorally (cold-after-relaunch, zero restores). Caveat: the
specific ~92.6K user event predates surviving logs, so confirmation is
mechanism-level + behavioral, not a single logged trace. The ~91.8 K park dirs
are probe artifacts (repeated ~91.8 K parks with distinct `prefix_hash`es across
Oct 8–9), not the user's single growing conversation.

**Implication.** **GO.** The fix must add *after-turn idle checkpointing* (G1);
park-on-shutdown alone would miss nothing-crashes and mid-turn cases and cannot
cover a killed process.

---

## A4 — LCP divergence in practice

**Method.** Compare parked `ids.bin` heads vs later request prefixes; inspect the
recurring colds; check for a persisted snapshot ≤ LCP.

**Evidence.**

- All 44 `ids.bin` share the same first 8 int32 = `0 128803 61 82004 …` (shared
  system-prompt/template head) → LCP extends at least through the common head.
- The **12× recurring 91,043-row colds are the same prompt length but cold every
  run** → each probe rewrites the token-0 salt, collapsing LCP to ~0.
- The **16:02 123,186-row cold** follows a token-0 preamble change → LCP ≈ 0 →
  full cold despite any resident/parked cache.
- park.py **does** persist rewind snapshots (`snapshot.<pos>.ring/carry/carrysc.
  <layer>.bin`) and prompt-end anchors (`anchor.<pos>.bin`, e.g.
  `anchor.91044.bin` = 517120 B in a 91.8 K dir), but a **token-0** divergence
  leaves **no** position ≤ LCP to rewind to.

**Finding.** LCP divergence is **REAL and located at TOKEN 0** (template /
preamble / salt), not only mid-stream stripped reasoning. Any change to the head
invalidates the whole parked prefix; snapshots cannot recover a head change.

**Implication.** The whole-prefix-from-token-0 sensitivity dominates. G6
(rewind-to-LCP) helps only mid-stream divergence and should be **deprioritised**;
the real mitigation is to fingerprint the template/head and honestly take a cold
start when the head changed.

---

## A5 — Are cross-build restores already happening?

**Method.** Restore-event timestamps vs restored-dir manifest mtimes vs build
switches (reflog); inspect manifest fingerprint fields.

**Evidence.**

- 15 restore events (10:33:35 → 11:42:23, both nodes) restored dirs whose
  manifests date **Oct 8 17:23 → Oct 9 23:59**; reflog shows **≥12 reset/checkout
  build switches Oct 8 17:00 → Oct 10 12:00** → **≈12 of 15 restores are
  structurally cross-build**.
- Manifests carry **no build fingerprint**: keys are `magic(=DSV41-PARK)`,
  `version(=1)`, `created, key, offset, capacity, max_seq_len, n_layers, window,
  bsz, token_count, prefix_key, ids, n_turns, cache_turns, base, gen, …` — no
  code/model/mlx/layout hash.
- Corroboration that the source changed between dir creation and restore: the
  `_evict` log line number moved **1120/1125 → 1145** between the Oct-8/9 build
  and the 12:04 Oct-10 build.

**Finding.** **Cross-build restores are CONFIRMED to have already served
traffic (~12/15), unguarded.** Silent staleness (including sampling from
old-build exact-repeat anchors) is a realised risk, not hypothetical.

**Implication.** **A5 > 0 → ship the build fingerprint + refusal policy (G2)
immediately.** Treat the existing 44 legacy dirs as foreign (refuse, do not
quarantine; LRU reclaims).

---

## A6 — What does restore do to the dir?

**Method.** Code read of the restore path.

**Evidence.** `park.py` `try_restore(...)` docstring: *"Find, restore, and
consume a parked session"*; on success `self.remove(d)` (`park.py:776`) →
`remove` = `shutil.rmtree(d, ignore_errors=True)` (`park.py:702-703`). Bad dir →
`_quarantine(d)` (`park.py:773-775`) → `os.replace(d, f"{d.name}.bad")` else
rmtree (`park.py:693-700`). `park_conversation` refuses to clobber:
`if d.exists(): return False` (`park.py:279-280`). Observed cycle in logs:
restore 10:33:35 → re-evict 10:34:50 → re-park 10:34:50.927 (new uuid, same
`prefix_hash`).

**Finding.** **Restore is CONSUMING** — it deletes the dir. After a restore the
conversation is RAM-only and is lost at the **next** restart unless it is
idle-evicted and re-parked first.

**Implication.** A parked restore buys exactly **one** restart. G5
(non-consuming restore + generation retention) is a prerequisite before G1
relies on parked state.

## A7 — Topology + multi-rank park/restore coordination

**Method.** Code read of park/session for rank/world-size handling; live park-dir
comparison across nodes.

**Evidence.**

- **No coordination exists.** `grep` for `rank|world_size|barrier|all_gather`
  in `park.py`/`session.py` is clean. The store is built per worker
  (`engine.py:342`; `session.py:1019-1043` builds a bare `ParkedStore` with no
  rank/root override). Each rank parks (`session.py:1096-1127`) and restores
  (`session.py:1052-1076`) **independently** — no barrier, no cross-rank commit,
  no shared "parked" record.
- Dir naming: `key_hash = prefix_hash(tokens)`; `d = root / f"{key_hash}__
  {uuid.uuid4().hex[:8]}"` (`park.py:722-723`). The 8-hex suffix is a **random
  uuid4, not a rank/host salt** (studio1 `__2b60c6ef` vs studio2 `__09f79b88`
  are two randoms). Root = `~/.exo/dsv41_park`, local per node.
- Live: **44 dirs on each node**; full names identical **0/44** (suffix differs);
  **minus suffix 44/44 exact match** → same conversations (deterministic
  `prefix_hash` from the shared token stream), mirrored per rank.

**Finding.** TP2 ranks park/restore with **zero coordination**. Nothing prevents
rank A restoring while rank B cold-prefills the same request → collective hang /
state skew. **Blind spot #1 CONFIRMED.**

**Implication.** **G4 (all-or-nothing, cluster-level, generations) is mandatory**
for this deployment.

---

## A8 — How is shutdown performed?

**Method.** Read the deploy/shutdown scripts; inspect live launch mechanism.

**Evidence.**

- **No launchd for exo** (`launchctl list | grep -i exo` empty on both; only
  netwatch/tailscale/CleanMyMac agents). exo runs under **`screen -dmS exorun`**
  (`start_cluster.sh:3474`) ⇒ **no `ExitTimeOut` exists**.
- Graceful path: `scripts/exo_graceful_shutdown.sh:137-139` `kill -TERM`, wait
  `GRACE` (`:83` default **15 s**), `:154-159` escalate to `kill -9` if
  survivors; invoked at `start_cluster.sh:1537`.
- Ordering (worktree == production for park.py; launcher from production branch):
  loop A kills the old process (`:1495/:1537`) → **rsync** (`:1641-1653`) →
  build → loop B launches (`:1856` → `:3474`). **`git reset --hard` no longer
  exists** (removed 2026-08-15; only comments remain). Node `~/relaunch_exo.sh`
  (`:3403`) also SIGTERM → 20 s → kill -9.
- Launch cmd (`start_cluster.sh:3400`): `cd ~/repos/exo && $EXO_ENV … .venv/bin/
  python -m exo -v >> ~/exo.log 2>&1 & … caffeinate -s … wait`.

**Finding.** Graceful shutdown = SIGTERM → ≤15 s → SIGKILL. The old process dies
**before** new source lands ⇒ **R3 does not occur**. The plan's G3 premise
("0.7 × supervisor ExitTimeOut") is **wrong**: there is no launchd/ExitTimeOut;
the real budget is the script's 15 s GRACE.

**Implication.** Correct G3 to key off `EXO_SHUTDOWN_GRACE_SECONDS` (default 15 s)
and keep the kill→rsync order. Any launcher change must land on the **production
branch** (worktree `origin/main` `start_cluster.sh` lacks `DSV41_DENSE`; the
production branch carries `DSV41_DENSE=affine6`).

---

## A9 — Park/restore cost

**Method.** Inspect park-dir sizes and any timing log lines.

**Evidence.**

- **No park/restore duration is logged** (park logs only success+rows; restore
  logs rows) → no direct p50/p95 GiB/s.
- Store size: **11,575,736 KB ≈ 11.04 GiB / 44 dirs** on studio1. Largest class
  (~91.8 K rows) = **482,428 KB ≈ 471 MiB each** (dominated by `comp_kv.*.bin`
  at 47–94 MB per layer). A 20.8 K-row dir is proportionally smaller.
- Anchor blob sanity: `anchor.91044.bin` = 517120 B (prompt-end logits for a
  91.8 K session).

**Finding.** **A9 is PARTIAL.** A 92 K-row session park ≈ **471 MiB** written;
whole-store 11 GiB. Exact throughput is **not** derivable from logs.

**Implication.** Sub-second per-session park/restore on NVMe is plausible, but
the shutdown budget must assume the measured **471 MiB/session** figure.
**NEEDS-GO** for an exact throughput number (requires instrumented run; out of
read-only scope).

---

## A10 — Memory headroom

**Method.** One-shot read-only: `sysctl hw.memsize`, `vm_stat`, `memory_pressure`,
`ps aux`.

**Evidence.** `hw.memsize` = 137,438,953,472 B (**128 GiB**). studio1: pages free
5,784 (~90 MB); wired ~6.9 M pages (**~108 GB**); swapins 67.0 M / swapouts
67.4 M (historical heavy swap); exo main RSS 286.8 MB + worker fork RSS
**18.98 GB (14.1 %)**. studio2: free ~8.6 GB; wired ~105 GB; exo ~238 MB + MLX
fork ~18.0 GB.

**Finding.** Unified memory is **near-saturated** (wired 105–108 GB of 128 GB;
free 0.09–8.6 GB). There is **no large contiguous CPU-copy headroom** available.
R5 is real. Exact peak-at-92K-session is not measured (would need a run).

**Implication.** Park must **stream one layer at a time** (copy→write→release);
never stage a whole session in CPU RAM. **A10 PARTIAL** (peak-at-92K needs a run).

---

## A11 — Do active conversations carry images?

**Method.** grep chat requests for image/vision; grep forced-cold lines.

**Evidence.** Yes. studio1: 3 image requests (17:42:57, 17:44:41, 18:36:37) and
**6×** `[DSV41] session forced cold: image-carrying request must not reuse a
resident conversation` (17:42:45–18:36:37); vision tower loaded 12:06:48
(`image_token_id=129264`). studio2: 8 image requests (12:06→18:36), vision loaded
4× (once per build), **5×** forced-cold.

**Finding.** Active conversations **do** carry images, and image-carrying
requests are **forced cold by design** (never reuse/park). **R7 confirmed in
scope.**

**Implication.** Parked restore does not cover image conversations. If the user's
active conversation carries an image, G1 alone will not cover it — needs a
separate plan item.

---

## A12 — Is the conversation key stable across processes?

**Method.** Read the key derivation; grep for per-process salts; corroborate with
live log keys before/after relaunch.

**Evidence.** `session.py:972-977` `key_for`: `if key: return f"id:{key}"` else
`return "p:" + _sc.prefix_hash(_ids_of(prompt_ids))`; `prefix_hash` =
`hashlib.blake2b(…, digest_size=16).hexdigest()` (`mlx_lm/models/deepseek_v41/
session_cache.py:112-120`) — **deterministic, platform-independent**. No
`os.getpid/urandom/secrets/random/time` anywhere in the key path; `uuid4` appears
only in park dir/tmp names. Live proof of stability: the **same 3 keys**
(`p:8a28da3b…`, `p:2039f2fa…`, `p:629c1503…`) are parked 09:52–10:00, restored
**10:33 (after the 10:31 relaunch)**, re-parked 10:34, restored 10:50 (after a
later relaunch).

**Finding.** **Key IS stable across restarts** — the plan's R6 fear ("unstable
key → restore-by-key never hits post-restart") is **KILLED**. All live manifests
carry `"key": null`, so these sessions are keyed by prefix and restore uses the
**keyless prefix-match branch** (`session.py:1011`), which **does** hit
post-restart (proven).

**Implication.** No key fix needed. Post-restart recovery rides on prefix-match,
which inherits A4 (token-0 sensitivity) and A6 (consume) — those are the real
constraints, not key stability.

---

## Go / no-go

Plan gate: *proceed if A3 = "never parked"; if A5 > 0, ship the fingerprint
immediately.*

- **A3 = never parked** (code-definitive) → **GO**.
- **A5 ≈ 12/15 restores cross-build, no fingerprint** → **ship G2 now**.
- **A6 = consuming** and **A7 = zero multi-rank coordination** → both must be
  fixed before G1 is safe.

**Verdict: GO on the core hypothesis and the plan's fix direction (after-turn
idle checkpoint as the primary mechanism).** The mechanism is not "park-on-
shutdown"; it must be **G1 after-turn idle checkpointing**, gated by **G2**
(fingerprint) and preceded by **G5** (non-consuming restore) and **G4**
(multi-rank).

---

## The plan's four blind spots

1. **Multi-rank consistency (A7) — CONFIRMED RISK.** Each TP2 rank parks/restores
   independently, no barrier/commit; store is node-local; suffix is random (not a
   rank id). A one-rank restore vs other-rank cold-prefill can hang the
   collective. → **G4 mandatory.**
2. **Silent staleness / cross-build restores already happening (A5) —
   CONFIRMED.** ≈12/15 restores crossed build switches; manifests carry no
   fingerprint; exact-repeat anchors mean sampling from old-build logits is
   possible. → **ship G2 immediately; refuse the 44 legacy dirs.**
3. **Restore-consume semantics (A6) — CONFIRMED CONSUMING.** `try_restore` →
   `rmtree`; a restored conversation is RAM-only and lost at the next restart.
   → **G5 non-consuming restore required.** Without it, parked restore survives
   exactly one restart.
4. **LCP divergence (A4) — CONFIRMED, and at TOKEN 0.** Template/preamble/salt
   changes at the head collapse LCP to ~0 (16:02 123,186-row cold; 12× recurring
   91,043-row colds). Snapshots cannot rewind a head change. → mitigations that
   assume mid-stream divergence (G6) are **low value**; fingerprint the head and
   accept cold on head change.

---

## Ranked next steps

1. **G2 — build fingerprint + refusal policy.** *Highest.* A5 proves cross-build
   restores already happen with no guard; it is a prerequisite for G1 (which will
   make restores common). Cheap (~1 d). Refuse the 44 legacy `key:null` dirs.
2. **G5 — non-consuming restore + atomic generations + fsync.** A6: restore
   currently consumes, so a restore buys one restart only; without this, G1's
   whole value is capped. (~0.5–1 d.)
3. **G1 — after-turn idle checkpoint.** *Core fix for A3.* Add per-session dirty
   tracking + `maybe_checkpoint()` on the engine thread when idle; keep the
   session resident. (~2–3 d.)
4. **G4 — multi-rank all-or-nothing restore + generations.** Mandatory for TP2
   (A7). (~2–4 d.)
5. **G3 — park-on-shutdown (fast path).** Correct the budget to the real
   `EXO_SHUTDOWN_GRACE_SECONDS`=15 s (no launchd/ExitTimeOut); keep the
   kill→rsync order (R3 struck). (~1–1.5 d.)
6. **(low) G6 — rewind-to-LCP.** A4 shows divergence is at token 0, where rewind
   is useless; deprioritise.
7. **G7 — incremental checkpoints.** Defer.

---

## Corrections to the plan's stated assumptions (audit contradicts)

- "Park dirs span **Oct 4–10**" → actually **Oct 8 17:23 → Oct 10 15:10**.
- "studio1 **46** dirs / studio2 44" → **both 44**.
- "**~15.6 K** park/restore log lines" → park/restore/evict lines are **~19/15/21
  per node** in surviving logs; 15.6 K does not match park/restore events (likely
  counts all `[DSV41]` lines).
- "restore-by-key" → in practice **keyless prefix-match** (all manifests
  `"key": null`).
- G3 "0.7 × supervisor **ExitTimeOut**" → **no launchd**; budget = script
  `GRACE`=15 s.
- **R3** (reset-before-signal) → **does NOT occur** (no `git reset --hard` since
  2026-08-15; kill precedes rsync).
- **R6** (unstable key) → **KILLED** (key is deterministic blake2b).
- "**~92.6K ≈ 282 s**" → the exact event **predates surviving logs**; closest
  logged cold = 91,043 rows ≈ 379 s. Use 471 MiB/session for park sizing.

**Marked NEEDS-GO (not answerable read-only):** A9 exact park/restore throughput
(no timing instrumentation); A10 exact peak unified memory *at* a 92 K session
(needs a run). Both require an instrumented boot — deferred per the read-only
constraint.

