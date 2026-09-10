# `pkill -f` self-match forces a SIGKILL of exo on the control node

**Found:** 2026-09-10, during the (successful) Stage A redeploy of `origin/main`
onto the two M4 Studios. Not the bug Stage A was chasing — found incidentally
while auditing why the *previous*, failed Stage A attempt had escalated to
`pkill -9`.

**Severity:** high latent. It does not break a deploy, but it silently spends
the one recovery lever this project does not have (see §5), on **every single
run**, and it has been doing so for as long as the deploy has been driven from
a Studio.

---

## 1. Symptom

Every `start_cluster.sh` run driven from `macstudio-m4-1` logs, for the m4-1
node iteration only:

```
Killing existing Exo processes on macstudio-m4-1...
  WARNING: Exo on macstudio-m4-1 did not exit on SIGTERM after 15s — escalating to SIGKILL (may leak RDMA QPs; reboot if TB wedges).
./start_cluster.sh: line 1214: 95846 Killed: 9               ssh "$NODE" "pkill -9 -f 'exo.main' || true"
./start_cluster.sh: line 1214: 95853 Killed: 9               ssh "$NODE" "pkill -9 -f 'python.*exo' || true"
```

`macstudio-m4-2` in the same run is killed silently, with no warning.

Reproduced 4/4 on m4-1 and 0/4 on m4-2 across the 2026-09-09 incident logs
(`deploy.log:62`, `rollback.log:32`, `recover.log:28`, `recover.log:908`;
m4-2 counterparts at lines 489 / 455 / 451 / 1331).

## 2. Root cause

The kill sequence checks whether exo is still alive with

```bash
ssh "$NODE" "pgrep -f 'python.*exo'"
```

When `$NODE` is the node the script is *running on*, that `ssh` client process
is itself local and visible to `pgrep`. Its own argv contains the literal
pattern text `python.*exo`, and the ERE `python.*exo` matches that literal
string (`python`, `.*` consumes `.*`, `exo`).

So on the self-node the liveness probe **matches its own command line** and
returns a hit even when exo is completely dead. Demonstrated directly, with
exo confirmed dead:

```
$ ssh macstudio-m4-1 "pgrep -lf 'python.*exo'"
22670 ssh macstudio-m4-1 pgrep -lf 'python.*exo'      <-- the probe matching itself
```

Consequences, in order:

1. The 15 s "did it exit?" wait **can never succeed** on the self-node.
2. The script therefore **always** escalates to `pkill -9` on the self-node,
   regardless of whether exo already exited cleanly on SIGTERM.
3. The `Killed: 9` lines are the same bug biting a second time: the
   `pkill -9 -f 'python.*exo'` kills the *sibling ssh client processes* that
   are themselves carrying the pattern in their argv, which is why bash
   reports them as killed.

`exo.main` self-matches identically — verified — so all three escalation
commands are affected, not just the first probe.

## 3. Why it matters

`pkill -9` skips the C++ static-duration destructors that free RDMA queue
pairs and release Metal buffers. The script's own comments say so, and the
generated `~/relaunch_exo.sh` repeats the warning:

> Never use `screen -X quit` / `pkill -9` here: both skip the destructors,
> leaking QPs (TB-stack wedge) and orphaning ~60-80 GB of wired pages the OS
> then takes ~a minute (or a reboot) to reclaim.

The documented escape hatch for a leaked-QP Thunderbolt wedge is **reboot the
node**. Per `docs/DSV4_VISION_PORT_PHASE5_PROCEDURE.md` §1.10 that hatch is
unavailable to an agent session: FileVault is ON, `sudo -n reboot` needs a
password, and the sandbox `op` CLI has no broker socket. So this bug quietly
rolls the dice on an unrecoverable state on every deploy.

## 4. Mitigation used for the 2026-09-10 Stage A deploy (no code change)

The deploy did **not** patch `start_cluster.sh` — editing a script while bash
is executing it is itself hazardous (bash reads scripts incrementally by byte
offset; a mid-run edit shifts offsets and is the most likely cause of the
`recover.log` `line 3206: syntax error near unexpected token 'done'`, which
appeared while the file was being hand-edited and did not reproduce under
`bash -n`).

Instead the shutdown was performed **by hand, before** invoking the deploy,
with a self-match-immune, pattern-complete check:

- Both patterns written with the bracket trick (`[p]ython.*exo`, `[e]xo.main`)
  so they cannot match the text that launched them.
- The checker lives in a **file invoked by path**, so no pattern text appears
  in any `ssh`/shell argv at all.
- The check is pattern-complete: `python.*exo` **and** `exo.main` **and** the
  `lsof -ti:52415,52416` port holders — matching all three patterns the kill
  sequence uses.
- SIGTERM only; the gate **fails by design** rather than escalating.

Result: both nodes reported `CLEAN_EXIT after 1s (NO SIGKILL USED)`, RDMA ports
stayed `PORT_ACTIVE`, the TB link stayed at 0% loss, and wired+compressor
drained to 3.6 / 3.8 GB.

Once exo is already cleanly dead, the script's guaranteed-to-fire `pkill -9`
on the self-node hits nothing real — it can only kill sibling ssh clients,
which is cosmetic. The escalation warning still appeared in the deploy log,
**as expected**, and was verified to be a no-op.

Two residual conditions were checked explicitly rather than assumed:

- **No auto-respawn** could refill the window between the manual check and the
  script's SIGKILL: neither node has a crontab, an exo LaunchAgent/LaunchDaemon,
  or a loaded launchctl job matching exo. `~/relaunch_exo.sh` is a generated
  artifact, invoked only by hand.
- **Pattern completeness**, as above — checking only `python.*exo` would have
  left the `exo.main` and port-holder escalations unproven.

## 5. Real fix (not done here — deliberately out of Stage A's scope)

Make the liveness probe unable to see itself. Any of:

- Use the bracket trick in the probe and the kills:
  `pgrep -f '[p]ython.*exo'`.
- Anchor on the real command instead of a substring, e.g.
  `pgrep -f '\.venv/bin/python -m exo'`, and/or match with `pgrep -x`.
- Exclude the probe's own PID and its ancestors (`pgrep -f ... | grep -v "^$$"`).
- Prefer the port holders (`lsof -ti:52415,52416`) as the liveness signal,
  since that cannot be spoofed by a command line.

Best combined with the still-outstanding root-cause fix from `04524f3a`'s
commit message — having warmup emit progress events so the pre-serving
watchdogs do not need operator-level timeout overrides at all.

## 6. Cross-reference

The 2026-09-09 cold-start incident (`04524f3a`) and this bug are **independent**.
That one fires on the *startup* path of the new process (283 s warmup exceeding
both pre-serving watchdogs); this one fires on the *shutdown* path of the old
process, before any new code is launched. They share no state. Fixing the
timeout forwarding does not address this, and this was still firing on the
successful 2026-09-10 deploy — as a verified no-op, only because of the manual
gate in §4.
