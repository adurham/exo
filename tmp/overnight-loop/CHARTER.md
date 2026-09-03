# OVERNIGHT AUTOMATION LOOP — CHARTER (authorized 2026-09-02 by user)

## The loop (runs until the user says STOP)

1. Supervisor consults `claude-fable-5` (mcp__consult) with the latest round's results.
2. Fable returns ranked, pre-registerable suggestions.
3. Supervisor dispatches a PM subagent (agent_type='pm', role='orchestrator') to execute them.
4. PM completes -> supervisor records findings in `docs/PERFORMANCE_HISTORY.md` (commit+push same turn).
5. Supervisor feeds results back to fable. GOTO 2.

The loop is event-driven: each delegation completion re-enters the conversation and triggers the
next cycle. There must ALWAYS be exactly one PM running, or the loop stalls.

## STANDING AUTHORIZATION (granted 2026-09-02, "full automation loop overnight, no human
intervention needed")

- Cluster relaunches via `start_cluster.sh` ARE authorized for pre-registered experiments.
  This is a scoped exception to the normal relaunch-needs-explicit-yes rule, granted for the
  duration of this loop.
- Repo commits + pushes to `adurham/exo` and `adurham/hermes-agent` are authorized.
- Cluster time is authorized without per-round approval.

## HARD GUARDRAILS (never waived, even in autonomous mode)

1. **Leave the cluster HEALTHY.** Every round ends with: patches reverted, production config
   restored (shipped defaults), API responding, both nodes READY. Verify before reporting done.
2. **Root-cause only.** No mitigations, no backoffs, no retry-hacks, no defensive timeouts.
3. **Pre-registered experiments only.** Bands written BEFORE measurement; applied verbatim;
   never rationalize a near-miss.
4. **No boot experiments / no P16** (parked by user).
5. **No destructive ops:** no `git add -A`, no force push, no history rewrite, no `rm -rf`,
   no deleting another session's work. Exact-file staging only. Foreign dirty files untouched.
6. **NOT-FUNDED list stays closed:** further head-sharding, rowseq-GEMM custom kernels, tree
   drafting, context caps, Sinkhorn truncation, pad-to-M=8, Fix B (decode-KV retention),
   SDPA batched-path anomaly (c=1 makes it moot). Reopen only on the recorded trigger conditions.
7. **Bounded rounds:** each PM round <= ~2h of wall time. If a round needs more, split it.
8. **Every finding lands in `docs/PERFORMANCE_HISTORY.md`** in the same turn it is learned,
   committed and pushed. Non-negotiable (standing rule).
9. **If a round finds nothing fundable**, say so plainly to fable and ask for a different
   direction rather than manufacturing work.
10. **Escalate to the user (stop the loop) only for:** hardware faults, data loss, a change that
    would need credentials/new hardware, or a genuine safety issue. Everything else: keep going.

## Cost discipline

- fable consults: user granted runway; use freely but one consult per round (not per subagent).
- PM model: claude-opus-5 via `agent_type='pm'`. Children tiered per `dev-tier-task-routing`.

## State file

`/Users/adam.durham/repos/exo/tmp/overnight-loop/STATE.md` — updated every round. On context
loss, read the charter + STATE.md to resume without asking the user anything.
