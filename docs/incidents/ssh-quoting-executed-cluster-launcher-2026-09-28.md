# Incident — quoted `ssh` command executed the cluster launcher on a live cluster (2026-09-28)

**Severity: near-miss, no impact.** Production was not disturbed (verified four
independent ways, below). Recorded because the failure mode is one command
character away from a full production outage.

## What was attempted

Reading a variable out of the launcher script over SSH — a plain inspection:

```
ssh macstudio-m4-1 'grep -n 'DSV4_MODEL_ID\|EXO_DSV4\|env ' ~/repos/exo/start_cluster.sh | head -30'
```

## What actually ran

The nested single quotes closed the outer quote at the first inner one. The
shell then re-parsed the remainder, and the fragment `env ~/repos/exo/start_cluster.sh`
is **`env PROGRAM` — which EXECUTES the program.** The launcher ran.

Observed on the node:

```
zsh:1: command not found: EXO_DSV4
Starting cluster setup...
-----------------------------------------------------
Discovering active Thunderbolt IPs...
CRITICAL ERROR: Could not map Studio-to-Studio Thunderbolt topology!
```

## Why production survived (this is luck, not design)

`start_cluster.sh` performs a **graceful-shutdown step as part of normal
startup** — its own teardown gate, which prints `EXO_SHUTDOWN_VERDICT=CLEAN_EXIT`
or `=ALREADY_DEAD`. Had the run proceeded past its early phases, that step would
have stopped the live cluster.

It **aborted before reaching the shutdown phase**, because Thunderbolt topology
discovery failed in that non-interactive context (`Could not map Studio-to-Studio
Thunderbolt topology!` → `Unknown host IPs`). The same condition is documented
in the skill as a known degradation ("could not resolve Tailscale IPs … falling
back to LAN IPs … expect split-brain under macOS 27 unless exo is launched from
an interactive session"). Here the degraded path **failed closed** and exited.

A different environment (resolvable Thunderbolt/Tailscale) would have taken
production down.

## Proof production was untouched

Four independent checks, all on the live nodes:

1. **Process start times** — the exo processes are `Wed Sep 23 19:07:26 2026`,
   elapsed `04-01:15:16`. They predate the incident by four days. The runner PID
   (61922 / 63999) is the same one logged on Sep 23.
2. **No restart in the log** — grepping the last 2000 lines for
   `Shutdown|CreateRunner|bootstrap|graceful|SIGTERM|terminating|Starting cluster`
   returns nothing. The only lines in the 20:2x window are two routine
   `fetch_file_list` HF warnings.
3. **No errors** — no exception/traceback/failure lines in the window.
4. **Real generation at baseline speed** — a 256-token completion returned
   **21.8 tok/s** decode, against the documented 20.86 tok/s mean baseline.
   A cold first request measured 58 s (JIT/page-in after the node had paged out;
   `Pageouts: 1849`), then 1.6 s for 24 tokens once warm.

## The rule

**Never build an inspection command by nesting quotes inside an SSH one-liner.**
Write the script to a file, `scp` it, and run it by path. This session already
had a documented pitfall for the same class of failure
(`~/repos/exo/docs/…` "nested-SSH heredoc quoting fails repeatedly → write
scripts as files and scp them"); the mistake was reading `env` as an
inert argument to `grep` when the quoting had already broken, which turned an
inspection into an execution.

Second rule, specific to this repo: **`start_cluster.sh` is not safe to invoke
speculatively.** It shuts down whatever is running as part of its startup
sequence. Treat any accidental invocation as a potential production event and
verify the four checks above immediately.
