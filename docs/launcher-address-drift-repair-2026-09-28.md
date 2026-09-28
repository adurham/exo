# Launcher address-drift repair (2026-09-28)

## Why

The cluster launcher could no longer complete a restart. A measurement-boot
attempt (for phase 7) aborted at "Discovering active Thunderbolt IPs" with
`ssh: connect to host 192.168.86.201 port 22: Host is down` — **fail-closed,
before the launcher's shutdown step; production was never touched** (process
elapsed time and API serving verified unchanged).

## Root cause (two independent drifts)

1. **Node ssh aliases pointed at dead DHCP addresses.** macstudio-m4-1's
   `~/.ssh/config` mapped `macstudio-m4-1` → `192.168.86.201` and
   `macstudio-m4-2` → `192.168.86.202`. Both are dead; the current addresses
   are LAN `.48`/`.47`, Thunderbolt `192.168.201.1`/`.2`, Tailscale
   `100.91.246.26` / `100.66.38.13`. Every launcher ssh hop
   (`get_node_tb_ips`, `resolve_tailscale_ip`, route repair, launch, health
   check) rides these aliases — all failed.
2. **mDNS no longer resolves the launcher's health-check name.**
   `adams-mac-studio-m4-1.local` doesn't resolve: the machine's LocalHostName
   drifted (m4-1 reports `Adams-Mac-Studio-M4-4`). The hardcoded fallback
   (`$M4_1_IP` = `.201`) is dead too, and the health-check `curl` has no
   timeout, so "Waiting for cluster to stabilize" would hang through TCP
   timeouts even when the cluster is fine.

## Fixes applied

1. On **macstudio-m4-1** (the launcher host), `~/.ssh/config`:
   `macstudio-m4-1` → `100.91.246.26`, `macstudio-m4-2` → `100.66.38.13`
   (Tailscale; stable across DHCP churn, and the same transport the gateway
   already uses). Backup: `~/.ssh/config.bak-20260928-prefix-aliases`.
   m4-2's own config was left untouched (the launcher only runs from m4-1).
   Side effect: other consumers of those aliases on m4-1 (interactive shells,
   rsyncs) now reach the nodes over Tailscale; if Tailscale is down, use the
   LAN/TB IPs directly.
2. In `start_cluster.sh`, the health-check `API_HOST` fallback chain is now
   **mDNS → Tailscale (`$M4_1_TS_IP`, already resolved at the top of the
   script) → hardcoded** (was mDNS → hardcoded).

## Verification

- `ssh macstudio-m4-1 hostname` / `ssh macstudio-m4-2 hostname` through the
  config aliases, BatchMode: resolve to the right hosts.
- The launcher's exact discovery commands (`networksetup -listallhardwareports`
  over the aliases) return both nodes' TB devices (en2–en5).
- `bash -n start_cluster.sh` + a 3-case logic harness on the patched block
  (mDNS alive / mDNS dead + Tailscale up / both dead): all pass.
- API reachable at the new host: `http://100.91.246.26:52415/state` → 200.

## NOT verified (deliberate)

The full launcher run was **not** executed — a relaunch is a separate risk
tier, and it wasn't needed (phase 7 was measured live). Known residual deltas
for the next relaunch attempt: the `IS_M4_*` self-identification still compares
against the stale `.201`/`.202` constants so the launcher takes its "remote
controller" path — the same path the last good boot (2026-09-23) took — and
launch-time discovery falls back to the Tailscale unicast target (resolved via
the now-working aliases). If the next relaunch fails, start here.
