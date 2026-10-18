[BLOCK] WRITEUP.md:1201 "cd prompts"
SRC: `git ls-tree b88d65dd --name-only | grep -c '^prompts'` -> 0 (no such dir at the pinned SHA)
FIX: "mkdir -p prompts first; the capture session's prompt files and their generator are not in this repo (§11.5 text)."

[BLOCK] WRITEUP.md:1123 "Tailscale MagicDNS (node0/node1); the launcher discovers the nodes over the tailnet and"
SRC: `git show b88d65dd:start_cluster.sh | sed -n '1025,1028p'` -> M4_1_IP="192.168.86.48" / M4_1_PEER_ID="12D3KooWDGQK..." (hardcoded)
FIX: "the launcher hardcodes both nodes' IPs, libp2p peer IDs and NODES (`start_cluster.sh:1025-1028`, `:1113`); edit those before the first launch."

[FIX] WRITEUP.md:1169 "git checkout main && git submodule update --init --recursive"
SRC: `git rev-parse main` -> 1f91afe3a4d30e4a8b61a352ea7aacdb523eb53d (table at :1135 pins b88d65dd)
FIX: "`git checkout b88d65dd330478e8e1b8a565848af3c863be7a06` (the SHA in the table above), then `git submodule update --init --recursive`."

[FIX] WRITEUP.md:1173 "Shutdown is SIGTERM-only via `scripts/exo_graceful_shutdown.sh`; SIGKILL leaks RDMA queue"
SRC: `git show b88d65dd:scripts/exo_graceful_shutdown.sh | sed -n '5,6p'` -> "Runs ON the target node. start_cluster.sh delivers it over `ssh "$NODE" bash -s` STDIN"
FIX: "Shutdown is SIGTERM-only: `scripts/exo_graceful_shutdown.sh` runs ON each node (the launcher pipes it over `ssh node bash -s`), not on the controller."

[NOTE] WRITEUP.md:1249 "`reasoning_budget_tokens` field over its `0.75 x max_tokens` default (capped at 16384)."
SRC: `git show b88d65dd:src/exo/worker/engines/mlx/dsv41/reasoning_budget.py | sed -n '91,92p'` -> "if request_value is not None and int(request_value) > 0: return int(request_value)"
FIX: "specify the 16384 ceiling bounds the default only; a positive per-request value is returned verbatim, uncapped."

[NOTE] WRITEUP.md:944 "bands differ (~68 tok/s vs 98.1 at the deepest, a cross-attempt swing of the size the repo's"
SRC: evidence.md:291 -> "between-boot spread 6.02 tok/s" (decode medians ~32 tok/s); 68.3 vs 98.1 is +44%
FIX: "a 44% same-position swing (68.3 to 98.1 tok/s), larger than the 6.02 tok/s between-boot spread the corpus documents for decode."

CHECKED: 47
END S4
