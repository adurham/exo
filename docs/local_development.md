# Local Development & Cluster Operations

## Environment Context

* **Local Machine**: MacBook (controller/development — NOT a cluster node)
* **Cluster Nodes**: `macstudio-m4-1`, `macstudio-m4-2`, `macbook-m4`

## Development Workflow

1. Make changes locally.
2. Run local checks: `just check`, `just lint`, `just test`, `just fmt`.
3. Commit and push to `origin/main`.
4. Run `start_cluster.sh` — it pulls from `origin/main` on all nodes, rebuilds as needed, and starts the cluster.

## Connecting to the Cluster

SSH aliases: `ssh macstudio-m4-1`, `ssh macstudio-m4-2`, `ssh macbook-m4`

Inference is via REST API (`http://192.168.86.201:52415`), not SSH.

## Troubleshooting

1. **Check remote logs**:
    ```bash
    ssh macstudio-m4-1 "tail -f /tmp/exo.log"
    ssh macstudio-m4-2 "tail -f /tmp/exo.log"
    ssh macbook-m4 "tail -f /tmp/exo.log"
    ```

2. **Verify topology**:
    ```bash
    curl http://192.168.86.201:52415/state | jq .
    ```

3. **Snapshot logs**: `scripts/snapshot_logs.sh <label>`

## Running Locally (Standalone)

You can run `exo` locally for testing standalone features or dashboard development. It will form a cluster of 1 node unless explicitly peered.

```bash
uv run exo
```
