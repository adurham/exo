# Local Development & Cluster Operations

This document outlines the workflow for developing `exo` locally while interacting with the physical cluster.

## Environment Context

* **Local Machine**: MacBook (Controller)
* **Cluster Nodes**: `macstudio-m4-1`, `macstudio-m4-2`

**Crucial Warning**: **NEVER** run `start_cluster.sh` on your local machine. This script is designed specifically for the physical cluster nodes to initialize their environment and networking.

## Connecting to the Cluster

The cluster is managed remotely via SSH. The `start_cluster.sh` script resides on the repository root but is meant to be executed (or its logic applied) on the remote nodes, typically triggered by an orchestration command or manual SSH session.

### SSH Aliases

* `ssh macstudio-m4-1`
* `ssh macstudio-m4-2`

## Cluster Management

### Starting the Cluster

Do not run the startup script locally. Instead, use the established workflows (e.g., GitHub Actions or manual SSH) to trigger updates on the nodes.

### Troubleshooting

When debugging cluster issues:

1. **Do not rely on local logs.** Your local instance is likely just a client or a separate node.
2. **Check Remote Logs**:

    ```bash
    ssh macstudio-m4-1 "tail -f /tmp/exo.log"
    ssh macstudio-m4-2 "tail -f /tmp/exo.log"
    ```

3. **Verify Topology**:
    Check the cluster state via the primary node's API:

    ```bash
    curl http://192.168.86.201:52415/state | jq .
    ```

## Development Workflow

1. Make changes locally on the MacBook.
2. Push changes to the repository.
3. Pull changes on the cluster nodes (handled by update scripts or manually).
4. Restart services on the cluster nodes.

## Running Locally (Standalone)

You *can* run `exo` locally for testing standalone features or dashboard development, but be aware it will form a separate "cluster" of 1 node unless explicitly peered.

```bash
uv run exo
```
