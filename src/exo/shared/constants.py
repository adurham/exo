"""Constants used throughout the EXO system.

This module defines file paths, topic names, and performance bounds used
across different components of the system.
"""

import os
from pathlib import Path

# Home directory configuration
EXO_HOME_RELATIVE_PATH: str = os.environ.get("EXO_HOME", ".exo")
"""Relative path for EXO home directory. Defaults to '.exo' in user's home."""

EXO_HOME: Path = Path.home() / EXO_HOME_RELATIVE_PATH
"""Base directory for all EXO configuration and data files."""

# Model directory configuration
EXO_MODELS_DIR_ENV: str | None = os.environ.get("EXO_MODELS_DIR")
"""Environment variable override for models directory."""

EXO_MODELS_DIR: Path = Path(EXO_MODELS_DIR_ENV) if EXO_MODELS_DIR_ENV else EXO_HOME / "models"
"""Directory where model files are stored."""

# Database file paths
EXO_GLOBAL_EVENT_DB: Path = EXO_HOME / "global_events.db"
"""SQLite database for global events."""

EXO_WORKER_EVENT_DB: Path = EXO_HOME / "worker_events.db"
"""SQLite database for worker events."""

# State file paths
EXO_MASTER_STATE: Path = EXO_HOME / "master_state.json"
"""JSON file for master state persistence."""

EXO_WORKER_STATE: Path = EXO_HOME / "worker_state.json"
"""JSON file for worker state persistence."""

# Log file paths
EXO_MASTER_LOG: Path = EXO_HOME / "master.log"
"""Log file for master component."""

EXO_WORKER_LOG: Path = EXO_HOME / "worker.log"
"""Log file for worker component."""

EXO_LOG: Path = EXO_HOME / "exo.log"
"""Main log file for EXO."""

EXO_TEST_LOG: Path = EXO_HOME / "exo_test.log"
"""Log file for tests."""

# Keypair and keyring files
EXO_NODE_ID_KEYPAIR: Path = EXO_HOME / "node_id.keypair"
"""File storing the node's libp2p keypair for identification."""

EXO_WORKER_KEYRING_FILE: Path = EXO_HOME / "worker_keyring"
"""Keyring file for worker authentication."""

EXO_MASTER_KEYRING_FILE: Path = EXO_HOME / "master_keyring"
"""Keyring file for master authentication."""

# IPC directory
EXO_IPC_DIR: Path = EXO_HOME / "ipc"
"""Directory for inter-process communication files."""

# libp2p topic names for event forwarding
LIBP2P_LOCAL_EVENTS_TOPIC: str = "worker_events"
"""Topic name for local events from workers."""

LIBP2P_GLOBAL_EVENTS_TOPIC: str = "global_events"
"""Topic name for global events from master."""

LIBP2P_ELECTION_MESSAGES_TOPIC: str = "election_message"
"""Topic name for master election messages."""

LIBP2P_COMMANDS_TOPIC: str = "commands"
"""Topic name for commands to workers."""

# Performance lower bounds (used for timeouts)
# These values are based on M1 chip specifications
LB_TFLOPS: float = 2.3
"""Lower bound for TFLOPs performance (used for timeout calculations)."""

LB_MEMBW_GBPS: float = 68
"""Lower bound for memory bandwidth in GB/s (used for timeout calculations)."""

LB_DISK_GBPS: float = 1.5
"""Lower bound for disk bandwidth in GB/s (used for timeout calculations)."""
