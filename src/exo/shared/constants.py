import os
import sys
from pathlib import Path

from exo.utils.dashboard_path import find_dashboard, find_resources

_EXO_HOME_ENV = os.environ.get("EXO_HOME", None)


def _get_xdg_dir(env_var: str, fallback: str) -> Path:
    """Get XDG directory, prioritising EXO_HOME environment variable if its set. On non-Linux platforms, default to ~/.exo."""

    if _EXO_HOME_ENV is not None:
        return Path.home() / _EXO_HOME_ENV

    if sys.platform != "linux":
        return Path.home() / ".exo"

    xdg_value = os.environ.get(env_var, None)
    if xdg_value is not None:
        return Path(xdg_value) / "exo"
    return Path.home() / fallback / "exo"


EXO_CONFIG_HOME = _get_xdg_dir("XDG_CONFIG_HOME", ".config")
EXO_DATA_HOME = _get_xdg_dir("XDG_DATA_HOME", ".local/share")
EXO_CACHE_HOME = _get_xdg_dir("XDG_CACHE_HOME", ".cache")

# Models directory (data)
_EXO_MODELS_DIR_ENV = os.environ.get("EXO_MODELS_DIR", None)
EXO_MODELS_DIR = (
    EXO_DATA_HOME / "models"
    if _EXO_MODELS_DIR_ENV is None
    else Path.home() / _EXO_MODELS_DIR_ENV
)

# Read-only search path for pre-downloaded models (colon-separated directories)
_EXO_MODELS_PATH_ENV = os.environ.get("EXO_MODELS_PATH", None)
EXO_MODELS_PATH: tuple[Path, ...] | None = (
    tuple(Path(p).expanduser() for p in _EXO_MODELS_PATH_ENV.split(":") if p)
    if _EXO_MODELS_PATH_ENV is not None
    else None
)

_RESOURCES_DIR_ENV = os.environ.get("EXO_RESOURCES_DIR", None)
RESOURCES_DIR = (
    find_resources() if _RESOURCES_DIR_ENV is None else Path.home() / _RESOURCES_DIR_ENV
)
_DASHBOARD_DIR_ENV = os.environ.get("EXO_DASHBOARD_DIR", None)
DASHBOARD_DIR = (
    find_dashboard() if _DASHBOARD_DIR_ENV is None else Path.home() / _DASHBOARD_DIR_ENV
)

# Log files (data/logs or cache)
EXO_LOG_DIR = EXO_CACHE_HOME / "exo_log"
EXO_LOG = EXO_LOG_DIR / "exo.log"
EXO_TEST_LOG = EXO_CACHE_HOME / "exo_test.log"

# Identity (config)
EXO_NODE_ID_KEYPAIR = EXO_CONFIG_HOME / "node_id.keypair"
EXO_CONFIG_FILE = EXO_CONFIG_HOME / "config.toml"

# libp2p topics for event forwarding
LIBP2P_LOCAL_EVENTS_TOPIC = "worker_events"
LIBP2P_GLOBAL_EVENTS_TOPIC = "global_events"
LIBP2P_ELECTION_MESSAGES_TOPIC = "election_message"
LIBP2P_COMMANDS_TOPIC = "commands"

EXO_FILE_SERVER_PORT = 52416


EXO_MAX_CHUNK_SIZE = 512 * 1024

EXO_CUSTOM_MODEL_CARDS_DIR = EXO_DATA_HOME / "custom_model_cards"

EXO_EVENT_LOG_DIR = EXO_DATA_HOME / "event_log"
EXO_IMAGE_CACHE_DIR = EXO_CACHE_HOME / "images"
EXO_TRACING_CACHE_DIR = EXO_CACHE_HOME / "traces"

EXO_ENABLE_IMAGE_MODELS = (
    os.getenv("EXO_ENABLE_IMAGE_MODELS", "false").lower() == "true"
)

EXO_OFFLINE = os.getenv("EXO_OFFLINE", "false").lower() == "true"

EXO_TRACING_ENABLED = os.getenv("EXO_TRACING_ENABLED", "false").lower() == "true"

EXO_MAX_CONCURRENT_REQUESTS = int(os.getenv("EXO_MAX_CONCURRENT_REQUESTS", "8"))


def _int_or_none(env: str, default: int | None) -> int | None:
    val = os.environ.get(env, "")
    if not val or val.lower() in ("none", "null", "false"):
        return default
    return int(val)


EXO_MAX_CONTEXT_TOKENS: int | None = _int_or_none("EXO_MAX_CONTEXT_TOKENS", None)

# Context token limit applied to requests that required model fallback (subagents).
# Defaults to EXO_MAX_CONTEXT_TOKENS / 3 when unset.
EXO_SUBAGENT_MAX_CONTEXT_TOKENS: int | None = _int_or_none(
    "EXO_SUBAGENT_MAX_CONTEXT_TOKENS", None
)

# Output token limit applied to subagent requests to prevent massive responses
# from bloating the main conversation context. Defaults to 4096 when unset.
EXO_SUBAGENT_MAX_OUTPUT_TOKENS: int | None = _int_or_none(
    "EXO_SUBAGENT_MAX_OUTPUT_TOKENS", 10000
)

# When true, strip <system-reminder> and other tagged blocks from subagent
# input messages.  Disable when the subagent model has enough context to
# handle the full prompt from Claude Code.
EXO_SUBAGENT_TRIM_MESSAGES: bool = os.environ.get(
    "EXO_SUBAGENT_TRIM_MESSAGES", "true"
).lower() in ("true", "1", "yes")

# Rules injected into system prompts.
# Loaded from markdown files at the repo root for easy editing.
_RULES_DIR = os.path.join(os.path.dirname(__file__), "..", "..", "..")

def _load_rules(filename: str) -> str:
    path = os.path.join(_RULES_DIR, filename)
    if os.path.isfile(path):
        with open(path) as f:
            return f.read().strip()
    return ""

EXO_SUBAGENT_RULES: str = _load_rules("exo-subagent-rules.md")
EXO_MAIN_RULES: str = _load_rules("exo-rules.md")

# When set, any request for an unknown model silently resolves to this model.
# When unset (default), falls back to the sole active model if exactly one is loaded.
EXO_DEFAULT_MODEL: str | None = os.environ.get("EXO_DEFAULT_MODEL", None) or None

# When set, subagent requests (was_fallback=True) resolve to this model
# instead of EXO_DEFAULT_MODEL. Enables routing subagents to a smaller,
# faster model on a dedicated node.
EXO_SUBAGENT_MODEL: str | None = os.environ.get("EXO_SUBAGENT_MODEL", None) or None
