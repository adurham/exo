# TODO: Do we want so many constants?
#  I think we want a lot of these as parameters?

import os


def _int_or_none(env: str, default: int | None) -> int | None:
    val = os.environ.get(env, "")
    if not val or val.lower() in ("none", "null", "false"):
        return default
    return int(val)


KV_GROUP_SIZE: int | None = 32
KV_BITS: int | None = _int_or_none("EXO_KV_BITS", None)
MAX_TOKENS: int = 32168
MAX_KV_SIZE: int | None = _int_or_none("EXO_MAX_KV_SIZE", 3200)
KEEP_KV_SIZE: int | None = _int_or_none("EXO_KEEP_KV_SIZE", 1600)
QUANTIZE_MODEL_MODE: str | None = "affine"
CACHE_GROUP_SIZE: int = 64

DEFAULT_TOP_LOGPROBS: int = 5

PREFILL_STEP_SIZE: int = int(os.environ.get("EXO_PREFILL_STEP_SIZE", "4096"))

# Maximum prefill chunk size for heartbeat liveness.  Even when
# EXO_PREFILL_STEP_SIZE is set very large (e.g. 524288 for pipeline
# parallel throughput), generate_step must break the prefill into
# chunks no larger than this so that prompt_progress_callback fires
# between chunks and the runner heartbeat stays alive.
MAX_PREFILL_CHUNK: int = int(os.environ.get("EXO_MAX_PREFILL_CHUNK", "16384"))

# TODO: We should really make this opt-in, but Kimi requires trust_remote_code=True
TRUST_REMOTE_CODE: bool = True
