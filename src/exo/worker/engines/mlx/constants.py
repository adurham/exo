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
ATTENTION_KV_BITS: int | None = _int_or_none("EXO_KV_BITS", 4)
MAX_TOKENS: int = 32168
MAX_KV_SIZE: int | None = _int_or_none("EXO_MAX_KV_SIZE", 3200)
KEEP_KV_SIZE: int | None = _int_or_none("EXO_KEEP_KV_SIZE", 1600)
QUANTIZE_MODEL_MODE: str | None = "affine"
CACHE_GROUP_SIZE: int = 64
KV_CACHE_BITS: int | None = _int_or_none("EXO_KV_BITS", None)

DEFAULT_TOP_LOGPROBS: int = 5

# TODO: We should really make this opt-in, but Kimi requires trust_remote_code=True
TRUST_REMOTE_CODE: bool = True
