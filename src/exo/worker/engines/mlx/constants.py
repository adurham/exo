# TODO: Do we want so many constants?
#  I think we want a lot of these as parameters?

import os


def _get_kv_bits() -> int | None:
    bits = os.environ.get("EXO_KV_BITS", "none")
    if bits.lower() in ("none", "null", "false"):
        return None
    return int(bits)

KV_GROUP_SIZE: int | None = 32
KV_BITS: int | None = _get_kv_bits()
ATTENTION_KV_BITS: int | None = _get_kv_bits()
def _get_int_or_none(env_var: str, default: str) -> int | None:
    val = os.environ.get(env_var, default)
    if val.lower() in ("none", "null", "false"):
        return None
    return int(val)

MAX_TOKENS: int = 32168
MAX_KV_SIZE: int | None = _get_int_or_none("EXO_MAX_KV_SIZE", "none")
KEEP_KV_SIZE: int | None = _get_int_or_none("EXO_KEEP_KV_SIZE", "none")
QUANTIZE_MODEL_MODE: str | None = "affine"
CACHE_GROUP_SIZE: int = 64
KV_CACHE_BITS: int | None = _get_kv_bits()

DEFAULT_TOP_LOGPROBS: int = 5

# TODO: We should really make this opt-in, but Kimi requires trust_remote_code=True
TRUST_REMOTE_CODE: bool = True
