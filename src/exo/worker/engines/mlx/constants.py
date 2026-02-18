# TODO: Do we want so many constants?
#  I think we want a lot of these as parameters?

import os


def _get_kv_bits() -> int | None:
    bits = os.environ.get("EXO_KV_BITS", "4")
    if bits.lower() in ("none", "null", "false"):
        return None
    return int(bits)

KV_GROUP_SIZE: int | None = 32
KV_BITS: int | None = _get_kv_bits()
ATTENTION_KV_BITS: int | None = _get_kv_bits()
MAX_TOKENS: int = 32168
MAX_KV_SIZE: int | None = 3200
KEEP_KV_SIZE: int | None = 1600
QUANTIZE_MODEL_MODE: str | None = "affine"
CACHE_GROUP_SIZE: int = 64
KV_CACHE_BITS: int | None = 4

DEFAULT_TOP_LOGPROBS: int = 5

# TODO: We should really make this opt-in, but Kimi requires trust_remote_code=True
TRUST_REMOTE_CODE: bool = True
