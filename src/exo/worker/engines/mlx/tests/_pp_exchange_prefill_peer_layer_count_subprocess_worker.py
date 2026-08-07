#!/usr/bin/env python3
# pyright: reportPrivateUsage=false, reportUnknownVariableType=false
# pyright: reportUnknownMemberType=false, reportUnknownArgumentType=false
# pyright: reportUnknownParameterType=false, reportMissingParameterType=false
# pyright: reportArgumentType=false, reportUnknownLambdaType=false
# pyright: reportCallIssue=false, reportAny=false
"""Worker script for
``test_pp_exchange_prefill_peer_layer_count_subprocess.py``: one real
OS process, real MLX ring transport, calling the ACTUAL production
``exchange_prefill_peer_layer_count()`` function with its own local
layer count, writing the result (or any error) to a JSON file for the
parent to assert on.
"""

from __future__ import annotations

import json
import sys
import traceback

import mlx.core as mx


def main() -> None:
    rank = int(sys.argv[1])
    out_path = sys.argv[2]
    local_layer_count = int(sys.argv[3])

    result: dict[str, object] = {"rank": rank}

    try:
        group = mx.distributed.init(backend="ring")
        if group.rank() != rank:
            raise RuntimeError(
                f"MLX ring group.rank()={group.rank()} does not match "
                f"expected rank={rank} from argv"
            )
        if group.size() != 2:
            raise RuntimeError(f"expected group.size()==2, got {group.size()}")

        sys.path.insert(0, "src")
        from exo.worker.engines.mlx.pp_batched_decode_glue import (
            exchange_prefill_peer_layer_count,
        )

        dst_rank = 1 if rank == 0 else 0
        peer_layer_count = exchange_prefill_peer_layer_count(
            local_layer_count=local_layer_count,
            dst_rank=dst_rank,
            group=group,
        )
        result["ok"] = True
        result["peer_layer_count"] = peer_layer_count
    except Exception as e:  # noqa: BLE001 -- deliberately broad, this is a
        # subprocess worker whose ONLY job is to report failure to the
        # parent, not to handle it
        result["ok"] = False
        result["error"] = str(e)
        result["traceback"] = traceback.format_exc()

    with open(out_path, "w") as f:
        json.dump(result, f)


if __name__ == "__main__":
    main()
