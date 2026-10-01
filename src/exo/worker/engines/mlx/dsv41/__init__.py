"""DeepSeek-V4.1-Flash (EXL3) engine package.

Modules, and who owns what:

* ``dsml``     -- the V4.1 DSML dialect (`<|DSML| calls>` / `<|DSML| invoke>`
                  / `<|DSML| parameter>`) expressed with exo's existing DSML
                  machinery. Real sentinel strings verified against the
                  checkpoint's ``tokenizer.json`` / ``chat_template.jinja``.
* ``thinking`` -- reasoning-marker resolution for this tokenizer
                  (`` thinking`` / ``</think>``, single added tokens).
* ``output``   -- the engine's output pipeline: thinking split -> DSML tool
                  calls -> chunks.
* ``rounds``   -- request gating, response construction and the decode round.
* ``load``     -- the EXL3 loader (TP geometry, wired limit, engram token map).
* ``builder``  -- the exo ``Builder`` for this engine.
* ``dispatch`` -- the dispatch decision (which card gets this engine).
* ``engine``   -- the batch-size-1 greedy/speculative engine.
* ``agreement``/``engram``/``errors`` -- rank agreement, engram token map,
                  error types.

The package has no import of ``mlx_lm.models.deepseek_v41`` at import time:
that module only exists in the DSv4.1 mlx-lm fork (the Macs), so every use of
it is a lazy import inside a function. That keeps ``exo`` importable (and the
control plane / gateway test runs honest) without the fork present.
"""

from __future__ import annotations

__all__ = [
    "agreement",
    "builder",
    "dispatch",
    "dsml",
    "engine",
    "engram",
    "errors",
    "load",
    "output",
    "rounds",
    "thinking",
]
