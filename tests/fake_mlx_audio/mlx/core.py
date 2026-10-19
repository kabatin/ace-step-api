"""A stand-in for ``mlx.core``: the buffer-cache release the MiniMax worker calls after a track.

``FAKE_MLX_CACHE_API`` picks where it lives: core (``mx.clear_cache``, MLX 0.32), metal
(``mx.metal.clear_cache``, older MLX) or none. Each call is recorded in ``FAKE_MINIMAX_RECORD``.
"""

import json
import os

_API = os.environ.get("FAKE_MLX_CACHE_API", "core")


def _record(where):
    path = os.environ.get("FAKE_MINIMAX_RECORD")
    if path:
        with open(path, "a", encoding="utf-8") as f:
            f.write(json.dumps({"clear_cache": where}) + "\n")


if _API == "core":

    def clear_cache():
        _record("core")


if _API == "metal":

    class metal:  # the module name in MLX
        @staticmethod
        def clear_cache():
            _record("metal")
