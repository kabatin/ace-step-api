"""A stand-in for the ``acestep`` package: the real worker script runs against it in tests.

``FAKE_ACESTEP_MODE`` picks the behaviour: ok, load_fail, gen_error, hang (generation never
ends), crash (the process dies mid-track), noisy (prints to stdout like the real engine).
"""

import os

MODE = os.environ.get("FAKE_ACESTEP_MODE", "ok")
