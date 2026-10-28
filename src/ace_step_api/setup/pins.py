"""Everything ``setup`` installs, pinned. Change a pin only after measuring the new version on
the target Mac (README "Measurements")."""

from __future__ import annotations

# -- ACE-Step 1.5 --------------------------------------------------------------------------------
# The Apple Silicon port: the checkout the server was measured with (PyTorch on MPS, 19 GB peak,
# 2x the official checkout's speed; the official one peaked at 37 GB with MLX).
ACE_STEP_REPO = "https://github.com/clockworksquirrel/ace-step-apple-silicon"
ACE_STEP_COMMIT = "a97f0519ac1a1735caa7bee3fd15a9ca70822901"
ACE_STEP_DIR = "ace-step-apple-silicon"
# the port has no uv.lock: resolve as of the day the measured environment existed
ACE_STEP_EXCLUDE_NEWER = "2026-10-17T00:00:00Z"
# the port's MLX language model fails to load its checkpoint and falls back to PyTorch
ACE_STEP_LM_BACKEND = "pt"
ACE_STEP_HF_HOME = "~/.local/share/ace-step-api/hf"
# the checkout's download_main_model() would fetch main: setup fetches this commit instead
ACE_STEP_MODEL_REPO = "ACE-Step/Ace-Step1.5"
ACE_STEP_MODEL_REVISION = "19671f406d603126926c1b7e2adc169acbcade22"
ACE_STEP_MODEL_BYTES = 10_092_102_593  # the repository's files at that revision
# estimates: the port's environment (PyTorch, transformers, gradio, ...) and its checkout
ACE_STEP_ENV_BYTES = 2_500_000_000
ACE_STEP_CLONE_BYTES = 150_000_000
ACE_STEP_LICENCE = "ACE-Step 1.5 code and weights: MIT"

# -- MiniMax Music 3 -----------------------------------------------------------------------------
MINIMAX_PYTHON_VERSION = "3.12"
MINIMAX_MLX_AUDIO = "mlx-audio==0.5.9"
MINIMAX_EXCLUDE_NEWER = "2026-10-19T00:00:00Z"  # the day mlx-audio 0.5.9 was measured
MINIMAX_DIR = "minimax-music3"
MINIMAX_MODEL_REPO = "mlx-community/MiniMax-Music3-mxfp8"
MINIMAX_MODEL_REVISION = "d00a12c3c7f80eb66379dd02dd0f30ed0ce2d96e"
MINIMAX_MODEL_BYTES = 13_868_676_861  # the repository's files at that revision
MINIMAX_ENV_BYTES = 1_000_000_000  # estimate: mlx, mlx-audio and their dependencies
MINIMAX_LICENCE = (
    "MiniMax Music 3: commercial use allowed if the product shows 'MiniMax-Music3' prominently; "
    "over USD 20 million annual revenue needs MiniMax's permission"
)

# -- disk ----------------------------------------------------------------------------------------
# uv keeps a cache next to the environment and Hugging Face writes .incomplete files first
DISK_MARGIN = 1.2
DISK_RESERVE_BYTES = 5_000_000_000
