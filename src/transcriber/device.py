"""Choosing the compute device.

The only place where the project decides what to compute on. ASR, diarization and
the LLM ask here instead of working it out themselves — otherwise steps drift onto
different devices on the same machine and there is no telling which one is at
fault.

The split is deliberate: `is_apple_silicon` answers from the platform alone and
never touches torch, so the config can ask it at startup for free. `detect` knows
about the real hardware but pulls torch in, so it is called lazily and cached.
"""

from __future__ import annotations

import platform
import sys
from functools import cache
from typing import Literal

Device = Literal["cuda", "mps", "cpu"]


def is_apple_silicon() -> bool:
    """A Mac on Apple Silicon — the only platform where MLX is available."""
    return sys.platform == "darwin" and platform.machine() == "arm64"


@cache
def detect() -> Device:
    """The best device available: CUDA → Metal → CPU.

    The result is cached: hardware does not change while the process runs, and
    importing torch costs seconds.
    """
    try:
        import torch
    except ImportError:
        # torch is a diarization dependency. Without it ASR on mlx still runs on
        # Metal, and faster-whisper runs on the CPU.
        return "cpu"

    if torch.cuda.is_available():
        return "cuda"
    if torch.backends.mps.is_available():
        return "mps"
    return "cpu"


def describe() -> str:
    """A human-readable device name.

    It is shown to the user, so these are not backend identifiers (`mps`, `cpu`)
    but an answer to the question "why is this fast or slow".
    """
    device = detect()
    if device == "cuda":
        import torch

        return f"CUDA · {torch.cuda.get_device_name(0)}"
    return {"mps": "Metal · Apple GPU", "cpu": "CPU"}[device]
