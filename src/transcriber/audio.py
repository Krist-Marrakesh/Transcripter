"""Low-level audio handling.

Inside the pipeline there is exactly one format: float32, mono, 16 kHz. ffmpeg
brings everything to it during ingest, so nothing downstream has to think
about containers, sample rates or channel counts.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import soundfile as sf

SAMPLE_RATE = 16_000


def load(path: Path) -> np.ndarray:
    """Reads the pipeline WAV into a mono float32 array.

    Raises:
        ValueError: the file is not in pipeline format, meaning ingest was bypassed.
    """
    samples, rate = sf.read(path, dtype="float32", always_2d=False)
    if rate != SAMPLE_RATE:
        raise ValueError(f"{path}: expected {SAMPLE_RATE} Hz, got {rate}")
    if samples.ndim != 1:
        raise ValueError(f"{path}: expected mono, got {samples.shape[1]} channels")
    return samples


def save(path: Path, samples: np.ndarray) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    sf.write(path, samples, SAMPLE_RATE, subtype="PCM_16")
    return path


def duration(samples: np.ndarray) -> float:
    return len(samples) / SAMPLE_RATE


def to_samples(seconds: float) -> int:
    return int(round(seconds * SAMPLE_RATE))


def to_seconds(sample_index: int) -> float:
    return sample_index / SAMPLE_RATE
