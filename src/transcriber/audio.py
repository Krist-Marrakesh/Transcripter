"""Low-level audio handling.

Inside the pipeline there is exactly one format: float32, mono, 16 kHz. ffmpeg
brings everything to it during ingest, so nothing downstream has to think
about containers, sample rates or channel counts.
"""

from __future__ import annotations

from collections.abc import Iterator, Sequence
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


def length(path: Path) -> float:
    """How long the recording is, without reading it."""
    return sf.info(path).duration


def blocks(path: Path, seconds: float) -> Iterator[tuple[float, np.ndarray]]:
    """Walks the file in pieces, handing out the offset of each and its samples.

    So that the memory a step needs stops depending on how long the recording is.
    Eight hours at 16 kHz is 1.8 GB as float32 — enough to end a run on a laptop
    that has a model loaded as well, and to end it with an error about memory
    rather than about the recording.
    """
    with sf.SoundFile(path) as handle:
        size = to_samples(seconds)
        while True:
            offset = handle.tell()
            chunk = handle.read(size, dtype="float32", always_2d=False)
            if not len(chunk):
                return
            yield to_seconds(offset), chunk


def read_spans(path: Path, spans: Sequence[tuple[float, float]]) -> np.ndarray:
    """Reads only the named stretches and glues them into one array.

    The point is what is *not* read: the pauses between them never reach memory,
    and on a lecture that is three quarters of the file.
    """
    pieces: list[np.ndarray] = []
    with sf.SoundFile(path) as handle:
        for start, end in spans:
            handle.seek(to_samples(start))
            pieces.append(handle.read(to_samples(end - start), dtype="float32", always_2d=False))
    return np.concatenate(pieces) if pieces else np.zeros(0, dtype="float32")
