"""Diarization through sherpa-onnx — the one that needs nothing arranged.

The models are published openly on GitHub releases, so this backend works on a
machine where nobody has accepted anything or obtained anything. That is the
whole reason it is the default: a person who just downloaded the application
gets speaker labels, rather than a message about somebody else's licence.

The segmentation stage is pyannote's own `segmentation-3.0`, exported to ONNX and
redistributed under its MIT licence — the gate on HuggingFace is a consent form,
not a restriction on sharing. What differs from the pyannote pipeline is the
voice embedding and the clustering, and that is where the two part ways.
"""

from __future__ import annotations

import tarfile
from collections.abc import Callable
from pathlib import Path

from .. import paths
from ..models import SpeakerTurn
from ..report import Report, Stopped
from . import DiarizationError

# Measured, not chosen. The library's own default of 0.5 turned one lecturer into
# sixty-four speakers; the count falls with the threshold rising — 0.5 → 64,
# 0.7 → 37, 0.9 → 26, 1.1 → 12, 1.3 → 2 — and 1.3 was then checked on five
# recordings, four and a half hours in all. Three of them are monologues, and on
# each it found exactly one speaker with no divergence at all; on the two with
# several, real disagreement with pyannote stayed between 1.8% and 2.8% of speech.
#
# Anyone tempted to restore the library default should repeat that measurement
# first. It is the single most load-bearing number in this file.
THRESHOLD = 1.3

RELEASES = "https://github.com/k2-fsa/sherpa-onnx/releases/download"
SEGMENTATION = (
    f"{RELEASES}/speaker-segmentation-models/sherpa-onnx-pyannote-segmentation-3-0.tar.bz2"
)
# The misspelling is theirs and is part of the address.
EMBEDDING = f"{RELEASES}/speaker-recongition-models/nemo_en_titanet_large.onnx"

TOTAL_BYTES = 104 * 1024 * 1024
"""Roughly what the pair weighs, for saying so before fetching it."""


def home() -> Path:
    """Where the models live. Not the HuggingFace cache — they are not from there.

    In the cache rather than beside the settings: losing them costs one download,
    and `weights.py` deliberately knows nothing about this pair.
    """
    return paths.cache_dir() / "diarization"


def ready() -> bool:
    return _segmentation().exists() and _embedding().exists()


def _segmentation() -> Path:
    return home() / "sherpa-onnx-pyannote-segmentation-3-0" / "model.onnx"


def _embedding() -> Path:
    return home() / "nemo_en_titanet_large.onnx"


def fetch(report: Report | None = None) -> None:
    """Downloads both models. Does nothing when they are already there."""
    import httpx

    told = report or Report()
    if ready():
        return

    home().mkdir(parents=True, exist_ok=True)
    told.say(f"downloading the speaker models, about {TOTAL_BYTES // (1024 * 1024)} MB")

    done = 0
    for url in (SEGMENTATION, EMBEDDING):
        target = home() / url.rsplit("/", 1)[-1]
        try:
            with httpx.stream("GET", url, follow_redirects=True, timeout=60.0) as answer:
                answer.raise_for_status()
                with target.open("wb") as file:
                    for chunk in answer.iter_bytes(1 << 16):
                        file.write(chunk)
                        done += len(chunk)
                        told.at(done / TOTAL_BYTES)
        except Exception as exc:
            # A half-arrived file would pass the existence check next time and
            # fail as a broken model instead of a missing one.
            target.unlink(missing_ok=True)
            raise DiarizationError(f"could not download the speaker models: {exc}") from exc

        if target.suffix == ".bz2":
            with tarfile.open(target) as archive:
                archive.extractall(home(), filter="data")
            target.unlink()

    if not ready():
        raise DiarizationError("the speaker models arrived, but not the files we expected")


def _watch(report: Report) -> Callable[[int, int], int]:
    """Turns the library's progress callback into the report the rest of us use.

    It is the only moment this backend asks anything during a run that lasts
    minutes, so it carries both the bar and the stop.

    Stopping by raising, not by the documented non-zero answer: measured on a
    three-minute recording, the answer changed nothing — labelling ran its full
    10.8 seconds and only then noticed. The exception travels out of the
    extension and ends the work where it stands, in 0.1 s.
    """

    def watch(done: int, total: int) -> int:
        report.stop_if_asked()
        report.at(done / total if total else 0.0)
        return 0

    return watch


def run(
    audio: Path, *, num_speakers: int | None = None, report: Report | None = None
) -> list[SpeakerTurn]:
    """Labels a recording by speaker."""
    import sherpa_onnx
    import soundfile

    fetch(report)

    config = sherpa_onnx.OfflineSpeakerDiarizationConfig(
        segmentation=sherpa_onnx.OfflineSpeakerSegmentationModelConfig(
            pyannote=sherpa_onnx.OfflineSpeakerSegmentationPyannoteModelConfig(
                model=str(_segmentation())
            ),
            num_threads=4,
        ),
        embedding=sherpa_onnx.SpeakerEmbeddingExtractorConfig(
            model=str(_embedding()), num_threads=4
        ),
        # A known count beats any threshold: guessing it is the harder half of the
        # job, and it is exactly the half this backend is weakest at.
        clustering=sherpa_onnx.FastClusteringConfig(
            num_clusters=num_speakers or -1, threshold=THRESHOLD
        ),
    )
    if not config.validate():
        raise DiarizationError("the speaker models did not load — try deleting the cache")

    # The invariant is the same as for pyannote: what arrives here is the full
    # recording, not the VAD-compressed array. The timings have to sit on the
    # original axis, because the ASR segments already do.
    samples, _ = soundfile.read(audio, dtype="float32", always_2d=False)
    told = report or Report()
    try:
        result = sherpa_onnx.OfflineSpeakerDiarization(config).process(
            samples, callback=_watch(told)
        )
    except Stopped:
        # A stop is a decision, not a failure. Wrapped as one it would be caught
        # by the pipeline as "speakers not labelled", and a run the person asked
        # to end would report itself finished.
        raise
    except Exception as exc:
        raise DiarizationError(f"could not label speakers: {exc}") from exc

    return [
        SpeakerTurn(start=segment.start, end=segment.end, speaker=f"SPEAKER_{segment.speaker:02d}")
        for segment in result.sort_by_start_time()
    ]
