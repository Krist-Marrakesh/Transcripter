"""Локальный транскрибатор аудио, видео и ссылок с YouTube.

Пайплайн: ingest (ffmpeg / yt-dlp) → VAD → ASR (Whisper) → диаризация →
перевод и саммари локальной LLM → экспорт.

Всё считается на своём железе; сеть нужна один раз, чтобы скачать веса моделей.
"""

from __future__ import annotations

import os
from importlib.metadata import PackageNotFoundError, version

from . import paths
from .config import Settings, load_settings
from .models import Diarization, Segment, SpeakerTurn, Summary, Transcript, Translation, Word
from .pipeline import Pipeline
from .weights import hub

# Asked of the installed distribution rather than written down here. The updater
# compares what is running against what is published, and a literal in the source
# answers a different question than the one being asked: after an update installs
# a new wheel into the environment, the literal would still name the old version
# and the same update would be offered forever.
try:
    __version__ = version("transcript")
except PackageNotFoundError:
    # Running straight from a source tree that was never installed. Nothing to
    # compare against, and saying so beats inventing a number.
    __version__ = "0"


def _point_huggingface_at_our_folder() -> None:
    """Tells the download libraries where this copy keeps its weights.

    Said here, at the import of our own package, because `huggingface_hub` reads
    the variable once and freezes it into a module constant — every setting after
    the first import of the library is ignored in silence. mlx, pyannote and
    faster-whisper all pull it in, and each of them is imported lazily inside a
    function, so this comes first by a wide margin.

    `HF_HUB_CACHE` rather than `HF_HOME`: the narrow one moves the models and
    leaves the rest — tokens above all — where the person keeps them. It is also
    the one that wins between the two, so setting the wide one would have left the
    library obeying an `HF_HUB_CACHE` we never looked at.

    Set outright rather than by default. A bundle that keeps its weights inside
    itself cannot honour a folder somewhere else: the library would download to
    one place while everything here counted files in another.
    """
    if paths.models_dir() is not None:
        os.environ["HF_HUB_CACHE"] = str(hub())


def _keep_downloads_on_plain_http() -> None:
    """Turns Xet off, for the downloads that happen inside this process.

    Said here for the reason the folder above is said here: the library freezes
    the variable into a module constant at its own import, and every setting after
    that is ignored in silence.

    Said at all because `weights._spawn` — which has carried this since the
    transport threw away 952 MB at the thirteenth minute, and where the numbers
    behind it are written — is not on every path. `weights.required()` knows only
    the mlx backends, so wherever recognition runs on faster-whisper the model is
    fetched in-process by the library itself, and nothing was setting it there.

    Left alone if someone has set it already: the measurement is one machine's,
    and the variable is how a person with a better connection says so.
    """
    os.environ.setdefault("HF_HUB_DISABLE_XET", "1")


_point_huggingface_at_our_folder()
_keep_downloads_on_plain_http()

__all__ = [
    "Diarization",
    "Pipeline",
    "Segment",
    "Settings",
    "SpeakerTurn",
    "Summary",
    "Transcript",
    "Translation",
    "Word",
    "load_settings",
]
