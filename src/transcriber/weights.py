"""Model weights: what is already on disk, how much it takes and how to fetch it.

The only place that knows how HuggingFace lays out its cache. The knowledge is
needed for exactly one purpose — to answer "is it downloaded" without going to
the network. With no files on disk the very first transcription silently leaves
for a few gigabytes, and a person spends minutes watching a motionless window
with nothing to say what is happening.

Downloading lives here too, because fetching weights is a step of its own with a
duration of its own, and it should be visible rather than hidden inside the first
run.
"""

from __future__ import annotations

import os
import subprocess
import sys
import time
from collections.abc import Callable
from dataclasses import dataclass
from pathlib import Path

from . import paths
from .config import Settings
from .report import Report, Stopped
from .subproc import interpreter, quiet_flags

# Different builds name their weight files differently: mlx-lm keeps
# `*.safetensors` shards, mlx-whisper a single `weights.npz`, CTranslate2 a
# `model.bin`. The question has to be about the weights specifically — config and
# tokenizer arrive first, so an interrupted download leaves a folder that looks
# like a finished model.
#
# `.bin` was missing while only mlx was asked about, and its absence was silent
# in the worst way: the 2.9 GB of `faster-whisper-large-v3` sat on the disk while
# every check called the model absent.
WEIGHT_SUFFIXES = (".safetensors", ".npz", ".bin")

BytesProgress = Callable[[int, int], None]
"""Bytes already on disk and bytes expected in total (0 — size unknown).

Named apart from `Report.at` on purpose: that one is a fraction, this one is two
byte counts, and the window shows them differently — "13 MB of 2.9 GB" cannot be
recovered from a percentage. They used to share the name `Progress` in different
modules, which promised one thing and delivered another.
"""


@dataclass(frozen=True)
class ModelWeights:
    """One model's weights, described the way they are spoken of to the user."""

    repo: str
    role: str
    """`asr` or `llm` — what this model does in the pipeline."""

    title: str
    """The short registry name: it is what a person picks the model by."""

    ready: bool
    size: int
    """Bytes on disk. For an unfinished model, the size of what arrived."""


def hub() -> Path:
    """The root of the HuggingFace cache: ours where we own one, theirs otherwise.

    Ours is not asked of the library on purpose. `HF_HUB_CACHE` is a module
    constant frozen the moment anything first imports `huggingface_hub`, so the
    answer would be whatever the environment happened to hold by then. Get that
    order wrong once and downloads land in one folder while the check for "is it
    already downloaded" reads another — with nothing said by either.
    """
    owned = paths.models_dir()
    if owned is not None:
        return owned / "huggingface" / "hub"
    # No folder of our own, so nothing of ours has touched the library's answer
    # and it is the honest one — a developer who moved their cache meant it.
    try:
        from huggingface_hub.constants import HF_HUB_CACHE

        return Path(HF_HUB_CACHE)
    except ImportError:
        return _previous_hub()


def _previous_hub() -> Path:
    """Where weights downloaded before this version are still lying.

    Worked out by HuggingFace's own rule rather than asked of the library. By the
    time anything can ask, our folder is already in the environment and the
    library answers with it — so the move would look for the old weights in the
    new place, find nothing there, and quietly download twenty-two gigabytes that
    are on the disk already. That is precisely what happened when it did ask.
    """
    cache = os.environ.get("XDG_CACHE_HOME")
    return (Path(cache) if cache else Path.home() / ".cache") / "huggingface" / "hub"


def cache_dir(repo: str) -> Path:
    """The model folder — the name the library itself gives it."""
    return hub() / _folder(repo)


def _folder(repo: str) -> str:
    """The folder name a repository gets in the cache. The library's rule, not ours."""
    return "models--" + repo.replace("/", "--")


def adopt(previous: Path, current: Path) -> bool:
    """Moves a folder of weights under the application, when that costs nothing.

    A rename, and deliberately nothing else. On one volume it is instant whatever
    the folder weighs; across two the system would fall back to copying, and
    twenty-two gigabytes copied before the window opens is a launch that looks
    hung with nothing to explain it. A model left behind is merely downloaded
    again, through the panel that already shows progress — and the old folder
    stays where it is, to be moved by hand by anyone who would rather not wait.
    """
    if current.exists() or not previous.is_dir():
        return False
    current.parent.mkdir(parents=True, exist_ok=True)
    try:
        previous.rename(current)
    except OSError:
        # Another volume, or someone still holding the folder open. Neither is
        # worth a failed launch: the weights are exactly where they were.
        return False
    return True


def adopt_model(repo: str) -> bool:
    """Takes a model downloaded by an earlier version into the folder we own.

    Only this model's folder, never the cache around it: that cache belongs to
    the whole machine, and every other tool on it expects to find its own models
    where it left them.
    """
    if paths.models_dir() is None:
        return False
    return adopt(_previous_hub() / _folder(repo), cache_dir(repo))


def is_ready(repo: str) -> bool:
    """Whether the weights themselves are on disk, not just the files around them."""
    # A missing folder passes through `glob` quietly — no separate check needed.
    files = (cache_dir(repo) / "snapshots").glob("*/*")
    return any(path.suffix in WEIGHT_SUFFIXES for path in files)


def local_size(repo: str) -> int:
    """How much of the model is already on disk.

    Links are skipped: in the HuggingFace cache the snapshot files are symlinks
    into `blobs`, and counting both would double the size. Where the system has
    no symlinks — Windows outside developer mode — the library moves the blob
    into the snapshot instead of linking to it, so nothing is counted twice
    there either.

    Unfinished pieces must not be summed. Every interrupted attempt leaves its
    own `*.incomplete`, and the total quickly outgrows the model itself — 4.5 GB
    was seen where the weights are 2.9. Only the largest is counted: that one is
    the current attempt, the rest are abandoned litter.
    """
    done = partial = 0
    for root, _, files in os.walk(cache_dir(repo)):
        for name in files:
            path = Path(root) / name
            try:
                if path.is_symlink():
                    continue
                size = path.stat().st_size
            except OSError:
                # The file vanished between the walk and the measurement — a
                # download is running, no harm done.
                continue
            if name.endswith(".incomplete"):
                partial = max(partial, size)
            else:
                done += size
    return done + partial


def remote_size(repo: str, *, attempts: int = 3) -> int:
    """What the model weighs on HuggingFace. Zero means "could not ask".

    We ask several times: the request leaves at the same moment as the download
    starts, and a single miss is expensive — the bar stays without a scale to the
    very end, though the network was fine a second later.

    Without a size the download still runs: progress then counts gigabytes
    instead of percent.
    """
    from huggingface_hub import HfApi

    for attempt in range(attempts):
        try:
            info = HfApi().model_info(repo, files_metadata=True)
        except Exception:
            # The size is a helper, not a reason to cancel a download.
            time.sleep(attempt)
            continue
        return sum(file.size or 0 for file in (info.siblings or []))
    return 0


def required(settings: Settings) -> list[ModelWeights]:
    """The weights needed under the current settings.

    Recognition is always here, whichever backend does it. It used to be listed
    only for mlx — CTranslate2 was said to keep its models "under names of its
    own", and the names are indeed its own, but the files land in the same
    HuggingFace cache and are ours to fetch just the same. The cost of that
    sentence was the whole panel outside Apple Silicon: a person on Windows was
    told nothing, and their first recognition went off to fetch three gigabytes
    with a progress bar that belonged to the recognition.

    Ollama is the real case of weights kept elsewhere: they live inside the
    daemon, and nothing here can see or fetch them.

    Speaker labelling is here whatever the settings say about it, because the
    window has a tick-box for it beside every recording: weights nobody asked for
    at startup are wanted the moment it is ticked, and finding that out then means
    finding it out during the work.
    """
    items = [_describe(asr_repo(settings), "asr", settings.asr_model)]
    if settings.llm_backend == "mlx":
        items.append(_describe(settings.llm_repo, "llm", settings.llm_model))
    items.append(_speakers(settings))
    return items


SHERPA = "sherpa-onnx"
"""What the pair of speaker models is called where a repository name is expected.

Not a repository and never was: they are two archives on GitHub releases. The name
is what the panel keys a row by and what comes back when its button is pressed, so
it has to be something — and something that cannot collide with a real one.
"""


def _speakers(settings: Settings) -> ModelWeights:
    """The weights that labelling speakers needs, under the chosen backend.

    Two backends, two different worlds. pyannote is a repository like any other and
    the machinery here fits it — except that it is gated, so the button below works
    only for someone who has taken a token and accepted the terms. The offer is
    made all the same: a person who chose pyannote has done both, and one who has
    not is on sherpa, where nothing is asked of anybody.
    """
    from .diarize import sherpa_backend

    if settings.diarization_backend == "pyannote":
        name = settings.diarization_model.rsplit("/", maxsplit=1)[-1]
        return _describe(settings.diarization_model, "speakers", name)
    return ModelWeights(
        repo=SHERPA,
        role="speakers",
        title="sherpa-onnx",
        size=sherpa_backend.local_size(),
        ready=sherpa_backend.ready(),
    )


def asr_repo(settings: Settings) -> str:
    """The repository recognition will fetch from, which the backend decides.

    Asked of the backend rather than read off the settings, for the reason the log
    line and the recorded model name are: a short name expands differently per
    backend, and off Apple Silicon the settings still hold the mlx repository
    while the work goes to a CTranslate2 one.
    """
    if settings.asr_backend == "mlx":
        return settings.asr_repo

    from .asr.faster_backend import hub_repo, normalize_model_name

    return hub_repo(normalize_model_name(settings.asr_repo))


def missing(settings: Settings) -> list[ModelWeights]:
    return [item for item in required(settings) if not item.ready]


def _describe(repo: str, role: str, title: str) -> ModelWeights:
    ready = is_ready(repo)
    return ModelWeights(
        repo=repo,
        role=role,
        title=title,
        # The size of an unfinished model is what already arrived: it shows the
        # download was interrupted rather than never started.
        size=local_size(repo),
        ready=ready,
    )


def sweep(repo: str) -> int:
    """Clears abandoned pieces of earlier attempts. Returns the bytes freed.

    An interrupted download leaves its `*.incomplete` behind and the next attempt
    starts a new one — the pieces pile up until the folder outgrows the model. The
    freshest piece is left alone: it either belongs to a running download or will
    serve one, should the library manage to finish what it started.
    """
    blobs = cache_dir(repo) / "blobs"
    pieces = sorted(blobs.glob("*.incomplete"), key=lambda path: path.stat().st_mtime)

    freed = 0
    # The newest is the current attempt; the rest are of no use to anyone.
    for piece in pieces[:-1]:
        try:
            freed += piece.stat().st_size
            piece.unlink()
        except OSError:
            # It could not be removed — this is litter, not work: carry on.
            continue
    return freed


# The HuggingFace downloader cannot be interrupted from the inside — it never asks
# whether anyone is still waiting. So it lives as a separate process: one of those
# stops at once.
#
# Progress is lost when it stops. The library itself can resume — it sends a
# `Range` from the size of the file it began — but with Xet on (the `hf_xet`
# package, installed by default) every attempt starts a temporary file of its own.
# An interrupted download was seen with four pieces, together larger than the
# model. So stopping means "start again", not "continue from the break".
_FETCH = "import sys; from huggingface_hub import snapshot_download; snapshot_download(sys.argv[1])"


def _spawn(repo: str) -> subprocess.Popen[str]:
    env = os.environ | {
        # A short read timeout: a hung connection should break quickly enough for
        # the watchdog below to notice and restart it.
        "HF_HUB_DOWNLOAD_TIMEOUT": "20",
        # Paths are inherited explicitly: even if the interpreter turns out to be
        # a stranger's, it will still find our packages.
        "PYTHONPATH": os.pathsep.join(p for p in sys.path if p),
        # Xet is the new HuggingFace transport, and on this machine it turned out
        # worse than plain HTTP on every count: 0.2 MB in fifteen seconds against
        # 13 MB/s, and the file grows in its own chunk cache rather than in the
        # model folder — progress is invisible and every attempt starts over. The
        # ordinary path writes to `blob.incomplete` and resumes by `Range`.
        "HF_HUB_DISABLE_XET": "1",
    }
    # The downloader does not import our package, so where we keep weights has to
    # travel as a variable — `hub()` in that process would answer for HuggingFace
    # instead of for us, and the model would arrive somewhere nobody looks.
    if paths.models_dir() is not None:
        env["HF_HUB_CACHE"] = str(hub())
    return subprocess.Popen(
        [interpreter(), "-c", _FETCH, repo],
        stdout=subprocess.DEVNULL,
        stderr=subprocess.PIPE,
        text=True,
        env=env,
        **quiet_flags(),
    )


# When a download counts as hung. A connection to HuggingFace can freeze without
# breaking: the process is alive, the file is open, no data comes. Seen live —
# first not a byte of growth, then a speed of single kilobytes.
#
# The threshold is in bytes rather than "strictly zero": a barely crawling
# connection is indistinguishable from a dead one by its consequences. A megabyte
# in three quarters of a minute is 23 KB/s; at that speed a three-gigabyte model
# would take a day and a half.
STALL_SECONDS = 45.0
STALL_BYTES = 1 << 20
STALL_RESTARTS = 5


def download(
    repo: str,
    on_bytes: BytesProgress | None = None,
    *,
    report: Report | None = None,
    period: float = 0.7,
) -> Path:
    """Downloads the weights, reporting how many bytes are on disk.

    Progress is measured by the size of the folder rather than by the innards of
    the downloader: that way it does not depend on how many files the model
    arrives in or in what order, and it shows exactly what takes up space.

    A hung connection is restarted. Waiting for it is pointless: the process is
    alive, the file is open, and no bytes come — from the outside that is
    indistinguishable from a slow network, and a person stares at a frozen
    counter not knowing the download is already dead. A restart costs the file
    begun, so the stall threshold is generous.
    """
    total = remote_size(repo) if on_bytes else 0
    told = report or Report()
    sweep(repo)

    for restart in range(STALL_RESTARTS + 1):
        process = _spawn(repo)
        if _pump(repo, process, total, on_bytes, told, period):
            break

        _stop(process)
        if restart == STALL_RESTARTS:
            raise RuntimeError(f"the download of {repo} is not moving: the network sends nothing")
        told.say(f"the download stalled, starting over ({restart + 1} of {STALL_RESTARTS})")

    if process.returncode != 0:
        reason = (process.stderr.read() if process.stderr else "").strip().splitlines()
        raise RuntimeError("\n".join([f"could not download {repo}", *reason[-3:]]))

    if on_bytes is not None:
        final = local_size(repo)
        on_bytes(final, total or final)
    return cache_dir(repo)


def _pump(
    repo: str,
    process: subprocess.Popen[str],
    total: int,
    on_bytes: BytesProgress | None,
    report: Report,
    period: float,
) -> bool:
    """Watches a running download. `True` — it finished on its own, `False` — hung."""
    seen, moved_at = local_size(repo), time.monotonic()

    while process.poll() is None:
        if report.stopping:
            _stop(process)
            raise Stopped(repo)

        done = local_size(repo)
        if done - seen >= STALL_BYTES:
            seen, moved_at = done, time.monotonic()
        elif time.monotonic() - moved_at > STALL_SECONDS:
            return False

        if on_bytes is not None:
            on_bytes(done, total)
        time.sleep(period)
    return True


def _stop(process: subprocess.Popen[str]) -> None:
    process.terminate()
    try:
        process.wait(timeout=30)
    except subprocess.TimeoutExpired:
        process.kill()
