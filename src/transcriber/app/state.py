"""What the window remembers between launches.

Kept apart from `Settings`: those come from the environment and `.env`, while
this is what was chosen with the mouse. Mixing them would mean a button in the
interface silently rewriting the project's configuration.

The file sits beside the configs rather than in the cache: a cache is something
one wipes without thinking, and losing the chosen folder would be a shame.
"""

from __future__ import annotations

import json
from pathlib import Path

from .. import paths

PATH = paths.config_dir() / "app.json"


def load() -> dict[str, str]:
    """Reads the state. A missing or damaged file is simply an empty choice."""
    try:
        data = json.loads(PATH.read_text(encoding="utf-8"))
    except (OSError, ValueError):
        return {}
    return data if isinstance(data, dict) else {}


def save(**changes: str) -> None:
    """Adds values without wiping the ones already there."""
    data = load() | {key: str(value) for key, value in changes.items()}
    PATH.parent.mkdir(parents=True, exist_ok=True)
    tmp = PATH.with_suffix(".tmp")
    tmp.write_text(json.dumps(data, ensure_ascii=False, indent=2), encoding="utf-8")
    tmp.replace(PATH)


# The same path as in `Settings`: the default used to be written in two places
# and could drift apart at the first edit to either.
DEFAULT_OUTPUT = paths.default_output()


def output_dir(configured: Path) -> Path:
    """The output folder: chosen with the mouse, then configured, then default.

    A relative path from `.env` is ignored by the window. In a terminal "output"
    means "beside the current directory", which is reasonable there; but Finder
    starts an application from the root, where such a path either fails to be
    created or lands somewhere nobody expects.
    """
    chosen = load().get("output_dir")
    if chosen:
        return Path(chosen)
    return configured if configured.is_absolute() else DEFAULT_OUTPUT


def token() -> str | None:
    """The HuggingFace token entered in the window. `None` — none was.

    Here rather than in `.env`: that file belongs to the project, and a button in
    the interface has no business rewriting it silently, still less clobbering
    the keys around it. The price is honest — `app.json` is plain text, exactly
    as `.env` would have been.
    """
    return load().get("hf_token") or None


def declined_speakers() -> bool:
    """Whether pyannote was turned down. Reversible — a token overrides it."""
    return load().get("speakers_declined") == "yes"


def diarization() -> str | None:
    """The labelling backend chosen with the mouse. `None` — no choice made."""
    return load().get("diarization_backend") or None


def ensure(directory: Path) -> bool:
    """Creates the folder in advance and says whether it worked.

    In advance, because a folder already sitting there explains where to look
    before anything has been saved. Failure comes back as a value rather than an
    exception: for the window it is no reason not to open — macOS asks about
    access to Documents separately, and until it is answered writing there is
    forbidden — and what to do about it is the caller's decision.
    """
    try:
        directory.mkdir(parents=True, exist_ok=True)
        return True
    except OSError:
        return False
