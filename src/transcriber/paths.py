"""Where the application puts files — the only place that decides it.

The default transcript folder used to be written down twice: in `Settings` and
in the window state. Such pairs drift apart silently — the edit lands in one copy
while the other keeps returning the old path, and files end up elsewhere.

The paths are macOS ones: XDG directories inside the home folder. Windows has
different conventions (`%LOCALAPPDATA%`, a Documents folder OneDrive may relocate)
— when its turn comes, only this module will need changing.
"""

from __future__ import annotations

from pathlib import Path

APP = "transcript"


def default_output() -> Path:
    """The folder finished transcripts are collected in.

    Documents, because Finder keeps it in the sidebar: that is where a person
    goes looking for a finished file without memorising a path.
    """
    return Path.home() / "Documents" / "Транскрипты"


def data_dir() -> Path:
    """Recording history: it outlives a cache purge, so it lives apart from it."""
    return Path.home() / ".local" / "share" / APP


def config_dir() -> Path:
    """Choices made with the mouse: the output folder, the colour theme."""
    return Path.home() / ".config" / APP


def cache_dir() -> Path:
    """Intermediate artifacts: losing them is fine, recomputing costs minutes."""
    return Path.home() / ".cache" / APP
