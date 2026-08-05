"""Where the application puts files — the only place that decides it.

The default transcript folder used to be written down twice: in `Settings` and
in the window state. Such pairs drift apart silently — the edit lands in one copy
while the other keeps returning the old path, and files end up elsewhere.

Two conventions live here. Unix splits an application's belongings across XDG
directories in the home folder; Windows keeps them together under
`%LOCALAPPDATA%\\<app>` and expects the separation to be made by subfolders. The
difference is confined to this module: everything else asks for `cache_dir` and
gets whatever the system considers right.
"""

from __future__ import annotations

import os
import sys
from pathlib import Path

APP = "transcript"

# Where Explorer keeps the already-expanded paths of the user's own folders.
# Microsoft calls the key a compatibility leftover and points at the shell API
# instead, but the key is still maintained, still follows a folder moved by
# OneDrive, and costs six lines against thirty of COM.
_SHELL_FOLDERS = r"Software\Microsoft\Windows\CurrentVersion\Explorer\Shell Folders"


def default_output() -> Path:
    """The folder finished transcripts are collected in.

    Documents, because both systems keep it one click away: that is where a
    person goes looking for a finished file without memorising a path.
    """
    return _documents() / "Транскрипты"


def data_dir() -> Path:
    """Recording history: it outlives a cache purge, so it lives apart from it."""
    return _app_dir(".local/share", "data")


def config_dir() -> Path:
    """Choices made with the mouse: the output folder, the colour theme."""
    return _app_dir(".config", "config")


def cache_dir() -> Path:
    """Intermediate artifacts: losing them is fine, recomputing costs minutes."""
    return _app_dir(".cache", "cache")


def _app_dir(xdg: str, windows: str) -> Path:
    """One of our folders, named the way the system in use names such things."""
    if sys.platform == "win32":
        return _local_app_data() / APP / windows
    return Path.home() / xdg / APP


def _local_app_data() -> Path:
    """The per-machine half of the profile: it is not copied between machines.

    Which is what we want for all three folders. The roaming half would drag the
    cache along behind a person moving to another computer.
    """
    return Path(os.environ.get("LOCALAPPDATA") or Path.home() / "AppData" / "Local")


def _documents() -> Path:
    """The folder the system itself considers Documents.

    On Windows that is not always `~/Documents`: OneDrive relocates it, and a
    transcript written to the old path would land somewhere the person no longer
    looks — while the folder shown in Explorer stays empty.
    """
    if sys.platform != "win32":
        return Path.home() / "Documents"

    import winreg

    try:
        with winreg.OpenKey(winreg.HKEY_CURRENT_USER, _SHELL_FOLDERS) as key:
            personal, _ = winreg.QueryValueEx(key, "Personal")
        return Path(personal)
    except OSError:
        # The key is gone or unreadable. A wrong guess here is better than no
        # folder at all: the person can pick another one in the window.
        return Path.home() / "Documents"
