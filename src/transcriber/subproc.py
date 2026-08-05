"""Starting outside programs: which interpreter, and without the window flinching.

Two questions that look unrelated until both are got wrong. Both are about a
child process, both have answers that differ per system, and both failed
silently rather than loudly when they were wrong.

The second rule exists because of one system. A shortcut on Windows
points at `pythonw.exe`, a program without a console; when such a process starts
a console one — ffmpeg, ffprobe, the weight downloader — Windows gives the child
a console of its own and shows it. Redirected streams do not help: the window
appears because of the subsystem the child was built for, not because of what it
writes. So a person would see a black rectangle blink over the interface for
every file probed and every download restarted.
"""

from __future__ import annotations

import sys
from pathlib import Path


def interpreter() -> str:
    """An interpreter that certainly has our packages.

    `sys.executable` will not do. pywebview relaunches the application on macOS
    through Python.app, which has neither the environment nor anything installed
    into it — the weight downloader died instantly this way while the window
    showed a speed of zero, and an update would install itself into a stranger.

    `sys.prefix` survives that relaunch and still points at our environment. The
    two systems lay one out differently, and the folder name is the only reliable
    sign of which one we are standing in.
    """
    folder, name = ("Scripts", "python.exe") if sys.platform == "win32" else ("bin", "python")
    inside = Path(sys.prefix) / folder / name
    return str(inside) if inside.exists() else sys.executable


# The same number `subprocess.CREATE_NO_WINDOW` holds — spelled out because that
# name exists only on Windows, and a function that cannot even be called
# elsewhere is a function no test on another system can check.
CREATE_NO_WINDOW = 0x08000000


def quiet_flags() -> dict[str, int]:
    """Arguments that stop a console child from opening a window of its own.

    Empty everywhere but Windows: there is nothing to hide on the other systems,
    and `subprocess` would not accept the argument there.

    The system is read at the moment of the call rather than at import. A
    constant frozen at import time would be a branch nothing but Windows itself
    could ever reach.
    """
    if sys.platform != "win32":
        return {}
    return {"creationflags": CREATE_NO_WINDOW}
