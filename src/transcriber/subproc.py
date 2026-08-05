"""Running outside programs without the interface flinching.

One rule lives here, and it exists because of one system. A shortcut on Windows
points at `pythonw.exe`, a program without a console; when such a process starts
a console one — ffmpeg, ffprobe, the weight downloader — Windows gives the child
a console of its own and shows it. Redirected streams do not help: the window
appears because of the subsystem the child was built for, not because of what it
writes. So a person would see a black rectangle blink over the interface for
every file probed and every download restarted.
"""

from __future__ import annotations

import sys

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
