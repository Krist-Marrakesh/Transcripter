"""Making the CUDA libraries findable, which only Windows needs.

Since Python 3.8 an extension module is not looked for next to whatever happens to
be on `PATH`: only directories the process has named itself are searched. So
CTranslate2 links against cuBLAS and cuDNN and finds neither, though pip has just
put both into `site-packages` — and the import fails with "DLL load failed while
importing", naming nothing at all. From there the application falls back to the
processor and a lecture takes forty minutes instead of four.

Nowhere else does the question arise. Linux wheels carry the folder inside them as
an RPATH, and on macOS there is no CUDA to find.
"""

from __future__ import annotations

import os
import sys
import sysconfig
from pathlib import Path

_named: list[object] = []
"""Handles of the directories already named, kept alive on purpose.

`os.add_dll_directory` hands back an object that *removes* the directory when it
is closed. Dropping it on the floor would mean the search path lasted until the
next collection — which is a defect that shows up as a random import failure.
"""


def library_paths() -> tuple[Path, ...]:
    """Folders in this environment that hold CUDA runtime libraries.

    Two shapes, because the packages disagree about where such things go. torch's
    Windows build carries the CUDA runtime beside its own DLLs in `torch/lib`;
    NVIDIA's packages give each library a folder of its own in `nvidia/*/bin`.
    Whichever is installed here is what is returned — on a machine without a card
    that is nothing, and nothing is the right answer.
    """
    root = Path(sysconfig.get_paths()["purelib"])
    candidates = [root / "torch" / "lib", *sorted((root / "nvidia").glob("*/bin"))]
    return tuple(path for path in candidates if path.is_dir())


def make_findable() -> tuple[Path, ...]:
    """Tells the loader where those libraries are, and says which it named.

    Safe to call again: the directories already named are remembered, and naming
    one twice would only add a handle nobody holds. Called immediately before the
    imports that need it rather than once at startup, because at startup we do not
    yet know whether anything will.
    """
    if sys.platform != "win32":
        return ()
    found = library_paths()
    if not _named:
        _named.extend(os.add_dll_directory(str(path)) for path in found)
    return found
