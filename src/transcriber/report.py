"""How a long step reports on itself and learns that it should stop.

Three concerns that always travel together: say something in words, say how much
is done, and answer whether anyone is still waiting. They were threaded through
every signature by hand — `prepare(target, cache, notify, advance, cancel)`,
`_download_audio(url, wav, declared, cache, key, say, advance, cancel)` — and
each new channel cost a parameter in half a dozen places.

They were also spelled differently in every module: `notify`/`say`, `advance`/
`on_progress`, and three separate exception classes meaning "the person pressed
stop". Worse, `Progress` meant a fraction in one module and a pair of byte counts
in another — the same name promising two different things.

One object, created once per run, passed as one argument. A fourth channel
becomes a field here rather than a parameter everywhere.
"""

from __future__ import annotations

import threading
from collections.abc import Callable
from dataclasses import dataclass, field

Notify = Callable[[str], None]
"""A line for a person to read. Appended to the log, never replaces anything."""

Advance = Callable[[float], None]
"""How much of the current step is done, from 0 to 1. Replaces itself."""


class Stopped(RuntimeError):
    """The work was stopped at a person's request.

    Not a failure, and callers have to tell the two apart: nothing is broken and
    there is nothing to report as an error. Whatever was finished stays in the
    cache, so starting again resumes rather than repeats.
    """


def _unsaid(_: str) -> None:
    """Nobody is listening. The default, so no step has to check for one."""


def _unwatched(_: float) -> None:
    """Nobody is watching the bar either."""


@dataclass(frozen=True)
class Report:
    """Where a step speaks, and how it asks whether to carry on.

    Frozen, and deliberately so: it crosses threads — the window sets `cancel`
    while a worker reads it — and the only mutable thing in it is an `Event`,
    which is built for exactly that. Any later channel that needs mutable state
    belongs beside the Event as one of its own, not as a field somebody will
    assign from two threads at once.
    """

    say: Notify = field(default=_unsaid)
    at: Advance = field(default=_unwatched)
    cancel: threading.Event | None = None

    def stop_if_asked(self) -> None:
        """Raises `Stopped` when the person asked to stop.

        Called where the work is already saved, not wherever it is convenient:
        stopping in the middle of a step throws that step away.
        """
        if self.cancel is not None and self.cancel.is_set():
            raise Stopped("stopped")

    @property
    def stopping(self) -> bool:
        """The same question without the exception, for loops that clean up."""
        return self.cancel is not None and self.cancel.is_set()
