"""The recording the window has open, and what has been written out of it.

One object because these move as one. They used to be seven fields on the bridge
— source, transcript, translation, which of the two is shown, the files saved,
which of those fell behind, the summary — and three separate places had to reset
all seven in step. Nothing enforced that but memory, and the price of forgetting
one was a translation from the previous recording sitting under the new one, or
a warning about stale files that belonged to somebody else's lecture.

Here the invariant is structural: opening a recording builds the whole thing, so
there is no state left over from the last one to forget.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path

from ..models import Transcript

ORIGINAL = "original"
TRANSLATION = "translation"


@dataclass
class OpenRecording:
    """What is on screen, in both its versions, and its trail on disk."""

    title: str
    transcript: Transcript
    # The translation lives beside the original rather than replacing it: saving
    # has to write what a person is looking at, and going back has to be possible.
    translated: Transcript | None = None
    showing: str = ORIGINAL
    # A summary is minutes of a model's work and used to live only on screen.
    summary: str | None = None
    # Files written out of this recording, and those of them that no longer match
    # what is shown. A person whose window updated has no way of knowing the
    # folder did not, and would carry off the older version certain it was current.
    saved: list[str] = field(default_factory=list)
    stale: list[str] = field(default_factory=list)

    @property
    def current(self) -> Transcript:
        """What is on screen — and therefore what gets saved."""
        if self.showing == TRANSLATION and self.translated is not None:
            return self.translated
        return self.transcript

    def wrote(self, path: str) -> None:
        """Remembers a file written out; it matches the screen again."""
        if path not in self.saved:
            self.saved.append(path)
        # Only this one. Its neighbours were written from the same screen but may
        # still be a version behind, and saying otherwise would be a guess.
        self.stale = [old for old in self.stale if old != path]

    def fell_behind(self) -> list[str]:
        """Declares every saved file older than the screen and names them."""
        self.stale = list(self.saved)
        return self.behind

    @property
    def behind(self) -> list[str]:
        """Names — not paths — of the files that no longer match the screen."""
        return [Path(old).name for old in self.stale]
