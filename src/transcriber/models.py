"""Pipeline data models.

`Transcript` is the central artifact: the recognition result plus speaker labels.
It is serialised to JSON and cached, so the expensive steps (ASR, diarization)
run exactly once. Translation and summaries live as separate artifacts and never
modify the original transcript.
"""

from __future__ import annotations

from datetime import UTC, datetime
from pathlib import Path
from typing import Self

from pydantic import BaseModel, ConfigDict, Field


class Word(BaseModel):
    """A word with a timestamp of its own.

    Filled in only when ASR ran with word-level timings: they cost noticeably
    more and are not always needed.
    """

    model_config = ConfigDict(frozen=True)

    start: float
    end: float
    text: str
    probability: float | None = None


class Segment(BaseModel):
    """A line of speech — a continuous chunk with timestamps.

    The base unit of the pipeline: diarization, translation and subtitle export
    all work on segments.
    """

    start: float
    end: float
    text: str
    speaker: str | None = None
    words: list[Word] = Field(default_factory=list)
    formulas: list[str] = Field(default_factory=list)
    """Formulas spoken here in words, written out in LaTeX. Empty until asked.

    Beside the text and never instead of it. A lecture says "the limit of sine
    x over x as x goes to zero equals one", and the person who was in the room
    reconstructs the blackboard from that while a reader of the transcript does
    not — but the words are still what was said, and a transcript owes them.
    """

    @property
    def duration(self) -> float:
        return self.end - self.start


class SpeakerTurn(BaseModel):
    """A stretch spoken by one person. The result of diarization."""

    model_config = ConfigDict(frozen=True)

    start: float
    end: float
    speaker: str


class Portion(BaseModel):
    """One portion of a recording, recognised and cached on its own.

    A long recording is recognised in pieces so that memory does not grow with its
    length. Each piece is stored as it is finished, which is what makes an
    interruption cheap: coming back after a crash three hours in resumes at the
    fourth hour rather than at the first.

    The timestamps here are already on the original axis — the caller's portion
    boundaries are not needed to read this back.
    """

    segments: list[Segment]
    language: str


class Diarization(BaseModel):
    """Speaker layout of a recording.

    Cached separately from the transcript: diarization and ASR are computed
    independently, and turning one on must not force recomputing the other.
    """

    turns: list[SpeakerTurn]
    model: str
    created_at: datetime = Field(default_factory=lambda: datetime.now(UTC))


class Transcript(BaseModel):
    """The recognition result for one recording."""

    source: str
    language: str
    duration: float
    segments: list[Segment]
    asr_model: str
    names: dict[str, str] = Field(default_factory=dict)
    """Who each speaker label turned out to be. Empty until anyone asks.

    A mapping and not a rewrite of the labels, because the two answer different
    questions and change at different times. Labelling again — with the speaker
    count given, say — produces the same `Спикер N` and would carry away any
    name written into them; correcting one name would otherwise mean editing
    every segment that person speaks. The two meet only where a transcript is
    rendered.
    """
    created_at: datetime = Field(default_factory=lambda: datetime.now(UTC))

    @property
    def text(self) -> str:
        """Solid text without timestamps."""
        return " ".join(s.text.strip() for s in self.segments if s.text.strip())

    @property
    def speakers(self) -> list[str]:
        """Speaker labels in order of first appearance, before any naming."""
        seen: dict[str, None] = {}
        for segment in self.segments:
            if segment.speaker is not None:
                seen.setdefault(segment.speaker, None)
        return list(seen)

    def as_named(self) -> Self:
        """A copy carrying the people instead of the labels, for rendering.

        Labels nobody identified stay as they are: `Спикер 2` says less than a
        name and more than a name that was made up.
        """
        if not self.names:
            return self
        return self.model_copy(
            update={
                "segments": [
                    segment.model_copy(update={"speaker": self.names[segment.speaker]})
                    if segment.speaker in self.names
                    else segment
                    for segment in self.segments
                ]
            }
        )

    @classmethod
    def load(cls, path: Path) -> Self:
        """Reads a saved transcript — this is how history opens its own copies."""
        return cls.model_validate_json(path.read_text(encoding="utf-8"))


class Translation(BaseModel):
    """A translated transcript. Stored separately so the original stays intact."""

    source_language: str
    target_language: str
    llm_model: str
    segments: list[Segment]
    created_at: datetime = Field(default_factory=lambda: datetime.now(UTC))

    @property
    def text(self) -> str:
        return " ".join(s.text.strip() for s in self.segments if s.text.strip())


class SpokenFormulas(BaseModel):
    """Formulas found in a transcript, keyed by segment number as a string.

    A string because JSON has no integer keys, and this is stored as JSON. The
    number rather than the text: two segments of a lecture are often word for
    word the same, and «то же самое» is not a key.
    """

    formulas: dict[str, list[str]]
    llm_model: str
    created_at: datetime = Field(default_factory=lambda: datetime.now(UTC))


class SpeakerNames(BaseModel):
    """Who the speaker labels turned out to be, read out of the speech itself.

    Cached in its own right, like a translation: it costs an LLM and the answer
    does not change while the text does not.
    """

    names: dict[str, str]
    llm_model: str
    created_at: datetime = Field(default_factory=lambda: datetime.now(UTC))


class Summary(BaseModel):
    """A transcript summary produced by a map-reduce fold over chunks."""

    language: str
    llm_model: str
    overview: str
    key_points: list[str] = Field(default_factory=list)
    action_items: list[str] = Field(default_factory=list)
    chunk_summaries: list[str] = Field(default_factory=list)
    created_at: datetime = Field(default_factory=lambda: datetime.now(UTC))
