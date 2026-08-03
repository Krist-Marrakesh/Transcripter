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

    @property
    def duration(self) -> float:
        return self.end - self.start


class SpeakerTurn(BaseModel):
    """A stretch spoken by one person. The result of diarization."""

    model_config = ConfigDict(frozen=True)

    start: float
    end: float
    speaker: str


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
    created_at: datetime = Field(default_factory=lambda: datetime.now(UTC))

    @property
    def text(self) -> str:
        """Solid text without timestamps."""
        return " ".join(s.text.strip() for s in self.segments if s.text.strip())

    @property
    def speakers(self) -> list[str]:
        """Speakers in order of first appearance."""
        seen: dict[str, None] = {}
        for segment in self.segments:
            if segment.speaker is not None:
                seen.setdefault(segment.speaker, None)
        return list(seen)

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


class Summary(BaseModel):
    """A transcript summary produced by a map-reduce fold over chunks."""

    language: str
    llm_model: str
    overview: str
    key_points: list[str] = Field(default_factory=list)
    action_items: list[str] = Field(default_factory=list)
    chunk_summaries: list[str] = Field(default_factory=list)
    created_at: datetime = Field(default_factory=lambda: datetime.now(UTC))
