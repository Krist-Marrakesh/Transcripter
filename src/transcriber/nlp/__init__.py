"""The NLP layer: translation and summaries on top of a finished transcript.

It works on text and never touches recognition. So it can be run as many times as
one likes without transcribing again — the original `Transcript` is in the cache.
"""

from __future__ import annotations

from .llm import LLM, create_llm
from .names import name_speakers
from .summarize import summarize, topic
from .translate import translate

__all__ = ["LLM", "create_llm", "name_speakers", "summarize", "topic", "translate"]
