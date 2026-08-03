"""NLP-слой: перевод и саммари поверх готового транскрипта.

Работает с текстом, в распознавание не вмешивается. Поэтому запускается сколько
угодно раз без повторной транскрибации — исходный `Transcript` лежит в кэше.
"""

from __future__ import annotations

from .llm import LLM, create_llm
from .summarize import summarize, topic
from .translate import translate

__all__ = ["LLM", "create_llm", "summarize", "topic", "translate"]
