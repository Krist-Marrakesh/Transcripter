"""The history of recognised recordings.

It keeps a copy of the transcript rather than a pointer into the cache. A cache
is wiped without thinking, and losing the history is the one thing this exists to
prevent. The copy costs kilobytes against hundreds of megabytes of source audio.

The index is a file of its own: the list opens at once, without reading every
entry to build it.
"""

from __future__ import annotations

import json
import re
from collections.abc import Callable
from datetime import UTC, datetime

from .. import paths
from ..cache import stable_key
from ..models import Transcript

ROOT = paths.data_dir()
INDEX = ROOT / "history.json"
ENTRIES = ROOT / "entries"

# More than this is not a list but a heap; nobody scrolls through one anyway.
LIMIT = 200

_SENTENCE = re.compile(r"(?<=[.!?…])\s+")


def remember(
    transcript: Transcript,
    *,
    origin: str,
    title: str,
    audio: str = "",
    topic: Callable[[str], str] | None = None,
) -> dict[str, str | float]:
    """Adds an entry to the history and returns it.

    The key is computed from the transcript itself — its text, language and
    model. So running the same recording again updates the row instead of
    duplicating it, while running it through another model starts a separate one:
    their texts differ, and comparing them is the whole reason models get changed.
    """
    key = stable_key(transcript.text, language=transcript.language, model=transcript.asr_model)

    ENTRIES.mkdir(parents=True, exist_ok=True)
    (ENTRIES / f"{key}.json").write_text(transcript.model_dump_json(indent=2), encoding="utf-8")

    entry = {
        "key": key,
        "origin": origin,
        "kind": "url" if origin.startswith(("http://", "https://")) else "file",
        "title": title,
        # The name of the normalised WAV in the cache: it cannot be derived from
        # the transcript, and without it a history entry opens without sound.
        "audio": audio,
        "topic": _topic(transcript, topic),
        "language": transcript.language,
        "duration": transcript.duration,
        "speakers": len(transcript.speakers),
        "added": datetime.now(UTC).isoformat(timespec="seconds"),
    }

    entries = [item for item in load() if item.get("key") != key]
    entries.insert(0, entry)
    _write(entries[:LIMIT])
    return entry


def load() -> list[dict]:
    """Entries newest first. A damaged index counts as an empty one."""
    try:
        data = json.loads(INDEX.read_text(encoding="utf-8"))
    except (OSError, ValueError):
        return []
    return data if isinstance(data, list) else []


def open_entry(key: str) -> Transcript | None:
    """Fetches the stored transcript — the source file is not needed for it."""
    path = ENTRIES / f"{key}.json"
    if not path.exists():
        return None
    try:
        return Transcript.load(path)
    except ValueError:
        return None


def forget(key: str) -> None:
    _write([item for item in load() if item.get("key") != key])
    (ENTRIES / f"{key}.json").unlink(missing_ok=True)


def _write(entries: list[dict]) -> None:
    INDEX.parent.mkdir(parents=True, exist_ok=True)
    tmp = INDEX.with_suffix(".tmp")
    tmp.write_text(json.dumps(entries, ensure_ascii=False, indent=2), encoding="utf-8")
    tmp.replace(INDEX)


def _topic(transcript: Transcript, describe: Callable[[str], str] | None) -> str:
    """A short "what this is about".

    With a local LLM, a real subject — "a lecture on Bayes' theorem". Without
    one, the opening phrase of the recording: it nearly always introduces the
    subject, and is in any case better than nothing.
    """
    text = transcript.text.strip()
    if not text:
        return "an empty recording"

    if describe is not None:
        try:
            topic = describe(text).strip()
            if topic:
                return _trim(topic)
        except Exception:
            # The subject decorates the list. Losing it is no reason to lose the save.
            pass

    return _trim(_opening(text))


def _opening(text: str, least: int = 45) -> str:
    """The opening of a recording, at least a meaningful phrase long.

    The first sentence is sometimes "Hello!" — a list of those tells nobody
    anything, so we keep taking the next until the line says something.
    """
    collected = ""
    for sentence in _SENTENCE.split(text):
        collected = f"{collected} {sentence}".strip()
        if len(collected) >= least:
            break
    return collected


def _trim(text: str, limit: int = 110) -> str:
    text = " ".join(text.split())
    if len(text) <= limit:
        return text
    return text[:limit].rsplit(" ", maxsplit=1)[0] + "…"
