"""История распознанных записей.

Хранит собственную копию транскрипта, а не ссылку в кэш. Кэш чистят не
задумываясь — историю терять нельзя, в этом её смысл. Копия занимает килобайты
против сотен мегабайт исходного аудио.

Индекс отдельным файлом: список открывается мгновенно, не читая каждую запись.
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

# Больше — не список, а свалка; листать такое всё равно никто не станет.
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
    """Добавляет запись в историю и возвращает её.

    Ключ считается по самому транскрипту — тексту, языку и модели. Поэтому
    повторный прогон той же записи обновляет строку вместо дубликата, а прогон
    другой моделью заводит отдельную: тексты у них разные, и сравнить их — то,
    ради чего модель и меняют.
    """
    key = stable_key(transcript.text, language=transcript.language, model=transcript.asr_model)

    ENTRIES.mkdir(parents=True, exist_ok=True)
    (ENTRIES / f"{key}.json").write_text(transcript.model_dump_json(indent=2), encoding="utf-8")

    entry = {
        "key": key,
        "origin": origin,
        "kind": "url" if origin.startswith(("http://", "https://")) else "file",
        "title": title,
        # Имя нормализованного WAV в кэше: из транскрипта его не вывести, а без
        # него запись из истории откроется без звука.
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
    """Записи от свежих к старым. Битый индекс трактуется как пустой."""
    try:
        data = json.loads(INDEX.read_text(encoding="utf-8"))
    except (OSError, ValueError):
        return []
    return data if isinstance(data, list) else []


def open_entry(key: str) -> Transcript | None:
    """Достаёт сохранённый транскрипт — исходный файл для этого не нужен."""
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
    """Короткое «о чём это».

    С локальной LLM — осмысленная тема вроде «лекция о теореме Байеса». Без неё
    первая фраза записи: она почти всегда вводит в тему и всяко лучше пустоты.
    """
    text = transcript.text.strip()
    if not text:
        return "пустая запись"

    if describe is not None:
        try:
            topic = describe(text).strip()
            if topic:
                return _trim(topic)
        except Exception:
            # Тема — украшение списка. Её потеря не повод ронять сохранение.
            pass

    return _trim(_opening(text))


def _opening(text: str, least: int = 45) -> str:
    """Начало записи длиной хотя бы в осмысленную фразу.

    Первое предложение бывает «Привет!» — такой темой список не пролистаешь,
    поэтому добираем следующие, пока строка не станет содержательной.
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
