"""Что окно помнит между запусками.

Отдельно от `Settings`: те задаются окружением и `.env`, а это — выбор,
сделанный мышью. Смешивать их значило бы, что кнопка в интерфейсе молча
переписывает конфигурацию проекта.

Файл лежит рядом с конфигами, а не в кэше: кэш можно чистить не задумываясь,
а выбранную папку терять обидно.
"""

from __future__ import annotations

import json
from pathlib import Path

from .. import paths

PATH = paths.config_dir() / "app.json"


def load() -> dict[str, str]:
    """Читает состояние. Битый или отсутствующий файл — просто пустой выбор."""
    try:
        data = json.loads(PATH.read_text(encoding="utf-8"))
    except (OSError, ValueError):
        return {}
    return data if isinstance(data, dict) else {}


def save(**changes: str) -> None:
    """Дописывает значения, не затирая остальные."""
    data = load() | {key: str(value) for key, value in changes.items()}
    PATH.parent.mkdir(parents=True, exist_ok=True)
    tmp = PATH.with_suffix(".tmp")
    tmp.write_text(json.dumps(data, ensure_ascii=False, indent=2), encoding="utf-8")
    tmp.replace(PATH)


# Один и тот же путь, что и в `Settings`: раньше умолчание было записано в двух
# местах и могло разъехаться при первой же правке.
DEFAULT_OUTPUT = paths.default_output()


def output_dir(configured: Path) -> Path:
    """Папка для файлов: выбранная мышью, затем из настроек, затем по умолчанию.

    Относительный путь из `.env` окно игнорирует. В терминале «output» означает
    «рядом с текущим каталогом» и это разумно, но Finder запускает приложение из
    корня, где такой путь либо не создастся, либо уедет в неожиданное место.
    """
    chosen = load().get("output_dir")
    if chosen:
        return Path(chosen)
    return configured if configured.is_absolute() else DEFAULT_OUTPUT


def ensure(directory: Path) -> bool:
    """Создаёт папку заранее и говорит, удалось ли.

    Заранее — потому что пустая папка на месте объясняет, куда смотреть, ещё до
    первого сохранения. Отказ возвращается значением, а не исключением: для
    окна это не повод не открыться (macOS спрашивает доступ к «Документам»
    отдельно, и до ответа запись туда запрещена), а решает это вызывающий.
    """
    try:
        directory.mkdir(parents=True, exist_ok=True)
        return True
    except OSError:
        return False
