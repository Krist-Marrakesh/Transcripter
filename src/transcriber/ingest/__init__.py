"""Ingest: приведение любого источника к WAV 16 кГц моно."""

from __future__ import annotations

import shutil
from collections.abc import Callable
from dataclasses import dataclass
from pathlib import Path

from ..cache import ArtifactCache, fingerprint, stable_key
from . import media, youtube

Notify = Callable[[str], None]
Advance = Callable[[float], None]
"""Доля выполненного у долгого шага — здесь это скачивание."""

# Сколько раз перекачивать оборвавшуюся загрузку, прежде чем работать с тем, что
# доехало. Обрыв — дело сети и обычно лечится повтором; но если длительность
# врёт сам источник (частая история у записей трансляций), повторы бессмысленны.
DOWNLOAD_ATTEMPTS = 3


@dataclass(frozen=True)
class Source:
    """Нормализованный вход пайплайна."""

    audio: Path
    """WAV 16 кГц моно — то, с чем работают все последующие шаги."""

    origin: str
    """Исходный путь или URL, для метаданных и имён выходных файлов."""

    title: str
    duration: float


def prepare(
    target: str,
    cache: ArtifactCache,
    notify: Notify | None = None,
    advance: Advance | None = None,
) -> Source:
    """Приводит файл или ссылку к формату пайплайна.

    Результат кэшируется по содержимому исходника: повторный запуск на том же
    файле не декодирует его заново.

    Доля выполненного считается только для скачивания: у локального файла
    сколько-нибудь долгий шаг один — перекодирование, а сообщать о его ходе
    ffmpeg умеет только парсингом собственного вывода.
    """
    say = notify or (lambda _: None)
    if youtube.is_url(target):
        return _prepare_url(target, cache, say, advance)
    return _prepare_file(Path(target).expanduser().resolve(), cache, say)


def _prepare_file(path: Path, cache: ArtifactCache, say: Notify) -> Source:
    if not path.exists():
        raise FileNotFoundError(f"файл не найден: {path}")

    key = stable_key(fingerprint(path), step="wav16k")
    wav = cache.reserve("audio", key, ".wav")
    if not wav.exists():
        media.extract_audio(path, wav)

    duration = media.probe(wav).duration
    # Локальный файл перекачать неоткуда, поэтому только предупреждаем. Проверка
    # идёт и при готовом WAV: недостача видна на исходнике, а не на результате.
    if warning := media.truncation_warning(media.probe(path).duration, duration):
        say(warning)

    return Source(audio=wav, origin=str(path), title=path.stem, duration=duration)


def _prepare_url(
    url: str, cache: ArtifactCache, say: Notify, advance: Advance | None = None
) -> Source:
    info = youtube.probe(url)

    # Ключ по URL, а не по содержимому: скачивать файл ради отпечатка,
    # чтобы потом понять, что он уже скачан, — бессмысленно.
    key = stable_key(url, step="wav16k")
    wav = cache.reserve("audio", key, ".wav")

    # Оборванная загрузка могла осесть в кэше прошлым запуском, поэтому готовый
    # WAV тоже проверяется — иначе обрезанная лекция закрепится там навсегда.
    if not wav.exists() or media.truncation_warning(info.duration, media.probe(wav).duration):
        _download_audio(url, wav, info.duration, cache, key, say, advance)

    return Source(
        audio=wav,
        origin=url,
        title=info.title,
        # Именно фактическая длительность: заявленная у оборванной записи
        # обещает то, чего в звуке нет, и транскрипт получился бы «на два часа».
        duration=media.probe(wav).duration,
    )


def _download_audio(
    url: str,
    wav: Path,
    declared: float,
    cache: ArtifactCache,
    key: str,
    say: Notify,
    advance: Advance | None = None,
) -> None:
    """Качает и конвертирует, повторяя попытку, если звук доехал не целиком."""
    for attempt in range(1, DOWNLOAD_ATTEMPTS + 1):
        staging = cache.reserve("downloads", key, "")
        # Остатки прерванной загрузки собьют выбор файла — начинаем с чистого места
        # и убираем за собой в любом случае.
        shutil.rmtree(staging, ignore_errors=True)
        staging.mkdir(parents=True)
        try:
            raw = youtube.download_audio(url, staging, advance)
            media.extract_audio(raw, wav)
        finally:
            shutil.rmtree(staging, ignore_errors=True)

        warning = media.truncation_warning(declared, media.probe(wav).duration)
        if warning is None:
            return
        if attempt < DOWNLOAD_ATTEMPTS:
            say(f"{warning}; downloading again, attempt {attempt + 1} of {DOWNLOAD_ATTEMPTS}")
        else:
            say(warning)
