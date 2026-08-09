"""Что происходит, когда человек нажал «стоп».

Проверяется не сам факт остановки, а два его последствия. Первое: остановка
должна быть замечена там, где работа длинная, иначе кнопка нажата, а окно
минутами молчит — и это выглядит как зависшее приложение, а не как «сейчас
доделаю». Второе: на месте недоделанного не должно остаться ничего, что
следующий запуск примет за готовое.
"""

from __future__ import annotations

import subprocess
import threading
import wave

import pytest

from transcriber import vad
from transcriber.cache import building
from transcriber.ingest import media
from transcriber.report import Report, Stopped


def recording(path, seconds: float = 3.0):
    """Тихая запись нужной длины — VAD в ней ничего не найдёт, и это неважно."""
    with wave.open(str(path), "wb") as handle:
        handle.setnchannels(1)
        handle.setsampwidth(2)
        handle.setframerate(16_000)
        handle.writeframes(b"\x00\x00" * int(16_000 * seconds))
    return path


def test_the_scan_answers_a_stop_between_blocks(tmp_path, monkeypatch):
    """Регрессия: поиск речи был сплошным отрезком без единой проверки.

    На часовой записи это минуты, в течение которых нажатая кнопка не значила
    ничего, а окно не двигалось — ровно то, что человек называет зависанием.
    """
    monkeypatch.setattr(vad, "SCAN_BLOCK", 1.0)
    stop = threading.Event()
    stop.set()

    with pytest.raises(Stopped):
        vad.scan(recording(tmp_path / "тихо.wav"), Report(cancel=stop))


def test_the_scan_reports_how_far_it_got(tmp_path, monkeypatch):
    """Полоса обязана двигаться и здесь: шаг длинный и до сих пор был немым."""
    monkeypatch.setattr(vad, "SCAN_BLOCK", 1.0)
    seen: list[float] = []

    vad.scan(recording(tmp_path / "тихо.wav"), Report(at=seen.append))

    assert seen == sorted(seen)
    assert seen[-1] == pytest.approx(1.0)


def test_a_running_tool_is_killed_on_a_stop():
    """Пока ffmpeg перекодирует четырёхчасовое видео, ждать его — не остановка."""
    stop = threading.Event()
    stop.set()

    with pytest.raises(Stopped):
        media._run(["sleep", "30"], Report(cancel=stop))


def test_a_tool_that_finishes_is_not_disturbed():
    """Обычный путь остался прежним: вывод возвращается целиком."""
    assert media._run(["echo", "готово"]).strip() == "готово"


def test_a_tool_that_failed_still_explains_itself():
    """Причину падения показывает stderr, и она не должна потеряться в ожидании."""
    with pytest.raises(media.FFmpegError, match="exited with code"):
        media._run(["sh", "-c", "echo сломалось >&2; exit 3"])


def test_an_interrupted_build_leaves_no_artifact(tmp_path):
    """Регрессия: оборванный ffmpeg оставлял недописанный WAV под готовым именем.

    Проверка `exists()` — единственное, что отделяет следующий запуск от этого
    файла, поэтому обрыв обязан не оставить под ним ничего: иначе запись молча
    окажется короче себя, и так навсегда.
    """
    target = tmp_path / "звук.wav"

    with pytest.raises(RuntimeError):
        with building(target) as partial:
            partial.write_bytes(b"half")
            raise RuntimeError("оборвалось")

    assert not target.exists()
    assert list(tmp_path.iterdir()) == []


def test_a_finished_build_publishes_the_artifact(tmp_path):
    target = tmp_path / "звук.wav"

    with building(target) as partial:
        partial.write_bytes(b"whole")

    assert target.read_bytes() == b"whole"
    assert list(tmp_path.iterdir()) == [target]


def test_extraction_publishes_only_a_whole_file(tmp_path, monkeypatch):
    """Тот же обрыв, но по настоящему пути: через `_prepare_file`."""
    from transcriber import ingest
    from transcriber.cache import ArtifactCache

    source = recording(tmp_path / "лекция.wav", seconds=1.0)
    cache = ArtifactCache(tmp_path / "кэш")

    def die(_source, target, _report=None):
        target.write_bytes(b"half")
        raise subprocess.SubprocessError("оборвалось")

    monkeypatch.setattr(media, "extract_audio", die)
    with pytest.raises(subprocess.SubprocessError):
        ingest.prepare(str(source), cache)

    monkeypatch.undo()
    prepared = ingest.prepare(str(source), cache)

    assert prepared.duration == pytest.approx(1.0, abs=0.01)
