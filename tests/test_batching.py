"""Разбиение записи на порции.

Ради этого память перестаёт зависеть от длины записи: в неё попадает одна
порция, а не восемь часов звука. Ошибка здесь дорогая вдвойне — либо потолок
возвращается, либо порции режут речь посередине слова.
"""

from __future__ import annotations

import threading
from pathlib import Path

import pytest

from transcriber import audio, vad
from transcriber.config import load_settings
from transcriber.ingest import Source
from transcriber.pipeline import Pipeline
from transcriber.report import Report, Stopped
from transcriber.vad import SpeechRegion, batches

CACHE = Path.home() / ".cache/transcript/audio/53d426fae351ad3a643a43c9840e693f.wav"


def speech(portion) -> float:
    return sum(region.duration for region in portion)


def test_a_portion_never_exceeds_the_limit():
    regions = [SpeechRegion(0, 10), SpeechRegion(20, 35), SpeechRegion(60, 64)]

    portions = batches(regions, 20)

    assert all(speech(p) <= 20 for p in portions)
    assert sum(speech(p) for p in portions) == pytest.approx(29)


def test_cuts_fall_in_the_pauses():
    """Границы порций совпадают с границами, которые нашёл VAD.

    Иначе рез пришёлся бы на середину слова, и обе половины распознались бы
    неверно — по одному обрывку с каждой стороны.
    """
    regions = [SpeechRegion(0, 10), SpeechRegion(20, 30), SpeechRegion(60, 70)]

    portions = batches(regions, 20)

    edges = {(r.start, r.end) for p in portions for r in p}
    assert edges == {(0, 10), (20, 30), (60, 70)}


def finishes(call, seconds: float = 5.0):
    """Ждёт результата ограниченное время.

    Обычному тесту сторож не нужен, но здесь проверяется как раз незавершаемость:
    без него сломанный код не падает, а вешает прогон и набивает память.
    """
    answer: list = []
    worker = threading.Thread(target=lambda: answer.append(call()), daemon=True)
    worker.start()
    worker.join(seconds)
    assert answer, "разбиение не завершилось"
    return answer[0]


# Первые восемнадцать речевых участков настоящей 33-минутной лекции, как их нашёл
# VAD. Первые семнадцать дают в сумме ровно 120 секунд по десятичному счёту и
# 119.99999999999999 по двоичному — придумать такое труднее, чем взять готовое.
BRIM = [
    (11.0, 14.4), (14.8, 18.5), (18.7, 51.4), (51.6, 56.0), (56.4, 59.6), (60.0, 62.5),
    (62.8, 68.1), (68.5, 70.3), (70.6, 78.7), (79.4, 94.6), (95.1, 99.7), (99.9, 105.3),
    (105.7, 110.6), (111.2, 119.7), (120.1, 127.8), (128.0, 134.0), (134.4, 137.0),
    (137.2, 139.9),
]  # fmt: skip


def test_a_portion_filled_to_the_brim_does_not_hang():
    """Регрессия: остаток в 1.4e-14 секунды считался местом под ещё один кусок.

    `start + остаток` округляется обратно в `start`, разрез не двигается с места,
    цикл не кончается и набивает память регионами нулевой длины. На этой самой
    записи было 19 ГБ и намертво занятое окно, а в журнале — тишина после строки
    про найденную речь.
    """
    regions = [SpeechRegion(start=start, end=end) for start, end in BRIM]

    portions = finishes(lambda: batches(regions, 120.0))

    assert all(region.end > region.start for portion in portions for region in portion)
    assert sum(len(portion) for portion in portions) == len(regions)


def test_speech_longer_than_a_portion_is_split():
    """Страховка: непрерывная речь длиннее лимита должна как-то поместиться."""
    portions = batches([SpeechRegion(0, 50)], 20)

    assert [speech(p) for p in portions] == [20, 20, 10]
    assert portions[0][0].end == portions[1][0].start


def test_without_regions_the_whole_recording_is_divided():
    """VAD выключен или ничего не услышал — путь у вызывающего остаётся один."""
    portions = batches([], 20, total=45)

    assert [(r.start, r.end) for p in portions for r in p] == [(0, 20), (20, 40), (40, 45)]


def test_nothing_at_all_gives_nothing():
    assert batches([], 20, total=0) == []


@pytest.mark.skipif(not CACHE.exists(), reason="нужна запись в кэше")
def test_streaming_detection_matches_reading_it_whole():
    """Регрессия по замыслу: шов между блоками не должен рождать лишних границ.

    `scan` читает файл кусками, `detect` — целиком. Речь от этого меняться не
    должна, иначе каждые десять минут в транскрипте появлялся бы разрыв фразы.
    """
    whole = vad.detect(audio.load(CACHE))
    streamed = vad.scan(CACHE)

    assert len(streamed) == len(whole)
    for a, b in zip(streamed, whole, strict=True):
        assert a.start == pytest.approx(b.start, abs=0.05)
        assert a.end == pytest.approx(b.end, abs=0.05)


@pytest.mark.skipif(not CACHE.exists(), reason="нужна запись в кэше")
def test_only_the_named_stretches_are_read():
    """Паузы в память не попадают — на лекции это три четверти файла."""
    spans = [(10.0, 20.0), (100.0, 105.0)]

    samples = audio.read_spans(CACHE, spans)

    assert len(samples) == pytest.approx(audio.to_samples(15), abs=2)


@pytest.mark.skipif(not CACHE.exists(), reason="нужна запись в кэше")
def test_stopping_between_portions(tmp_path):
    """Остановка — пауза, а не отмена: сделанное остаётся в кэше.

    Проверяется на границе порций, потому что именно там работа уже сохранена;
    внутри порции бэкенд прервать нельзя, и ждать его — честнее, чем терять.
    """
    stop = threading.Event()
    stop.set()
    settings = load_settings(cache_dir=tmp_path)

    with pytest.raises(Stopped):
        Pipeline(settings, Report(cancel=stop)).transcribe(
            Source(audio=CACHE, origin=str(CACHE), title="проба", duration=180.0)
        )
