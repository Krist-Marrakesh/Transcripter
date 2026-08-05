"""Сколько работы уже позади — у скачивания и у распознавания.

Проверяется не точность доли, а свойства, без которых полоса вредна: она не
должна врать про завершённость, застревать под конец и ронять сам шаг. Час
распознавания стоит дороже любой шкалы.

Отдельно проверяется, что отчёты действительно доходят. На этом уже попались:
заглушка была исправна, а подмена промахивалась мимо модуля, и распознавание
молча шло без прогресса при зелёных тестах.
"""

from __future__ import annotations

import importlib

import numpy as np
import pytest

from transcriber.asr import mlx_backend
from transcriber.asr.faster_backend import FasterWhisperBackend
from transcriber.ingest import youtube


def counter(report):
    """Счётчик, привязанный к конкретному получателю, — как это делает бэкенд."""
    return type("Bound", (mlx_backend._Counter,), {"report": staticmethod(report)})


def test_counter_reports_a_share_of_the_whole():
    seen: list[float] = []

    with counter(seen.append)(total=200) as bar:
        bar.update(50)
        bar.update(50)

    assert seen == [0.25, 0.5]


def test_counter_never_promises_more_than_done():
    """Последнее окно Whisper выходит за конец записи — доля не должна перевалить."""
    seen: list[float] = []

    with counter(seen.append)(total=100) as bar:
        bar.update(150)

    assert seen == [1.0]


def test_counter_survives_an_unknown_total():
    """`total=0` — делить не на что, но падать тут нельзя."""
    seen: list[float] = []

    with counter(seen.append)(total=0) as bar:
        bar.update(10)

    assert seen == []


def test_counter_takes_whatever_tqdm_takes():
    """Библиотека может завтра добавить свои аргументы — узкая сигнатура упала бы."""
    with counter(lambda _: None)(range(3), total=1, unit="frames", disable=True, desc="ходъ"):
        pass


def engine():
    """Модуль, где живёт полоса mlx-whisper.

    Именно модуль, а не одноимённая функция: пакет переопределяет это имя в
    своём `__init__`, и на этом уже один раз попались — подмена уходила в
    пустоту, а распознавание молча шло без отчётов.
    """
    pytest.importorskip("mlx_whisper")
    return importlib.import_module("mlx_whisper.transcribe")


def test_the_counter_actually_takes_the_place_of_the_bar():
    """Регрессия: подменялась функция вместо модуля, и патч не доезжал.

    Тест на самой заглушке этого не ловил — она исправна, промахивалась
    установка.
    """
    module = engine()
    original = module.tqdm

    with mlx_backend._reporting(lambda _: None):
        assert module.tqdm is not original
        assert issubclass(module.tqdm.tqdm, mlx_backend._Counter)


def test_patch_is_removed_after_a_failure():
    """Подмена видна всему процессу: оставить её после исключения нельзя."""
    module = engine()
    original = module.tqdm

    with pytest.raises(RuntimeError), mlx_backend._reporting(lambda _: None):
        raise RuntimeError("распознавание упало")

    assert module.tqdm is original


def test_nothing_is_patched_without_a_listener():
    """Без получателя трогать чужой модуль незачем."""
    module = engine()
    original = module.tqdm

    with mlx_backend._reporting(None):
        assert module.tqdm is original


def test_faster_measures_against_the_audio_it_was_given():
    """Мерой служит длина после VAD, а не исходной записи.

    Иначе на лекции, где вырезано три четверти пауз, полоса замерла бы на
    четверти и там осталась — при полностью распознанном файле.
    """
    seen: list[float] = []
    samples = np.zeros(16_000 * 100, dtype=np.float32)  # сто секунд
    spans = [type("S", (), {"end": 50.0})(), type("S", (), {"end": 100.0})()]

    assert len(list(FasterWhisperBackend._watched(spans, samples, seen.append))) == 2
    assert seen == [0.5, 1.0]


def test_faster_without_a_listener_still_yields_everything():
    samples = np.zeros(16_000, dtype=np.float32)
    spans = [type("S", (), {"end": 0.5})()]

    assert len(list(FasterWhisperBackend._watched(spans, samples, None))) == 1


def test_youtube_reports_a_share_of_the_download():
    seen: list[float] = []
    hook = youtube._reporter(seen.append)

    hook({"status": "downloading", "downloaded_bytes": 250, "total_bytes": 1000})
    hook({"status": "downloading", "downloaded_bytes": 750, "total_bytes": 1000})

    assert seen == [0.25, 0.75]


def test_youtube_falls_back_to_an_estimated_size():
    """Точную длину сервер отдаёт не всегда — оценка лучше пустой шкалы."""
    seen: list[float] = []

    youtube._reporter(seen.append)(
        {"status": "downloading", "downloaded_bytes": 100, "total_bytes_estimate": 400}
    )

    assert seen == [0.25]


def test_youtube_stays_silent_without_a_size():
    """Шкала, ползущая по догадке, хуже отсутствующей."""
    seen: list[float] = []

    youtube._reporter(seen.append)({"status": "downloading", "downloaded_bytes": 100})

    assert seen == []


def test_youtube_closes_the_bar_on_the_last_event():
    """Оценка почти всегда мимо, и без этого полоса застревала бы под конец."""
    seen: list[float] = []

    youtube._reporter(seen.append)({"status": "finished"})

    assert seen == [1.0]


def test_the_hook_reaches_yt_dlp():
    """Та же проверка, что и у mlx: доходит ли отчёт до чужой библиотеки.

    Дальше начинается её договор — вызывать переданное по ходу закачки; это
    проверяется только настоящей загрузкой и здесь не покрыто.
    """
    pytest.importorskip("yt_dlp")
    hook = youtube._reporter(lambda _: None)

    with youtube._ydl(progress_hooks=[hook]) as ydl:
        assert hook in ydl.params["progress_hooks"]
