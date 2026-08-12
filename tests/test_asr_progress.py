"""Сколько работы уже позади — у скачивания.

У распознавания прогресс считает пайплайн по числу порций, поэтому проверять в
бэкендах нечего: у них нет такого канала и изобретать его им не нужно.

Здесь остаётся загрузка. Проверяется не точность доли, а свойства, без которых
полоса вредна: она не должна врать про завершённость и застревать под конец.
"""

from __future__ import annotations

import pytest

from transcriber.ingest import youtube
from transcriber.report import Report


def test_youtube_reports_a_share_of_the_download():
    seen: list[float] = []
    hook = youtube._reporter(Report(at=seen.append))

    hook({"status": "downloading", "downloaded_bytes": 250, "total_bytes": 1000})
    hook({"status": "downloading", "downloaded_bytes": 750, "total_bytes": 1000})

    assert seen == [0.25, 0.75]


def test_youtube_falls_back_to_an_estimated_size():
    """Точную длину сервер отдаёт не всегда — оценка лучше пустой шкалы."""
    seen: list[float] = []

    youtube._reporter(Report(at=seen.append))(
        {"status": "downloading", "downloaded_bytes": 100, "total_bytes_estimate": 400}
    )

    assert seen == [0.25]


def test_youtube_stays_silent_without_a_size():
    """Шкала, ползущая по догадке, хуже отсутствующей."""
    seen: list[float] = []

    youtube._reporter(Report(at=seen.append))({"status": "downloading", "downloaded_bytes": 100})

    assert seen == []


def test_youtube_closes_the_bar_on_the_last_event():
    """Оценка почти всегда мимо, и без этого полоса застревала бы под конец."""
    seen: list[float] = []

    youtube._reporter(Report(at=seen.append))({"status": "finished"})

    assert seen == [1.0]


def test_the_hook_reaches_yt_dlp():
    """Та же проверка, что и у mlx: доходит ли отчёт до чужой библиотеки.

    Дальше начинается её договор — вызывать переданное по ходу закачки; это
    проверяется только настоящей загрузкой и здесь не покрыто.
    """
    pytest.importorskip("yt_dlp")
    hook = youtube._reporter(Report())

    with youtube._ydl(progress_hooks=[hook]) as ydl:
        assert hook in ydl.params["progress_hooks"]


def test_a_link_from_a_playlist_means_the_video_not_the_playlist():
    """Регрессия: ссылка с `list=` понималась как весь курс.

    yt-dlp по умолчанию видит в такой ссылке плейлист. На лекции №20 из курса
    он отдавал `YoutubeTab` с 47 записями, без длительности и с названием всего
    курса, а скачивание кончалось «HTTP Error 403: Forbidden».

    Тише и хуже другое: ключ кэша строится по этому идентификатору, так что все
    сорок семь лекций делили бы один — вторая открывалась бы звуком первой.
    Сети тест не касается, проверяется само решение.
    """
    assert youtube._ydl().params["noplaylist"] is True


def test_a_caller_may_still_ask_for_the_whole_playlist():
    """Умолчание, а не запрет: сверху опции перебиваются."""
    assert youtube._ydl(noplaylist=False).params["noplaylist"] is False
