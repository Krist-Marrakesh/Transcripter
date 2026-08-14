"""Откуда берутся ffmpeg и ffprobe.

Без них не работает ничего: на них стоит и чтение файла, и загрузка с YouTube.
Раньше их отсутствие означало «поставь brew install ffmpeg» — а окно, запущенное
из Finder, не наследует PATH оболочки, и то же сообщение видел человек, у
которого ffmpeg стоял годами.
"""

from __future__ import annotations

from pathlib import Path

import pytest

from transcriber.ingest import media, youtube


def test_the_system_binary_comes_first(monkeypatch):
    """Системный почти всегда свежее нашего и настроен под эту машину."""
    monkeypatch.setattr(media.shutil, "which", lambda name: f"/usr/local/bin/{name}")

    assert media._require("ffmpeg") == "/usr/local/bin/ffmpeg"


def test_without_a_system_binary_ours_is_used(monkeypatch, tmp_path):
    """Ровно случай чистой машины: Homebrew нет, а работать надо."""
    binary = tmp_path / "ffmpeg"
    binary.write_text("", encoding="utf-8")
    monkeypatch.setattr(media.shutil, "which", lambda name: None)
    monkeypatch.setattr(media, "_bundled", lambda tool: str(binary))

    assert media._require("ffmpeg") == str(binary)


def test_with_nothing_anywhere_it_says_so(monkeypatch):
    monkeypatch.setattr(media.shutil, "which", lambda name: None)
    monkeypatch.setattr(media, "_bundled", lambda tool: None)

    with pytest.raises(media.FFmpegError, match="ffmpeg-binaries"):
        media._require("ffmpeg")


def test_availability_is_asked_the_way_the_work_asks(monkeypatch, tmp_path):
    """Регрессия: `transcript info` писал «ffmpeg нет» на исправной установке.

    Он спрашивал `shutil.which`, то есть только PATH, — а запасная статическая
    сборка приезжает пакетом и лежит вне его. Это не редкий случай: README
    обещает, что ffmpeg ставить отдельно не нужно, и каждый, кто поверил
    обещанию, видел в диагностике красное «нет» на работающей установке.
    """
    binary = tmp_path / "ffmpeg"
    binary.write_text("", encoding="utf-8")
    monkeypatch.setattr(media.shutil, "which", lambda name: None)
    monkeypatch.setattr(media, "_bundled", lambda tool: str(binary))

    assert media.find_tool("ffmpeg") == str(binary)


def test_nothing_anywhere_is_answered_rather_than_raised(monkeypatch):
    """Отчёт о состоянии не имеет права падать: это отчёт, а не работа."""
    monkeypatch.setattr(media.shutil, "which", lambda name: None)
    monkeypatch.setattr(media, "_bundled", lambda tool: None)

    assert media.find_tool("ffmpeg") is None


def test_the_bundled_pair_is_really_there():
    """Пакет объявлен зависимостью — значит оба бинаря обязаны быть на месте."""
    for tool in ("ffmpeg", "ffprobe"):
        found = media._bundled(tool)
        assert found is not None, tool
        assert Path(found).exists()


def test_yt_dlp_is_told_where_to_look_only_when_needed(monkeypatch, tmp_path):
    """Свой каталог подсовываем, лишь когда системного ffmpeg нет.

    yt-dlp смотрит только в PATH и запасного не держит: без подсказки он молча
    отказался бы склеивать форматы на машине без Homebrew.
    """
    binary = tmp_path / "bin" / "ffmpeg"
    binary.parent.mkdir()
    binary.write_text("", encoding="utf-8")
    monkeypatch.setattr(media.shutil, "which", lambda name: None)
    monkeypatch.setattr(media, "_bundled", lambda tool: str(binary))

    assert media.ffmpeg_folder() == binary.parent

    monkeypatch.setattr(media.shutil, "which", lambda name: "/opt/homebrew/bin/ffmpeg")
    assert media.ffmpeg_folder() is None


# --- чем ключуется скачанная запись ---


def test_identity_ignores_where_playback_starts():
    """Регрессия: `?t=2940s` уводил в промах мимо уже скачанных 118 МБ.

    Метка времени говорит браузеру, с какой секунды играть, и к самой записи
    отношения не имеет. Ключ по адресу целиком заводил второй экземпляр того же
    баттла — а на дачном интернете это не мелочь.
    """
    info = {"extractor_key": "Youtube", "id": "f_KUzNwlMpo", "title": "батл"}

    assert youtube._identity(info) == "Youtube:f_KUzNwlMpo"


def test_identity_names_the_site_too():
    """Номера, которые раздают два разных сайта, совпасть не обязаны."""
    assert youtube._identity({"extractor_key": "vk", "id": "-77521_162"}) == "vk:-77521_162"


def test_identity_falls_back_to_the_address():
    """Сайт не назвал записи — тогда ключом остаётся адрес, как было раньше."""
    info = youtube.RemoteInfo(url="https://example.com/видео", title="x", duration=1.0)

    assert info.identity == "https://example.com/видео"


def test_identity_prefers_what_the_site_calls_it():
    info = youtube.RemoteInfo(
        url="https://www.youtube.com/watch?v=abc&t=99s",
        title="x",
        duration=1.0,
        media_id="Youtube:abc",
    )

    assert info.identity == "Youtube:abc"
