"""Откуда берутся ffmpeg и ffprobe.

Без них не работает ничего: на них стоит и чтение файла, и загрузка с YouTube.
Раньше их отсутствие означало «поставь brew install ffmpeg» — а окно, запущенное
из Finder, не наследует PATH оболочки, и то же сообщение видел человек, у
которого ffmpeg стоял годами.
"""

from __future__ import annotations

from pathlib import Path

import pytest

from transcriber.ingest import media


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
