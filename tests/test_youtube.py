"""Куки для yt-dlp.

Без них YouTube всё чаще отвечает «Sign in to confirm you're not a bot», и
ссылка не открывается вовсе — ни скачать, ни спросить длительность. Проверяется
без сети: важно не то, пустит ли нас сайт, а доезжает ли до yt-dlp то, что
человек написал в настройках. Разъедется — настройка будет стоять, а ошибка
останется прежней, и винить будут ссылку.
"""

from __future__ import annotations

from pathlib import Path

import pytest

from transcriber.config import Settings
from transcriber.ingest import youtube


def test_nothing_asked_is_nothing_told():
    """Обычный случай: большинство ссылок открываются без всяких кук."""
    assert youtube.Cookies().options() == {}


def test_a_browser_is_named_the_way_yt_dlp_expects_it():
    """Четвёрка, а не строка: браузер, профиль, связка ключей, контейнер."""
    asked = youtube.Cookies(from_browser="firefox").options()

    assert asked == {"cookiesfrombrowser": ("firefox", None, None, None)}


def test_a_file_is_named_as_a_path(tmp_path):
    """Запасной путь для Chrome и родни: выгруженный расширением `cookies.txt`."""
    jar = tmp_path / "cookies.txt"
    jar.write_text("# Netscape HTTP Cookie File\n", encoding="utf-8")

    assert youtube.Cookies(file=jar).options() == {"cookiefile": str(jar)}


def test_both_named_are_both_told():
    """yt-dlp сольёт их в один набор — выбирать за человека не надо."""
    asked = youtube.Cookies(from_browser="firefox", file=Path(__file__)).options()

    assert set(asked) == {"cookiesfrombrowser", "cookiefile"}


def test_a_file_that_is_not_there_is_named_outright(tmp_path):
    """Иначе опечатку в пути покажут как отказ скачивания, и виновата будет ссылка."""
    with pytest.raises(youtube.DownloadError, match="cookie file not found"):
        youtube.Cookies(file=tmp_path / "нет.txt").options()


def test_what_was_asked_reaches_yt_dlp():
    """Проверка стыка: настройка бесполезна, если не доходит до самой библиотеки."""
    pytest.importorskip("yt_dlp")

    with youtube._ydl(youtube.Cookies(from_browser="firefox")) as ydl:
        assert ydl.params["cookiesfrombrowser"] == ("firefox", None, None, None)


def test_the_probe_hands_the_cookies_over(monkeypatch):
    """Ссылку сначала спрашивают о длительности — и отказ приходит уже там."""
    seen: list[youtube.Cookies | None] = []
    monkeypatch.setattr(youtube, "_ydl", lambda cookies=None, **options: _Fake(seen, cookies))

    youtube.probe("https://example.com/video", cookies=youtube.Cookies(from_browser="firefox"))

    assert seen == [youtube.Cookies(from_browser="firefox")]


def test_the_download_hands_them_over_too(monkeypatch, tmp_path):
    """Второй вызов к сайту — и он тоже без кук не пройдёт."""
    seen: list[youtube.Cookies | None] = []
    monkeypatch.setattr(youtube, "_ydl", lambda cookies=None, **options: _Fake(seen, cookies))

    youtube.download_audio(
        "https://example.com/video", tmp_path, cookies=youtube.Cookies(from_browser="firefox")
    )

    assert seen == [youtube.Cookies(from_browser="firefox")]


def test_the_settings_carry_the_choice():
    """Настройка читается переменной среды, как и всё остальное."""
    settings = Settings(_env_file=None, cookies_from_browser="firefox")

    assert settings.cookies_from_browser == "firefox"


def test_an_unknown_browser_is_refused():
    """Опечатка в имени — ошибка настройки, а не пустой набор кук молча."""
    with pytest.raises(ValueError):
        Settings(_env_file=None, cookies_from_browser="файрфокс")


class _Fake:
    """Подставной YoutubeDL: запоминает, с какими куками его собрали."""

    def __init__(self, seen: list, cookies) -> None:
        seen.append(cookies)

    def __enter__(self):
        return self

    def __exit__(self, *_) -> None:
        return None

    def extract_info(self, url: str, download: bool = False) -> dict:
        return {"title": "запись", "duration": 1.0, "id": "1", "extractor_key": "Test"}

    def prepare_filename(self, info: dict) -> str:
        return "1.m4a"
