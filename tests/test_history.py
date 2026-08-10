"""История записей.

Главное свойство: она переживает чистку кэша. Поэтому хранит собственную копию
транскрипта, а не ссылку на него.
"""

from __future__ import annotations

import pytest

from transcriber.app import history
from transcriber.models import Segment, Transcript


@pytest.fixture(autouse=True)
def isolated(tmp_path, monkeypatch):
    monkeypatch.setattr(history, "ROOT", tmp_path)
    monkeypatch.setattr(history, "INDEX", tmp_path / "history.json")
    monkeypatch.setattr(history, "ENTRIES", tmp_path / "entries")


def make(text: str, language: str = "ru") -> Transcript:
    return Transcript(
        source="лекция.mp4",
        language=language,
        duration=61.0,
        segments=[Segment(start=0.0, end=61.0, text=text)],
        asr_model="stub",
    )


def test_remembers_and_reopens_without_source_file():
    entry = history.remember(
        make("Сегодня разберём теорему Байеса."), origin="лекция.mp4", title="лекция"
    )

    reopened = history.open_entry(entry["key"])

    assert reopened is not None
    assert "теорему Байеса" in reopened.text


def test_topic_from_llm_when_available():
    entry = history.remember(
        make("Много слов про птиц."),
        origin="лекция.mp4",
        title="лекция",
        topic=lambda text: "лекция о миграции птиц",
    )

    assert entry["topic"] == "лекция о миграции птиц"


def test_topic_falls_back_to_first_sentence():
    """Без LLM тема — начало записи: тему занятия называют в первых фразах."""
    entry = history.remember(
        make("Сегодня разберём теорему Байеса и её применение. Дальше всё остальное."),
        origin="лекция.mp4",
        title="лекция",
    )

    assert entry["topic"] == "Сегодня разберём теорему Байеса и её применение."


def test_short_opener_is_extended():
    """«Привет!» темой быть не может — по такому списку ничего не найти."""
    entry = history.remember(
        make("Привет! Сегодня разберём миграцию птиц. И ещё кое-что."),
        origin="лекция.mp4",
        title="лекция",
    )

    assert "миграцию птиц" in entry["topic"]


def test_broken_llm_does_not_lose_the_entry():
    """Тема — украшение списка; её сбой не повод терять запись."""

    def explode(text: str) -> str:
        raise RuntimeError("модель не загрузилась")

    entry = history.remember(
        make("Первая фраза записи."), origin="лекция.mp4", title="лекция", topic=explode
    )

    assert entry["topic"] == "Первая фраза записи."


def test_url_and_file_are_distinguished():
    file_entry = history.remember(make("раз"), origin="/путь/лекция.mp4", title="лекция")
    url_entry = history.remember(make("два"), origin="https://youtube.com/watch?v=x", title="ролик")

    assert file_entry["kind"] == "file"
    assert url_entry["kind"] == "url"


def test_repeat_updates_instead_of_duplicating():
    same = make("Один и тот же текст.")
    history.remember(same, origin="лекция.mp4", title="первый раз")
    history.remember(same, origin="лекция.mp4", title="второй раз")

    entries = history.load()
    assert len(entries) == 1
    assert entries[0]["title"] == "второй раз"


def test_newest_first():
    history.remember(make("старая"), origin="a.mp4", title="старая")
    history.remember(make("новая"), origin="b.mp4", title="новая")

    assert [item["title"] for item in history.load()] == ["новая", "старая"]


def test_forget_removes_entry_and_copy():
    entry = history.remember(make("текст"), origin="a.mp4", title="запись")

    history.forget(entry["key"])

    assert history.load() == []
    assert history.open_entry(entry["key"]) is None


def test_limit_keeps_the_list_usable(monkeypatch):
    monkeypatch.setattr(history, "LIMIT", 3)
    for index in range(5):
        history.remember(make(f"запись номер {index}"), origin="a.mp4", title=str(index))

    assert [item["title"] for item in history.load()] == ["4", "3", "2"]


def test_broken_index_is_not_fatal():
    history.INDEX.parent.mkdir(parents=True, exist_ok=True)
    history.INDEX.write_text("{сломано", encoding="utf-8")

    assert history.load() == []


def test_names_found_later_reach_the_saved_copy():
    """Имена спрашивают отдельной кнопкой, уже после сохранения записи.

    Без дописывания открытая заново запись снова была бы «Спикером 1», и
    нажатие выглядело бы ничего не сделавшим.
    """
    transcript = make("Меня зовут Юрий.")
    entry = history.remember(transcript, origin="баттл.mp4", title="баттл")

    assert history.keep_names(transcript.model_copy(update={"names": {"Спикер 1": "Юрий"}}))

    reopened = history.open_entry(entry["key"])
    assert reopened is not None
    assert reopened.names == {"Спикер 1": "Юрий"}


def test_naming_does_not_add_a_second_entry():
    """Ключ строится по тексту, а имена его не меняют — строка та же самая."""
    transcript = make("Меня зовут Юрий.")
    history.remember(transcript, origin="баттл.mp4", title="баттл")

    history.keep_names(transcript.model_copy(update={"names": {"Спикер 1": "Юрий"}}))

    assert len(history.load()) == 1


def test_names_for_a_recording_that_was_never_saved_change_nothing():
    """Запись открыли из кэша, а не из истории: дописывать нечего и не во что."""
    assert history.keep_names(make("текст").model_copy(update={"names": {"a": "b"}})) is False
