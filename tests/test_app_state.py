"""Папка для готовых файлов.

Приоритет: выбранное мышью → настройки → умолчание. Отдельная проверка на
относительный путь: из Finder приложение стартует в корне, и «output» из `.env`
там означает `/output`.
"""

from __future__ import annotations

from pathlib import Path

import pytest

from transcriber.app import state


@pytest.fixture(autouse=True)
def isolated(tmp_path, monkeypatch):
    """Состояние пишется в свой файл, домашний не трогаем."""
    monkeypatch.setattr(state, "PATH", tmp_path / "app.json")
    monkeypatch.setattr(state, "DEFAULT_OUTPUT", tmp_path / "Транскрипты")


def test_absolute_setting_is_respected(tmp_path):
    configured = tmp_path / "Лекции"
    assert state.output_dir(configured) == configured


def test_relative_setting_falls_back_to_default():
    """Регрессия: `TRANSCRIPT_OUTPUT_DIR=output` уводил экспорт в /output."""
    assert state.output_dir(Path("output")) == state.DEFAULT_OUTPUT


def test_chosen_folder_wins_over_setting(tmp_path):
    picked = tmp_path / "Выбранная"
    state.save(output_dir=str(picked))

    assert state.output_dir(tmp_path / "Из настроек") == picked


def test_choice_survives_restart(tmp_path):
    state.save(output_dir=str(tmp_path / "Первая"))
    state.save(something_else="значение")

    # Дозапись не должна терять уже сохранённое.
    assert state.load()["output_dir"] == str(tmp_path / "Первая")
    assert state.load()["something_else"] == "значение"


def test_folder_is_created_once_and_not_recreated(tmp_path):
    """Повторный вход в приложение не должен ничего трогать в готовой папке."""
    folder = tmp_path / "Транскрипты"
    assert state.ensure(folder) is True

    (folder / "лекция.txt").write_text("уже лежит", encoding="utf-8")
    assert state.ensure(folder) is True

    assert (folder / "лекция.txt").read_text(encoding="utf-8") == "уже лежит"


def test_refused_folder_is_reported_not_raised(tmp_path):
    """Окно обязано открыться, даже если папку создать не дали."""
    blocked = tmp_path / "занято"
    blocked.write_text("это файл, а не папка", encoding="utf-8")

    assert state.ensure(blocked / "Транскрипты") is False


def test_broken_file_is_not_fatal(tmp_path):
    state.PATH.parent.mkdir(parents=True, exist_ok=True)
    state.PATH.write_text("{это не json", encoding="utf-8")

    assert state.load() == {}
    assert state.output_dir(Path("output")) == state.DEFAULT_OUTPUT
