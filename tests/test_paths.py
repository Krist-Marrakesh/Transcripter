"""Пути приложения.

Главное здесь — умолчание для папки с транскриптами задано один раз. Раньше оно
было записано и в `Settings`, и в состоянии окна: правка в одном месте молча
расходилась со вторым, и файлы уезжали не туда, куда показывал интерфейс.
"""

from __future__ import annotations

import sys
from pathlib import Path

from transcriber import paths
from transcriber.app import history, state
from transcriber.config import Settings


def test_default_output_lives_in_documents():
    assert paths.default_output() == Path.home() / "Documents" / "Транскрипты"


def test_settings_and_window_agree_on_default():
    """Регрессия: умолчание было записано дважды и могло разъехаться.

    Берём именно фабрику поля, а не `Settings()`: та подхватит `.env` разработчика
    и проверит его настройку вместо умолчания, ради которого тест и написан.
    """
    factory = Settings.model_fields["output_dir"].default_factory

    assert factory() == state.DEFAULT_OUTPUT == paths.default_output()


def test_history_is_not_stored_inside_the_cache():
    """История переживает чистку кэша — значит, лежит не в нём."""
    assert paths.cache_dir() not in history.ROOT.parents
    assert history.ROOT == paths.data_dir()


def test_each_directory_is_distinct():
    """Кэш, история и выбор мышью не должны сливаться в одну папку."""
    assert len({paths.cache_dir(), paths.data_dir(), paths.config_dir()}) == 3


def test_windows_gathers_everything_under_local_app_data(monkeypatch, tmp_path):
    """Там нет деления на config/data/cache — разделяем сами, папками.

    Систему подменяем: функции спрашивают `sys.platform` в момент вызова, и это
    единственная часть Windows-кода, которую можно проверить с мака по-честному.
    """
    monkeypatch.setattr(sys, "platform", "win32")
    monkeypatch.setenv("LOCALAPPDATA", str(tmp_path))

    assert paths.cache_dir() == tmp_path / "transcript" / "cache"
    assert paths.data_dir() == tmp_path / "transcript" / "data"
    assert paths.config_dir() == tmp_path / "transcript" / "config"
    assert len({paths.cache_dir(), paths.data_dir(), paths.config_dir()}) == 3


def test_windows_survives_a_missing_variable(monkeypatch):
    """Переменной может не быть: без запасного пути приложение не откроется."""
    monkeypatch.setattr(sys, "platform", "win32")
    monkeypatch.delenv("LOCALAPPDATA", raising=False)

    assert paths.cache_dir() == Path.home() / "AppData" / "Local" / "transcript" / "cache"
