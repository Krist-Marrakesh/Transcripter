"""Пути приложения.

Главное здесь — умолчание для папки с транскриптами задано один раз. Раньше оно
было записано и в `Settings`, и в состоянии окна: правка в одном месте молча
расходилась со вторым, и файлы уезжали не туда, куда показывал интерфейс.
"""

from __future__ import annotations

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
