"""Настройки: откуда они берутся.

Главное здесь — где ищется `.env`. Ошибка тихая и полная: приложение не падает
и не жалуется, просто ведёт себя так, будто настроек нет.
"""

from __future__ import annotations

from transcriber import paths
from transcriber.config import Settings


def test_env_is_looked_for_in_a_place_that_does_not_move():
    """Регрессия: искали только в текущем каталоге, а он у окна чужой.

    Разработчик запускает из корня проекта и видит свой `.env`; приложение,
    запущенное из Finder, стартует неизвестно откуда — и в релизной сборке
    задать токен для диаризации было негде вовсе.
    """
    places = Settings.model_config["env_file"]

    assert paths.config_dir() / ".env" in places


def test_the_project_copy_wins():
    """Порядок не косметический: pydantic отдаёт победу последнему файлу.

    Значит `.env` в каталоге проекта должен идти вторым — иначе установленная
    копия перебивала бы настройки разработчика.
    """
    places = list(Settings.model_config["env_file"])

    assert places.index(".env") > places.index(paths.config_dir() / ".env")
