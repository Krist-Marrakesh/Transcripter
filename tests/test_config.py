"""Настройки: откуда они берутся.

Главное здесь — где ищется `.env`. Ошибка тихая и полная: приложение не падает
и не жалуется, просто ведёт себя так, будто настроек нет.
"""

from __future__ import annotations

from pathlib import Path

from transcriber import config, paths
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


def test_the_checkout_is_found_without_standing_in_it(tmp_path, monkeypatch):
    """Регрессия: ярлык стартует не в проекте, и `.env` проекта не читался вовсе.

    Относительное имя находит файл, только когда рабочий каталог и есть проект —
    в терминале так, у ярлыка никогда: `.lnk` стартует в домашней папке, Finder в
    корне. Токен в проектном `.env` при запуске иконкой оставался невидимым, и
    окно предлагало настроить настроенное.
    """
    named = config._project_env()
    monkeypatch.chdir(tmp_path)

    assert config._project_env() == named
    assert named.is_absolute()
    assert named.parent == Path(config.__file__).resolve().parents[2]


def test_the_checkout_still_wins_over_the_installed_copy():
    """Порядок тот же, что и у относительного имени: проект перебивает установленное."""
    places = list(Settings.model_config["env_file"])

    assert places.index(config._project_env()) > places.index(paths.config_dir() / ".env")
