"""Где лежат веса моделей.

Весами приложения владеет приложение: папка внутри бандла, чтобы перетаскивание
в корзину уносило и двадцать с лишним гигабайт вместе с ним. Раньше они ложились
в общий кэш HuggingFace и в кэш приложения — оба переживают любую программу,
когда-либо в них писавшую, и оба остаются занимать место после удаления.

Проверяется без сети и без самих весов: важно не как HuggingFace качает файлы, а
сходятся ли на одной папке все, кто про эти файлы знает. Разъедутся — закачка
уйдёт в одно место, а проверка «скачано ли» будет читать другое, и оба промолчат.
"""

from __future__ import annotations

from pathlib import Path

import pytest

from transcriber import paths, weights
from transcriber.diarize import sherpa_backend

REPO = "mlx-community/whisper-large-v3-mlx"


@pytest.fixture
def owned(tmp_path, monkeypatch):
    """Папка приложения — та, что лаунчер передаёт переменной."""
    monkeypatch.setenv(paths.MODELS, str(tmp_path))
    return tmp_path


def test_a_source_checkout_has_no_folder_of_its_own(monkeypatch):
    """Отсутствие бандла — это ответ, а не потерянное значение.

    В исходниках веса общие с машиной: тот же кэш читает любой другой инструмент
    на ней, и забирать двадцать гигабайт под проект значило бы отнять их у всех.
    """
    monkeypatch.delenv(paths.MODELS, raising=False)

    assert paths.models_dir() is None


def test_every_model_lands_in_the_one_folder(owned):
    """Регрессия: HuggingFace и sherpa качают порознь, удаляться должны вместе."""
    assert weights.hub() == owned / "huggingface" / "hub"
    assert sherpa_backend.home() == owned / "diarization"
    assert owned in weights.cache_dir(REPO).parents


def test_the_folder_is_ours_before_the_library_is_asked(owned, monkeypatch):
    """Ответ не зависит от того, что успело импортировать `huggingface_hub`.

    Константа `HF_HUB_CACHE` замерзает при первом импорте библиотеки, поэтому
    спрашивать её про нашу папку нельзя: порядок импортов решал бы, куда попадут
    веса. Подменяем её заведомо чужим путём — ответ обязан не измениться.
    """
    monkeypatch.setattr(weights, "_previous_hub", lambda: owned.parent / "чужой")

    assert weights.hub() == owned / "huggingface" / "hub"


def test_the_separate_downloader_is_told_where_to_put_them(owned, monkeypatch):
    """Закачка идёт отдельным процессом и наш пакет не импортирует вовсе."""
    seen: dict[str, str] = {}

    def remember(_command, **kwargs):
        seen.update(kwargs["env"])
        raise RuntimeError("до самой закачки дело не доходит")

    monkeypatch.setattr(weights.subprocess, "Popen", remember)
    with pytest.raises(RuntimeError):
        weights._spawn(REPO)

    assert seen["HF_HUB_CACHE"] == str(owned / "huggingface" / "hub")


def test_the_move_looks_where_the_weights_actually_are(owned, monkeypatch):
    """Регрессия: переезд искал прежние веса уже в новом месте.

    Прежнее место спрашивалось у библиотеки, а к моменту вопроса оно уже наше —
    мы сами его туда и поставили строкой выше. Переезд не находил ничего и молча
    качал заново двадцать два гигабайта, лежащие на диске. Поймано прогоном на
    настоящей машине, а не на заглушке: на заглушке всё сходилось.
    """
    monkeypatch.delenv("XDG_CACHE_HOME", raising=False)

    assert owned not in weights._previous_hub().parents
    assert weights._previous_hub() == Path.home() / ".cache" / "huggingface" / "hub"


def test_weights_downloaded_before_are_taken_over(owned, monkeypatch, tmp_path):
    """Уже скачанную модель не качают второй раз ради переезда."""
    theirs = tmp_path / "чужой-кэш"
    monkeypatch.setattr(weights, "_previous_hub", lambda: theirs)
    lying = theirs / weights._folder(REPO) / "snapshots" / "rev1"
    lying.mkdir(parents=True)
    (lying / "weights.npz").write_bytes(b"x" * 100)

    assert weights.adopt_model(REPO) is True
    assert weights.is_ready(REPO) is True
    assert not (theirs / weights._folder(REPO)).exists()


def test_what_is_already_here_is_left_alone(owned, monkeypatch, tmp_path):
    """Своё не затирается чужим: иначе переезд стёр бы более свежую загрузку."""
    theirs = tmp_path / "чужой-кэш"
    monkeypatch.setattr(weights, "_previous_hub", lambda: theirs)
    (theirs / weights._folder(REPO)).mkdir(parents=True)
    weights.cache_dir(REPO).mkdir(parents=True)

    assert weights.adopt_model(REPO) is False


def test_nothing_is_moved_without_a_folder_to_move_into(monkeypatch, tmp_path):
    """В исходниках переезжать некуда, и трогать общий кэш не за чем."""
    monkeypatch.delenv(paths.MODELS, raising=False)

    assert weights.adopt_model(REPO) is False


def test_a_folder_on_another_disk_stays_where_it_is(owned, monkeypatch, tmp_path):
    """Переезд — только переименование.

    Копирование двадцати двух гигабайт до открытия окна выглядит как зависший
    запуск, а модель, оставшаяся на месте, всего лишь скачается заново — панелью,
    которая уже показывает прогресс. Другой том подделан отказом: настоящий
    второй том на машине с тестами не создать.
    """
    previous = tmp_path / "снаружи"
    previous.mkdir()

    def refuse(_self, _target):
        raise OSError("Invalid cross-device link")

    monkeypatch.setattr(Path, "rename", refuse)

    assert weights.adopt(previous, owned / "сюда") is False
    assert previous.exists()
