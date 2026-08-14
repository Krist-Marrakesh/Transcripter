"""Поиск библиотек CUDA — единственное, что нужно только Windows.

Питон с 3.8 не ищет DLL расширения по PATH, поэтому CTranslate2 не находит
cuBLAS, который pip только что положил рядом в `site-packages`, и молча уезжает
на процессор. Лекция считается сорок минут вместо четырёх, и в логе ничего.

Проверяется с мака честно: обе функции спрашивают систему в момент вызова, а не
при импорте, поэтому подменённая система идёт по настоящей ветке.
"""

from __future__ import annotations

import os
import sys
import sysconfig

from transcriber import cuda


def packages(monkeypatch, tmp_path):
    """Подменяет `site-packages` пустой папкой."""
    monkeypatch.setattr(sysconfig, "get_paths", lambda: {"purelib": str(tmp_path)})
    return tmp_path


def test_torch_carries_the_runtime_beside_itself(monkeypatch, tmp_path):
    """Колесо torch под Windows несёт cuBLAS и cuDNN в своей же папке."""
    root = packages(monkeypatch, tmp_path)
    (root / "torch" / "lib").mkdir(parents=True)

    assert cuda.library_paths() == (root / "torch" / "lib",)


def test_nvidia_gives_each_library_a_folder(monkeypatch, tmp_path):
    """Свои пакеты NVIDIA раскладывает иначе — по `nvidia/<имя>/bin`."""
    root = packages(monkeypatch, tmp_path)
    (root / "nvidia" / "cublas" / "bin").mkdir(parents=True)
    (root / "nvidia" / "cudnn" / "bin").mkdir(parents=True)

    assert cuda.library_paths() == (
        root / "nvidia" / "cublas" / "bin",
        root / "nvidia" / "cudnn" / "bin",
    )


def test_a_machine_without_a_card_has_nothing_to_name(monkeypatch, tmp_path):
    """Пустой ответ — это ответ: называть нечего, и это не ошибка."""
    packages(monkeypatch, tmp_path)

    assert cuda.library_paths() == ()


def test_nothing_is_named_where_there_is_no_such_loader(monkeypatch, tmp_path):
    """На маке и в Linux искать не за чем: там путь зашит в само колесо."""
    root = packages(monkeypatch, tmp_path)
    (root / "torch" / "lib").mkdir(parents=True)
    monkeypatch.setattr(sys, "platform", "darwin")

    assert cuda.make_findable() == ()


def test_the_folders_also_go_into_path(monkeypatch, tmp_path):
    """Регрессия, найденная на настоящей карте: одного `add_dll_directory` мало.

    Он действует на то, что грузит сам питон, — расширения поднимаются с флагами
    `LOAD_LIBRARY_SEARCH_*`, и те названные папки читают. Но CTranslate2 просит
    `cublas64_12.dll` по имени изнутри своей уже загруженной DLL, а такой вызов
    идёт обычным порядком поиска, и обычный порядок читает `PATH`.

    Расхождение было полным: `ctypes.WinDLL("cublas64_12.dll")` грузился, а
    распознавание падало «Library cublas64_12.dll is not found or cannot be
    loaded» — и уезжало на процессор, ради ухода с которого модуль и написан.
    """
    root = packages(monkeypatch, tmp_path)
    (root / "nvidia" / "cublas" / "bin").mkdir(parents=True)
    monkeypatch.setattr(sys, "platform", "win32")
    monkeypatch.setattr(cuda, "_named", [])
    # `raising=False`, потому что на маке такой функции в `os` нет вовсе: она
    # только для Windows. Без этого тест падал бы ровно там, где файл обещает
    # проверяться честно.
    monkeypatch.setattr(os, "add_dll_directory", lambda path: object(), raising=False)
    monkeypatch.setenv("PATH", "C:\\чужое")

    cuda.make_findable()

    assert os.environ["PATH"].split(os.pathsep)[0] == str(root / "nvidia" / "cublas" / "bin")
    assert "C:\\чужое" in os.environ["PATH"]


def test_a_machine_without_a_card_leaves_path_alone(monkeypatch, tmp_path):
    """Нечего называть — не за чем и трогать окружение."""
    packages(monkeypatch, tmp_path)
    monkeypatch.setattr(sys, "platform", "win32")
    monkeypatch.setattr(cuda, "_named", [])
    monkeypatch.setenv("PATH", "C:\\чужое")

    cuda.make_findable()

    assert os.environ["PATH"] == "C:\\чужое"
