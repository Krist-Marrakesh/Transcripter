"""Одна копия приложения на машину.

Проверяется не запуск окна, а решение «открывать новое или поднять открытое».
Ошибка здесь тихая в обе стороны: лишняя копия расходится с первой по истории,
а ложное срабатывание замка не даёт открыть приложение вовсе.
"""

from __future__ import annotations

import os

import pytest

from transcriber.app import single


@pytest.fixture
def lock(tmp_path, monkeypatch):
    """Замок во временном каталоге вместо настоящего."""
    path = tmp_path / "app.pid"
    monkeypatch.setattr(single, "lock_file", lambda: path)
    return path


def test_no_lock_means_nothing_is_running(lock):
    assert single.running() is None


def test_broken_lock_is_ignored(lock):
    """Файл могли обрезать при жёстком выключении — это не повод не открыться."""
    lock.write_text("не число", encoding="utf-8")

    assert single.running() is None


def test_dead_process_leaves_no_claim(lock):
    """Копию убили, замок остался: следующий запуск обязан пройти."""
    lock.write_text("99999999\nMon Aug  3 01:00:00 2026", encoding="utf-8")

    assert single.running() is None


def test_own_process_is_not_a_rival(lock):
    """Собственный номер в замке — это мы сами, а не вторая копия."""
    lock.write_text(f"{os.getpid()}\nMon Aug  3 01:00:00 2026", encoding="utf-8")

    assert single.running() is None


def test_foreign_process_under_reused_pid_is_ignored(lock, monkeypatch):
    """Номера переиспользуются: под старым может оказаться чужая программа.

    Отличаем по времени запуска: у занявшего номер оно другое.
    """
    lock.write_text("4242\nMon Aug  3 01:00:00 2026", encoding="utf-8")
    monkeypatch.setattr(single, "_alive", lambda pid: True)
    monkeypatch.setattr(single, "_started", lambda pid: "Mon Aug  3 01:17:00 2026")

    assert single.running() is None


def test_live_copy_is_found(lock, monkeypatch):
    lock.write_text("4242\nMon Aug  3 01:00:00 2026", encoding="utf-8")
    monkeypatch.setattr(single, "_alive", lambda pid: True)
    monkeypatch.setattr(single, "_started", lambda pid: "Mon Aug  3 01:00:00 2026")

    assert single.running() == 4242


def test_lock_without_start_time_is_ignored(lock, monkeypatch):
    """Замок от старой версии: без времени запуска сверить нечего."""
    lock.write_text("4242", encoding="utf-8")
    monkeypatch.setattr(single, "_alive", lambda pid: True)

    assert single.running() is None


def test_claim_is_released_after_the_window_closes(lock):
    with single.claim():
        pid, _, started = lock.read_text(encoding="utf-8").partition("\n")
        assert int(pid) == os.getpid()
        # Время запуска пишется рядом: по нему отличают нас от занявшего номер.
        assert started.strip() != ""

    assert lock.exists() is False


def test_claim_is_released_even_if_the_window_crashes(lock):
    """Иначе упавшая копия навсегда запретила бы открыть приложение."""
    with pytest.raises(RuntimeError), single.claim():
        raise RuntimeError("окно упало")

    assert lock.exists() is False
