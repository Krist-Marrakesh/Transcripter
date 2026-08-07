"""Выбор между двумя способами разметить спикеров.

Проверяется не качество моделей, а решения вокруг них: что предлагается, когда
и обратимо ли. Ошибка тут тихая — человек либо навсегда теряет предложение
перейти на pyannote, либо получает его при каждом запуске.
"""

from __future__ import annotations

import pytest

from transcriber.app import state
from transcriber.app.bridge import Api


class FakeServer:
    def url_for(self, path):
        return ""


@pytest.fixture
def api(tmp_path, monkeypatch):
    """Состояние окна во временном файле вместо настоящего."""
    monkeypatch.setattr(state, "PATH", tmp_path / "app.json")
    return Api(FakeServer())


def test_a_clean_machine_gets_the_offer(api):
    """Токена нет, отказа не было — самое время предложить."""
    speakers = api.setup()["speakers"]

    assert speakers["offer"] is True
    assert speakers["backend"] == "sherpa"


def test_a_token_switches_to_pyannote_and_ends_the_offer(api):
    speakers = api.save_token("hf_секрет")

    assert speakers["backend"] == "pyannote"
    assert speakers["offer"] is False
    assert state.token() == "hf_секрет"


def test_an_empty_token_changes_nothing(api):
    """Пустое поле — это не выбор, а промах мимо кнопки."""
    speakers = api.save_token("   ")

    assert speakers["backend"] == "sherpa"
    assert state.token() is None


def test_a_refusal_is_remembered(api):
    speakers = api.decline_speakers()

    assert speakers["offer"] is False
    assert speakers["backend"] == "sherpa"
    assert api.setup()["speakers"]["offer"] is False


def test_a_token_overrides_an_earlier_refusal(api):
    """Отказ не приговор: передумать можно в любой момент."""
    api.decline_speakers()

    speakers = api.save_token("hf_передумал")

    assert speakers["backend"] == "pyannote"
    assert speakers["offer"] is False


def test_the_offer_returns_when_the_token_is_gone(api):
    """Регрессия по замыслу: флажок «уже показывали» запер бы предложение навсегда.

    Токен могли стереть или он мог протухнуть — и тогда человек снова остаётся
    без pyannote, причём молча.
    """
    api.save_token("hf_секрет")
    state.save(hf_token="", speakers_declined="no")

    assert api.setup()["speakers"]["offer"] is True


def test_the_choice_reaches_the_settings(api):
    """Между кнопкой и пайплайном не должно быть разрыва."""
    from transcriber.app.bridge import load_settings

    api.save_token("hf_секрет")
    settings = load_settings()

    assert settings.diarization_backend == "pyannote"
    assert settings.hf_token == "hf_секрет"
