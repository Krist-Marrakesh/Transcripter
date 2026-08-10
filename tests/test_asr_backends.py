"""Бэкенды распознавания: где они держат веса и как их отпускают.

Держат по-разному, и это не мелочь реализации. CTranslate2 хранит модель в самом
объекте, а mlx-whisper — в атрибуте своего класса, который переживает наш объект
целиком. Поэтому «отпустить» умеет только бэкенд: снаружи непонятно, что именно
нужно обнулить.
"""

from __future__ import annotations

import numpy as np
import pytest

from transcriber.asr.mlx_backend import MLXWhisperBackend


def test_mlx_release_empties_the_library_cache():
    """Регрессия по замыслу: `mlx_whisper.transcribe` — функция, а не модуль.

    Через атрибут пакета до `ModelHolder` не добраться: там лежит функция того же
    имени, и обращение к ней падает. Ровно так когда-то промахнулся мимо модуля
    патч tqdm — а промахнувшееся освобождение выглядит точно как сработавшее.
    """
    pytest.importorskip("mlx_whisper")
    from mlx_whisper.transcribe import ModelHolder

    ModelHolder.model = object()
    ModelHolder.model_path = "какой-то путь"

    MLXWhisperBackend("любой-репозиторий").release()

    assert ModelHolder.model is None
    assert ModelHolder.model_path is None


def test_faster_release_drops_its_own_model():
    """У CTranslate2 модель лежит в объекте, и отпустить её — это отпустить ссылку."""
    pytest.importorskip("faster_whisper")
    from transcriber.asr.faster_backend import FasterWhisperBackend

    backend = FasterWhisperBackend("large-v3")
    backend._model = object()

    backend.release()

    assert backend._model is None


def test_mlx_never_asks_for_a_beam(monkeypatch):
    """Регрессия по замыслу: на `beam_size` mlx падает NotImplementedError.

    Настройка называется шириной поиска, но у этого бэкенда поиска нет. Передать
    её как есть — уронить распознавание целиком, а не получить худшее качество.
    """
    pytest.importorskip("mlx_whisper")
    import mlx_whisper

    seen: dict = {}

    def fake(samples, **options):
        seen.update(options)
        return {"segments": [], "language": "ru"}

    monkeypatch.setattr(mlx_whisper, "transcribe", fake)
    MLXWhisperBackend("репозиторий").transcribe(np.zeros(16, dtype=np.float32), beam_size=5)

    assert "beam_size" not in seen
    assert seen["best_of"] == 5


def test_mlx_asks_for_nothing_when_the_width_is_one(monkeypatch):
    """Ширина в один — это и есть жадное декодирование, просить нечего."""
    pytest.importorskip("mlx_whisper")
    import mlx_whisper

    seen: dict = {}
    monkeypatch.setattr(
        mlx_whisper, "transcribe", lambda s, **o: (seen.update(o), {"segments": []})[1]
    )
    MLXWhisperBackend("репозиторий").transcribe(np.zeros(16, dtype=np.float32), beam_size=1)

    assert "best_of" not in seen
