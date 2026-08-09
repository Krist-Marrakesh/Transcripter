"""Бэкенды распознавания: где они держат веса и как их отпускают.

Держат по-разному, и это не мелочь реализации. CTranslate2 хранит модель в самом
объекте, а mlx-whisper — в атрибуте своего класса, который переживает наш объект
целиком. Поэтому «отпустить» умеет только бэкенд: снаружи непонятно, что именно
нужно обнулить.
"""

from __future__ import annotations

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
