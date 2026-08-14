"""Бэкенды распознавания: где они держат веса и как их отпускают.

Держат по-разному, и это не мелочь реализации. CTranslate2 хранит модель в самом
объекте, а mlx-whisper — в атрибуте своего класса, который переживает наш объект
целиком. Поэтому «отпустить» умеет только бэкенд: снаружи непонятно, что именно
нужно обнулить.
"""

from __future__ import annotations

from types import SimpleNamespace

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


def _refusing_card(monkeypatch, message: str) -> list[str]:
    """Подменяет `WhisperModel` так, что карта модель не принимает.

    Импорт внутри `_build` идёт на каждый вызов, поэтому подмена атрибута модуля
    доезжает — в отличие от промаха мимо модуля, на котором когда-то сломался
    патч tqdm.
    """
    import faster_whisper

    tried: list[str] = []

    def build(repo, *, device, compute_type):
        tried.append(device)
        if device != "cpu":
            raise RuntimeError(message)
        return object()

    monkeypatch.setattr(faster_whisper, "WhisperModel", build)
    return tried


def test_faster_falls_back_to_the_processor(monkeypatch):
    """Карта, не принявшая модель, стоит замедления, а не отказа от работы.

    Настоящая причина, а не выдуманная: колесо torch канала cu130 несёт ядра
    начиная с `sm_75`, поэтому на Pascal и старше `torch.cuda.is_available()`
    отвечает «да», а первая же операция падает вот этим сообщением. Пока модель
    поднималась без отката, у такого человека вместо часа расшифровки на
    процессоре была «Ошибка» и выход.
    """
    pytest.importorskip("faster_whisper")
    from transcriber.asr.faster_backend import FasterWhisperBackend

    tried = _refusing_card(monkeypatch, "no kernel image is available for execution on the device")
    backend = FasterWhisperBackend("large-v3", device="cuda")

    backend.load()

    assert tried == ["cuda", "cpu"]
    assert backend.device == "cpu · int8"
    assert "no kernel image" in backend.device_warning


def test_a_backend_on_the_card_promises_nothing_until_it_is_loaded(monkeypatch):
    """`device` до загрузки — намерение, после неё — факт, и разница видна."""
    pytest.importorskip("faster_whisper")
    from transcriber.asr.faster_backend import FasterWhisperBackend

    _refusing_card(monkeypatch, "CUDA failed with error out of memory")
    backend = FasterWhisperBackend("large-v3", device="cuda")

    assert backend.device == "cuda · float16"
    assert backend.device_warning == ""

    backend.load()

    assert backend.device == "cpu · int8"


def test_a_refusing_processor_is_a_real_failure(monkeypatch):
    """Отступать с процессора некуда, и притворяться, что всё хорошо, нельзя."""
    pytest.importorskip("faster_whisper")
    import faster_whisper

    from transcriber.asr.faster_backend import FasterWhisperBackend

    def die(repo, *, device, compute_type):
        raise RuntimeError("модель не читается")

    monkeypatch.setattr(faster_whisper, "WhisperModel", die)

    with pytest.raises(RuntimeError, match="не читается"):
        FasterWhisperBackend("large-v3", device="cpu").load()


def _card_that_computes_badly(monkeypatch, message: str) -> list[str]:
    """Карта, на которой модель собирается, но не считает.

    Именно так ведёт себя CTranslate2: библиотеки CUDA он подтягивает лениво, и
    до первого счёта отказ ничем себя не выдаёт.
    """
    import faster_whisper

    computed: list[str] = []

    class Model:
        def __init__(self, device: str) -> None:
            self.device = device

        def transcribe(self, audio, **kwargs):
            computed.append(self.device)
            if self.device != "cpu":
                raise RuntimeError(message)
            return iter(()), SimpleNamespace(language="ru")

    monkeypatch.setattr(
        faster_whisper, "WhisperModel", lambda repo, *, device, compute_type: Model(device)
    )
    return computed


def test_a_card_that_only_fails_when_it_computes(monkeypatch):
    """Настоящий случай, из-за которого откат переехал из `load` в счёт.

    Замерено на RTX 2070 без cuBLAS 12: `WhisperModel(device="cuda")` собирается
    молча и успешно, `get_cuda_device_count()` отвечает единицей, а первый же
    счёт падает `Library cublas64_12.dll is not found`. Пока откат стоял в
    `load`, он не срабатывал вовсе — ловить там было нечего.
    """
    pytest.importorskip("faster_whisper")
    from transcriber.asr.faster_backend import FasterWhisperBackend

    computed = _card_that_computes_badly(monkeypatch, "Library cublas64_12.dll is not found")
    backend = FasterWhisperBackend("large-v3", device="cuda")

    backend.load()
    assert backend.device == "cuda · float16", "загрузка и не обязана ничего заметить"

    result = backend.transcribe(np.zeros(16_000, dtype=np.float32))

    assert computed == ["cuda", "cpu"]
    assert backend.device == "cpu · int8"
    assert "cublas64_12" in backend.device_warning
    assert result.language == "ru"


def test_a_card_that_worked_once_is_not_abandoned_later(monkeypatch):
    """Досчитавшая карта доказана, и следующая беда — уже не про устройство.

    Иначе сломанная запись посреди лекции стоила бы пересборки модели на
    процессоре и повторного счёта того же куска — лечения не того.
    """
    pytest.importorskip("faster_whisper")
    import faster_whisper

    from transcriber.asr.faster_backend import FasterWhisperBackend

    calls: list[str] = []

    class Model:
        def transcribe(self, audio, **kwargs):
            calls.append("счёт")
            if len(calls) > 1:
                raise RuntimeError("запись не читается")
            return iter(()), SimpleNamespace(language="ru")

    monkeypatch.setattr(faster_whisper, "WhisperModel", lambda repo, **kwargs: Model())
    backend = FasterWhisperBackend("large-v3", device="cuda")
    backend.transcribe(np.zeros(16_000, dtype=np.float32))

    with pytest.raises(RuntimeError, match="не читается"):
        backend.transcribe(np.zeros(16_000, dtype=np.float32))

    assert backend.device == "cuda · float16"
    assert backend.device_warning == ""


def test_a_card_that_refused_is_not_asked_again(monkeypatch):
    """Освобождение весов не возвращает выбор на карту.

    Иначе каждая порция часовой лекции начиналась бы с попытки, о которой уже
    известно, чем она кончится, и с новой строки в журнале.
    """
    pytest.importorskip("faster_whisper")
    from transcriber.asr.faster_backend import FasterWhisperBackend

    tried = _refusing_card(monkeypatch, "no kernel image")
    backend = FasterWhisperBackend("large-v3", device="cuda")
    backend.load()
    backend.release()

    backend.load()

    assert tried == ["cuda", "cpu", "cpu"]


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
