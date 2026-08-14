"""ASR на faster-whisper (CTranslate2).

Основной бэкенд везде, кроме Apple Silicon: на NVIDIA считает через CUDA, на
остальных машинах — на процессоре. На macOS остаётся запасным, потому что Metal
в CTranslate2 не поддерживается и выигрыш там даёт только mlx.
"""

from __future__ import annotations

from collections.abc import Callable
from typing import TYPE_CHECKING, TypeVar

import numpy as np

from ..device import Device, detect
from ..models import Segment, Word
from .base import ASRResult, Task

if TYPE_CHECKING:
    from faster_whisper import WhisperModel

T = TypeVar("T")


def normalize_model_name(name: str) -> str:
    """Приводит имя модели к тому, что понимает faster-whisper.

    mlx-репозитории ему не подходят, но пользователю удобно указывать одно и то
    же имя для обоих бэкендов — снимаем mlx-обёртку с известных названий.
    """
    stripped = name.removeprefix("mlx-community/whisper-").removesuffix("-mlx")
    return stripped if "/" not in stripped else name


def hub_repo(name: str) -> str:
    """Репозиторий, который faster-whisper на самом деле скачает под это имя.

    Спрашивается у самой библиотеки, а не выводится по образцу. Образец кажется
    очевидным — `Systran/faster-whisper-<имя>` — и на трёх моделях из четырёх
    сходится, но `turbo` уезжает к `mobiuslabsgmbh/faster-whisper-large-v3-turbo`,
    и написанное по образцу правило показывало бы человеку кнопку «скачать» на
    модель, которая уже лежит, а качало бы её во второй раз под чужим именем.

    Ошибись здесь — и промолчат оба конца: панель считает веса отсутствующими, а
    распознавание всё равно работает, потому что библиотека качает своё сама.
    Ровно эта тишина и стоила Windows целой панели.

    Импорт ленивый и не бесплатный — 2.2 с вместе с CTranslate2, — поэтому
    спрашивается он на вызове из окна, когда интерфейс уже нарисован.
    """
    try:
        from faster_whisper.utils import _MODELS
    except ImportError:
        # Библиотеки нет или таблица переехала. Имя как есть — faster-whisper
        # поймёт его сам; панель при этом скажет «не скачано», и это честнее
        # молчания.
        return name
    return _MODELS.get(name, name)


def select_device(device: Device | None = None, compute_type: str | None = None) -> tuple[str, str]:
    """Подбирает устройство и точность для CTranslate2.

    Metal он не умеет, поэтому на Apple honest-ответ — процессор. На CUDA
    точность по умолчанию `float16`: она вдвое быстрее `int8` на тензорных ядрах,
    а на процессоре наоборот выигрывает `int8`.
    """
    resolved = device or detect()
    if resolved == "mps":
        resolved = "cpu"
    if compute_type is None:
        compute_type = "float16" if resolved == "cuda" else "int8"
    return resolved, compute_type


class FasterWhisperBackend:
    """Обёртка над `faster_whisper.WhisperModel`."""

    def __init__(
        self, repo: str, *, device: Device | None = None, compute_type: str | None = None
    ) -> None:
        self.repo = normalize_model_name(repo)
        self.target, self.compute_type = select_device(device, compute_type)
        self.device = f"{self.target} · {self.compute_type}"
        self.device_warning = ""
        self._model: WhisperModel | None = None
        self._proven = False

    def load(self) -> None:
        """Поднимает модель — насколько устройство вообще можно проверить заранее.

        Заранее его проверить нельзя до конца, и это свойство CTranslate2, а не
        недосмотр. Замерено на RTX 2070 без cuBLAS 12: `WhisperModel(device=
        "cuda")` собирается молча и успешно, `get_cuda_device_count()` отвечает
        единицей, а первый же счёт падает `Library cublas64_12.dll is not found`.
        Библиотеки подтягиваются лениво, поэтому загрузка доказывает только то,
        что модель прочиталась.

        Отсюда правило: **устройство доказано тем, что на нём посчитали**, и
        настоящий откат живёт в `transcribe`. Здесь ловится лишь то, что видно
        сразу, — нечитаемые веса, неподдерживаемая точность.
        """
        self._or_on_the_processor(self._ensure_model)

    def _or_on_the_processor(self, work: Callable[[], T]) -> T:
        """Делает работу, уводя её на процессор, если карта отказала.

        Откат живёт здесь, а не у вызывающего: чем всё кончилось и с какой
        точностью теперь считают — знает только бэкенд, и `select_device` лежит
        рядом с ним.

        Отступать есть от чего, и причина глубже случайного сбоя. Карту выбирает
        `device.detect()`, а он спрашивает torch; считает же CTranslate2 — со
        своей сборкой CUDA и своим набором ядер, так что работающий torch про
        него не доказывает ничего. Замерено: колесо torch канала cu130 несёт
        ядра начиная с `sm_75` (под Pascal и старше их нет вовсе) и cuBLAS
        версии 13, тогда как CTranslate2 4.8 просит двенадцатую. Туда же —
        видеопамять, занятая чужим процессом.

        Любая из этих причин стоит распознавания на процессоре, а не отказа от
        распознавания: лекция сосчитается медленнее, но сосчитается.

        Отступаем один раз и только пока карта ничего не досчитала. После
        первого удавшегося куска она доказана, и следующая беда — уже не про
        устройство: пересобирать ради неё модель на процессоре и повторять час
        лекции значило бы лечить не то.
        """
        try:
            return work()
        except Exception as exc:
            if self.target == "cpu" or self._proven:
                raise
            self.device_warning = f"{self.target} is unavailable, transcribing on the CPU: {exc}"
            self.target, self.compute_type = select_device("cpu")
            self.device = f"{self.target} · {self.compute_type}"
            # Модель собрана под прежнее устройство и на новом бесполезна.
            self._model = None
            return work()

    def _ensure_model(self) -> WhisperModel:
        if self._model is None:
            self._model = self._build()
        return self._model

    def _build(self) -> WhisperModel:
        from ..cuda import make_findable

        # CTranslate2 линкуется с cuBLAS и cuDNN, и на Windows не найдёт их сам:
        # питон не ищет DLL расширений по PATH. Без этой строки бэкенд молча
        # уезжает на процессор.
        make_findable()

        from faster_whisper import WhisperModel

        return WhisperModel(self.repo, device=self.target, compute_type=self.compute_type)

    def release(self) -> None:
        """Отпускает модель. CTranslate2 держит её здесь, и больше нигде.

        Выбранное устройство при этом остаётся выбранным: если карта уже
        отказала, следующая загрузка идёт сразу на процессор, а не пробует её
        снова на каждой порции.
        """
        self._model = None

    def transcribe(
        self,
        samples: np.ndarray,
        *,
        language: str | None = None,
        task: Task = "transcribe",
        beam_size: int = 5,
        word_timestamps: bool = False,
        initial_prompt: str | None = None,
    ) -> ASRResult:
        audio = np.ascontiguousarray(samples, dtype=np.float32)

        def recognise() -> ASRResult:
            segments, info = self._ensure_model().transcribe(
                audio,
                language=language,
                task=task,
                beam_size=beam_size,
                word_timestamps=word_timestamps,
                initial_prompt=initial_prompt,
                # Выключен по той же причине, что и у mlx, и там же лежит замер:
                # перенос контекста заводит зацикливание. Ради ровной пунктуации
                # его включать не стоит — она от этого почти не меняется.
                condition_on_previous_text=False,
                # Свой VAD выключен: тишину уже вырезал общий шаг пайплайна.
                vad_filter=False,
            )
            # segments — ленивый генератор, расчёт идёт по мере обхода, поэтому
            # список собирается здесь, внутри попытки. Снаружи неё отказ карты
            # случился бы уже за пределами отката.
            return ASRResult(
                segments=[_build_segment(item) for item in segments],
                language=info.language or language or "unknown",
            )

        result = self._or_on_the_processor(recognise)
        # Досчитанный кусок — единственное доказательство, что устройство живо.
        self._proven = True
        return result


def _build_segment(raw) -> Segment:
    return Segment(
        start=float(raw.start),
        end=float(raw.end),
        text=raw.text.strip(),
        words=[
            Word(
                start=float(word.start),
                end=float(word.end),
                text=word.word.strip(),
                probability=getattr(word, "probability", None),
            )
            for word in raw.words or ()
        ],
    )
