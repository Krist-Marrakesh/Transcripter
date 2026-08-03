"""Мост окна: что именно уходит на диск.

Главное здесь — сохраняется показанное. Человек, глядящий на перевод, ждёт в
файле перевод; молчаливая запись оригинала в этом месте выглядит как исчезнувшая
работа.
"""

from __future__ import annotations

import threading
from pathlib import Path

import pytest

from transcriber import weights
from transcriber.app.bridge import Api, _llm_ready
from transcriber.ingest import Source
from transcriber.models import Segment, Transcript


class FakeServer:
    """Раздача аудио к экспорту отношения не имеет."""

    def url_for(self, audio):
        return f"http://127.0.0.1:0/{audio.name}"


def transcript(language: str, text: str) -> Transcript:
    return Transcript(
        source="лекция.mp4",
        language=language,
        duration=1.0,
        segments=[Segment(start=0.0, end=1.0, text=text)],
        asr_model="stub",
    )


@pytest.fixture
def api(tmp_path):
    instance = Api(FakeServer())
    audio = tmp_path / "лекция.wav"
    audio.write_bytes(b"")
    instance._source = Source(audio=audio, origin="лекция.mp4", title="лекция", duration=1.0)
    instance._transcript = transcript("en", "The encoder produces vectors.")
    return instance


def test_exports_original_by_default(api, tmp_path):
    path = api.export("txt", directory=str(tmp_path))

    assert "encoder produces vectors" in open(path, encoding="utf-8").read()


def test_exports_translation_while_it_is_shown(api, tmp_path):
    """Регрессия: раньше на диск уходил оригинал, хотя на экране был перевод."""
    api._translated = transcript("ru", "Энкодер выдаёт векторы.")
    api._showing = "translation"

    path = api.export("txt", directory=str(tmp_path))

    assert "Энкодер выдаёт векторы" in open(path, encoding="utf-8").read()


def test_translation_does_not_overwrite_original(api, tmp_path):
    """Имена обязаны различаться, иначе перевод затрёт исходный файл."""
    original = api.export("txt", directory=str(tmp_path))

    api._translated = transcript("ru", "Энкодер выдаёт векторы.")
    api._showing = "translation"
    translated = api.export("txt", directory=str(tmp_path))

    assert original != translated
    assert translated.endswith(".ru.txt")


def test_switching_back_shows_original(api):
    api._translated = transcript("ru", "Энкодер выдаёт векторы.")
    api._showing = "translation"

    assert api.show("original") is True
    assert api._showing == "original"


def test_switching_to_missing_translation_is_refused(api):
    assert api.show("translation") is False
    assert api._showing == "original"


def test_export_without_transcript_explains_itself(api, tmp_path):
    api._transcript = None
    with pytest.raises(RuntimeError, match="transcribe a recording first"):
        api.export("txt", directory=str(tmp_path))


def test_unknown_format_refused(api, tmp_path):
    with pytest.raises(ValueError, match="unknown format"):
        api.export("rtf", directory=str(tmp_path))


@pytest.mark.parametrize("fmt", ["pdf", "docx"])
def test_binary_formats_write_real_files(api, tmp_path, fmt):
    """Эти форматы не отдают строку, а пишут файл сами."""
    path = Path(api.export(fmt, directory=str(tmp_path)))

    assert path.suffix == f".{fmt}"
    # Пустой файл означал бы, что запись сорвалась молча.
    assert path.stat().st_size > 1000


class Weights:
    """Ровно то, что читает проверка готовности LLM."""

    llm_backend = "mlx"
    llm_repo = "mlx-community/Qwen3-4B-Instruct-2507-4bit"


def snapshot(hub: Path) -> Path:
    """Каталог модели в кэше HuggingFace, как его раскладывает сама библиотека."""
    folder = "models--" + Weights.llm_repo.replace("/", "--")
    path = hub / folder / "snapshots" / "abc123"
    path.mkdir(parents=True)
    return path


def test_metadata_without_weights_is_not_ready(tmp_path, monkeypatch):
    """Регрессия: `any(glob(...))` по генераторам всегда истинно.

    Оборванная закачка оставляет config и токенизатор без весов, проверка
    считала модель готовой — и первое же распознавание тянуло гигабайты, чтобы
    подписать строчку в истории.
    """
    monkeypatch.setattr(weights, "hub", lambda: tmp_path)
    (snapshot(tmp_path) / "config.json").write_text("{}", encoding="utf-8")

    assert _llm_ready(Weights()) is False


def test_downloaded_weights_are_ready(tmp_path, monkeypatch):
    monkeypatch.setattr(weights, "hub", lambda: tmp_path)
    (snapshot(tmp_path) / "model.safetensors").write_bytes(b"weights")

    assert _llm_ready(Weights()) is True


def test_missing_model_is_not_ready(tmp_path, monkeypatch):
    """Модель не качали вовсе — каталога нет, и это не повод падать."""
    monkeypatch.setattr(weights, "hub", lambda: tmp_path)

    assert _llm_ready(Weights()) is False


class Downloading:
    """Идущая закачка: поток, который сам остановится по флагу."""

    def __init__(self) -> None:
        self.stop = threading.Event()
        self.finished = threading.Event()

    def run(self) -> None:
        self.stop.wait(5)
        self.finished.set()


def test_closing_the_window_stops_the_download():
    """Загрузчик — отдельный процесс и сам по себе закрытие окна переживёт.

    Без остановки он продолжал бы качать в фон, которого уже никто не видит,
    а окно закрывалось бы, пока кто-то ещё держит сеть.
    """
    api = Api(FakeServer())
    job = Downloading()
    api._stop_download = job.stop
    api._download_thread = threading.Thread(target=job.run, daemon=True)
    api._download_thread.start()

    api.shutdown(timeout=5)

    assert job.finished.is_set() is True
    assert api._download_thread.is_alive() is False


def test_shutdown_without_download_is_harmless():
    """Обычное закрытие окна — самый частый случай, и он не должен ничего ждать."""
    Api(FakeServer()).shutdown()
