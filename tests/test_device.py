"""Выбор платформы, бэкендов и устройства.

Смысл этих тестов — проверить поведение на чужой платформе, не имея её под
рукой: и определение Apple Silicon, и выбор бэкендов подменяются, поэтому
Windows-ветка проверяется с macOS.
"""

from __future__ import annotations

import pytest

from transcriber import config
from transcriber.asr.faster_backend import (
    FasterWhisperBackend,
    normalize_model_name,
    select_device,
)
from transcriber.asr.mlx_backend import MLXWhisperBackend
from transcriber.config import Settings, resolve_llm

BACKEND_VARS = ("TRANSCRIPT_ASR_BACKEND", "TRANSCRIPT_LLM_BACKEND")


@pytest.fixture
def clean_env(monkeypatch):
    """Убирает влияние .env и переменных окружения на значения по умолчанию."""
    for name in BACKEND_VARS:
        monkeypatch.delenv(name, raising=False)


@pytest.fixture
def on_platform(monkeypatch, clean_env):
    """Притворяется Apple Silicon или чем-то ещё."""

    def apply(apple: bool) -> Settings:
        monkeypatch.setattr(config, "is_apple_silicon", lambda: apple)
        return Settings(_env_file=None)

    return apply


def test_apple_silicon_gets_mlx(on_platform):
    settings = on_platform(True)
    assert settings.asr_backend == "mlx"
    assert settings.llm_backend == "mlx"


def test_other_platforms_get_ctranslate2_and_ollama(on_platform):
    """На Windows и Linux mlx недоступен физически — колёс под них не существует."""
    settings = on_platform(False)
    assert settings.asr_backend == "faster"
    assert settings.llm_backend == "ollama"


@pytest.mark.parametrize(
    ("device", "expected"),
    [
        # На тензорных ядрах float16 вдвое быстрее int8.
        ("cuda", ("cuda", "float16")),
        ("cpu", ("cpu", "int8")),
        # CTranslate2 не поддерживает Metal — честный ответ здесь процессор.
        ("mps", ("cpu", "int8")),
    ],
)
def test_select_device_picks_precision(device, expected):
    assert select_device(device) == expected


def test_explicit_compute_type_wins():
    assert select_device("cuda", "int8_float16") == ("cuda", "int8_float16")


def test_backend_reports_where_it_computes():
    """Молчаливый откат на процессор обязан быть виден до начала распознавания."""
    # Модель грузится лениво, поэтому конструктор безопасен без faster-whisper.
    assert FasterWhisperBackend("large-v3", device="cuda").device == "cuda · float16"
    assert FasterWhisperBackend("large-v3", device="cpu").device == "cpu · int8"
    assert MLXWhisperBackend("любой-репозиторий").device == "metal"


def test_llm_identifier_differs_per_backend():
    """У mlx это репозиторий HuggingFace, у Ollama — её собственный тег."""
    assert resolve_llm("qwen3-8b", "mlx") == "mlx-community/Qwen3-8B-4bit"
    assert resolve_llm("qwen3-8b", "ollama") == "qwen3:8b"


def test_unknown_llm_name_passes_through():
    """Так задаётся модель, которой нет в реестре."""
    assert resolve_llm("hf-user/своя-модель", "mlx") == "hf-user/своя-модель"


def test_llm_name_without_backend_build_passes_through():
    """Имя есть, сборки под этот бэкенд нет — пусть бэкенд и скажет, что не нашёл.

    KeyError на разборе конфига был бы хуже: он ничего не объясняет.
    """
    assert resolve_llm("qwen3.6-35b", "ollama") == "qwen3.6-35b"


def test_every_llm_alias_is_available_on_mlx():
    """mlx — бэкенд по умолчанию на Apple Silicon, без него имя бесполезно."""
    for alias, identifiers in config.LLM_MODELS.items():
        assert "mlx" in identifiers, alias
        assert set(identifiers) <= {"mlx", "ollama"}, alias


def test_mlx_repo_normalized_for_ctranslate2():
    """Короткое имя одно на оба бэкенда, а разворачивают они его по-своему."""
    assert normalize_model_name("mlx-community/whisper-large-v3-mlx") == "large-v3"
    assert normalize_model_name("large-v3") == "large-v3"
    # Чужой репозиторий трогать нельзя — вдруг это готовая CTranslate2-сборка.
    assert normalize_model_name("Systran/faster-whisper-large-v3") == (
        "Systran/faster-whisper-large-v3"
    )
