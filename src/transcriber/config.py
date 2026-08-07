"""Pipeline configuration.

Everything runs locally. We go out exactly once — for the model weights on the
first launch; after that they live in the HuggingFace cache and no network is
needed.

Any parameter is overridden by an environment variable prefixed `TRANSCRIPT_`
or by a line in `.env`, for example `TRANSCRIPT_ASR_MODEL=turbo`.
"""

from __future__ import annotations

from pathlib import Path
from typing import Literal

from pydantic import Field
from pydantic_settings import BaseSettings, SettingsConfigDict

from . import paths
from .device import is_apple_silicon

# Short names instead of HuggingFace repositories — for CLI arguments.
WHISPER_MODELS: dict[str, str] = {
    "large-v3": "mlx-community/whisper-large-v3-mlx",
    "turbo": "mlx-community/whisper-large-v3-turbo",
    "medium": "mlx-community/whisper-medium-mlx",
    "small": "mlx-community/whisper-small-mlx",
}

# Backends name models differently: mlx addresses a HuggingFace repository, Ollama
# its own tag. The short name is one; it expands per backend.
LLM_MODELS: dict[str, dict[str, str]] = {
    # MoE: 35B parameters, ~3B active. 19 GB in 4 bits, loads in 6 s and translates
    # a paragraph in one. The build is multimodal, but only the text path matters
    # to us — mlx-lm brings it up as qwen3_5_moe.
    "qwen3.6-35b": {"mlx": "mlx-community/Qwen3.6-35B-A3B-4bit"},
    # MoE: 30B parameters, ~3B active. Fast and good at Russian, but ~17 GB in
    # 4 bits — it does not fit whole into 16 GB of VRAM.
    "qwen3-30b": {
        "mlx": "mlx-community/Qwen3-30B-A3B-Instruct-2507-4bit",
        "ollama": "qwen3:30b-a3b",
    },
    "qwen3-14b": {"mlx": "mlx-community/Qwen3-14B-4bit", "ollama": "qwen3:14b"},
    "qwen3-8b": {"mlx": "mlx-community/Qwen3-8B-4bit", "ollama": "qwen3:8b"},
    "qwen3-4b": {"mlx": "mlx-community/Qwen3-4B-Instruct-2507-4bit", "ollama": "qwen3:4b"},
}


def resolve_model(name: str, registry: dict[str, str]) -> str:
    """Expands a short name into a repo id; a full id passes through unchanged."""
    return registry.get(name, name)


def resolve_llm(name: str, backend: str) -> str:
    """Short LLM name → an identifier for a particular backend.

    An unknown name is returned as is: that is how any repository or tag missing
    from the registry is given. A name the registry knows but which was never
    published for this backend behaves the same — let the backend say it found
    no such model.
    """
    return LLM_MODELS.get(name, {}).get(backend, name)


class Settings(BaseSettings):
    # Two places, and the second is not a nicety. `.env` alone is a relative path,
    # resolved against the working directory — which a developer has pointed at the
    # project, and which an application started from Finder does not control at
    # all. In a released bundle that meant there was nowhere to put a HuggingFace
    # token, and speaker labelling could not be switched on by any means.
    #
    # The project copy comes last on purpose: pydantic reads the files in order and
    # lets the later one win, so a checkout keeps overriding the installed copy.
    model_config = SettingsConfigDict(
        env_prefix="TRANSCRIPT_",
        env_file=(paths.config_dir() / ".env", ".env"),
        env_file_encoding="utf-8",
        extra="ignore",
    )

    # --- paths ---
    cache_dir: Path = Field(default_factory=paths.cache_dir)
    # An absolute path on principle: the application is launched from Finder, where
    # the working directory is the root, and a relative "output" would either fail
    # to be created or end up somewhere unexpected.
    output_dir: Path = Field(default_factory=paths.default_output)

    # --- ASR ---
    # MLX exists only for Apple Silicon, so everywhere else the default backend is
    # CTranslate2 — which is also what gives CUDA on NVIDIA.
    asr_backend: Literal["mlx", "faster"] = Field(
        default_factory=lambda: "mlx" if is_apple_silicon() else "faster"
    )
    asr_model: str = "large-v3"
    # Search width during decoding. On faster-whisper this is a real beam search;
    # mlx has no beam decoder, and the parameter becomes `best_of` — the number of
    # trajectories sampled during the temperature fallback.
    beam_size: int = 5
    word_timestamps: bool = False
    # A decoder hint for the opening seconds. It is the start of a sliding context
    # rather than a glossary: on a long recording it is displaced within a window
    # or two, and measurements showed it hurting more often than helping.
    initial_prompt: str | None = None

    # --- VAD ---
    # Whisper hallucinates on silence and music, so this is on by default.
    vad_enabled: bool = True
    vad_threshold: float = 0.5
    vad_min_speech: float = 0.25
    vad_min_silence: float = 0.5
    # A small margin around speech so attack and decay are not clipped.
    vad_padding: float = 0.15

    # --- diarization ---
    diarization_enabled: bool = False
    # sherpa по умолчанию: он работает на машине, где ничего не настраивали, а
    # pyannote требует принятых условий и токена. Разница в качестве измерена и
    # описана в `diarize.sherpa_backend`.
    diarization_backend: Literal["sherpa", "pyannote"] = "sherpa"
    diarization_model: str = "pyannote/speaker-diarization-community-1"
    # pyannote models are gated: a token and acceptance of the terms are required.
    hf_token: str | None = None
    num_speakers: int | None = None
    min_speakers: int | None = None
    max_speakers: int | None = None

    # --- LLM for translation and summaries ---
    # Outside Apple Silicon it is Ollama: one installer, and it claims the NVIDIA
    # GPU by itself, unlike building llama.cpp with CUDA from source.
    # `openai` is not OpenAI's cloud but its protocol: vLLM, llama.cpp server,
    # LM Studio, TGI and Ollama itself all speak it. One backend covers any machine
    # someone brought a model up on — the application stays local, only the text
    # step travels.
    llm_backend: Literal["mlx", "ollama", "openai"] = Field(
        default_factory=lambda: "mlx" if is_apple_silicon() else "ollama"
    )
    # On Apple Silicon memory is shared, so a 19 GB model runs freely. Elsewhere the
    # LLM lives in VRAM: only a smaller model fits whole into a 16 GB card, and
    # partial offloading to RAM costs more speed than the extra size buys quality.
    llm_model: str = Field(
        default_factory=lambda: "qwen3.6-35b" if is_apple_silicon() else "qwen3-8b"
    )
    llm_max_tokens: int = 4096
    llm_temperature: float = 0.3
    ollama_host: str = "http://localhost:11434"
    # Address of the server holding the model, e.g. http://192.168.1.50:8000/v1.
    # The model name is the server's to define, so the short-name registry does not
    # apply here.
    llm_base_url: str = ""
    llm_api_key: str | None = None
    # A long step on someone else's machine: folding an hour-long lecture takes more
    # than seconds, and a timeout in the middle costs more than extra waiting.
    llm_timeout: float = 300.0
    # The summary language does not follow the recording: an English lecture most
    # often needs retelling in Russian rather than in English.
    summary_language: str = "ru"
    # Chunk size for translation and summaries: local models degrade well before
    # their nominal context window, so we cut them ourselves.
    chunk_chars: int = 6000
    # The one request the application makes for its own sake rather than at a
    # person's: asking GitHub whether a newer version exists. It sends nothing and
    # takes an answer of a few kilobytes, but a program that promises to keep
    # everything on this computer owes an off switch for the one thing that leaves.
    update_check: bool = True

    @property
    def asr_repo(self) -> str:
        return resolve_model(self.asr_model, WHISPER_MODELS)

    @property
    def llm_repo(self) -> str:
        return resolve_llm(self.llm_model, self.llm_backend)


def load_settings(**overrides: object) -> Settings:
    """Builds settings, dropping None so CLI arguments do not overwrite `.env`."""
    return Settings(**{k: v for k, v in overrides.items() if v is not None})
