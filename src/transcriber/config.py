"""Pipeline configuration.

Everything runs locally. We go out exactly once — for the model weights on the
first launch; after that they live in the HuggingFace cache and no network is
needed.

Any parameter is overridden by an environment variable prefixed `TRANSCRIPT_`
or by a line in `.env`, for example `TRANSCRIPT_ASR_MODEL=turbo`.
"""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass
from pathlib import Path
from typing import Literal

from pydantic import Field
from pydantic_settings import BaseSettings, SettingsConfigDict

from . import paths
from .device import is_apple_silicon, memory_budget

# Short names instead of HuggingFace repositories — for CLI arguments.
WHISPER_MODELS: dict[str, str] = {
    "large-v3": "mlx-community/whisper-large-v3-mlx",
    "turbo": "mlx-community/whisper-large-v3-turbo",
    "medium": "mlx-community/whisper-medium-mlx",
    "small": "mlx-community/whisper-small-mlx",
}


@dataclass(frozen=True)
class LLMChoice:
    """A model the project offers, and what holding it costs.

    Backends name models differently: mlx addresses a HuggingFace repository,
    Ollama its own tag. The short name is one; `ids` expands it per backend.

    `gigabytes` is the published weights, added up from the repositories rather
    than worked out from a parameter count — quantisation and embeddings do not
    follow that arithmetic. It sits here and not in a list of its own because a
    second list keyed by the same names is the kind that drifts.
    """

    ids: Mapping[str, str]
    gigabytes: float


LLM_MODELS: dict[str, LLMChoice] = {
    # MoE: 35B parameters, ~3B active. Loads in 6 s and translates a paragraph in
    # one. The build is multimodal, but only the text path matters to us — mlx-lm
    # brings it up as qwen3_5_moe.
    "qwen3.6-35b": LLMChoice({"mlx": "mlx-community/Qwen3.6-35B-A3B-4bit"}, 19.00),
    # MoE: 30B parameters, ~3B active. Fast and good at Russian, but it does not
    # fit whole into 16 GB of VRAM.
    "qwen3-30b": LLMChoice(
        {
            "mlx": "mlx-community/Qwen3-30B-A3B-Instruct-2507-4bit",
            "ollama": "qwen3:30b-a3b",
        },
        16.00,
    ),
    "qwen3-14b": LLMChoice({"mlx": "mlx-community/Qwen3-14B-4bit", "ollama": "qwen3:14b"}, 7.74),
    "qwen3-8b": LLMChoice({"mlx": "mlx-community/Qwen3-8B-4bit", "ollama": "qwen3:8b"}, 4.29),
    "qwen3-4b": LLMChoice(
        {"mlx": "mlx-community/Qwen3-4B-Instruct-2507-4bit", "ollama": "qwen3:4b"}, 2.11
    ),
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
    choice = LLM_MODELS.get(name)
    return choice.ids.get(backend, name) if choice else name


def default_llm_model() -> str:
    """The largest model this machine can hold.

    Choosing by platform was the old answer, and it assumed the author's
    machine: every Mac was handed a 19 GB model, so one with 16 GB went looking
    for weights that cannot fit there at all.

    Whisper is still in memory alongside it while a recording is being
    recognised — some 3 GB more. That overlap is not accounted for here on
    purpose: it is not a property of the machine but a defect of the order in
    which the two are loaded, and it belongs to whoever releases the recognition
    before the model comes up.
    """
    if not is_apple_silicon():
        # Off Apple Silicon the model lives in VRAM, not in shared memory, and
        # the size of the card is a question we do not ask yet — that platform
        # is not supported for now. This one fits whole into 16 GB of it.
        return "qwen3-8b"

    budget = memory_budget()
    ladder = sorted(LLM_MODELS.items(), key=lambda item: item[1].gigabytes, reverse=True)
    for name, choice in ladder:
        if choice.gigabytes <= budget:
            return name
    # Nothing fits. Naming the smallest is still better than naming one that
    # certainly does not: it may yet run, and the download panel says the size.
    return ladder[-1][0]


def llm_size(name: str) -> float | None:
    """Gigabytes of weights behind a short name; `None` for one we were not told."""
    choice = LLM_MODELS.get(name)
    return choice.gigabytes if choice else None


def memory_warning(settings: Settings) -> str | None:
    """Why the chosen model will not fit this machine, or `None` when it will.

    Only a warning, never a refusal: the choice by memory already lands on a
    model that fits, so the only way to get here is to have asked for a bigger
    one on purpose, and that is a decision to respect rather than overrule.

    Said before the load rather than after, which is the whole point. mlx does
    not help here: its memory limit raises only once RAM *and* swap are both
    exhausted, which is to say after the half hour of thrashing, not instead of
    it. Comparing two numbers we already know costs nothing and comes first.
    """
    # Only a model loading into this process can overflow this machine. With
    # Ollama or a server the weights are somebody else's, on a machine that is
    # not ours to measure.
    if settings.llm_backend != "mlx":
        return None

    choice = LLM_MODELS.get(settings.llm_model)
    if choice is None:
        # A repository given by hand, of a size nobody told us. Silence is
        # honest here; inventing a size to warn about would not be.
        return None

    budget = memory_budget()
    if choice.gigabytes <= budget:
        return None

    return (
        f"{settings.llm_model} wants {choice.gigabytes:.1f} GB and this machine can spare "
        f"{budget:.1f} — it will swap and crawl. The largest that fits here is "
        f"{default_llm_model()}, set as TRANSCRIPT_LLM_MODEL."
    )


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
    # Search width during decoding, and it means two different things. On
    # faster-whisper it is a real beam search. On mlx there is no beam decoder at
    # all — passing one raises — so it becomes `best_of`, and mlx drops `best_of`
    # for every segment decoded at temperature zero.
    #
    # Which is all of them, measured: ten minutes of a battle came back as 118
    # segments, 118 of them at temperature zero. The number only ever acts on the
    # fallback, where a segment came out repetitive or improbable and is decoded
    # again warmer. So on mlx this tunes recovery, not quality — do not reach for
    # it to make the text better.
    #
    # Nor is there a cheap substitute. Sampling at 0.2 and keeping the best of
    # five does raise the model's own confidence (avg_logprob -0.179 against
    # -0.210) but earns it by leaving out what was actually said — "как бы, ну,
    # ты" becomes "ты", "Соля" becomes "соли". Likelihood rewards fluency, and a
    # transcript is owed faithfulness.
    beam_size: int = 5
    word_timestamps: bool = False
    # How much speech goes into recognition at a time. This is what stops memory
    # depending on the length of a recording: eight hours cost what two minutes
    # cost, because two minutes is all that is ever in memory at once.
    #
    # Two minutes is 7.7 MB as float32, and it is also the step of the progress bar
    # and the unit of resuming after an interruption. The size was measured, not
    # picked: on a 37-minute lecture, portions of 2 minutes against 15 took 120
    # seconds against 131, with 14 steps of the bar against 4. Smaller is not
    # dearer — it is noticeably more alive.
    asr_batch_minutes: float = 2.0
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
    # The largest model this machine can hold — see `default_llm_model`. Naming
    # one explicitly wins over the choice: a person may ask for a model bigger
    # than fits, and that is their call to make.
    llm_model: str = Field(default_factory=default_llm_model)
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
