"""Pipeline orchestration.

Here and only here the steps are joined into a chain and the cache is wired in.
The steps themselves stay pure and know nothing about caching.

Separate caching is a matter of principle. ASR and diarization cost minutes,
translation and summaries cost seconds. Every step is keyed by its own
parameters, so enabling diarization does not force recognition to be recomputed,
and changing the summary prompt does not drag the whole recording through
transcription again.
"""

from __future__ import annotations

from collections.abc import Callable

from . import audio, ingest, vad
from .asr import Task, create_backend
from .cache import ArtifactCache, fingerprint, stable_key
from .config import Settings
from .diarize import DiarizationError, assign_speakers
from .diarize import diarize as run_diarization
from .ingest import Source
from .models import Diarization, Summary, Transcript, Translation
from .nlp import create_llm
from .nlp import summarize as run_summarize
from .nlp import translate as run_translate

Notify = Callable[[str], None]


class Pipeline:
    """Wires ingest → VAD → ASR → diarization → NLP on top of a shared cache."""

    def __init__(self, settings: Settings, notify: Notify | None = None) -> None:
        self.settings = settings
        self.cache = ArtifactCache(settings.cache_dir)
        self._notify = notify or (lambda _: None)
        self._llm = None

    @property
    def llm(self):
        """The LLM loads lazily: transcription without translation never touches it."""
        if self._llm is None:
            self._notify(f"loading LLM {self.settings.llm_repo}")
            self._llm = create_llm(
                self.settings.llm_backend,
                self.settings.llm_repo,
                host=self.settings.ollama_host,
                base_url=self.settings.llm_base_url,
                api_key=self.settings.llm_api_key,
                timeout=self.settings.llm_timeout,
            )
        return self._llm

    # --- steps ---

    def prepare(self, target: str) -> Source:
        self._notify(f"preparing audio: {target}")
        return ingest.prepare(target, self.cache, self._notify)

    def transcribe(
        self,
        source: Source,
        *,
        language: str | None = None,
        task: Task = "transcribe",
        force: bool = False,
    ) -> Transcript:
        settings = self.settings
        key = stable_key(
            fingerprint(source.audio),
            backend=settings.asr_backend,
            model=settings.asr_repo,
            task=task,
            language=language,
            beam_size=settings.beam_size,
            word_timestamps=settings.word_timestamps,
            initial_prompt=settings.initial_prompt,
            vad=(
                settings.vad_enabled,
                settings.vad_threshold,
                settings.vad_min_speech,
                settings.vad_min_silence,
                settings.vad_padding,
            ),
        )

        if not force and (cached := self.cache.load("transcript", key, Transcript)) is not None:
            self._notify("transcript taken from cache")
            return cached

        samples = audio.load(source.audio)
        timeline = vad.Timeline()

        if settings.vad_enabled:
            regions = vad.detect(
                samples,
                threshold=settings.vad_threshold,
                min_speech=settings.vad_min_speech,
                min_silence=settings.vad_min_silence,
                padding=settings.vad_padding,
            )
            if regions:
                timeline = vad.Timeline(regions)
                total = audio.duration(samples)
                samples = vad.compress(samples, regions)
                ratio = vad.speech_ratio(regions, total)
                noun = "segment" if len(regions) == 1 else "segments"
                self._notify(
                    f"VAD: {len(regions)} speech {noun}, "
                    f"{timeline.compressed_duration / 60:.1f} min left of "
                    f"{total / 60:.1f} ({ratio:.0%})"
                )

        backend = create_backend(settings.asr_backend, settings.asr_repo)
        # The device is named before the start: falling back to the CPU does not
        # break the result but stretches the step several times over, and finding
        # that out afterwards is too late.
        self._notify(f"transcribing with {settings.asr_repo} · {backend.device}")
        result = backend.transcribe(
            samples,
            language=language,
            task=task,
            beam_size=settings.beam_size,
            word_timestamps=settings.word_timestamps,
            initial_prompt=settings.initial_prompt,
        )

        transcript = Transcript(
            source=source.origin,
            language=result.language,
            duration=source.duration,
            # Timestamps come back from the compressed VAD axis onto the original one.
            segments=[timeline.remap(segment) for segment in result.segments],
            asr_model=settings.asr_repo,
        )
        self.cache.store("transcript", key, transcript)
        return transcript

    def add_speakers(
        self, source: Source, transcript: Transcript, *, force: bool = False
    ) -> Transcript:
        """Labels the transcript by speaker."""
        settings = self.settings
        key = stable_key(
            fingerprint(source.audio),
            model=settings.diarization_model,
            num_speakers=settings.num_speakers,
            min_speakers=settings.min_speakers,
            max_speakers=settings.max_speakers,
        )

        cached = None if force else self.cache.load("diarization", key, Diarization)
        if cached is not None:
            self._notify("diarization taken from cache")
            turns = cached.turns
        else:
            self._notify(f"labelling speakers with {settings.diarization_model}")
            turns = run_diarization(
                source.audio,
                model=settings.diarization_model,
                token=settings.hf_token,
                num_speakers=settings.num_speakers,
                min_speakers=settings.min_speakers,
                max_speakers=settings.max_speakers,
            )
            self.cache.store(
                "diarization", key, Diarization(turns=turns, model=settings.diarization_model)
            )

        segments = assign_speakers(transcript.segments, turns)
        found = len({t.speaker for t in turns})
        self._notify(f"{found} speaker{'s' if found != 1 else ''}")
        return transcript.model_copy(update={"segments": segments})

    def translate(
        self, transcript: Transcript, *, target_language: str, force: bool = False
    ) -> Translation:
        settings = self.settings
        key = stable_key(
            _content_key(transcript),
            target=target_language,
            model=settings.llm_repo,
            chunk_chars=settings.chunk_chars,
        )

        if not force and (cached := self.cache.load("translation", key, Translation)) is not None:
            self._notify("translation taken from cache")
            return cached

        translation = run_translate(
            transcript,
            self.llm,
            target_language=target_language,
            chunk_chars=settings.chunk_chars,
            max_tokens=settings.llm_max_tokens,
            temperature=settings.llm_temperature,
            progress=lambda done, total: self._notify(f"translating: batch {done}/{total}"),
        )
        self.cache.store("translation", key, translation)
        return translation

    def summarize(
        self, transcript: Transcript, *, language: str = "ru", force: bool = False
    ) -> Summary:
        settings = self.settings
        key = stable_key(
            _content_key(transcript),
            language=language,
            model=settings.llm_repo,
            chunk_chars=settings.chunk_chars,
        )

        if not force and (cached := self.cache.load("summary", key, Summary)) is not None:
            self._notify("summary taken from cache")
            return cached

        summary = run_summarize(
            transcript,
            self.llm,
            language=language,
            chunk_chars=settings.chunk_chars,
            max_tokens=settings.llm_max_tokens,
            temperature=settings.llm_temperature,
            progress=lambda done, total: self._notify(f"summarizing: step {done}/{total}"),
        )
        self.cache.store("summary", key, summary)
        return summary

    # --- end-to-end run ---

    def run(
        self,
        target: str,
        *,
        language: str | None = None,
        task: Task = "transcribe",
        diarize: bool | None = None,
        force: bool = False,
    ) -> tuple[Source, Transcript]:
        """Ingest → VAD → ASR → diarization (optional).

        Diarization is enrichment, not a mandatory step. Its failure leaves the
        transcript without speaker labels but does not devalue recognition that
        cost minutes: the result is returned and the reason goes to `notify`.
        Ingest and ASR errors propagate — without them there is nothing to return.
        """
        source = self.prepare(target)
        transcript = self.transcribe(source, language=language, task=task, force=force)

        want_speakers = self.settings.diarization_enabled if diarize is None else diarize
        if want_speakers:
            try:
                transcript = self.add_speakers(source, transcript, force=force)
            except DiarizationError as exc:
                self._notify(f"speakers not labelled: {exc}")

        return source, transcript


def _content_key(transcript: Transcript) -> str:
    """A key over transcript content: editing segments invalidates the NLP steps."""
    return stable_key(transcript.text, language=transcript.language, model=transcript.asr_model)
