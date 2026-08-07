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

from collections.abc import Callable, Sequence

from . import audio, ingest, vad
from .asr import Task, create_backend
from .cache import ArtifactCache, fingerprint, stable_key
from .config import Settings
from .diarize import DiarizationError, assign_speakers
from .diarize import diarize as run_diarization
from .ingest import Source
from .models import Diarization, Portion, Segment, Summary, Transcript, Translation
from .nlp import create_llm
from .nlp import summarize as run_summarize
from .nlp import translate as run_translate

Notify = Callable[[str], None]

Advance = Callable[[float], None]
"""How far the current long step has got, from 0 to 1.

Kept apart from `notify` rather than folded into it as another line of text: one
is a record of what happened, the other is a single number that replaces itself.
Recognition is the only step here measured in minutes, and it is the one place
where "it is running" is not an answer to "how much longer".
"""


class Pipeline:
    """Wires ingest → VAD → ASR → diarization → NLP on top of a shared cache."""

    def __init__(
        self,
        settings: Settings,
        notify: Notify | None = None,
        advance: Advance | None = None,
    ) -> None:
        self.settings = settings
        self.cache = ArtifactCache(settings.cache_dir)
        self._notify = notify or (lambda _: None)
        self._advance = advance or (lambda _: None)
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
        return ingest.prepare(target, self.cache, self._notify, self._advance)

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

        total = audio.length(source.audio)
        regions: list[vad.SpeechRegion] = []

        if settings.vad_enabled:
            regions = vad.scan(
                source.audio,
                threshold=settings.vad_threshold,
                min_speech=settings.vad_min_speech,
                min_silence=settings.vad_min_silence,
                padding=settings.vad_padding,
            )
            if regions:
                speech = sum(region.duration for region in regions)
                ratio = vad.speech_ratio(regions, total)
                noun = "segment" if len(regions) == 1 else "segments"
                self._notify(
                    f"VAD: {len(regions)} speech {noun}, "
                    f"{speech / 60:.1f} min left of {total / 60:.1f} ({ratio:.0%})"
                )

        # The recording goes through in portions, and this is the whole reason
        # memory stops depending on its length: eight hours cost what a quarter of
        # an hour costs, because that is all that is ever in memory at once.
        portions = vad.batches(regions, settings.asr_batch_minutes * 60, total)

        backend = create_backend(settings.asr_backend, settings.asr_repo)
        # The device is named before the start: falling back to the CPU does not
        # break the result but stretches the step several times over, and finding
        # that out afterwards is too late.
        self._notify(f"transcribing with {settings.asr_repo} · {backend.device}")
        # The bar belongs to whichever long step is running now, and downloading
        # left it full. A full bar through the minutes of recognition would read
        # as "finished, why is it still going".
        self._advance(0.0)

        segments: list[Segment] = []
        spoken = language
        for number, portion in enumerate(portions):
            piece = self._recognise(source, portion, key, number, backend, spoken, task, settings)
            segments += piece.segments
            # What the first portion heard is passed to the rest instead of being
            # decided again. Detection is per call, so without this a recording
            # could change language halfway on nothing but a quiet passage.
            spoken = spoken or piece.language
            # Progress counts portions rather than the innards of a backend: with
            # several of them, a bar that filled and reset each time would say less
            # than one that crosses the whole recording once.
            self._advance((number + 1) / len(portions))

        # The step is over, whatever the last window reported: Whisper can stop
        # short of the end, and a bar frozen at 98% reads as a hang.
        self._advance(1.0)

        transcript = Transcript(
            source=source.origin,
            language=spoken or language or "unknown",
            duration=source.duration,
            segments=segments,
            asr_model=settings.asr_repo,
        )
        self.cache.store("transcript", key, transcript)
        return transcript

    def _recognise(
        self,
        source: Source,
        portion: Sequence[vad.SpeechRegion],
        key: str,
        number: int,
        backend,
        language: str | None,
        task: Task,
        settings: Settings,
    ) -> Portion:
        """Recognises one portion, remembering the result on its own.

        Cached separately from the transcript so that an interruption three hours
        in does not throw away those three hours. The portion's key is the
        transcript's plus its number: change anything the transcript is keyed by
        and every portion is recomputed, as it should be.
        """
        piece = stable_key(key, portion=number)
        cached = self.cache.load("portion", piece, Portion)
        if cached is not None:
            return cached

        # Only this portion's speech is read; the pauses between never reach memory.
        samples = audio.read_spans(source.audio, [(r.start, r.end) for r in portion])
        result = backend.transcribe(
            samples,
            language=language,
            task=task,
            beam_size=settings.beam_size,
            word_timestamps=settings.word_timestamps,
            initial_prompt=settings.initial_prompt,
        )

        # A timeline built from this portion's regions maps its compressed axis
        # straight onto absolute time — there is no offset to add afterwards.
        timeline = vad.Timeline(portion)
        done = Portion(
            segments=[timeline.remap(segment) for segment in result.segments],
            language=result.language,
        )
        self.cache.store("portion", piece, done)
        return done

    def add_speakers(
        self, source: Source, transcript: Transcript, *, force: bool = False
    ) -> Transcript:
        """Labels the transcript by speaker."""
        settings = self.settings
        # The backend belongs in the key. Without it, switching to pyannote after
        # obtaining a token would hand back the labelling sherpa made — the very
        # thing the switch was meant to replace, and silently.
        key = stable_key(
            fingerprint(source.audio),
            backend=settings.diarization_backend,
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
            named = (
                settings.diarization_model
                if settings.diarization_backend == "pyannote"
                else "sherpa-onnx"
            )
            self._notify(f"labelling speakers with {named}")
            turns = run_diarization(
                source.audio,
                backend=settings.diarization_backend,
                model=settings.diarization_model,
                token=settings.hf_token,
                num_speakers=settings.num_speakers,
                min_speakers=settings.min_speakers,
                max_speakers=settings.max_speakers,
                notify=self._notify,
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
