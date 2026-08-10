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

from collections.abc import Sequence
from dataclasses import dataclass
from pathlib import Path

from . import audio, ingest, vad
from .asr import ASRBackend, Task, create_backend
from .cache import ArtifactCache, fingerprint, stable_key
from .config import Settings, llm_size, memory_warning
from .diarize import DiarizationError, assign_speakers
from .diarize import diarize as run_diarization
from .ingest import Source
from .models import (
    Diarization,
    Portion,
    Segment,
    SpeakerNames,
    SpokenFormulas,
    Summary,
    Transcript,
    Translation,
)
from .nlp import LLM, create_llm
from .nlp import name_speakers as run_name_speakers
from .nlp import read_formulas as run_read_formulas
from .nlp import summarize as run_summarize
from .nlp import translate as run_translate
from .report import Report


class LLMSlot:
    """The one language model of a session, built no earlier than first needed.

    A model is assembled here and nowhere else, so everyone asking gets the same
    one. That matters beyond tidiness: a second owner means a second read of the
    weights off the disk, and for the length of that read two copies of nineteen
    gigabytes are alive at once — which on a machine with sixteen is the whole
    difference between working and being killed.

    An empty slot is also an answer. Work that is worth doing with a model
    already up, and not worth raising one for, asks `loaded` and takes the
    honest no.
    """

    def __init__(self, settings: Settings) -> None:
        self._settings = settings
        self._llm: LLM | None = None

    @property
    def loaded(self) -> bool:
        """Whether the model can answer without reading its weights first."""
        return self._llm is not None and self._llm.loaded

    def get(self) -> LLM:
        if self._llm is None:
            self._llm = create_llm(
                self._settings.llm_backend,
                self._settings.llm_repo,
                host=self._settings.ollama_host,
                base_url=self._settings.llm_base_url,
                api_key=self._settings.llm_api_key,
                timeout=self._settings.llm_timeout,
            )
        return self._llm


@dataclass(frozen=True)
class _Recognition:
    """What stays the same while a recording is recognised portion by portion.

    Only two things vary between portions — which regions and which number — so
    only those are arguments. The rest travels here instead of through a
    signature that would grow with every parameter recognition ever gains.
    """

    audio: Path
    key: str
    backend: ASRBackend
    task: Task
    settings: Settings


class Pipeline:
    """Wires ingest → VAD → ASR → diarization → NLP on top of a shared cache."""

    def __init__(
        self, settings: Settings, report: Report | None = None, llm_slot: LLMSlot | None = None
    ) -> None:
        self.settings = settings
        self.cache = ArtifactCache(settings.cache_dir)
        # One object instead of three channels threaded through every signature.
        # A step that needs to speak, show progress or check for a stop takes this
        # and nothing else.
        self.report = report or Report()
        # A caller living longer than one run passes its own slot, and the model
        # then outlives the run instead of being read off the disk again for the
        # next one. A single run — the CLI — keeps its model to itself.
        self._slot = llm_slot or LLMSlot(settings)

    @property
    def llm(self) -> LLM:
        """The LLM loads lazily: transcription without translation never touches it."""
        if not self._slot.loaded:
            self._say_what_is_loading()
        return self._slot.get()

    def _say_what_is_loading(self) -> None:
        """Names the model and its size before the wait rather than after it.

        Reading the weights is seconds with nothing visible happening — measured
        at seven here with the file already in the page cache, and longer
        without one. A window silent that long is indistinguishable from one
        that has stopped answering, and the size is what tells a person which of
        the two they are looking at.
        """
        name = self.settings.llm_model
        size = llm_size(name)
        self.report.say(f"loading LLM {name}" + (f" · {size:.1f} GB" if size is not None else ""))
        # Said here and not at startup: this is the moment the memory is about
        # to be wanted, and the only moment when the warning still changes what
        # a person can do about it.
        if warning := memory_warning(self.settings):
            self.report.say(warning)

    # --- steps ---

    def prepare(self, target: str) -> Source:
        self.report.say(f"preparing audio: {target}")
        return ingest.prepare(target, self.cache, self.report)

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
            self.report.say("transcript taken from cache")
            return cached

        self.report.stop_if_asked()
        total = audio.length(source.audio)
        regions: list[vad.SpeechRegion] = []

        if settings.vad_enabled:
            # Named before it starts, not after: on a long recording the scan is
            # minutes of its own, and a window that says nothing for that long is
            # indistinguishable from one that has stopped answering.
            self.report.say(f"looking for speech in {total / 60:.0f} min of audio")
            regions = vad.scan(
                source.audio,
                self.report,
                threshold=settings.vad_threshold,
                min_speech=settings.vad_min_speech,
                min_silence=settings.vad_min_silence,
                padding=settings.vad_padding,
            )
            if regions:
                speech = sum(region.duration for region in regions)
                ratio = vad.speech_ratio(regions, total)
                noun = "segment" if len(regions) == 1 else "segments"
                self.report.say(
                    f"VAD: {len(regions)} speech {noun}, "
                    f"{speech / 60:.1f} min left of {total / 60:.1f} ({ratio:.0%})"
                )

        # The recording goes through in portions, and this is the whole reason
        # memory stops depending on its length: eight hours cost what a quarter of
        # an hour costs, because that is all that is ever in memory at once.
        portions = vad.batches(regions, settings.asr_batch_minutes * 60, total)

        # Everything that does not change from portion to portion, gathered once.
        # Passing them one by one made a method of eight parameters, of which six
        # were the same on every call — the caller's locals wearing a signature.
        job = _Recognition(
            audio=source.audio,
            key=key,
            backend=create_backend(settings.asr_backend, settings.asr_repo),
            task=task,
            settings=settings,
        )
        backend = job.backend
        # The device is named before the start: falling back to the CPU does not
        # break the result but stretches the step several times over, and finding
        # that out afterwards is too late.
        self.report.say(f"transcribing with {settings.asr_repo} · {backend.device}")
        # The bar belongs to whichever long step is running now, and downloading
        # left it full. A full bar through the minutes of recognition would read
        # as "finished, why is it still going".
        self.report.at(0.0)

        segments: list[Segment] = []
        spoken = language
        try:
            for number, portion in enumerate(portions):
                self.report.stop_if_asked()
                piece = self._recognise(job, portion, number, spoken)
                segments += piece.segments
                # What the first portion heard is passed to the rest instead of
                # being decided again. Detection is per call, so without this a
                # recording could change language halfway on nothing but a quiet
                # passage.
                spoken = spoken or piece.language
                # Progress counts portions rather than the innards of a backend:
                # with several of them, a bar that filled and reset each time
                # would say less than one that crosses the whole recording once.
                self.report.at((number + 1) / len(portions))
        finally:
            # Recognition is over, however it ended, and nothing after it needs
            # the recogniser: diarization brings its own model and translation
            # brings the LLM. Measured on a recording recognised and then
            # translated — holding on to whisper puts the peak at 21.38 GB
            # against 18.50, and the 2.88 GB between them is exactly what
            # whisper keeps. Reading the weights again next time costs 0.9 s,
            # which is the cheaper side of that trade by a wide margin.
            backend.release()

        # The step is over, whatever the last window reported: Whisper can stop
        # short of the end, and a bar frozen at 98% reads as a hang.
        self.report.at(1.0)

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
        job: _Recognition,
        portion: Sequence[vad.SpeechRegion],
        number: int,
        language: str | None,
    ) -> Portion:
        """Recognises one portion, remembering the result on its own.

        Cached separately from the transcript so that an interruption three hours
        in does not throw away those three hours. The portion's key is the
        transcript's plus its number: change anything the transcript is keyed by
        and every portion is recomputed, as it should be.
        """
        piece = stable_key(job.key, portion=number)
        cached = self.cache.load("portion", piece, Portion)
        if cached is not None:
            return cached

        # Only this portion's speech is read; the pauses between never reach memory.
        samples = audio.read_spans(job.audio, [(r.start, r.end) for r in portion])
        result = job.backend.transcribe(
            samples,
            language=language,
            task=job.task,
            beam_size=job.settings.beam_size,
            word_timestamps=job.settings.word_timestamps,
            initial_prompt=job.settings.initial_prompt,
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
            self.report.say("diarization taken from cache")
            turns = cached.turns
        else:
            named = (
                settings.diarization_model
                if settings.diarization_backend == "pyannote"
                else "sherpa-onnx"
            )
            self.report.say(f"labelling speakers with {named}")
            # Recognition left the bar full, and this step has its own minutes to
            # run. A bar that stays at the end through them reads as "finished".
            self.report.at(0.0)
            turns = run_diarization(
                source.audio,
                backend=settings.diarization_backend,
                model=settings.diarization_model,
                token=settings.hf_token,
                num_speakers=settings.num_speakers,
                min_speakers=settings.min_speakers,
                max_speakers=settings.max_speakers,
                report=self.report,
            )
            self.cache.store(
                "diarization", key, Diarization(turns=turns, model=settings.diarization_model)
            )

        segments = assign_speakers(transcript.segments, turns)
        found = len({t.speaker for t in turns})
        self.report.say(f"{found} speaker{'s' if found != 1 else ''}")
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
            self.report.say("translation taken from cache")
            return cached

        translation = run_translate(
            transcript,
            self.llm,
            target_language=target_language,
            chunk_chars=settings.chunk_chars,
            max_tokens=settings.llm_max_tokens,
            temperature=settings.llm_temperature,
            progress=lambda done, total: self.report.say(f"translating: batch {done}/{total}"),
        )
        self.cache.store("translation", key, translation)
        return translation

    def name_speakers(self, transcript: Transcript, *, force: bool = False) -> Transcript:
        """Puts names to the speaker labels, where the recording says them.

        Returns a transcript rather than the mapping alone: the names belong
        with the text they were read out of, and a caller that has to remember
        to attach them will one day forget.
        """
        if not transcript.speakers:
            return transcript

        key = stable_key(_content_key(transcript), model=self.settings.llm_repo, step="names")
        cached = None if force else self.cache.load("names", key, SpeakerNames)
        if cached is not None:
            self.report.say("speaker names taken from cache")
            return transcript.model_copy(update={"names": cached.names})

        found = run_name_speakers(transcript, self.llm)
        self.report.say(
            f"named {len(found)} of {len(transcript.speakers)} speakers"
            if found
            else "no names were said in the recording"
        )
        self.cache.store("names", key, SpeakerNames(names=found, llm_model=self.llm.model))
        return transcript.model_copy(update={"names": found})

    def read_formulas(self, transcript: Transcript, *, force: bool = False) -> Transcript:
        """Writes out, in LaTeX, the formulas the recording says in words.

        Beside the text and not instead of it: the words are what was said.
        """
        key = stable_key(_content_key(transcript), model=self.settings.llm_repo, step="formulas")
        cached = None if force else self.cache.load("formulas", key, SpokenFormulas)
        if cached is not None:
            self.report.say("formulas taken from cache")
            written = cached.formulas
        else:
            self.report.say("looking for formulas spoken in words")
            written = {
                str(number): formulas
                for number, formulas in run_read_formulas(
                    transcript.segments, self.llm, chunk_chars=self.settings.chunk_chars
                ).items()
            }
            self.cache.store(
                "formulas", key, SpokenFormulas(formulas=written, llm_model=self.llm.model)
            )

        self.report.say(f"formulas written out: {sum(len(v) for v in written.values())}")
        return transcript.model_copy(
            update={
                "segments": [
                    segment.model_copy(update={"formulas": written[str(number)]})
                    if str(number) in written
                    else segment
                    for number, segment in enumerate(transcript.segments)
                ]
            }
        )

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
            self.report.say("summary taken from cache")
            return cached

        summary = run_summarize(
            transcript,
            self.llm,
            language=language,
            chunk_chars=settings.chunk_chars,
            max_tokens=settings.llm_max_tokens,
            temperature=settings.llm_temperature,
            progress=lambda done, total: self.report.say(f"summarizing: step {done}/{total}"),
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
                self.report.say(f"speakers not labelled: {exc}")

        return source, transcript


def _content_key(transcript: Transcript) -> str:
    """A key over transcript content: editing segments invalidates the NLP steps."""
    return stable_key(transcript.text, language=transcript.language, model=transcript.asr_model)
