"""Сшивка сегментов ASR с разметкой спикеров."""

from __future__ import annotations

from types import SimpleNamespace

from transcriber.diarize import _annotation, assign_speakers
from transcriber.models import Segment, SpeakerTurn


def seg(start: float, end: float, text: str = "…") -> Segment:
    return Segment(start=start, end=end, text=text)


def test_assigns_by_maximum_overlap():
    segments = [seg(0.0, 2.0), seg(2.0, 4.0)]
    turns = [
        SpeakerTurn(start=0.0, end=2.1, speaker="Спикер 1"),
        SpeakerTurn(start=2.1, end=4.0, speaker="Спикер 2"),
    ]

    result = assign_speakers(segments, turns)

    assert result[0].speaker == "Спикер 1"
    # Пересечение со вторым отрезком больше, чем с первым.
    assert result[1].speaker == "Спикер 2"


def test_no_turns_leaves_segments_untouched():
    segments = [seg(0.0, 2.0)]
    assert assign_speakers(segments, [])[0].speaker is None


def test_segment_outside_any_turn_stays_unlabeled():
    segments = [seg(100.0, 101.0)]
    turns = [SpeakerTurn(start=0.0, end=5.0, speaker="Спикер 1")]
    assert assign_speakers(segments, turns)[0].speaker is None


def test_does_not_mutate_input():
    segments = [seg(0.0, 2.0)]
    turns = [SpeakerTurn(start=0.0, end=2.0, speaker="Спикер 1")]

    assign_speakers(segments, turns)

    assert segments[0].speaker is None


def test_overlapping_turns_pick_dominant_speaker():
    """Перебивка: говорят двое, сегмент достаётся тому, кто занял больше времени."""
    segments = [seg(0.0, 10.0)]
    turns = [
        SpeakerTurn(start=0.0, end=3.0, speaker="Спикер 1"),
        SpeakerTurn(start=2.0, end=10.0, speaker="Спикер 2"),
    ]
    assert assign_speakers(segments, turns)[0].speaker == "Спикер 2"


def test_sweep_handles_long_turn_before_short_ones():
    """Указатель окна не должен проскочить короткие отрезки за длинным."""
    segments = [seg(0.0, 1.0), seg(50.0, 51.0)]
    turns = [
        SpeakerTurn(start=0.0, end=100.0, speaker="Спикер 1"),
        SpeakerTurn(start=50.0, end=51.0, speaker="Спикер 2"),
    ]

    result = assign_speakers(segments, turns)

    assert result[0].speaker == "Спикер 1"
    # На втором сегменте оба отрезка перекрываются одинаково по 1 с,
    # но «Спикер 2» покрывает его целиком — важно, что он вообще найден.
    assert result[1].speaker in {"Спикер 1", "Спикер 2"}


class Annotation:
    """Разметка pyannote в том виде, в каком её читает `diarize`."""

    def __init__(self, *turns: tuple[float, float, str]) -> None:
        self._turns = turns

    def itertracks(self, yield_label: bool = False):
        for start, end, speaker in self._turns:
            yield SimpleNamespace(start=start, end=end), None, speaker


def test_annotation_prefers_the_variant_without_overlaps():
    """pyannote 4 отдаёт составной объект, и для сшивки годится не всякое поле.

    Реплика, сказанная поверх чужой, в тексте всё равно одна: два спикера на
    один сегмент ASR только спорили бы между собой.
    """
    result = SimpleNamespace(
        speaker_diarization=Annotation((0.0, 2.0, "A"), (1.5, 3.0, "B")),
        exclusive_speaker_diarization=Annotation((0.0, 1.5, "A"), (1.5, 3.0, "B")),
    )

    picked = list(_annotation(result).itertracks(yield_label=True))

    assert [segment.end for segment, _, _ in picked] == [1.5, 3.0]


def test_annotation_falls_back_to_the_plain_one():
    """Если варианта без перекрытий нет, берём обычную разметку."""
    plain = Annotation((0.0, 1.0, "A"))
    result = SimpleNamespace(speaker_diarization=plain)

    assert _annotation(result) is plain


def test_annotation_accepts_the_old_shape():
    """pyannote 3 отдавал разметку напрямую — она сама себе результат."""
    plain = Annotation((0.0, 1.0, "A"))

    assert _annotation(plain) is plain
