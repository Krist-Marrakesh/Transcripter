"""Мост окна: что именно уходит на диск.

Главное здесь — сохраняется показанное. Человек, глядящий на перевод, ждёт в
файле перевод; молчаливая запись оригинала в этом месте выглядит как исчезнувшая
работа.
"""

from __future__ import annotations

import threading
from pathlib import Path

import pytest

from transcriber.app.bridge import Api
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


def test_history_is_closed_while_work_is_running(api):
    """Регрессия: открытая из истории запись подменяла показанный транскрипт.

    Идущее распознавание этого не отменяет, и на экране остаётся чужая лекция —
    человек читает её, считая, что дождался своей.
    """
    api._busy = True

    assert api.open_history("любой") is False


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


class UpModel:
    """Модель, уже поднятая переводом или саммари: спросить её ничего не стоит."""

    model = "stub"
    loaded = True

    def complete(self, prompt: str, **kwargs) -> str:
        return "лекция о миграции птиц"


def test_topic_does_not_raise_the_model(api, monkeypatch):
    """Регрессия: строчка «о чём запись» стоила девятнадцати гигабайт на прогон.

    Веса поднимались в конце **каждого** распознавания — даже тому, кто саммари
    ни разу не нажимал, — ради сорока восьми токенов. Проверяем не «вернулся
    None», а что собрать модель никто даже не попытался: прежний код отдавал
    функцию, и гигабайты приезжали уже внутри истории.
    """

    def explode(*args, **kwargs):
        raise AssertionError("тема не стоит загрузки модели")

    monkeypatch.setattr("transcriber.pipeline.create_llm", explode)

    assert api._topic_writer() is None


def test_topic_is_written_by_a_model_already_up(api, monkeypatch):
    """Поднятая ради перевода модель называет тему даром — сорок восемь токенов."""
    monkeypatch.setattr("transcriber.pipeline.create_llm", lambda *a, **kw: UpModel())
    # Ровно то, что делает нажатие «Перевести»: слот занят, веса в памяти.
    api._llm_slot.get()

    describe = api._topic_writer()

    assert describe is not None
    assert describe("много слов про птиц") == "лекция о миграции птиц"


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


def test_advance_is_thinned_before_it_reaches_the_window(api, monkeypatch):
    """Whisper отчитывается на каждом окне — на часовой записи это полторы сотни
    переходов границы в webview. Глазу столько не нужно, а стоят они реально."""
    sent: list[float] = []
    monkeypatch.setattr(api, "_emit", lambda kind, **data: sent.append(data["done"]))
    advance = api._advance()

    for step in (0.001, 0.002, 0.05, 0.051, 0.4):
        advance(step)

    assert sent == [0.001, 0.05, 0.4]


def test_advance_always_reports_the_end(api, monkeypatch):
    """Полоса, замершая на 99.6%, читается как зависание."""
    sent: list[float] = []
    monkeypatch.setattr(api, "_emit", lambda kind, **data: sent.append(data["done"]))
    advance = api._advance()
    advance(0.996)

    advance(1.0)

    assert sent[-1] == 1.0


def test_advance_starts_over_for_the_next_recording(api, monkeypatch):
    """Своё прореживание у каждого запуска: иначе вторая запись начнётся с конца."""
    sent: list[float] = []
    monkeypatch.setattr(api, "_emit", lambda kind, **data: sent.append(data["done"]))
    first = api._advance()
    first(0.8)

    api._advance()(0.01)

    assert sent == [0.8, 0.01]


# --- дописанное к транскрипту и устаревшие файлы ---


@pytest.fixture
def speaking(api):
    """Транскрипт с размеченными спикерами: без них имена спрашивать не у кого."""
    api._transcript = Transcript(
        source="батл.mp4",
        language="ru",
        duration=1.0,
        segments=[Segment(start=0.0, end=1.0, text="Салют.", speaker="Спикер 1")],
        asr_model="stub",
    )
    return api


def stub_steps(monkeypatch):
    """Оба шага обогащения — без модели, но с тем же следом в транскрипте."""
    from transcriber.pipeline import Pipeline

    monkeypatch.setattr(
        Pipeline,
        "name_speakers",
        lambda self, t, **kw: t.model_copy(update={"names": {"Спикер 1": "Ресторатор"}}),
    )
    monkeypatch.setattr(
        Pipeline,
        "read_formulas",
        lambda self, t, **kw: t.model_copy(
            update={
                "segments": [s.model_copy(update={"formulas": ["E = mc^2"]}) for s in t.segments]
            }
        ),
    )


def test_both_additions_survive_each_other(speaking, monkeypatch):
    """Регрессия по замыслу: шаги идут по одному транскрипту и не затирают друг друга.

    Имена живут отображением на транскрипте, формулы — полями сегментов, и
    каждый шаг обязан скопировать сделанное предыдущим.
    """
    stub_steps(monkeypatch)

    speaking._run_nlp("enrich", "", ("names", "formulas"))

    assert speaking._transcript.names == {"Спикер 1": "Ресторатор"}
    assert speaking._transcript.segments[0].formulas == ["E = mc^2"]


def test_saving_then_adding_marks_the_file_behind(speaking, monkeypatch, tmp_path):
    """Файл на диске не меняется от того, что изменилось окно.

    Без этой отметки человек уносит прошлую версию, уверенный, что унёс нынешнюю.
    """
    stub_steps(monkeypatch)
    saved = Path(speaking.export("txt", directory=str(tmp_path)))
    assert speaking._stale == []

    speaking._run_nlp("enrich", "", ("names",))

    assert speaking._stale == [str(saved)]


def test_saving_again_catches_the_file_up(speaking, monkeypatch, tmp_path):
    stub_steps(monkeypatch)
    speaking.export("txt", directory=str(tmp_path))
    speaking._run_nlp("enrich", "", ("names",))

    speaking.export("txt", directory=str(tmp_path))

    assert speaking._stale == []


def test_only_the_format_saved_again_catches_up(speaking, monkeypatch, tmp_path):
    """Переписали txt — md всё ещё отстаёт, и молчать об этом нельзя."""
    stub_steps(monkeypatch)
    speaking.export("txt", directory=str(tmp_path))
    stale_md = Path(speaking.export("md", directory=str(tmp_path)))
    speaking._run_nlp("enrich", "", ("names",))

    speaking.export("txt", directory=str(tmp_path))

    assert speaking._stale == [str(stale_md)]


def test_nothing_saved_means_nothing_to_warn_about(speaking, monkeypatch):
    """Строка про устаревшее не должна появляться у того, кто ничего не сохранял."""
    stub_steps(monkeypatch)

    speaking._run_nlp("enrich", "", ("names", "formulas"))

    assert speaking._stale == []


def test_naming_without_speakers_still_writes_the_formulas(api, monkeypatch):
    """Называть некого — не повод отменять вторую половину работы.

    Правило одно на оба пути — и на кнопку, и на распознавание с галочкой, —
    поэтому живёт в самой работе, а не в том, кто её начал.
    """
    stub_steps(monkeypatch)

    api._run_nlp("enrich", "", ("names", "formulas"))

    assert api._transcript.segments[0].formulas == ["E = mc^2"]
    assert api._transcript.names == {}


def test_asking_only_for_names_without_speakers_starts_nothing(api, monkeypatch):
    """Работы не осталось — и поднимать ради неё модель не за что."""
    monkeypatch.setattr(Api, "_start_nlp", lambda *a, **kw: pytest.fail("работать не над чем"))

    assert api.enrich(["names"]) is False


def test_an_empty_choice_is_refused(api, monkeypatch):
    monkeypatch.setattr(Api, "_start_nlp", lambda *a, **kw: pytest.fail("ничего не выбрано"))

    assert api.enrich([]) is False


def test_the_summary_is_saved_beside_the_transcript_not_instead_of_it(api, tmp_path):
    """Регрессия: в окне саммари только показывалось и уходило с закрытием.

    Отдельным файлом и с отдельным суффиксом — иначе `md` с самим транскриптом
    затёрся бы тем, чего у него не просили.
    """
    api._summary = "# лекция\n\nо чём была речь"
    transcript_md = Path(api.export("md", directory=str(tmp_path)))

    summary_md = Path(api.save_summary(directory=str(tmp_path)))

    assert summary_md != transcript_md
    assert summary_md.name.endswith(".summary.md")
    assert "о чём была речь" in summary_md.read_text(encoding="utf-8")
    assert "о чём была речь" not in transcript_md.read_text(encoding="utf-8")


def test_saving_a_summary_nobody_asked_for_explains_itself(api, tmp_path):
    with pytest.raises(RuntimeError, match="ask for a summary first"):
        api.save_summary(directory=str(tmp_path))


def test_ticking_before_the_run_writes_it_into_the_saved_copy(api, monkeypatch, tmp_path):
    """Отмеченное до запуска делается в конце распознавания, а не после показа.

    Порядок важен: сначала дописать, потом отдать в историю и на экран. Иначе
    сохранённая копия остаётся без имён, а показанное устаревает у человека на
    глазах — сразу, ещё до того как он что-либо сохранил.
    """
    from transcriber.app import history
    from transcriber.pipeline import Pipeline

    stub_steps(monkeypatch)
    monkeypatch.setattr(history, "ROOT", tmp_path)
    monkeypatch.setattr(history, "INDEX", tmp_path / "history.json")
    monkeypatch.setattr(history, "ENTRIES", tmp_path / "entries")

    audio = tmp_path / "батл.wav"
    audio.write_bytes(b"")
    source = Source(audio=audio, origin="батл.mp4", title="батл", duration=1.0)
    spoken = Transcript(
        source="батл.mp4",
        language="ru",
        duration=1.0,
        segments=[Segment(start=0.0, end=1.0, text="Салют.", speaker="Спикер 1")],
        asr_model="stub",
    )
    monkeypatch.setattr(Pipeline, "run", lambda self, *a, **kw: (source, spoken))

    api._stop_job = threading.Event()
    api._run("батл.mp4", {"add": ["names", "formulas"]})

    assert api._transcript.names == {"Спикер 1": "Ресторатор"}
    assert api._transcript.segments[0].formulas == ["E = mc^2"]
    stored = history.open_entry(history.load()[0]["key"])
    assert stored.names == {"Спикер 1": "Ресторатор"}
    # Ничего не сохраняли — и пугать устаревшими файлами не с чего.
    assert api._stale == []
