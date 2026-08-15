"""Учёт весов на диске.

Проверяется без сети и без самих моделей: важно не то, как HuggingFace качает
файлы, а то, отличаем ли мы готовую модель от оборванной закачки. Ошибка здесь
стоит дорого — приложение либо молча уходит за гигабайтами, либо предлагает
скачать то, что уже лежит.
"""

from __future__ import annotations

import io
import os
import threading
from pathlib import Path

import pytest

import transcriber
from transcriber import weights
from transcriber.config import Settings
from transcriber.diarize import sherpa_backend
from transcriber.report import Report, Stopped

REPO = "mlx-community/Qwen3.6-35B-A3B-4bit"
WHISPER = "mlx-community/whisper-large-v3-mlx"


@pytest.fixture
def hub(tmp_path, monkeypatch):
    """Пустая машина: ни одной скачанной модели, ни в одном из двух мест.

    Мест именно два, и подменять надо оба. Кэш HuggingFace держит распознавание и
    LLM, а модели спикеров приезжают не оттуда, и папка у них своя. Пока в панели
    не было строки со спикерами, хватало и одного; с её появлением тест начал
    отвечать не про код, а про машину — на чистой проходил, на рабочей падал,
    потому что 104 МБ моделей на ней уже лежали.
    """
    monkeypatch.setattr(weights, "hub", lambda: tmp_path)
    monkeypatch.setattr(sherpa_backend, "home", lambda: tmp_path / "diarization")
    return tmp_path


def put(hub: Path, repo: str, name: str, data: bytes = b"x") -> Path:
    """Кладёт файл в снапшот модели — так, как это делает сама библиотека."""
    path = weights.cache_dir(repo) / "snapshots" / "rev1" / name
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_bytes(data)
    return path


def test_missing_model_is_not_ready(hub):
    assert weights.is_ready(REPO) is False


def test_metadata_alone_is_not_ready(hub):
    """Конфиг и токенизатор приезжают первыми — модели ещё нет."""
    put(hub, REPO, "config.json")
    put(hub, REPO, "tokenizer.json")

    assert weights.is_ready(REPO) is False


def test_safetensors_make_model_ready(hub):
    put(hub, REPO, "model-00001-of-00004.safetensors")

    assert weights.is_ready(REPO) is True


def test_npz_weights_make_model_ready(hub):
    """mlx-whisper хранит веса одним `weights.npz`, а не шардами safetensors."""
    put(hub, WHISPER, "weights.npz")

    assert weights.is_ready(WHISPER) is True


def test_local_size_counts_downloaded_bytes(hub):
    put(hub, REPO, "model.safetensors", b"0123456789")

    assert weights.local_size(REPO) >= 10


def symlinks_allowed(where: Path) -> bool:
    """Разрешает ли система создавать символические ссылки.

    Спрашиваем попыткой, а не по имени системы: на Windows это решают режим
    разработчика и права процесса, а не версия, и угадать снаружи нельзя.
    """
    target = where / "цель"
    target.write_bytes(b"")
    try:
        (where / "ссылка").symlink_to(target)
    except OSError:
        return False
    return True


def test_local_size_does_not_double_count_symlinks(hub):
    """В кэше файл снапшота — ссылка на blob; учёт обоих удвоил бы размер.

    Там, где ссылок нельзя, проверять нечего: библиотека кладёт blob в снапшот
    переносом, а не ссылкой, и дважды считать становится нечего. Ровно это и
    записано рядом с `local_size`, так что пропуск здесь — не обход неудобства,
    а отсутствие самого случая.
    """
    if not symlinks_allowed(hub):
        pytest.skip("система не разрешает символические ссылки")

    blob = weights.cache_dir(REPO) / "blobs" / "abc"
    blob.parent.mkdir(parents=True)
    blob.write_bytes(b"0123456789")

    link = weights.cache_dir(REPO) / "snapshots" / "rev1" / "model.safetensors"
    link.parent.mkdir(parents=True)
    link.symlink_to(blob)

    assert weights.local_size(REPO) < 20


def test_required_lists_both_models_on_mlx(hub):
    items = weights.required(Settings(asr_backend="mlx", llm_backend="mlx"))

    assert [item.role for item in items] == ["asr", "llm", "speakers"]
    assert all(item.ready is False for item in items)


def test_required_names_recognition_on_every_backend(hub):
    """Регрессия: вне Apple Silicon список был пуст, и панель не показывалась вовсе.

    Человеку на Windows не говорили ничего, а первое же распознавание уходило
    качать три гигабайта — с полосой, которая относится к распознаванию. Со
    стороны это выглядит как зависшее приложение.

    Спрашивается наличие строки, и только оно. Имя репозитория разворачивает
    `hub_repo`, спрашивая саму библиотеку, а на Apple Silicon её нет — маркер в
    зависимостях туда её и не ставит; разворот проверен ниже, и там стоит
    `importorskip`. Название модели тоже не спрашивается: оно приезжает из
    настроек, а те подхватывают `.env` разработчика — тест отвечал бы про машину,
    на которой запущен, ровно как и до правки.
    """
    items = weights.required(Settings(asr_backend="faster", llm_backend="ollama"))

    assert [item.role for item in items] == ["asr", "speakers"]


def test_required_leaves_the_daemons_models_to_the_daemon(hub):
    """У Ollama веса внутри демона: отсюда их не видно и качать нечего."""
    items = weights.required(Settings(asr_backend="mlx", llm_backend="ollama"))

    assert "llm" not in [item.role for item in items]


def test_the_repository_is_the_one_the_backend_will_fetch(hub):
    """Короткое имя разворачивается под каждый бэкенд по-своему.

    Спросить настройки мало: в них остаётся mlx-репозиторий, а считает
    CTranslate2 из своего. Ошибка здесь молчалива с обоих концов — панель зовёт
    скачать уже скачанное, распознавание при этом работает.
    """
    pytest.importorskip("faster_whisper")

    mlx = weights.asr_repo(Settings(asr_backend="mlx", asr_model="large-v3"))
    faster = weights.asr_repo(Settings(asr_backend="faster", asr_model="large-v3"))

    assert mlx == "mlx-community/whisper-large-v3-mlx"
    assert faster == "Systran/faster-whisper-large-v3"


def test_the_turbo_repository_is_not_guessed_by_pattern(hub):
    """`turbo` уезжает к другому владельцу — образец «Systran/…» на нём врёт."""
    pytest.importorskip("faster_whisper")

    assert weights.asr_repo(Settings(asr_backend="faster", asr_model="turbo")) == (
        "mobiuslabsgmbh/faster-whisper-large-v3-turbo"
    )


def test_a_ctranslate2_model_on_disk_counts_as_ready(hub):
    """Регрессия: `model.bin` не считался весами, и скачанное звали отсутствующим."""
    put(hub, "Systran/faster-whisper-large-v3", "model.bin", b"x" * 10)

    assert weights.is_ready("Systran/faster-whisper-large-v3") is True


def test_missing_hides_downloaded_models(hub):
    put(hub, WHISPER, "weights.npz")
    settings = Settings(asr_backend="mlx", llm_backend="mlx", asr_model="large-v3")

    assert "asr" not in [item.role for item in weights.missing(settings)]


def test_speaker_models_are_named_though_they_are_not_a_repository(hub, monkeypatch):
    """Регрессия: панель молчала про разметку, а её пара весит сотню мегабайт.

    Галочку «разметить говорящих» ставят рядом с записью, поэтому веса к ней
    нужны в любой момент — узнать об этом посреди распознавания значит узнать
    поздно. Репозитория у пары нет: это два архива с GitHub, и имя ей дано
    только затем, чтобы панели было что нажать.
    """
    monkeypatch.setattr(sherpa_backend, "ready", lambda: False)
    monkeypatch.setattr(sherpa_backend, "local_size", lambda: 40 * 1024**2)

    speakers = [
        item
        for item in weights.required(Settings(diarization_backend="sherpa"))
        if item.role == "speakers"
    ]

    assert [item.repo for item in speakers] == [weights.SHERPA]
    assert speakers[0].ready is False
    assert speakers[0].size == 40 * 1024**2


def test_pyannote_is_named_as_the_repository_it_is(hub):
    """У pyannote репозиторий настоящий, и вся здешняя механика ему подходит."""
    settings = Settings(diarization_backend="pyannote")

    speakers = [item for item in weights.required(settings) if item.role == "speakers"]

    assert speakers[0].repo == settings.diarization_model
    assert speakers[0].ready is False


class FakeDownloader:
    """Процесс-загрузчик: сколько он успевает скачать и чем кончает — задаёт тест."""

    def __init__(self, *, writes=(), code=0, stderr="", hangs=False) -> None:
        self._writes = list(writes)
        self._code = code
        self._hangs = hangs
        self.stderr = io.StringIO(stderr)
        self.returncode = None
        self.killed = False

    def poll(self):
        if self._writes:
            self._writes.pop(0)()
            return None
        if self._hangs:
            return None
        self.returncode = self._code
        return self._code

    def terminate(self):
        self.killed = True
        self.returncode = -15

    def wait(self, timeout=None):
        return self.returncode

    def kill(self):
        self.killed = True


def spawning(monkeypatch, *processes):
    """Подменяет запуск загрузчика: каждая новая попытка берёт следующий процесс."""
    queue = list(processes)
    started: list[FakeDownloader] = []

    def spawn(repo: str):
        process = queue.pop(0) if queue else FakeDownloader()
        started.append(process)
        return process

    monkeypatch.setattr(weights, "_spawn", spawn)
    return started


def test_xet_is_off_for_a_download_inside_the_process(monkeypatch):
    """Регрессия: Xet гасился только в отдельном загрузчике, а тот идёт не всегда.

    `required()` знает лишь mlx-бэкенды, поэтому там, где ASR идёт через
    faster-whisper, веса тянет сама библиотека изнутри процесса — и гасить Xet в
    той закачке было некому. Она упала на тринадцатой минуте и выбросила все
    952 МБ: файл копится в чанк-кэше Xet, а не в папке модели.
    """
    monkeypatch.delenv("HF_HUB_DISABLE_XET", raising=False)

    transcriber._keep_downloads_on_plain_http()

    assert os.environ["HF_HUB_DISABLE_XET"] == "1"


def test_xet_asked_for_outright_is_left_alone(monkeypatch):
    """Замер сделан на одной машине и одной сети — переменная и есть способ возразить."""
    monkeypatch.setenv("HF_HUB_DISABLE_XET", "0")

    transcriber._keep_downloads_on_plain_http()

    assert os.environ["HF_HUB_DISABLE_XET"] == "0"


def test_download_reports_what_reached_disk(hub, monkeypatch):
    """Прогресс снимается с диска, а не с внутренностей загрузчика.

    Значит он не зависит от того, сколькими файлами приезжает модель, и в конце
    обязан совпасть с тем, что реально занято.
    """
    seen: list[tuple[int, int]] = []
    spawning(
        monkeypatch,
        FakeDownloader(writes=[lambda: put(hub, REPO, "model.safetensors", b"x" * 1000)]),
    )
    monkeypatch.setattr(weights, "remote_size", lambda repo: 1000)

    weights.download(REPO, lambda done, total: seen.append((done, total)), period=0.01)

    assert seen[-1] == (1000, 1000)
    assert weights.is_ready(REPO) is True


def test_download_survives_unknown_remote_size(hub, monkeypatch):
    """Сеть не сказала размер — качаем всё равно, просто без процентов."""
    seen: list[tuple[int, int]] = []
    spawning(
        monkeypatch,
        FakeDownloader(writes=[lambda: put(hub, WHISPER, "weights.npz", b"x" * 42)]),
    )
    monkeypatch.setattr(weights, "remote_size", lambda repo: 0)

    weights.download(WHISPER, lambda done, total: seen.append((done, total)), period=0.01)

    assert seen[-1] == (42, 42)


def test_stalled_download_is_restarted(hub, monkeypatch):
    """Соединение умеет замирать не разрываясь: процесс жив, а байты не идут.

    Наблюдалось вживую — 751 МБ и ни байта прироста. Ждать такое бесполезно,
    поэтому загрузчик перезапускается: скачанное остаётся, докачка продолжается.
    """
    monkeypatch.setattr(weights, "STALL_SECONDS", 0.05)
    monkeypatch.setattr(weights, "remote_size", lambda repo: 100)
    notes: list[str] = []
    started = spawning(
        monkeypatch,
        FakeDownloader(hangs=True),
        FakeDownloader(writes=[lambda: put(hub, REPO, "model.safetensors", b"x" * 100)]),
    )

    weights.download(REPO, period=0.01, report=Report(say=notes.append))

    assert len(started) == 2
    assert started[0].killed is True
    assert "stalled" in notes[0]


def test_hopeless_download_gives_up_with_a_reason(hub, monkeypatch):
    monkeypatch.setattr(weights, "STALL_SECONDS", 0.05)
    monkeypatch.setattr(weights, "STALL_RESTARTS", 1)
    spawning(monkeypatch, FakeDownloader(hangs=True), FakeDownloader(hangs=True))

    with pytest.raises(RuntimeError, match="not moving"):
        weights.download(REPO, period=0.01)


def test_download_can_be_stopped(hub, monkeypatch):
    """Пауза — решение человека, а не отказ: скачанное остаётся на диске."""
    put(hub, REPO, "part.incomplete", b"x" * 10)
    process = FakeDownloader(hangs=True)
    spawning(monkeypatch, process)
    stop = threading.Event()
    stop.set()

    with pytest.raises(Stopped):
        weights.download(REPO, period=0.01, report=Report(cancel=stop))

    assert process.killed is True
    assert weights.local_size(REPO) == 10


def test_failed_download_reports_what_the_loader_said(hub, monkeypatch):
    spawning(monkeypatch, FakeDownloader(code=1, stderr="OSError: no space left"))

    with pytest.raises(RuntimeError, match="no space left"):
        weights.download(REPO, period=0.01)


def put_blob(hub, repo: str, name: str, data: bytes, *, age: float = 0.0):
    """Кладёт файл прямо в blobs — там живут байты и недокачанные куски."""
    path = weights.cache_dir(repo) / "blobs" / name
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_bytes(data)
    if age:
        stamp = path.stat().st_mtime - age
        os.utime(path, (stamp, stamp))
    return path


def test_local_size_does_not_add_up_abandoned_attempts(hub):
    """Найдено вживую: 4.5 ГБ в каталоге там, где вся модель весит 2.9.

    Каждая прерванная закачка оставляет свой кусок. Складывать их нельзя —
    прогресс тогда переваливает за сто процентов ещё до конца работы.
    """
    put_blob(hub, REPO, "abc.11111111.incomplete", b"x" * 700, age=600)
    put_blob(hub, REPO, "abc.22222222.incomplete", b"x" * 900)

    assert weights.local_size(REPO) == 900


def test_local_size_counts_finished_files_together_with_the_current_piece(hub):
    put_blob(hub, REPO, "done", b"x" * 1000)
    put_blob(hub, REPO, "abc.33333333.incomplete", b"x" * 250)

    assert weights.local_size(REPO) == 1250


def test_sweep_keeps_the_freshest_piece(hub):
    """С него закачка продолжится; остальные — брошенный мусор."""
    old = put_blob(hub, REPO, "abc.11111111.incomplete", b"x" * 700, age=600)
    fresh = put_blob(hub, REPO, "abc.22222222.incomplete", b"x" * 900)

    freed = weights.sweep(REPO)

    assert freed == 700
    assert old.exists() is False
    assert fresh.exists() is True


def test_sweep_is_harmless_when_there_is_nothing_to_clean(hub):
    assert weights.sweep(REPO) == 0
