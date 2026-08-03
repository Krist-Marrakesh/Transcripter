"""Веса моделей: что уже на диске, сколько занимает и как докачать.

Единственное место, которое знает, как HuggingFace раскладывает свой кэш. Знание
это нужно ровно затем, чтобы спрашивать «скачано ли», не выходя в сеть: без
файлов на диске первое же распознавание молча уходит за гигабайтами, и человек
несколько минут смотрит в неподвижное окно, не понимая, что происходит.

Скачивание здесь же, потому что качать веса — тоже отдельный шаг со своей
длительностью, и он должен быть виден, а не прятаться внутри первого запуска.
"""

from __future__ import annotations

import os
import subprocess
import sys
import threading
import time
from collections.abc import Callable
from dataclasses import dataclass
from pathlib import Path

from .config import Settings

# Файлы весов у разных сборок называются по-разному: mlx-lm хранит шарды
# `*.safetensors`, mlx-whisper — один `weights.npz`. Спрашивать надо именно про
# веса: конфиг и токенизатор приезжают первыми, поэтому оборванная закачка
# оставляет каталог, который выглядит как готовая модель.
WEIGHT_SUFFIXES = (".safetensors", ".npz")

Progress = Callable[[int, int], None]
"""Сколько байт уже на диске и сколько ожидается всего (0 — размер неизвестен)."""


@dataclass(frozen=True)
class ModelWeights:
    """Веса одной модели в том виде, в каком о них говорят пользователю."""

    repo: str
    role: str
    """`asr` или `llm` — чем эта модель занята в пайплайне."""

    title: str
    """Короткое имя из реестра: человек выбирает модель именно им."""

    ready: bool
    size: int
    """Байты на диске. У недокачанной модели это размер того, что доехало."""


def hub() -> Path:
    """Корень кэша HuggingFace с учётом переменных окружения."""
    try:
        from huggingface_hub.constants import HF_HUB_CACHE

        return Path(HF_HUB_CACHE)
    except ImportError:
        return Path.home() / ".cache" / "huggingface" / "hub"


def cache_dir(repo: str) -> Path:
    """Каталог модели — так его называет сама библиотека."""
    return hub() / ("models--" + repo.replace("/", "--"))


def is_ready(repo: str) -> bool:
    """Есть ли на диске сами веса, а не только сопроводительные файлы."""
    # Отсутствующий каталог `glob` переживает молча — отдельная проверка не нужна.
    files = (cache_dir(repo) / "snapshots").glob("*/*")
    return any(path.suffix in WEIGHT_SUFFIXES for path in files)


def local_size(repo: str) -> int:
    """Сколько из модели уже лежит на диске.

    Ссылки пропускаем: в кэше HuggingFace файлы снапшота — это симлинки на
    `blobs`, и учёт тех и других удвоил бы размер.

    Незавершённые куски складывать нельзя. Каждая прерванная попытка оставляет
    свой `*.incomplete`, и сумма быстро перерастает саму модель — наблюдалось
    4.5 ГБ там, где весов 2.9. Считается только самый большой из них: он и есть
    текущая попытка, остальные — брошенный мусор.
    """
    done = partial = 0
    for root, _, files in os.walk(cache_dir(repo)):
        for name in files:
            path = Path(root) / name
            try:
                if path.is_symlink():
                    continue
                size = path.stat().st_size
            except OSError:
                # Файл исчез между обходом и замером — идёт закачка, не беда.
                continue
            if name.endswith(".incomplete"):
                partial = max(partial, size)
            else:
                done += size
    return done + partial


def remote_size(repo: str, *, attempts: int = 3) -> int:
    """Сколько модель весит на HuggingFace. Ноль означает «спросить не удалось».

    Спрашиваем несколько раз: запрос уходит одновременно со стартом закачки, и
    единственная осечка стоит дорого — полоса остаётся без шкалы до конца, хотя
    сеть уже через секунду в порядке.

    Совсем без размера закачка всё равно идёт: прогресс тогда считает гигабайты,
    а не проценты.
    """
    from huggingface_hub import HfApi

    for attempt in range(attempts):
        try:
            info = HfApi().model_info(repo, files_metadata=True)
        except Exception:
            # Размер — вещь вспомогательная, ради него закачку не отменяют.
            time.sleep(attempt)
            continue
        return sum(file.size or 0 for file in (info.siblings or []))
    return 0


def required(settings: Settings) -> list[ModelWeights]:
    """Веса, нужные при текущих настройках.

    Только для бэкендов, которые качают файлы сами. У Ollama модели живут внутри
    демона, у CTranslate2 — под своими именами; ни то, ни другое отсюда не видно
    и докачивать нам нечего.
    """
    items: list[ModelWeights] = []
    if settings.asr_backend == "mlx":
        items.append(_describe(settings.asr_repo, "asr", settings.asr_model))
    if settings.llm_backend == "mlx":
        items.append(_describe(settings.llm_repo, "llm", settings.llm_model))
    return items


def missing(settings: Settings) -> list[ModelWeights]:
    return [item for item in required(settings) if not item.ready]


def _describe(repo: str, role: str, title: str) -> ModelWeights:
    ready = is_ready(repo)
    return ModelWeights(
        repo=repo,
        role=role,
        title=title,
        ready=ready,
        # Размер недокачанной модели — то, что уже доехало: по нему видно, что
        # закачка была прервана, а не что её не начинали.
        size=local_size(repo),
    )


class DownloadStopped(RuntimeError):
    """Закачку остановили по просьбе человека."""


def sweep(repo: str) -> int:
    """Убирает брошенные куски прошлых попыток. Возвращает освобождённые байты.

    Прерванная закачка оставляет свой `*.incomplete`, и следующая попытка заводит
    новый — куски копятся, пока каталог не перерастёт саму модель. Свежий кусок
    не трогаем: он либо принадлежит идущей закачке, либо пригодится ей, если
    библиотека всё-таки сумеет дописать начатое.
    """
    blobs = cache_dir(repo) / "blobs"
    pieces = sorted(blobs.glob("*.incomplete"), key=lambda path: path.stat().st_mtime)

    freed = 0
    # Последний по времени — текущая попытка, остальные уже никому не нужны.
    for piece in pieces[:-1]:
        try:
            freed += piece.stat().st_size
            piece.unlink()
        except OSError:
            # Не удалось убрать — это мусор, а не работа: продолжаем.
            continue
    return freed


# Загрузчик HuggingFace прервать изнутри нельзя — он не спрашивает, ждут ли его
# ещё. Поэтому он живёт отдельным процессом: такой останавливается сразу.
#
# Прогресс при остановке теряется. Сама библиотека докачку умеет — шлёт `Range`
# от размера начатого файла, — но при включённом Xet (пакет `hf_xet`, он стоит по
# умолчанию) каждая попытка заводит собственный временный файл. У прерванной
# закачки наблюдалось четыре куска, в сумме больше самой модели. Поэтому
# остановка означает «начать заново», а не «продолжить с места обрыва».
_FETCH = "import sys; from huggingface_hub import snapshot_download; snapshot_download(sys.argv[1])"


def _interpreter() -> str:
    """Интерпретатор, у которого точно есть наши зависимости.

    `sys.executable` для этого не годится: pywebview на macOS перезапускает
    приложение через Python.app, и у того нет ни venv, ни `huggingface_hub` —
    загрузчик падал мгновенно, а окно показывало нулевую скорость.
    """
    inside_venv = Path(sys.prefix) / "bin" / "python"
    return str(inside_venv) if inside_venv.exists() else sys.executable


def _spawn(repo: str) -> subprocess.Popen[str]:
    env = os.environ | {
        # Короткий таймаут чтения: повисшее соединение должно оборваться быстро,
        # чтобы его успел заметить и перезапустить сторож ниже.
        "HF_HUB_DOWNLOAD_TIMEOUT": "20",
        # Пути наследуются явно: даже если интерпретатор окажется чужим, наши
        # пакеты он всё равно найдёт.
        "PYTHONPATH": os.pathsep.join(p for p in sys.path if p),
        # Xet — новый транспорт HuggingFace, и на этой машине он оказался хуже
        # обычного HTTP по всем статьям: 0.2 МБ за пятнадцать секунд против
        # 13 МБ/с, файл при этом растёт в собственном кэше чанков, а не в
        # каталоге модели — прогресс не виден, и каждая попытка начинает заново.
        # Обычный путь пишет в `blob.incomplete` и умеет докачку по `Range`.
        "HF_HUB_DISABLE_XET": "1",
    }
    return subprocess.Popen(
        [_interpreter(), "-c", _FETCH, repo],
        stdout=subprocess.DEVNULL,
        stderr=subprocess.PIPE,
        text=True,
        env=env,
    )


# Когда считать закачку повисшей. Соединение с HuggingFace умеет замирать, не
# разрываясь: процесс жив, файл открыт, данных нет. Наблюдалось вживую — сперва
# ни байта прироста, потом скорость в единицы килобайт.
#
# Порог задан в байтах, а не «строго ноль»: еле ползущее соединение неотличимо
# от мёртвого по последствиям. Мегабайт за три четверти минуты — это 23 КБ/с;
# на такой скорости трёхгигабайтная модель качалась бы полтора суток.
STALL_SECONDS = 45.0
STALL_BYTES = 1 << 20
STALL_RESTARTS = 5


def download(
    repo: str,
    on_progress: Progress | None = None,
    *,
    period: float = 0.7,
    cancel: threading.Event | None = None,
    notify: Callable[[str], None] | None = None,
) -> Path:
    """Скачивает веса, докладывая, сколько байт уже на диске.

    Прогресс считается по размеру каталога, а не по внутренностям загрузчика:
    так он не зависит от того, сколькими файлами и в каком порядке приезжает
    модель, и показывает ровно то, что занимает место.

    Повисшее соединение перезапускается. Ждать его бесполезно: процесс жив, файл
    открыт, а байты не идут — со стороны это неотличимо от медленной сети, и
    человек смотрит на замерший счётчик, не зная, что закачка уже мертва. Цена
    перезапуска — начатый файл, поэтому порог застоя выбран щедрым.
    """
    total = remote_size(repo) if on_progress else 0
    say = notify or (lambda _: None)
    sweep(repo)

    for restart in range(STALL_RESTARTS + 1):
        process = _spawn(repo)
        if _pump(repo, process, total, on_progress, cancel, period):
            break

        _stop(process)
        if restart == STALL_RESTARTS:
            raise RuntimeError(f"закачка {repo} не двигается: сеть не отдаёт данные")
        say(f"закачка встала, продолжаю с этого места ({restart + 1} из {STALL_RESTARTS})")

    if process.returncode != 0:
        reason = (process.stderr.read() if process.stderr else "").strip().splitlines()
        raise RuntimeError("\n".join([f"не удалось скачать {repo}", *reason[-3:]]))

    if on_progress is not None:
        final = local_size(repo)
        on_progress(final, total or final)
    return cache_dir(repo)


def _pump(
    repo: str,
    process: subprocess.Popen[str],
    total: int,
    on_progress: Progress | None,
    cancel: threading.Event | None,
    period: float,
) -> bool:
    """Следит за идущей закачкой. `True` — процесс закончил сам, `False` — завис."""
    seen, moved_at = local_size(repo), time.monotonic()

    while process.poll() is None:
        if cancel is not None and cancel.is_set():
            _stop(process)
            raise DownloadStopped(repo)

        done = local_size(repo)
        if done - seen >= STALL_BYTES:
            seen, moved_at = done, time.monotonic()
        elif time.monotonic() - moved_at > STALL_SECONDS:
            return False

        if on_progress is not None:
            on_progress(done, total)
        time.sleep(period)
    return True


def _stop(process: subprocess.Popen[str]) -> None:
    process.terminate()
    try:
        process.wait(timeout=30)
    except subprocess.TimeoutExpired:
        process.kill()
