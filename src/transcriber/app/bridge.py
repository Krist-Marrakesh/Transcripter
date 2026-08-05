"""Мост между окном и пайплайном.

Разделение ролей жёсткое: здесь нет ни одного шага обработки — только запуск
пайплайна в рабочем потоке и перекладывание результата в окно. Вся логика
остаётся в `pipeline`, поэтому интерфейс и CLI считают одно и то же.

Почему поток: `webview.start()` занимает главный поток, а распознавание часовой
записи идёт минуты. Проверено, что `evaluate_js` можно звать из рабочего потока,
на этом и держится передача прогресса.
"""

from __future__ import annotations

import json
import subprocess
import sys
import threading
from collections.abc import Callable
from pathlib import Path
from typing import Any

from .. import weights
from ..config import WHISPER_MODELS, load_settings
from ..device import describe as describe_device
from ..documents import write_docx, write_pdf
from ..export import FORMATS, render, summary_to_markdown
from ..models import Transcript
from ..pipeline import Pipeline
from . import history, state
from .server import AudioServer

AUDIO_SUFFIXES = ("mp3", "wav", "m4a", "aac", "flac", "ogg", "opus", "mp4", "mov", "mkv", "webm")

# Форматы, которые пишут файл сами, в отличие от текстовых из `export`.
DOCUMENTS = {"pdf": write_pdf, "docx": write_docx}

# Пояснения к моделям — это текст интерфейса, поэтому живут здесь, а не в конфиге.
MODEL_HINTS = {
    "large-v3": "most accurate",
    "turbo": "faster, weaker on Russian",
    "medium": "a compromise",
    "small": "draft quality",
}


class Api:
    """Методы, вызываемые из JS через `window.pywebview.api`."""

    def __init__(self, audio_server: AudioServer) -> None:
        self._audio = audio_server
        self._window: Any = None
        self._lock = threading.Lock()
        self._busy = False
        self._source = None
        self._llm_instance = None
        # Флаг остановки закачки и поток, который её ведёт: живут от нажатия
        # «скачать» до её конца. Нужны оба — закрытие окна обязано её унести.
        self._stop_download: threading.Event | None = None
        self._download_thread: threading.Thread | None = None
        # Найденное обновление и защёлка на случай, когда оно уже поставлено.
        # Прежний код всё ещё в памяти, версия в ней старая, и без защёлки та же
        # проверка предлагала бы то же самое до перезапуска.
        self._update = None
        self._updated = False
        self._transcript: Transcript | None = None
        # Перевод живёт рядом с оригиналом, а не вместо него: сохранять нужно то,
        # что человек видит на экране, и уметь вернуться к исходному тексту.
        self._translated: Transcript | None = None
        self._showing = "original"

    def attach(self, window: Any) -> None:
        """Окно появляется после создания Api, поэтому связываем их отдельно."""
        self._window = window
        window.events.loaded += self._wire_drop

    def _wire_drop(self) -> None:
        """Подключает приём файлов перетаскиванием.

        Обязательно через DOM-события pywebview, а не через `addEventListener`
        в JS: полный путь к файлу подставляет сама библиотека в поле
        `pywebviewFullPath`, а в браузерный объект `File` он не попадает вовсе.
        """
        from webview.dom import DOMEventHandler

        element = self._window.dom.get_element("#drop")
        if element is None:
            return
        element.events.drop += DOMEventHandler(self._on_drop, prevent_default=True)

    def _on_drop(self, event: dict[str, Any]) -> None:
        files = event.get("dataTransfer", {}).get("files") or []
        path = files[0].get("pywebviewFullPath") if files else None
        if path:
            self._emit("dropped", path=path, name=Path(path).name)
        else:
            self._emit("error", message="could not read the path — use the file button instead")

    # --- вызовы из интерфейса ---

    def setup(self) -> dict[str, Any]:
        """Чем и на чём считаем — для шапки и списка моделей.

        Список приходит отсюда, а не из разметки: модели заданы в конфиге, и
        дублировать их в двух местах — верный способ разойтись.
        """
        settings = load_settings()
        return {
            "device": describe_device(),
            "models": [
                {"name": alias, "hint": MODEL_HINTS.get(alias, "")} for alias in WHISPER_MODELS
            ],
            "current": settings.asr_model,
            "llm": settings.llm_repo.rsplit("/", maxsplit=1)[-1],
            "output": str(state.output_dir(settings.output_dir)),
        }

    def weights_status(self) -> list[dict[str, Any]]:
        """Веса, нужные при текущих настройках, и что из них уже на диске."""
        return [
            {
                "repo": item.repo,
                "role": item.role,
                "title": item.title,
                "ready": item.ready,
                "size": item.size,
            }
            for item in weights.required(load_settings())
        ]

    def download_weights(self, repo: str) -> bool:
        """Качает веса. Возвращает False, если работа уже идёт."""
        with self._lock:
            if self._busy:
                return False
            self._busy = True
        self._stop_download = threading.Event()
        self._download_thread = threading.Thread(
            target=self._run_download, args=(repo,), daemon=True
        )
        self._download_thread.start()
        return True

    def pause_download(self) -> bool:
        """Останавливает закачку.

        Продолжить с того же места нельзя: загрузчик HuggingFace заводит
        временный файл заново, поэтому следующая попытка качает всё сначала.
        """
        if self._stop_download is None:
            return False
        self._stop_download.set()
        return True

    def shutdown(self, timeout: float = 10.0) -> None:
        """Уносит фоновую работу вместе с окном.

        Загрузчик — отдельный процесс, и сам по себе он закрытие окна переживёт:
        останется качать гигабайты в фон, которого уже никто не видит. Поэтому
        закачку останавливаем и дожидаемся, пока поток добьёт своего потомка.
        """
        if self._stop_download is not None:
            self._stop_download.set()
        if self._download_thread is not None:
            self._download_thread.join(timeout)

    def check_update(self) -> None:
        """Спрашивает, не вышла ли версия новее. Молчит, если нет.

        В отдельном потоке: запрос уходит при открытии окна, а сеть отвечает не
        мгновенно — на главном потоке это была бы пауза на пустом месте.
        """
        if not load_settings().update_check or self._updated:
            return
        threading.Thread(target=self._look_for_update, daemon=True).start()

    def install_update(self) -> bool:
        """Ставит найденное обновление. `False` — предлагать нечего."""
        if self._update is None or self._updated:
            return False
        threading.Thread(target=self._run_update, daemon=True).start()
        return True

    def _look_for_update(self) -> None:
        from .. import __version__, updates

        # Ошибки сюда не поднимаются: проверку никто не заказывал, и сообщать о
        # её неудаче — значит ругаться на человека за неработающий вайфай.
        if found := updates.available(__version__):
            self._update = found
            self._emit("update-found", version=found.version, notes=found.notes[:400])

    def _run_update(self) -> None:
        from .. import updates

        try:
            self._emit("progress", message=f"installing version {self._update.version}")
            updates.install(self._update)
        except updates.UpdateError as exc:
            self._emit("error", message=f"{exc}")
            # Отдельным событием, а не только ошибкой: строку в шапке надо
            # вернуть в исходное состояние, иначе она навсегда останется
            # «installing…» с погашенной кнопкой, и повторить будет нечем.
            self._emit(
                "update-failed",
                version=self._update.version,
                notes=self._update.notes[:400],
            )
            return
        # Запущенный процесс держит прежний код, и версия в памяти осталась
        # старой: без защёлки та же проверка предлагала бы обновление вечно.
        self._updated = True
        self._emit("update-installed", version=self._update.version)

    def pick_folder(self) -> str | None:
        """Выбор папки для готовых файлов. Выбор запоминается."""
        import webview

        chosen = self._window.create_file_dialog(webview.FOLDER_DIALOG)
        if not chosen:
            return None
        state.save(output_dir=chosen[0])
        return chosen[0]

    def reveal(self, path: str) -> None:
        """Показать файл в файловом менеджере: иначе путь надо копировать руками."""
        commands = {
            "darwin": ["open", "-R", path],
            "win32": ["explorer", "/select,", path.replace("/", "\\")],
        }
        command = commands.get(sys.platform, ["xdg-open", str(Path(path).parent)])
        try:
            subprocess.run(command, check=False)
        except OSError:
            # Показать не удалось — путь всё равно назван в уведомлении.
            pass

    def open_url(self, url: str) -> None:
        """Открывает ссылку в системном браузере.

        Внутри окна ей не место: webview здесь — интерфейс приложения, и уводить
        его на страницу значит потерять открытый транскрипт. Схема проверяется,
        чтобы через этот вызов нельзя было запустить что-нибудь ещё.
        """
        import webbrowser

        if url.startswith(("http://", "https://")):
            webbrowser.open(url)

    def pick_file(self) -> str | None:
        """Нативный диалог выбора файла."""
        import webview

        patterns = ";".join(f"*.{suffix}" for suffix in AUDIO_SUFFIXES)
        chosen = self._window.create_file_dialog(
            webview.OPEN_DIALOG,
            allow_multiple=False,
            file_types=(f"Audio and video ({patterns})", "All files (*.*)"),
        )
        return chosen[0] if chosen else None

    def transcribe(self, target: str, options: dict[str, Any]) -> bool:
        """Запускает распознавание. Возвращает False, если работа уже идёт."""
        if not target.strip():
            self._emit("error", message="no file or link given")
            return False
        # Модель выбирают тут же в форме, поэтому спрашиваем про выбранную, а не
        # про ту, что стоит в настройках.
        if not self._weights_ready(load_settings(asr_model=options.get("model") or None), "asr"):
            return False
        with self._lock:
            if self._busy:
                return False
            self._busy = True
        threading.Thread(target=self._run, args=(target.strip(), options), daemon=True).start()
        return True

    def translate(self, language: str) -> bool:
        return self._start_nlp("translate", language)

    def summarize(self, language: str) -> bool:
        return self._start_nlp("summarize", language)

    def show(self, which: str) -> bool:
        """Переключает показ между оригиналом и переводом."""
        target = self._translated if which == "translation" else self._transcript
        if target is None:
            return False
        self._showing = which
        self._emit("shown", which=which, **_payload(target))
        return True

    def export(self, fmt: str, directory: str | None = None) -> str:
        """Пишет на диск то, что показано на экране, и возвращает путь.

        Именно показанное: пользователь, глядящий на перевод, ждёт в файле
        перевод. Суффикс с языком не даёт переводу затереть оригинал.
        """
        current = self._current()
        if current is None:
            raise RuntimeError("nothing to save yet — transcribe a recording first")
        if fmt not in FORMATS and fmt not in DOCUMENTS:
            raise ValueError(f"unknown format: {fmt}")

        settings = load_settings()
        target_dir = Path(directory) if directory else state.output_dir(settings.output_dir)
        target_dir.mkdir(parents=True, exist_ok=True)

        stem = _stem(self._source)
        if self._showing == "translation":
            stem = f"{stem}.{current.language}"
        path = target_dir / f"{stem}.{fmt}"

        # Текстовые форматы отдают строку, бинарные умеют только писать файл.
        if fmt in DOCUMENTS:
            DOCUMENTS[fmt](current, path, title=_stem(self._source))
        else:
            path.write_text(render(current, fmt), encoding="utf-8")
        return str(path)

    def _current(self) -> Transcript | None:
        return self._translated if self._showing == "translation" else self._transcript

    # --- работа в потоке ---

    def _advance(self) -> Callable[[float], None]:
        """Отдаёт долю выполненного в окно, прореживая поток событий.

        Whisper отчитывается на каждом тридцатисекундном окне — на часовой записи
        это полторы сотни вызовов, и каждый переходит границу в webview. Глазу
        столько не нужно: шаг в один процент меняет полосу на видимую величину,
        а всё, что мельче, теряется в анимации перехода.
        """
        seen: float | None = None

        def advance(done: float) -> None:
            nonlocal seen
            # Первый отчёт проходит всегда, каким бы малым ни был: именно по нему
            # полоса появляется в окне. Ждать процента значило бы держать её
            # спрятанной до конца первого окна, а на длинной записи и дольше.
            if seen is None or done < seen or done - seen >= 0.01 or done >= 1.0:
                seen = done
                self._emit("advance", done=done)

        return advance

    def _run(self, target: str, options: dict[str, Any]) -> None:
        try:
            settings = load_settings(
                asr_model=options.get("model") or None,
                vad_enabled=False if options.get("no_vad") else None,
                num_speakers=options.get("speakers") or None,
            )
            pipeline = Pipeline(
                settings,
                notify=lambda text: self._emit("progress", message=text),
                advance=self._advance(),
            )

            source, transcript = pipeline.run(
                target,
                language=options.get("language") or None,
                diarize=bool(options.get("diarize")),
            )
            # Новая запись обнуляет перевод от предыдущей — иначе кнопка
            # «оригинал/перевод» покажет текст от другого файла.
            self._source, self._transcript = source, transcript
            self._translated, self._showing = None, "original"
            self._emit(
                "transcript",
                audio=self._audio.url_for(source.audio),
                title=source.title,
                **_payload(transcript),
            )

            entry = history.remember(
                transcript,
                origin=source.origin,
                title=source.title,
                audio=source.audio.name,
                topic=self._topic_writer(),
            )
            self._emit("history", entries=history.load(), latest=entry["key"])
        except Exception as exc:
            self._emit("error", message=f"{exc}")
        finally:
            self._release()

    def _run_download(self, repo: str) -> None:
        try:
            self._emit("weights-started", repo=repo)
            weights.download(
                repo,
                lambda done, total: self._emit(
                    "weights-progress", repo=repo, done=done, total=total
                ),
                # Чаще, чем меняется картинка: полоса тогда ползёт, а не прыгает.
                period=0.4,
                cancel=self._stop_download,
                notify=lambda text: self._emit("progress", message=text),
            )
            self._emit("weights-ready", repo=repo, items=self.weights_status())
        except weights.DownloadStopped:
            # Не отказ, а решение человека. Начатый кусок остаётся на диске,
            # но следующая попытка начнёт файл заново — так устроен загрузчик.
            self._emit("weights-paused", repo=repo, items=self.weights_status())
        except Exception as exc:
            self._emit("error", message=f"could not download the weights: {exc}")
        finally:
            self._stop_download = None
            self._release()

    def _weights_ready(self, settings, role: str) -> bool:
        """Не даёт начать работу, которая иначе молча ушла бы качать гигабайты.

        Панель весов при этом перерисовывается: человек должен увидеть, чего не
        хватает, а не только прочитать, что чего-то не хватает.
        """
        pending = [item for item in weights.missing(settings) if item.role == role]
        if not pending:
            return True
        self._emit("weights-needed", items=self.weights_status())
        self._emit("error", message=f"download the model first: {pending[0].title}")
        return False

    def _start_nlp(self, step: str, language: str) -> bool:
        if self._transcript is None:
            self._emit("error", message="transcribe a recording first")
            return False
        if not self._weights_ready(load_settings(), "llm"):
            return False
        with self._lock:
            if self._busy:
                return False
            self._busy = True
        threading.Thread(target=self._run_nlp, args=(step, language), daemon=True).start()
        return True

    def _run_nlp(self, step: str, language: str) -> None:
        try:
            settings = load_settings()
            pipeline = Pipeline(settings, notify=lambda text: self._emit("progress", message=text))
            assert self._transcript is not None

            if step == "translate":
                result = pipeline.translate(self._transcript, target_language=language)
                self._translated = self._transcript.model_copy(
                    update={"segments": result.segments, "language": result.target_language}
                )
                self._showing = "translation"
                self._emit("translation", **_payload(self._translated))
            else:
                summary = pipeline.summarize(self._transcript, language=language)
                self._emit(
                    "summary",
                    markdown=summary_to_markdown(summary, title=_stem(self._source)),
                )
        except Exception as exc:
            self._emit("error", message=f"{exc}")
        finally:
            self._release()

    def _topic_writer(self):
        """Функция, называющая тему записи, — только если LLM уже скачана.

        Иначе первое же распознавание потянуло бы гигабайты весов ради строчки
        в списке. Без неё история подставит первую фразу записи.
        """
        settings = load_settings()
        if not _llm_ready(settings):
            return None

        def describe(text: str) -> str:
            from ..nlp import topic as make_topic

            return make_topic(text, self._llm())

        return describe

    def _llm(self):
        """Модель для служебных мелочей вроде темы записи.

        Собирается пайплайном, а не своим вызовом `create_llm`: у бэкендов
        появились адреса и ключи, и вторая точка сборки разъехалась бы с первой
        при первой же новой настройке.
        """
        if self._llm_instance is None:
            self._llm_instance = Pipeline(load_settings()).llm
        return self._llm_instance

    def history(self) -> list[dict]:
        """Список распознанных записей, свежие сверху."""
        return history.load()

    def open_history(self, key: str) -> bool:
        """Открывает сохранённый транскрипт. Исходный файл для этого не нужен."""
        transcript = history.open_entry(key)
        if transcript is None:
            self._emit("error", message="recording not found — it may have been deleted")
            return False

        entry = next((item for item in history.load() if item.get("key") == key), {})
        self._transcript, self._translated, self._showing = transcript, None, "original"
        self._source = _Origin(entry.get("title") or Path(transcript.source).stem)

        # Звука может не быть: кэш чистят, а транскрипт остаётся.
        audio = self._audio.root / str(entry.get("audio") or "")
        playable = bool(entry.get("audio")) and audio.exists()

        self._emit(
            "transcript",
            audio=self._audio.url_for(audio) if playable else "",
            title=self._source.title,
            **_payload(transcript),
        )
        return True

    def forget_history(self, key: str) -> list[dict]:
        history.forget(key)
        return history.load()

    def _release(self) -> None:
        with self._lock:
            self._busy = False
        self._emit("idle")

    def _emit(self, kind: str, **payload: Any) -> None:
        """Отправляет событие в окно. Данные уходят как JSON, а не как код."""
        if self._window is None:
            return
        message = json.dumps({"kind": kind, **payload}, ensure_ascii=False)
        try:
            self._window.evaluate_js(f"window.appEvent({message})")
        except Exception:
            # Окно закрыли, пока шёл долгий шаг — доставлять уже некому.
            pass


def _payload(transcript: Transcript) -> dict[str, Any]:
    return {
        "language": transcript.language,
        "duration": transcript.duration,
        "speakers": transcript.speakers,
        "segments": [
            {
                "start": segment.start,
                "end": segment.end,
                "text": segment.text,
                "speaker": segment.speaker,
            }
            for segment in transcript.segments
        ],
    }


class _Origin:
    """Заглушка источника для записи, открытой из истории: файла уже может не быть."""

    def __init__(self, title: str) -> None:
        self.title = title


def _llm_ready(settings) -> bool:
    """Скачаны ли веса LLM.

    Проверка нужна, чтобы обычное распознавание не потянуло гигабайты ради
    строчки «о чём запись». Для Ollama файлов на нашей стороне нет — там демон
    отвечает сам, а неудача перехватывается на уровне истории.
    """
    return settings.llm_backend != "mlx" or weights.is_ready(settings.llm_repo)


def _stem(source) -> str:
    """Имя файла без символов, ломающих путь."""
    title = getattr(source, "title", None) or "transcript"
    cleaned = "".join(ch if ch.isalnum() or ch in " -_.," else "_" for ch in title).strip()
    return (cleaned[:120] or "transcript").rstrip(". ")
