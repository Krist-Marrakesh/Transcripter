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
from ..pipeline import LLMSlot, Pipeline
from ..report import Report, Stopped
from . import history, state
from .server import AudioServer

AUDIO_SUFFIXES = ("mp3", "wav", "m4a", "aac", "flac", "ogg", "opus", "mp4", "mov", "mkv", "webm")

# Форматы, которые пишут файл сами, в отличие от текстовых из `export`.
DOCUMENTS = {"pdf": write_pdf, "docx": write_docx}

# Что можно дописать к готовому транскрипту. Порядок здесь — порядок работы:
# имена раньше формул, потому что имена меняют подписи, а формулы текст, и
# читать в журнале «назвал троих, выписал сорок формул» естественнее наоборот.
ENRICHMENTS = ("names", "formulas")

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
        # Модель у окна одна на всё время его жизни. Раньше её строил каждый
        # запуск, и второй перевод читал девятнадцать гигабайт с диска заново,
        # а на время чтения в памяти лежали обе копии.
        #
        # Настройки для неё берутся один раз и здесь: кнопкой в окне меняются
        # папка, токен и бэкенд разметки, а модель и её адрес — только через
        # `.env`, который читают на запуске.
        self._llm_slot = LLMSlot(chosen_settings())
        # Флаг остановки закачки и поток, который её ведёт: живут от нажатия
        # «скачать» до её конца. Нужны оба — закрытие окна обязано её унести.
        self._stop_download: threading.Event | None = None
        self._download_thread: threading.Thread | None = None
        # Найденное обновление и защёлка на случай, когда оно уже поставлено.
        # Прежний код всё ещё в памяти, версия в ней старая, и без защёлки та же
        # проверка предлагала бы то же самое до перезапуска.
        self._update = None
        self._updated = False
        # Флаг остановки распознавания: живёт от нажатия «Transcribe» до конца
        # работы. Отдельный от закачки — это разные работы с разной ценой обрыва.
        self._stop_job: threading.Event | None = None
        self._transcript: Transcript | None = None
        # Перевод живёт рядом с оригиналом, а не вместо него: сохранять нужно то,
        # что человек видит на экране, и уметь вернуться к исходному тексту.
        self._translated: Transcript | None = None
        self._showing = "original"
        # Что уже сохранено из этой записи и что из сохранённого успело устареть.
        # Дописанные имена и формулы меняют показанное, но не файл на диске, и
        # человеку неоткуда узнать, что его надо переписать: окно выглядит
        # обновившимся, а папка — нет.
        self._saved: list[str] = []
        self._stale: list[str] = []
        # Готовое саммари держится здесь, а не только на экране: оно стоит минут
        # работы модели, и терять его при закрытии окна не за что.
        self._summary: str | None = None

    def _claim(self) -> bool:
        """Занимает приложение под одну работу. `False` — уже занято.

        Одна за раз, и это не ограничение реализации: распознавание упирается в
        GPU, и две параллельные шли бы вдвое медленнее каждая, а не быстрее
        вместе. Проверка и захват под одним замком — между ними не должно
        помещаться второе нажатие.
        """
        with self._lock:
            if self._busy:
                return False
            self._busy = True
            return True

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
        settings = chosen_settings()
        return {
            "device": describe_device(),
            "models": [
                {"name": alias, "hint": MODEL_HINTS.get(alias, "")} for alias in WHISPER_MODELS
            ],
            "current": settings.asr_model,
            "llm": settings.llm_repo.rsplit("/", maxsplit=1)[-1],
            "output": str(state.output_dir(settings.output_dir)),
            "speakers": self._speakers(settings),
        }

    def _speakers(self, settings) -> dict[str, Any]:
        """Чем размечаются спикеры и надо ли предложить что-то получше.

        Предложение выводится из отсутствия токена, а не из флажка «уже
        показывали»: иначе человек, стерший токен, никогда бы его больше не
        увидел. Отказ обратим, токен его перебивает.

        Токен берётся из настроек, а не только из `app.json`: положивший его в
        `.env` уже всё сделал, и предлагать ему настроить то, что настроено, —
        значит показывать, что мы его не заметили.
        """
        return {
            "backend": settings.diarization_backend,
            "offer": not settings.hf_token and not state.declined_speakers(),
            "model": settings.diarization_model,
        }

    def save_token(self, token: str) -> dict[str, Any]:
        """Принимает токен и переводит разметку на pyannote.

        Проверять токен запросом не станем: он понадобится не сейчас, а на
        разметке, и там же честно скажет, если не подошёл.
        """
        clean = (token or "").strip()
        if not clean:
            return self._speakers(chosen_settings())
        state.save(hf_token=clean, diarization_backend="pyannote", speakers_declined="no")
        return self._speakers(chosen_settings())

    def decline_speakers(self) -> dict[str, Any]:
        """Отказ от pyannote: размечать будет sherpa, и больше не спрашиваем."""
        state.save(speakers_declined="yes", diarization_backend="sherpa")
        return self._speakers(chosen_settings())

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
            for item in weights.required(chosen_settings())
        ]

    def download_weights(self, repo: str) -> bool:
        """Качает веса. Возвращает False, если работа уже идёт."""
        if not self._claim():
            return False
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
        if not chosen_settings().update_check or self._updated:
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
        if not self._weights_ready(chosen_settings(asr_model=options.get("model") or None), "asr"):
            return False
        # Про модель для дописывания спрашиваем здесь же, а не в конце работы:
        # узнать, что её нет, после трёх минут распознавания — узнать поздно.
        if any(step in ENRICHMENTS for step in (options.get("add") or ())):
            if not self._weights_ready(chosen_settings(), "llm"):
                return False
        if not self._claim():
            return False
        self._stop_job = threading.Event()
        threading.Thread(target=self._run, args=(target.strip(), options), daemon=True).start()
        return True

    def stop(self) -> bool:
        """Останавливает идущую работу — любую. `False` — останавливать нечего.

        Любую, потому что кнопка одна: пока что-то идёт, она называется «Stop»,
        и человеку неоткуда знать, что распознавание передумать можно, а перевод
        часовой лекции — нет.

        Не отмена, а пауза: распознанные порции остаются в кэше, переведённые
        куски — нет, но и то и другое переигрывается без потери исходного. Ничего
        не разрушается, поэтому подтверждения не спрашиваем.
        """
        if self._stop_job is None:
            return False
        self._stop_job.set()
        # Строкой в журнал, а не молча: начатый кусок не прерывается на полуслове,
        # и до его конца окно выглядит так же, как до нажатия. Без объяснения эти
        # полминуты читаются как зависшее приложение.
        self._emit("progress", message="stopping — the piece already started is being finished")
        return True

    def translate(self, language: str) -> bool:
        return self._start_nlp("translate", language)

    def summarize(self, language: str) -> bool:
        return self._start_nlp("summarize", language)

    def enrich(self, steps: list[str]) -> bool:
        """Дописывает к транскрипту выбранное: имена спикеров, формулы.

        Одной работой, а не по кнопке на каждое: модель поднимается один раз, а
        шаги друг друга не трогают — имена живут отображением на транскрипте,
        формулы полями сегментов. Отдельным нажатием от распознавания, потому
        что это стоит загрузки основной модели, и платить за неё должен тот,
        кто просил.
        """
        if self._transcript is None:
            self._emit("error", message="transcribe a recording first")
            return False

        wanted = tuple(step for step in ENRICHMENTS if step in steps)
        if not wanted:
            self._emit("error", message="choose what to add first")
            return False
        # Не отказ всей работе: формулы имени спикера не требуют. О самом отказе
        # скажет `_enrich`, туда же смотрит и путь из распознавания.
        if wanted == ("names",) and not self._transcript.speakers:
            self.report_line("no speakers are labelled, so there is nobody to name")
            return False

        return self._start_nlp("enrich", "", wanted)

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

        settings = chosen_settings()
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

        if str(path) not in self._saved:
            self._saved.append(str(path))
        # Переписали — значит этот файл снова совпадает с показанным, даже если
        # соседние всё ещё отстают.
        self._stale = [old for old in self._stale if old != str(path)]
        self._emit("stale", files=[Path(old).name for old in self._stale])
        return str(path)

    def save_summary(self, directory: str | None = None) -> str:
        """Пишет саммари отдельным файлом и возвращает путь.

        Отдельным, а не вместе с транскриптом: это другая вещь, и подменять ею
        расшифровку никто не просил. Имя с суффиксом `.summary`, чтобы соседний
        `md` с самим транскриптом не затёрся.
        """
        if self._summary is None:
            raise RuntimeError("nothing to save yet — ask for a summary first")

        settings = chosen_settings()
        target_dir = Path(directory) if directory else state.output_dir(settings.output_dir)
        target_dir.mkdir(parents=True, exist_ok=True)
        path = target_dir / f"{_stem(self._source)}.summary.md"
        path.write_text(self._summary, encoding="utf-8")
        return str(path)

    def _mark_stale(self) -> list[str]:
        """Объявляет всё сохранённое отставшим и называет файлы.

        Зовётся там, где показанное изменилось: сам файл на диске от этого не
        меняется, и без такой отметки человек уносит с собой прошлую версию,
        уверенный, что унёс нынешнюю.
        """
        self._stale = list(self._saved)
        return [Path(old).name for old in self._stale]

    def _forget_saved(self) -> None:
        """Новая запись — новый счёт: чужие файлы устареть от неё не могли.

        Саммари уходит вместе с ними: оно про прошлую запись, и предлагать его
        сохранить рядом с новой значило бы записать не то.
        """
        self._saved, self._stale = [], []
        self._summary = None

    def _current(self) -> Transcript | None:
        return self._translated if self._showing == "translation" else self._transcript

    # --- работа в потоке ---

    def _report(self, cancel: threading.Event | None = None) -> Report:
        """Как шаг пайплайна разговаривает с окном.

        Собирается на каждый запуск, потому что прореживание прогресса держит
        своё состояние: у второй записи оно должно начинаться с нуля.
        """
        return Report(
            say=lambda text: self._emit("progress", message=text),
            at=self._advance(),
            cancel=cancel,
        )

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
            settings = chosen_settings(
                asr_model=options.get("model") or None,
                vad_enabled=False if options.get("no_vad") else None,
                num_speakers=options.get("speakers") or None,
            )
            pipeline = Pipeline(settings, self._report(self._stop_job), self._llm_slot)

            source, transcript = pipeline.run(
                target,
                language=options.get("language") or None,
                diarize=bool(options.get("diarize")),
            )
            # Новая запись обнуляет перевод от предыдущей — иначе кнопка
            # «оригинал/перевод» покажет текст от другого файла.
            self._source, self._transcript = source, transcript
            self._translated, self._showing = None, "original"
            self._forget_saved()

            # Дописываем до того, как запись уйдёт в историю и на экран: тогда
            # сохранённая копия сразу с именами и формулами, а показанное не
            # успевает устареть у человека на глазах. Здесь же и модель уже
            # поднята, поэтому тема записи достанется истории настоящей.
            wanted = tuple(step for step in ENRICHMENTS if step in (options.get("add") or ()))
            if wanted:
                self._enrich(pipeline, wanted, quiet=True)
            transcript = self._transcript

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
        except Stopped:
            # Не отказ, а решение человека. Распознанные порции лежат в кэше, и
            # следующий запуск на этой записи продолжит с того же места, поэтому
            # говорим «остановлено», а не «прервано».
            self._emit("stopped", kept=True)
        except Exception as exc:
            self._emit("error", message=f"{exc}")
        finally:
            self._stop_job = None
            self._release()

    def _run_download(self, repo: str) -> None:
        try:
            self._emit("weights-started", repo=repo)
            weights.download(
                repo,
                lambda done, total: self._emit(
                    "weights-progress", repo=repo, done=done, total=total
                ),
                report=self._report(self._stop_download),
                # Чаще, чем меняется картинка: полоса тогда ползёт, а не прыгает.
                period=0.4,
            )
            self._emit("weights-ready", repo=repo, items=self.weights_status())
        except Stopped:
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

    def _start_nlp(self, step: str, language: str, wanted: tuple[str, ...] = ()) -> bool:
        if self._transcript is None:
            self._emit("error", message="transcribe a recording first")
            return False
        if not self._weights_ready(chosen_settings(), "llm"):
            return False
        if not self._claim():
            return False
        # Тот же флаг, что у распознавания: работа в окне одна за раз, и кнопка,
        # которая её останавливает, тоже одна.
        self._stop_job = threading.Event()
        threading.Thread(target=self._run_nlp, args=(step, language, wanted), daemon=True).start()
        return True

    def _run_nlp(self, step: str, language: str, wanted: tuple[str, ...] = ()) -> None:
        try:
            settings = chosen_settings()
            pipeline = Pipeline(settings, self._report(self._stop_job), self._llm_slot)
            assert self._transcript is not None

            match step:
                case "translate":
                    result = pipeline.translate(self._transcript, target_language=language)
                    self._translated = self._transcript.model_copy(
                        update={"segments": result.segments, "language": result.target_language}
                    )
                    self._showing = "translation"
                    self._emit("translation", **_payload(self._translated))
                case "enrich":
                    self._enrich(pipeline, wanted)
                case _:
                    summary = pipeline.summarize(self._transcript, language=language)
                    self._summary = summary_to_markdown(summary, title=_stem(self._source))
                    self._emit("summary", markdown=self._summary)
        except Stopped:
            # То же, что и у распознавания: решение человека, а не отказ. Здесь
            # сделанное не сохраняется — куски перевода без остальных бесполезны, —
            # но исходный транскрипт цел, и запустить заново можно когда угодно.
            self._emit("stopped", kept=False)
        except Exception as exc:
            self._emit("error", message=f"{exc}")
        finally:
            self._stop_job = None
            self._release()

    def _enrich(self, pipeline: Pipeline, wanted: tuple[str, ...], *, quiet: bool = False) -> None:
        """Дописывает выбранное и рассказывает, что из этого вышло.

        Шаги идут подряд по одному транскрипту и не мешают друг другу: имена —
        отображение на самом транскрипте, формулы — поля сегментов, и каждый шаг
        копирует то, что сделал предыдущий.

        `quiet` — когда это часть распознавания: показывать там нечего, запись
        ещё не появилась на экране, и перерисовывать её незачем. Сказанное всё
        равно попадёт в журнал строкой прогресса.
        """
        assert self._transcript is not None
        done: list[str] = []

        if "names" in wanted and not self._transcript.speakers:
            self.report_line("no speakers are labelled, so there is nobody to name")
            wanted = tuple(step for step in wanted if step != "names")

        if "names" in wanted:
            self._transcript = pipeline.name_speakers(self._transcript)
            names = self._transcript.names
            # Перевод получает те же имена: он копия того же транскрипта, и
            # разойтись подписи в двух половинах одного экрана не должны.
            if self._translated is not None:
                self._translated = self._translated.model_copy(update={"names": names})
            # Сохранённая копия переписывается, иначе открытая заново запись
            # снова окажется безымянной, а человек решит, что кнопка не сработала.
            history.keep_names(self._transcript)
            done.append(
                f"named {len(names)} of {len(self._transcript.speakers)} speakers"
                if names
                else "no names are said in the recording"
            )

        if "formulas" in wanted:
            self._transcript = pipeline.read_formulas(self._transcript)
            written = sum(len(s.formulas) for s in self._transcript.segments)
            done.append(
                f"{written} formulas written out in LaTeX"
                if written
                else "no formulas were spoken here"
            )

        for line in done:
            self.report_line(line)
        if quiet:
            return

        # Показанное могло измениться, поэтому перерисовывается ровно то, на что
        # человек сейчас смотрит, — а не то, что мы считаем главным.
        shown = self._translated if self._showing == "translation" else self._transcript
        assert shown is not None
        self._emit("enriched", done=done, stale=self._mark_stale(), **_payload(shown))

    def report_line(self, text: str) -> None:
        """Строка в журнал окна — то же, чем говорят шаги пайплайна."""
        self._emit("progress", message=text)

    def _topic_writer(self):
        """Функция, называющая тему записи, — только если модель уже в памяти.

        Тема украшает список истории, и поднимать ради неё модель не за что.
        Замерено на прогоне, где всё остальное взято из кэша, так что цена
        принадлежит одной этой строчке: 18.5 ГБ у аллокатора MLX и 8.9 секунды
        против 0.0 ГБ и 0.1 секунды. Восемь секунд — тот самый лаг в конце
        распознавания, а восемнадцать гигабайт на машине с шестнадцатью не
        помещаются вовсе, и всё это ради сорока восьми токенов.

        Кто уже переводил или просил саммари, получит настоящую тему даром;
        остальным история подставит первую фразу записи, и это честнее
        ожидания.
        """
        if not self._llm_slot.loaded:
            return None

        def describe(text: str) -> str:
            from ..nlp import topic as make_topic

            return make_topic(text, self._llm_slot.get())

        return describe

    def history(self) -> list[dict]:
        """Список распознанных записей, свежие сверху."""
        return history.load()

    def open_history(self, key: str) -> bool:
        """Открывает сохранённый транскрипт. Исходный файл для этого не нужен.

        Пока идёт распознавание — отказ. Иначе на экране оказывается чужая
        запись, а идущая работа никак этого не отменяет: человек видит готовый
        текст там, где ждал свой, и считает, что приложение выдало не то.
        """
        if self._busy:
            self._emit("error", message="finish or stop the current recording first")
            return False

        transcript = history.open_entry(key)
        if transcript is None:
            self._emit("error", message="recording not found — it may have been deleted")
            return False

        entry = next((item for item in history.load() if item.get("key") == key), {})
        self._transcript, self._translated, self._showing = transcript, None, "original"
        self._source = _Origin(entry.get("title") or Path(transcript.source).stem)
        self._forget_saved()

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


def chosen_settings(**overrides: object):
    """Настройки с учётом выбранного мышью.

    Имя своё, а не `load_settings`: это не та же функция под тем же именем, а
    обёртка вокруг неё, и читающий должен видеть разницу, не сверяясь с
    импортами.

    Токен и бэкенд разметки живут в `app.json`, а не в `.env`: их вводят кнопкой,
    а файл настроек принадлежит проекту. Отброс `None` внутри означает, что
    «не выбирали» — это «оставить как в конфиге».
    """
    return load_settings(
        hf_token=state.token(), diarization_backend=state.diarization(), **overrides
    )


def _payload(transcript: Transcript) -> dict[str, Any]:
    # Окно показывает людей, а не ярлыки, — но только там, где имя нашлось.
    # Само отображение хранится отдельно от подписей, чтобы повторная разметка
    # его не унесла, поэтому свести их надо здесь, на границе показа.
    transcript = transcript.as_named()
    return {
        "language": transcript.language,
        "duration": transcript.duration,
        "speakers": transcript.speakers,
        # Чем размечено — вместе с самой разметкой. Оговорку про слитые короткие
        # реплики человек должен прочитать там, где смотрит на ярлыки, а не в
        # документации, до которой он дойдёт когда-нибудь потом.
        "labelled_by": chosen_settings().diarization_backend if transcript.speakers else "",
        "segments": [
            {
                "start": segment.start,
                "end": segment.end,
                "text": segment.text,
                "speaker": segment.speaker,
                "formulas": segment.formulas,
            }
            for segment in transcript.segments
        ],
    }


class _Origin:
    """Заглушка источника для записи, открытой из истории: файла уже может не быть."""

    def __init__(self, title: str) -> None:
        self.title = title


def _stem(source) -> str:
    """Имя файла без символов, ломающих путь."""
    title = getattr(source, "title", None) or "transcript"
    cleaned = "".join(ch if ch.isalnum() or ch in " -_.," else "_" for ch in title).strip()
    return (cleaned[:120] or "transcript").rstrip(". ")
