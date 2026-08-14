"""Десктопное окно приложения.

Окно — системный webview, а не свой браузер: на macOS это WKWebView, на Windows
WebView2. Отсюда вес порядка мегабайта вместо полутора сотен у Electron.

Тяжёлые импорты живут внутри `run`, чтобы `python -m transcriber.app --help` и
сам CLI не тянули GUI-стек.
"""

from __future__ import annotations

import subprocess
import sys
from datetime import datetime
from pathlib import Path
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from ..config import Settings

WEB = Path(__file__).parent / "web"
ASSETS = Path(__file__).parent / "assets"


def _window_icon() -> str | None:
    """The icon for the window, where nothing else gives it one.

    On macOS the bundle already carries `icon.icns` and `Info.plist` points at it,
    so the dock has the icon before any Python runs; a second one would only
    override it with the worse of the two.

    On Windows nobody gives it one. The shortcut's icon belongs to the shortcut
    and stops at the desktop, and pywebview falls back to whatever it can pull out
    of `sys.executable` — where a virtual environment built by uv keeps a
    trampoline with no icon resource at all. Left alone, the window shows the
    stock WinForms icon instead.
    """
    icon = ASSETS / "icon.ico"
    return str(icon) if sys.platform == "win32" and icon.exists() else None


def _console_of_our_own() -> int:
    """The console window opened for this application alone — 0 when there is none.

    A shortcut points at `pythonw.exe`, a program without a console, and the
    intention is a window with nothing standing behind it. In an environment built
    by uv that file is a trampoline rather than the interpreter, and the
    interpreter it starts is the console build: Windows gives it a console of its
    own, which then stands behind the interface for as long as the application
    runs. Measured here — the window is real, class `ConsoleWindowClass`, visible.

    Also zero when the console is somebody's terminal, which is the whole
    difficulty: the same command typed into PowerShell must keep its window. The
    two are told apart by the cursor. A shell has printed a prompt into its
    console before ever starting us, so the cursor has moved; a console opened for
    us alone has never been written to and stands at its origin.
    """
    if sys.platform != "win32":
        return 0

    import ctypes
    from ctypes import wintypes

    kernel32 = ctypes.WinDLL("kernel32", use_last_error=True)
    window = kernel32.GetConsoleWindow()
    if not window:
        return 0

    class _Point(ctypes.Structure):
        _fields_ = [("x", ctypes.c_short), ("y", ctypes.c_short)]

    class _Buffer(ctypes.Structure):
        _fields_ = [
            ("size", _Point),
            ("cursor", _Point),
            ("attributes", wintypes.WORD),
            ("window", wintypes.SMALL_RECT),
            ("maximum", _Point),
        ]

    state = _Buffer()
    # -11 is STD_OUTPUT_HANDLE. A redirected stream answers no to this, and that
    # is the right answer: output going to a file needs no window hidden.
    if not kernel32.GetConsoleScreenBufferInfo(kernel32.GetStdHandle(-11), ctypes.byref(state)):
        return 0
    return window if state.cursor.x == 0 and state.cursor.y == 0 else 0


def _hide_our_console() -> bool:
    """Hides that console, and says whether there was one to hide."""
    window = _console_of_our_own()
    if not window:
        return False

    import ctypes

    # 0 is SW_HIDE. The console keeps running behind it — the streams stay valid,
    # which is why the log below is opened whenever this succeeds.
    ctypes.WinDLL("user32").ShowWindow(window, 0)
    return True


def _keep_a_log(hidden: bool = False) -> None:
    """Gives output somewhere to go when there is nowhere visible to write to.

    Two ways to end up there. A launcher that starts `pythonw.exe` and gets the
    windowed interpreter leaves `sys.stdout` and `sys.stderr` as None: `print`
    quietly does nothing and a library writing to the stream directly dies on it.
    Or the streams are real but their console has just been hidden, and every
    line written goes to a window nobody will see again.

    Either way, whatever the reason a launch failed, from the outside it looks
    like nothing happened. On macOS the launcher script redirects the streams
    itself, so neither case fires.
    """
    if sys.stderr is not None and not hidden:
        return

    from .. import paths

    log = paths.data_dir() / "launch.log"
    log.parent.mkdir(parents=True, exist_ok=True)
    # Line buffering: a crash must not take the last message down with it.
    stream = log.open("a", encoding="utf-8", buffering=1)
    sys.stdout = sys.stderr = stream
    print(f"--- запуск {datetime.now():%Y-%m-%d %H:%M:%S} ---")


def _say_it_is_open() -> None:
    """Сообщает, что копия уже работает, когда поднять её окно не вышло.

    Поднять получается не всегда: делается это через System Events, а тем нужны
    права Универсального доступа, которых у приложения по умолчанию нет. Без
    сообщения второй запуск выглядит как сломанный ярлык — иконка подпрыгнула и
    ничего не произошло, — и человек жмёт по ней снова и снова.

    Уведомление, а не диалог: копия действительно открыта, и требовать за это
    нажатия кнопки не за что.
    """
    print("окно уже открыто", file=sys.stderr)
    if sys.platform != "darwin":
        return

    from ..subproc import quiet_flags

    # По-английски, как и остальные уведомления лаунчера.
    script = (
        'display notification "Already running — its window is under Cmd-Tab" '
        'with title "Transcripter"'
    )
    subprocess.run(["osascript", "-e", script], capture_output=True, check=False, **quiet_flags())


def _gather_weights(settings: Settings) -> None:
    """Собирает под приложение веса, скачанные его прежними версиями.

    Раньше они ложились в общие кэши машины, откуда ничего не удаляется вместе с
    приложением. Теперь папка внутри бандла есть, и уже скачанную модель незачем
    качать второй раз, чтобы она туда попала: переименование на том же диске
    ничего не стоит.
    """
    from .. import weights
    from ..diarize import sherpa_backend

    for item in weights.required(settings):
        # Кроме пары для разметки: она не из кэша HuggingFace, и переносит её
        # свой же бэкенд строкой ниже.
        if item.repo != weights.SHERPA:
            weights.adopt_model(item.repo)
    sherpa_backend.adopt()


def run(*, debug: bool = False) -> None:
    """Открывает окно и держит его до закрытия пользователем."""
    _keep_a_log(_hide_our_console())

    import webview

    from ..config import load_settings
    from . import single, state
    from .bridge import Api
    from .server import AudioServer

    # Вторая копия не открывается: у неё была бы своя история и свой признак
    # занятости при общих папке вывода и кэше. Вместо неё поднимаем открытое окно.
    if (opened := single.running()) is not None:
        if not single.focus(opened):
            _say_it_is_open()
        return

    settings = load_settings()
    _gather_weights(settings)

    # Папка для готовых файлов создаётся на старте, а не при первом сохранении:
    # её должно быть видно в Finder до того, как понадобится. Неудача сюда не
    # пускается: упавший `mkdir` до открытия окна выглядит как «иконка
    # подпрыгнула и погасла», а сменить папку можно только в самом окне.
    output = state.output_dir(settings.output_dir)
    if not state.ensure(output):
        print(f"не удалось создать папку {output} — выбери другую в окне", file=sys.stderr)

    # Каталог соответствует неймспейсу, куда `ingest` кладёт нормализованный WAV.
    audio = AudioServer(settings.cache_dir / "audio")
    audio.start()

    api = Api(audio)
    window = webview.create_window(
        "Транскрибатор",
        url=str(WEB / "index.html"),
        js_api=api,
        width=1080,
        height=760,
        min_size=(720, 520),
    )
    api.attach(window)

    try:
        with single.claim():
            webview.start(debug=debug, icon=_window_icon())
    finally:
        # Сначала фоновая работа, потом раздача звука: закачка живёт отдельным
        # процессом и без этого пережила бы окно, а само окно закрывалось бы,
        # пока кто-то ещё держит сеть.
        api.shutdown()
        audio.stop()
