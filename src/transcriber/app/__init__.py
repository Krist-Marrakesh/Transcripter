"""Десктопное окно приложения.

Окно — системный webview, а не свой браузер: на macOS это WKWebView, на Windows
WebView2. Отсюда вес порядка мегабайта вместо полутора сотен у Electron.

Тяжёлые импорты живут внутри `run`, чтобы `python -m transcriber.app --help` и
сам CLI не тянули GUI-стек.
"""

from __future__ import annotations

import sys
from datetime import datetime
from pathlib import Path

WEB = Path(__file__).parent / "web"


def _keep_a_log() -> None:
    """Gives output somewhere to go when there is no console to write to.

    A shortcut on Windows points at `pythonw.exe`, which has no console at all:
    `sys.stdout` and `sys.stderr` are None, `print` quietly does nothing, and a
    library writing to the stream directly dies on it. Whatever the reason a
    launch failed, from the outside it looks like nothing happened.

    On macOS the launcher script redirects the streams itself, so there is
    nothing to do here — the check simply does not fire.
    """
    if sys.stderr is not None:
        return

    from .. import paths

    log = paths.data_dir() / "launch.log"
    log.parent.mkdir(parents=True, exist_ok=True)
    # Line buffering: a crash must not take the last message down with it.
    stream = log.open("a", encoding="utf-8", buffering=1)
    sys.stdout = sys.stderr = stream
    print(f"--- запуск {datetime.now():%Y-%m-%d %H:%M:%S} ---")


def run(*, debug: bool = False) -> None:
    """Открывает окно и держит его до закрытия пользователем."""
    _keep_a_log()

    import webview

    from ..config import load_settings
    from . import single, state
    from .bridge import Api
    from .server import AudioServer

    # Вторая копия не открывается: у неё была бы своя история и свой признак
    # занятости при общих папке вывода и кэше. Вместо неё поднимаем открытое окно.
    if (opened := single.running()) is not None:
        if not single.focus(opened):
            print("окно уже открыто", file=sys.stderr)
        return

    settings = load_settings()

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
            webview.start(debug=debug)
    finally:
        # Сначала фоновая работа, потом раздача звука: закачка живёт отдельным
        # процессом и без этого пережила бы окно, а само окно закрывалось бы,
        # пока кто-то ещё держит сеть.
        api.shutdown()
        audio.stop()
