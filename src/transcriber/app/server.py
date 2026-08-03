"""Локальная раздача аудио в окно приложения.

WKWebView не даёт `<audio>` читать файлы по `file://` за пределами страницы —
проверено, элемент отдаёт ошибку `code=4`. Поэтому аудио уходит в окно по HTTP
с петлевого адреса.

Range-запросы обязательны: без них браузер тянет файл целиком, прежде чем
разрешить перемотку, а час лекции в 16 кГц моно — это около 115 МБ. Готового
обработчика с Range в stdlib нет, отсюда свой.
"""

from __future__ import annotations

import threading
from http import HTTPStatus
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from urllib.parse import quote, unquote, urlparse

_CHUNK = 1 << 16


class _AudioHandler(BaseHTTPRequestHandler):
    """Отдаёт файлы из одного каталога, поддерживая частичные запросы."""

    root: Path

    def do_GET(self) -> None:  # noqa: N802 — имя задано базовым классом
        target = self._resolve()
        if target is None:
            self.send_error(HTTPStatus.NOT_FOUND)
            return

        size = target.stat().st_size
        start, end = self._range(size)

        with target.open("rb") as handle:
            if start is None:
                self._send_headers(HTTPStatus.OK, size, size)
                self._copy(handle, size)
                return

            length = end - start + 1
            self._send_headers(HTTPStatus.PARTIAL_CONTENT, length, size, content_range=(start, end))
            handle.seek(start)
            self._copy(handle, length)

    def _resolve(self) -> Path | None:
        """Путь внутри корня. Всё, что пытается выйти наружу, отбрасывается."""
        name = unquote(urlparse(self.path).path).lstrip("/")
        if not name:
            return None
        candidate = (self.root / name).resolve()
        if not candidate.is_file() or self.root.resolve() not in candidate.parents:
            return None
        return candidate

    def _range(self, size: int) -> tuple[int | None, int]:
        """Разбирает заголовок Range. Возвращает (None, size-1), если его нет."""
        header = self.headers.get("Range", "")
        if not header.startswith("bytes="):
            return None, size - 1
        first, _, last = header.removeprefix("bytes=").partition("-")
        try:
            start = int(first) if first else 0
            end = int(last) if last else size - 1
        except ValueError:
            return None, size - 1
        return max(0, start), min(end, size - 1)

    def _send_headers(
        self,
        status: HTTPStatus,
        length: int,
        total: int,
        *,
        content_range: tuple[int, int] | None = None,
    ) -> None:
        self.send_response(status)
        self.send_header("Content-Type", "audio/wav")
        self.send_header("Content-Length", str(length))
        self.send_header("Accept-Ranges", "bytes")
        if content_range is not None:
            first, last = content_range
            self.send_header("Content-Range", f"bytes {first}-{last}/{total}")
        self.end_headers()

    def _copy(self, handle, length: int) -> None:
        remaining = length
        try:
            while remaining > 0 and (chunk := handle.read(min(_CHUNK, remaining))):
                self.wfile.write(chunk)
                remaining -= len(chunk)
        except (BrokenPipeError, ConnectionResetError):
            # Перемотка обрывает предыдущий запрос — это нормальный ход событий.
            pass

    def log_message(self, *args) -> None:
        """Молчим: обращения за аудио идут потоком и засоряют вывод."""


class AudioServer:
    """Раздаёт каталог с аудио на петлевом адресе.

    Порт выбирает система: фиксированный номер рано или поздно окажется занят
    чужим процессом.
    """

    def __init__(self, root: Path) -> None:
        self.root = root
        handler = type("Handler", (_AudioHandler,), {"root": root})
        self._server = ThreadingHTTPServer(("127.0.0.1", 0), handler)
        self._thread = threading.Thread(target=self._server.serve_forever, daemon=True)

    @property
    def origin(self) -> str:
        host, port = self._server.server_address[:2]
        return f"http://{host}:{port}"

    def url_for(self, audio: Path) -> str:
        """Ссылка на файл. Имя экранируется: в кэше это хэши, но не только."""
        return f"{self.origin}/{quote(audio.name)}"

    def start(self) -> None:
        self._thread.start()

    def stop(self) -> None:
        self._server.shutdown()
        self._server.server_close()
