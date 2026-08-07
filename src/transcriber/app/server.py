"""Serving audio to the application window from this machine.

WKWebView does not let `<audio>` read files over `file://` outside the page
directory — measured, the element answers with `code=4`. So the audio reaches the
window over HTTP from the loopback address.

Range requests are not optional: without them a browser pulls the whole file
before it will allow seeking, and an hour of lecture at 16 kHz mono is about
115 MB. The standard library has no handler that speaks Range, hence this one.
"""

from __future__ import annotations

import threading
from http import HTTPStatus
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from urllib.parse import quote, unquote, urlparse

_CHUNK = 1 << 16


class _AudioHandler(BaseHTTPRequestHandler):
    """Serves files from one directory, partial requests included."""

    root: Path

    def do_GET(self) -> None:  # noqa: N802 — the name is the base class's
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
        """A path inside the root. Anything reaching outside is refused."""
        name = unquote(urlparse(self.path).path).lstrip("/")
        if not name:
            return None
        candidate = (self.root / name).resolve()
        if not candidate.is_file() or self.root.resolve() not in candidate.parents:
            return None
        return candidate

    def _range(self, size: int) -> tuple[int | None, int]:
        """Parses the Range header. Returns (None, size-1) when there is none."""
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
            # Seeking cuts off the previous request — an ordinary turn of events.
            pass

    def log_message(self, *args) -> None:
        """Silence: audio requests come in a stream and would bury everything else."""


class AudioServer:
    """Serves a folder of audio on the loopback address.

    The system picks the port: a fixed number is one someone else's process will
    be holding sooner or later.
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
        """A link to the file. The name is escaped: cache names are hashes, mostly."""
        return f"{self.origin}/{quote(audio.name)}"

    def start(self) -> None:
        self._thread.start()

    def stop(self) -> None:
        self._server.shutdown()
        self._server.server_close()
