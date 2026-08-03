"""Локальная раздача аудио.

Range-запросы здесь не украшение: без них webview тянет файл целиком, прежде чем
разрешить перемотку, а час записи в 16 кГц моно — это около 115 МБ.
"""

from __future__ import annotations

import urllib.error
import urllib.request

import pytest

from transcriber.app.server import AudioServer

PAYLOAD = bytes(range(256)) * 8  # 2048 байт с заведомо различимым содержимым


@pytest.fixture
def server(tmp_path):
    (tmp_path / "запись.wav").write_bytes(PAYLOAD)
    instance = AudioServer(tmp_path)
    instance.start()
    yield instance
    instance.stop()


def fetch(url: str, headers: dict[str, str] | None = None):
    request = urllib.request.Request(url, headers=headers or {})
    return urllib.request.urlopen(request, timeout=5)


def test_serves_whole_file(server, tmp_path):
    response = fetch(server.url_for(tmp_path / "запись.wav"))

    assert response.status == 200
    assert response.read() == PAYLOAD
    # Без этого заголовка браузер даже не попробует запросить кусок.
    assert response.headers["Accept-Ranges"] == "bytes"


def test_range_returns_requested_slice(server, tmp_path):
    response = fetch(server.url_for(tmp_path / "запись.wav"), {"Range": "bytes=100-199"})

    assert response.status == 206
    assert response.read() == PAYLOAD[100:200]
    assert response.headers["Content-Range"] == f"bytes 100-199/{len(PAYLOAD)}"
    assert response.headers["Content-Length"] == "100"


def test_open_ended_range_runs_to_end(server, tmp_path):
    """Перемотка в конец записи просит «отсюда и до упора»."""
    response = fetch(server.url_for(tmp_path / "запись.wav"), {"Range": "bytes=2000-"})

    assert response.status == 206
    assert response.read() == PAYLOAD[2000:]


def test_range_beyond_end_is_clamped(server, tmp_path):
    response = fetch(server.url_for(tmp_path / "запись.wav"), {"Range": "bytes=2040-9999"})

    assert response.read() == PAYLOAD[2040:]
    assert response.headers["Content-Range"] == f"bytes 2040-{len(PAYLOAD) - 1}/{len(PAYLOAD)}"


def test_broken_range_header_falls_back_to_whole_file(server, tmp_path):
    response = fetch(server.url_for(tmp_path / "запись.wav"), {"Range": "bytes=abc-def"})

    assert response.status == 200
    assert response.read() == PAYLOAD


def test_unknown_file_is_not_found(server):
    with pytest.raises(urllib.error.HTTPError) as info:
        fetch(f"{server.origin}/missing.wav")
    assert info.value.code == 404


def test_escaping_the_root_is_refused(server):
    """Сервер слушает на петлевом адресе, но выход из каталога всё равно закрыт."""
    with pytest.raises(urllib.error.HTTPError) as info:
        fetch(f"{server.origin}/../../etc/passwd")
    assert info.value.code == 404
