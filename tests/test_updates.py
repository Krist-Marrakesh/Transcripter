"""Обновление приложения из окна.

Сеть здесь не трогается: проверяется разбор ответа и решения, которые на нём
строятся. Ошибка в них стоит либо вечного предложения обновиться, либо тихого
отката к прежней версии — и то и другое человек видит, но объяснить не может.
"""

from __future__ import annotations

import json

import httpx
import pytest

from transcriber import updates


def answer(
    monkeypatch,
    payload: dict | None,
    *,
    status: int = 200,
    fails: Exception | None = None,
    body: str | None = None,
) -> None:
    """Подменяет ответ GitHub, не выходя в сеть."""

    def get(url, **kwargs):
        if fails is not None:
            raise fails
        content = body if body is not None else json.dumps(payload)
        return httpx.Response(status, text=content, request=httpx.Request("GET", url))

    monkeypatch.setattr(httpx, "get", get)


def release(version: str = "0.2.0", assets: list[str] | None = None) -> dict:
    names = ["Transcripter-macos.zip", "transcript-0.2.0-py3-none-any.whl"]
    return {
        "tag_name": f"v{version}",
        "body": "что нового",
        "assets": [
            {"name": name, "browser_download_url": f"https://example/{name}"}
            for name in (names if assets is None else assets)
        ],
    }


@pytest.mark.parametrize(
    ("later", "earlier"),
    [("0.2.0", "0.1.0"), ("0.1.1", "0.1.0"), ("1.0.0", "0.9.9"), ("0.1.10", "0.1.9")],
)
def test_versions_compare_as_numbers(later, earlier):
    """Строкой `0.1.10` оказалась бы раньше `0.1.9` — и обновление не предложили бы."""
    assert updates.newer(later, earlier) is True
    assert updates.newer(earlier, later) is False


def test_the_same_version_is_not_newer():
    assert updates.newer("0.1.0", "0.1.0") is False


def test_the_leading_v_of_a_tag_is_not_part_of_the_version():
    assert updates.parse("v1.2.3") == (1, 2, 3)


def test_a_tail_that_is_not_a_number_ends_the_version():
    """Предрелизы не публикуются, но упасть на них нельзя."""
    assert updates.parse("1.2.0rc1") == (1, 2, 0)


def test_a_newer_release_is_offered(monkeypatch):
    answer(monkeypatch, release("0.2.0"))

    found = updates.available("0.1.0")

    assert found is not None
    assert found.version == "0.2.0"
    assert found.wheel.endswith(".whl")


def test_the_current_version_is_not_offered(monkeypatch):
    answer(monkeypatch, release("0.1.0"))

    assert updates.available("0.1.0") is None


def test_an_older_release_is_not_offered(monkeypatch):
    """Установленное может быть новее опубликованного — откатывать нельзя."""
    answer(monkeypatch, release("0.1.0"))

    assert updates.available("0.2.0") is None


def test_the_wheel_is_picked_by_suffix(monkeypatch):
    """По имени искать нельзя: в нём версия, и первый же выпуск сломал бы поиск."""
    answer(monkeypatch, release("0.2.0", ["Transcripter.zip", "transcript-9.9.9-py3-none-any.whl"]))

    assert updates.available("0.1.0").wheel.endswith("transcript-9.9.9-py3-none-any.whl")


def test_a_release_without_a_package_is_not_offered(monkeypatch):
    """Иначе появилась бы кнопка, которой нечего ставить."""
    answer(monkeypatch, release("0.2.0", ["Transcripter-macos.zip"]))

    assert updates.available("0.1.0") is None


def test_a_repository_without_releases_says_nothing(monkeypatch):
    """Ровно то, что наш репозиторий отдаёт сейчас: 404, релизов ещё нет."""
    answer(monkeypatch, {"message": "Not Found"}, status=404)

    assert updates.latest() is None


def test_rate_limiting_says_nothing(monkeypatch):
    """GitHub ограничивает запросы без токена — это не отказ приложения."""
    answer(monkeypatch, {"message": "API rate limit exceeded"}, status=403)

    assert updates.latest() is None


def test_a_dead_network_says_nothing(monkeypatch):
    """Проверку никто не заказывал: ругаться на чужой вайфай не за что."""
    answer(monkeypatch, None, fails=httpx.ConnectError("нет сети"))

    assert updates.latest() is None


def test_a_failed_certificate_says_nothing(monkeypatch):
    """Найдено вживую: у framework-питона нет хранилища сертификатов.

    Через `urllib` проверка обновлений на такой машине не работала бы никогда,
    и выглядело бы это как «обновлений нет». Отсюда httpx со своим бандлом.
    """
    answer(monkeypatch, None, fails=httpx.ConnectError("CERTIFICATE_VERIFY_FAILED"))

    assert updates.latest() is None


def test_rubbish_instead_of_json_says_nothing(monkeypatch):
    answer(monkeypatch, None, body="<html>сервис недоступен</html>")

    assert updates.latest() is None


def test_the_bundled_uv_is_preferred(monkeypatch, tmp_path):
    """Внутри релиза он единственный: в окружении из `uv venv` нет даже pip."""
    binary = tmp_path / "uv"
    binary.write_text("", encoding="utf-8")
    monkeypatch.setenv("TRANSCRIPT_UV", str(binary))

    assert updates.uv() == str(binary)


def test_the_update_installs_what_the_launcher_installed(monkeypatch, tmp_path):
    """Регрессия: набор экстр был выписан дважды и мог разъехаться.

    Лаунчер ставил `[app,documents]`, а обновление из окна — `[app]`. Не падало
    только потому, что экстры добавляют и не убирают: первая же зависимость,
    добавленная в `documents`, пропала бы у всех, кто обновился кнопкой.
    """
    monkeypatch.setenv(updates.EXTRAS, "app,documents")
    wheel = tmp_path / "transcript-0.2.3-py3-none-any.whl"

    assert updates._asked_for(wheel) == f"{wheel}[app,documents]"


def test_without_a_launcher_the_wheel_is_asked_for_bare(monkeypatch, tmp_path):
    """Запуск из исходников: набор никто не называл, и выдумывать его не за что."""
    monkeypatch.delenv(updates.EXTRAS, raising=False)
    wheel = tmp_path / "transcript-0.2.3-py3-none-any.whl"

    assert updates._asked_for(wheel) == str(wheel)


def test_a_stale_path_falls_back_to_the_system(monkeypatch, tmp_path):
    """Бандл могли перенести или удалить — переменная переживёт это, файл нет."""
    monkeypatch.setenv("TRANSCRIPT_UV", str(tmp_path / "нет"))
    monkeypatch.setattr(updates.shutil, "which", lambda name: "/usr/local/bin/uv")

    assert updates.uv() == "/usr/local/bin/uv"


def test_without_uv_anywhere_it_says_so(monkeypatch):
    monkeypatch.delenv("TRANSCRIPT_UV", raising=False)
    monkeypatch.setattr(updates.shutil, "which", lambda name: None)

    with pytest.raises(updates.UpdateError, match="uv"):
        updates.uv()
