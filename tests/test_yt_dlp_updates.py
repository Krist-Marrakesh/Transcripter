"""Свежий yt-dlp и решатель, который он требует.

Сеть не трогается: подменяются ответы PyPI, метаданные окружения и сам uv. Ошибки
здесь дорогие и тихие. Неверно прочитанная версия — и yt-dlp не обновляется
никогда, а YouTube отвечает 403. Обновлённый без своего решателя — и из списка
форматов пропадает звук, а окно говорит «Requested format is not available».
"""

from __future__ import annotations

import json
import subprocess
import sys
from importlib.metadata import PackageNotFoundError

import httpx
import pytest

from transcriber import updates, yt_dlp_updates
from transcriber.yt_dlp_updates import FEED, PROJECT, RELEASE, YtDlpRelease


def feed(*versions: str) -> str:
    """Лента PyPI в том виде, в каком она приходит: версии — заголовками записей."""
    items = "".join(
        f"<item><title>{v}</title><link>https://pypi.org/project/yt-dlp/{v}/</link></item>"
        for v in versions
    )
    return (
        '<?xml version="1.0" encoding="UTF-8"?><rss version="2.0"><channel>'
        f"<title>PyPI recent updates for yt-dlp</title>{items}</channel></rss>"
    )


def described(version: str, solver: str | None = "0.9.0", *, yanked: bool = False) -> str:
    """Описание релиза: решатель прибит точной версией, и не в одной экстре."""
    requires = [
        'brotli; (implementation_name == "cpython") and extra == "default"',
        'requests<3,>=2.32.2; extra == "default"',
        'deno>=2.6.6; extra == "deno"',
    ]
    if solver:
        requires += [
            f'yt-dlp-ejs=={solver}; extra == "default"',
            f'yt-dlp-ejs=={solver}; extra == "pin"',
        ]
    info = {"version": version, "requires_dist": requires, "yanked": yanked}
    return json.dumps({"info": info, "releases": {}})


class PyPI:
    """Подменённый PyPI: отвечает по адресу и помнит, о чём его спросили."""

    def __init__(self, monkeypatch, answers: dict[str, str | int | Exception]) -> None:
        self.asked: list[str] = []

        def get(url, **kwargs):
            self.asked.append(url)
            answer = answers.get(url, 404)
            if isinstance(answer, Exception):
                raise answer
            status, text = (answer, "") if isinstance(answer, int) else (200, answer)
            return httpx.Response(status, text=text, request=httpx.Request("GET", url))

        monkeypatch.setattr(httpx, "get", get)


def release(version: str) -> str:
    return RELEASE.format(version=version)


def installed_as(monkeypatch, found: str | None | Exception) -> None:
    """Что стоит в окружении — без самого окружения."""

    def lookup(name: str):
        assert name == "yt-dlp"
        if isinstance(found, Exception):
            raise found
        return found

    monkeypatch.setattr(yt_dlp_updates, "version", lookup)


# --- сколько стоит обычный запуск ---


def test_an_ordinary_launch_asks_the_feed_and_nothing_else(monkeypatch):
    """Ради этого лента и заведена: килобайт вместо 264 КБ полного описания.

    Новее ничего нет — значит, и спрашивать описание релиза не о чем.
    """
    installed_as(monkeypatch, "2026.8.19")
    pypi = PyPI(monkeypatch, {FEED: feed("2026.9.16.232951.dev0", "2026.8.19", "2026.7.4")})

    assert yt_dlp_updates.outdated() is None
    assert pypi.asked == [FEED]


def test_a_newer_release_comes_with_the_solver_it_pins(monkeypatch):
    installed_as(monkeypatch, "2026.8.19")
    pypi = PyPI(
        monkeypatch,
        {FEED: feed("2026.9.20", "2026.8.19"), release("2026.9.20"): described("2026.9.20")},
    )

    assert yt_dlp_updates.outdated() == YtDlpRelease(version="2026.9.20", solver="0.9.0")
    assert pypi.asked == [FEED, release("2026.9.20")]


# --- что считать релизом ---


def test_a_nightly_newer_than_the_release_is_not_offered(monkeypatch):
    """Ночные сборки лежат в ленте рядом с релизами; ставить их никто не просил."""
    installed_as(monkeypatch, "2026.8.19")
    PyPI(
        monkeypatch,
        {
            FEED: feed("2026.9.22.232951.dev0", "2026.9.20"),
            release("2026.9.20"): described("2026.9.20"),
        },
    )

    assert yt_dlp_updates.outdated().version == "2026.9.20"


def test_the_newest_is_taken_not_the_first_listed(monkeypatch):
    """Лента идёт в порядке загрузки, а не версий."""
    installed_as(monkeypatch, "2026.7.4")
    PyPI(
        monkeypatch,
        {FEED: feed("2026.8.19", "2026.9.20"), release("2026.9.20"): described("2026.9.20")},
    )

    assert yt_dlp_updates.outdated().version == "2026.9.20"


def test_months_compare_as_numbers(monkeypatch):
    """Строкой `2026.10.2` оказалась бы раньше `2026.9.30` — и осень прошла бы без обновлений."""
    installed_as(monkeypatch, "2026.9.30")
    PyPI(monkeypatch, {FEED: feed("2026.10.2"), release("2026.10.2"): described("2026.10.2")})

    assert yt_dlp_updates.outdated().version == "2026.10.2"


def test_a_nightly_put_in_on_purpose_is_not_downgraded(monkeypatch):
    """Ночная сборка позже релиза того же дня — откатывать её к нему нельзя."""
    installed_as(monkeypatch, "2026.9.16.232951.dev0")
    pypi = PyPI(monkeypatch, {FEED: feed("2026.9.16")})

    assert yt_dlp_updates.outdated() is None
    assert pypi.asked == [FEED]


def test_a_nightly_older_than_the_release_gives_way_to_it(monkeypatch):
    installed_as(monkeypatch, "2026.8.30.232658.dev0")
    PyPI(monkeypatch, {FEED: feed("2026.9.16"), release("2026.9.16"): described("2026.9.16")})

    assert yt_dlp_updates.outdated().version == "2026.9.16"


# --- отозванные и неотвеченные ---


def test_a_withdrawn_release_gives_way_to_the_one_before(monkeypatch):
    """Лента отозванных не помечает — это знает только описание релиза."""
    installed_as(monkeypatch, "2026.8.19")
    PyPI(
        monkeypatch,
        {
            FEED: feed("2026.9.20", "2026.9.10", "2026.8.19"),
            release("2026.9.20"): described("2026.9.20", yanked=True),
            release("2026.9.10"): described("2026.9.10", solver="0.8.0"),
        },
    )

    assert yt_dlp_updates.outdated() == YtDlpRelease(version="2026.9.10", solver="0.8.0")


def test_withdrawn_down_to_ours_offers_nothing(monkeypatch):
    installed_as(monkeypatch, "2026.8.19")
    pypi = PyPI(
        monkeypatch,
        {
            FEED: feed("2026.9.20", "2026.8.19"),
            release("2026.9.20"): described("2026.9.20", yanked=True),
        },
    )

    assert yt_dlp_updates.outdated() is None
    assert pypi.asked == [FEED, release("2026.9.20")]


def test_an_unanswered_release_ends_the_search(monkeypatch):
    """Не ответил — не значит отозван: остальные ждали бы той же тишины, по таймауту на каждый."""
    installed_as(monkeypatch, "2026.8.19")
    pypi = PyPI(
        monkeypatch,
        {FEED: feed("2026.9.20", "2026.9.10"), release("2026.9.20"): 503},
    )

    assert yt_dlp_updates.outdated() is None
    assert pypi.asked == [FEED, release("2026.9.20")]


def test_a_feed_without_a_release_asks_the_full_document(monkeypatch):
    """Сорок ночных сборок подряд за два года не случалось, но ответ всё равно есть."""
    installed_as(monkeypatch, "2026.8.19")
    nightlies = [f"2026.9.{day}.232951.dev0" for day in range(1, 29)]
    pypi = PyPI(monkeypatch, {FEED: feed(*nightlies), PROJECT: described("2026.9.20")})

    assert yt_dlp_updates.outdated() == YtDlpRelease(version="2026.9.20", solver="0.9.0")
    assert pypi.asked == [FEED, PROJECT]


@pytest.mark.parametrize(
    "answer",
    [
        404,
        503,
        "<rss><channel><item><title>2026.9.20",
        httpx.ConnectError("нет сети"),
    ],
)
def test_a_feed_that_cannot_be_read_says_nothing(monkeypatch, answer):
    """Проверку никто не заказывал — её неудача не повод говорить в окне."""
    installed_as(monkeypatch, "2026.8.19")
    PyPI(monkeypatch, {FEED: answer})

    assert yt_dlp_updates.outdated() is None


@pytest.mark.parametrize("body", ["<html>сервис недоступен</html>", '{"info": null}', "{}"])
def test_a_description_that_is_not_one_says_nothing(monkeypatch, body):
    installed_as(monkeypatch, "2026.8.19")
    PyPI(monkeypatch, {FEED: feed("2026.9.20"), release("2026.9.20"): body})

    assert yt_dlp_updates.outdated() is None


# --- решатель ---


def test_both_are_asked_for_exactly():
    """Решатель «последний» не годится: он бывает новее того, что yt-dlp примет."""
    pinned = YtDlpRelease(version="2026.9.20", solver="0.9.0")

    assert pinned.requirements() == ["yt-dlp==2026.9.20", "yt-dlp-ejs==0.9.0"]


def test_a_release_that_pins_no_solver_goes_in_alone(monkeypatch):
    """Выдумать версию решателя хуже, чем оставить стоящую."""
    installed_as(monkeypatch, "2026.8.19")
    PyPI(
        monkeypatch,
        {FEED: feed("2026.9.20"), release("2026.9.20"): described("2026.9.20", solver=None)},
    )

    assert yt_dlp_updates.outdated().requirements() == ["yt-dlp==2026.9.20"]


@pytest.mark.parametrize(
    ("requirement", "solver"),
    [
        ('yt-dlp-ejs>=0.8; extra == "default"', None),
        ('yt_dlp_ejs==0.9.1; extra == "default"', "0.9.1"),
    ],
)
def test_only_an_exact_pin_is_a_solver_version(monkeypatch, requirement, solver):
    """Нестрогое требование — не версия, которую можно поставить; имя пишут по-разному."""
    body = json.loads(described("2026.9.20", solver=None))
    body["info"]["requires_dist"].append(requirement)
    installed_as(monkeypatch, "2026.8.19")
    PyPI(monkeypatch, {FEED: feed("2026.9.20"), release("2026.9.20"): json.dumps(body)})

    assert yt_dlp_updates.outdated().solver == solver


# --- что стоит сейчас ---


@pytest.mark.parametrize("found", [PackageNotFoundError("yt-dlp"), None])
def test_a_missing_or_broken_yt_dlp_is_to_be_installed(monkeypatch, found):
    """Так выглядит установка, оборванная закрытым окном: пакета нет или нет метаданных.

    Следующий запуск должен её починить, а не споткнуться о неё.
    """
    installed_as(monkeypatch, found)
    PyPI(monkeypatch, {FEED: feed("2026.9.20"), release("2026.9.20"): described("2026.9.20")})

    assert yt_dlp_updates.installed() == "0"
    assert yt_dlp_updates.outdated().version == "2026.9.20"


# --- установка ---


def test_the_installation_asks_uv_for_both_pinned_and_nothing_more(monkeypatch):
    """Экстра `default` принесла бы requests, websockets и ещё четыре пакета."""
    asked: list[list[str]] = []
    monkeypatch.setattr(updates, "uv", lambda: "uv")
    monkeypatch.setattr(
        updates.subprocess,
        "run",
        lambda command, **kwargs: asked.append(command) or subprocess.CompletedProcess(command, 0),
    )

    yt_dlp_updates.install(YtDlpRelease(version="2026.9.20", solver="0.9.0"))

    (command,) = asked
    assert command[:3] == ["uv", "pip", "install"]
    assert "--python" in command
    assert command[-2:] == ["yt-dlp==2026.9.20", "yt-dlp-ejs==0.9.0"]
    assert not any("[" in part for part in command)


def test_the_next_import_sees_the_new_files(monkeypatch):
    """Каталог, прочитанный до установки, не знает о файлах, которые она принесла."""
    forgotten: list[bool] = []
    monkeypatch.setattr(updates, "uv", lambda: "uv")
    monkeypatch.setattr(
        updates.subprocess, "run", lambda command, **kw: subprocess.CompletedProcess(command, 0)
    )
    monkeypatch.setattr(
        yt_dlp_updates.importlib, "invalidate_caches", lambda: forgotten.append(True)
    )

    yt_dlp_updates.install(YtDlpRelease(version="2026.9.20"))

    assert forgotten == [True]


def test_a_refused_installation_names_the_version(monkeypatch):
    monkeypatch.setattr(updates, "uv", lambda: "uv")
    monkeypatch.setattr(
        updates.subprocess,
        "run",
        lambda command, **kw: subprocess.CompletedProcess(
            command, 2, stderr="error: Failed to fetch `yt-dlp==2026.9.20`\n"
        ),
    )

    with pytest.raises(updates.UpdateError, match="yt-dlp 2026.9.20") as refused:
        yt_dlp_updates.install(YtDlpRelease(version="2026.9.20"))

    assert "Failed to fetch" in str(refused.value)


def test_a_process_that_runs_yt_dlp_already_needs_a_restart(monkeypatch):
    """Импортированный модуль остаётся прежним, что бы ни лежало потом на диске."""
    monkeypatch.setitem(sys.modules, "yt_dlp", object())

    assert yt_dlp_updates.imported() is True


def test_a_process_that_never_opened_a_link_takes_the_new_one(monkeypatch):
    monkeypatch.delitem(sys.modules, "yt_dlp", raising=False)

    assert yt_dlp_updates.imported() is False
