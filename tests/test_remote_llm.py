"""Модель на чужой машине.

Сеть здесь поддельная: проверяется не то, что сервер работает, а то, что мы
правильно с ним разговариваем. Самое важное — выключатель рассуждений: локально
он уезжает в шаблон чата, а на сервере шаблон применяют без нас, и флаг обязан
ехать полем запроса. Без этого перевод возвращается с ходом мысли внутри.
"""

from __future__ import annotations

import json

import httpx
import pytest

from transcriber.nlp.llm import RemoteLanguageModel, RemoteLLMError, create_llm

URL = "http://server:8000/v1"


def answering(reply: str = "Готовый перевод.", *, status: int = 200, capture: list | None = None):
    """Сервер, который отвечает как OpenAI и запоминает, о чём его просили."""

    def handler(request: httpx.Request) -> httpx.Response:
        if capture is not None:
            capture.append(json.loads(request.content))
        if request.url.path.endswith("/models"):
            return httpx.Response(200, json={"data": [{"id": "Qwen/Qwen3.6-35B"}]})
        if status != 200:
            return httpx.Response(status, text="model not found")
        return httpx.Response(200, json={"choices": [{"message": {"content": reply}}]})

    return httpx.Client(base_url=URL, transport=httpx.MockTransport(handler))


def model(**kwargs) -> RemoteLanguageModel:
    return RemoteLanguageModel("qwen3.6-35b", URL, client=answering(**kwargs))


def test_returns_server_answer():
    assert model().complete("переведи") == "Готовый перевод."


def test_asks_server_to_skip_reasoning():
    """Локальный `enable_thinking=False` до сервера не доедет — нужен свой путь."""
    asked: list[dict] = []
    model(capture=asked).complete("переведи")

    assert asked[0]["chat_template_kwargs"] == {"enable_thinking": False}


def test_system_prompt_and_limits_reach_the_server():
    asked: list[dict] = []
    model(capture=asked).complete(
        "переведи", system="ты переводчик", max_tokens=64, temperature=0.1
    )

    sent = asked[0]
    assert sent["messages"][0] == {"role": "system", "content": "ты переводчик"}
    assert (sent["max_tokens"], sent["temperature"]) == (64, 0.1)


def test_reasoning_trace_is_stripped_from_remote_answer():
    """Сервер мог применить шаблон по-своему — сетка на своей стороне остаётся."""
    assert model(reply="Разбираю просьбу.\n</think>\n\nОтвет").complete("переведи") == "Ответ"


def test_server_without_template_kwargs_is_retried_without_them():
    """Не всякий сервер знает про параметры шаблона: отказ — не повод сдаваться."""
    seen: list[dict] = []

    def handler(request: httpx.Request) -> httpx.Response:
        body = json.loads(request.content)
        seen.append(body)
        if "chat_template_kwargs" in body:
            return httpx.Response(400, text="unknown field chat_template_kwargs")
        return httpx.Response(200, json={"choices": [{"message": {"content": "Ответ"}}]})

    client = httpx.Client(base_url=URL, transport=httpx.MockTransport(handler))
    remote = RemoteLanguageModel("qwen3.6-35b", URL, client=client)

    assert remote.complete("переведи") == "Ответ"
    assert len(seen) == 2
    # Второй запрос той же модели уже не тратит попытку впустую.
    remote.complete("ещё раз")
    assert "chat_template_kwargs" not in seen[-1]


def test_refusal_names_model_and_what_server_offers():
    """Имя модели на сервере своё; без подсказки причину ищут в своём железе."""
    with pytest.raises(RemoteLLMError) as failure:
        model(status=404).complete("переведи")

    message = str(failure.value)
    assert "qwen3.6-35b" in message
    assert "Qwen/Qwen3.6-35B" in message


def test_unreachable_server_is_reported_as_such():
    def refuse(request: httpx.Request) -> httpx.Response:
        raise httpx.ConnectError("connection refused")

    client = httpx.Client(base_url=URL, transport=httpx.MockTransport(refuse))

    with pytest.raises(RemoteLLMError, match="not responding"):
        RemoteLanguageModel("qwen3.6-35b", URL, client=client).complete("переведи")


def test_backend_requires_an_address():
    """Без адреса бэкенд бессмыслен, и сказать об этом надо сразу."""
    with pytest.raises(ValueError, match="TRANSCRIPT_LLM_BASE_URL"):
        create_llm("openai", "qwen3.6-35b")


def test_backend_is_built_from_settings():
    remote = create_llm("openai", "qwen3.6-35b", base_url=URL, api_key="secret", timeout=5.0)

    assert isinstance(remote, RemoteLanguageModel)
    assert remote.model == "qwen3.6-35b"


def test_answer_made_only_of_reasoning_is_refused():
    """Найдено на живом сервере: без выключателя рассуждений `content` не приходит.

    Модель успевает только подумать, ответ не помещается в лимит, и сообщение
    приезжает с одним `reasoning`. Пустая строка выглядела бы как потерянный
    перевод, поэтому это отказ с объяснением.
    """

    def only_thinking(request: httpx.Request) -> httpx.Response:
        return httpx.Response(
            200,
            json={"choices": [{"message": {"role": "assistant", "reasoning": "Here's a plan…"}}]},
        )

    client = httpx.Client(base_url=URL, transport=httpx.MockTransport(only_thinking))

    with pytest.raises(RemoteLLMError, match="only a reasoning trace"):
        RemoteLanguageModel("qwen3.6-35b", URL, client=client).complete("переведи")
