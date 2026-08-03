"""Локальная LLM — общий интерфейс и бэкенды.

По умолчанию mlx-lm: считает на Metal, не требует отдельного демона, веса
скачиваются с HuggingFace при первом запуске. Ollama доступна как альтернатива,
если она уже стоит в системе.
"""

from __future__ import annotations

import re
from typing import Literal, Protocol

# Рассуждающие модели оборачивают ход мысли в <think>…</think>. В ответ это
# попадать не должно.
_THINKING = re.compile(r"<think>.*?</think>\s*", re.DOTALL)
# Открывающий тег модель не пишет сама: его дописывает шаблон чата в конец
# промпта. Поэтому в ответе остаётся только рассуждение и закрывающий тег.
_DANGLING_THINKING = re.compile(r"\A.*?</think>\s*", re.DOTALL)


def strip_thinking(text: str) -> str:
    """Убирает ход мысли рассуждающей модели, оставляя один ответ."""
    return _DANGLING_THINKING.sub("", _THINKING.sub("", text)).strip()


class LLM(Protocol):
    """Контракт языковой модели: промпт на входе, готовый текст на выходе."""

    model: str

    def complete(
        self,
        prompt: str,
        *,
        system: str | None = None,
        max_tokens: int = 2048,
        temperature: float = 0.3,
    ) -> str: ...


class MLXLanguageModel:
    """Бэкенд на mlx-lm. Модель грузится лениво и живёт до конца процесса."""

    def __init__(self, model: str) -> None:
        self.model = model
        self._loaded: tuple[object, object] | None = None

    def _ensure_loaded(self) -> tuple[object, object]:
        if self._loaded is None:
            from mlx_lm import load

            self._loaded = load(self.model)
        return self._loaded

    def complete(
        self,
        prompt: str,
        *,
        system: str | None = None,
        max_tokens: int = 2048,
        temperature: float = 0.3,
    ) -> str:
        from mlx_lm import generate
        from mlx_lm.sample_utils import make_sampler

        model, tokenizer = self._ensure_loaded()
        messages = [
            *([{"role": "system", "content": system}] if system else []),
            {"role": "user", "content": prompt},
        ]
        formatted = tokenizer.apply_chat_template(
            messages,
            add_generation_prompt=True,
            tokenize=False,
            # Переводу и саммари ход мысли не нужен: на Qwen3.6 рассуждения дают
            # вчетверо большее время ответа и лезут в результат. Шаблоны, которые
            # флага не знают, лишнюю переменную просто игнорируют.
            enable_thinking=False,
        )

        raw = generate(
            model,
            tokenizer,
            prompt=formatted,
            max_tokens=max_tokens,
            sampler=make_sampler(temp=temperature),
            verbose=False,
        )
        return strip_thinking(raw)


class OllamaLanguageModel:
    """Бэкенд на Ollama — для тех, у кого демон уже поднят."""

    def __init__(self, model: str, host: str = "http://localhost:11434") -> None:
        self.model = model
        self._host = host
        self._client = None

    def _ensure_client(self):
        if self._client is None:
            from ollama import Client

            self._client = Client(host=self._host)
        return self._client

    def complete(
        self,
        prompt: str,
        *,
        system: str | None = None,
        max_tokens: int = 2048,
        temperature: float = 0.3,
    ) -> str:
        messages = [
            *([{"role": "system", "content": system}] if system else []),
            {"role": "user", "content": prompt},
        ]
        response = self._ensure_client().chat(
            model=self.model,
            messages=messages,
            options={"temperature": temperature, "num_predict": max_tokens},
        )
        return strip_thinking(response["message"]["content"])


class RemoteLLMError(RuntimeError):
    """Сервер с моделью недоступен или ответил не тем."""


class RemoteLanguageModel:
    """Модель на чужой машине, говорящая протоколом OpenAI.

    Этим протоколом отвечают vLLM, llama.cpp server, LM Studio, TGI и сама
    Ollama, поэтому одна реализация покрывает любой способ поднять модель на
    сервере. Приложение при этом остаётся локальным: наружу уходит только текст
    транскрипта, аудио не покидает машину.
    """

    def __init__(
        self,
        model: str,
        base_url: str,
        *,
        api_key: str | None = None,
        timeout: float = 300.0,
        client: object | None = None,
    ) -> None:
        self.model = model
        self._base = base_url.rstrip("/")
        self._api_key = api_key
        self._timeout = timeout
        # Соединение живёт между запросами: перевод часовой лекции — это десятки
        # запросов пачками, и заново открывать сессию на каждый незачем.
        self._http = client
        # Шаблон чата применяет сервер, поэтому выключатель рассуждений едет к
        # нему полем запроса. Понимают его не все: не понявший отвечает отказом,
        # и тогда мы повторяем запрос без него, а ход мысли срезаем на своей
        # стороне.
        self._ask_without_thinking = True

    def complete(
        self,
        prompt: str,
        *,
        system: str | None = None,
        max_tokens: int = 2048,
        temperature: float = 0.3,
    ) -> str:
        payload: dict[str, object] = {
            "model": self.model,
            "messages": [
                *([{"role": "system", "content": system}] if system else []),
                {"role": "user", "content": prompt},
            ],
            "max_tokens": max_tokens,
            "temperature": temperature,
        }
        if self._ask_without_thinking:
            payload["chat_template_kwargs"] = {"enable_thinking": False}

        response = self._post(payload)
        if response.status_code == 400 and self._ask_without_thinking:
            # Сервер не знает про параметры шаблона — дальше обходимся без них.
            self._ask_without_thinking = False
            payload.pop("chat_template_kwargs")
            response = self._post(payload)

        if response.status_code != 200:
            raise RemoteLLMError(self._explain(response))

        message = response.json()["choices"][0]["message"]
        # Рассуждение сервер может оставить внутри ответа, а может вынести в
        # отдельное поле — из ответа его убираем в любом случае.
        if content := strip_thinking(message.get("content") or ""):
            return content

        reasoning = message.get("reasoning") or message.get("reasoning_content")
        if reasoning:
            # Ответа нет вовсе: модель рассуждала, пока не кончился лимит. Пустая
            # строка выглядела бы как «перевод потерялся», поэтому говорим прямо.
            raise RemoteLLMError(
                f"{self._base} returned only a reasoning trace and no answer: the model thinks "
                "before replying and the server did not accept our request to skip it. "
                "Raise the token limit or disable thinking on the server side."
            )
        return ""

    def _session(self):
        import httpx

        if self._http is None:
            headers = {"Authorization": f"Bearer {self._api_key}"} if self._api_key else {}
            self._http = httpx.Client(
                base_url=self._base, headers=headers, timeout=self._timeout
            )
        return self._http

    def _post(self, payload: dict[str, object]):
        import httpx

        try:
            return self._session().post("/chat/completions", json=payload)
        except httpx.HTTPError as exc:
            raise RemoteLLMError(f"{self._base} is not responding: {exc}") from exc

    def _explain(self, response) -> str:
        """Отказ сервера с названием модели и тем, что он вообще обслуживает.

        Имя модели на сервере своё и с нашим коротким именем совпадает редко —
        без этого списка человек ищет причину в собственной машине.
        """
        detail = response.text.strip()[:200]
        available = ", ".join(self.available()) or "server did not list any"
        return (
            f"{self._base} rejected model '{self.model}' "
            f"(HTTP {response.status_code}: {detail}). Available: {available}"
        )

    def available(self) -> list[str]:
        """Что сервер обслуживает. Пустой список — спросить не удалось."""
        import httpx

        try:
            response = self._session().get("/models")
            return [item["id"] for item in response.json().get("data", [])]
        except (httpx.HTTPError, ValueError, KeyError, TypeError):
            # Диагностика необязательна: отказ и без неё будет назван.
            return []


def check_remote(base_url: str, *, api_key: str | None = None, timeout: float = 5.0) -> str | None:
    """Отвечает ли сервер с моделью. `None` — да, иначе причина отказа.

    Таймаут короткий намеренно: это справка о состоянии, а не работа, и ждать
    её дольше нескольких секунд незачем.
    """
    import httpx

    headers = {"Authorization": f"Bearer {api_key}"} if api_key else {}
    try:
        response = httpx.get(f"{base_url.rstrip('/')}/models", headers=headers, timeout=timeout)
    except httpx.HTTPError as exc:
        return f"{type(exc).__name__}: {exc}"
    if response.status_code != 200:
        return f"HTTP {response.status_code}"
    return None


def create_llm(
    kind: Literal["mlx", "ollama", "openai"],
    model: str,
    *,
    host: str = "",
    base_url: str = "",
    api_key: str | None = None,
    timeout: float = 300.0,
) -> LLM:
    match kind:
        case "mlx":
            return MLXLanguageModel(model)
        case "ollama":
            try:
                import ollama  # noqa: F401
            except ImportError as exc:
                raise ImportError(
                    'бэкенд ollama не установлен: uv pip install -e ".[ollama]"'
                ) from exc
            return OllamaLanguageModel(model, host or "http://localhost:11434")
        case "openai":
            if not base_url:
                raise ValueError(
                    "не задан адрес сервера с моделью: TRANSCRIPT_LLM_BASE_URL=http://хост:порт/v1"
                )
            return RemoteLanguageModel(model, base_url, api_key=api_key, timeout=timeout)
        case _:
            raise ValueError(f"неизвестный LLM-бэкенд: {kind}")
