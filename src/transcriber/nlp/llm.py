"""The local LLM — one interface and the backends behind it.

mlx-lm by default: it computes on Metal, needs no daemon of its own, and its
weights come from HuggingFace on the first run. Ollama is there as an alternative
for anyone who already has it installed.
"""

from __future__ import annotations

import re
from typing import Literal, Protocol

# Reasoning models wrap their train of thought in <think>…</think>. None of it
# belongs in the answer.
_THINKING = re.compile(r"<think>.*?</think>\s*", re.DOTALL)
# The model does not write the opening tag itself: the chat template appends it
# to the end of the prompt. So what comes back is the reasoning and a closing tag.
_DANGLING_THINKING = re.compile(r"\A.*?</think>\s*", re.DOTALL)


def strip_thinking(text: str) -> str:
    """Strips a reasoning model's train of thought, leaving only the answer."""
    return _DANGLING_THINKING.sub("", _THINKING.sub("", text)).strip()


class LLM(Protocol):
    """The contract: a prompt in, finished text out."""

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
    """The mlx-lm backend. The model loads lazily and lives out the process."""

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
            # Translation and summaries have no use for the reasoning: on Qwen3.6
            # it makes answers four times slower and leaks into the result.
            # Templates that do not know the flag simply ignore the extra variable.
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
    """The Ollama backend, for anyone whose daemon is already running."""

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
    """The server holding the model is unreachable or answered with nonsense."""


class RemoteLanguageModel:
    """A model on someone else's machine, speaking the OpenAI protocol.

    vLLM, llama.cpp server, LM Studio, TGI and Ollama itself all answer with this
    protocol, so one implementation covers every way of putting a model on a
    server. The application stays local even so: only the text of the transcript
    leaves, and the audio never does.
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
        # The connection outlives a request: translating an hour-long lecture is
        # dozens of batched calls, and opening a session for each is waste.
        self._http = client
        # The server applies the chat template, so the switch that turns reasoning
        # off travels to it as a field of the request. Not everything understands
        # it: what does not answers with a refusal, and then we repeat the request
        # without it and cut the reasoning off at our end.
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
            # The server knows nothing of template parameters — carry on without them.
            self._ask_without_thinking = False
            payload.pop("chat_template_kwargs")
            response = self._post(payload)

        if response.status_code != 200:
            raise RemoteLLMError(self._explain(response))

        message = response.json()["choices"][0]["message"]
        # A server may leave the reasoning inside the answer or move it to a field
        # of its own — either way it is taken out of what we return.
        if content := strip_thinking(message.get("content") or ""):
            return content

        reasoning = message.get("reasoning") or message.get("reasoning_content")
        if reasoning:
            # No answer at all: the model reasoned until the limit ran out. An
            # empty string would read as "the translation went missing", so say it.
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
            self._http = httpx.Client(base_url=self._base, headers=headers, timeout=self._timeout)
        return self._http

    def _post(self, payload: dict[str, object]):
        import httpx

        try:
            return self._session().post("/chat/completions", json=payload)
        except httpx.HTTPError as exc:
            raise RemoteLLMError(f"{self._base} is not responding: {exc}") from exc

    def _explain(self, response) -> str:
        """A refusal, naming the model asked for and what the server does serve.

        A server names its models its own way, which rarely matches our short
        name — and without this list a person hunts for the cause on their side.
        """
        detail = response.text.strip()[:200]
        available = ", ".join(self.available()) or "server did not list any"
        return (
            f"{self._base} rejected model '{self.model}' "
            f"(HTTP {response.status_code}: {detail}). Available: {available}"
        )

    def available(self) -> list[str]:
        """What the server serves. An empty list means the asking failed."""
        import httpx

        try:
            response = self._session().get("/models")
            return [item["id"] for item in response.json().get("data", [])]
        except (httpx.HTTPError, ValueError, KeyError, TypeError):
            # The diagnosis is optional: the refusal gets named regardless.
            return []


def check_remote(base_url: str, *, api_key: str | None = None, timeout: float = 5.0) -> str | None:
    """Whether the server answers with the model. `None` — yes; otherwise why not.

    The timeout is deliberately short: this reports a state rather than doing
    work, and waiting more than a few seconds for it serves nobody.
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
                    'the ollama backend is not installed: uv pip install -e ".[ollama]"'
                ) from exc
            return OllamaLanguageModel(model, host or "http://localhost:11434")
        case "openai":
            if not base_url:
                raise ValueError(
                    "no address for the model server: TRANSCRIPT_LLM_BASE_URL=http://host:port/v1"
                )
            return RemoteLanguageModel(model, base_url, api_key=api_key, timeout=timeout)
        case _:
            raise ValueError(f"unknown LLM backend: {kind}")
