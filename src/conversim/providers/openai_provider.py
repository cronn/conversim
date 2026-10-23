"""OpenAI and OpenAI-compatible provider.

Covers both hosted OpenAI and anything speaking the same wire protocol -- vLLM,
Ollama, llama.cpp's server, LM Studio -- by pointing ``base_url`` at it. That matters
for on-premises use, where the whole reason to generate data locally is that it must
not leave the building.

Requires the ``openai`` extra::

    pip install 'conversim[openai]'
"""

from __future__ import annotations

import asyncio
import json
import logging
from typing import Any, Literal

from pydantic import BaseModel, ValidationError

from ..errors import MissingDependencyError, ProviderError, StructuredOutputError
from ..models import TokenUsage
from .base import RetryableProviderError, RetryPolicy, StructuredResult, TextResult, retry_async

__all__ = ["OpenAIProvider", "StructuredMode"]

logger = logging.getLogger(__name__)

StructuredMode = Literal["parse", "json_object"]
"""How structured output is requested.

``parse`` uses OpenAI's strict structured-output support, which guarantees schema
conformance server-side. ``json_object`` only asks for valid JSON and validates
client-side -- slower and occasionally retried, but it is what most
OpenAI-compatible local servers actually support.
"""


def _require_openai() -> Any:
    try:
        import openai
    except ModuleNotFoundError as error:  # pragma: no cover - exercised by install shape
        raise MissingDependencyError("OpenAIProvider", "openai", "openai") from error
    return openai


class OpenAIProvider:
    """Text and structured generation against an OpenAI-compatible chat endpoint.

    Args:
        model: Default model identifier, e.g. ``"gpt-5-mini"``.
        api_key: API key. Falls back to the SDK's own resolution (``OPENAI_API_KEY``)
            when omitted. conversim never reads credentials from disk itself.
        base_url: Override to target a compatible server instead of OpenAI.
        structured_mode: See :data:`StructuredMode`. Use ``"json_object"`` for local
            servers without strict structured-output support.
        temperature: Sent only when set. Several current OpenAI models reject the
            parameter outright, so the default is to omit it.
        max_concurrency: Cap on simultaneous in-flight requests from this provider.
        retry: Backoff policy for transient failures.
        timeout: Per-request timeout in seconds.
        default_headers: Extra headers, e.g. for a gateway.

    Example:
        >>> provider = OpenAIProvider(model="gpt-5-mini", api_key="sk-...")  # doctest: +SKIP
        >>> result = await provider.generate_text(prompt="Say hello")        # doctest: +SKIP
    """

    def __init__(
        self,
        *,
        model: str,
        api_key: str | None = None,
        base_url: str | None = None,
        structured_mode: StructuredMode = "parse",
        temperature: float | None = None,
        max_concurrency: int = 8,
        retry: RetryPolicy | None = None,
        timeout: float | None = 600.0,
        default_headers: dict[str, str] | None = None,
    ) -> None:
        openai = _require_openai()
        self._openai = openai
        self._model = model
        self._structured_mode: StructuredMode = structured_mode
        self._temperature = temperature
        self._retry = retry or RetryPolicy()
        self._semaphore = asyncio.Semaphore(max_concurrency)
        self._client = openai.AsyncOpenAI(
            api_key=api_key,
            base_url=base_url,
            timeout=timeout,
            default_headers=default_headers,
            max_retries=0,  # retries are handled here, with our own backoff and logging
        )

    @property
    def model(self) -> str:
        """The default model identifier."""
        return self._model

    async def aclose(self) -> None:
        """Close the underlying HTTP client."""
        await self._client.close()

    async def __aenter__(self) -> OpenAIProvider:
        return self

    async def __aexit__(self, *exc_info: object) -> None:
        await self.aclose()

    # -- text ---------------------------------------------------------------

    async def generate_text(
        self,
        *,
        prompt: str,
        max_output_tokens: int | None = None,
    ) -> TextResult:
        """Generate free-form text. See :class:`~conversim.providers.base.TextGenerator`."""

        async def call() -> TextResult:
            async with self._semaphore:
                try:
                    response = await self._client.chat.completions.create(
                        model=self._model,
                        messages=[{"role": "user", "content": prompt}],
                        max_completion_tokens=max_output_tokens,
                        **self._optional_params(),
                    )
                except Exception as error:
                    raise self._classify(error) from error

            choice = response.choices[0]
            content = choice.message.content
            if not content:
                raise RetryableProviderError(
                    f"{self._model} returned an empty completion "
                    f"(finish_reason={choice.finish_reason!r})"
                )
            return TextResult(
                text=content,
                usage=_usage_from(response.usage),
                model=getattr(response, "model", self._model),
                truncated=choice.finish_reason == "length",
            )

        return await retry_async(
            call, policy=self._retry, description=f"OpenAI text generation ({self._model})"
        )

    # -- structured ---------------------------------------------------------

    async def generate_structured(
        self,
        *,
        prompt: str,
        schema: type[BaseModel],
    ) -> StructuredResult:
        """Generate a validated ``schema`` instance.

        See :class:`~conversim.providers.base.StructuredGenerator`.
        """
        if self._structured_mode == "parse":
            return await self._generate_structured_parse(prompt=prompt, schema=schema)
        return await self._generate_structured_json(prompt=prompt, schema=schema)

    async def _generate_structured_parse(
        self, *, prompt: str, schema: type[BaseModel]
    ) -> StructuredResult:
        parse = self._resolve_parse_helper()

        async def call() -> StructuredResult:
            async with self._semaphore:
                try:
                    response = await parse(
                        model=self._model,
                        messages=[{"role": "user", "content": prompt}],
                        response_format=schema,
                        **self._optional_params(),
                    )
                except Exception as error:
                    raise self._classify(error) from error

            message = response.choices[0].message
            refusal = getattr(message, "refusal", None)
            if refusal:
                raise StructuredOutputError(f"{self._model} refused the request: {refusal}")
            parsed = getattr(message, "parsed", None)
            if parsed is None:
                raise RetryableProviderError(f"{self._model} returned no parsed structured output")
            return StructuredResult(
                value=parsed,
                usage=_usage_from(response.usage),
                model=getattr(response, "model", self._model),
                raw_text=message.content,
            )

        return await retry_async(
            call,
            policy=self._retry,
            description=f"OpenAI structured generation ({self._model})",
        )

    async def _generate_structured_json(
        self, *, prompt: str, schema: type[BaseModel]
    ) -> StructuredResult:
        instructed = (
            f"{prompt}\n\n"
            "Antworte ausschliesslich mit einem JSON-Objekt, das dieses JSON-Schema "
            "erfuellt. Keine Erklaerungen, kein Markdown.\n\n"
            f"{json.dumps(schema.model_json_schema(), ensure_ascii=False, indent=2)}"
        )

        async def call() -> StructuredResult:
            async with self._semaphore:
                try:
                    response = await self._client.chat.completions.create(
                        model=self._model,
                        messages=[{"role": "user", "content": instructed}],
                        response_format={"type": "json_object"},
                        **self._optional_params(),
                    )
                except Exception as error:
                    raise self._classify(error) from error

            content = response.choices[0].message.content
            if not content:
                raise RetryableProviderError(f"{self._model} returned an empty completion")
            try:
                value = schema.model_validate_json(_strip_code_fence(content))
            except ValidationError as error:
                # Retryable: a re-roll often produces conforming output.
                raise RetryableProviderError(
                    f"{self._model} output failed {schema.__name__} validation: {error}"
                ) from error
            return StructuredResult(
                value=value,
                usage=_usage_from(response.usage),
                model=getattr(response, "model", self._model),
                raw_text=content,
            )

        return await retry_async(
            call,
            policy=self._retry,
            description=f"OpenAI JSON generation ({self._model})",
        )

    # -- internals ----------------------------------------------------------

    def _optional_params(self) -> dict[str, Any]:
        return {} if self._temperature is None else {"temperature": self._temperature}

    def _resolve_parse_helper(self) -> Any:
        """Find the structured-output helper across SDK layouts.

        It moved from ``client.beta.chat.completions.parse`` to
        ``client.chat.completions.parse``; support both rather than pinning the SDK.
        """
        completions = self._client.chat.completions
        if hasattr(completions, "parse"):
            return completions.parse
        beta = getattr(self._client, "beta", None)
        parse = getattr(getattr(getattr(beta, "chat", None), "completions", None), "parse", None)
        if parse is not None:
            return parse
        raise ProviderError(
            "The installed openai SDK exposes no structured-output parse helper. "
            "Upgrade openai, or construct the provider with structured_mode='json_object'."
        )

    def _classify(self, error: Exception) -> Exception:
        """Map an SDK exception onto retryable or terminal.

        Rate limits, timeouts, connection failures and 5xx are worth another try;
        authentication and malformed-request errors are not.
        """
        openai = self._openai
        transient = (
            openai.RateLimitError,
            openai.APITimeoutError,
            openai.APIConnectionError,
            openai.InternalServerError,
        )
        if isinstance(error, transient):
            return RetryableProviderError(str(error))
        status = getattr(error, "status_code", None)
        if isinstance(status, int) and (status >= 500 or status == 429):
            return RetryableProviderError(str(error))
        if isinstance(error, openai.OpenAIError):
            return ProviderError(str(error))
        return error


def _usage_from(usage: Any) -> TokenUsage:
    if usage is None:
        return TokenUsage()
    return TokenUsage(
        input_tokens=getattr(usage, "prompt_tokens", 0) or 0,
        output_tokens=getattr(usage, "completion_tokens", 0) or 0,
    )


def _strip_code_fence(text: str) -> str:
    """Remove a surrounding ```json fence, which smaller models like to add."""
    stripped = text.strip()
    if not stripped.startswith("```"):
        return stripped
    lines = stripped.splitlines()
    if len(lines) < 2:
        return stripped
    body = lines[1:-1] if lines[-1].strip().startswith("```") else lines[1:]
    return "\n".join(body).strip()
