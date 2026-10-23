"""Google Gemini provider.

Gemini's ``response_schema`` support makes it a natural fit for the ground-truth
stage: the schema is enforced server-side against the Pydantic model directly, so
what comes back either validates or the call fails.

Requires the ``google`` extra::

    pip install 'conversim[google]'
"""

from __future__ import annotations

import logging
from typing import Any

from pydantic import BaseModel, ValidationError

from ..errors import MissingDependencyError, ProviderError, StructuredOutputError
from ..models import TokenUsage
from .base import RetryableProviderError, RetryPolicy, StructuredResult, TextResult, retry_async

__all__ = ["GoogleProvider"]

logger = logging.getLogger(__name__)

_RETRYABLE_STATUS = frozenset({408, 429, 500, 502, 503, 504})


def _require_genai() -> tuple[Any, Any, Any]:
    try:
        from google import genai
        from google.genai import errors, types
    except ModuleNotFoundError as error:  # pragma: no cover - exercised by install shape
        raise MissingDependencyError("GoogleProvider", "google", "google-genai") from error
    return genai, types, errors


class GoogleProvider:
    """Text and structured generation against the Gemini API.

    Args:
        model: Default model identifier, e.g. ``"gemini-2.5-flash"``.
        api_key: API key. Falls back to the SDK's own resolution
            (``GOOGLE_API_KEY`` / ``GEMINI_API_KEY``) when omitted.
        temperature: Sent only when set.
        thinking_budget: Token budget for models that expose one. Pass ``0`` to switch
            reasoning off, which is usually the right call for schema filling.
        max_concurrency: Cap on simultaneous in-flight requests.
        retry: Backoff policy for transient failures.

    Example:
        >>> provider = GoogleProvider(model="gemini-2.5-flash", api_key="...")  # doctest: +SKIP
        >>> result = await provider.generate_structured(                        # doctest: +SKIP
        ...     prompt="Fill this in", schema=MySchema
        ... )
    """

    def __init__(
        self,
        *,
        model: str,
        api_key: str | None = None,
        temperature: float | None = None,
        thinking_budget: int | None = None,
        max_concurrency: int = 8,
        retry: RetryPolicy | None = None,
    ) -> None:
        genai, types, errors = _require_genai()
        self._types = types
        self._errors = errors
        self._model = model
        self._temperature = temperature
        self._thinking_budget = thinking_budget
        self._retry = retry or RetryPolicy()
        self._client = genai.Client(api_key=api_key)

        import asyncio

        self._semaphore = asyncio.Semaphore(max_concurrency)

    @property
    def model(self) -> str:
        """The default model identifier."""
        return self._model

    # -- text ---------------------------------------------------------------

    async def generate_text(
        self,
        *,
        prompt: str,
        max_output_tokens: int | None = None,
    ) -> TextResult:
        """Generate free-form text. See :class:`~conversim.providers.base.TextGenerator`."""

        async def call() -> TextResult:
            config = self._config(max_output_tokens=max_output_tokens)
            async with self._semaphore:
                try:
                    response = await self._client.aio.models.generate_content(
                        model=self._model, contents=prompt, config=config
                    )
                except Exception as error:
                    raise self._classify(error) from error

            text = getattr(response, "text", None)
            if not text:
                raise RetryableProviderError(
                    f"{self._model} returned an empty response "
                    f"(finish_reason={_finish_reason(response)!r})"
                )
            return TextResult(
                text=text,
                usage=_usage_from(response),
                model=self._model,
                truncated=_finish_reason(response) == "MAX_TOKENS",
            )

        return await retry_async(
            call, policy=self._retry, description=f"Gemini text generation ({self._model})"
        )

    # -- structured ---------------------------------------------------------

    async def generate_structured(
        self,
        *,
        prompt: str,
        schema: type[BaseModel],
    ) -> StructuredResult:
        """Generate a validated ``schema`` instance using Gemini's schema enforcement.

        See :class:`~conversim.providers.base.StructuredGenerator`.
        """

        async def call() -> StructuredResult:
            config = self._config(response_schema=schema)
            async with self._semaphore:
                try:
                    response = await self._client.aio.models.generate_content(
                        model=self._model, contents=prompt, config=config
                    )
                except Exception as error:
                    raise self._classify(error) from error

            text = getattr(response, "text", None)
            if not text:
                raise RetryableProviderError(
                    f"{self._model} returned an empty structured response "
                    f"(finish_reason={_finish_reason(response)!r})"
                )

            parsed = getattr(response, "parsed", None)
            if isinstance(parsed, schema):
                value: BaseModel = parsed
            else:
                try:
                    value = schema.model_validate_json(text)
                except ValidationError as error:
                    if _finish_reason(response) == "MAX_TOKENS":
                        raise StructuredOutputError(
                            f"{self._model} hit the output-token limit before completing "
                            f"a {schema.__name__} instance; raise max_output_tokens"
                        ) from error
                    raise RetryableProviderError(
                        f"{self._model} output failed {schema.__name__} validation: {error}"
                    ) from error

            return StructuredResult(
                value=value,
                usage=_usage_from(response),
                model=self._model,
                raw_text=text,
            )

        return await retry_async(
            call,
            policy=self._retry,
            description=f"Gemini structured generation ({self._model})",
        )

    # -- internals ----------------------------------------------------------

    def _config(
        self,
        *,
        response_schema: type[BaseModel] | None = None,
        max_output_tokens: int | None = None,
    ) -> Any:
        # We never pass tools, but the SDK still warns about automatic function
        # calling on every direct generate_content call. Disabling it explicitly
        # keeps that warning out of long runs, where it looks like a fault.
        kwargs: dict[str, Any] = {
            "automatic_function_calling": self._types.AutomaticFunctionCallingConfig(disable=True)
        }
        if response_schema is not None:
            kwargs["response_mime_type"] = "application/json"
            kwargs["response_schema"] = response_schema
        if max_output_tokens is not None:
            kwargs["max_output_tokens"] = max_output_tokens
        if self._temperature is not None:
            kwargs["temperature"] = self._temperature
        if self._thinking_budget is not None:
            kwargs["thinking_config"] = self._types.ThinkingConfig(
                thinking_budget=self._thinking_budget
            )
        return self._types.GenerateContentConfig(**kwargs)

    def _classify(self, error: Exception) -> Exception:
        """Map an SDK exception onto retryable or terminal."""
        if isinstance(error, self._errors.APIError):
            code = getattr(error, "code", None)
            if isinstance(code, int) and code in _RETRYABLE_STATUS:
                return RetryableProviderError(str(error))
            return ProviderError(str(error))
        return error


def _usage_from(response: Any) -> TokenUsage:
    usage = getattr(response, "usage_metadata", None)
    if usage is None:
        return TokenUsage()
    return TokenUsage(
        input_tokens=getattr(usage, "prompt_token_count", 0) or 0,
        output_tokens=getattr(usage, "candidates_token_count", 0) or 0,
    )


def _finish_reason(response: Any) -> str | None:
    candidates = getattr(response, "candidates", None) or []
    if not candidates:
        return None
    reason = getattr(candidates[0], "finish_reason", None)
    if reason is None:
        return None
    return getattr(reason, "name", None) or str(reason)
