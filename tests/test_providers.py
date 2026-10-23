"""Retry behaviour and the protocol contracts."""

from __future__ import annotations

import random

import pytest

from conversim.errors import MissingDependencyError, ProviderError
from conversim.providers.base import (
    RetryableProviderError,
    RetryPolicy,
    StructuredGenerator,
    TextGenerator,
    retry_async,
)

from .conftest import StubStructuredProvider, StubTextProvider


def test_stubs_satisfy_the_protocols() -> None:
    """The practical payoff of Protocol over inheritance."""
    assert isinstance(StubStructuredProvider(), StructuredGenerator)
    assert isinstance(StubTextProvider(), TextGenerator)


async def test_retry_returns_on_first_success() -> None:
    calls = 0

    async def operation() -> str:
        nonlocal calls
        calls += 1
        return "ok"

    assert await retry_async(operation, policy=RetryPolicy(), description="test") == "ok"
    assert calls == 1


async def test_retry_recovers_after_transient_failures() -> None:
    calls = 0

    async def operation() -> str:
        nonlocal calls
        calls += 1
        if calls < 3:
            raise RetryableProviderError("temporarily unavailable")
        return "ok"

    policy = RetryPolicy(attempts=3, initial_backoff=0.0, max_backoff=0.0)
    assert await retry_async(operation, policy=policy, description="test") == "ok"
    assert calls == 3


async def test_retry_gives_up_and_reraises() -> None:
    async def operation() -> str:
        raise RetryableProviderError("still down")

    policy = RetryPolicy(attempts=2, initial_backoff=0.0, max_backoff=0.0)
    with pytest.raises(RetryableProviderError, match="still down"):
        await retry_async(operation, policy=policy, description="test")


async def test_non_retryable_errors_are_not_retried() -> None:
    calls = 0

    async def operation() -> str:
        nonlocal calls
        calls += 1
        raise ProviderError("bad api key")

    policy = RetryPolicy(attempts=5, initial_backoff=0.0)
    with pytest.raises(ProviderError, match="bad api key"):
        await retry_async(operation, policy=policy, description="test")
    assert calls == 1  # retrying an auth failure only wastes time


def test_backoff_grows_and_is_bounded() -> None:
    policy = RetryPolicy(initial_backoff=1.0, multiplier=2.0, max_backoff=5.0)
    rng = random.Random(0)
    for attempt in range(1, 8):
        delay = policy.delay_for(attempt, rng)
        assert 0.0 <= delay <= 5.0  # full jitter, capped


def test_single_attempt_policy_disables_retrying() -> None:
    assert RetryPolicy(attempts=1).attempts == 1


def test_missing_extra_raises_an_actionable_error() -> None:
    error = MissingDependencyError("QwenTTS", "qwen", "qwen-tts")
    message = str(error)
    assert "conversim[qwen]" in message
    assert "qwen-tts" in message


def test_provider_adapters_are_importable_without_their_sdk() -> None:
    """Importing the package must not require any provider SDK."""
    import importlib

    for module in (
        "conversim.providers.openai_provider",
        "conversim.providers.google_provider",
        "conversim.speech.openai_tts",
        "conversim.speech.qwen_tts",
    ):
        importlib.import_module(module)


def test_lazy_attribute_access_reports_unknown_names() -> None:
    import conversim.providers as providers
    import conversim.speech as speech

    with pytest.raises(AttributeError):
        providers.NotAProvider  # noqa: B018
    with pytest.raises(AttributeError):
        speech.NotABackend  # noqa: B018
