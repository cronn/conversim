"""Provider-agnostic interfaces for language-model calls.

conversim needs exactly two capabilities from a model:

* :class:`StructuredGenerator` -- produce an instance of a given Pydantic schema.
  Used for the ground truth, where schema conformance is the whole point.
* :class:`TextGenerator` -- produce free text. Used for the dialogue script.

Both are :class:`~typing.Protocol` types, so anything with the right method shape
works: the bundled adapters, a local model behind an OpenAI-compatible endpoint, or a
stub in your tests. Nothing in this package imports a provider SDK at module level.
"""

from __future__ import annotations

import asyncio
import logging
import random
from collections.abc import Awaitable, Callable
from dataclasses import dataclass
from typing import Protocol, TypeVar, runtime_checkable

from pydantic import BaseModel

from ..errors import ProviderError
from ..models import TokenUsage

__all__ = [
    "RetryPolicy",
    "RetryableProviderError",
    "StructuredGenerator",
    "StructuredResult",
    "TextGenerator",
    "TextResult",
    "retry_async",
]

logger = logging.getLogger(__name__)

T = TypeVar("T")


class RetryableProviderError(ProviderError):
    """A provider failure that is worth trying again -- rate limits, 5xx, timeouts.

    Adapters raise this (instead of plain :class:`~conversim.errors.ProviderError`)
    to opt a failure in to the retry loop. Authentication and request-shape errors
    deliberately do not qualify: retrying them only wastes time.
    """


@dataclass(frozen=True)
class StructuredResult:
    """A validated schema instance returned by a provider.

    Attributes:
        value: The parsed and validated model instance.
        usage: Tokens consumed by the call.
        model: Identifier of the model that produced it, as reported or requested.
        raw_text: The raw response body, kept for provenance when available.
    """

    value: BaseModel
    usage: TokenUsage
    model: str
    raw_text: str | None = None


@dataclass(frozen=True)
class TextResult:
    """Free-form text returned by a provider.

    Attributes:
        text: The generated text.
        usage: Tokens consumed by the call.
        model: Identifier of the model that produced it.
        truncated: Whether generation stopped at the output-token limit rather than
            finishing on its own. A truncated dialogue is usually worth discarding.
    """

    text: str
    usage: TokenUsage
    model: str
    truncated: bool = False


@runtime_checkable
class StructuredGenerator(Protocol):
    """Produces instances of a caller-supplied Pydantic schema."""

    @property
    def model(self) -> str:
        """The model identifier this generator calls by default."""
        ...

    async def generate_structured(
        self,
        *,
        prompt: str,
        schema: type[BaseModel],
    ) -> StructuredResult:
        """Generate one instance of ``schema``.

        Implementations must return a *validated* instance, or raise
        :class:`~conversim.errors.StructuredOutputError` if the model's output does
        not conform.
        """
        ...


@runtime_checkable
class TextGenerator(Protocol):
    """Produces free-form text."""

    @property
    def model(self) -> str:
        """The model identifier this generator calls by default."""
        ...

    async def generate_text(
        self,
        *,
        prompt: str,
        max_output_tokens: int | None = None,
    ) -> TextResult:
        """Generate text for ``prompt``."""
        ...


@dataclass(frozen=True)
class RetryPolicy:
    """Exponential backoff with full jitter.

    Args:
        attempts: Total tries, including the first. ``1`` disables retrying.
        initial_backoff: Seconds to wait before the second attempt.
        max_backoff: Ceiling for the computed delay.
        multiplier: Growth factor applied per attempt.
    """

    attempts: int = 3
    initial_backoff: float = 1.0
    max_backoff: float = 30.0
    multiplier: float = 2.0

    def delay_for(self, attempt: int, rng: random.Random) -> float:
        """Seconds to sleep after a failed ``attempt`` (one-based).

        Full jitter (uniform over ``[0, computed]``) avoids a thundering herd when
        many concurrent samples hit the same rate limit at once.
        """
        exponential = self.initial_backoff * (self.multiplier ** (attempt - 1))
        return rng.uniform(0.0, min(exponential, self.max_backoff))


async def retry_async(
    operation: Callable[[], Awaitable[T]],
    *,
    policy: RetryPolicy,
    description: str,
    retryable: type[Exception] | tuple[type[Exception], ...] = RetryableProviderError,
    rng: random.Random | None = None,
) -> T:
    """Run ``operation``, retrying failures that match ``retryable``.

    Jitter is drawn from a dedicated generator rather than a seeded one, so retry
    timing -- which depends on the network, not on the dataset definition -- cannot
    perturb the reproducible sampling stream.

    Raises:
        Exception: the last failure, once attempts are exhausted or the error is not
            retryable.
    """
    jitter_rng = rng or random.Random()
    last_error: Exception | None = None

    for attempt in range(1, policy.attempts + 1):
        try:
            return await operation()
        except retryable as error:
            last_error = error
            if attempt == policy.attempts:
                break
            delay = policy.delay_for(attempt, jitter_rng)
            logger.warning(
                "%s failed (attempt %d/%d): %s -- retrying in %.2fs",
                description,
                attempt,
                policy.attempts,
                error,
                delay,
            )
            await asyncio.sleep(delay)

    assert last_error is not None  # unreachable: the loop only breaks after a failure
    raise last_error
