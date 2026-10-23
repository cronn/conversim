"""Language-model providers.

The protocols in :mod:`conversim.providers.base` are what the pipeline depends on.
Concrete adapters are imported lazily so that installing conversim without any
provider extra still works.
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Any

from .base import (
    RetryableProviderError,
    RetryPolicy,
    StructuredGenerator,
    StructuredResult,
    TextGenerator,
    TextResult,
    retry_async,
)

if TYPE_CHECKING:
    from .google_provider import GoogleProvider
    from .openai_provider import OpenAIProvider, StructuredMode

__all__ = [
    "GoogleProvider",
    "OpenAIProvider",
    "RetryPolicy",
    "RetryableProviderError",
    "StructuredGenerator",
    "StructuredMode",
    "StructuredResult",
    "TextGenerator",
    "TextResult",
    "retry_async",
]

_LAZY = {
    "OpenAIProvider": ("conversim.providers.openai_provider", "OpenAIProvider"),
    "StructuredMode": ("conversim.providers.openai_provider", "StructuredMode"),
    "GoogleProvider": ("conversim.providers.google_provider", "GoogleProvider"),
}


def __getattr__(name: str) -> Any:
    """Import provider adapters on first access, so missing extras stay silent."""
    target = _LAZY.get(name)
    if target is None:
        raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
    from importlib import import_module

    module = import_module(target[0])
    value = getattr(module, target[1])
    globals()[name] = value
    return value


def __dir__() -> list[str]:
    return sorted(__all__)
