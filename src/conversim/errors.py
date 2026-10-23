"""Exception hierarchy for :mod:`conversim`.

Every error raised deliberately by this library derives from :class:`ConversimError`,
so callers can catch the whole family with a single ``except``.
"""

from __future__ import annotations

__all__ = [
    "ConfigurationError",
    "ConversimError",
    "DialogueFormatError",
    "MissingDependencyError",
    "ProviderError",
    "SchemaError",
    "StructuredOutputError",
    "SynthesisError",
]


class ConversimError(Exception):
    """Base class for all errors raised by conversim."""


class ConfigurationError(ConversimError):
    """A configuration object is internally inconsistent or incomplete."""


class SchemaError(ConfigurationError):
    """A ground-truth schema, or a reference into one, is invalid.

    Raised for example when a topic category points at a field path that does not
    exist in the schema.
    """


class ProviderError(ConversimError):
    """A language-model provider call failed and could not be retried further."""


class StructuredOutputError(ProviderError):
    """A provider returned output that does not validate against the target schema."""


class DialogueFormatError(ConversimError):
    """A generated dialogue could not be parsed or violates the requested format."""


class SynthesisError(ConversimError):
    """A speech-synthesis backend failed to render an utterance."""


class MissingDependencyError(ConversimError):
    """An optional dependency is required for the requested feature but not installed."""

    def __init__(self, feature: str, extra: str, package: str) -> None:
        super().__init__(
            f"{feature} requires the optional dependency {package!r}. "
            f"Install it with: pip install 'conversim[{extra}]'"
        )
        self.feature = feature
        self.extra = extra
        self.package = package
