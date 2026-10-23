"""Introspection over the ground-truth schema.

conversim treats the ground truth as "some Pydantic model the caller owns". This
module is the only place that reasons *about* such a model, and it provides the two
things the rest of the library needs:

* **Sentinel awareness** -- schemas for this kind of task usually distinguish
  "not talked about" from "talked about, answer unknown" from "does not apply".
  Those are ordinary string values in the schema, and both prompt building and
  topic planning need to recognise them.
* **Field-path validation** -- topic categories point at schema fields by dotted
  path. Validating those paths against the model turns a silent typo into a loud
  error at construction time.
"""

from __future__ import annotations

import types
import typing
from dataclasses import dataclass, field
from enum import Enum
from typing import Any

from pydantic import BaseModel

from .errors import SchemaError

__all__ = ["DEFAULT_SENTINELS", "SchemaSpec"]

DEFAULT_SENTINELS: frozenset[str] = frozenset({"not_discussed", "unknown", "not_applicable"})
"""Sentinel values recognised unless the caller overrides them.

``not_discussed`` is the load-bearing one: it is what keeps generated conversations
short and natural, because the dialogue prompt is told never to mention such fields.
"""

_MISSING = object()


def _unwrap_annotation(annotation: Any) -> list[Any]:
    """Flatten ``Optional``/``Union``/container annotations into candidate leaf types.

    ``list[Symptom] | None`` yields ``[Symptom]``; ``dict[str, Finding]`` yields
    ``[Finding]``. Used to find the nested model to descend into when resolving a
    dotted field path.
    """
    origin = typing.get_origin(annotation)
    if origin is None:
        return [annotation]

    # Union / Optional (both typing.Union and PEP 604 ``X | Y``).
    if origin is typing.Union or origin is types.UnionType:
        results: list[Any] = []
        for argument in typing.get_args(annotation):
            if argument is type(None):
                continue
            results.extend(_unwrap_annotation(argument))
        return results

    # Containers: descend into the value type.
    arguments = typing.get_args(annotation)
    if not arguments:
        return []
    if origin in (dict, typing.Dict):  # noqa: UP006 - runtime comparison
        arguments = arguments[1:]
    results = []
    for argument in arguments:
        if argument is Ellipsis or argument is type(None):
            continue
        results.extend(_unwrap_annotation(argument))
    return results


def _nested_model(annotation: Any) -> type[BaseModel] | None:
    """Return the Pydantic model reachable through ``annotation``, if any."""
    for candidate in _unwrap_annotation(annotation):
        if isinstance(candidate, type) and issubclass(candidate, BaseModel):
            return candidate
    return None


@dataclass(frozen=True)
class SchemaSpec:
    """A ground-truth schema plus the conventions used to interpret its values.

    Args:
        model: The Pydantic model describing one ground-truth instance.
        sentinels: String values that mean "no substantive information here".
        sentinel_docs: Optional per-sentinel explanation, surfaced in prompts so the
            generating model knows what each one implies.

    Example:
        >>> from pydantic import BaseModel
        >>> class Visit(BaseModel):
        ...     reason: str = "not_discussed"
        >>> spec = SchemaSpec(Visit)
        >>> spec.is_substantive("not_discussed")
        False
        >>> spec.is_substantive("chest pain")
        True
    """

    model: type[BaseModel]
    sentinels: frozenset[str] = DEFAULT_SENTINELS
    sentinel_docs: dict[str, str] = field(default_factory=dict)

    def __post_init__(self) -> None:
        if not (isinstance(self.model, type) and issubclass(self.model, BaseModel)):
            raise SchemaError(f"Ground-truth schema must be a Pydantic model, got {self.model!r}")
        unknown_docs = set(self.sentinel_docs) - set(self.sentinels)
        if unknown_docs:
            raise SchemaError(
                f"sentinel_docs documents values that are not sentinels: {sorted(unknown_docs)}"
            )

    @property
    def name(self) -> str:
        """The schema class name, as referenced in prompts."""
        return self.model.__name__

    # -- value interpretation -------------------------------------------------

    def is_sentinel(self, value: Any) -> bool:
        """Whether ``value`` is one of the configured sentinel strings."""
        if isinstance(value, Enum):
            value = value.value
        return isinstance(value, str) and value in self.sentinels

    def is_substantive(self, value: Any) -> bool:
        """Whether ``value`` carries information worth talking about.

        The rule, applied recursively:

        * ``None``, empty strings and empty containers are not substantive.
        * Sentinel values are not substantive.
        * ``False`` is *not* substantive -- a bare negative answer does not by itself
          justify spending dialogue turns on a topic. ``0`` on the other hand is,
          because a count of zero is a stated fact.
        * Containers and nested models are substantive if any member is.
        """
        if value is None:
            return False
        if self.is_sentinel(value):
            return False
        if isinstance(value, bool):
            return value
        if isinstance(value, Enum):
            return True
        if isinstance(value, str):
            return bool(value.strip())
        if isinstance(value, BaseModel):
            return self.is_substantive(value.model_dump())
        if isinstance(value, dict):
            return any(self.is_substantive(item) for item in value.values())
        if isinstance(value, (list, tuple, set, frozenset)):
            return any(self.is_substantive(item) for item in value)
        return True

    # -- field paths ---------------------------------------------------------

    def validate_field_path(self, path: str) -> None:
        """Check that ``path`` addresses a real field in the schema.

        Paths are dotted and may traverse nested models, including through
        ``Optional`` and list-valued fields: ``partners.hiv_status`` is valid when
        ``partners`` is a ``list[Partner]``.

        Raises:
            SchemaError: if any segment does not exist.
        """
        if not path:
            raise SchemaError("Field path must not be empty")

        current: type[BaseModel] | None = self.model
        walked: list[str] = []
        for segment in path.split("."):
            if current is None:
                raise SchemaError(
                    f"Field path {path!r} descends into {'.'.join(walked)!r}, "
                    f"which is not a nested model"
                )
            field_info = current.model_fields.get(segment)
            if field_info is None:
                available = ", ".join(sorted(current.model_fields))
                location = ".".join(walked) or self.name
                raise SchemaError(
                    f"Field path {path!r} is invalid: {location} has no field {segment!r}. "
                    f"Available fields: {available}"
                )
            walked.append(segment)
            current = _nested_model(field_info.annotation)

    def resolve(self, data: Any, path: str) -> Any:
        """Read ``path`` out of a ground-truth mapping or model instance.

        Returns ``None`` when any segment is absent. When a segment traverses a list,
        the result is the list of resolved values from its items, so
        ``partners.hiv_status`` over two partners yields two statuses.
        """
        current: Any = data.model_dump() if isinstance(data, BaseModel) else data
        for segment in path.split("."):
            current = _descend(current, segment)
            if current is _MISSING:
                return None
        return current

    def has_content(self, data: Any, paths: typing.Iterable[str]) -> bool:
        """Whether any of ``paths`` resolves to a substantive value in ``data``."""
        return any(self.is_substantive(self.resolve(data, path)) for path in paths)

    # -- prompt helpers ------------------------------------------------------

    def json_schema(self) -> dict[str, Any]:
        """The JSON Schema for the model, as handed to structured-output APIs."""
        return self.model.model_json_schema()

    def field_guide(self) -> str:
        """Render every documented field as ``path: description``, one per line.

        Useful for models that ignore field descriptions embedded in a JSON schema
        but do follow them when restated in the prompt.
        """
        lines: list[str] = []

        def walk(model: type[BaseModel], prefix: str, seen: frozenset[type[BaseModel]]) -> None:
            if model in seen:
                return  # guard against recursive schemas
            seen = seen | {model}
            for name, info in model.model_fields.items():
                path = f"{prefix}{name}"
                if info.description:
                    description = " ".join(info.description.split())
                    lines.append(f"- {path}: {description}")
                nested = _nested_model(info.annotation)
                if nested is not None:
                    walk(nested, f"{path}.", seen)

        walk(self.model, "", frozenset())
        return "\n".join(lines)

    def sentinel_guide(self) -> str:
        """Render the sentinel conventions for inclusion in a prompt."""
        lines = []
        for sentinel in sorted(self.sentinels):
            explanation = self.sentinel_docs.get(sentinel)
            lines.append(f"- `{sentinel}`: {explanation}" if explanation else f"- `{sentinel}`")
        return "\n".join(lines)


def _descend(current: Any, segment: str) -> Any:
    """One step of :meth:`SchemaSpec.resolve`; returns ``_MISSING`` when absent."""
    if isinstance(current, BaseModel):
        current = current.model_dump()
    if isinstance(current, dict):
        return current.get(segment, _MISSING)
    if isinstance(current, (list, tuple)):
        collected = [_descend(item, segment) for item in current]
        present = [item for item in collected if item is not _MISSING]
        return present if present else _MISSING
    return _MISSING
