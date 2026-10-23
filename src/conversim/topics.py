"""Deciding which topics a conversation covers, and in what order.

A ground-truth instance is a bag of facts with no inherent narrative order. Handing
it to a model as-is produces checklist dialogue: the model walks the JSON top to
bottom. A topic plan fixes that by deriving, per conversation, which subjects
actually have content and arranging them into a plausible sequence.

Categories declare the schema fields they cover as dotted paths, and those paths are
validated against the schema when the planner is built. That matters more than it
sounds: a category pointing at a misspelled field silently never activates, and the
resulting dataset is missing a whole subject area with nothing in the logs to say so.
"""

from __future__ import annotations

import random
from typing import Any, Literal, Protocol, runtime_checkable

from pydantic import BaseModel, ConfigDict, Field, model_validator

from .errors import ConfigurationError
from .schema import SchemaSpec
from .variation import VariationConfig

__all__ = [
    "Position",
    "PriorityRule",
    "SchemaTopicPlanner",
    "TopicCategory",
    "TopicPlan",
    "TopicPlanEntry",
    "TopicPlanner",
]

Position = Literal["early", "any", "late"]
"""Where a category may appear: pinned to the front, free, or pinned to the back."""


class TopicCategory(BaseModel):
    """One subject a conversation may cover.

    Args:
        name: Identifier, also used to target the category from a :class:`PriorityRule`.
        description: Prompt-facing text describing what to cover. This is what the
            dialogue model actually reads, so write it as guidance, not as a label.
        field_paths: Dotted schema paths whose content activates this category. Leave
            empty together with ``always_include`` for categories that are not tied to
            ground-truth data, such as a greeting.
        priority: Ordering key within the category's position band; lower comes first.
        position: Whether the category is pinned early, late, or free to float.
        always_include: Include regardless of whether any field has content.
        inclusion_probability: Chance of including the category when it *is* active.
            Below 1.0 this adds run-to-run variation; useful for optional colour such
            as personal background chatter.
        priority_jitter: Inclusive ``(low, high)`` range to redraw ``priority`` from,
            letting a category land at different points across conversations.
    """

    model_config = ConfigDict(frozen=True)

    name: str
    description: str
    field_paths: tuple[str, ...] = ()
    priority: int = 5
    position: Position = "any"
    always_include: bool = False
    inclusion_probability: float = Field(default=1.0, ge=0.0, le=1.0)
    priority_jitter: tuple[int, int] | None = None

    @model_validator(mode="after")
    def _activatable(self) -> TopicCategory:
        if not self.field_paths and not self.always_include:
            raise ValueError(
                f"Category {self.name!r} declares no field_paths and is not "
                f"always_include, so it can never activate"
            )
        if self.priority_jitter is not None:
            low, high = self.priority_jitter
            if low > high:
                raise ValueError(
                    f"Category {self.name!r} has an inverted priority_jitter: {(low, high)}"
                )
        return self


class PriorityRule(BaseModel):
    """Reorder categories when a variation axis takes a particular value.

    This is how "a symptomatic visit opens with the symptoms" is expressed without
    branching code: the rule fires when the sampled variation matches, and rewrites
    the priorities it names.

    Args:
        axis: Name of the variation axis to match.
        value: Axis value that triggers the rule.
        priorities: New priority per category name.
    """

    model_config = ConfigDict(frozen=True)

    axis: str
    value: str
    priorities: dict[str, int]


class TopicPlanEntry(BaseModel):
    """One step of a planned conversation."""

    model_config = ConfigDict(frozen=True)

    name: str
    description: str


class TopicPlan(BaseModel):
    """An ordered sequence of topics for one conversation.

    Rendering is deliberately left to the prompt builder: the wording around the list
    ("cover these roughly in this order") is language- and domain-specific, while the
    ordering itself is not.
    """

    model_config = ConfigDict(frozen=True)

    entries: tuple[TopicPlanEntry, ...] = ()

    def __len__(self) -> int:
        return len(self.entries)

    def __bool__(self) -> bool:
        return bool(self.entries)

    @property
    def names(self) -> tuple[str, ...]:
        """Category names in planned order."""
        return tuple(entry.name for entry in self.entries)

    @property
    def descriptions(self) -> tuple[str, ...]:
        """Prompt-facing descriptions in planned order."""
        return tuple(entry.description for entry in self.entries)


@runtime_checkable
class TopicPlanner(Protocol):
    """Produces a :class:`TopicPlan` for one conversation.

    Implement this to replace the default content-driven planning with something
    else -- a fixed clinical protocol, say, or a learned ordering.
    """

    def plan(
        self,
        ground_truth: Any,
        variation: VariationConfig,
        rng: random.Random,
    ) -> TopicPlan:
        """Plan the topic sequence for one conversation."""
        ...


class SchemaTopicPlanner:
    """Default planner: include topics that have ground-truth content, then order them.

    Planning happens in four steps:

    1. **Activate** categories whose ``field_paths`` resolve to substantive values
       (see :meth:`~conversim.schema.SchemaSpec.is_substantive`), plus any marked
       ``always_include``.
    2. **Thin out** optional categories according to ``inclusion_probability``.
    3. **Reprioritise** using the :class:`PriorityRule` set matching this
       conversation's variation config, then apply per-category ``priority_jitter``.
    4. **Sort** within the early/free/late bands, breaking ties randomly so that
       equally-ranked topics do not always appear in the same order.

    Args:
        schema: The ground-truth schema, used to validate field paths and to decide
            what counts as content.
        categories: The catalogue of topics to choose from.
        priority_rules: Variation-dependent reordering rules.

    Raises:
        SchemaError: if a category references a field path that the schema lacks.
        ConfigurationError: on duplicate category names, or a rule naming an unknown
            category.
    """

    def __init__(
        self,
        schema: SchemaSpec,
        categories: tuple[TopicCategory, ...] | list[TopicCategory],
        priority_rules: tuple[PriorityRule, ...] | list[PriorityRule] = (),
    ) -> None:
        self.schema = schema
        self.categories = tuple(categories)
        self.priority_rules = tuple(priority_rules)

        names = [category.name for category in self.categories]
        duplicates = {name for name in names if names.count(name) > 1}
        if duplicates:
            raise ConfigurationError(f"Duplicate topic category names: {sorted(duplicates)}")

        known = set(names)
        for category in self.categories:
            for path in category.field_paths:
                # Raises SchemaError with the available field names on a typo.
                schema.validate_field_path(path)

        for rule in self.priority_rules:
            unknown = set(rule.priorities) - known
            if unknown:
                raise ConfigurationError(
                    f"Priority rule {rule.axis}={rule.value} targets unknown "
                    f"categories: {sorted(unknown)}"
                )

    def plan(
        self,
        ground_truth: Any,
        variation: VariationConfig,
        rng: random.Random,
    ) -> TopicPlan:
        """Plan the topic sequence for one conversation."""
        priorities = self._resolved_priorities(variation)

        selected: list[tuple[TopicCategory, int]] = []
        for category in self.categories:
            if not self._is_active(category, ground_truth):
                continue
            if category.inclusion_probability < 1.0 and rng.random() >= (
                category.inclusion_probability
            ):
                continue
            priority = priorities[category.name]
            if category.priority_jitter is not None:
                low, high = category.priority_jitter
                priority = rng.randint(low, high)
            selected.append((category, priority))

        ordered = self._sort_bands(selected, rng)
        return TopicPlan(
            entries=tuple(
                TopicPlanEntry(name=category.name, description=category.description)
                for category in ordered
            )
        )

    def _is_active(self, category: TopicCategory, ground_truth: Any) -> bool:
        if category.always_include:
            return True
        return self.schema.has_content(ground_truth, category.field_paths)

    def _resolved_priorities(self, variation: VariationConfig) -> dict[str, int]:
        priorities = {category.name: category.priority for category in self.categories}
        for rule in self.priority_rules:
            if variation.value(rule.axis) == rule.value:
                priorities.update(rule.priorities)
        return priorities

    @staticmethod
    def _sort_bands(
        selected: list[tuple[TopicCategory, int]],
        rng: random.Random,
    ) -> list[TopicCategory]:
        bands: dict[Position, list[tuple[TopicCategory, int]]] = {
            "early": [],
            "any": [],
            "late": [],
        }
        for category, priority in selected:
            bands[category.position].append((category, priority))

        ordered: list[TopicCategory] = []
        for position in ("early", "any", "late"):
            band = bands[position]
            band.sort(key=lambda item: (item[1], rng.random()))
            ordered.extend(category for category, _ in band)
        return ordered
