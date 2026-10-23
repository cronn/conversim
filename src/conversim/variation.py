"""Explicit control over dataset diversity.

Left to its own devices, a language model asked 400 times for "a realistic
conversation" produces 400 variations on its favourite conversation. A variation
space makes diversity a design parameter instead of a hope: each axis is sampled
per conversation and its instruction text is injected into *both* the ground-truth
and the dialogue prompt, so the two stay consistent.

The space is also the honest place to state how much diversity was designed in --
:meth:`VariationSpace.cardinality` counts the reachable configurations.
"""

from __future__ import annotations

import math
import random
from typing import Any

from pydantic import BaseModel, ConfigDict, Field, model_validator

__all__ = [
    "CategoricalAxis",
    "FlagAxis",
    "VariationConfig",
    "VariationOption",
    "VariationSpace",
]


class VariationOption(BaseModel):
    """One possible setting of a categorical axis.

    Args:
        value: Machine-readable identifier, recorded in the dataset for analysis.
        instruction: Prompt-facing sentence describing this setting. Omit to have the
            option influence sampling and metadata without appearing in the prompt.
        weight: Relative sampling weight. Defaults to uniform.
        metadata: Arbitrary extras read by scenario builders -- for example a turn
            range attached to a "concise" or "detailed" style.
    """

    model_config = ConfigDict(frozen=True)

    value: str
    instruction: str | None = None
    weight: float = Field(default=1.0, gt=0.0)
    metadata: dict[str, Any] = Field(default_factory=dict)


class CategoricalAxis(BaseModel):
    """An axis with a fixed set of mutually exclusive settings.

    Args:
        name: Key under which the selection is recorded.
        label: Prefix used when rendering the instruction, e.g. ``"Setting"`` becomes
            ``"Setting: STI outpatient clinic"``.
        options: The settings to choose between.
        skip_probability: Chance of leaving the axis unset, letting the model decide.
            Useful for axes that should only sometimes be constrained.
    """

    model_config = ConfigDict(frozen=True)

    name: str
    label: str | None = None
    options: tuple[VariationOption, ...]
    skip_probability: float = Field(default=0.0, ge=0.0, le=1.0)

    @model_validator(mode="after")
    def _options_present_and_unique(self) -> CategoricalAxis:
        if not self.options:
            raise ValueError(f"Axis {self.name!r} must define at least one option")
        values = [option.value for option in self.options]
        duplicates = {value for value in values if values.count(value) > 1}
        if duplicates:
            raise ValueError(
                f"Axis {self.name!r} has duplicate option values: {sorted(duplicates)}"
            )
        return self

    def sample(self, rng: random.Random) -> VariationOption | None:
        """Draw one option, or ``None`` if the axis is skipped this time."""
        if self.skip_probability and rng.random() < self.skip_probability:
            return None
        weights = [option.weight for option in self.options]
        return rng.choices(self.options, weights=weights, k=1)[0]

    def render(self, option: VariationOption) -> str | None:
        """Render ``option`` as a prompt instruction, or ``None`` if it has no text."""
        if option.instruction is None:
            return None
        if self.label:
            return f"{self.label}: {option.instruction}"
        return option.instruction


class FlagAxis(BaseModel):
    """A boolean axis included with a given probability.

    This models the "sometimes there is small talk" kind of variation, where the
    absence of the instruction is the interesting other case.

    Args:
        name: Key under which the outcome is recorded.
        instruction: Prompt-facing sentence, added only when the flag comes up true.
        probability: Chance of the flag being set.
    """

    model_config = ConfigDict(frozen=True)

    name: str
    instruction: str
    probability: float = Field(ge=0.0, le=1.0)

    def sample(self, rng: random.Random) -> bool:
        """Draw the flag."""
        return rng.random() < self.probability


class VariationConfig(BaseModel):
    """The sampled configuration for a single conversation.

    Recorded verbatim alongside each sample, which makes the dataset sliceable after
    the fact ("how does extraction fare on hesitant patients?").
    """

    model_config = ConfigDict(frozen=True)

    choices: dict[str, VariationOption] = Field(
        default_factory=dict, description="Selected option per categorical axis."
    )
    flags: dict[str, bool] = Field(default_factory=dict, description="Outcome per flag axis.")
    instructions: tuple[str, ...] = Field(
        default=(), description="Rendered prompt lines, in axis order."
    )

    def value(self, axis: str) -> str | None:
        """The selected value for ``axis``, or ``None`` if unset or skipped."""
        option = self.choices.get(axis)
        return option.value if option else None

    def metadata(self, axis: str, key: str, default: Any = None) -> Any:
        """Read ``key`` from the selected option's metadata on ``axis``."""
        option = self.choices.get(axis)
        if option is None:
            return default
        return option.metadata.get(key, default)

    def flag(self, axis: str) -> bool:
        """Whether flag ``axis`` came up true (``False`` if the axis is unknown)."""
        return self.flags.get(axis, False)

    def summary(self) -> dict[str, Any]:
        """A flat mapping of axis name to value, for logging and analysis."""
        summary: dict[str, Any] = {name: option.value for name, option in self.choices.items()}
        summary.update(self.flags)
        return summary


class VariationSpace(BaseModel):
    """The full set of axes sampled once per conversation."""

    model_config = ConfigDict(frozen=True)

    categorical: tuple[CategoricalAxis, ...] = ()
    flags: tuple[FlagAxis, ...] = ()

    @model_validator(mode="after")
    def _names_unique(self) -> VariationSpace:
        names = [axis.name for axis in self.categorical] + [axis.name for axis in self.flags]
        duplicates = {name for name in names if names.count(name) > 1}
        if duplicates:
            raise ValueError(f"Axis names must be unique across the space: {sorted(duplicates)}")
        return self

    def sample(self, rng: random.Random) -> VariationConfig:
        """Draw one configuration.

        All randomness comes from ``rng``, so a seeded generator reproduces the same
        dataset regardless of how the work is scheduled.
        """
        choices: dict[str, VariationOption] = {}
        instructions: list[str] = []

        for axis in self.categorical:
            option = axis.sample(rng)
            if option is None:
                continue
            choices[axis.name] = option
            rendered = axis.render(option)
            if rendered:
                instructions.append(rendered)

        flags: dict[str, bool] = {}
        for flag_axis in self.flags:
            enabled = flag_axis.sample(rng)
            flags[flag_axis.name] = enabled
            if enabled:
                instructions.append(flag_axis.instruction)

        return VariationConfig(choices=choices, flags=flags, instructions=tuple(instructions))

    def cardinality(self) -> int:
        """Count the distinct configurations this space can produce.

        Skippable axes contribute one extra state, flags contribute two. This counts
        the *designed* space; it says nothing about how distinguishable the resulting
        conversations actually are.
        """
        total = 1
        for axis in self.categorical:
            states = len(axis.options) + (1 if axis.skip_probability > 0 else 0)
            total *= states
        total *= 2 ** len(self.flags)
        return int(total)

    def describe(self) -> str:
        """A short human-readable summary of the space and its size."""
        lines = [f"{len(self.categorical)} categorical axes, {len(self.flags)} flags"]
        for axis in self.categorical:
            optional = " (skippable)" if axis.skip_probability > 0 else ""
            lines.append(f"  {axis.name}: {len(axis.options)} options{optional}")
        for flag_axis in self.flags:
            lines.append(f"  {flag_axis.name}: flag at p={flag_axis.probability:g}")
        magnitude = self.cardinality()
        exponent = int(math.log10(magnitude)) if magnitude > 0 else 0
        lines.append(f"  => {magnitude:,} configurations (~1e{exponent})")
        return "\n".join(lines)
