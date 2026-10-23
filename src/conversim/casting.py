"""Choosing who plays which role, and in what situation.

Both concerns are strategies rather than settings because the interesting variants are
structurally different: drawing personas at random is not a parameterisation of
enumerating every pairing, and a scenario derived from sampled variation is not a
parameterisation of a fixed one.

All strategies take the sample index and a seeded generator, which is what keeps a run
reproducible even though samples complete out of order under concurrency.

Pool ordering is significant: the personas in a cast follow the order the role pools
were declared in, so declaring ``{"host": ..., "guest": ...}`` puts the host first.
Dictionaries preserve insertion order, so this is as deterministic as sorting while
matching what the caller wrote.
"""

from __future__ import annotations

import itertools
import random
from typing import Protocol, runtime_checkable

from pydantic import BaseModel, ConfigDict, Field, model_validator

from .errors import ConfigurationError
from .models import Cast, Persona, Scenario, TurnTarget
from .variation import VariationConfig

__all__ = [
    "CastingStrategy",
    "ExhaustiveCasting",
    "FixedCasting",
    "RandomCasting",
    "ScenarioBuilder",
    "StaticScenarioBuilder",
]


@runtime_checkable
class CastingStrategy(Protocol):
    """Chooses the personas for one conversation."""

    def cast(self, *, index: int, rng: random.Random) -> Cast:
        """Return the cast for sample ``index``."""
        ...


class FixedCasting(BaseModel):
    """Always use the same cast."""

    model_config = ConfigDict(frozen=True)

    value: Cast

    def cast(self, *, index: int, rng: random.Random) -> Cast:
        """Return the configured cast."""
        return self.value


class RandomCasting(BaseModel):
    """Draw one persona per role from per-role pools.

    Args:
        pools: Candidate personas per role id. Every persona in a pool must declare
            that role.

    Raises:
        ConfigurationError: if a pool is empty or contains a persona for another role.
    """

    model_config = ConfigDict(frozen=True)

    pools: dict[str, tuple[Persona, ...]]

    @model_validator(mode="after")
    def _pools_consistent(self) -> RandomCasting:
        _validate_pools(self.pools)
        return self

    def cast(self, *, index: int, rng: random.Random) -> Cast:
        """Draw one persona per role, in the order the pools were declared."""
        return Cast(personas=tuple(rng.choice(personas) for personas in self.pools.values()))


class ExhaustiveCasting(BaseModel):
    """Walk every combination of personas across roles, in a stable order.

    Use this when coverage matters more than independence -- for example to guarantee
    each persona pairing appears equally often. With ``shuffle`` the order is permuted
    once per run so that a truncated run is not biased toward the first personas in
    each pool, while still covering combinations uniformly.

    Args:
        pools: Candidate personas per role id.
        shuffle: Permute the combination order using the sample's generator.

    Raises:
        ConfigurationError: if a pool is empty or contains a persona for another role.
    """

    model_config = ConfigDict(frozen=True)

    pools: dict[str, tuple[Persona, ...]]
    shuffle: bool = False

    @model_validator(mode="after")
    def _pools_consistent(self) -> ExhaustiveCasting:
        _validate_pools(self.pools)
        return self

    @property
    def combination_count(self) -> int:
        """How many distinct casts exist."""
        total = 1
        for personas in self.pools.values():
            total *= len(personas)
        return total

    def combinations(self) -> tuple[Cast, ...]:
        """Every cast, in a stable order following how the pools were declared."""
        return tuple(
            Cast(personas=tuple(combination))
            for combination in itertools.product(*self.pools.values())
        )

    def cast(self, *, index: int, rng: random.Random) -> Cast:
        """Return the cast for ``index``, cycling once the combinations run out."""
        options = self.combinations()
        if self.shuffle:
            position = rng.randrange(len(options))
            return options[position]
        return options[index % len(options)]


def _validate_pools(pools: dict[str, tuple[Persona, ...]]) -> None:
    if not pools:
        raise ConfigurationError("Casting needs at least one role pool")
    for role, personas in pools.items():
        if not personas:
            raise ConfigurationError(f"Persona pool for role {role!r} is empty")
        mismatched = [persona.name for persona in personas if persona.role != role]
        if mismatched:
            raise ConfigurationError(
                f"Personas in the {role!r} pool declare a different role: {', '.join(mismatched)}"
            )


@runtime_checkable
class ScenarioBuilder(Protocol):
    """Derives the situational frame for one conversation."""

    def build(self, *, variation: VariationConfig, rng: random.Random) -> Scenario:
        """Return the scenario for one conversation."""
        ...


class StaticScenarioBuilder(BaseModel):
    """Use one fixed scenario for every conversation.

    Sensible when the variation space carries all the situational variety, or as a
    starting point before writing a builder that reads sampled variation.

    Args:
        topic: What every conversation is about.
        context: Setting and circumstances.
        language: Target language name.
        turns: Requested dialogue length.
        turn_range_axis: Optionally take the turn range from a variation axis's option
            metadata instead, under the key ``"turn_range"`` as ``(min, max)``. This
            covers the common case where a "concise vs. detailed" axis should also
            determine length, without needing a custom builder.
    """

    model_config = ConfigDict(frozen=True)

    topic: str
    context: str
    language: str = "German"
    turns: TurnTarget = Field(default_factory=lambda: TurnTarget(minimum=20, maximum=35))
    turn_range_axis: str | None = None

    def build(self, *, variation: VariationConfig, rng: random.Random) -> Scenario:
        """Return the configured scenario, honouring ``turn_range_axis`` when set."""
        turns = self.turns
        if self.turn_range_axis:
            span = variation.metadata(self.turn_range_axis, "turn_range")
            if span is not None:
                minimum, maximum = span
                turns = TurnTarget(minimum=int(minimum), maximum=int(maximum))
        return Scenario(
            topic=self.topic,
            context=self.context,
            language=self.language,
            turns=turns,
        )
