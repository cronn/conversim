"""Casting and scenario strategies."""

from __future__ import annotations

import random

import pytest

from conversim import (
    Cast,
    ExhaustiveCasting,
    FixedCasting,
    Persona,
    RandomCasting,
    StaticScenarioBuilder,
    TurnTarget,
)
from conversim.errors import ConfigurationError
from conversim.variation import (
    CategoricalAxis,
    VariationConfig,
    VariationOption,
    VariationSpace,
)

AGENTS = tuple(Persona(name=f"Agent{i}", role="agent") for i in range(3))
CUSTOMERS = tuple(Persona(name=f"Cust{i}", role="customer") for i in range(2))
POOLS = {"agent": AGENTS, "customer": CUSTOMERS}


def test_fixed_casting_always_returns_the_same_cast() -> None:
    cast = Cast(personas=(AGENTS[0], CUSTOMERS[0]))
    strategy = FixedCasting(value=cast)
    assert strategy.cast(index=0, rng=random.Random(0)) == cast
    assert strategy.cast(index=9, rng=random.Random(9)) == cast


def test_random_casting_draws_one_per_role() -> None:
    strategy = RandomCasting(pools=POOLS)
    for seed in range(10):
        cast = strategy.cast(index=seed, rng=random.Random(seed))
        assert sorted(cast.role_ids) == ["agent", "customer"]


def test_random_casting_is_reproducible() -> None:
    strategy = RandomCasting(pools=POOLS)
    assert strategy.cast(index=0, rng=random.Random(4)) == strategy.cast(
        index=0, rng=random.Random(4)
    )


def test_random_casting_varies() -> None:
    strategy = RandomCasting(pools=POOLS)
    seen = {strategy.cast(index=s, rng=random.Random(s)).names for s in range(30)}
    assert len(seen) > 1


def test_exhaustive_casting_walks_every_combination() -> None:
    strategy = ExhaustiveCasting(pools=POOLS)
    assert strategy.combination_count == 6
    seen = {strategy.cast(index=i, rng=random.Random(0)).names for i in range(6)}
    assert len(seen) == 6


def test_exhaustive_casting_cycles_past_the_end() -> None:
    strategy = ExhaustiveCasting(pools=POOLS)
    assert strategy.cast(index=0, rng=random.Random(0)) == strategy.cast(
        index=6, rng=random.Random(0)
    )


def test_a_persona_in_the_wrong_pool_is_rejected() -> None:
    with pytest.raises(ConfigurationError, match="different role"):
        RandomCasting(pools={"agent": CUSTOMERS})


def test_an_empty_pool_is_rejected() -> None:
    with pytest.raises(ConfigurationError, match="is empty"):
        RandomCasting(pools={"agent": ()})


def test_no_pools_is_rejected() -> None:
    with pytest.raises(ConfigurationError, match="at least one role pool"):
        RandomCasting(pools={})


def test_static_scenario_is_constant() -> None:
    builder = StaticScenarioBuilder(topic="t", context="c", language="English")
    scenario = builder.build(variation=VariationConfig(), rng=random.Random(0))
    assert (scenario.topic, scenario.context, scenario.language) == ("t", "c", "English")
    assert scenario.turns == TurnTarget(minimum=20, maximum=35)


def test_static_scenario_can_take_its_length_from_an_axis() -> None:
    space = VariationSpace(
        categorical=(
            CategoricalAxis(
                name="style",
                options=(VariationOption(value="brief", metadata={"turn_range": (4, 6)}),),
            ),
        )
    )
    variation = space.sample(random.Random(0))
    builder = StaticScenarioBuilder(topic="t", context="c", turn_range_axis="style")
    scenario = builder.build(variation=variation, rng=random.Random(0))
    assert scenario.turns == TurnTarget(minimum=4, maximum=6)


def test_axis_without_turn_range_falls_back() -> None:
    builder = StaticScenarioBuilder(
        topic="t", context="c", turns=TurnTarget(minimum=9, maximum=9), turn_range_axis="absent"
    )
    scenario = builder.build(variation=VariationConfig(), rng=random.Random(0))
    assert scenario.turns == TurnTarget(minimum=9, maximum=9)


def test_cast_order_follows_pool_declaration_order() -> None:
    """So that Cast.names lines up with the roles in the DatasetSpec, not the alphabet."""
    declared = {"customer": CUSTOMERS, "agent": AGENTS}
    cast = RandomCasting(pools=declared).cast(index=0, rng=random.Random(0))
    assert cast.role_ids == ("customer", "agent")

    exhaustive = ExhaustiveCasting(pools=declared).cast(index=0, rng=random.Random(0))
    assert exhaustive.role_ids == ("customer", "agent")
