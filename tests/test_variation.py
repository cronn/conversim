"""Variation sampling: reproducibility, weights, cardinality."""

from __future__ import annotations

import random

import pytest

from conversim import CategoricalAxis, FlagAxis, VariationOption, VariationSpace

STYLE = CategoricalAxis(
    name="style",
    label="Register",
    options=(
        VariationOption(value="concise", instruction="brisk", metadata={"turn_range": (10, 15)}),
        VariationOption(
            value="detailed", instruction="thorough", metadata={"turn_range": (30, 40)}
        ),
    ),
)
MOOD = CategoricalAxis(
    name="mood",
    options=(VariationOption(value="calm"), VariationOption(value="tense")),
    skip_probability=0.5,
)
SMALL_TALK = FlagAxis(name="small_talk", instruction="Opens with small talk.", probability=0.5)


@pytest.fixture
def space() -> VariationSpace:
    return VariationSpace(categorical=(STYLE, MOOD), flags=(SMALL_TALK,))


def test_same_seed_gives_the_same_configuration(space: VariationSpace) -> None:
    first = space.sample(random.Random(11))
    second = space.sample(random.Random(11))
    assert first == second


def test_different_seeds_diverge(space: VariationSpace) -> None:
    configurations = {
        tuple(sorted(space.sample(random.Random(seed)).summary().items())) for seed in range(40)
    }
    assert len(configurations) > 1


def test_instructions_carry_the_axis_label(space: VariationSpace) -> None:
    config = space.sample(random.Random(3))
    style = config.value("style")
    assert style in {"concise", "detailed"}
    assert any(line.startswith("Register: ") for line in config.instructions)


def test_options_without_instruction_stay_out_of_the_prompt(space: VariationSpace) -> None:
    for seed in range(30):
        config = space.sample(random.Random(seed))
        if config.value("mood") is not None:
            assert not any("calm" in line or "tense" in line for line in config.instructions)
            return
    pytest.fail("mood was skipped in all 30 draws")


def test_metadata_is_reachable(space: VariationSpace) -> None:
    config = space.sample(random.Random(3))
    assert config.metadata("style", "turn_range") in {(10, 15), (30, 40)}
    assert config.metadata("style", "absent") is None
    assert config.metadata("nonexistent", "turn_range", "fallback") == "fallback"


def test_flag_appears_in_instructions_only_when_set(space: VariationSpace) -> None:
    for seed in range(30):
        config = space.sample(random.Random(seed))
        has_instruction = SMALL_TALK.instruction in config.instructions
        assert config.flag("small_talk") is has_instruction


def test_skip_probability_sometimes_omits_the_axis(space: VariationSpace) -> None:
    outcomes = {space.sample(random.Random(seed)).value("mood") is None for seed in range(40)}
    assert outcomes == {True, False}


def test_cardinality_counts_skip_and_flag_states(space: VariationSpace) -> None:
    # style(2) * mood(2 + 1 skip) * small_talk(2)
    assert space.cardinality() == 2 * 3 * 2


def test_weights_bias_sampling() -> None:
    space = VariationSpace(
        categorical=(
            CategoricalAxis(
                name="axis",
                options=(
                    VariationOption(value="rare", weight=1.0),
                    VariationOption(value="common", weight=99.0),
                ),
            ),
        )
    )
    rng = random.Random(0)
    draws = [space.sample(rng).value("axis") for _ in range(400)]
    assert draws.count("common") > draws.count("rare") * 5


def test_duplicate_axis_names_rejected() -> None:
    axis = CategoricalAxis(name="dup", options=(VariationOption(value="a"),))
    with pytest.raises(ValueError, match="unique"):
        VariationSpace(categorical=(axis, axis))


def test_duplicate_option_values_rejected() -> None:
    with pytest.raises(ValueError, match="duplicate option"):
        CategoricalAxis(
            name="axis",
            options=(VariationOption(value="same"), VariationOption(value="same")),
        )


def test_describe_mentions_every_axis(space: VariationSpace) -> None:
    described = space.describe()
    assert "style" in described
    assert "mood" in described
    assert "small_talk" in described
    assert "configurations" in described
