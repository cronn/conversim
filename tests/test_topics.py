"""Topic planning: activation, ordering, and the field-path guard."""

from __future__ import annotations

import random

import pytest

from conversim import (
    CategoricalAxis,
    PriorityRule,
    SchemaSpec,
    SchemaTopicPlanner,
    TopicCategory,
    VariationOption,
    VariationSpace,
)
from conversim.errors import ConfigurationError, SchemaError
from conversim.variation import VariationConfig

GREETING = TopicCategory(
    name="greeting", description="Greeting.", position="early", always_include=True
)
PROBLEM = TopicCategory(
    name="problem",
    description="What is failing.",
    field_paths=("problem.summary", "problem.first_seen"),
    priority=2,
)
SEVERITY = TopicCategory(
    name="severity", description="How bad it is.", field_paths=("problem.severity",), priority=3
)
RESOLUTION = TopicCategory(
    name="resolution",
    description="Next steps.",
    field_paths=("resolution.actions",),
    priority=9,
    position="late",
)

CATEGORIES = (GREETING, PROBLEM, SEVERITY, RESOLUTION)
EMPTY_VARIATION = VariationConfig()


def planner(schema: SchemaSpec, **kwargs: object) -> SchemaTopicPlanner:
    return SchemaTopicPlanner(schema, CATEGORIES, **kwargs)  # type: ignore[arg-type]


def test_only_categories_with_content_are_planned(ticket_schema: SchemaSpec) -> None:
    ground_truth = {
        "problem": {
            "summary": "printer offline",
            "first_seen": "not_discussed",
            "severity": "not_discussed",
        },
        "resolution": {"actions": []},
    }
    plan = planner(ticket_schema).plan(ground_truth, EMPTY_VARIATION, random.Random(0))
    assert plan.names == ("greeting", "problem")


def test_always_include_survives_an_empty_ground_truth(ticket_schema: SchemaSpec) -> None:
    plan = planner(ticket_schema).plan({}, EMPTY_VARIATION, random.Random(0))
    assert plan.names == ("greeting",)


def test_position_bands_pin_early_and_late(ticket_schema: SchemaSpec) -> None:
    ground_truth = {
        "problem": {"summary": "a", "first_seen": "b", "severity": "c"},
        "resolution": {"actions": ["x"]},
    }
    plan = planner(ticket_schema).plan(ground_truth, EMPTY_VARIATION, random.Random(0))
    assert plan.names[0] == "greeting"
    assert plan.names[-1] == "resolution"


def test_priority_rule_reorders_on_a_matching_axis(ticket_schema: SchemaSpec) -> None:
    space = VariationSpace(
        categorical=(
            CategoricalAxis(name="focus", options=(VariationOption(value="severity_first"),)),
        )
    )
    variation = space.sample(random.Random(0))
    ground_truth = {"problem": {"summary": "a", "first_seen": "b", "severity": "c"}}

    rules = (PriorityRule(axis="focus", value="severity_first", priorities={"severity": 1}),)
    ordered = planner(ticket_schema, priority_rules=rules).plan(
        ground_truth, variation, random.Random(0)
    )
    unruled = planner(ticket_schema).plan(ground_truth, variation, random.Random(0))

    assert ordered.names.index("severity") < ordered.names.index("problem")
    assert unruled.names.index("problem") < unruled.names.index("severity")


def test_priority_rule_ignored_when_the_axis_does_not_match(ticket_schema: SchemaSpec) -> None:
    rules = (PriorityRule(axis="focus", value="other", priorities={"severity": 1}),)
    ground_truth = {"problem": {"summary": "a", "first_seen": "b", "severity": "c"}}
    plan = planner(ticket_schema, priority_rules=rules).plan(
        ground_truth, EMPTY_VARIATION, random.Random(0)
    )
    assert plan.names.index("problem") < plan.names.index("severity")


def test_inclusion_probability_thins_optional_categories(ticket_schema: SchemaSpec) -> None:
    optional = TopicCategory(
        name="colour", description="Chit-chat.", always_include=True, inclusion_probability=0.5
    )
    plan_maker = SchemaTopicPlanner(ticket_schema, (GREETING, optional))
    outcomes = {
        "colour" in plan_maker.plan({}, EMPTY_VARIATION, random.Random(seed)).names
        for seed in range(30)
    }
    assert outcomes == {True, False}


def test_priority_jitter_moves_a_category_around(ticket_schema: SchemaSpec) -> None:
    floating = TopicCategory(
        name="floating", description="Varies.", always_include=True, priority_jitter=(1, 9)
    )
    plan_maker = SchemaTopicPlanner(ticket_schema, (PROBLEM, SEVERITY, RESOLUTION, floating))
    ground_truth = {
        "problem": {"summary": "a", "first_seen": "b", "severity": "c"},
        "resolution": {"actions": ["x"]},
    }
    positions = {
        plan_maker.plan(ground_truth, EMPTY_VARIATION, random.Random(seed)).names.index("floating")
        for seed in range(30)
    }
    assert len(positions) > 1


def test_planning_is_reproducible(ticket_schema: SchemaSpec) -> None:
    ground_truth = {"problem": {"summary": "a", "first_seen": "b", "severity": "c"}}
    plan_maker = planner(ticket_schema)
    first = plan_maker.plan(ground_truth, EMPTY_VARIATION, random.Random(5))
    second = plan_maker.plan(ground_truth, EMPTY_VARIATION, random.Random(5))
    assert first == second


def test_a_misspelled_field_path_fails_at_construction(ticket_schema: SchemaSpec) -> None:
    """The guard that the original pipeline lacked.

    Without it a category simply never activates, which is invisible at runtime and
    silently removes a whole subject from every generated conversation.
    """
    broken = TopicCategory(name="broken", description="x", field_paths=("problem.sevrity",))
    with pytest.raises(SchemaError, match="severity"):
        SchemaTopicPlanner(ticket_schema, (broken,))


def test_a_rule_targeting_an_unknown_category_is_rejected(ticket_schema: SchemaSpec) -> None:
    rules = (PriorityRule(axis="focus", value="v", priorities={"nonexistent": 1}),)
    with pytest.raises(ConfigurationError, match="nonexistent"):
        planner(ticket_schema, priority_rules=rules)


def test_duplicate_category_names_are_rejected(ticket_schema: SchemaSpec) -> None:
    with pytest.raises(ConfigurationError, match="Duplicate"):
        SchemaTopicPlanner(ticket_schema, (GREETING, GREETING))


def test_a_category_that_can_never_activate_is_rejected() -> None:
    with pytest.raises(ValueError, match="never activate"):
        TopicCategory(name="dead", description="x")


def test_no_categories_yields_an_empty_plan(ticket_schema: SchemaSpec) -> None:
    plan = SchemaTopicPlanner(ticket_schema, ()).plan({}, EMPTY_VARIATION, random.Random(0))
    assert not plan
    assert plan.descriptions == ()
