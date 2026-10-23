"""Building your own domain pack, with designed diversity and topic planning.

A domain pack is five things: a schema, a variation space, topic categories bound to
schema fields, roles with personas, and (optionally) prompt builders in your language.
This example builds all five for job interviews. Needs API keys.

    uv run --extra openai --extra google python examples/04_custom_domain.py
"""

from __future__ import annotations

import logging

from pydantic import BaseModel, Field

from conversim import (
    CategoricalAxis,
    DatasetSpec,
    ExhaustiveCasting,
    FlagAxis,
    GenerationConfig,
    Persona,
    PriorityRule,
    Role,
    SchemaSpec,
    SchemaTopicPlanner,
    StaticScenarioBuilder,
    TopicCategory,
    VariationOption,
    VariationSpace,
    generate_dataset,
)
from conversim.providers import GoogleProvider, OpenAIProvider

# -- 1. the schema ----------------------------------------------------------
# Every categorical field carries "not_discussed". That is what lets a short
# interview be short: the dialogue prompt is told never to mention such fields.


class Experience(BaseModel):
    """The candidate's background."""

    years_total: int | None = Field(default=None, description="Total years of experience.")
    current_role: str = Field(default="not_discussed", description="Current job title.")
    notable_project: str = Field(default="not_discussed", description="A project discussed.")


class Compensation(BaseModel):
    """Money, where it came up."""

    expectation_discussed: str = Field(
        default="not_discussed", description="Did compensation come up? yes/no."
    )
    range_stated: str = Field(default="not_discussed", description="Range mentioned, if any.")


class InterviewRecord(BaseModel):
    """What a screening interview should yield."""

    role_applied_for: str = Field(default="not_discussed", description="The open position.")
    experience: Experience = Field(default_factory=Experience)
    skills_confirmed: list[str] = Field(
        default_factory=list, description="Skills the candidate demonstrated."
    )
    compensation: Compensation = Field(default_factory=Compensation)
    next_step: str = Field(default="not_discussed", description="Agreed next step.")


# -- 2. designed diversity --------------------------------------------------
# Each axis is injected into both generation stages, so ground truth and dialogue
# agree. The style axis also carries the turn range, keeping length and content
# from pulling in opposite directions.

VARIATION = VariationSpace(
    categorical=(
        CategoricalAxis(
            name="style",
            label="Interview style",
            options=(
                VariationOption(
                    value="screening",
                    instruction="a brisk 15-minute phone screen",
                    metadata={"turn_range": (10, 16)},
                ),
                VariationOption(
                    value="deep_dive",
                    instruction="a thorough technical deep dive",
                    metadata={"turn_range": (28, 40)},
                ),
            ),
        ),
        CategoricalAxis(
            name="seniority",
            label="Candidate seniority",
            options=(
                VariationOption(value="junior", instruction="early career, one or two roles"),
                VariationOption(value="mid", instruction="mid level, several years in"),
                VariationOption(value="senior", instruction="senior, has led teams"),
            ),
        ),
        CategoricalAxis(
            name="candidate_manner",
            label="Candidate manner",
            options=(
                VariationOption(value="confident", instruction="confident, volunteers detail"),
                VariationOption(value="nervous", instruction="nervous, needs prompting"),
                VariationOption(value="terse", instruction="terse, answers only what is asked"),
            ),
        ),
    ),
    flags=(
        FlagAxis(
            name="salary_talk",
            instruction="Compensation comes up explicitly.",
            probability=0.4,
        ),
        FlagAxis(
            name="candidate_questions",
            instruction="The candidate asks their own questions near the end.",
            probability=0.5,
        ),
    ),
)


# -- 3. topic planning ------------------------------------------------------
# Field paths are validated against the schema when the planner is constructed, so a
# typo raises instead of silently deactivating a whole subject.

CATEGORIES = (
    TopicCategory(
        name="opening",
        description="Greeting, introductions, what the role is",
        position="early",
        always_include=True,
        priority=1,
    ),
    TopicCategory(
        name="background",
        description="The candidate's experience so far and current role",
        field_paths=("experience.years_total", "experience.current_role"),
        priority=3,
    ),
    TopicCategory(
        name="project",
        description="A concrete project the candidate worked on, in some depth",
        field_paths=("experience.notable_project",),
        priority=4,
    ),
    TopicCategory(
        name="skills",
        description="Specific skills probed and demonstrated",
        field_paths=("skills_confirmed",),
        priority=5,
    ),
    TopicCategory(
        name="compensation",
        description="Salary expectations, if either side raises them",
        field_paths=("compensation.expectation_discussed", "compensation.range_stated"),
        priority=8,
    ),
    TopicCategory(
        name="closing",
        description="What happens next and when",
        field_paths=("next_step",),
        position="late",
        priority=10,
    ),
)

RULES = (
    # A deep dive front-loads the substance.
    PriorityRule(axis="style", value="deep_dive", priorities={"project": 2, "skills": 3}),
)


def main() -> None:
    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(name)s: %(message)s")

    schema = SchemaSpec(
        InterviewRecord,
        sentinel_docs={
            "not_discussed": "The topic never came up in the interview.",
            "unknown": "It came up, but no clear answer was given.",
            "not_applicable": "It does not apply to this candidate.",
        },
    )

    interviewers = (
        Persona(name="Dana", role="interviewer", description="Warm, asks follow-ups."),
        Persona(name="Ravi", role="interviewer", description="Structured, keeps to time."),
    )
    candidates = (
        Persona(name="Jo", role="candidate", description="Talks in concrete examples."),
        Persona(name="Lin", role="candidate", description="Abstract, needs grounding."),
    )

    spec = DatasetSpec(
        schema=schema,
        roles=(
            Role(id="interviewer", label="Interviewer", description="Runs the interview."),
            Role(id="candidate", label="Candidate", description="Applying for the role."),
        ),
        # Exhaustive casting guarantees each pairing appears equally often.
        casting=ExhaustiveCasting(pools={"interviewer": interviewers, "candidate": candidates}),
        scenario=StaticScenarioBuilder(
            topic="a screening interview for a backend engineering role",
            context="Video call, first round.",
            language="English",
            turn_range_axis="style",
        ),
        variation=VARIATION,
        topic_planner=SchemaTopicPlanner(schema, CATEGORIES, RULES),
        name="interviews",
        description="Synthetic screening interviews with structured outcomes.",
    )

    print(spec.variation.describe())

    result = generate_dataset(
        "out/interviews",
        spec=spec,
        structured_provider=GoogleProvider(model="gemini-2.5-flash", thinking_budget=0),
        text_provider=OpenAIProvider(model="gpt-5-mini"),
        config=GenerationConfig(count=8, seed=1, concurrency=4),
    )
    print(result.summary())


if __name__ == "__main__":
    main()
