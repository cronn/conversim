"""Prompt construction.

Prompts are the part of a generation pipeline most in need of domain knowledge and
most resistant to abstraction, so conversim keeps them behind two small protocols
and ships workable defaults rather than a templating language. Overriding means
writing a class with one method, in Python, with a typed context object -- which is
easier to reason about than a template DSL once the prompt starts branching on
sampled variation.

The defaults here are domain-neutral and instruct in English while asking for output
in :attr:`~conversim.models.Scenario.language`. For a fully localised, domain-tuned
example see :mod:`conversim.presets.sti`.
"""

from __future__ import annotations

import json
import random
from dataclasses import dataclass
from typing import Protocol, runtime_checkable

from pydantic import BaseModel

from .models import Cast, Role, Scenario
from .schema import SchemaSpec
from .topics import TopicPlan
from .variation import VariationConfig

__all__ = [
    "DefaultDialoguePromptBuilder",
    "DefaultGroundTruthPromptBuilder",
    "DialoguePromptBuilder",
    "DialoguePromptContext",
    "GroundTruthPromptBuilder",
    "GroundTruthPromptContext",
]


@dataclass(frozen=True)
class GroundTruthPromptContext:
    """Everything available when asking a model to invent the facts.

    Attributes:
        schema: The target schema and its sentinel conventions.
        scenario: Topic, context, language and requested length.
        variation: This conversation's sampled variation configuration.
        rng: Seeded generator. Use it for any randomised phrasing, so that the prompt
            itself remains reproducible.
    """

    schema: SchemaSpec
    scenario: Scenario
    variation: VariationConfig
    rng: random.Random


@dataclass(frozen=True)
class DialoguePromptContext:
    """Everything available when asking a model to write the conversation.

    Attributes:
        schema: The target schema and its sentinel conventions.
        scenario: Topic, context, language and requested length.
        roles: Structural roles taking part, in order.
        cast: The personas filling those roles.
        ground_truth: The facts the dialogue must convey, already validated.
        variation: This conversation's sampled variation configuration.
        topic_plan: Suggested topic sequence derived from the ground truth.
        include_cues: Whether to invite ``(cue)`` delivery annotations. Only useful
            when the speech backend can act on them.
        rng: Seeded generator for randomised phrasing.
    """

    schema: SchemaSpec
    scenario: Scenario
    roles: tuple[Role, ...]
    cast: Cast
    ground_truth: BaseModel
    variation: VariationConfig
    topic_plan: TopicPlan
    include_cues: bool
    rng: random.Random

    def ground_truth_json(self) -> str:
        """The ground truth as indented JSON, ready to embed in a prompt."""
        return json.dumps(
            self.ground_truth.model_dump(mode="json"),
            indent=2,
            ensure_ascii=False,
            default=str,
        )


@runtime_checkable
class GroundTruthPromptBuilder(Protocol):
    """Builds the prompt that produces a ground-truth instance."""

    def build(self, context: GroundTruthPromptContext) -> str:
        """Render the prompt."""
        ...


@runtime_checkable
class DialoguePromptBuilder(Protocol):
    """Builds the prompt that produces a dialogue script."""

    def build(self, context: DialoguePromptContext) -> str:
        """Render the prompt."""
        ...


class DefaultGroundTruthPromptBuilder:
    """Domain-neutral ground-truth prompt.

    Emphasises the point that makes generated ground truth usable: a realistic
    instance is mostly *incomplete*. A model asked to fill a schema will fill all of
    it, which then forces the dialogue stage to cover every field and produces
    interrogation transcripts. Instructing liberal sentinel use is what keeps the
    resulting conversations plausibly short.

    Args:
        extra_instructions: Additional lines appended to the guidance section.
    """

    def __init__(self, *, extra_instructions: tuple[str, ...] = ()) -> None:
        self.extra_instructions = extra_instructions

    def build(self, context: GroundTruthPromptContext) -> str:
        """Render the prompt."""
        scenario = context.scenario
        sections = [
            f"Produce one realistic instance of the {context.schema.name} schema, "
            f"describing the outcome of a single conversation.",
            "",
            "SETTING",
            f"- Topic: {scenario.topic}",
            f"- Context: {scenario.context}",
            f"- Language for all free-text values: {scenario.language}",
        ]

        if context.variation.instructions:
            sections += ["", "CONSTRAINTS FOR THIS INSTANCE"]
            sections += [f"- {line}" for line in context.variation.instructions]

        sections += [
            "",
            "GUIDANCE",
            "- Fill only what a real conversation of this kind would actually cover.",
            "- Values must be mutually consistent; never contradict yourself.",
            "- Do not pad list fields. Prefer a few plausible entries over many.",
        ]
        if context.schema.sentinels:
            sections += [
                "- Use the following reserved values instead of inventing content:",
                context.schema.sentinel_guide(),
            ]
        sections += [f"- {line}" for line in self.extra_instructions]

        field_guide = context.schema.field_guide()
        if field_guide:
            sections += ["", "FIELD REFERENCE", field_guide]

        return "\n".join(sections)


class DefaultDialoguePromptBuilder:
    """Domain-neutral dialogue prompt.

    The instruction set is built around one asymmetry: every substantive ground-truth
    value must be recoverable from the dialogue, while sentinel-valued fields must not
    be mentioned at all. Without the second half, models cover the whole schema by
    saying "we didn't discuss that", which is both unnatural and unlearnable.

    Args:
        extra_instructions: Additional lines appended to the realism section.
    """

    def __init__(self, *, extra_instructions: tuple[str, ...] = ()) -> None:
        self.extra_instructions = extra_instructions

    def build(self, context: DialoguePromptContext) -> str:
        """Render the prompt."""
        scenario = context.scenario
        sections = [
            f"Write a realistic conversation in {scenario.language}.",
            "",
            "PURPOSE",
            "This conversation is training data for a system that extracts structured "
            "records from transcripts. It must sound like people talking, not like a "
            "form being filled in.",
            "",
            "SETTING",
            f"- Topic: {scenario.topic}",
            f"- Context: {scenario.context}",
            f"- Target length: {scenario.turns.describe()} turns",
            "",
            "PARTICIPANTS",
        ]

        for role in context.roles:
            persona = context.cast.by_role(role.id)
            line = f"- {role.label} -- {persona.name}"
            if role.description:
                line += f"\n    Role: {role.description}"
            if persona.description:
                line += f"\n    Character: {persona.description}"
            sections.append(line)

        if context.variation.instructions:
            sections += ["", "SPECIFICS FOR THIS CONVERSATION"]
            sections += [f"- {line}" for line in context.variation.instructions]

        sections += [
            "",
            "GROUND TRUTH",
            "These are the facts the conversation conveys. Never contradict them.",
            "```json",
            context.ground_truth_json(),
            "```",
            "",
            "HOW TO USE THE GROUND TRUTH",
            "- Every value that is not a reserved placeholder must be recoverable from "
            "the dialogue, stated explicitly or unambiguously implied.",
            "- Every list entry must come up.",
        ]
        if context.schema.sentinels:
            reserved = ", ".join(f"`{value}`" for value in sorted(context.schema.sentinels))
            sections.append(
                f"- Fields set to a reserved placeholder ({reserved}) must NOT be "
                f"mentioned at all. Skipping them is what keeps the conversation short."
            )
        sections.append("- Convey facts in natural phrasing. Do not read values out verbatim.")

        if context.topic_plan:
            sections += ["", "SUGGESTED TOPIC ORDER (guidance, not a script)"]
            sections += [
                f"{position}. {description}"
                for position, description in enumerate(context.topic_plan.descriptions, 1)
            ]
            sections.append("Natural flow takes precedence; topics may blend or overlap.")

        sections += ["", "REALISM"]
        sections += [
            "- Speakers may hesitate, backtrack, digress and misunderstand.",
            "- Background detail beyond the ground truth is welcome as long as it does "
            "not conflict with it.",
            "- Avoid long runs of yes/no exchanges.",
        ]
        sections += [f"- {line}" for line in self.extra_instructions]

        speaker_names = " or ".join(f"`{name}: ...`" for name in context.cast.names)
        sections += [
            "",
            "OUTPUT FORMAT",
            f"- One turn per line, prefixed with the speaker name: {speaker_names}",
            "- Use only these speaker names, exactly as written.",
        ]
        if context.include_cues:
            sections.append(
                "- A short delivery cue in parentheses may precede the text, "
                "e.g. `Name: (quietly) ...`. Use it only where it genuinely helps."
            )
        sections += [
            "- No narration, no stage directions, no headings, no JSON.",
            f"- At least {scenario.turns.minimum} turns.",
            "",
            "Output only the dialogue lines:",
        ]

        return "\n".join(sections)
