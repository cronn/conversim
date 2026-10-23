"""A complete domain pack: German STI/HIV history-taking consultations.

This is the worked example that the domain-agnostic core is built around, and the
reference for assembling your own. A domain pack is five things:

1. a ground-truth schema (:mod:`~conversim.presets.sti.schema`),
2. a variation space (:mod:`~conversim.presets.sti.variation`),
3. topic categories tied to schema fields (:mod:`~conversim.presets.sti.topics`),
4. roles and personas (:mod:`~conversim.presets.sti.personas`),
5. localised prompt builders (:mod:`~conversim.presets.sti.prompts`).

Usage::

    from conversim import GenerationConfig, generate_dataset
    from conversim.presets.sti import sti_dataset_spec, sti_prompt_builders
    from conversim.providers import GoogleProvider, OpenAIProvider

    ground_truth_prompt, dialogue_prompt = sti_prompt_builders()
    generate_dataset(
        "out/sti",
        spec=sti_dataset_spec(),
        structured_provider=GoogleProvider(model="gemini-2.5-flash", api_key=...),
        text_provider=OpenAIProvider(model="gpt-5-mini", api_key=...),
        config=GenerationConfig(count=100, seed=1, concurrency=8),
        ground_truth_prompt=ground_truth_prompt,
        dialogue_prompt=dialogue_prompt,
    )

A note on the subject matter: everything generated here is fictitious by construction.
No real patient data is involved at any point, which is the reason this approach exists
-- the domain's privacy constraints rule out training on real consultations.
"""

from __future__ import annotations

from ...casting import CastingStrategy, ExhaustiveCasting, RandomCasting
from ...pipeline import DatasetSpec
from ...schema import SchemaSpec
from .personas import (
    CLINICIAN,
    CLINICIAN_PERSONAS,
    PATIENT,
    PATIENT_PERSONAS,
    STI_ROLES,
)
from .prompts import StiDialoguePrompt, StiGroundTruthPrompt
from .scenario import StiScenarioBuilder
from .schema import StiAnamnesis
from .topics import STI_PRIORITY_RULES, STI_TOPIC_CATEGORIES, sti_topic_planner
from .variation import sti_variation_space

__all__ = [
    "CLINICIAN",
    "CLINICIAN_PERSONAS",
    "PATIENT",
    "PATIENT_PERSONAS",
    "STI_PRIORITY_RULES",
    "STI_ROLES",
    "STI_TOPIC_CATEGORIES",
    "StiAnamnesis",
    "StiDialoguePrompt",
    "StiGroundTruthPrompt",
    "StiScenarioBuilder",
    "sti_dataset_spec",
    "sti_prompt_builders",
    "sti_schema_spec",
    "sti_topic_planner",
    "sti_variation_space",
]

_SENTINEL_DOCS = {
    "not_discussed": "Das Thema kam im Gespräch nicht zur Sprache.",
    "unknown": "Das Thema kam zur Sprache, die Patient:in weiß es aber nicht.",
    "not_applicable": "Trifft auf diese Patient:in sachlich nicht zu.",
}


def sti_schema_spec() -> SchemaSpec:
    """The STI schema together with its sentinel conventions.

    Returns:
        A :class:`~conversim.schema.SchemaSpec` over
        :class:`~conversim.presets.sti.schema.StiAnamnesis`.
    """
    return SchemaSpec(model=StiAnamnesis, sentinel_docs=_SENTINEL_DOCS)


def sti_dataset_spec(
    *,
    casting: CastingStrategy | None = None,
    exhaustive_casting: bool = False,
    name: str | None = "sti-anamnesis",
    description: str | None = (
        "Synthetic German STI/HIV history-taking consultations with schema-conformant ground truth."
    ),
) -> DatasetSpec:
    """Assemble the full STI dataset specification.

    Args:
        casting: Override how personas are chosen.
        exhaustive_casting: Walk all 40 clinician-patient pairings in a stable order
            instead of drawing at random. Ignored when ``casting`` is given.
        name: Dataset name recorded in the manifest.
        description: Dataset description recorded in the manifest.

    Returns:
        A specification ready to hand to
        :func:`~conversim.pipeline.generate_dataset`.

    Example:
        >>> spec = sti_dataset_spec()
        >>> spec.schema.name
        'StiAnamnesis'
        >>> spec.variation.cardinality()
        1270080
    """
    schema = sti_schema_spec()
    pools = {"clinician": CLINICIAN_PERSONAS, "patient": PATIENT_PERSONAS}

    if casting is None:
        casting = (
            ExhaustiveCasting(pools=pools) if exhaustive_casting else RandomCasting(pools=pools)
        )

    return DatasetSpec(
        schema=schema,
        roles=STI_ROLES,
        casting=casting,
        scenario=StiScenarioBuilder(),
        variation=sti_variation_space(),
        topic_planner=sti_topic_planner(schema),
        name=name,
        description=description,
    )


def sti_prompt_builders() -> tuple[StiGroundTruthPrompt, StiDialoguePrompt]:
    """The German prompt builders for both stages.

    Returns:
        A ``(ground_truth, dialogue)`` pair, ready to pass to
        :func:`~conversim.pipeline.generate_dataset`.
    """
    return StiGroundTruthPrompt(), StiDialoguePrompt()
