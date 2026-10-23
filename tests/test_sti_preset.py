"""The STI domain pack: schema, planner, prompts."""

from __future__ import annotations

import json
import random

import pytest

from conversim import Cast, TopicCategory
from conversim.errors import SchemaError
from conversim.presets.sti import (
    CLINICIAN_PERSONAS,
    PATIENT_PERSONAS,
    STI_ROLES,
    StiAnamnesis,
    StiDialoguePrompt,
    StiGroundTruthPrompt,
    StiScenarioBuilder,
    sti_dataset_spec,
    sti_schema_spec,
    sti_topic_planner,
    sti_variation_space,
)
from conversim.prompts import DialoguePromptContext, GroundTruthPromptContext
from conversim.topics import SchemaTopicPlanner


def test_spec_assembles() -> None:
    spec = sti_dataset_spec()
    assert spec.schema.name == "StiAnamnesis"
    assert [role.id for role in spec.roles] == ["clinician", "patient"]
    assert spec.variation.cardinality() > 1_000_000


def test_every_topic_field_path_exists() -> None:
    """The check that the original pipeline lacked; five categories were broken there."""
    planner = sti_topic_planner()
    assert len(planner.categories) == 15


def test_a_typo_in_a_field_path_is_caught() -> None:
    broken = TopicCategory(
        name="symptoms",
        description="x",
        # The name the original pipeline actually looked for.
        field_paths=("symptome.symptom_locations",),
    )
    with pytest.raises(SchemaError, match="symptom_sites"):
        SchemaTopicPlanner(sti_schema_spec(), (broken,))


def test_default_instance_activates_only_the_unconditional_topic() -> None:
    plan = sti_topic_planner().plan(StiAnamnesis(), _variation(0), random.Random(0))
    assert "greeting" in plan.names
    assert "symptoms" not in plan.names


def test_symptoms_activate_when_present() -> None:
    ground_truth = StiAnamnesis.model_validate(
        {"symptome": {"symptom_sites": ["penis_urethra"], "symptom_types": ["discharge"]}}
    )
    plan = sti_topic_planner().plan(ground_truth, _variation(0), random.Random(0))
    assert "symptoms" in plan.names


def test_prevention_activates_on_a_vaccination_status() -> None:
    ground_truth = StiAnamnesis.model_validate({"praevention": {"vaccination_hpv": "complete"}})
    plan = sti_topic_planner().plan(ground_truth, _variation(0), random.Random(0))
    assert "prevention" in plan.names


def test_a_stated_no_activates_its_topic() -> None:
    """The reason TriState exists: 'asked, answer no' is content, 'never came up' is not."""
    planner = sti_topic_planner()
    asked = StiAnamnesis.model_validate({"praevention": {"prep_current_use": "no"}})
    never = StiAnamnesis.model_validate({"praevention": {"prep_current_use": "not_discussed"}})
    assert "prevention" in planner.plan(asked, _variation(0), random.Random(0)).names
    assert "prevention" not in planner.plan(never, _variation(0), random.Random(0)).names


def test_greeting_is_first_and_tests_are_last() -> None:
    ground_truth = StiAnamnesis.model_validate(
        {
            "symptome": {"symptom_types": ["itch"]},
            "diagnostik_und_kommunikation": {"tests_recommended": ["hiv"]},
        }
    )
    plan = sti_topic_planner().plan(ground_truth, _variation(0), random.Random(0))
    assert plan.names[0] == "greeting"
    assert plan.names[-1] == "planned_tests"


def test_scenario_reads_setting_and_style_metadata() -> None:
    space = sti_variation_space()
    builder = StiScenarioBuilder()
    for seed in range(20):
        variation = space.sample(random.Random(seed))
        scenario = builder.build(variation=variation, rng=random.Random(seed))
        assert scenario.language == "Deutsch"
        assert scenario.turns.minimum <= scenario.turns.maximum
        location = variation.metadata("setting", "location")
        if location:
            assert location in scenario.context


def test_concise_style_yields_a_shorter_target() -> None:
    space = sti_variation_space()
    builder = StiScenarioBuilder()
    lengths = {}
    for seed in range(200):
        variation = space.sample(random.Random(seed))
        style = variation.value("style")
        if style in {"concise", "detailed"} and style not in lengths:
            lengths[style] = builder.build(
                variation=variation, rng=random.Random(seed)
            ).turns.maximum
        if len(lengths) == 2:
            break
    assert lengths["concise"] < lengths["detailed"]


def test_ground_truth_prompt_is_german_and_reproducible() -> None:
    schema = sti_schema_spec()
    variation = sti_variation_space().sample(random.Random(4))
    scenario = StiScenarioBuilder().build(variation=variation, rng=random.Random(4))
    builder = StiGroundTruthPrompt()

    def render(seed: int) -> str:
        return builder.build(
            GroundTruthPromptContext(
                schema=schema, scenario=scenario, variation=variation, rng=random.Random(seed)
            )
        )

    prompt = render(1)
    assert "not_discussed" in prompt
    assert "STI-/HIV-Anamnese" in prompt
    assert "Das Thema kam im Gespräch nicht zur Sprache." in prompt  # sentinel docs
    # Reproducible: this was not true of the pipeline this was ported from.
    assert prompt == render(1)
    assert prompt != render(2)


def test_ground_truth_prompt_can_restate_the_field_guide() -> None:
    schema = sti_schema_spec()
    variation = sti_variation_space().sample(random.Random(0))
    scenario = StiScenarioBuilder().build(variation=variation, rng=random.Random(0))
    context = GroundTruthPromptContext(
        schema=schema, scenario=scenario, variation=variation, rng=random.Random(0)
    )
    assert "FELDREFERENZ" not in StiGroundTruthPrompt().build(context)
    assert "FELDREFERENZ" in StiGroundTruthPrompt(include_field_guide=True).build(context)


def test_dialogue_prompt_contains_everything_the_model_needs() -> None:
    schema = sti_schema_spec()
    variation = sti_variation_space().sample(random.Random(4))
    scenario = StiScenarioBuilder().build(variation=variation, rng=random.Random(4))
    cast = Cast(personas=(CLINICIAN_PERSONAS[0], PATIENT_PERSONAS[0]))
    ground_truth = StiAnamnesis.model_validate(
        {"symptome": {"symptom_types": ["discharge"]}, "patient": {"gender_identity": "Mann"}}
    )
    plan = sti_topic_planner().plan(ground_truth, variation, random.Random(4))

    context = DialoguePromptContext(
        schema=schema,
        scenario=scenario,
        roles=STI_ROLES,
        cast=cast,
        ground_truth=ground_truth,
        variation=variation,
        topic_plan=plan,
        include_cues=False,
        rng=random.Random(4),
    )
    prompt = StiDialoguePrompt().build(context)

    assert "Lehmann" in prompt and "Robin" in prompt
    assert "```json" in prompt
    assert "discharge" in prompt  # ground truth is embedded
    assert "THEMATISCHE REIHENFOLGE" in prompt  # topic plan is rendered
    assert "dürfen NICHT erwähnt werden" in prompt  # the instruction that keeps it short
    assert "(ruhig)" not in prompt  # cues were not requested


def test_dialogue_prompt_offers_cues_when_asked() -> None:
    context = _dialogue_context(include_cues=True)
    assert "(ruhig)" in StiDialoguePrompt().build(context)


def test_dialogue_prompt_is_reproducible() -> None:
    builder = StiDialoguePrompt()
    first = builder.build(_dialogue_context(rng_seed=3))
    second = builder.build(_dialogue_context(rng_seed=3))
    third = builder.build(_dialogue_context(rng_seed=4))
    assert first == second
    assert first != third


def test_schema_defaults_are_all_sentinels() -> None:
    """A fresh instance must read as 'nothing discussed', or topic planning inverts."""
    schema = sti_schema_spec()
    assert not schema.is_substantive(StiAnamnesis())


def test_exhaustive_casting_covers_all_pairings() -> None:
    spec = sti_dataset_spec(exhaustive_casting=True)
    assert spec.casting.combination_count == 40  # type: ignore[attr-defined]


# -- helpers -----------------------------------------------------------------


def _variation(seed: int):
    return sti_variation_space().sample(random.Random(seed))


def _dialogue_context(*, include_cues: bool = False, rng_seed: int = 4) -> DialoguePromptContext:
    variation = _variation(4)
    return DialoguePromptContext(
        schema=sti_schema_spec(),
        scenario=StiScenarioBuilder().build(variation=variation, rng=random.Random(4)),
        roles=STI_ROLES,
        cast=Cast(personas=(CLINICIAN_PERSONAS[0], PATIENT_PERSONAS[0])),
        ground_truth=StiAnamnesis(),
        variation=variation,
        topic_plan=sti_topic_planner().plan(StiAnamnesis(), variation, random.Random(4)),
        include_cues=include_cues,
        rng=random.Random(rng_seed),
    )


async def test_sti_preset_runs_end_to_end_offline(tmp_path) -> None:
    """The real preset through the real pipeline, with canned providers."""
    from conversim import GenerationConfig, load_dataset
    from conversim.pipeline import DatasetGenerator
    from conversim.presets.sti import sti_prompt_builders

    from .conftest import StubStructuredProvider, StubTextProvider

    class StiStructured(StubStructuredProvider):
        async def generate_structured(self, *, prompt: str, schema):
            self.calls += 1
            value = schema.model_validate(
                {
                    "patient": {"gender_identity": "Mann", "anatomy_present": ["penis"]},
                    "symptome": {
                        "symptom_types": ["discharge"],
                        "symptom_sites": ["penis_urethra"],
                    },
                    "praevention": {"vaccination_hpv": "complete"},
                    "diagnostik_und_kommunikation": {"tests_recommended": ["hiv", "chlamydia"]},
                }
            )
            from conversim.models import TokenUsage
            from conversim.providers.base import StructuredResult

            return StructuredResult(
                value=value, usage=TokenUsage(output_tokens=32), model=self.model
            )

    spec = sti_dataset_spec()
    ground_truth_prompt, dialogue_prompt = sti_prompt_builders()
    text = StubTextProvider(turns=22, personas=CLINICIAN_PERSONAS + PATIENT_PERSONAS)

    generator = DatasetGenerator.from_providers(
        spec=spec,
        structured_provider=StiStructured(),
        text_provider=text,
        ground_truth_prompt=ground_truth_prompt,
        dialogue_prompt=dialogue_prompt,
    )
    result = await generator.generate(tmp_path / "sti", GenerationConfig(count=3, seed=5))

    assert len(result.generated) == 3
    assert not result.failed

    dataset = load_dataset(tmp_path / "sti")
    for sample in dataset:
        assert sample.record.schema_name == "StiAnamnesis"
        assert sample.record.is_clean
        # Topic planning reacted to the ground truth that was actually present.
        assert "symptoms" in sample.record.topic_plan.names
        assert "prevention" in sample.record.topic_plan.names
        assert "substance_use" not in sample.record.topic_plan.names
        assert sample.record.scenario.language == "Deutsch"
        StiAnamnesis.model_validate(sample.ground_truth)


async def test_sti_dataset_is_reproducible_offline(tmp_path) -> None:
    from conversim import GenerationConfig, load_dataset
    from conversim.pipeline import DatasetGenerator
    from conversim.presets.sti import sti_prompt_builders

    from .conftest import StubStructuredProvider, StubTextProvider

    async def run(name: str, concurrency: int) -> list[tuple[str, tuple[str, ...], dict]]:
        ground_truth_prompt, dialogue_prompt = sti_prompt_builders()
        generator = DatasetGenerator.from_providers(
            spec=sti_dataset_spec(),
            structured_provider=StubStructuredProvider(),
            text_provider=StubTextProvider(
                turns=22, personas=CLINICIAN_PERSONAS + PATIENT_PERSONAS
            ),
            ground_truth_prompt=ground_truth_prompt,
            dialogue_prompt=dialogue_prompt,
        )
        await generator.generate(
            tmp_path / name, GenerationConfig(count=5, seed=99, concurrency=concurrency)
        )
        return [
            (s.id, s.record.cast.names, s.record.variation.summary())
            for s in load_dataset(tmp_path / name)
        ]

    assert await run("serial", 1) == await run("parallel", 5)
