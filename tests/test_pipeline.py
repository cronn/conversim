"""End-to-end generation: reproducibility, resume, failure handling, audio."""

from __future__ import annotations

import json

import pytest

from conversim import (
    AudioConfig,
    DatasetGenerator,
    DatasetSpec,
    DialogueValidator,
    GenerationConfig,
    RandomCasting,
    load_dataset,
)
from conversim.speech.mixing import MixConfig
from conversim.speech.voices import FixedAssignment, RandomAssignment, SwappedAssignment

from .conftest import (
    StubStructuredProvider,
    StubSynthesizer,
    StubTextProvider,
)


def generator(spec: DatasetSpec, **kwargs: object) -> DatasetGenerator:
    structured = kwargs.pop("structured", None) or StubStructuredProvider()
    text = kwargs.pop("text", None) or StubTextProvider()
    return DatasetGenerator.from_providers(
        spec=spec,
        structured_provider=structured,
        text_provider=text,
        **kwargs,  # type: ignore[arg-type]
    )


async def test_generates_all_expected_artefacts(tmp_path, spec: DatasetSpec) -> None:
    result = await generator(spec).generate(tmp_path / "ds", GenerationConfig(count=3, seed=1))

    assert len(result.generated) == 3
    assert not result.failed

    layout = load_dataset(tmp_path / "ds").layout
    for outcome in result.generated:
        sample_id = outcome.id
        for path in (
            layout.sample_record(sample_id),
            layout.ground_truth(sample_id),
            layout.dialogue_jsonl(sample_id),
            layout.dialogue_txt(sample_id),
            layout.transcript(sample_id),
            layout.prompt(sample_id, "ground_truth"),
            layout.prompt(sample_id, "dialogue"),
        ):
            assert path.exists(), path
    assert layout.manifest_path.exists()
    assert layout.index_path.exists()


async def test_manifest_records_the_run(tmp_path, spec: DatasetSpec) -> None:
    result = await generator(spec).generate(tmp_path / "ds", GenerationConfig(count=2, seed=42))

    manifest = result.manifest
    assert manifest.seed == 42
    assert manifest.sample_count == 2
    assert manifest.schema_name == "Ticket"
    assert manifest.schema_definition["title"] == "Ticket"
    assert manifest.stage_models == {"ground_truth": "stub-structured", "dialogue": "stub-text"}
    assert manifest.name == "test-tickets"


async def test_an_unset_seed_is_still_recorded(tmp_path, spec: DatasetSpec) -> None:
    result = await generator(spec).generate(tmp_path / "ds", GenerationConfig(count=1))
    assert result.manifest.seed is not None


async def test_same_seed_reproduces_the_dataset(tmp_path, spec: DatasetSpec) -> None:
    async def run(name: str, concurrency: int) -> list[tuple[str, str, int]]:
        await generator(spec).generate(
            tmp_path / name, GenerationConfig(count=6, seed=7, concurrency=concurrency)
        )
        dataset = load_dataset(tmp_path / name)
        return [
            (sample.id, json.dumps(sample.ground_truth, sort_keys=True), sample.record.seed)
            for sample in dataset
        ]

    # Different concurrency, same result: per-sample seeds do not depend on scheduling.
    assert await run("a", 1) == await run("b", 4)


async def test_different_seeds_give_different_sample_ids(tmp_path, spec: DatasetSpec) -> None:
    first = await generator(spec).generate(tmp_path / "a", GenerationConfig(count=2, seed=1))
    second = await generator(spec).generate(tmp_path / "b", GenerationConfig(count=2, seed=2))
    assert {o.id for o in first.generated}.isdisjoint({o.id for o in second.generated})


async def test_resume_skips_existing_samples(tmp_path, spec: DatasetSpec) -> None:
    root = tmp_path / "ds"
    await generator(spec).generate(root, GenerationConfig(count=2, seed=1))

    structured = StubStructuredProvider()
    second = await generator(spec, structured=structured).generate(
        root, GenerationConfig(count=4, seed=1)
    )

    assert len(second.skipped) == 2
    assert len(second.generated) == 2
    assert structured.calls == 2  # the existing two cost nothing
    assert len(load_dataset(root)) == 4


async def test_resume_can_be_disabled(tmp_path, spec: DatasetSpec) -> None:
    root = tmp_path / "ds"
    await generator(spec).generate(root, GenerationConfig(count=2, seed=1))

    structured = StubStructuredProvider()
    result = await generator(spec, structured=structured).generate(
        root, GenerationConfig(count=2, seed=1, resume=False)
    )
    assert len(result.generated) == 2
    assert structured.calls == 2


async def test_failures_are_isolated(tmp_path, spec: DatasetSpec) -> None:
    structured = StubStructuredProvider(fail_times=1)
    result = await generator(spec, structured=structured).generate(
        tmp_path / "ds", GenerationConfig(count=3, seed=1, concurrency=1)
    )

    assert len(result.failed) == 1
    assert len(result.generated) == 2
    assert "stub failure" in (result.failed[0].error or "")
    # The failed sample left nothing behind, so a resume will retry it.
    assert len(load_dataset(tmp_path / "ds")) == 2


async def test_failures_can_abort_the_run(tmp_path, spec: DatasetSpec) -> None:
    structured = StubStructuredProvider(fail_times=1)
    with pytest.raises(RuntimeError, match="stub failure"):
        await generator(spec, structured=structured).generate(
            tmp_path / "ds",
            GenerationConfig(count=3, seed=1, concurrency=1, continue_on_error=False),
        )


async def test_a_one_sided_dialogue_is_rejected(tmp_path, spec: DatasetSpec) -> None:
    class OneSided(StubTextProvider):
        async def generate_text(self, *, prompt: str, max_output_tokens: int | None = None):
            result = await super().generate_text(prompt=prompt, max_output_tokens=max_output_tokens)
            lines = [line for line in result.text.splitlines() if line.startswith("Sam:")]
            return type(result)(text="\n".join(lines), usage=result.usage, model=result.model)

    result = await generator(spec, text=OneSided()).generate(
        tmp_path / "ds", GenerationConfig(count=1, seed=1)
    )
    assert len(result.failed) == 1
    assert "never speak" in (result.failed[0].error or "")


async def test_lenient_validation_records_instead_of_failing(tmp_path, spec: DatasetSpec) -> None:
    text = StubTextProvider(turns=40)  # far outside the requested 6-12
    result = await generator(spec, text=text, validator=DialogueValidator(strict=False)).generate(
        tmp_path / "ds", GenerationConfig(count=1, seed=1)
    )
    assert len(result.generated) == 1
    record = result.generated[0].record
    assert record is not None
    turn_issues = [issue for issue in record.issues if issue.kind == "turn_count"]
    assert turn_issues
    # Overrunning the requested turn range is a missed instruction, not damage,
    # so it is recorded as minor and leaves the sample usable.
    assert turn_issues[0].severity == "minor"
    assert record.is_clean
    assert not record.major_issues


async def test_major_issue_makes_sample_unclean(tmp_path, spec: DatasetSpec) -> None:
    """The counterpart to the test above: a parse problem does disqualify."""
    text = StubTextProvider(preamble="Sure, here is the dialogue:")
    result = await generator(spec, text=text).generate(
        tmp_path / "ds", GenerationConfig(count=1, seed=1)
    )
    record = result.generated[0].record
    assert record is not None
    assert {issue.kind for issue in record.major_issues} == {"preamble"}
    assert not record.is_clean


async def test_truncation_is_recorded(tmp_path, spec: DatasetSpec) -> None:
    result = await generator(spec, text=StubTextProvider(truncated=True)).generate(
        tmp_path / "ds", GenerationConfig(count=1, seed=1)
    )
    record = result.generated[0].record
    assert record is not None
    assert "truncated" in {issue.kind for issue in record.issues}


async def test_preamble_is_stripped_and_raw_output_kept(tmp_path, spec: DatasetSpec) -> None:
    text = StubTextProvider(preamble="Sure! Here is the conversation:")
    result = await generator(spec, text=text).generate(
        tmp_path / "ds", GenerationConfig(count=1, seed=1)
    )
    sample_id = result.generated[0].id
    dataset = load_dataset(tmp_path / "ds")
    sample = dataset.load(sample_id)

    assert "Here is the conversation" not in sample.dialogue.transcript()
    # Parsing changed something, so the raw output is preserved for inspection.
    assert dataset.layout.dialogue_raw(sample_id).exists()


async def test_clean_output_does_not_duplicate_the_raw_file(tmp_path, spec: DatasetSpec) -> None:
    result = await generator(spec).generate(tmp_path / "ds", GenerationConfig(count=1, seed=1))
    dataset = load_dataset(tmp_path / "ds")
    assert not dataset.layout.dialogue_raw(result.generated[0].id).exists()


async def test_usage_and_progress_are_reported(tmp_path, spec: DatasetSpec) -> None:
    events: list[tuple[str, str | None]] = []
    result = await generator(spec).generate(
        tmp_path / "ds",
        GenerationConfig(count=2, seed=1, on_event=lambda e: events.append((e.kind, e.status))),
    )

    assert result.total_usage().output_tokens > 0
    assert ("sample_started", None) in events
    assert ("sample_finished", "generated") in events
    assert "generated" in result.summary()


async def test_a_broken_progress_callback_does_not_break_the_run(
    tmp_path, spec: DatasetSpec
) -> None:
    def explode(event: object) -> None:
        raise ValueError("boom")

    result = await generator(spec).generate(
        tmp_path / "ds", GenerationConfig(count=1, seed=1, on_event=explode)
    )
    assert len(result.generated) == 1


async def test_custom_sample_ids(tmp_path, spec: DatasetSpec) -> None:
    result = await generator(spec).generate(
        tmp_path / "ds",
        GenerationConfig(count=2, seed=1, sample_id_factory=lambda i, s: f"call-{i:03d}"),
    )
    assert sorted(o.id for o in result.generated) == ["call-000", "call-001"]


async def test_random_casting_covers_the_pools(tmp_path, ticket_schema) -> None:
    from conversim import Persona, StaticScenarioBuilder, TurnTarget
    from conversim.models import Role

    agents = tuple(Persona(name=f"Agent{i}", role="agent") for i in range(3))
    customers = tuple(Persona(name=f"Cust{i}", role="customer") for i in range(3))
    spec = DatasetSpec(
        schema=ticket_schema,
        roles=(Role(id="agent", label="Agent"), Role(id="customer", label="Customer")),
        casting=RandomCasting(pools={"agent": agents, "customer": customers}),
        scenario=StaticScenarioBuilder(
            topic="t", context="c", turns=TurnTarget(minimum=6, maximum=12)
        ),
    )
    text = StubTextProvider(personas=agents + customers)
    await generator(spec, text=text).generate(tmp_path / "ds", GenerationConfig(count=8, seed=3))

    speakers = {
        speaker for sample in load_dataset(tmp_path / "ds") for speaker in sample.record.cast.names
    }
    assert len(speakers) > 2


# -- audio -------------------------------------------------------------------


async def test_audio_pass_produces_takes(tmp_path, spec: DatasetSpec, voice_library) -> None:
    root = tmp_path / "ds"
    await generator(spec).generate(root, GenerationConfig(count=2, seed=1))

    synthesizer = StubSynthesizer()
    result = await DatasetGenerator.synthesize(
        root,
        AudioConfig(
            synthesizer=synthesizer,
            library=voice_library,
            assignment=FixedAssignment(voices={"agent": "anna", "customer": "ben"}, take_id="main"),
            mix=MixConfig(pause_sd_s=0.0),
        ),
    )

    assert len(result.rendered) == 2
    assert not result.failed
    assert result.total_duration_s() > 0

    dataset = load_dataset(root)
    for sample in dataset:
        assert [take.id for take in sample.takes] == ["main"]
        take = sample.takes[0]
        assert sample.audio_path("main").exists()
        assert len(take.timings) == sample.record.turn_count
        assert take.backend == "stub-tts"
        assert take.assignment.voice_for("agent") == "anna"
        assert sample.record.take_ids == ("main",)


async def test_sparse_default_renders_only_the_voices_actually_cast(
    tmp_path, spec: DatasetSpec, voice_library
) -> None:
    """One take costs one utterance per turn, not one per turn per library voice."""
    root = tmp_path / "ds"
    await generator(spec).generate(root, GenerationConfig(count=1, seed=1))
    dataset = load_dataset(root)
    sample = dataset.load(dataset.sample_ids()[0])
    turns = sample.record.turn_count

    synthesizer = StubSynthesizer()
    await DatasetGenerator.synthesize(
        root,
        AudioConfig(
            synthesizer=synthesizer,
            library=voice_library,
            assignment=FixedAssignment(voices={"agent": "anna", "customer": "ben"}),
        ),
    )
    assert len(synthesizer.calls) == turns
    assert len(voice_library) == 2  # rendering every voice would have cost 2 * turns


async def test_swapped_takes_render_each_voice_turn_pair_once(
    tmp_path, spec: DatasetSpec, voice_library
) -> None:
    """Exchanging two voices genuinely needs both voices on every turn.

    The cache does not reduce that -- every ``(voice, turn)`` pair is required by one
    permutation or the other. What it guarantees is that none is rendered twice.
    """
    root = tmp_path / "ds"
    await generator(spec).generate(root, GenerationConfig(count=1, seed=1))
    dataset = load_dataset(root)
    turns = dataset.load(dataset.sample_ids()[0]).record.turn_count

    synthesizer = StubSynthesizer()
    result = await DatasetGenerator.synthesize(
        root,
        AudioConfig(
            synthesizer=synthesizer, library=voice_library, assignment=SwappedAssignment(limit=2)
        ),
    )

    assert len(result.rendered) == 2
    assert len(synthesizer.calls) == 2 * turns
    assert len(set(synthesizer.calls)) == len(synthesizer.calls)  # nothing rendered twice
    assert result.segments_rendered == 2 * turns


async def test_a_third_take_reuses_already_rendered_voices(tmp_path, spec: DatasetSpec) -> None:
    """Where the cache does pay off: a new take sharing voices with an existing one."""
    from conversim.speech.base import Voice
    from conversim.speech.voices import VoiceLibrary

    root = tmp_path / "ds"
    await generator(spec).generate(root, GenerationConfig(count=1, seed=1))
    dataset = load_dataset(root)
    turns = dataset.load(dataset.sample_ids()[0]).record.turn_count

    library = VoiceLibrary(
        voices=(
            Voice(id="anna", preset="alloy"),
            Voice(id="ben", preset="echo"),
            Voice(id="cleo", preset="sage"),
        )
    )
    synthesizer = StubSynthesizer()

    # First take: anna and ben on every turn -> 2 * turns segments.
    await DatasetGenerator.synthesize(
        root,
        AudioConfig(
            synthesizer=synthesizer,
            library=library,
            assignment=SwappedAssignment(restrict_to_tags=(), limit=2),
        ),
    )
    after_first = len(synthesizer.calls)

    # A take pairing an already-rendered voice with a new one only pays for the new one.
    await DatasetGenerator.synthesize(
        root,
        AudioConfig(
            synthesizer=synthesizer,
            library=library,
            assignment=FixedAssignment(
                voices={"agent": "cleo", "customer": _already_rendered(synthesizer)},
                take_id="third",
            ),
        ),
    )
    added = len(synthesizer.calls) - after_first
    assert added < turns  # only cleo's half of the turns needed rendering


def _already_rendered(synthesizer: StubSynthesizer) -> str:
    return synthesizer.calls[0][0]


async def test_segments_are_reused_on_a_second_pass(
    tmp_path, spec: DatasetSpec, voice_library
) -> None:
    root = tmp_path / "ds"
    await generator(spec).generate(root, GenerationConfig(count=1, seed=1))
    assignment = FixedAssignment(voices={"agent": "anna", "customer": "ben"})

    first = StubSynthesizer()
    await DatasetGenerator.synthesize(
        root, AudioConfig(synthesizer=first, library=voice_library, assignment=assignment)
    )

    second = StubSynthesizer()
    result = await DatasetGenerator.synthesize(
        root,
        AudioConfig(
            synthesizer=second, library=voice_library, assignment=assignment, overwrite=True
        ),
    )
    assert second.calls == []  # every utterance came from cache
    assert len(result.rendered) == 1  # but the mix was redone


async def test_existing_takes_are_skipped(tmp_path, spec: DatasetSpec, voice_library) -> None:
    root = tmp_path / "ds"
    await generator(spec).generate(root, GenerationConfig(count=1, seed=1))
    assignment = FixedAssignment(voices={"agent": "anna", "customer": "ben"})
    config = AudioConfig(
        synthesizer=StubSynthesizer(), library=voice_library, assignment=assignment
    )

    await DatasetGenerator.synthesize(root, config)
    again = await DatasetGenerator.synthesize(root, config)
    assert len(again.skipped) == 1
    assert not again.rendered


async def test_audio_limit_and_selection(tmp_path, spec: DatasetSpec, voice_library) -> None:
    root = tmp_path / "ds"
    generated = await generator(spec).generate(root, GenerationConfig(count=3, seed=1))
    chosen = generated.generated[0].id

    result = await DatasetGenerator.synthesize(
        root,
        AudioConfig(
            synthesizer=StubSynthesizer(),
            library=voice_library,
            assignment=RandomAssignment(),
            sample_ids=[chosen],
        ),
    )
    assert {outcome.sample_id for outcome in result.rendered} == {chosen}


async def test_audio_totals_land_in_the_manifest(
    tmp_path, spec: DatasetSpec, voice_library
) -> None:
    root = tmp_path / "ds"
    await generator(spec).generate(root, GenerationConfig(count=2, seed=1))
    await DatasetGenerator.synthesize(
        root,
        AudioConfig(
            synthesizer=StubSynthesizer(), library=voice_library, assignment=RandomAssignment()
        ),
    )
    manifest = load_dataset(root).manifest()
    assert manifest is not None
    assert manifest.take_count == 2
    assert manifest.audio_duration_s > 0


async def test_index_reflects_audio(tmp_path, spec: DatasetSpec, voice_library) -> None:
    root = tmp_path / "ds"
    await generator(spec).generate(root, GenerationConfig(count=1, seed=1))
    await DatasetGenerator.synthesize(
        root,
        AudioConfig(
            synthesizer=StubSynthesizer(), library=voice_library, assignment=RandomAssignment()
        ),
    )
    entry = load_dataset(root).index()[0]
    assert entry.has_audio
    assert entry.audio_duration_s is not None
