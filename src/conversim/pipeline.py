"""The orchestrator: many samples, one dataset.

Two entry points, deliberately separate:

* :func:`generate_dataset` runs the text stages. They are network-bound, so they run
  concurrently with a configurable limit.
* :func:`synthesize_dataset` adds audio to an existing dataset. It reads what the
  first pass wrote, which means the expensive GPU phase can happen later, elsewhere,
  or repeatedly with different voices -- and an interrupted run picks up where it left
  off at the granularity of a single utterance.

Both are thin synchronous wrappers around :class:`DatasetGenerator`, so notebooks need
no ``await`` and asynchronous callers keep full control.

Reproducibility is built on per-sample seeds derived from ``(root_seed, index)`` by
hash rather than by drawing from a shared generator. A shared generator would make
every sample depend on the order in which earlier ones were scheduled, which under
concurrency is not stable. With derived seeds, sample 37 is the same sample whether it
ran first, last, or alone.
"""

from __future__ import annotations

import asyncio
import hashlib
import logging
import random
from collections.abc import Callable, Sequence
from dataclasses import dataclass, field
from pathlib import Path
from typing import Literal

from .casting import CastingStrategy, ScenarioBuilder
from .dataset.records import DatasetManifest, SampleRecord, TakeRecord
from .dataset.writer import DatasetWriter
from .errors import ConfigurationError
from .models import Cast, Dialogue, Role, Scenario, TokenUsage
from .prompts import DialoguePromptBuilder, GroundTruthPromptBuilder
from .providers.base import StructuredGenerator, TextGenerator
from .schema import SchemaSpec
from .speech.base import SpeechSynthesizer
from .speech.mixing import ConversationMixer, MixConfig, MixTurn
from .speech.voices import AssignmentStrategy, VoiceAssignment, VoiceLibrary
from .stages.dialogue import DialogueGenerator, DialogueValidator
from .stages.ground_truth import GroundTruthGenerator
from .topics import SchemaTopicPlanner, TopicPlan, TopicPlanner
from .variation import VariationConfig, VariationSpace

__all__ = [
    "AudioConfig",
    "AudioResult",
    "DatasetGenerator",
    "DatasetResult",
    "DatasetSpec",
    "GenerationConfig",
    "GenerationEvent",
    "SampleOutcome",
    "TakeOutcome",
    "generate_dataset",
    "synthesize_dataset",
]

logger = logging.getLogger(__name__)

Status = Literal["generated", "skipped", "failed"]


@dataclass(frozen=True)
class DatasetSpec:
    """What the dataset is made of.

    Args:
        schema: Ground-truth schema and its sentinel conventions.
        roles: Structural roles, in speaking order.
        casting: How personas are chosen per conversation.
        scenario: How the situational frame is derived.
        variation: The diversity axes sampled per conversation.
        topic_planner: How topic order is derived. Defaults to a
            :class:`~conversim.topics.SchemaTopicPlanner` with no categories, which
            yields an empty plan and simply omits topic guidance from the prompt.
        name: Optional dataset name, recorded in the manifest.
        description: Optional description, recorded in the manifest.

    Raises:
        ConfigurationError: if roles are empty or duplicated.
    """

    schema: SchemaSpec
    roles: tuple[Role, ...]
    casting: CastingStrategy
    scenario: ScenarioBuilder
    variation: VariationSpace = field(default_factory=VariationSpace)
    topic_planner: TopicPlanner | None = None
    name: str | None = None
    description: str | None = None

    def __post_init__(self) -> None:
        if not self.roles:
            raise ConfigurationError("A dataset spec needs at least one role")
        ids = [role.id for role in self.roles]
        duplicates = {value for value in ids if ids.count(value) > 1}
        if duplicates:
            raise ConfigurationError(f"Duplicate role ids: {sorted(duplicates)}")

    def planner(self) -> TopicPlanner:
        """The configured planner, or a no-op one."""
        if self.topic_planner is not None:
            return self.topic_planner
        return SchemaTopicPlanner(self.schema, categories=())


class GenerationEvent:
    """A progress notification.

    Passed to the ``on_event`` callback so callers can drive a progress bar, a log
    line, or nothing at all. The library itself only logs; it never writes to stdout.

    Attributes:
        kind: ``"sample_started"``, ``"sample_finished"``, ``"take_finished"`` or
            ``"segment_finished"``.
        index: Zero-based position in the run.
        total: Total planned items.
        sample_id: Sample the event concerns.
        status: Outcome, for ``*_finished`` events.
        detail: Free-text detail, e.g. an error message.
    """

    __slots__ = ("detail", "index", "kind", "sample_id", "status", "total")

    def __init__(
        self,
        kind: str,
        *,
        index: int,
        total: int,
        sample_id: str,
        status: Status | None = None,
        detail: str | None = None,
    ) -> None:
        self.kind = kind
        self.index = index
        self.total = total
        self.sample_id = sample_id
        self.status = status
        self.detail = detail

    def __repr__(self) -> str:
        parts = [f"{self.kind} {self.index + 1}/{self.total} {self.sample_id}"]
        if self.status:
            parts.append(self.status)
        if self.detail:
            parts.append(self.detail)
        return f"<GenerationEvent {' '.join(parts)}>"


ProgressCallback = Callable[[GenerationEvent], None]


@dataclass(frozen=True)
class GenerationConfig:
    """How to run the text stages.

    Args:
        count: Number of samples to produce.
        seed: Root seed. When omitted, one is drawn and recorded in the manifest, so a
            run is reproducible after the fact even if you did not plan to repeat it.
        concurrency: Samples in flight at once. Providers apply their own limits on top;
            this bounds how many conversations are partially complete.
        resume: Skip samples whose record already exists.
        continue_on_error: Record a failure and carry on instead of aborting the run.
            A handful of failures out of hundreds is normal and not worth losing the
            rest of the run over.
        sample_id_factory: Override sample naming. Receives the index and the effective
            root seed.
        on_event: Progress callback.
    """

    count: int
    seed: int | None = None
    concurrency: int = 4
    resume: bool = True
    continue_on_error: bool = True
    sample_id_factory: Callable[[int, int], str] | None = None
    on_event: ProgressCallback | None = None

    def __post_init__(self) -> None:
        if self.count < 1:
            raise ConfigurationError(f"count must be >= 1, got {self.count}")
        if self.concurrency < 1:
            raise ConfigurationError(f"concurrency must be >= 1, got {self.concurrency}")


@dataclass(frozen=True)
class AudioConfig:
    """How to render audio for an existing dataset.

    Args:
        synthesizer: Speech backend.
        library: Voices available for casting.
        assignment: How voices are cast onto roles, and how many takes to render.
        mix: Mixing parameters.
        overwrite: Re-render takes that already exist. Per-utterance segments are
            reused regardless unless this is set, which is what makes an interrupted
            run cheap to resume.
        concurrency: Samples in flight at once. Leave at ``1`` for a local GPU backend:
            it serialises generation internally anyway, and more in flight only means
            more partially finished conversations.
        limit: Stop after this many samples.
        sample_ids: Render only these samples, in this order.
        on_event: Progress callback.
    """

    synthesizer: SpeechSynthesizer
    library: VoiceLibrary
    assignment: AssignmentStrategy
    mix: MixConfig = field(default_factory=MixConfig)
    overwrite: bool = False
    concurrency: int = 1
    limit: int | None = None
    sample_ids: Sequence[str] | None = None
    on_event: ProgressCallback | None = None

    def __post_init__(self) -> None:
        if self.concurrency < 1:
            raise ConfigurationError(f"concurrency must be >= 1, got {self.concurrency}")


@dataclass(frozen=True)
class SampleOutcome:
    """What happened to one sample."""

    id: str
    status: Status
    record: SampleRecord | None = None
    error: str | None = None


@dataclass(frozen=True)
class DatasetResult:
    """Outcome of a generation run.

    Attributes:
        root: Dataset root.
        manifest: Manifest as written.
        outcomes: One entry per planned sample, in run order.
    """

    root: Path
    manifest: DatasetManifest
    outcomes: tuple[SampleOutcome, ...]

    def _with_status(self, status: Status) -> tuple[SampleOutcome, ...]:
        return tuple(outcome for outcome in self.outcomes if outcome.status == status)

    @property
    def generated(self) -> tuple[SampleOutcome, ...]:
        """Samples produced by this run."""
        return self._with_status("generated")

    @property
    def skipped(self) -> tuple[SampleOutcome, ...]:
        """Samples that already existed."""
        return self._with_status("skipped")

    @property
    def failed(self) -> tuple[SampleOutcome, ...]:
        """Samples that could not be produced."""
        return self._with_status("failed")

    def total_usage(self) -> TokenUsage:
        """Token usage across every sample produced by this run."""
        total = TokenUsage()
        for outcome in self.generated:
            if outcome.record is not None:
                total = total + outcome.record.total_usage()
        return total

    def summary(self) -> str:
        """A one-line summary suitable for logging."""
        usage = self.total_usage()
        return (
            f"{len(self.generated)} generated, {len(self.skipped)} skipped, "
            f"{len(self.failed)} failed; "
            f"{usage.input_tokens:,} in / {usage.output_tokens:,} out tokens"
        )


@dataclass(frozen=True)
class TakeOutcome:
    """What happened to one take."""

    sample_id: str
    take_id: str
    status: Status
    record: TakeRecord | None = None
    error: str | None = None


@dataclass(frozen=True)
class AudioResult:
    """Outcome of an audio run.

    Attributes:
        root: Dataset root.
        outcomes: One entry per attempted take.
        segments_rendered: Utterances synthesised in this run, excluding cache hits.
    """

    root: Path
    outcomes: tuple[TakeOutcome, ...]
    segments_rendered: int

    @property
    def rendered(self) -> tuple[TakeOutcome, ...]:
        """Takes mixed in this run."""
        return tuple(outcome for outcome in self.outcomes if outcome.status == "generated")

    @property
    def skipped(self) -> tuple[TakeOutcome, ...]:
        """Takes that already existed."""
        return tuple(outcome for outcome in self.outcomes if outcome.status == "skipped")

    @property
    def failed(self) -> tuple[TakeOutcome, ...]:
        """Takes that could not be produced."""
        return tuple(outcome for outcome in self.outcomes if outcome.status == "failed")

    def total_duration_s(self) -> float:
        """Audio produced in this run, in seconds."""
        return sum(
            outcome.record.duration_s for outcome in self.rendered if outcome.record is not None
        )

    def summary(self) -> str:
        """A one-line summary suitable for logging."""
        return (
            f"{len(self.rendered)} takes rendered ({self.total_duration_s() / 60:.1f} min), "
            f"{len(self.skipped)} skipped, {len(self.failed)} failed; "
            f"{self.segments_rendered} utterances synthesised"
        )


class DatasetGenerator:
    """Runs the generation stages across many samples.

    Args:
        spec: What the dataset is made of.
        ground_truth: Configured ground-truth stage.
        dialogue: Configured dialogue stage.

    Example:
        >>> generator = DatasetGenerator.from_providers(      # doctest: +SKIP
        ...     spec=spec,
        ...     structured_provider=GoogleProvider(model="gemini-2.5-flash"),
        ...     text_provider=OpenAIProvider(model="gpt-5-mini"),
        ... )
        >>> result = await generator.generate(                # doctest: +SKIP
        ...     "out/dataset", GenerationConfig(count=100, seed=7)
        ... )
    """

    def __init__(
        self,
        *,
        spec: DatasetSpec,
        ground_truth: GroundTruthGenerator,
        dialogue: DialogueGenerator,
    ) -> None:
        self.spec = spec
        self.ground_truth = ground_truth
        self.dialogue = dialogue

    @classmethod
    def from_providers(
        cls,
        *,
        spec: DatasetSpec,
        structured_provider: StructuredGenerator,
        text_provider: TextGenerator,
        ground_truth_prompt: GroundTruthPromptBuilder | None = None,
        dialogue_prompt: DialoguePromptBuilder | None = None,
        validator: DialogueValidator | None = None,
        max_output_tokens: int | None = 16_000,
        include_cues: bool = False,
    ) -> DatasetGenerator:
        """Build a generator from two providers, using the default stage wiring.

        Args:
            spec: What the dataset is made of.
            structured_provider: Provider for the ground-truth stage.
            text_provider: Provider for the dialogue stage.
            ground_truth_prompt: Override the ground-truth prompt builder.
            dialogue_prompt: Override the dialogue prompt builder.
            validator: Override dialogue validation thresholds.
            max_output_tokens: Output cap for the dialogue call.
            include_cues: Ask for ``(cue)`` delivery annotations. Only enable when the
                speech backend acts on them.
        """
        return cls(
            spec=spec,
            ground_truth=GroundTruthGenerator(
                schema=spec.schema,
                provider=structured_provider,
                prompt_builder=ground_truth_prompt,
            ),
            dialogue=DialogueGenerator(
                schema=spec.schema,
                provider=text_provider,
                roles=spec.roles,
                prompt_builder=dialogue_prompt,
                validator=validator,
                max_output_tokens=max_output_tokens,
                include_cues=include_cues,
            ),
        )

    # -- text stages --------------------------------------------------------

    async def generate(
        self,
        root: Path | str,
        config: GenerationConfig,
    ) -> DatasetResult:
        """Generate ground truth and dialogue for ``config.count`` samples."""
        writer = DatasetWriter(root)
        seed = config.seed if config.seed is not None else random.SystemRandom().getrandbits(32)
        if config.seed is None:
            logger.info("No seed given; using %d (recorded in the manifest)", seed)

        planner = self.spec.planner()
        semaphore = asyncio.Semaphore(config.concurrency)
        outcomes: list[SampleOutcome | None] = [None] * config.count

        async def run_one(index: int) -> None:
            sample_id = self._sample_id(index, seed, config)
            async with semaphore:
                if config.resume and writer.has_sample(sample_id):
                    logger.debug("Skipping existing sample %s", sample_id)
                    outcomes[index] = SampleOutcome(id=sample_id, status="skipped")
                    _emit(
                        config.on_event,
                        "sample_finished",
                        index,
                        config.count,
                        sample_id,
                        "skipped",
                    )
                    return

                _emit(config.on_event, "sample_started", index, config.count, sample_id)
                try:
                    record = await self._generate_sample(
                        writer=writer,
                        planner=planner,
                        sample_id=sample_id,
                        index=index,
                        seed=seed,
                    )
                except Exception as error:
                    if not config.continue_on_error:
                        raise
                    logger.error("Sample %s failed: %s", sample_id, error, exc_info=True)
                    outcomes[index] = SampleOutcome(
                        id=sample_id, status="failed", error=f"{type(error).__name__}: {error}"
                    )
                    _emit(
                        config.on_event,
                        "sample_finished",
                        index,
                        config.count,
                        sample_id,
                        "failed",
                        str(error),
                    )
                    return

                outcomes[index] = SampleOutcome(id=sample_id, status="generated", record=record)
                _emit(
                    config.on_event,
                    "sample_finished",
                    index,
                    config.count,
                    sample_id,
                    "generated",
                )

        await asyncio.gather(*(run_one(index) for index in range(config.count)))

        writer.rebuild_index()
        manifest = self._build_manifest(writer, seed=seed)
        writer.write_manifest(manifest)

        result = DatasetResult(
            root=writer.root,
            manifest=manifest,
            outcomes=tuple(outcome for outcome in outcomes if outcome is not None),
        )
        logger.info("Generation complete: %s", result.summary())
        return result

    async def _generate_sample(
        self,
        *,
        writer: DatasetWriter,
        planner: TopicPlanner,
        sample_id: str,
        index: int,
        seed: int,
    ) -> SampleRecord:
        sample_seed = _derive_seed(seed, index)
        rng = random.Random(sample_seed)

        variation = self.spec.variation.sample(rng)
        scenario = self.spec.scenario.build(variation=variation, rng=rng)
        cast = self.spec.casting.cast(index=index, rng=rng)
        self._check_cast_covers_roles(cast)

        ground_truth = await self.ground_truth.generate(
            scenario=scenario, variation=variation, rng=rng
        )
        topic_plan = planner.plan(ground_truth.value, variation, rng)
        dialogue_result = await self.dialogue.generate(
            scenario=scenario,
            cast=cast,
            ground_truth=ground_truth.value,
            variation=variation,
            topic_plan=topic_plan,
            rng=rng,
        )

        record = self._build_record(
            sample_id=sample_id,
            sample_seed=sample_seed,
            scenario=scenario,
            cast=cast,
            variation=variation,
            topic_plan=topic_plan,
            dialogue=dialogue_result.dialogue,
            models={
                "ground_truth": ground_truth.model,
                "dialogue": dialogue_result.model,
            },
            usage={
                "ground_truth": ground_truth.usage,
                "dialogue": dialogue_result.usage,
            },
        )

        return writer.write_sample(
            record,
            ground_truth=ground_truth.value,
            dialogue=dialogue_result.dialogue,
            prompts={
                "ground_truth": ground_truth.prompt,
                "dialogue": dialogue_result.prompt,
            },
        )

    def _build_record(
        self,
        *,
        sample_id: str,
        sample_seed: int,
        scenario: Scenario,
        cast: Cast,
        variation: VariationConfig,
        topic_plan: TopicPlan,
        dialogue: Dialogue,
        models: dict[str, str],
        usage: dict[str, TokenUsage],
    ) -> SampleRecord:
        transcript = dialogue.transcript()
        return SampleRecord(
            id=sample_id,
            schema_name=self.spec.schema.name,
            seed=sample_seed,
            scenario=scenario,
            roles=self.spec.roles,
            cast=cast,
            variation=variation,
            topic_plan=topic_plan,
            turn_count=len(dialogue),
            word_count=len(transcript.split()),
            models=models,
            usage=usage,
            issues=dialogue.issues,
        )

    def _check_cast_covers_roles(self, cast: Cast) -> None:
        missing = [role.id for role in self.spec.roles if role.id not in cast.role_ids]
        if missing:
            raise ConfigurationError(
                f"Cast does not cover roles {missing}; it covers {sorted(cast.role_ids)}"
            )

    def _sample_id(self, index: int, seed: int, config: GenerationConfig) -> str:
        if config.sample_id_factory is not None:
            return config.sample_id_factory(index, seed)
        # A short seed tag keeps ids from colliding when several runs, each with its
        # own seed, are written into one dataset root.
        tag = hashlib.blake2b(str(seed).encode(), digest_size=2).hexdigest()
        return f"{tag}-{index:04d}"

    def _build_manifest(self, writer: DatasetWriter, *, seed: int) -> DatasetManifest:
        take_count, audio_duration = writer.audio_totals()
        return DatasetManifest(
            name=self.spec.name,
            description=self.spec.description,
            schema_name=self.spec.schema.name,
            schema_definition=self.spec.schema.json_schema(),
            roles=self.spec.roles,
            seed=seed,
            variation_cardinality=self.spec.variation.cardinality(),
            sample_count=len(writer.sample_ids()),
            take_count=take_count,
            audio_duration_s=audio_duration,
            stage_models={
                "ground_truth": self.ground_truth.provider.model,
                "dialogue": self.dialogue.provider.model,
            },
        )

    # -- audio stage --------------------------------------------------------

    @staticmethod
    async def synthesize(
        root: Path | str,
        config: AudioConfig,
    ) -> AudioResult:
        """Render audio for samples that already exist on disk.

        Utterances are synthesised once per ``(voice, turn)`` pair and cached, so
        rendering the same dialogue with the voices swapped costs one pass, not two.
        """
        writer = DatasetWriter(root)
        sample_ids = (
            list(config.sample_ids)
            if config.sample_ids is not None
            else [
                directory.name
                for directory in sorted(writer.layout.samples_dir.iterdir())
                if directory.is_dir() and writer.has_sample(directory.name)
            ]
        )
        if config.limit is not None:
            sample_ids = sample_ids[: config.limit]

        if not sample_ids:
            logger.warning("No samples to synthesise in %s", writer.root)
            return AudioResult(root=writer.root, outcomes=(), segments_rendered=0)

        mixer = ConversationMixer(config.mix)
        semaphore = asyncio.Semaphore(config.concurrency)
        outcomes: list[TakeOutcome] = []
        rendered_segments = 0
        lock = asyncio.Lock()
        total = len(sample_ids)

        async def run_one(position: int, sample_id: str) -> None:
            nonlocal rendered_segments
            async with semaphore:
                _emit(config.on_event, "sample_started", position, total, sample_id)
                try:
                    sample_outcomes, segments = await _synthesize_sample(
                        writer=writer,
                        mixer=mixer,
                        config=config,
                        sample_id=sample_id,
                        position=position,
                        total=total,
                    )
                except Exception as error:
                    logger.error("Audio for sample %s failed: %s", sample_id, error, exc_info=True)
                    async with lock:
                        outcomes.append(
                            TakeOutcome(
                                sample_id=sample_id,
                                take_id="-",
                                status="failed",
                                error=f"{type(error).__name__}: {error}",
                            )
                        )
                    _emit(
                        config.on_event,
                        "sample_finished",
                        position,
                        total,
                        sample_id,
                        "failed",
                        str(error),
                    )
                    return

                async with lock:
                    outcomes.extend(sample_outcomes)
                    rendered_segments += segments
                _emit(config.on_event, "sample_finished", position, total, sample_id, "generated")

        await asyncio.gather(
            *(run_one(position, sample_id) for position, sample_id in enumerate(sample_ids))
        )

        writer.rebuild_index()
        manifest = writer.read_manifest()
        if manifest is not None:
            take_count, audio_duration = writer.audio_totals()
            writer.write_manifest(
                manifest.model_copy(
                    update={"take_count": take_count, "audio_duration_s": audio_duration}
                )
            )

        result = AudioResult(
            root=writer.root, outcomes=tuple(outcomes), segments_rendered=rendered_segments
        )
        logger.info("Synthesis complete: %s", result.summary())
        return result


async def _synthesize_sample(
    *,
    writer: DatasetWriter,
    mixer: ConversationMixer,
    config: AudioConfig,
    sample_id: str,
    position: int,
    total: int,
) -> tuple[list[TakeOutcome], int]:
    """Render every take for one sample, reusing cached utterances."""
    from .dataset.reader import DatasetReader

    reader = DatasetReader(writer.root)
    sample = reader.load(sample_id)
    dialogue = sample.dialogue
    if not dialogue.utterances:
        logger.warning("Sample %s has no utterances; nothing to synthesise", sample_id)
        return [], 0

    rng = random.Random(sample.record.seed)
    assignments = config.assignment.assign(cast=sample.record.cast, library=config.library, rng=rng)
    if not assignments:
        logger.warning("Assignment strategy produced no takes for sample %s", sample_id)
        return [], 0

    pending = [
        assignment
        for assignment in assignments
        if config.overwrite or not writer.layout.take_record(sample_id, assignment.id).exists()
    ]
    outcomes: list[TakeOutcome] = [
        TakeOutcome(sample_id=sample_id, take_id=assignment.id, status="skipped")
        for assignment in assignments
        if assignment not in pending
    ]
    if not pending:
        return outcomes, 0

    # Every (voice, turn) pair needed by any pending take, each rendered at most once.
    required: dict[tuple[str, int], str] = {}
    for assignment in pending:
        for utterance in dialogue.utterances:
            required[(assignment.voice_for(utterance.role), utterance.index)] = utterance.text

    rendered = 0
    for (voice_id, turn_index), text in sorted(required.items()):
        if writer.has_segment(sample_id, voice_id, turn_index):
            continue
        utterance = dialogue.utterances[turn_index]
        audio = await config.synthesizer.synthesize(
            text=text, voice=config.library.get(voice_id), cue=utterance.cue
        )
        writer.write_segment(sample_id, voice_id, turn_index, audio)
        rendered += 1
        _emit(
            config.on_event,
            "segment_finished",
            position,
            total,
            sample_id,
            detail=f"{voice_id}/{turn_index:04d}",
        )

    for assignment in pending:
        outcomes.append(
            await _mix_take(
                writer=writer,
                mixer=mixer,
                config=config,
                sample_id=sample_id,
                assignment=assignment,
                dialogue=dialogue,
                rng=rng,
                position=position,
                total=total,
            )
        )

    writer.set_take_ids(
        sample_id,
        tuple(take.id for take in writer.read_take_records(sample_id)),
    )
    return outcomes, rendered


async def _mix_take(
    *,
    writer: DatasetWriter,
    mixer: ConversationMixer,
    config: AudioConfig,
    sample_id: str,
    assignment: VoiceAssignment,
    dialogue: Dialogue,
    rng: random.Random,
    position: int,
    total: int,
) -> TakeOutcome:
    """Mix one take from cached utterance renders."""
    from .speech.base import load_audio

    turns: list[MixTurn] = []
    for utterance in dialogue.utterances:
        voice_id = assignment.voice_for(utterance.role)
        path = writer.segment_path(sample_id, voice_id, utterance.index)
        if not path.exists():
            return TakeOutcome(
                sample_id=sample_id,
                take_id=assignment.id,
                status="failed",
                error=f"missing segment for turn {utterance.index} in voice {voice_id!r}",
            )
        voice = config.library.get(voice_id)
        turns.append(
            MixTurn(
                index=utterance.index,
                role=utterance.role,
                speaker=utterance.speaker,
                voice_id=voice_id,
                audio=await asyncio.to_thread(load_audio, path),
                voice_tags=voice.tags,
            )
        )

    mixed = await asyncio.to_thread(mixer.mix, turns, rng=rng)
    record = TakeRecord(
        id=assignment.id,
        sample_id=sample_id,
        backend=config.synthesizer.name,
        assignment=assignment,
        audio_path=writer.layout.relative(writer.layout.take_audio(sample_id, assignment.id)),
        sample_rate=mixed.audio.sample_rate,
        channels=mixed.audio.channels,
        duration_s=mixed.audio.duration_s,
        timings=mixed.timings,
        pan_positions=mixed.pan_positions,
        backchannel_count=mixed.backchannel_count,
        mix_config=config.mix,
    )
    writer.write_take(record, mixed.audio)
    _emit(
        config.on_event,
        "take_finished",
        position,
        total,
        sample_id,
        "generated",
        f"{assignment.id} ({mixed.audio.duration_s:.1f}s)",
    )
    return TakeOutcome(
        sample_id=sample_id, take_id=assignment.id, status="generated", record=record
    )


# -- synchronous facade -----------------------------------------------------


def generate_dataset(
    root: Path | str,
    *,
    spec: DatasetSpec,
    structured_provider: StructuredGenerator,
    text_provider: TextGenerator,
    config: GenerationConfig,
    ground_truth_prompt: GroundTruthPromptBuilder | None = None,
    dialogue_prompt: DialoguePromptBuilder | None = None,
    validator: DialogueValidator | None = None,
    max_output_tokens: int | None = 16_000,
    include_cues: bool = False,
) -> DatasetResult:
    """Generate a text dataset, blocking until done.

    The synchronous entry point. Use :class:`DatasetGenerator` directly from an
    existing event loop.

    Args:
        root: Dataset root directory, created if absent.
        spec: What the dataset is made of.
        structured_provider: Provider for the ground-truth stage.
        text_provider: Provider for the dialogue stage.
        config: Run settings: count, seed, concurrency, resume.
        ground_truth_prompt: Override the ground-truth prompt builder.
        dialogue_prompt: Override the dialogue prompt builder.
        validator: Override dialogue validation thresholds.
        max_output_tokens: Output cap for the dialogue call.
        include_cues: Ask for ``(cue)`` delivery annotations.

    Returns:
        The run outcome, including per-sample status and token usage.

    Example:
        >>> result = generate_dataset(                        # doctest: +SKIP
        ...     "out/demo",
        ...     spec=spec,
        ...     structured_provider=GoogleProvider(model="gemini-2.5-flash"),
        ...     text_provider=OpenAIProvider(model="gpt-5-mini"),
        ...     config=GenerationConfig(count=10, seed=1),
        ... )
        >>> print(result.summary())                           # doctest: +SKIP
    """
    generator = DatasetGenerator.from_providers(
        spec=spec,
        structured_provider=structured_provider,
        text_provider=text_provider,
        ground_truth_prompt=ground_truth_prompt,
        dialogue_prompt=dialogue_prompt,
        validator=validator,
        max_output_tokens=max_output_tokens,
        include_cues=include_cues,
    )
    return asyncio.run(generator.generate(root, config))


def synthesize_dataset(root: Path | str, config: AudioConfig) -> AudioResult:
    """Render audio for an existing dataset, blocking until done.

    Args:
        root: Dataset root directory.
        config: Backend, voices, assignment strategy and mixing parameters.

    Returns:
        The run outcome, including per-take status and total duration.

    Example:
        >>> result = synthesize_dataset(                      # doctest: +SKIP
        ...     "out/demo",
        ...     AudioConfig(
        ...         synthesizer=QwenTTS(device="cuda:1"),
        ...         library=VoiceLibrary.from_directory("actors"),
        ...         assignment=SwappedAssignment(limit=2),
        ...     ),
        ... )
    """
    return asyncio.run(DatasetGenerator.synthesize(root, config))


# -- helpers ----------------------------------------------------------------


def _derive_seed(root_seed: int, index: int) -> int:
    """Derive a per-sample seed from the root seed and the sample index.

    Hashing rather than drawing from a shared generator makes each sample independent
    of scheduling order, which is what allows concurrency without giving up
    reproducibility.
    """
    digest = hashlib.blake2b(f"{root_seed}:{index}".encode(), digest_size=8).digest()
    return int.from_bytes(digest, "big")


def _emit(
    callback: ProgressCallback | None,
    kind: str,
    index: int,
    total: int,
    sample_id: str,
    status: Status | None = None,
    detail: str | None = None,
) -> None:
    """Invoke a progress callback, never letting it break the run."""
    if callback is None:
        return
    try:
        callback(
            GenerationEvent(
                kind, index=index, total=total, sample_id=sample_id, status=status, detail=detail
            )
        )
    except Exception:
        logger.warning("Progress callback raised; continuing", exc_info=True)
