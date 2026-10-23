"""Serialised records: what a generated dataset says about itself.

These models are the dataset's public contract. Anything a consumer might need to
know -- how a sample was produced, with which models and seed, under which sampled
variation, which voice spoke which role -- is written down rather than reconstructable,
because the code that produced a dataset drifts away from it almost immediately.
"""

from __future__ import annotations

from datetime import datetime, timezone
from importlib.metadata import PackageNotFoundError, version
from typing import Any

from pydantic import BaseModel, ConfigDict, Field

from ..models import Cast, DialogueIssue, Role, Scenario, TokenUsage
from ..speech.mixing import MixConfig, TurnTiming
from ..speech.voices import VoiceAssignment
from ..topics import TopicPlan
from ..variation import VariationConfig

__all__ = [
    "DatasetManifest",
    "SampleIndexEntry",
    "SampleRecord",
    "TakeRecord",
    "conversim_version",
    "utc_now",
]


def conversim_version() -> str:
    """The installed conversim version, or ``"unknown"`` when running from a checkout."""
    try:
        return version("conversim")
    except PackageNotFoundError:  # pragma: no cover - editable/source use
        return "unknown"


def utc_now() -> datetime:
    """Timezone-aware current time.

    Naive timestamps in a dataset are ambiguous the moment it leaves the machine that
    produced it, so everything recorded here carries an offset.
    """
    return datetime.now(timezone.utc)


class SampleRecord(BaseModel):
    """Provenance for one generated conversation.

    Attributes:
        id: Sample identifier, matching its directory name.
        created_at: When the sample was written.
        conversim_version: Version that produced it.
        schema_name: Name of the ground-truth schema class.
        seed: Seed of this sample's random stream. Together with the same library
            version and models, this is what makes a single sample reproducible.
        scenario: Topic, context, language and requested length.
        roles: Structural roles, in speaking order.
        cast: Personas filling those roles.
        variation: The sampled variation configuration.
        topic_plan: Topic sequence handed to the dialogue model.
        turn_count: Number of parsed utterances.
        word_count: Words in the transcript.
        models: Model identifier per stage.
        usage: Token usage per stage.
        issues: Problems found during parsing and validation, minor ones included.
            See :attr:`is_clean` for the distinction. An empty tuple means the
            sample passed every check.
        take_ids: Identifiers of rendered audio takes, empty when text-only.
    """

    model_config = ConfigDict(frozen=True)

    id: str
    created_at: datetime = Field(default_factory=utc_now)
    conversim_version: str = Field(default_factory=conversim_version)
    schema_name: str
    seed: int
    scenario: Scenario
    roles: tuple[Role, ...]
    cast: Cast
    variation: VariationConfig
    topic_plan: TopicPlan = Field(default_factory=TopicPlan)
    turn_count: int
    word_count: int
    models: dict[str, str] = Field(default_factory=dict)
    usage: dict[str, TokenUsage] = Field(default_factory=dict)
    issues: tuple[DialogueIssue, ...] = ()
    take_ids: tuple[str, ...] = ()

    @property
    def is_clean(self) -> bool:
        """Whether the sample is usable, i.e. carries no *major* issue.

        Minor issues are reported in :attr:`issues` but do not make a sample
        unclean. A dialogue that overran its requested turn range is the usual
        case: the instruction was missed, the sample is fine. Filter on
        :attr:`issues` directly if you want every deviation.
        """
        return not self.major_issues

    @property
    def major_issues(self) -> tuple[DialogueIssue, ...]:
        """The issues that make this sample unusable."""
        return tuple(issue for issue in self.issues if issue.severity == "major")

    def total_usage(self) -> TokenUsage:
        """Token usage summed across stages."""
        total = TokenUsage()
        for usage in self.usage.values():
            total = total + usage
        return total


class TakeRecord(BaseModel):
    """One rendered recording of a sample.

    A take is a casting of voices onto roles plus the mix that resulted. Several takes
    of one sample share the same words and differ only in who says them and how the
    room sounded.

    Attributes:
        id: Take identifier, matching its audio filename.
        sample_id: Sample this take belongs to.
        created_at: When the take was written.
        backend: Speech backend that rendered the segments.
        assignment: Which voice played which role.
        audio_path: Recording path, relative to the dataset root.
        sample_rate: Sample rate of the recording.
        channels: Channel count of the recording.
        duration_s: Length of the recording in seconds.
        timings: Start and end of every turn, in dialogue order. Directly usable as
            diarization reference and as segment boundaries for streaming evaluation.
        pan_positions: Stereo position assigned to each role.
        backchannel_count: Listener acknowledgements injected.
        mix_config: The mixing parameters used.
    """

    model_config = ConfigDict(frozen=True)

    id: str
    sample_id: str
    created_at: datetime = Field(default_factory=utc_now)
    backend: str
    assignment: VoiceAssignment
    audio_path: str
    sample_rate: int
    channels: int
    duration_s: float
    timings: tuple[TurnTiming, ...]
    pan_positions: dict[str, float] = Field(default_factory=dict)
    backchannel_count: int = 0
    mix_config: MixConfig


class SampleIndexEntry(BaseModel):
    """One line of ``samples.jsonl``: enough to filter a dataset without opening it.

    Derived from the per-sample records, and safe to regenerate at any time via
    :meth:`~conversim.dataset.writer.DatasetWriter.rebuild_index`.
    """

    model_config = ConfigDict(frozen=True)

    id: str
    path: str
    turn_count: int
    word_count: int
    speakers: tuple[str, ...]
    variation: dict[str, Any] = Field(default_factory=dict)
    issue_kinds: tuple[str, ...] = ()
    is_clean: bool = True
    take_ids: tuple[str, ...] = ()
    audio_duration_s: float | None = None

    @property
    def has_audio(self) -> bool:
        """Whether any take was rendered for this sample."""
        return bool(self.take_ids)


class DatasetManifest(BaseModel):
    """Run-level description of a dataset.

    Attributes:
        conversim_version: Version that produced the dataset.
        created_at: When the dataset root was first written.
        updated_at: When it was last touched.
        name: Optional human-readable dataset name.
        description: Optional free-text description.
        schema_name: Ground-truth schema class name.
        schema_definition: The JSON Schema itself, embedded so a published dataset stays
            interpretable without its generating code.
        roles: Structural roles used across the dataset.
        seed: Root seed the run was derived from, when one was set.
        variation_cardinality: Size of the designed variation space. Reported so a
            dataset can state how much diversity was built in, as distinct from how
            much the generator happened to realise.
        sample_count: Number of samples.
        take_count: Number of rendered takes.
        audio_duration_s: Total rendered audio in seconds.
        stage_models: Model identifiers per stage, as configured for the run.
        notes: Free-form extras.
    """

    model_config = ConfigDict(frozen=True)

    conversim_version: str = Field(default_factory=conversim_version)
    created_at: datetime = Field(default_factory=utc_now)
    updated_at: datetime = Field(default_factory=utc_now)
    name: str | None = None
    description: str | None = None
    schema_name: str
    schema_definition: dict[str, Any] = Field(default_factory=dict)
    roles: tuple[Role, ...] = ()
    seed: int | None = None
    variation_cardinality: int | None = None
    sample_count: int = 0
    take_count: int = 0
    audio_duration_s: float = 0.0
    stage_models: dict[str, str] = Field(default_factory=dict)
    notes: dict[str, Any] = Field(default_factory=dict)
