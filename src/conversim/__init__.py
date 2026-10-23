"""conversim -- synthetic dialogue datasets with aligned structured ground truth.

conversim generates conversations for which the correct structured record is known by
construction, because the record comes first. Samples are produced in reverse order of
consumption:

1. **Ground truth** -- a validated instance of your Pydantic schema, invented by a
   model that supports structured output.
2. **Dialogue** -- a naturalistic conversation written from that ground truth, parsed
   into utterances.
3. **Audio** -- optional utterance-level speech synthesis, mixed into a recording with
   panning, varied pacing, backchannels and room tone.

Deriving a label from a transcript afterwards would only ever be as good as the
extractor used to derive it, which is usually the very thing under evaluation. Starting
from the label avoids that circularity.

Nothing in the core is domain-specific: you supply the schema, the roles, the personas
and the variation axes. :mod:`conversim.presets.sti` is a complete worked example --
a German clinical history-taking dataset.

Quick start::

    from conversim import (
        DatasetSpec, GenerationConfig, Role, SchemaSpec,
        RandomCasting, StaticScenarioBuilder, generate_dataset,
    )
    from conversim.providers import GoogleProvider, OpenAIProvider

    spec = DatasetSpec(
        schema=SchemaSpec(MyRecord),
        roles=(Role(id="agent", label="Agent"), Role(id="customer", label="Customer")),
        casting=RandomCasting(pools={"agent": agents, "customer": customers}),
        scenario=StaticScenarioBuilder(topic="a support call", context="Helpdesk."),
    )

    result = generate_dataset(
        "out/support-calls",
        spec=spec,
        structured_provider=GoogleProvider(model="gemini-2.5-flash"),
        text_provider=OpenAIProvider(model="gpt-5-mini"),
        config=GenerationConfig(count=50, seed=1),
    )

API keys are never read from disk by conversim. Pass them explicitly, or let the
provider SDK resolve its own environment variable.
"""

from __future__ import annotations

from .casting import (
    CastingStrategy,
    ExhaustiveCasting,
    FixedCasting,
    RandomCasting,
    ScenarioBuilder,
    StaticScenarioBuilder,
)
from .dataset import (
    DatasetManifest,
    DatasetReader,
    DatasetWriter,
    LoadedSample,
    SampleIndexEntry,
    SampleRecord,
    TakeRecord,
    load_dataset,
)
from .errors import (
    ConfigurationError,
    ConversimError,
    DialogueFormatError,
    MissingDependencyError,
    ProviderError,
    SchemaError,
    StructuredOutputError,
    SynthesisError,
)
from .models import (
    Cast,
    Dialogue,
    DialogueIssue,
    Persona,
    Role,
    Scenario,
    TokenUsage,
    TurnTarget,
    Utterance,
)
from .parsing import DialogueParser
from .pipeline import (
    AudioConfig,
    AudioResult,
    DatasetGenerator,
    DatasetResult,
    DatasetSpec,
    GenerationConfig,
    GenerationEvent,
    SampleOutcome,
    TakeOutcome,
    generate_dataset,
    synthesize_dataset,
)
from .prompts import (
    DefaultDialoguePromptBuilder,
    DefaultGroundTruthPromptBuilder,
    DialoguePromptBuilder,
    DialoguePromptContext,
    GroundTruthPromptBuilder,
    GroundTruthPromptContext,
)
from .schema import DEFAULT_SENTINELS, SchemaSpec
from .stages import (
    DialogueGenerator,
    DialogueResult,
    DialogueValidator,
    GroundTruthGenerator,
    GroundTruthResult,
)
from .topics import (
    PriorityRule,
    SchemaTopicPlanner,
    TopicCategory,
    TopicPlan,
    TopicPlanner,
)
from .variation import (
    CategoricalAxis,
    FlagAxis,
    VariationConfig,
    VariationOption,
    VariationSpace,
)

__all__ = [
    # schema
    "SchemaSpec",
    "DEFAULT_SENTINELS",
    # domain models
    "Role",
    "Persona",
    "Cast",
    "Scenario",
    "TurnTarget",
    "Utterance",
    "Dialogue",
    "DialogueIssue",
    "TokenUsage",
    # variation
    "VariationSpace",
    "CategoricalAxis",
    "FlagAxis",
    "VariationOption",
    "VariationConfig",
    # topics
    "TopicCategory",
    "TopicPlan",
    "TopicPlanner",
    "SchemaTopicPlanner",
    "PriorityRule",
    # casting and scenario
    "CastingStrategy",
    "FixedCasting",
    "RandomCasting",
    "ExhaustiveCasting",
    "ScenarioBuilder",
    "StaticScenarioBuilder",
    # prompts
    "GroundTruthPromptBuilder",
    "DialoguePromptBuilder",
    "GroundTruthPromptContext",
    "DialoguePromptContext",
    "DefaultGroundTruthPromptBuilder",
    "DefaultDialoguePromptBuilder",
    # stages
    "GroundTruthGenerator",
    "GroundTruthResult",
    "DialogueGenerator",
    "DialogueResult",
    "DialogueValidator",
    "DialogueParser",
    # pipeline
    "DatasetSpec",
    "GenerationConfig",
    "AudioConfig",
    "DatasetGenerator",
    "DatasetResult",
    "AudioResult",
    "SampleOutcome",
    "TakeOutcome",
    "GenerationEvent",
    "generate_dataset",
    "synthesize_dataset",
    # dataset io
    "DatasetWriter",
    "DatasetReader",
    "LoadedSample",
    "SampleRecord",
    "TakeRecord",
    "DatasetManifest",
    "SampleIndexEntry",
    "load_dataset",
    # errors
    "ConversimError",
    "ConfigurationError",
    "SchemaError",
    "ProviderError",
    "StructuredOutputError",
    "DialogueFormatError",
    "SynthesisError",
    "MissingDependencyError",
]

__version__ = "0.1.0"
