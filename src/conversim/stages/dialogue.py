"""Stage 2: turn the facts into a conversation.

The dialogue is both the ASR transcription target and the input a structured-extraction
model sees, so two properties matter: every substantive ground-truth value has to be
recoverable from it, and it has to sound like speech. Those pull against each other,
and the prompt resolves the tension in favour of naturalness by allowing most schema
fields to go unmentioned (see :mod:`conversim.prompts`).
"""

from __future__ import annotations

import logging
import random
from dataclasses import dataclass

from pydantic import BaseModel

from ..errors import DialogueFormatError
from ..models import Cast, Dialogue, DialogueIssue, Role, Scenario, TokenUsage
from ..parsing import ContinuationPolicy, DialogueParser
from ..prompts import (
    DefaultDialoguePromptBuilder,
    DialoguePromptBuilder,
    DialoguePromptContext,
)
from ..providers.base import TextGenerator
from ..schema import SchemaSpec
from ..topics import TopicPlan
from ..variation import VariationConfig

__all__ = ["DialogueGenerator", "DialogueResult", "DialogueValidator"]

logger = logging.getLogger(__name__)


@dataclass(frozen=True)
class DialogueResult:
    """Output of the dialogue stage.

    Attributes:
        dialogue: Parsed utterances, raw model output, and any issues found.
        prompt: The prompt that produced it.
        usage: Tokens consumed.
        model: Model that wrote the dialogue.
    """

    dialogue: Dialogue
    prompt: str
    usage: TokenUsage
    model: str


@dataclass
class DialogueValidator:
    """Checks a parsed dialogue for problems worth knowing about.

    Two classes of problem are distinguished:

    * **Fatal** -- nothing parsed, or a role never speaks. Such a sample is unusable:
      a one-sided "conversation" teaches an extraction model nothing and gives a
      speech backend no one to cast.
    * **Reported** -- turn count outside the requested range, discarded lines,
      truncated output. Recorded on the sample and left to the caller to filter on.

    Args:
        turn_tolerance: How far the turn count may fall outside the requested range
            before it is reported.
        require_all_roles: Treat a silent role as fatal.
        strict: Raise on fatal problems instead of reporting them.
    """

    turn_tolerance: int = 5
    require_all_roles: bool = True
    strict: bool = True

    def validate(
        self,
        *,
        dialogue: Dialogue,
        cast: Cast,
        scenario: Scenario,
        truncated: bool = False,
    ) -> tuple[DialogueIssue, ...]:
        """Return issues found, raising on fatal ones when :attr:`strict`."""
        issues: list[DialogueIssue] = list(dialogue.issues)
        fatal: list[str] = []

        if not dialogue.utterances:
            fatal.append("no utterances could be parsed from the model output")

        if self.require_all_roles and dialogue.utterances:
            spoken = {utterance.role for utterance in dialogue.utterances}
            silent = [role for role in cast.role_ids if role not in spoken]
            if silent:
                fatal.append(f"roles never speak: {', '.join(sorted(silent))}")

        if truncated:
            issues.append(
                DialogueIssue(
                    kind="truncated",
                    detail="Generation stopped at the output-token limit; "
                    "the dialogue is probably cut off mid-turn",
                )
            )

        count = len(dialogue.utterances)
        target = scenario.turns
        if count and not (
            target.minimum - self.turn_tolerance <= count <= target.maximum + self.turn_tolerance
        ):
            issues.append(
                DialogueIssue(
                    kind="turn_count",
                    detail=f"{count} turns, requested {target.describe()} "
                    f"(tolerance {self.turn_tolerance})",
                    # A dialogue outside the requested range is still a usable
                    # dialogue, so this must not disqualify the sample.
                    severity="minor",
                )
            )

        if fatal:
            summary = "; ".join(fatal)
            if self.strict:
                raise DialogueFormatError(f"Unusable dialogue: {summary}")
            issues.append(DialogueIssue(kind="unusable", detail=summary))

        return tuple(issues)


class DialogueGenerator:
    """Produces, parses and validates the dialogue for one conversation.

    Args:
        schema: The ground-truth schema, used by the prompt builder for sentinel and
            field guidance.
        provider: A text generator, e.g.
            :class:`~conversim.providers.openai_provider.OpenAIProvider`.
        roles: Structural roles taking part, in speaking order.
        prompt_builder: Override the default domain-neutral prompt.
        validator: Override the default validation thresholds.
        max_output_tokens: Output cap for the dialogue call. Long conversations need
            real headroom; too low a value silently truncates them.
        include_cues: Ask for ``(cue)`` delivery annotations. Only worth enabling when
            the speech backend can act on them.
        continuation_policy: How the parser treats lines without a speaker prefix.
    """

    def __init__(
        self,
        *,
        schema: SchemaSpec,
        provider: TextGenerator,
        roles: tuple[Role, ...],
        prompt_builder: DialoguePromptBuilder | None = None,
        validator: DialogueValidator | None = None,
        max_output_tokens: int | None = 16_000,
        include_cues: bool = False,
        continuation_policy: ContinuationPolicy = "drop",
    ) -> None:
        self.schema = schema
        self.provider = provider
        self.roles = roles
        self.prompt_builder = prompt_builder or DefaultDialoguePromptBuilder()
        self.validator = validator or DialogueValidator()
        self.max_output_tokens = max_output_tokens
        self.include_cues = include_cues
        self.continuation_policy = continuation_policy

    async def generate(
        self,
        *,
        scenario: Scenario,
        cast: Cast,
        ground_truth: BaseModel,
        variation: VariationConfig,
        topic_plan: TopicPlan,
        rng: random.Random,
    ) -> DialogueResult:
        """Generate, parse and validate one dialogue.

        Raises:
            ProviderError: if the provider fails after exhausting its retries.
            DialogueFormatError: if the result is unusable and the validator is strict.
        """
        prompt = self.prompt_builder.build(
            DialoguePromptContext(
                schema=self.schema,
                scenario=scenario,
                roles=self.roles,
                cast=cast,
                ground_truth=ground_truth,
                variation=variation,
                topic_plan=topic_plan,
                include_cues=self.include_cues,
                rng=rng,
            )
        )
        logger.debug("Requesting dialogue (%d prompt chars)", len(prompt))

        result = await self.provider.generate_text(
            prompt=prompt, max_output_tokens=self.max_output_tokens
        )

        parser = DialogueParser(
            cast=cast,
            continuation_policy=self.continuation_policy,
            capture_cues=self.include_cues,
        )
        dialogue = parser.parse(result.text)
        issues = self.validator.validate(
            dialogue=dialogue, cast=cast, scenario=scenario, truncated=result.truncated
        )
        dialogue = dialogue.model_copy(update={"issues": issues})

        for issue in issues:
            logger.info("Dialogue issue [%s]: %s", issue.kind, issue.detail)

        return DialogueResult(
            dialogue=dialogue,
            prompt=prompt,
            usage=result.usage,
            model=result.model,
        )
