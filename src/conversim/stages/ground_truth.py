"""Stage 1: invent the facts.

Samples are generated in reverse order of consumption -- ground truth first, then a
dialogue that conveys it, then audio. Doing it the other way round (write a dialogue,
then extract a label from it) makes the label only as good as the extractor, which is
precisely the thing the dataset is supposed to evaluate.
"""

from __future__ import annotations

import logging
import random
from dataclasses import dataclass

from pydantic import BaseModel

from ..models import Scenario, TokenUsage
from ..prompts import (
    DefaultGroundTruthPromptBuilder,
    GroundTruthPromptBuilder,
    GroundTruthPromptContext,
)
from ..providers.base import StructuredGenerator
from ..schema import SchemaSpec
from ..variation import VariationConfig

__all__ = ["GroundTruthGenerator", "GroundTruthResult"]

logger = logging.getLogger(__name__)


@dataclass(frozen=True)
class GroundTruthResult:
    """Output of the ground-truth stage.

    Attributes:
        value: The validated schema instance. This is the dataset's extraction label.
        prompt: The prompt that produced it, kept so a sample can be traced or replayed.
        usage: Tokens consumed.
        model: Model that produced the instance.
    """

    value: BaseModel
    prompt: str
    usage: TokenUsage
    model: str


class GroundTruthGenerator:
    """Produces schema-conformant ground truth for one conversation.

    Args:
        schema: The target schema and its sentinel conventions.
        provider: A structured generator, e.g.
            :class:`~conversim.providers.google_provider.GoogleProvider`.
        prompt_builder: Override the default domain-neutral prompt.
    """

    def __init__(
        self,
        *,
        schema: SchemaSpec,
        provider: StructuredGenerator,
        prompt_builder: GroundTruthPromptBuilder | None = None,
    ) -> None:
        self.schema = schema
        self.provider = provider
        self.prompt_builder = prompt_builder or DefaultGroundTruthPromptBuilder()

    async def generate(
        self,
        *,
        scenario: Scenario,
        variation: VariationConfig,
        rng: random.Random,
    ) -> GroundTruthResult:
        """Generate one ground-truth instance.

        Raises:
            ProviderError: if the provider fails after exhausting its retries.
            StructuredOutputError: if the output cannot be made to conform.
        """
        prompt = self.prompt_builder.build(
            GroundTruthPromptContext(
                schema=self.schema,
                scenario=scenario,
                variation=variation,
                rng=rng,
            )
        )
        logger.debug("Requesting %s instance (%d prompt chars)", self.schema.name, len(prompt))

        result = await self.provider.generate_structured(prompt=prompt, schema=self.schema.model)
        return GroundTruthResult(
            value=result.value,
            prompt=prompt,
            usage=result.usage,
            model=result.model,
        )
