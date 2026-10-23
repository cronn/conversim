"""Shared fixtures and stub backends.

The stubs are what let the whole pipeline be exercised without touching a network or a
GPU. They implement the same protocols the real adapters do, which is the practical
argument for those protocols existing: a fake that satisfies ``StructuredGenerator`` is
twenty lines, and no part of the pipeline can tell the difference.
"""

from __future__ import annotations

import math

import numpy as np
import pytest
from pydantic import BaseModel, Field

from conversim import (
    Cast,
    DatasetSpec,
    FixedCasting,
    Persona,
    Role,
    SchemaSpec,
    StaticScenarioBuilder,
    TurnTarget,
)
from conversim.models import TokenUsage
from conversim.providers.base import StructuredResult, TextResult
from conversim.speech.base import Audio, Voice
from conversim.speech.voices import VoiceLibrary

# -- a small domain to test against ----------------------------------------


class Problem(BaseModel):
    """Nested section, to exercise dotted field paths."""

    summary: str = Field(default="not_discussed", description="What is failing.")
    first_seen: str = Field(default="not_discussed", description="When it started.")
    severity: str = Field(default="not_discussed", description="How bad it is.")


class Resolution(BaseModel):
    """Nested section with a list field."""

    actions: list[str] = Field(default_factory=list, description="Agreed next steps.")
    follow_up_days: int | None = Field(default=None, description="Days until follow-up.")


class Ticket(BaseModel):
    """Test schema: two nested sections and a top-level field."""

    reference: str = Field(default="not_discussed", description="Ticket reference.")
    problem: Problem = Field(default_factory=Problem)
    resolution: Resolution = Field(default_factory=Resolution)


AGENT = Role(id="agent", label="Agent", description="Handles the call.")
CUSTOMER = Role(id="customer", label="Customer", description="Reports the problem.")

SAM = Persona(name="Sam", role="agent", description="Patient and methodical.")
KIM = Persona(name="Kim", role="customer", description="In a hurry.")


# -- stub providers ---------------------------------------------------------


class StubStructuredProvider:
    """Returns a filled schema instance without calling anything.

    The instance varies with the prompt, so tests can tell reproducible sampling from
    accidental constancy.
    """

    def __init__(self, *, model: str = "stub-structured", fail_times: int = 0) -> None:
        self._model = model
        self.calls = 0
        self.fail_times = fail_times

    @property
    def model(self) -> str:
        return self._model

    async def generate_structured(self, *, prompt: str, schema: type[BaseModel]):
        self.calls += 1
        if self.calls <= self.fail_times:
            raise RuntimeError(f"stub failure {self.calls}")
        marker = f"len{len(prompt)}"
        value = schema.model_validate(
            {
                "reference": marker,
                "problem": {
                    "summary": "printer offline",
                    "first_seen": "last Tuesday",
                    "severity": "not_discussed",
                },
                "resolution": {"actions": ["restart spooler"], "follow_up_days": 3},
            }
        )
        return StructuredResult(
            value=value,
            usage=TokenUsage(input_tokens=len(prompt) // 4, output_tokens=64),
            model=self._model,
            raw_text=value.model_dump_json(),
        )


class StubTextProvider:
    """Writes a well-formed script using whichever cast names appear in the prompt."""

    def __init__(
        self,
        *,
        model: str = "stub-text",
        turns: int = 8,
        personas: tuple[Persona, ...] = (SAM, KIM),
        truncated: bool = False,
        preamble: str = "",
    ) -> None:
        self._model = model
        self.turns = turns
        self.personas = personas
        self.truncated = truncated
        self.preamble = preamble
        self.calls = 0

    @property
    def model(self) -> str:
        return self._model

    async def generate_text(self, *, prompt: str, max_output_tokens: int | None = None):
        self.calls += 1
        present = [persona.name for persona in self.personas if persona.name in prompt]
        if len(present) < 2:
            present = [persona.name for persona in self.personas[:2]]
        lines = [self.preamble] if self.preamble else []
        for index in range(self.turns):
            speaker = present[index % len(present)]
            lines.append(f"{speaker}: Turn {index} about the printer being offline.")
        return TextResult(
            text="\n".join(lines),
            usage=TokenUsage(input_tokens=len(prompt) // 4, output_tokens=self.turns * 12),
            model=self._model,
            truncated=self.truncated,
        )


class StubSynthesizer:
    """Renders a tone whose length scales with the text, at a fixed sample rate."""

    def __init__(self, *, sample_rate: int = 16_000, seconds_per_char: float = 0.02) -> None:
        self.sample_rate = sample_rate
        self.seconds_per_char = seconds_per_char
        self.calls: list[tuple[str, str]] = []

    @property
    def name(self) -> str:
        return "stub-tts"

    async def synthesize(self, *, text: str, voice: Voice, cue: str | None = None) -> Audio:
        self.calls.append((voice.id, text))
        frames = max(1, int(len(text) * self.seconds_per_char * self.sample_rate))
        time = np.arange(frames, dtype=np.float32) / self.sample_rate
        # A distinct pitch per voice makes it audible which voice a mix used.
        frequency = 110.0 + 40.0 * (sum(map(ord, voice.id)) % 8)
        samples = 0.4 * np.sin(2.0 * math.pi * frequency * time).astype(np.float32)
        return Audio(samples, self.sample_rate)


# -- fixtures ---------------------------------------------------------------


@pytest.fixture
def ticket_schema() -> SchemaSpec:
    return SchemaSpec(Ticket)


@pytest.fixture
def cast() -> Cast:
    return Cast(personas=(SAM, KIM))


@pytest.fixture
def spec(ticket_schema: SchemaSpec, cast: Cast) -> DatasetSpec:
    return DatasetSpec(
        schema=ticket_schema,
        roles=(AGENT, CUSTOMER),
        casting=FixedCasting(value=cast),
        scenario=StaticScenarioBuilder(
            topic="a support call",
            context="Helpdesk, business hours.",
            language="English",
            turns=TurnTarget(minimum=6, maximum=12),
        ),
        name="test-tickets",
    )


@pytest.fixture
def voice_library() -> VoiceLibrary:
    return VoiceLibrary(
        voices=(
            Voice(id="anna", language="English", preset="alloy", tags=("female",)),
            Voice(id="ben", language="English", preset="echo", tags=("male",)),
        )
    )
