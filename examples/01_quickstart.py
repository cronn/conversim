"""Minimal end-to-end run against real providers.

Needs API keys and will spend money. Set OPENAI_API_KEY and GOOGLE_API_KEY, or pass
the keys explicitly as shown. To see the pipeline work without either, run
``05_offline_demo.py`` instead.

    uv run --extra openai --extra google python examples/01_quickstart.py
"""

from __future__ import annotations

import logging

from pydantic import BaseModel, Field

from conversim import (
    DatasetSpec,
    GenerationConfig,
    Persona,
    RandomCasting,
    Role,
    SchemaSpec,
    StaticScenarioBuilder,
    TurnTarget,
    generate_dataset,
    load_dataset,
)
from conversim.providers import GoogleProvider, OpenAIProvider


class Problem(BaseModel):
    """What the customer is reporting."""

    summary: str = Field(default="not_discussed", description="Was ausgefallen ist.")
    product: str = Field(default="not_discussed", description="Betroffenes Produkt.")
    started: str = Field(default="not_discussed", description="Seit wann.")


class SupportTicket(BaseModel):
    """The structured record a support call should yield."""

    problem: Problem = Field(default_factory=Problem)
    resolved_on_call: str = Field(
        default="not_discussed", description="Wurde das Problem im Gespräch gelöst? yes/no."
    )
    callback_requested: str = Field(
        default="not_discussed", description="Wurde ein Rückruf vereinbart? yes/no."
    )


def main() -> None:
    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(name)s: %(message)s")

    spec = DatasetSpec(
        schema=SchemaSpec(SupportTicket),
        roles=(
            Role(id="agent", label="Support-Mitarbeiter:in"),
            Role(id="customer", label="Kund:in"),
        ),
        casting=RandomCasting(
            pools={
                "agent": (
                    Persona(name="Sam", role="agent", description="Ruhig und methodisch."),
                    Persona(name="Nora", role="agent", description="Schnell und direkt."),
                ),
                "customer": (
                    Persona(name="Kim", role="customer", description="In Eile, leicht genervt."),
                    Persona(name="Ben", role="customer", description="Geduldig, technikfern."),
                ),
            }
        ),
        scenario=StaticScenarioBuilder(
            topic="ein Support-Anruf zu einem Softwareproblem",
            context="Telefon-Hotline eines Softwareanbieters, Geschäftszeiten.",
            language="Deutsch",
            turns=TurnTarget(minimum=10, maximum=18),
        ),
        name="support-calls",
    )

    result = generate_dataset(
        "out/support-calls",
        spec=spec,
        # Gemini enforces the schema server-side, which suits the ground-truth stage.
        structured_provider=GoogleProvider(model="gemini-2.5-flash", thinking_budget=0),
        text_provider=OpenAIProvider(model="gpt-5-mini"),
        config=GenerationConfig(count=4, seed=1, concurrency=2),
    )
    print(result.summary())

    for sample in load_dataset(result.root):
        print(f"\n--- {sample.id} ({sample.record.turn_count} turns) ---")
        print(sample.dialogue.to_script()[:400])


if __name__ == "__main__":
    main()
