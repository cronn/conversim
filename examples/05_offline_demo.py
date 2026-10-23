"""The whole pipeline, offline: no API key, no GPU, no network.

Substitutes canned providers and a tone generator for the real backends, then walks the
generated dataset. Useful for seeing the shape of the output and for verifying an
install.

    uv run python examples/05_offline_demo.py

The fakes below are also the shortest possible answer to "how do I test code that uses
conversim?" -- both provider interfaces are ``typing.Protocol``, so a class with the
right method satisfies them.
"""

from __future__ import annotations

import json
import logging
import math
import shutil
from pathlib import Path

import numpy as np
from pydantic import BaseModel, Field

from conversim import (
    AudioConfig,
    CategoricalAxis,
    DatasetSpec,
    FlagAxis,
    GenerationConfig,
    Persona,
    RandomCasting,
    Role,
    SchemaSpec,
    SchemaTopicPlanner,
    StaticScenarioBuilder,
    TokenUsage,
    TopicCategory,
    VariationOption,
    VariationSpace,
    generate_dataset,
    load_dataset,
    synthesize_dataset,
)
from conversim.providers.base import StructuredResult, TextResult
from conversim.speech import (
    Audio,
    MixConfig,
    SwappedAssignment,
    Voice,
    VoiceLibrary,
)

OUT = Path("out/offline-demo")


# -- the domain --------------------------------------------------------------


class Booking(BaseModel):
    """What a restaurant reservation call should yield."""

    party_size: int | None = Field(default=None, description="How many people.")
    date: str = Field(default="not_discussed", description="Requested date.")
    time: str = Field(default="not_discussed", description="Requested time.")
    dietary_notes: list[str] = Field(default_factory=list, description="Dietary requirements.")
    confirmed: str = Field(default="not_discussed", description="Was it confirmed? yes/no.")


# -- fake backends -----------------------------------------------------------


class CannedStructured:
    """Satisfies StructuredGenerator by returning a fixed record."""

    model = "canned-structured"

    async def generate_structured(self, *, prompt: str, schema: type[BaseModel]):
        value = schema.model_validate(
            {
                "party_size": 4,
                "date": "next Friday",
                "time": "19:30",
                "dietary_notes": ["one vegetarian", "one nut allergy"],
                "confirmed": "yes",
            }
        )
        return StructuredResult(
            value=value,
            usage=TokenUsage(input_tokens=len(prompt) // 4, output_tokens=48),
            model=self.model,
        )


class CannedDialogue:
    """Satisfies TextGenerator by writing a fixed script in the right format."""

    model = "canned-dialogue"

    async def generate_text(self, *, prompt: str, max_output_tokens: int | None = None):
        # Pick whichever personas the prompt actually named.
        host = "Mara" if "Mara" in prompt else "Tom"
        guest = "Ida" if "Ida" in prompt else "Rui"
        lines = [
            f"{host}: Guten Abend, Restaurant Fenster, was kann ich für Sie tun?",
            f"{guest}: Hallo, ich wollte fragen, ob Sie nächsten Freitag noch was frei haben.",
            f"{host}: Für wie viele Personen wäre das denn?",
            f"{guest}: Wir wären vier. Am liebsten so gegen halb acht.",
            f"{host}: Halb acht bekomme ich hin, ja. Gibt es etwas, das ich der Küche "
            f"sagen sollte?",
            f"{guest}: Ja, eine Person isst vegetarisch, und einer von uns hat eine "
            f"Nussallergie, das ist wichtig.",
            f"{host}: Notiere ich beides. Dann steht das für Freitag, 19:30, vier "
            f"Personen. Auf welchen Namen?",
            f"{guest}: Perfekt, danke. Das passt so.",
        ]
        return TextResult(
            text="\n".join(lines),
            usage=TokenUsage(input_tokens=len(prompt) // 4, output_tokens=180),
            model=self.model,
        )


class ToneSynthesizer:
    """Satisfies SpeechSynthesizer with a tone whose pitch identifies the voice."""

    name = "tone-generator"

    def __init__(self, sample_rate: int = 16_000) -> None:
        self.sample_rate = sample_rate

    async def synthesize(self, *, text: str, voice: Voice, cue: str | None = None) -> Audio:
        seconds = max(0.4, len(text) * 0.045)
        frames = int(seconds * self.sample_rate)
        t = np.arange(frames, dtype=np.float32) / self.sample_rate
        frequency = 130.0 + 55.0 * (sum(map(ord, voice.id)) % 5)
        envelope = np.clip(np.sin(math.pi * t / seconds), 0.0, 1.0).astype(np.float32)
        return Audio(
            (0.45 * envelope * np.sin(2 * math.pi * frequency * t)).astype(np.float32),
            self.sample_rate,
        )


# -- the run -----------------------------------------------------------------


def build_spec() -> DatasetSpec:
    schema = SchemaSpec(
        Booking,
        sentinel_docs={
            "not_discussed": "Kam im Gespräch nicht vor.",
            "unknown": "Kam vor, blieb aber offen.",
            "not_applicable": "Trifft nicht zu.",
        },
    )

    variation = VariationSpace(
        categorical=(
            CategoricalAxis(
                name="style",
                label="Tempo",
                options=(
                    VariationOption(
                        value="quick",
                        instruction="kurz und geschäftig",
                        metadata={"turn_range": (6, 10)},
                    ),
                    VariationOption(
                        value="chatty",
                        instruction="freundlich und ausschweifend",
                        metadata={"turn_range": (12, 20)},
                    ),
                ),
            ),
        ),
        flags=(
            FlagAxis(
                name="fully_booked", instruction="Das Wunschdatum ist ausgebucht.", probability=0.3
            ),
        ),
    )

    categories = (
        TopicCategory(
            name="greeting",
            description="Begrüßung und Anliegen",
            position="early",
            always_include=True,
            priority=1,
        ),
        TopicCategory(
            name="details",
            description="Datum, Uhrzeit, Personenzahl",
            field_paths=("date", "time", "party_size"),
            priority=3,
        ),
        TopicCategory(
            name="dietary",
            description="Besondere Wünsche an die Küche",
            field_paths=("dietary_notes",),
            priority=5,
        ),
        TopicCategory(
            name="closing",
            description="Bestätigung und Abschluss",
            field_paths=("confirmed",),
            position="late",
            priority=9,
        ),
    )

    return DatasetSpec(
        schema=schema,
        roles=(
            Role(id="host", label="Restaurant", description="Nimmt die Reservierung auf."),
            Role(id="guest", label="Gast", description="Möchte reservieren."),
        ),
        casting=RandomCasting(
            pools={
                "host": (
                    Persona(name="Mara", role="host", description="Freundlich, routiniert."),
                    Persona(name="Tom", role="host", description="Knapp, effizient."),
                ),
                "guest": (
                    Persona(name="Ida", role="guest", description="Höflich, gut vorbereitet."),
                    Persona(name="Rui", role="guest", description="Unentschlossen."),
                ),
            }
        ),
        scenario=StaticScenarioBuilder(
            topic="eine Tischreservierung am Telefon",
            context="Kleines Restaurant, früher Abend.",
            language="Deutsch",
            turn_range_axis="style",
        ),
        variation=variation,
        topic_planner=SchemaTopicPlanner(schema, categories),
        name="offline-demo",
        description="Offline demonstration of the conversim pipeline.",
    )


def main() -> None:
    logging.basicConfig(level=logging.WARNING, format="%(levelname)s %(name)s: %(message)s")
    if OUT.exists():
        shutil.rmtree(OUT)

    spec = build_spec()
    print("Variation space")
    print(spec.variation.describe())

    print("\n== stage 1+2: ground truth and dialogue ==")
    result = generate_dataset(
        OUT,
        spec=spec,
        structured_provider=CannedStructured(),
        text_provider=CannedDialogue(),
        config=GenerationConfig(count=4, seed=7, concurrency=2),
    )
    print(result.summary())

    print("\n== stage 3: audio ==")
    library = VoiceLibrary(
        voices=(
            Voice(id="lo", language="Deutsch", tags=("male",)),
            Voice(id="hi", language="Deutsch", tags=("female",)),
        )
    )
    audio = synthesize_dataset(
        OUT,
        AudioConfig(
            synthesizer=ToneSynthesizer(),
            library=library,
            # Two takes with the voices exchanged.
            assignment=SwappedAssignment(limit=2),
            mix=MixConfig(pause_mean_s=0.3, pause_sd_s=0.25, pan_width=0.4),
        ),
    )
    print(audio.summary())

    print("\n== what came out ==")
    dataset = load_dataset(OUT)
    manifest = dataset.manifest()
    assert manifest is not None
    print(f"root:   {dataset.root}")
    print(f"seed:   {manifest.seed}")
    print(f"space:  {manifest.variation_cardinality:,} configurations")
    print(f"audio:  {manifest.take_count} takes, {manifest.audio_duration_s:.1f}s")

    sample = dataset.load(dataset.sample_ids()[0])
    print(f"\nsample {sample.id}")
    print(f"  variation: {sample.record.variation.summary()}")
    print(f"  topics:    {' -> '.join(sample.record.topic_plan.names)}")
    print(f"  turns:     {sample.record.turn_count}, words: {sample.record.word_count}")
    print(f"  clean:     {sample.record.is_clean}")
    print(f"  label:     {json.dumps(sample.ground_truth, ensure_ascii=False)}")
    print(f"  transcript: {sample.dialogue.transcript()[:90]}...")
    for take in sample.takes:
        cast = ", ".join(f"{role}={voice}" for role, voice in take.assignment.voices.items())
        print(f"  take {take.id}: {take.duration_s:.1f}s, {cast}")
        first = take.timings[0]
        print(f"    turn 0: {first.start_s:.2f}-{first.end_s:.2f}s by {first.speaker}")

    print(f"\nBrowse it: find {OUT} -type f | sort")


if __name__ == "__main__":
    main()
