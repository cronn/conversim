"""Generate the German STI/HIV consultation dataset from the bundled preset.

This is the configuration the library was extracted from. Needs API keys and will
spend money.

    uv run --extra openai --extra google python examples/02_sti_dataset.py --count 20

The preset supplies everything: schema, variation space, topic categories with
priority rules, personas, and German prompt builders.
"""

from __future__ import annotations

import argparse
import logging
from pathlib import Path

from conversim import GenerationConfig, GenerationEvent, generate_dataset, load_dataset
from conversim.presets.sti import sti_dataset_spec, sti_prompt_builders
from conversim.providers import GoogleProvider, OpenAIProvider


def progress(event: GenerationEvent) -> None:
    """Print one line per finished sample. The library itself never prints."""
    if event.kind == "sample_finished":
        marker = {"generated": "ok", "skipped": "--", "failed": "!!"}[event.status or "failed"]
        detail = f"  {event.detail}" if event.detail else ""
        print(f"[{event.index + 1}/{event.total}] {marker} {event.sample_id}{detail}")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--count", type=int, default=10, help="How many consultations")
    parser.add_argument("--out", type=Path, default=Path("out/sti"), help="Dataset root")
    parser.add_argument("--seed", type=int, default=1, help="Root seed")
    parser.add_argument("--concurrency", type=int, default=4, help="Samples in flight")
    parser.add_argument(
        "--dialogue-model", default="gpt-5-mini", help="Model writing the dialogues"
    )
    parser.add_argument(
        "--schema-model", default="gemini-2.5-flash", help="Model filling the schema"
    )
    parser.add_argument(
        "--all-pairings",
        action="store_true",
        help="Walk all 40 clinician-patient pairings instead of drawing at random",
    )
    args = parser.parse_args()

    logging.basicConfig(level=logging.WARNING, format="%(levelname)s %(name)s: %(message)s")

    spec = sti_dataset_spec(exhaustive_casting=args.all_pairings)
    ground_truth_prompt, dialogue_prompt = sti_prompt_builders()

    print(f"Variation space: {spec.variation.cardinality():,} configurations")

    result = generate_dataset(
        args.out,
        spec=spec,
        structured_provider=GoogleProvider(model=args.schema_model, thinking_budget=0),
        text_provider=OpenAIProvider(model=args.dialogue_model),
        config=GenerationConfig(
            count=args.count,
            seed=args.seed,
            concurrency=args.concurrency,
            on_event=progress,
        ),
        ground_truth_prompt=ground_truth_prompt,
        dialogue_prompt=dialogue_prompt,
        # Long consultations need real headroom; too low a cap truncates them silently.
        max_output_tokens=16_000,
        # Qwen3-TTS takes no delivery cues, and this dataset is voiced with it.
        include_cues=False,
    )

    print(f"\n{result.summary()}")
    for outcome in result.failed:
        print(f"  failed {outcome.id}: {outcome.error}")

    dataset = load_dataset(result.root)
    turns = [entry.turn_count for entry in dataset.index()]
    words = sum(entry.word_count for entry in dataset.index())
    flagged = [entry.id for entry in dataset.index() if not entry.is_clean]
    if turns:
        print(
            f"{len(turns)} samples, {sum(turns) / len(turns):.1f} turns on average "
            f"({min(turns)}-{max(turns)}), {words:,} words total"
        )
    if flagged:
        print(f"{len(flagged)} samples carry major issues: {', '.join(flagged[:5])}")
    print(f"\nNext: add audio with examples/03_synthesize_audio.py --dataset {result.root}")


if __name__ == "__main__":
    main()
