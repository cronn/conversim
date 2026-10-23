"""Add speech to an existing dataset using local Qwen3-TTS voice cloning.

Runs entirely on your machine; no API key involved.

    uv run --extra qwen python examples/03_synthesize_audio.py \
        --dataset out/sti --voices actors --device cuda:1 --takes 2

The voices directory holds ``name.wav`` plus ``name.txt`` pairs, where the text is the
verbatim transcript of the recording. A few seconds of clean speech per voice is
enough; the transcript needs to be accurate, because a wrong one degrades the clone in
a way that is easy to miss on one sample and obvious across a whole dataset.

Pick the CUDA device explicitly on a shared machine. ``--takes 2`` renders each
consultation twice with the two voices exchanged: identical words, different speaker
per line.
"""

from __future__ import annotations

import argparse
import logging
from pathlib import Path

from conversim import AudioConfig, GenerationEvent, load_dataset, synthesize_dataset
from conversim.speech import (
    BackchannelConfig,
    BackgroundConfig,
    MixConfig,
    QwenTTS,
    RandomAssignment,
    SwappedAssignment,
    VoiceLibrary,
)


def progress(event: GenerationEvent) -> None:
    if event.kind == "take_finished":
        print(f"  take {event.detail}")
    elif event.kind == "sample_finished":
        print(f"[{event.index + 1}/{event.total}] {event.status} {event.sample_id}")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset", type=Path, required=True, help="Dataset root")
    parser.add_argument("--voices", type=Path, required=True, help="Voice directory")
    parser.add_argument("--device", default="cuda:0", help="Torch device, e.g. cuda:1")
    parser.add_argument("--takes", type=int, default=1, help="Takes per consultation")
    parser.add_argument("--limit", type=int, default=None, help="Stop after N samples")
    parser.add_argument("--background", type=Path, default=None, help="Room tone WAV")
    parser.add_argument(
        "--backchannels", type=Path, default=None, help="Backchannel clips (tag subdirectories)"
    )
    parser.add_argument("--overlap", action="store_true", help="Allow turns to overlap")
    args = parser.parse_args()

    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(name)s: %(message)s")

    library = VoiceLibrary.from_directory(args.voices)
    print(f"Voices: {', '.join(library.ids)}")

    mix = MixConfig(
        pause_mean_s=-0.1 if args.overlap else 0.25,
        pause_sd_s=0.45,
        allow_overlap=args.overlap,
        pan_width=0.35,
        background=BackgroundConfig(path=args.background) if args.background else None,
        backchannels=(
            BackchannelConfig(directory=args.backchannels) if args.backchannels else None
        ),
    )

    result = synthesize_dataset(
        args.dataset,
        AudioConfig(
            synthesizer=QwenTTS(device=args.device),
            library=library,
            assignment=(
                SwappedAssignment(limit=args.takes) if args.takes > 1 else RandomAssignment(takes=1)
            ),
            mix=mix,
            limit=args.limit,
            # One conversation at a time: the backend serialises GPU work internally.
            concurrency=1,
            on_event=progress,
        ),
    )

    print(f"\n{result.summary()}")
    for outcome in result.failed:
        print(f"  failed {outcome.sample_id}/{outcome.take_id}: {outcome.error}")

    dataset = load_dataset(args.dataset)
    for sample in dataset:
        for take in sample.takes:
            speakers = ", ".join(
                f"{role}={voice}" for role, voice in take.assignment.voices.items()
            )
            print(f"{sample.id}/{take.id}: {take.duration_s:6.1f}s  {speakers}")
        break  # one example is enough


if __name__ == "__main__":
    main()
