"""On-disk layout of a generated dataset.

The shape is deliberately boring -- one directory per sample, plain JSON and text --
because a synthetic dataset gets read by hand far more often than its authors expect.
Every fact about a sample sits next to the sample, and the root-level index exists to
make bulk loading cheap rather than to be the source of truth.

::

    <root>/
      dataset.json                        run-level manifest
      samples.jsonl                       one index line per sample (derived)
      samples/<sample_id>/
        sample.json                       provenance: scenario, cast, variation, models
        ground_truth.json                 the extraction label
        dialogue.jsonl                    utterances, one JSON object per line
        dialogue.txt                      the same, as a readable script
        dialogue.raw.txt                  raw model output, only if it differed
        transcript.txt                    spoken words only: the ASR target
        prompts/ground_truth.txt
        prompts/dialogue.txt
        audio/
          segments/<voice_id>/0000.wav    per-turn renders, shared across takes
          takes/<take_id>.wav             mixed conversation
          takes/<take_id>.json            assignment, timings, mix settings

Segments are keyed by voice *and* turn rather than by take. Rendering the same
dialogue with two voices swapped therefore needs each (voice, turn) pair exactly once
instead of once per take, and an interrupted run resumes at the granularity of a
single utterance.
"""

from __future__ import annotations

import re
from dataclasses import dataclass
from pathlib import Path

from ..errors import ConfigurationError

__all__ = ["INDEX_NAME", "MANIFEST_NAME", "DatasetLayout"]

MANIFEST_NAME = "dataset.json"
INDEX_NAME = "samples.jsonl"

_SAFE_NAME = re.compile(r"^[A-Za-z0-9][A-Za-z0-9._-]*$")


def _check_component(value: str, kind: str) -> str:
    """Reject identifiers that would escape the dataset directory or break filesystems."""
    if not _SAFE_NAME.match(value):
        raise ConfigurationError(
            f"{kind} {value!r} is not usable as a path component; use letters, digits, "
            f"'.', '_' and '-', starting with a letter or digit"
        )
    return value


@dataclass(frozen=True)
class DatasetLayout:
    """Resolves paths within a dataset root.

    All methods are pure path arithmetic -- nothing here touches the filesystem, which
    keeps the layout usable for inspecting a dataset that is not present locally.

    Args:
        root: Dataset root directory.
    """

    root: Path

    def __post_init__(self) -> None:
        object.__setattr__(self, "root", Path(self.root))

    @property
    def manifest_path(self) -> Path:
        """Path of the run-level manifest."""
        return self.root / MANIFEST_NAME

    @property
    def index_path(self) -> Path:
        """Path of the derived sample index."""
        return self.root / INDEX_NAME

    @property
    def samples_dir(self) -> Path:
        """Directory holding all samples."""
        return self.root / "samples"

    def sample_dir(self, sample_id: str) -> Path:
        """Directory of one sample."""
        return self.samples_dir / _check_component(sample_id, "Sample id")

    def sample_record(self, sample_id: str) -> Path:
        """Provenance record of one sample."""
        return self.sample_dir(sample_id) / "sample.json"

    def ground_truth(self, sample_id: str) -> Path:
        """Ground-truth label of one sample."""
        return self.sample_dir(sample_id) / "ground_truth.json"

    def dialogue_jsonl(self, sample_id: str) -> Path:
        """Structured utterances of one sample."""
        return self.sample_dir(sample_id) / "dialogue.jsonl"

    def dialogue_txt(self, sample_id: str) -> Path:
        """Readable script of one sample."""
        return self.sample_dir(sample_id) / "dialogue.txt"

    def transcript(self, sample_id: str) -> Path:
        """Spoken words of one sample, without speaker labels."""
        return self.sample_dir(sample_id) / "transcript.txt"

    def dialogue_raw(self, sample_id: str) -> Path:
        """Unmodified model output for one sample.

        Written only when it differs from the canonical script -- that is, when
        something had to be discarded or normalised during parsing. Its absence is
        itself information: the model's output was already clean.
        """
        return self.sample_dir(sample_id) / "dialogue.raw.txt"

    def prompt(self, sample_id: str, stage: str) -> Path:
        """Prompt used for ``stage`` of one sample."""
        return self.sample_dir(sample_id) / "prompts" / f"{_check_component(stage, 'Stage')}.txt"

    def audio_dir(self, sample_id: str) -> Path:
        """Audio directory of one sample."""
        return self.sample_dir(sample_id) / "audio"

    def segment(self, sample_id: str, voice_id: str, turn_index: int) -> Path:
        """Per-turn render for one voice.

        Turn indices are zero-padded to four digits so that a plain directory listing
        sorts in dialogue order.
        """
        return (
            self.audio_dir(sample_id)
            / "segments"
            / _check_component(voice_id, "Voice id")
            / f"{turn_index:04d}.wav"
        )

    def segments_dir(self, sample_id: str, voice_id: str) -> Path:
        """Directory of per-turn renders for one voice."""
        return self.audio_dir(sample_id) / "segments" / _check_component(voice_id, "Voice id")

    def take_audio(self, sample_id: str, take_id: str) -> Path:
        """Mixed recording for one take."""
        return self.audio_dir(sample_id) / "takes" / f"{_check_component(take_id, 'Take id')}.wav"

    def take_record(self, sample_id: str, take_id: str) -> Path:
        """Metadata for one take."""
        return self.audio_dir(sample_id) / "takes" / f"{_check_component(take_id, 'Take id')}.json"

    def relative(self, path: Path) -> str:
        """Express ``path`` relative to the dataset root, using forward slashes.

        Stored paths are relative so a dataset stays valid after being moved or
        published, and POSIX-style so a manifest written on one platform reads on
        another.
        """
        return path.relative_to(self.root).as_posix()
