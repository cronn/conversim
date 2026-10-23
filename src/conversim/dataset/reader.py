"""Reading a generated dataset back.

The reader exists so that consuming a conversim dataset does not mean re-implementing
its layout. It also deliberately does *not* need the code that produced the dataset:
ground truth comes back as plain JSON, and validating it into a schema class is an
opt-in step for callers who still have that class.
"""

from __future__ import annotations

import json
import logging
from collections.abc import Iterator
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from pydantic import BaseModel

from ..errors import ConfigurationError
from ..models import Dialogue, Utterance
from .layout import DatasetLayout
from .records import DatasetManifest, SampleIndexEntry, SampleRecord, TakeRecord

__all__ = ["DatasetReader", "LoadedSample", "load_dataset"]

logger = logging.getLogger(__name__)


@dataclass(frozen=True)
class LoadedSample:
    """One sample read back from disk.

    Attributes:
        record: Provenance, including scenario, cast, variation and models.
        ground_truth: The extraction label, as parsed JSON.
        dialogue: Parsed utterances. ``dialogue.raw`` holds the original model output
            when it was kept, otherwise the canonical script.
        takes: Rendered recordings, empty for a text-only sample.
        root: Dataset root, for resolving the relative paths in ``takes``.
    """

    record: SampleRecord
    ground_truth: dict[str, Any]
    dialogue: Dialogue
    takes: tuple[TakeRecord, ...]
    root: Path

    @property
    def id(self) -> str:
        """The sample identifier."""
        return self.record.id

    def ground_truth_as(self, model: type[BaseModel]) -> BaseModel:
        """Validate the ground truth into ``model``.

        Useful for round-tripping a dataset through the schema it was generated
        against; it will also surface any drift if that schema has since changed.
        """
        return model.model_validate(self.ground_truth)

    def audio_path(self, take_id: str) -> Path:
        """Absolute path of a take's recording.

        Raises:
            ConfigurationError: if the sample has no such take.
        """
        for take in self.takes:
            if take.id == take_id:
                return self.root / take.audio_path
        available = ", ".join(take.id for take in self.takes) or "none"
        raise ConfigurationError(
            f"Sample {self.id!r} has no take {take_id!r}; available: {available}"
        )


class DatasetReader:
    """Reads samples, takes and metadata from a dataset root.

    Args:
        root: Dataset root directory.

    Raises:
        ConfigurationError: if the directory does not look like a dataset.

    Example:
        >>> dataset = load_dataset("out/my-dataset")            # doctest: +SKIP
        >>> len(dataset)                                        # doctest: +SKIP
        400
        >>> clean = [s for s in dataset if s.record.is_clean]   # doctest: +SKIP
    """

    def __init__(self, root: Path | str) -> None:
        self.layout = DatasetLayout(Path(root))
        if not self.layout.samples_dir.is_dir():
            raise ConfigurationError(
                f"{self.layout.root} does not look like a conversim dataset "
                f"(no 'samples/' directory)"
            )

    @property
    def root(self) -> Path:
        """The dataset root directory."""
        return self.layout.root

    def __len__(self) -> int:
        return len(self.sample_ids())

    def __iter__(self) -> Iterator[LoadedSample]:
        return self.iter_samples()

    def manifest(self) -> DatasetManifest | None:
        """The run-level manifest, or ``None`` if absent."""
        path = self.layout.manifest_path
        if not path.exists():
            return None
        return DatasetManifest.model_validate_json(path.read_text(encoding="utf-8"))

    def index(self) -> tuple[SampleIndexEntry, ...]:
        """The sample index.

        Falls back to scanning ``samples/`` when ``samples.jsonl`` is missing, so a
        partially written dataset is still readable.
        """
        path = self.layout.index_path
        if not path.exists():
            logger.debug("No index at %s; scanning sample directories", path)
            return tuple(
                SampleIndexEntry(
                    id=record.id,
                    path=self.layout.relative(self.layout.sample_dir(record.id)),
                    turn_count=record.turn_count,
                    word_count=record.word_count,
                    speakers=record.cast.names,
                    variation=record.variation.summary(),
                    issue_kinds=tuple(sorted({issue.kind for issue in record.issues})),
                    is_clean=record.is_clean,
                    take_ids=record.take_ids,
                )
                for record in self._scan_records()
            )

        entries = []
        for line in path.read_text(encoding="utf-8").splitlines():
            if line.strip():
                entries.append(SampleIndexEntry.model_validate_json(line))
        return tuple(entries)

    def sample_ids(self) -> tuple[str, ...]:
        """Ids of all samples, sorted."""
        return tuple(
            sorted(
                directory.name
                for directory in self.layout.samples_dir.iterdir()
                if directory.is_dir() and self.layout.sample_record(directory.name).exists()
            )
        )

    def load(self, sample_id: str) -> LoadedSample:
        """Read one sample in full.

        Raises:
            ConfigurationError: if the sample does not exist.
        """
        record_path = self.layout.sample_record(sample_id)
        if not record_path.exists():
            raise ConfigurationError(f"No sample {sample_id!r} in {self.root}")

        record = SampleRecord.model_validate_json(record_path.read_text(encoding="utf-8"))
        ground_truth = json.loads(self.layout.ground_truth(sample_id).read_text(encoding="utf-8"))
        return LoadedSample(
            record=record,
            ground_truth=ground_truth,
            dialogue=self._load_dialogue(sample_id, record),
            takes=self._load_takes(sample_id),
            root=self.root,
        )

    def iter_samples(self) -> Iterator[LoadedSample]:
        """Iterate every sample in id order."""
        for sample_id in self.sample_ids():
            yield self.load(sample_id)

    def transcripts(self) -> Iterator[tuple[str, str]]:
        """Iterate ``(sample_id, transcript)`` pairs.

        The cheap path for ASR work: reads one small text file per sample instead of
        reconstructing dialogues.
        """
        for sample_id in self.sample_ids():
            path = self.layout.transcript(sample_id)
            if path.exists():
                yield sample_id, path.read_text(encoding="utf-8").strip()

    def _scan_records(self) -> Iterator[SampleRecord]:
        for sample_id in self.sample_ids():
            path = self.layout.sample_record(sample_id)
            try:
                yield SampleRecord.model_validate_json(path.read_text(encoding="utf-8"))
            except Exception as error:
                logger.warning("Skipping unreadable sample record %s: %s", path, error)

    def _load_dialogue(self, sample_id: str, record: SampleRecord) -> Dialogue:
        utterances = []
        jsonl_path = self.layout.dialogue_jsonl(sample_id)
        if jsonl_path.exists():
            for line in jsonl_path.read_text(encoding="utf-8").splitlines():
                if line.strip():
                    utterances.append(Utterance.model_validate_json(line))

        raw_path = self.layout.dialogue_raw(sample_id)
        script_path = self.layout.dialogue_txt(sample_id)
        if raw_path.exists():
            raw = raw_path.read_text(encoding="utf-8")
        elif script_path.exists():
            raw = script_path.read_text(encoding="utf-8")
        else:
            raw = ""

        return Dialogue(utterances=tuple(utterances), raw=raw, issues=record.issues)

    def _load_takes(self, sample_id: str) -> tuple[TakeRecord, ...]:
        takes_dir = self.layout.audio_dir(sample_id) / "takes"
        if not takes_dir.is_dir():
            return ()
        takes = []
        for path in sorted(takes_dir.glob("*.json")):
            try:
                takes.append(TakeRecord.model_validate_json(path.read_text(encoding="utf-8")))
            except Exception as error:
                logger.warning("Skipping unreadable take record %s: %s", path, error)
        return tuple(takes)


def load_dataset(root: Path | str) -> DatasetReader:
    """Open a generated dataset for reading.

    Args:
        root: Dataset root directory.

    Returns:
        A reader over the dataset.
    """
    return DatasetReader(root)
