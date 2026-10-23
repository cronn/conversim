"""Writing a dataset to disk.

Two properties drive the implementation. **Atomic writes**: every file lands via a
temporary file and a rename, so an interrupted run leaves either a complete file or
none -- a half-written ``sample.json`` would otherwise make the next resume treat a
broken sample as finished. **A derived index**: ``samples.jsonl`` is rebuilt from the
per-sample records rather than appended to, which keeps concurrent writers from
interleaving lines and makes the index safe to regenerate at any time.
"""

from __future__ import annotations

import json
import logging
import os
import tempfile
from pathlib import Path

from pydantic import BaseModel

from ..models import Dialogue
from ..speech.base import Audio
from .layout import DatasetLayout
from .records import (
    DatasetManifest,
    SampleIndexEntry,
    SampleRecord,
    TakeRecord,
    utc_now,
)

__all__ = ["DatasetWriter"]

logger = logging.getLogger(__name__)


class DatasetWriter:
    """Creates and updates a dataset on disk.

    Args:
        root: Dataset root directory. Created if absent.

    Example:
        >>> writer = DatasetWriter(Path("out/my-dataset"))     # doctest: +SKIP
        >>> writer.write_sample(record, ground_truth=gt, dialogue=dialogue,
        ...                     prompts={"ground_truth": p1, "dialogue": p2})  # doctest: +SKIP
        >>> writer.rebuild_index()                            # doctest: +SKIP
    """

    def __init__(self, root: Path | str) -> None:
        self.layout = DatasetLayout(Path(root))
        self.layout.root.mkdir(parents=True, exist_ok=True)
        self.layout.samples_dir.mkdir(parents=True, exist_ok=True)

    @property
    def root(self) -> Path:
        """The dataset root directory."""
        return self.layout.root

    # -- samples ------------------------------------------------------------

    def has_sample(self, sample_id: str) -> bool:
        """Whether a complete sample record already exists.

        This is the resume predicate: a sample whose record is present is treated as
        done and skipped.
        """
        return self.layout.sample_record(sample_id).exists()

    def sample_ids(self) -> tuple[str, ...]:
        """Ids of every sample with a complete record, sorted."""
        if not self.layout.samples_dir.is_dir():
            return ()
        return tuple(
            sorted(
                directory.name
                for directory in self.layout.samples_dir.iterdir()
                if directory.is_dir() and self.has_sample(directory.name)
            )
        )

    def read_sample_record(self, sample_id: str) -> SampleRecord | None:
        """Load an existing sample record, or ``None`` if it is absent or unreadable."""
        path = self.layout.sample_record(sample_id)
        if not path.exists():
            return None
        try:
            return SampleRecord.model_validate_json(path.read_text(encoding="utf-8"))
        except Exception as error:
            logger.warning("Ignoring unreadable sample record %s: %s", path, error)
            return None

    def write_sample(
        self,
        record: SampleRecord,
        *,
        ground_truth: BaseModel,
        dialogue: Dialogue,
        prompts: dict[str, str],
    ) -> SampleRecord:
        """Write every text artefact of one sample.

        The sample record is written *last*, so a crash mid-write cannot leave a
        sample that resume would consider complete but whose contents are missing.

        Args:
            record: Provenance for the sample.
            ground_truth: The validated schema instance.
            dialogue: Parsed dialogue.
            prompts: Prompt text per stage name.

        Returns:
            The record as written.
        """
        directory = self.layout.sample_dir(record.id)
        directory.mkdir(parents=True, exist_ok=True)

        _write_text(
            self.layout.ground_truth(record.id),
            json.dumps(
                ground_truth.model_dump(mode="json"), indent=2, ensure_ascii=False, default=str
            ),
        )

        _write_text(
            self.layout.dialogue_jsonl(record.id),
            "\n".join(
                json.dumps(utterance.model_dump(mode="json"), ensure_ascii=False)
                for utterance in dialogue.utterances
            ),
        )

        script = dialogue.to_script()
        _write_text(self.layout.dialogue_txt(record.id), script)
        _write_text(self.layout.transcript(record.id), dialogue.transcript())

        # Only keep the raw output when parsing actually changed something.
        if dialogue.raw.strip() and dialogue.raw.strip() != script.strip():
            _write_text(self.layout.dialogue_raw(record.id), dialogue.raw)

        for stage, prompt in prompts.items():
            _write_text(self.layout.prompt(record.id, stage), prompt)

        _write_text(self.layout.sample_record(record.id), record.model_dump_json(indent=2))
        logger.debug("Wrote sample %s (%d turns)", record.id, record.turn_count)
        return record

    def set_take_ids(self, sample_id: str, take_ids: tuple[str, ...]) -> SampleRecord | None:
        """Record which takes exist for a sample, after audio has been rendered."""
        record = self.read_sample_record(sample_id)
        if record is None:
            logger.warning("Cannot record takes for unknown sample %s", sample_id)
            return None
        updated = record.model_copy(update={"take_ids": tuple(take_ids)})
        _write_text(self.layout.sample_record(sample_id), updated.model_dump_json(indent=2))
        return updated

    # -- audio --------------------------------------------------------------

    def segment_path(self, sample_id: str, voice_id: str, turn_index: int) -> Path:
        """Where a per-turn render for one voice belongs."""
        return self.layout.segment(sample_id, voice_id, turn_index)

    def has_segment(self, sample_id: str, voice_id: str, turn_index: int) -> bool:
        """Whether a non-empty per-turn render already exists.

        Zero-byte files are treated as absent: an interrupted write should be redone,
        not silently mixed in as silence.
        """
        path = self.segment_path(sample_id, voice_id, turn_index)
        return path.exists() and path.stat().st_size > 0

    def write_segment(self, sample_id: str, voice_id: str, turn_index: int, audio: Audio) -> Path:
        """Write a per-turn render."""
        path = self.segment_path(sample_id, voice_id, turn_index)
        path.parent.mkdir(parents=True, exist_ok=True)
        audio.write(path)
        return path

    def write_take(self, record: TakeRecord, audio: Audio) -> TakeRecord:
        """Write a mixed recording and its metadata."""
        audio_path = self.layout.take_audio(record.sample_id, record.id)
        audio_path.parent.mkdir(parents=True, exist_ok=True)
        audio.write(audio_path)
        _write_text(
            self.layout.take_record(record.sample_id, record.id),
            record.model_dump_json(indent=2),
        )
        logger.debug("Wrote take %s/%s (%.1fs)", record.sample_id, record.id, record.duration_s)
        return record

    def read_take_records(self, sample_id: str) -> tuple[TakeRecord, ...]:
        """Load every take record for a sample, skipping unreadable ones."""
        takes_dir = self.layout.audio_dir(sample_id) / "takes"
        if not takes_dir.is_dir():
            return ()
        records = []
        for path in sorted(takes_dir.glob("*.json")):
            try:
                records.append(TakeRecord.model_validate_json(path.read_text(encoding="utf-8")))
            except Exception as error:
                logger.warning("Ignoring unreadable take record %s: %s", path, error)
        return tuple(records)

    # -- dataset-level ------------------------------------------------------

    def rebuild_index(self) -> int:
        """Regenerate ``samples.jsonl`` from the per-sample records.

        Returns:
            The number of samples indexed.
        """
        entries: list[SampleIndexEntry] = []
        for directory in sorted(self.layout.samples_dir.iterdir()):
            if not directory.is_dir():
                continue
            record = self.read_sample_record(directory.name)
            if record is None:
                continue
            takes = self.read_take_records(record.id)
            entries.append(
                SampleIndexEntry(
                    id=record.id,
                    path=self.layout.relative(directory),
                    turn_count=record.turn_count,
                    word_count=record.word_count,
                    speakers=record.cast.names,
                    variation=record.variation.summary(),
                    issue_kinds=tuple(sorted({issue.kind for issue in record.issues})),
                    is_clean=record.is_clean,
                    take_ids=tuple(take.id for take in takes),
                    audio_duration_s=(sum(take.duration_s for take in takes) if takes else None),
                )
            )

        _write_text(
            self.layout.index_path,
            "\n".join(entry.model_dump_json() for entry in entries),
        )
        logger.info("Indexed %d samples in %s", len(entries), self.layout.index_path)
        return len(entries)

    def write_manifest(self, manifest: DatasetManifest) -> DatasetManifest:
        """Write the run-level manifest, preserving the original creation time."""
        existing = self.read_manifest()
        if existing is not None:
            manifest = manifest.model_copy(
                update={"created_at": existing.created_at, "updated_at": utc_now()}
            )
        _write_text(self.layout.manifest_path, manifest.model_dump_json(indent=2))
        return manifest

    def read_manifest(self) -> DatasetManifest | None:
        """Load the existing manifest, or ``None`` if absent or unreadable."""
        path = self.layout.manifest_path
        if not path.exists():
            return None
        try:
            return DatasetManifest.model_validate_json(path.read_text(encoding="utf-8"))
        except Exception as error:
            logger.warning("Ignoring unreadable manifest %s: %s", path, error)
            return None

    def audio_totals(self) -> tuple[int, float]:
        """Count rendered takes and their total duration in seconds."""
        count = 0
        duration = 0.0
        for directory in sorted(self.layout.samples_dir.iterdir()):
            if not directory.is_dir():
                continue
            for take in self.read_take_records(directory.name):
                count += 1
                duration += take.duration_s
        return count, duration


def _write_text(path: Path, content: str) -> Path:
    """Write ``content`` to ``path`` atomically, adding a trailing newline."""
    path.parent.mkdir(parents=True, exist_ok=True)
    body = content if content.endswith("\n") else f"{content}\n"
    # delete=False and manual cleanup, because the file has to outlive the handle in
    # order to be renamed into place; a plain context manager would remove it first.
    handle = tempfile.NamedTemporaryFile(  # noqa: SIM115
        "w", encoding="utf-8", dir=path.parent, prefix=f".{path.name}.", delete=False
    )
    try:
        with handle:
            handle.write(body)
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(handle.name, path)
    except BaseException:
        Path(handle.name).unlink(missing_ok=True)
        raise
    return path
