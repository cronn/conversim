"""Dataset layout, writing and reading."""

from __future__ import annotations

import pytest

from conversim import (
    Cast,
    DatasetReader,
    DatasetWriter,
    Dialogue,
    DialogueIssue,
    SampleRecord,
    Scenario,
    TurnTarget,
    Utterance,
    load_dataset,
)
from conversim.dataset.layout import DatasetLayout
from conversim.dataset.records import DatasetManifest
from conversim.errors import ConfigurationError
from conversim.variation import VariationConfig

from .conftest import AGENT, CUSTOMER, KIM, SAM, Ticket


def make_record(sample_id: str = "s-0000", **overrides: object) -> SampleRecord:
    defaults: dict[str, object] = dict(
        id=sample_id,
        schema_name="Ticket",
        seed=12345,
        scenario=Scenario(
            topic="a support call",
            context="Helpdesk.",
            language="English",
            turns=TurnTarget(minimum=2, maximum=4),
        ),
        roles=(AGENT, CUSTOMER),
        cast=Cast(personas=(SAM, KIM)),
        variation=VariationConfig(),
        turn_count=2,
        word_count=4,
        models={"ground_truth": "stub", "dialogue": "stub"},
    )
    defaults.update(overrides)
    return SampleRecord(**defaults)  # type: ignore[arg-type]


def make_dialogue() -> Dialogue:
    return Dialogue(
        utterances=(
            Utterance(index=0, speaker="Sam", role="agent", text="Hello there.", cue="warmly"),
            Utterance(index=1, speaker="Kim", role="customer", text="Hi back."),
        ),
        raw="Sam: (warmly) Hello there.\nKim: Hi back.",
    )


def test_round_trip(tmp_path) -> None:
    writer = DatasetWriter(tmp_path / "ds")
    record = make_record()
    writer.write_sample(
        record,
        ground_truth=Ticket.model_validate({"reference": "R-1"}),
        dialogue=make_dialogue(),
        prompts={"ground_truth": "gt prompt", "dialogue": "dlg prompt"},
    )
    writer.rebuild_index()
    writer.write_manifest(DatasetManifest(schema_name="Ticket"))

    dataset = load_dataset(tmp_path / "ds")
    assert dataset.sample_ids() == ("s-0000",)

    sample = dataset.load("s-0000")
    assert sample.record.seed == 12345
    assert sample.ground_truth["reference"] == "R-1"
    assert sample.dialogue.transcript() == "Hello there. Hi back."
    assert sample.dialogue.utterances[0].cue == "warmly"
    assert sample.ground_truth_as(Ticket).reference == "R-1"


def test_transcript_file_matches_the_dialogue(tmp_path) -> None:
    writer = DatasetWriter(tmp_path / "ds")
    writer.write_sample(make_record(), ground_truth=Ticket(), dialogue=make_dialogue(), prompts={})
    transcripts = dict(DatasetReader(tmp_path / "ds").transcripts())
    assert transcripts["s-0000"] == "Hello there. Hi back."


def test_index_carries_filterable_fields(tmp_path) -> None:
    writer = DatasetWriter(tmp_path / "ds")
    writer.write_sample(
        make_record(issues=(DialogueIssue(kind="turn_count", detail="off by 3"),)),
        ground_truth=Ticket(),
        dialogue=make_dialogue(),
        prompts={},
    )
    writer.rebuild_index()

    entry = load_dataset(tmp_path / "ds").index()[0]
    assert entry.id == "s-0000"
    assert entry.speakers == ("Sam", "Kim")
    assert entry.issue_kinds == ("turn_count",)
    assert not entry.has_audio


def test_index_is_regenerable(tmp_path) -> None:
    writer = DatasetWriter(tmp_path / "ds")
    for index in range(3):
        writer.write_sample(
            make_record(f"s-{index:04d}"),
            ground_truth=Ticket(),
            dialogue=make_dialogue(),
            prompts={},
        )
    assert writer.rebuild_index() == 3
    writer.layout.index_path.unlink()
    assert len(load_dataset(tmp_path / "ds").index()) == 3  # falls back to scanning
    assert writer.rebuild_index() == 3


def test_resume_predicate(tmp_path) -> None:
    writer = DatasetWriter(tmp_path / "ds")
    assert not writer.has_sample("s-0000")
    writer.write_sample(make_record(), ground_truth=Ticket(), dialogue=make_dialogue(), prompts={})
    assert writer.has_sample("s-0000")


def test_manifest_preserves_its_creation_time(tmp_path) -> None:
    writer = DatasetWriter(tmp_path / "ds")
    first = writer.write_manifest(DatasetManifest(schema_name="Ticket", sample_count=1))
    second = writer.write_manifest(DatasetManifest(schema_name="Ticket", sample_count=2))
    assert second.created_at == first.created_at
    assert second.sample_count == 2


def test_writes_are_atomic(tmp_path) -> None:
    """A failed write must not leave a partial file that resume would trust."""
    writer = DatasetWriter(tmp_path / "ds")
    writer.write_sample(make_record(), ground_truth=Ticket(), dialogue=make_dialogue(), prompts={})
    leftovers = list(writer.layout.sample_dir("s-0000").glob(".*"))
    assert leftovers == []


def test_unreadable_records_are_skipped_not_fatal(tmp_path) -> None:
    writer = DatasetWriter(tmp_path / "ds")
    writer.write_sample(make_record(), ground_truth=Ticket(), dialogue=make_dialogue(), prompts={})
    writer.layout.sample_record("s-0000").write_text("{ this is not json")
    assert writer.read_sample_record("s-0000") is None
    assert writer.rebuild_index() == 0


def test_loading_an_absent_sample_is_an_error(tmp_path) -> None:
    DatasetWriter(tmp_path / "ds")
    with pytest.raises(ConfigurationError, match="No sample"):
        load_dataset(tmp_path / "ds").load("nope")


def test_a_non_dataset_directory_is_rejected(tmp_path) -> None:
    (tmp_path / "random").mkdir()
    with pytest.raises(ConfigurationError, match="does not look like"):
        load_dataset(tmp_path / "random")


def test_missing_take_is_an_informative_error(tmp_path) -> None:
    writer = DatasetWriter(tmp_path / "ds")
    writer.write_sample(make_record(), ground_truth=Ticket(), dialogue=make_dialogue(), prompts={})
    sample = load_dataset(tmp_path / "ds").load("s-0000")
    with pytest.raises(ConfigurationError, match="none"):
        sample.audio_path("absent")


def test_paths_are_relative_and_posix(tmp_path) -> None:
    layout = DatasetLayout(tmp_path / "ds")
    assert layout.relative(layout.segment("s-0000", "anna", 7)) == (
        "samples/s-0000/audio/segments/anna/0007.wav"
    )


@pytest.mark.parametrize("bad", ["../escape", "with/slash", "", ".hidden", "with space"])
def test_unsafe_path_components_are_rejected(tmp_path, bad: str) -> None:
    layout = DatasetLayout(tmp_path / "ds")
    with pytest.raises(ConfigurationError):
        layout.sample_dir(bad)


def test_segment_indices_are_zero_padded(tmp_path) -> None:
    layout = DatasetLayout(tmp_path / "ds")
    assert layout.segment("s", "v", 3).name == "0003.wav"
    assert layout.segment("s", "v", 1234).name == "1234.wav"


def test_zero_byte_segments_count_as_absent(tmp_path) -> None:
    writer = DatasetWriter(tmp_path / "ds")
    path = writer.segment_path("s-0000", "anna", 0)
    path.parent.mkdir(parents=True)
    path.touch()
    assert not writer.has_segment("s-0000", "anna", 0)
