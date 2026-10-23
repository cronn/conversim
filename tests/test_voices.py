"""Voice libraries and assignment strategies."""

from __future__ import annotations

import random

import numpy as np
import pytest

from conversim import Cast, Persona
from conversim.errors import ConfigurationError
from conversim.speech.base import Audio, Voice
from conversim.speech.voices import (
    FixedAssignment,
    RandomAssignment,
    SwappedAssignment,
    VoiceAssignment,
    VoiceLibrary,
)

CAST = Cast(personas=(Persona(name="Sam", role="agent"), Persona(name="Kim", role="customer")))


def library(*ids: str) -> VoiceLibrary:
    return VoiceLibrary(voices=tuple(Voice(id=name, preset="alloy") for name in ids))


def test_lookup_and_tags() -> None:
    voices = VoiceLibrary(
        voices=(
            Voice(id="anna", tags=("female",)),
            Voice(id="ben", tags=("male",)),
        )
    )
    assert voices.get("anna").id == "anna"
    assert [v.id for v in voices.tagged("male")] == ["ben"]
    assert len(voices) == 2
    with pytest.raises(ConfigurationError, match="Unknown voice"):
        voices.get("nope")


def test_duplicate_voice_ids_rejected() -> None:
    with pytest.raises(ValueError, match="Duplicate voice ids"):
        library("anna", "anna")


def test_empty_library_rejected() -> None:
    with pytest.raises(ValueError, match="at least one"):
        VoiceLibrary(voices=())


def test_loads_wav_and_transcript_pairs(tmp_path) -> None:
    for name in ("joshua", "pascal"):
        Audio(np.zeros(1600, dtype=np.float32), 16_000).write(tmp_path / f"{name}.wav")
        (tmp_path / f"{name}.txt").write_text("Der flinke braune Fuchs.", encoding="utf-8")

    voices = VoiceLibrary.from_directory(tmp_path, tags={"joshua": ("male",)})
    assert voices.ids == ("joshua", "pascal")
    assert voices.get("joshua").reference_text == "Der flinke braune Fuchs."
    assert voices.get("joshua").tags == ("male",)


def test_audio_without_a_transcript_is_skipped(tmp_path) -> None:
    Audio(np.zeros(1600, dtype=np.float32), 16_000).write(tmp_path / "orphan.wav")
    Audio(np.zeros(1600, dtype=np.float32), 16_000).write(tmp_path / "paired.wav")
    (tmp_path / "paired.txt").write_text("hello", encoding="utf-8")

    assert VoiceLibrary.from_directory(tmp_path).ids == ("paired",)


def test_a_directory_with_no_usable_voices_is_an_error(tmp_path) -> None:
    with pytest.raises(ConfigurationError, match="No usable voices"):
        VoiceLibrary.from_directory(tmp_path)


def test_a_missing_directory_is_an_error(tmp_path) -> None:
    with pytest.raises(ConfigurationError, match="does not exist"):
        VoiceLibrary.from_directory(tmp_path / "absent")


def test_reference_requires_both_halves(tmp_path) -> None:
    path = tmp_path / "v.wav"
    Audio(np.zeros(100, dtype=np.float32), 16_000).write(path)

    with pytest.raises(ConfigurationError, match="no reference_audio"):
        Voice(id="v").require_reference()
    with pytest.raises(ConfigurationError, match="no reference_text"):
        Voice(id="v", reference_audio=path).require_reference()
    with pytest.raises(ConfigurationError, match="does not exist"):
        Voice(id="v", reference_audio=tmp_path / "gone.wav", reference_text="x").require_reference()

    assert Voice(id="v", reference_audio=path, reference_text="x").require_reference()[1] == "x"


def test_fixed_assignment_validates_voice_ids() -> None:
    strategy = FixedAssignment(voices={"agent": "anna", "customer": "nope"})
    with pytest.raises(ConfigurationError, match="Unknown voice"):
        strategy.assign(cast=CAST, library=library("anna", "ben"), rng=random.Random(0))


def test_random_assignment_is_distinct_by_default() -> None:
    takes = RandomAssignment().assign(
        cast=CAST, library=library("anna", "ben", "cleo"), rng=random.Random(0)
    )
    assert len(takes) == 1
    assert len(set(takes[0].voices.values())) == 2


def test_random_assignment_needs_enough_voices() -> None:
    with pytest.raises(ConfigurationError, match="distinct voices"):
        RandomAssignment().assign(cast=CAST, library=library("solo"), rng=random.Random(0))


def test_random_assignment_can_reuse_a_voice_when_asked() -> None:
    takes = RandomAssignment(distinct=False).assign(
        cast=CAST, library=library("solo"), rng=random.Random(0)
    )
    assert set(takes[0].voices.values()) == {"solo"}


def test_random_assignment_honours_tags() -> None:
    voices = VoiceLibrary(
        voices=(
            Voice(id="anna", tags=("de",)),
            Voice(id="ben", tags=("de",)),
            Voice(id="cleo", tags=("en",)),
        )
    )
    takes = RandomAssignment(restrict_to_tags=("de",)).assign(
        cast=CAST, library=voices, rng=random.Random(0)
    )
    assert set(takes[0].voices.values()) <= {"anna", "ben"}


def test_an_unmatched_tag_is_an_error() -> None:
    with pytest.raises(ConfigurationError, match="tags"):
        RandomAssignment(restrict_to_tags=("nope",)).assign(
            cast=CAST, library=library("anna", "ben"), rng=random.Random(0)
        )


def test_swapped_assignment_produces_both_permutations() -> None:
    takes = SwappedAssignment(limit=2).assign(
        cast=CAST, library=library("anna", "ben"), rng=random.Random(0)
    )
    assert len(takes) == 2
    first, second = takes
    assert first.voices["agent"] == second.voices["customer"]
    assert first.voices["customer"] == second.voices["agent"]
    assert first.id != second.id


def test_swapped_assignment_limit_caps_permutations() -> None:
    takes = SwappedAssignment(limit=1).assign(
        cast=CAST, library=library("anna", "ben"), rng=random.Random(0)
    )
    assert len(takes) == 1


def test_assignment_reports_missing_roles() -> None:
    assignment = VoiceAssignment(id="t", voices={"agent": "anna"})
    assert assignment.voice_for("agent") == "anna"
    with pytest.raises(ConfigurationError, match="does not cover role"):
        assignment.voice_for("customer")


def test_an_assignment_covering_nothing_is_rejected() -> None:
    with pytest.raises(ValueError, match="maps no roles"):
        VoiceAssignment(id="t", voices={})
