"""Mixing: timeline correctness, stereo placement, level control."""

from __future__ import annotations

import math
import random

import numpy as np
import pytest

from conversim.errors import ConfigurationError
from conversim.speech.base import Audio
from conversim.speech.mixing import (
    BackchannelConfig,
    BackgroundConfig,
    ConversationMixer,
    MixConfig,
    MixTurn,
)


def tone(seconds: float, sample_rate: int = 16_000, frequency: float = 220.0) -> Audio:
    time = np.arange(int(seconds * sample_rate), dtype=np.float32) / sample_rate
    return Audio((0.5 * np.sin(2 * math.pi * frequency * time)).astype(np.float32), sample_rate)


def turns(*durations: float, sample_rate: int = 16_000) -> list[MixTurn]:
    return [
        MixTurn(
            index=index,
            role="agent" if index % 2 == 0 else "customer",
            speaker="Sam" if index % 2 == 0 else "Kim",
            voice_id="anna" if index % 2 == 0 else "ben",
            audio=tone(duration, sample_rate),
            voice_tags=("female",) if index % 2 == 0 else ("male",),
        )
        for index, duration in enumerate(durations)
    ]


def deterministic_config(**kwargs: object) -> MixConfig:
    defaults: dict[str, object] = {
        "pause_mean_s": 0.5,
        "pause_sd_s": 0.0,
        "gain_jitter_db": (0.0, 0.0),
    }
    defaults.update(kwargs)
    return MixConfig(**defaults)  # type: ignore[arg-type]


def test_empty_conversation_is_rejected() -> None:
    with pytest.raises(ConfigurationError, match="empty"):
        ConversationMixer().mix([], rng=random.Random(0))


def test_timeline_matches_durations_and_pauses() -> None:
    mixer = ConversationMixer(deterministic_config())
    result = mixer.mix(turns(1.0, 2.0, 0.5), rng=random.Random(0))

    starts = [round(timing.start_s, 3) for timing in result.timings]
    ends = [round(timing.end_s, 3) for timing in result.timings]
    assert starts == [0.0, 1.5, 4.0]  # 1.0 + 0.5 pause; then 2.0 + 0.5
    assert ends == [1.0, 3.5, 4.5]


def test_timings_carry_speaker_and_voice() -> None:
    result = ConversationMixer(deterministic_config()).mix(turns(1.0, 1.0), rng=random.Random(0))
    assert [(t.index, t.speaker, t.voice_id) for t in result.timings] == [
        (0, "Sam", "anna"),
        (1, "Kim", "ben"),
    ]


def test_recording_is_at_least_as_long_as_the_last_turn() -> None:
    result = ConversationMixer(deterministic_config()).mix(turns(1.0, 1.0), rng=random.Random(0))
    assert result.audio.duration_s >= result.timings[-1].end_s


def test_output_is_stereo_by_default() -> None:
    result = ConversationMixer(deterministic_config()).mix(turns(0.5), rng=random.Random(0))
    assert result.audio.channels == 2


def test_mono_output_when_requested() -> None:
    result = ConversationMixer(deterministic_config(stereo=False)).mix(
        turns(0.5), rng=random.Random(0)
    )
    assert result.audio.channels == 1


def test_speakers_land_at_different_stereo_positions() -> None:
    result = ConversationMixer(deterministic_config(pan_width=0.9)).mix(
        turns(1.0, 1.0), rng=random.Random(2)
    )
    assert result.pan_positions["agent"] != result.pan_positions["customer"]
    assert all(abs(position) <= 0.9 for position in result.pan_positions.values())


def test_a_speaker_keeps_one_position_across_turns() -> None:
    result = ConversationMixer(deterministic_config(pan_width=0.9)).mix(
        turns(1.0, 1.0, 1.0, 1.0), rng=random.Random(2)
    )
    assert set(result.pan_positions) == {"agent", "customer"}


def test_peak_is_normalised_to_the_target() -> None:
    config = deterministic_config(target_peak_dbfs=-3.0)
    result = ConversationMixer(config).mix(turns(1.0, 1.0), rng=random.Random(0))
    expected = 10.0 ** (-3.0 / 20.0)
    assert result.audio.peak() == pytest.approx(expected, abs=1e-3)


def test_overlap_is_only_possible_when_enabled() -> None:
    no_overlap = ConversationMixer(
        MixConfig(pause_mean_s=0.0, pause_sd_s=2.0, allow_overlap=False)
    ).mix(turns(1.0, 1.0, 1.0), rng=random.Random(1))
    for previous, following in zip(no_overlap.timings, no_overlap.timings[1:], strict=False):
        assert following.start_s >= previous.end_s - 1e-9

    overlapping = ConversationMixer(
        MixConfig(pause_mean_s=-0.2, pause_sd_s=0.05, allow_overlap=True, min_pause_s=-0.3)
    ).mix(turns(1.0, 1.0, 1.0), rng=random.Random(1))
    assert any(
        following.start_s < previous.end_s
        for previous, following in zip(overlapping.timings, overlapping.timings[1:], strict=False)
    )


def test_mixing_is_reproducible() -> None:
    mixer = ConversationMixer(MixConfig(pause_sd_s=0.5, pan_width=0.5))
    first = mixer.mix(turns(1.0, 1.0, 1.0), rng=random.Random(9))
    second = mixer.mix(turns(1.0, 1.0, 1.0), rng=random.Random(9))
    assert first.timings == second.timings
    assert first.pan_positions == second.pan_positions
    np.testing.assert_allclose(first.audio.samples, second.audio.samples)


def test_differing_sample_rates_are_resampled() -> None:
    mixed = [
        MixTurn(index=0, role="agent", speaker="Sam", voice_id="anna", audio=tone(1.0, 16_000)),
        MixTurn(index=1, role="customer", speaker="Kim", voice_id="ben", audio=tone(1.0, 24_000)),
    ]
    result = ConversationMixer(deterministic_config()).mix(mixed, rng=random.Random(0))
    assert result.audio.sample_rate == 16_000
    assert result.timings[1].duration_s == pytest.approx(1.0, abs=0.01)


def test_forced_sample_rate_is_honoured() -> None:
    result = ConversationMixer(deterministic_config(sample_rate=8_000)).mix(
        turns(1.0), rng=random.Random(0)
    )
    assert result.audio.sample_rate == 8_000


def test_backchannels_are_injected_into_long_turns(tmp_path) -> None:
    pool = tmp_path / "backchannels" / "male"
    pool.mkdir(parents=True)
    tone(0.3).write(pool / "mhm.wav")

    config = deterministic_config(
        backchannels=BackchannelConfig(
            directory=tmp_path / "backchannels",
            probability=1.0,
            min_turn_duration_s=1.0,
            double_probability=0.0,
        )
    )
    result = ConversationMixer(config).mix(turns(5.0, 5.0), rng=random.Random(0))
    assert result.backchannel_count >= 1


def test_short_turns_get_no_backchannels(tmp_path) -> None:
    pool = tmp_path / "backchannels"
    pool.mkdir()
    tone(0.3).write(pool / "mhm.wav")

    config = deterministic_config(
        backchannels=BackchannelConfig(directory=pool, probability=1.0, min_turn_duration_s=10.0)
    )
    result = ConversationMixer(config).mix(turns(1.0, 1.0), rng=random.Random(0))
    assert result.backchannel_count == 0


def test_a_missing_backchannel_directory_is_survivable(tmp_path) -> None:
    config = deterministic_config(
        backchannels=BackchannelConfig(directory=tmp_path / "absent", probability=1.0)
    )
    result = ConversationMixer(config).mix(turns(5.0), rng=random.Random(0))
    assert result.backchannel_count == 0


def test_background_is_laid_under_the_whole_recording(tmp_path) -> None:
    bed = tmp_path / "room.wav"
    tone(1.0, frequency=60.0).write(bed)

    config = deterministic_config(background=BackgroundConfig(path=bed, gain_db=-20.0, tail_s=1.0))
    result = ConversationMixer(config).mix(turns(3.0), rng=random.Random(0))
    # Room tone continues past the final turn, so the tail is not pure silence.
    tail = result.audio.samples[int(3.5 * result.audio.sample_rate) :]
    assert float(np.abs(tail).max()) > 0.0


def test_a_missing_background_falls_back_to_silence(tmp_path) -> None:
    config = deterministic_config(background=BackgroundConfig(path=tmp_path / "absent.wav"))
    result = ConversationMixer(config).mix(turns(1.0), rng=random.Random(0))
    assert result.audio.frames > 0


def test_audio_round_trips_through_a_file(tmp_path) -> None:
    from conversim.speech.base import load_audio

    original = tone(0.5)
    path = original.write(tmp_path / "nested" / "tone.wav")
    reloaded = load_audio(path)
    assert reloaded.sample_rate == original.sample_rate
    assert reloaded.frames == original.frames
    np.testing.assert_allclose(reloaded.samples, original.samples, atol=1e-4)


def test_invalid_target_peak_is_rejected() -> None:
    with pytest.raises(ValueError, match="dBFS"):
        MixConfig(target_peak_dbfs=6.0)


def test_inverted_gain_range_is_rejected() -> None:
    with pytest.raises(ValueError, match="low, high"):
        MixConfig(gain_jitter_db=(2.0, -2.0))
