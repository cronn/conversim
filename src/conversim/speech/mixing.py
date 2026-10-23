"""Assembling utterances into a conversation recording.

Concatenating TTS output back to back sounds like TTS output back to back. The
post-processing here is what makes the result plausible as a room recording: speakers
sit at different points in the stereo field, gaps between turns vary, levels drift, a
listener hums acknowledgement mid-sentence, and room tone runs underneath.

Everything is computed in NumPy and written with ``soundfile``, so no external
decoder binary is needed at any point.

A recording also gets a **timeline**: exact start and end of every turn. Those come
for free during mixing, and without them a downstream diarization or streaming
evaluation has to re-derive alignment with forced alignment or by hand.
"""

from __future__ import annotations

import logging
import math
import random
from dataclasses import dataclass
from pathlib import Path

import numpy as np
from pydantic import BaseModel, ConfigDict, Field, model_validator

from ..errors import ConfigurationError
from .base import Audio, load_audio

__all__ = [
    "BackchannelConfig",
    "BackgroundConfig",
    "ConversationMixer",
    "MixConfig",
    "MixResult",
    "MixTurn",
    "TurnTiming",
]

logger = logging.getLogger(__name__)

_AUDIO_SUFFIXES = (".wav", ".flac", ".ogg", ".mp3")


class BackchannelConfig(BaseModel):
    """Short listener acknowledgements ("mhm") dropped into long turns.

    Args:
        directory: Root of the backchannel assets. Either a flat directory of clips,
            or one subdirectory per voice tag (``male/``, ``female/``) so a listener
            gets an acknowledgement that plausibly belongs to their voice.
        probability: Chance of injecting into any turn long enough to qualify.
        min_turn_duration_s: Turns shorter than this are left alone -- there is no
            room to interject in a two-word answer.
        gain_db: Attenuation range applied to the clip, since a backchannel is
            further from the microphone focus than the speaker.
        position: Fractional window of the turn to place it in. Front-loading sounds
            like an interruption; the default sits it in the back half.
        double_probability: Chance of a second acknowledgement following.
        double_gap_s: Gap before that second clip.
    """

    model_config = ConfigDict(frozen=True)

    directory: Path
    probability: float = Field(default=0.6, ge=0.0, le=1.0)
    min_turn_duration_s: float = Field(default=3.0, ge=0.0)
    gain_db: tuple[float, float] = (-10.0, -6.0)
    position: tuple[float, float] = (0.50, 0.85)
    double_probability: float = Field(default=0.2, ge=0.0, le=1.0)
    double_gap_s: float = Field(default=0.4, ge=0.0)


class BackgroundConfig(BaseModel):
    """Room tone laid under the whole conversation.

    Args:
        path: Audio file to loop. A steady, featureless recording works best; anything
            with events in it will be heard repeating.
        gain_db: Attenuation. Around -15 dB reads as presence rather than noise.
        tail_s: Silence-with-room-tone appended after the last turn, so the recording
            does not end on a hard cut.
    """

    model_config = ConfigDict(frozen=True)

    path: Path
    gain_db: float = -15.0
    tail_s: float = Field(default=1.0, ge=0.0)


class MixConfig(BaseModel):
    """How to assemble utterances into one recording.

    Args:
        pan_width: Maximum absolute stereo offset per speaker, ``0`` (centred) to
            ``1`` (hard left/right). Each speaker keeps a fixed position for the whole
            conversation, as they would in a room.
        pause_mean_s: Mean gap between turns. May be negative, meaning turns
            typically overlap -- which requires ``allow_overlap``.
        pause_sd_s: Standard deviation of that gap. Generous by design: uniform
            pacing is one of the strongest giveaways of synthetic dialogue.
        allow_overlap: Permit negative gaps, so turns can overlap as real speech does.
        min_pause_s: Floor for the sampled gap. Only reached when ``allow_overlap``;
            otherwise the floor is zero.
        gain_jitter_db: Per-turn level variation.
        normalize_turns: Peak-normalise each turn before mixing. Worth keeping on when
            utterances come from a cloning backend, whose level varies with the
            reference recording.
        target_peak_dbfs: Peak the finished mix is normalised to.
        stereo: Produce two channels. Required for panning to have any effect.
        sample_rate: Force a specific rate. By default the first turn's rate wins and
            anything differing is resampled.
        background: Optional room tone.
        backchannels: Optional listener acknowledgements.
    """

    model_config = ConfigDict(frozen=True)

    pan_width: float = Field(default=0.3, ge=0.0, le=1.0)
    pause_mean_s: float = Field(default=0.2)
    pause_sd_s: float = Field(default=0.4, ge=0.0)
    allow_overlap: bool = False
    min_pause_s: float = -0.3
    gain_jitter_db: tuple[float, float] = (-5.0, 1.0)
    normalize_turns: bool = True
    target_peak_dbfs: float = -1.0
    stereo: bool = True
    sample_rate: int | None = None
    background: BackgroundConfig | None = None
    backchannels: BackchannelConfig | None = None

    @model_validator(mode="after")
    def _ranges_ordered(self) -> MixConfig:
        if self.gain_jitter_db[0] > self.gain_jitter_db[1]:
            raise ValueError(f"gain_jitter_db must be (low, high), got {self.gain_jitter_db}")
        if self.target_peak_dbfs > 0:
            raise ValueError(
                f"target_peak_dbfs must be <= 0 (dBFS is relative to full scale), "
                f"got {self.target_peak_dbfs}"
            )
        if self.pause_mean_s < 0 and not self.allow_overlap:
            raise ValueError(
                f"pause_mean_s is negative ({self.pause_mean_s}), which only makes "
                f"sense with allow_overlap=True; otherwise gaps are clamped to zero "
                f"and every turn would simply follow immediately"
            )
        return self


@dataclass(frozen=True)
class MixTurn:
    """One rendered utterance, ready to place on the timeline.

    Attributes:
        index: Position of the utterance in the dialogue.
        role: Role id of the speaker. Drives pan position and listener selection.
        speaker: Persona name, carried through to the timeline for reference.
        voice_id: Voice that rendered this turn.
        audio: The rendered audio.
        voice_tags: Tags of that voice, used to pick matching backchannels.
    """

    index: int
    role: str
    speaker: str
    voice_id: str
    audio: Audio
    voice_tags: tuple[str, ...] = ()


class TurnTiming(BaseModel):
    """Where a turn ended up in the finished recording."""

    model_config = ConfigDict(frozen=True)

    index: int
    role: str
    speaker: str
    voice_id: str
    start_s: float
    end_s: float

    @property
    def duration_s(self) -> float:
        """Length of the turn in seconds."""
        return self.end_s - self.start_s


@dataclass(frozen=True)
class MixResult:
    """A finished recording and its turn timeline.

    Attributes:
        audio: The mixed conversation.
        timings: Per-turn start and end times, in dialogue order.
        pan_positions: Stereo position assigned to each role.
        backchannel_count: How many acknowledgements were injected.
    """

    audio: Audio
    timings: tuple[TurnTiming, ...]
    pan_positions: dict[str, float]
    backchannel_count: int


class ConversationMixer:
    """Assembles rendered utterances into one recording.

    Args:
        config: Mixing parameters.

    Example:
        >>> mixer = ConversationMixer(MixConfig(pause_mean_s=0.3))
        >>> result = mixer.mix(turns, rng=random.Random(0))   # doctest: +SKIP
        >>> result.audio.write(Path("conversation.wav"))       # doctest: +SKIP
    """

    def __init__(self, config: MixConfig | None = None) -> None:
        self.config = config or MixConfig()
        self._backchannels: dict[str, list[Audio]] | None = None

    def mix(self, turns: list[MixTurn], *, rng: random.Random) -> MixResult:
        """Mix ``turns`` into a single recording.

        Raises:
            ConfigurationError: if ``turns`` is empty.
        """
        if not turns:
            raise ConfigurationError("Cannot mix an empty conversation")

        config = self.config
        sample_rate = config.sample_rate or turns[0].audio.sample_rate
        roles = sorted({turn.role for turn in turns})
        pan_positions = {role: rng.uniform(-config.pan_width, config.pan_width) for role in roles}

        placements: list[tuple[int, np.ndarray, float]] = []  # (start frame, mono, pan)
        timings: list[TurnTiming] = []
        backchannel_count = 0
        cursor = 0

        for position, turn in enumerate(turns):
            mono = _resampled_mono(turn.audio, sample_rate)
            if config.normalize_turns:
                mono = _peak_normalize(mono, config.target_peak_dbfs)
            mono = mono * _db_to_gain(rng.uniform(*config.gain_jitter_db))

            if position > 0:
                floor = config.min_pause_s if config.allow_overlap else 0.0
                gap = max(floor, rng.gauss(config.pause_mean_s, config.pause_sd_s))
                cursor = max(0, cursor + round(gap * sample_rate))

            pan = pan_positions[turn.role]
            placements.append((cursor, mono, pan))
            timings.append(
                TurnTiming(
                    index=turn.index,
                    role=turn.role,
                    speaker=turn.speaker,
                    voice_id=turn.voice_id,
                    start_s=cursor / sample_rate,
                    end_s=(cursor + len(mono)) / sample_rate,
                )
            )

            injected = self._maybe_backchannel(
                turn=turn,
                turns=turns,
                turn_start=cursor,
                turn_frames=len(mono),
                sample_rate=sample_rate,
                pan_positions=pan_positions,
                placements=placements,
                rng=rng,
            )
            backchannel_count += injected

            cursor += len(mono)

        audio = self._render(placements, sample_rate=sample_rate)
        return MixResult(
            audio=audio,
            timings=tuple(timings),
            pan_positions=pan_positions,
            backchannel_count=backchannel_count,
        )

    # -- backchannels -------------------------------------------------------

    def _maybe_backchannel(
        self,
        *,
        turn: MixTurn,
        turns: list[MixTurn],
        turn_start: int,
        turn_frames: int,
        sample_rate: int,
        pan_positions: dict[str, float],
        placements: list[tuple[int, np.ndarray, float]],
        rng: random.Random,
    ) -> int:
        config = self.config.backchannels
        if config is None:
            return 0
        if turn_frames / sample_rate < config.min_turn_duration_s:
            return 0
        if rng.random() >= config.probability:
            return 0

        listeners = [other for other in turns if other.role != turn.role]
        if not listeners:
            return 0
        listener = rng.choice(listeners)
        pool = self._pool_for(listener.voice_tags)
        if not pool:
            return 0

        listener_pan = pan_positions.get(listener.role, 0.0)
        injected = 0

        def place(clip: Audio, at_frame: int) -> None:
            nonlocal injected
            mono = _resampled_mono(clip, sample_rate) * _db_to_gain(rng.uniform(*config.gain_db))
            placements.append((max(0, at_frame), mono, listener_pan))
            injected += 1

        offset = int(turn_frames * rng.uniform(*config.position))
        place(rng.choice(pool), turn_start + offset)

        if rng.random() < config.double_probability:
            gap = int(config.double_gap_s * sample_rate)
            place(rng.choice(pool), turn_start + offset + gap)

        return injected

    def _pool_for(self, tags: tuple[str, ...]) -> list[Audio]:
        """Backchannel clips matching any of ``tags``, falling back to the shared pool."""
        library = self._load_backchannels()
        for tag in tags:
            if library.get(tag):
                return library[tag]
        return library.get("", [])

    def _load_backchannels(self) -> dict[str, list[Audio]]:
        if self._backchannels is not None:
            return self._backchannels

        library: dict[str, list[Audio]] = {}
        config = self.config.backchannels
        if config is None:
            self._backchannels = library
            return library

        root = Path(config.directory)
        if not root.is_dir():
            logger.warning("Backchannel directory %s does not exist; disabling injection", root)
            self._backchannels = library
            return library

        subdirectories = [path for path in sorted(root.iterdir()) if path.is_dir()]
        if subdirectories:
            for subdirectory in subdirectories:
                clips = _load_clips(subdirectory)
                if clips:
                    library[subdirectory.name] = clips
        else:
            clips = _load_clips(root)
            if clips:
                library[""] = clips

        if not library:
            logger.warning("No backchannel clips found under %s", root)
        else:
            logger.info(
                "Loaded backchannels: %s",
                ", ".join(f"{tag or 'shared'}={len(clips)}" for tag, clips in library.items()),
            )
        self._backchannels = library
        return library

    # -- rendering ----------------------------------------------------------

    def _render(
        self,
        placements: list[tuple[int, np.ndarray, float]],
        *,
        sample_rate: int,
    ) -> Audio:
        config = self.config
        channels = 2 if config.stereo else 1
        tail = config.background.tail_s if config.background else 0.5
        total_frames = max(start + len(mono) for start, mono, _ in placements)
        total_frames += int(tail * sample_rate)

        canvas = self._base_canvas(total_frames, channels, sample_rate)
        for start, mono, pan in placements:
            block = _pan(mono, pan) if channels == 2 else mono[:, None]
            end = start + block.shape[0]
            if end > canvas.shape[0]:  # defensive: a backchannel may run past the tail
                canvas = np.pad(canvas, ((0, end - canvas.shape[0]), (0, 0)))
            canvas[start:end] += block

        canvas = _peak_normalize(canvas, config.target_peak_dbfs)
        samples = canvas[:, 0] if channels == 1 else canvas
        return Audio(np.ascontiguousarray(samples, dtype=np.float32), sample_rate)

    def _base_canvas(self, frames: int, channels: int, sample_rate: int) -> np.ndarray:
        config = self.config.background
        if config is None:
            return np.zeros((frames, channels), dtype=np.float32)

        path = Path(config.path)
        if not path.exists():
            logger.warning("Background audio %s does not exist; using silence", path)
            return np.zeros((frames, channels), dtype=np.float32)

        bed = _resampled_mono(load_audio(path), sample_rate)
        if bed.size == 0:
            return np.zeros((frames, channels), dtype=np.float32)

        repeats = math.ceil(frames / bed.size)
        looped = np.tile(bed, repeats)[:frames] * _db_to_gain(config.gain_db)
        return np.repeat(looped[:, None], channels, axis=1).astype(np.float32)


# -- signal helpers ---------------------------------------------------------


def _db_to_gain(decibels: float) -> float:
    """Convert a dB offset to a linear gain factor."""
    return float(10.0 ** (decibels / 20.0))


def _peak_normalize(samples: np.ndarray, target_dbfs: float) -> np.ndarray:
    """Scale ``samples`` so the largest absolute value sits at ``target_dbfs``."""
    peak = float(np.abs(samples).max()) if samples.size else 0.0
    if peak <= 0.0:
        return samples.astype(np.float32)
    return (samples * (_db_to_gain(target_dbfs) / peak)).astype(np.float32)


def _pan(mono: np.ndarray, position: float) -> np.ndarray:
    """Place mono audio in the stereo field using a constant-power pan law.

    Constant power (rather than linear) keeps perceived loudness steady as a speaker
    moves off centre, so panning does not double as an unintended level change.
    """
    position = float(np.clip(position, -1.0, 1.0))
    angle = (position + 1.0) * (math.pi / 4.0)
    return np.stack([mono * math.cos(angle), mono * math.sin(angle)], axis=1).astype(np.float32)


def _resampled_mono(audio: Audio, sample_rate: int) -> np.ndarray:
    """Return mono samples at ``sample_rate``.

    Rate conversion uses linear interpolation, which is adequate for the assets that
    need it -- room tone and short backchannels. Utterances within a conversation come
    from a single backend and share a rate, so they pass through untouched.
    """
    mono = audio.to_mono().samples.astype(np.float32)
    if audio.sample_rate == sample_rate or mono.size == 0:
        return mono

    logger.debug("Resampling %d Hz -> %d Hz (%d frames)", audio.sample_rate, sample_rate, mono.size)
    duration = mono.size / audio.sample_rate
    target_frames = max(1, round(duration * sample_rate))
    source_positions = np.linspace(0.0, mono.size - 1, num=mono.size, dtype=np.float64)
    target_positions = np.linspace(0.0, mono.size - 1, num=target_frames, dtype=np.float64)
    return np.interp(target_positions, source_positions, mono).astype(np.float32)


def _load_clips(directory: Path) -> list[Audio]:
    return [
        load_audio(path)
        for path in sorted(directory.iterdir())
        if path.suffix.lower() in _AUDIO_SUFFIXES
    ]
