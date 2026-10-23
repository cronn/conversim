"""Audio primitives and the speech-synthesis interface."""

from __future__ import annotations

import logging
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Protocol, runtime_checkable

import numpy as np
import soundfile as sf
from pydantic import BaseModel, ConfigDict, Field

from ..errors import ConfigurationError

__all__ = ["Audio", "SpeechSynthesizer", "Voice", "load_audio"]

logger = logging.getLogger(__name__)


@dataclass(frozen=True)
class Audio:
    """A block of PCM audio.

    Samples are float32 in ``[-1, 1]``, shaped ``(frames,)`` for mono or
    ``(frames, channels)`` otherwise. Keeping audio in memory as arrays rather than
    file paths lets the mixer compose without a round-trip through disk, and keeps
    file-format concerns at the edges.
    """

    samples: np.ndarray
    sample_rate: int

    def __post_init__(self) -> None:
        if self.samples.ndim not in (1, 2):
            raise ConfigurationError(
                f"Audio must be 1-D (mono) or 2-D (frames, channels), got shape "
                f"{self.samples.shape}"
            )
        if self.sample_rate <= 0:
            raise ConfigurationError(f"Sample rate must be positive, got {self.sample_rate}")

    @property
    def frames(self) -> int:
        """Number of sample frames."""
        return int(self.samples.shape[0])

    @property
    def channels(self) -> int:
        """Number of channels."""
        return 1 if self.samples.ndim == 1 else int(self.samples.shape[1])

    @property
    def duration_s(self) -> float:
        """Duration in seconds."""
        return self.frames / self.sample_rate

    def to_mono(self) -> Audio:
        """Collapse to a single channel by averaging."""
        if self.channels == 1:
            return self if self.samples.ndim == 1 else Audio(self.samples[:, 0], self.sample_rate)
        return Audio(self.samples.mean(axis=1).astype(np.float32), self.sample_rate)

    def peak(self) -> float:
        """Largest absolute sample value, or ``0.0`` for empty audio."""
        return float(np.abs(self.samples).max()) if self.frames else 0.0

    def write(self, path: Path, subtype: str = "PCM_16") -> Path:
        """Write to ``path``, creating parent directories as needed."""
        path.parent.mkdir(parents=True, exist_ok=True)
        sf.write(str(path), self.samples, self.sample_rate, subtype=subtype)
        return path


def load_audio(path: Path, *, mono: bool = True) -> Audio:
    """Read an audio file into an :class:`Audio`.

    Format support comes from libsndfile via ``soundfile``. WAV and FLAC are always
    available; MP3 needs a reasonably recent libsndfile. Prefer WAV for reference and
    ambience assets so the library never depends on an external decoder binary.
    """
    samples, sample_rate = sf.read(str(path), dtype="float32", always_2d=False)
    audio = Audio(np.ascontiguousarray(samples), int(sample_rate))
    return audio.to_mono() if mono else audio


class Voice(BaseModel):
    """A synthesis voice.

    Which fields matter depends on the backend, so the model carries the union rather
    than forcing a class hierarchy on what is really configuration:

    * :class:`~conversim.speech.openai_tts.OpenAITTS` uses :attr:`preset`.
    * :class:`~conversim.speech.qwen_tts.QwenTTS` uses :attr:`reference_audio` and
      :attr:`reference_text`.

    Args:
        id: Stable identifier. Appears in take ids and segment paths, so keep it
            filesystem-safe.
        language: Natural-language name, passed to backends that need it.
        preset: Named voice for backends that offer a fixed catalogue.
        reference_audio: Reference recording for cloning backends. A handful of
            seconds of clean speech is the sweet spot; more adds little.
        reference_text: Verbatim transcript of ``reference_audio``. Cloning quality
            depends on this being accurate, so it is worth checking by ear.
        tags: Free-form labels. Used to pick matching backchannels (``"male"``,
            ``"female"``) and to slice datasets afterwards.
        params: Backend-specific extras passed through untouched.
    """

    model_config = ConfigDict(frozen=True)

    id: str
    language: str = "German"
    preset: str | None = None
    reference_audio: Path | None = None
    reference_text: str | None = None
    tags: tuple[str, ...] = ()
    params: dict[str, Any] = Field(default_factory=dict)

    def require_reference(self) -> tuple[Path, str]:
        """Return the reference pair, raising if either half is missing.

        Raises:
            ConfigurationError: if reference audio or its transcript is absent.
        """
        if self.reference_audio is None:
            raise ConfigurationError(f"Voice {self.id!r} has no reference_audio for cloning")
        if not self.reference_audio.exists():
            raise ConfigurationError(
                f"Reference audio for voice {self.id!r} does not exist: {self.reference_audio}"
            )
        if not self.reference_text:
            raise ConfigurationError(
                f"Voice {self.id!r} has reference_audio but no reference_text; "
                f"cloning backends need the transcript of the reference"
            )
        return self.reference_audio, self.reference_text


@runtime_checkable
class SpeechSynthesizer(Protocol):
    """Renders one utterance to audio.

    Declared ``async`` even though local backends are compute-bound: the pipeline is
    async throughout, and local implementations should offload to a worker thread
    (see :func:`asyncio.to_thread`) so the event loop stays responsive.
    """

    @property
    def name(self) -> str:
        """Short identifier recorded in take metadata, e.g. ``"qwen3-tts"``."""
        ...

    async def synthesize(
        self,
        *,
        text: str,
        voice: Voice,
        cue: str | None = None,
    ) -> Audio:
        """Render ``text`` in ``voice``.

        Args:
            text: What to say.
            voice: The voice to use.
            cue: Optional delivery hint, honoured only by backends that accept one.
        """
        ...
