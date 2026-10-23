"""Qwen3-TTS voice-cloning backend.

The local option, and the better one for German: cloning from a few seconds of a real
speaker's voice handles domain vocabulary and German prosody more convincingly than a
catalogue voice, and nothing leaves the machine -- which matters when the point of the
pipeline is that data stays on premises.

Requires the ``qwen`` extra::

    pip install 'conversim[qwen]'

Operationally: the model is loaded once, lazily, on first synthesis, and generation is
serialised behind a lock. GPU inference is not reentrant in a useful way here, and
running two conversations against one model instance would only trade throughput for
memory pressure. To use several GPUs, construct one backend per device.
"""

from __future__ import annotations

import asyncio
import logging
from typing import Any

import numpy as np

from ..errors import ConfigurationError, MissingDependencyError, SynthesisError
from .base import Audio, Voice, load_audio

__all__ = ["DEFAULT_QWEN_MODEL", "QwenTTS"]

logger = logging.getLogger(__name__)

DEFAULT_QWEN_MODEL = "Qwen/Qwen3-TTS-12Hz-1.7B-Base"
"""Model used unless overridden."""


class QwenTTS:
    """Voice cloning with Qwen3-TTS.

    Voices supply :attr:`~conversim.speech.base.Voice.reference_audio` together with
    :attr:`~conversim.speech.base.Voice.reference_text`. The transcript must match the
    reference recording: a wrong transcript degrades the clone in ways that are easy to
    miss by ear on a single sample and obvious across a whole dataset.

    Args:
        model_id: Hugging Face model identifier.
        device: Torch device string, e.g. ``"cuda:1"``. Pick the device explicitly
            rather than relying on ``cuda`` when a machine is shared.
        dtype: Torch dtype name. ``bfloat16`` is the usual choice.
        attn_implementation: Attention backend passed to the model loader. ``eager``
            works everywhere; ``flash_attention_2`` is faster where available.
        x_vector_only_mode: Condition on a speaker embedding alone, ignoring the
            reference transcript. Faster and needs no transcript, at some cost in
            fidelity.
        reference_cache: Keep decoded reference recordings in memory. Worth it: the
            same handful of voices is reused across every conversation.

    Example:
        >>> tts = QwenTTS(device="cuda:1")                            # doctest: +SKIP
        >>> library = VoiceLibrary.from_directory("actors")            # doctest: +SKIP
        >>> audio = await tts.synthesize(                              # doctest: +SKIP
        ...     text="Guten Tag.", voice=library.get("joshua")
        ... )
    """

    def __init__(
        self,
        *,
        model_id: str = DEFAULT_QWEN_MODEL,
        device: str = "cuda:0",
        dtype: str = "bfloat16",
        attn_implementation: str = "eager",
        x_vector_only_mode: bool = False,
        reference_cache: bool = True,
    ) -> None:
        self.model_id = model_id
        self.device = device
        self.dtype = dtype
        self.attn_implementation = attn_implementation
        self.x_vector_only_mode = x_vector_only_mode
        self._reference_cache_enabled = reference_cache
        self._references: dict[str, tuple[np.ndarray, int, str]] = {}
        self._model: Any | None = None
        self._lock = asyncio.Lock()

    @property
    def name(self) -> str:
        """Backend identifier recorded in take metadata."""
        return f"qwen3-tts:{self.model_id}"

    @property
    def is_loaded(self) -> bool:
        """Whether the model is currently resident."""
        return self._model is not None

    async def synthesize(
        self,
        *,
        text: str,
        voice: Voice,
        cue: str | None = None,
    ) -> Audio:
        """Render ``text`` in ``voice``.

        ``cue`` is accepted for interface compatibility and ignored: this backend
        takes no free-text delivery instruction. Generate scripts without cues when
        using it (``include_cues=False``), so nothing implies control that is absent.

        Raises:
            ConfigurationError: if the voice lacks a usable reference.
            SynthesisError: if generation fails or yields empty audio.
        """
        if cue:
            logger.debug("Ignoring delivery cue %r: %s takes no cues", cue, self.name)

        reference = await asyncio.to_thread(self._reference_for, voice)

        async with self._lock:
            return await asyncio.to_thread(self._generate, text, voice, reference)

    def _generate(
        self,
        text: str,
        voice: Voice,
        reference: tuple[np.ndarray, int, str],
    ) -> Audio:
        model = self._ensure_model()
        samples, sample_rate, transcript = reference
        try:
            waveforms, output_rate = model.generate_voice_clone(
                text=text,
                language=voice.language,
                ref_audio=(samples, sample_rate),
                ref_text=transcript,
                x_vector_only_mode=self.x_vector_only_mode,
                **voice.params,
            )
        except Exception as error:
            raise SynthesisError(
                f"Qwen3-TTS failed to render {len(text)} characters in voice {voice.id!r}: {error}"
            ) from error

        if not len(waveforms):
            raise SynthesisError(f"Qwen3-TTS returned no audio for voice {voice.id!r}")

        waveform = np.asarray(waveforms[0], dtype=np.float32)
        if waveform.size == 0:
            raise SynthesisError(f"Qwen3-TTS returned empty audio for voice {voice.id!r}")
        return Audio(np.ascontiguousarray(waveform), int(output_rate))

    def _reference_for(self, voice: Voice) -> tuple[np.ndarray, int, str]:
        cached = self._references.get(voice.id)
        if cached is not None:
            return cached

        if self.x_vector_only_mode:
            if voice.reference_audio is None:
                raise ConfigurationError(
                    f"Voice {voice.id!r} has no reference_audio; cloning needs one "
                    f"even in x_vector_only_mode"
                )
            path, transcript = voice.reference_audio, voice.reference_text or ""
        else:
            path, transcript = voice.require_reference()

        audio = load_audio(path)
        logger.info(
            "Loaded reference for voice %r: %.2fs at %d Hz",
            voice.id,
            audio.duration_s,
            audio.sample_rate,
        )
        reference = (audio.samples, audio.sample_rate, transcript)
        if self._reference_cache_enabled:
            self._references[voice.id] = reference
        return reference

    def _ensure_model(self) -> Any:
        if self._model is not None:
            return self._model

        try:
            import torch
            from qwen_tts import Qwen3TTSModel
        except ModuleNotFoundError as error:  # pragma: no cover - install shape
            missing = getattr(error, "name", "qwen-tts")
            raise MissingDependencyError("QwenTTS", "qwen", missing) from error

        torch_dtype = getattr(torch, self.dtype, None)
        if torch_dtype is None:
            raise ConfigurationError(f"Unknown torch dtype: {self.dtype!r}")

        if self.device.startswith("cuda"):
            if not torch.cuda.is_available():
                raise ConfigurationError(
                    f"Device {self.device!r} requested but CUDA is unavailable"
                )
            index = int(self.device.split(":")[1]) if ":" in self.device else 0
            if index >= torch.cuda.device_count():
                raise ConfigurationError(
                    f"Device {self.device!r} requested but only "
                    f"{torch.cuda.device_count()} CUDA device(s) are visible"
                )

        logger.info("Loading %s onto %s (%s)", self.model_id, self.device, self.dtype)
        self._model = Qwen3TTSModel.from_pretrained(
            self.model_id,
            device_map=self.device,
            dtype=torch_dtype,
            attn_implementation=self.attn_implementation,
        )
        logger.info("Model ready")
        return self._model

    def unload(self) -> None:
        """Release the model and free device memory.

        Worth calling before a long non-TTS phase in the same process; the weights are
        several gigabytes of VRAM that nothing else can use while they are resident.
        """
        if self._model is None:
            return
        self._model = None
        self._references.clear()
        try:
            import gc

            import torch

            gc.collect()
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
        except ModuleNotFoundError:  # pragma: no cover - torch absent means nothing to free
            pass
        logger.info("Unloaded %s", self.model_id)
