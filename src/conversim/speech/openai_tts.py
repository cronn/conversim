"""OpenAI text-to-speech backend.

Useful as the zero-hardware option: no GPU, no model download, and it accepts a
free-text delivery instruction, so ``(cue)`` annotations in a script actually change
the performance.

Two caveats worth knowing before using it for a German dataset. The catalogue voices
handle German unevenly, and a German sentence containing several English words tends
to pull the whole sentence into English pronunciation -- explicit instruction reduces
this but does not eliminate it. Second, the ``tts-1`` family ignores ``instructions``
entirely; only ``gpt-4o-mini-tts`` and its successors act on them.

Requires the ``openai`` extra::

    pip install 'conversim[openai]'
"""

from __future__ import annotations

import asyncio
import io
import logging
from collections.abc import Callable
from typing import Any, Literal

import numpy as np
import soundfile as sf

from ..errors import ConfigurationError, MissingDependencyError, SynthesisError
from ..providers.base import RetryableProviderError, RetryPolicy, retry_async
from .base import Audio, Voice

__all__ = ["OPENAI_VOICES", "OpenAITTS"]

logger = logging.getLogger(__name__)

OPENAI_VOICES: tuple[str, ...] = (
    "alloy",
    "ash",
    "ballad",
    "coral",
    "echo",
    "fable",
    "onyx",
    "nova",
    "sage",
    "shimmer",
    "verse",
)
"""Voice names offered by the service at the time of writing.

Used for a friendly error on typos, not as a hard constraint: a name outside this
tuple is passed through with a warning, so a newly added voice keeps working.
"""

AudioFormat = Literal["wav", "flac", "mp3", "opus", "aac", "pcm"]


def _require_openai() -> Any:
    try:
        import openai
    except ModuleNotFoundError as error:  # pragma: no cover - exercised by install shape
        raise MissingDependencyError("OpenAITTS", "openai", "openai") from error
    return openai


def default_instructions(voice: Voice, cue: str | None) -> str:
    """Build the delivery instruction for one utterance.

    Naming the target language explicitly is not redundant: it measurably reduces the
    model's tendency to switch accent mid-sentence around loanwords.
    """
    instruction = (
        f"Speak like a native {voice.language} speaker. Stay in {voice.language} for "
        f"every word, including loanwords and abbreviations."
    )
    if cue:
        instruction += f"\nDelivery: {cue}"
    return instruction


class OpenAITTS:
    """Speech synthesis through OpenAI's audio API.

    Voices are selected via :attr:`~conversim.speech.base.Voice.preset`.

    Args:
        model: TTS model. Keep the default unless you have a reason; the older
            ``tts-1`` models silently ignore delivery instructions.
        api_key: API key. Falls back to the SDK's own resolution when omitted.
        base_url: Override for a compatible endpoint.
        audio_format: Wire format requested. ``wav`` and ``flac`` decode without any
            external binary, which is why ``wav`` is the default.
        instructions: Override the delivery-instruction builder.
        max_concurrency: Cap on simultaneous requests.
        retry: Backoff policy for transient failures.
        timeout: Per-request timeout in seconds.

    Example:
        >>> tts = OpenAITTS()                                        # doctest: +SKIP
        >>> voice = Voice(id="doctor", preset="sage", language="German")
        >>> audio = await tts.synthesize(text="Guten Tag.", voice=voice)  # doctest: +SKIP
    """

    def __init__(
        self,
        *,
        model: str = "gpt-4o-mini-tts",
        api_key: str | None = None,
        base_url: str | None = None,
        audio_format: AudioFormat = "wav",
        instructions: Callable[[Voice, str | None], str] = default_instructions,
        max_concurrency: int = 4,
        retry: RetryPolicy | None = None,
        timeout: float | None = 300.0,
    ) -> None:
        openai = _require_openai()
        self._openai = openai
        self._model = model
        self._audio_format: AudioFormat = audio_format
        self._instructions = instructions
        self._retry = retry or RetryPolicy()
        self._semaphore = asyncio.Semaphore(max_concurrency)
        self._client = openai.AsyncOpenAI(
            api_key=api_key, base_url=base_url, timeout=timeout, max_retries=0
        )

    @property
    def name(self) -> str:
        """Backend identifier recorded in take metadata."""
        return f"openai-tts:{self._model}"

    async def aclose(self) -> None:
        """Close the underlying HTTP client."""
        await self._client.close()

    async def __aenter__(self) -> OpenAITTS:
        return self

    async def __aexit__(self, *exc_info: object) -> None:
        await self.aclose()

    async def synthesize(
        self,
        *,
        text: str,
        voice: Voice,
        cue: str | None = None,
    ) -> Audio:
        """Render ``text`` in ``voice``.

        Raises:
            ConfigurationError: if the voice has no ``preset``.
            SynthesisError: if the response cannot be decoded.
            ProviderError: if the request fails after exhausting retries.
        """
        if not voice.preset:
            raise ConfigurationError(
                f"Voice {voice.id!r} has no 'preset'; OpenAI TTS needs a catalogue voice "
                f"name (one of: {', '.join(OPENAI_VOICES)})"
            )
        if voice.preset not in OPENAI_VOICES:
            logger.warning(
                "Voice preset %r is not in the known catalogue; passing it through",
                voice.preset,
            )

        instructions = self._instructions(voice, cue)

        async def call() -> Audio:
            async with self._semaphore:
                try:
                    response = await self._client.audio.speech.create(
                        model=self._model,
                        voice=voice.preset,
                        input=text,
                        instructions=instructions,
                        response_format=self._audio_format,
                        **voice.params,
                    )
                except Exception as error:
                    raise self._classify(error) from error
            return _decode(await _read_bytes(response), self._audio_format)

        return await retry_async(
            call, policy=self._retry, description=f"OpenAI TTS ({self._model}, {voice.id})"
        )

    def _classify(self, error: Exception) -> Exception:
        openai = self._openai
        transient = (
            openai.RateLimitError,
            openai.APITimeoutError,
            openai.APIConnectionError,
            openai.InternalServerError,
        )
        if isinstance(error, transient):
            return RetryableProviderError(str(error))
        status = getattr(error, "status_code", None)
        if isinstance(status, int) and (status >= 500 or status == 429):
            return RetryableProviderError(str(error))
        return error


async def _read_bytes(response: Any) -> bytes:
    """Extract the audio payload across SDK response shapes."""
    for attribute in ("content",):
        payload = getattr(response, attribute, None)
        if isinstance(payload, bytes):
            return payload
    reader = getattr(response, "aread", None)
    if callable(reader):
        return bytes(await reader())
    reader = getattr(response, "read", None)
    if callable(reader):
        return bytes(reader())
    raise SynthesisError(f"Could not read audio bytes from a {type(response).__name__} response")


def _decode(payload: bytes, audio_format: AudioFormat) -> Audio:
    """Decode returned bytes into an :class:`~conversim.speech.base.Audio`."""
    if not payload:
        raise SynthesisError("OpenAI TTS returned an empty audio payload")
    try:
        samples, sample_rate = sf.read(io.BytesIO(payload), dtype="float32", always_2d=False)
    except Exception as error:
        raise SynthesisError(
            f"Could not decode {audio_format!r} audio from OpenAI TTS. Formats other "
            f"than 'wav' and 'flac' may need a newer libsndfile."
        ) from error
    return Audio(np.ascontiguousarray(samples), int(sample_rate)).to_mono()
