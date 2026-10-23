"""Speech synthesis, voice management and conversation mixing.

Backends are imported lazily, so installing conversim without a TTS extra still lets
you import this package and use the mixer on audio you produced elsewhere.
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Any

from .base import Audio, SpeechSynthesizer, Voice, load_audio
from .mixing import (
    BackchannelConfig,
    BackgroundConfig,
    ConversationMixer,
    MixConfig,
    MixResult,
    MixTurn,
    TurnTiming,
)
from .voices import (
    AssignmentStrategy,
    FixedAssignment,
    RandomAssignment,
    SwappedAssignment,
    VoiceAssignment,
    VoiceLibrary,
)

if TYPE_CHECKING:
    from .openai_tts import OpenAITTS
    from .qwen_tts import QwenTTS

__all__ = [
    "AssignmentStrategy",
    "Audio",
    "BackchannelConfig",
    "BackgroundConfig",
    "ConversationMixer",
    "FixedAssignment",
    "MixConfig",
    "MixResult",
    "MixTurn",
    "OpenAITTS",
    "QwenTTS",
    "RandomAssignment",
    "SpeechSynthesizer",
    "SwappedAssignment",
    "TurnTiming",
    "Voice",
    "VoiceAssignment",
    "VoiceLibrary",
    "load_audio",
]

_LAZY = {
    "OpenAITTS": ("conversim.speech.openai_tts", "OpenAITTS"),
    "QwenTTS": ("conversim.speech.qwen_tts", "QwenTTS"),
}


def __getattr__(name: str) -> Any:
    """Import TTS backends on first access, so missing extras stay silent."""
    target = _LAZY.get(name)
    if target is None:
        raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
    from importlib import import_module

    value = getattr(import_module(target[0]), target[1])
    globals()[name] = value
    return value


def __dir__() -> list[str]:
    return sorted(__all__)
