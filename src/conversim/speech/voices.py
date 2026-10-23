"""Collecting voices and assigning them to roles.

Two ideas earn their own module here. First, a **voice library** loaded from a
directory, because reference recordings for cloning naturally live as
``name.wav`` + ``name.txt`` pairs. Second, **assignment as a separate concept from
casting**: a persona is a character in the script, a voice is a timbre. Keeping them
apart is what makes it possible to render one dialogue twice with the roles swapped --
useful augmentation, since it doubles ASR training material at zero generation cost
and controls for voice when comparing.
"""

from __future__ import annotations

import itertools
import logging
import random
from collections.abc import Iterator
from pathlib import Path
from typing import Protocol, runtime_checkable

from pydantic import BaseModel, ConfigDict, Field, model_validator

from ..errors import ConfigurationError
from ..models import Cast
from .base import Voice

__all__ = [
    "AssignmentStrategy",
    "FixedAssignment",
    "RandomAssignment",
    "SwappedAssignment",
    "VoiceAssignment",
    "VoiceLibrary",
]

logger = logging.getLogger(__name__)

_AUDIO_SUFFIXES = (".wav", ".flac", ".ogg", ".mp3", ".m4a")


class VoiceLibrary(BaseModel):
    """A named collection of voices."""

    model_config = ConfigDict(frozen=True)

    voices: tuple[Voice, ...]

    @model_validator(mode="after")
    def _ids_unique(self) -> VoiceLibrary:
        if not self.voices:
            raise ValueError("VoiceLibrary must contain at least one voice")
        ids = [voice.id for voice in self.voices]
        duplicates = {value for value in ids if ids.count(value) > 1}
        if duplicates:
            raise ValueError(f"Duplicate voice ids: {sorted(duplicates)}")
        return self

    def __len__(self) -> int:
        return len(self.voices)

    def __iter__(self) -> Iterator[Voice]:  # type: ignore[override]
        return iter(self.voices)

    @property
    def ids(self) -> tuple[str, ...]:
        """All voice ids, in library order."""
        return tuple(voice.id for voice in self.voices)

    def get(self, voice_id: str) -> Voice:
        """Look up a voice by id.

        Raises:
            ConfigurationError: if no such voice exists.
        """
        for voice in self.voices:
            if voice.id == voice_id:
                return voice
        raise ConfigurationError(f"Unknown voice {voice_id!r}; library has {', '.join(self.ids)}")

    def tagged(self, tag: str) -> tuple[Voice, ...]:
        """All voices carrying ``tag``."""
        return tuple(voice for voice in self.voices if tag in voice.tags)

    @classmethod
    def from_directory(
        cls,
        directory: Path | str,
        *,
        language: str = "German",
        tags: dict[str, tuple[str, ...]] | None = None,
        require_transcript: bool = True,
    ) -> VoiceLibrary:
        """Load cloning voices from a directory of ``name.wav`` + ``name.txt`` pairs.

        Args:
            directory: Directory to scan. Not searched recursively.
            language: Language assigned to every loaded voice.
            tags: Optional per-voice-id tags, e.g. ``{"joshua": ("male",)}``. Tags
                drive backchannel selection during mixing.
            require_transcript: Skip audio files with no matching ``.txt`` and log a
                warning. Set ``False`` to load them anyway, which only works with
                backends that do not need a reference transcript.

        Raises:
            ConfigurationError: if the directory is missing or yields no voices.
        """
        base = Path(directory)
        if not base.is_dir():
            raise ConfigurationError(f"Voice directory does not exist: {base}")

        tags = tags or {}
        voices: list[Voice] = []
        for audio_path in sorted(
            path for path in base.iterdir() if path.suffix.lower() in _AUDIO_SUFFIXES
        ):
            voice_id = audio_path.stem
            transcript_path = audio_path.with_suffix(".txt")
            transcript: str | None = None
            if transcript_path.exists():
                transcript = transcript_path.read_text(encoding="utf-8").strip()
            elif require_transcript:
                logger.warning(
                    "Skipping voice %r: no transcript at %s", voice_id, transcript_path.name
                )
                continue

            voices.append(
                Voice(
                    id=voice_id,
                    language=language,
                    reference_audio=audio_path,
                    reference_text=transcript,
                    tags=tags.get(voice_id, ()),
                )
            )

        if not voices:
            raise ConfigurationError(
                f"No usable voices found in {base}. Expected pairs of "
                f"'<name>.wav' and '<name>.txt'."
            )
        logger.info("Loaded %d voices from %s", len(voices), base)
        return cls(voices=tuple(voices))


class VoiceAssignment(BaseModel):
    """One casting of voices onto roles -- a "take" of the same dialogue.

    Args:
        id: Identifier for this take, used in output filenames.
        voices: Mapping of role id to voice id.
    """

    model_config = ConfigDict(frozen=True)

    id: str
    voices: dict[str, str]

    @model_validator(mode="after")
    def _non_empty(self) -> VoiceAssignment:
        if not self.voices:
            raise ValueError(f"Assignment {self.id!r} maps no roles to voices")
        return self

    def voice_for(self, role_id: str) -> str:
        """Voice id cast in ``role_id``.

        Raises:
            ConfigurationError: if the role is not covered.
        """
        try:
            return self.voices[role_id]
        except KeyError as error:
            raise ConfigurationError(
                f"Assignment {self.id!r} does not cover role {role_id!r}; "
                f"covers {sorted(self.voices)}"
            ) from error


@runtime_checkable
class AssignmentStrategy(Protocol):
    """Decides which voice assignments to render for a conversation."""

    def assign(
        self,
        *,
        cast: Cast,
        library: VoiceLibrary,
        rng: random.Random,
    ) -> tuple[VoiceAssignment, ...]:
        """Return one assignment per take to render."""
        ...


class FixedAssignment(BaseModel):
    """Always use the same role-to-voice mapping.

    Args:
        voices: Mapping of role id to voice id.
        take_id: Identifier for the single take produced.
    """

    model_config = ConfigDict(frozen=True)

    voices: dict[str, str]
    take_id: str = "take"

    def assign(
        self,
        *,
        cast: Cast,
        library: VoiceLibrary,
        rng: random.Random,
    ) -> tuple[VoiceAssignment, ...]:
        """Return the configured mapping as a single take."""
        for voice_id in self.voices.values():
            library.get(voice_id)  # fail fast on typos
        return (VoiceAssignment(id=self.take_id, voices=dict(self.voices)),)


class RandomAssignment(BaseModel):
    """Draw voices at random, optionally producing several takes.

    Args:
        takes: How many distinct assignments to render per conversation.
        distinct: Require a different voice per role. Two roles sharing a voice makes
            the recording near-useless for diarization and unpleasant to listen to.
        restrict_to_tags: Only draw from voices carrying at least one of these tags.

    Raises:
        ConfigurationError: at assignment time, if the library cannot supply enough
            distinct voices.
    """

    model_config = ConfigDict(frozen=True)

    takes: int = Field(default=1, ge=1)
    distinct: bool = True
    restrict_to_tags: tuple[str, ...] = ()

    def assign(
        self,
        *,
        cast: Cast,
        library: VoiceLibrary,
        rng: random.Random,
    ) -> tuple[VoiceAssignment, ...]:
        """Draw ``takes`` assignments."""
        candidates = _candidates(library, self.restrict_to_tags)
        roles = cast.role_ids

        if self.distinct and len(candidates) < len(roles):
            raise ConfigurationError(
                f"Need {len(roles)} distinct voices for roles {list(roles)} but only "
                f"{len(candidates)} are available"
                + (f" with tags {list(self.restrict_to_tags)}" if self.restrict_to_tags else "")
            )

        assignments: list[VoiceAssignment] = []
        seen: set[tuple[str, ...]] = set()
        for index in range(self.takes):
            chosen = (
                rng.sample(candidates, len(roles))
                if self.distinct
                else [rng.choice(candidates) for _ in roles]
            )
            key = tuple(chosen)
            if key in seen and self.distinct:
                continue  # skip a duplicate take rather than render it twice
            seen.add(key)
            assignments.append(
                VoiceAssignment(
                    id="-".join(chosen) if self.takes > 1 else f"take{index}",
                    voices=dict(zip(roles, chosen, strict=True)),
                )
            )
        return tuple(assignments)


class SwappedAssignment(BaseModel):
    """Render every permutation of one drawn set of voices.

    With two roles this yields the two takes of the same dialogue with the voices
    exchanged: identical words, different speaker per line. Cheap augmentation, and it
    isolates voice as a variable when comparing ASR results.

    Args:
        restrict_to_tags: Only draw from voices carrying at least one of these tags.
        limit: Cap on the number of permutations rendered. Permutation count grows
            factorially, so a cast of four roles would otherwise mean 24 takes.
    """

    model_config = ConfigDict(frozen=True)

    restrict_to_tags: tuple[str, ...] = ()
    limit: int = Field(default=2, ge=1)

    def assign(
        self,
        *,
        cast: Cast,
        library: VoiceLibrary,
        rng: random.Random,
    ) -> tuple[VoiceAssignment, ...]:
        """Draw one voice set and return its permutations."""
        candidates = _candidates(library, self.restrict_to_tags)
        roles = cast.role_ids
        if len(candidates) < len(roles):
            raise ConfigurationError(
                f"Need {len(roles)} distinct voices but only {len(candidates)} are available"
            )

        drawn = rng.sample(candidates, len(roles))
        assignments = []
        for permutation in itertools.islice(itertools.permutations(drawn), self.limit):
            assignments.append(
                VoiceAssignment(
                    id="-".join(permutation),
                    voices=dict(zip(roles, permutation, strict=True)),
                )
            )
        return tuple(assignments)


def _candidates(library: VoiceLibrary, tags: tuple[str, ...]) -> list[str]:
    if not tags:
        return list(library.ids)
    matching = [voice.id for voice in library.voices if any(tag in voice.tags for tag in tags)]
    if not matching:
        raise ConfigurationError(
            f"No voices carry any of the tags {list(tags)}; "
            f"library tags are {sorted({tag for v in library.voices for tag in v.tags})}"
        )
    return matching
