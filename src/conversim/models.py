"""Core domain models: who speaks, what they say, and in what setting.

These types are deliberately domain-agnostic. Nothing here knows about medicine;
a :class:`Role` may be ``clinician``/``patient`` just as well as
``interviewer``/``candidate`` or ``agent``/``customer``.
"""

from __future__ import annotations

from typing import Any, ClassVar, Literal

from pydantic import BaseModel, ConfigDict, Field, field_validator, model_validator

from .errors import ConfigurationError

__all__ = [
    "Cast",
    "Dialogue",
    "DialogueIssue",
    "Persona",
    "Role",
    "Scenario",
    "TokenUsage",
    "TurnTarget",
    "Utterance",
]


class Role(BaseModel):
    """A structural position in the conversation, independent of who fills it.

    Roles are referenced by :attr:`id` throughout the library: personas declare the
    role they play, voice assignments map roles to voices, and prompt builders use
    :attr:`label` when describing participants to the model.
    """

    model_config = ConfigDict(frozen=True)

    id: str = Field(description="Stable identifier, e.g. 'clinician'.")
    label: str = Field(description="Human-readable name used in prompts, e.g. 'Arzt'.")
    description: str | None = Field(
        default=None,
        description="What this role does in the conversation. Included in prompts when set.",
    )

    @field_validator("id")
    @classmethod
    def _id_is_slug(cls, value: str) -> str:
        if not value or not all(char.isalnum() or char in "_-" for char in value):
            raise ValueError(f"Role id must be a non-empty slug, got {value!r}")
        return value


class Persona(BaseModel):
    """A concrete character filling a :class:`Role`.

    :attr:`name` is the label that appears in the generated script (``Name: text``),
    which makes it the key used to attribute utterances back to a role.
    """

    model_config = ConfigDict(frozen=True)

    name: str = Field(description="Speaker name as it appears in the dialogue script.")
    role: str = Field(description="Id of the :class:`Role` this persona plays.")
    description: str | None = Field(
        default=None, description="Character sketch handed to the dialogue model."
    )
    traits: dict[str, Any] = Field(
        default_factory=dict,
        description="Free-form attributes for prompt builders and downstream analysis.",
    )

    @field_validator("name")
    @classmethod
    def _name_is_usable_as_speaker_label(cls, value: str) -> str:
        stripped = value.strip()
        if not stripped:
            raise ValueError("Persona name must not be empty")
        if ":" in stripped:
            raise ValueError(
                f"Persona name must not contain ':' (it delimits speaker from text): {value!r}"
            )
        if "\n" in stripped:
            raise ValueError(f"Persona name must be single-line, got {value!r}")
        return stripped

    def __str__(self) -> str:
        return self.name


class Cast(BaseModel):
    """The set of personas playing a conversation, one per role.

    Exactly one persona per role is required: the dialogue format attributes each
    line to a speaker name, so two personas sharing a role would be ambiguous.
    """

    model_config = ConfigDict(frozen=True)

    personas: tuple[Persona, ...] = Field(description="Personas taking part, in speaking order.")

    @model_validator(mode="after")
    def _roles_and_names_unique(self) -> Cast:
        if not self.personas:
            raise ValueError("Cast must contain at least one persona")

        roles = [persona.role for persona in self.personas]
        duplicate_roles = {role for role in roles if roles.count(role) > 1}
        if duplicate_roles:
            raise ValueError(
                f"Each role may be filled by exactly one persona; duplicated: "
                f"{sorted(duplicate_roles)}"
            )

        names = [persona.name.casefold() for persona in self.personas]
        duplicate_names = {name for name in names if names.count(name) > 1}
        if duplicate_names:
            raise ValueError(
                f"Persona names must be distinct (case-insensitively); duplicated: "
                f"{sorted(duplicate_names)}"
            )
        return self

    @property
    def names(self) -> tuple[str, ...]:
        """Speaker names in speaking order."""
        return tuple(persona.name for persona in self.personas)

    @property
    def role_ids(self) -> tuple[str, ...]:
        """Role ids in speaking order."""
        return tuple(persona.role for persona in self.personas)

    def by_role(self, role_id: str) -> Persona:
        """Return the persona playing ``role_id``.

        Raises:
            ConfigurationError: if no persona plays that role.
        """
        for persona in self.personas:
            if persona.role == role_id:
                return persona
        raise ConfigurationError(
            f"No persona plays role {role_id!r}; cast covers {sorted(self.role_ids)}"
        )

    def role_of(self, speaker_name: str) -> str | None:
        """Return the role id for ``speaker_name``, or ``None`` if unknown.

        Matching is case-insensitive, mirroring how models tend to vary capitalisation.
        """
        folded = speaker_name.strip().casefold()
        for persona in self.personas:
            if persona.name.casefold() == folded:
                return persona.role
        return None


class TurnTarget(BaseModel):
    """The requested dialogue length, in speaker turns.

    Treated as guidance rather than a hard constraint: language models overshoot and
    undershoot, and forcing an exact count degrades naturalness. Validation reports
    deviations instead of rejecting them -- see
    :class:`~conversim.stages.dialogue.DialogueValidator`.
    """

    model_config = ConfigDict(frozen=True)

    minimum: int = Field(ge=1, description="Lower bound handed to the model.")
    maximum: int = Field(ge=1, description="Upper bound handed to the model.")

    @model_validator(mode="after")
    def _ordered(self) -> TurnTarget:
        if self.maximum < self.minimum:
            raise ValueError(f"maximum ({self.maximum}) must be >= minimum ({self.minimum})")
        return self

    def describe(self) -> str:
        """Render the range for inclusion in a prompt."""
        if self.minimum == self.maximum:
            return str(self.minimum)
        return f"{self.minimum}-{self.maximum}"


class Scenario(BaseModel):
    """The situational frame of one conversation.

    Everything here is prompt-facing text: it tells the models *where* the
    conversation happens and *what* it is about, without constraining the facts
    (those live in the ground truth).
    """

    model_config = ConfigDict(frozen=True)

    topic: str = Field(description="What the conversation is about.")
    context: str = Field(description="Setting and circumstances, e.g. location and atmosphere.")
    language: str = Field(
        default="German",
        description="Natural-language name of the target language, as used in prompts.",
    )
    turns: TurnTarget = Field(description="Requested dialogue length.")


class Utterance(BaseModel):
    """One line of dialogue, attributed to a persona and its role."""

    model_config = ConfigDict(frozen=True)

    index: int = Field(ge=0, description="Zero-based position in the dialogue.")
    speaker: str = Field(description="Persona name as written in the script.")
    role: str = Field(description="Id of the role this speaker plays.")
    text: str = Field(description="What is said.")
    cue: str | None = Field(
        default=None,
        description="Optional delivery cue (tone/emotion) captured from '(cue)' in the script.",
    )

    def to_script_line(self) -> str:
        """Render back into the canonical ``Name: (cue) text`` script format."""
        if self.cue:
            return f"{self.speaker}: ({self.cue}) {self.text}"
        return f"{self.speaker}: {self.text}"


class DialogueIssue(BaseModel):
    """A non-fatal problem observed while parsing or validating a dialogue.

    Severity separates "this sample is damaged" from "this sample is fine but did
    not follow an instruction". A dialogue that came back longer than the sampled
    turn range is still a usable dialogue; one with an unparsable line is not.
    The default is ``"major"`` so a new issue kind has to opt into being minor.
    """

    model_config = ConfigDict(frozen=True)

    #: Kinds that are minor unless stated otherwise. The emitting stage sets the
    #: severity itself; this is what makes datasets written before the field
    #: existed read back with the right one instead of defaulting to major.
    MINOR_KINDS: ClassVar[frozenset[str]] = frozenset({"turn_count"})

    kind: str = Field(description="Machine-readable category, e.g. 'unparsed_line'.")
    detail: str = Field(description="Human-readable explanation.")
    severity: Literal["minor", "major"] = Field(
        default="major",
        description="'major' makes the sample unclean; 'minor' is reported but harmless.",
    )
    line_number: int | None = Field(
        default=None, description="One-based line number in the raw script, when applicable."
    )

    @model_validator(mode="before")
    @classmethod
    def _severity_from_kind(cls, data: Any) -> Any:
        """Fill in the severity of a known-minor kind when it was not stored."""
        if isinstance(data, dict) and "severity" not in data and data.get("kind") in cls.MINOR_KINDS:
            return {**data, "severity": "minor"}
        return data


class Dialogue(BaseModel):
    """A parsed dialogue together with the raw model output it came from."""

    utterances: tuple[Utterance, ...] = Field(description="Parsed lines, in order.")
    raw: str = Field(description="Unmodified model output, kept for provenance.")
    issues: tuple[DialogueIssue, ...] = Field(
        default=(), description="Problems found during parsing and validation."
    )

    def __len__(self) -> int:
        return len(self.utterances)

    def to_script(self) -> str:
        """Render all utterances as a script, one line per turn."""
        return "\n".join(utterance.to_script_line() for utterance in self.utterances)

    def transcript(self, separator: str = " ") -> str:
        """Render the spoken words only, without speaker labels or cues.

        This is the form an ASR system is expected to produce, which makes it the
        natural transcription target for datasets built with conversim.
        """
        return separator.join(utterance.text for utterance in self.utterances)

    def by_role(self, role_id: str) -> tuple[Utterance, ...]:
        """Return only the utterances spoken by ``role_id``."""
        return tuple(utterance for utterance in self.utterances if utterance.role == role_id)


class TokenUsage(BaseModel):
    """Token counts reported by a provider.

    conversim reports usage but deliberately does not price it: published rates change
    faster than a library release cycle, and a stale hard-coded price table produces
    confidently wrong cost figures. Multiply by your own rates if you need currency.
    """

    model_config = ConfigDict(frozen=True)

    input_tokens: int = Field(default=0, ge=0)
    output_tokens: int = Field(default=0, ge=0)

    @property
    def total_tokens(self) -> int:
        return self.input_tokens + self.output_tokens

    def __add__(self, other: TokenUsage) -> TokenUsage:
        return TokenUsage(
            input_tokens=self.input_tokens + other.input_tokens,
            output_tokens=self.output_tokens + other.output_tokens,
        )
