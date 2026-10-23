"""Parsing dialogue scripts back into structured utterances.

A model asked for ``Name: text`` complies most of the time. The interesting cases are
the rest: a bold speaker name, a bullet, a stray ``Here is the conversation:``
preamble, a trailing ``Let me know if you'd like more!``.

The approach here is to match against the *known* cast rather than a generic
"word followed by a colon" pattern. A generic pattern happily accepts
``GROUND TRUTH: ...`` as an utterance by a speaker named "GROUND TRUTH", and a
trailing pleasantry appended to the final turn silently corrupts the transcription
target for every ASR experiment that uses the dataset.
"""

from __future__ import annotations

import re
from dataclasses import dataclass
from typing import Literal

from .models import Cast, Dialogue, DialogueIssue, Utterance

__all__ = ["ContinuationPolicy", "DialogueParser"]

ContinuationPolicy = Literal["drop", "append"]
"""What to do with a line that carries no speaker prefix.

``drop`` discards it and records an issue -- the safe default, because model
pleasantries appended to the last turn end up in the transcript. ``append`` folds it
into the preceding utterance, which is right when the model hard-wraps long turns.
"""

# Leading list markers and surrounding emphasis that models add unprompted.
_LEADING_NOISE = re.compile(r"^\s*(?:[-*•]\s+|\d+[.)]\s+)?")
_EMPHASIS = re.compile(r"^[*_`\"'\s]+|[*_`\s]+$")


@dataclass(frozen=True)
class DialogueParser:
    """Parses a raw script into :class:`~conversim.models.Dialogue`.

    Args:
        cast: The personas expected to speak. Only their names are accepted as
            speaker labels, which is what keeps stray prose out of the transcript.
        continuation_policy: Handling for lines without a speaker prefix.
        capture_cues: Whether a leading ``(...)`` is captured as a delivery cue. When
            ``False`` such text stays part of the utterance.

    Example:
        >>> from conversim.models import Cast, Persona
        >>> cast = Cast(personas=(
        ...     Persona(name="Ada", role="a"), Persona(name="Bo", role="b"),
        ... ))
        >>> dialogue = DialogueParser(cast=cast).parse("Ada: (warmly) Hi.\\nBo: Hello!")
        >>> [(u.speaker, u.cue, u.text) for u in dialogue.utterances]
        [('Ada', 'warmly', 'Hi.'), ('Bo', None, 'Hello!')]
    """

    cast: Cast
    continuation_policy: ContinuationPolicy = "drop"
    capture_cues: bool = True

    def parse(self, raw: str) -> Dialogue:
        """Parse ``raw`` into utterances, collecting issues rather than raising."""
        pattern = self._speaker_pattern()
        utterances: list[Utterance] = []
        issues: list[DialogueIssue] = []
        pending_texts: list[list[str]] = []

        for line_number, line in enumerate(raw.splitlines(), start=1):
            stripped = line.strip()
            if not stripped or stripped.startswith("#"):
                continue

            match = pattern.match(_LEADING_NOISE.sub("", stripped, count=1))
            if match is None:
                self._handle_continuation(stripped, line_number, utterances, pending_texts, issues)
                continue

            speaker_label = _EMPHASIS.sub("", match.group("speaker"))
            role = self.cast.role_of(speaker_label)
            if role is None:  # unreachable: the pattern only admits known names
                issues.append(
                    DialogueIssue(
                        kind="unknown_speaker",
                        detail=f"Speaker {speaker_label!r} is not in the cast",
                        line_number=line_number,
                    )
                )
                continue

            # groupdict, not group: the cue group is absent when capture_cues is off.
            cue = match.groupdict().get("cue")
            text = match.group("text").strip()
            if not text:
                issues.append(
                    DialogueIssue(
                        kind="empty_utterance",
                        detail=f"{speaker_label} has a speaker label but no text",
                        line_number=line_number,
                    )
                )
                continue

            canonical_name = self.cast.by_role(role).name
            utterances.append(
                Utterance(
                    index=len(utterances),
                    speaker=canonical_name,
                    role=role,
                    text=text,
                    cue=cue.strip() if (cue and self.capture_cues) else None,
                )
            )
            pending_texts.append([text])

        merged = self._merge(utterances, pending_texts)
        return Dialogue(utterances=tuple(merged), raw=raw, issues=tuple(issues))

    def _handle_continuation(
        self,
        line: str,
        line_number: int,
        utterances: list[Utterance],
        pending_texts: list[list[str]],
        issues: list[DialogueIssue],
    ) -> None:
        if not utterances:
            issues.append(
                DialogueIssue(
                    kind="preamble",
                    detail=f"Discarded text before the first utterance: {_excerpt(line)}",
                    line_number=line_number,
                )
            )
            return
        if self.continuation_policy == "append":
            pending_texts[-1].append(line)
            return
        issues.append(
            DialogueIssue(
                kind="unparsed_line",
                detail=f"Discarded line without a speaker prefix: {_excerpt(line)}",
                line_number=line_number,
            )
        )

    @staticmethod
    def _merge(utterances: list[Utterance], pending_texts: list[list[str]]) -> list[Utterance]:
        """Fold appended continuation lines into their utterance."""
        merged: list[Utterance] = []
        for utterance, parts in zip(utterances, pending_texts, strict=True):
            if len(parts) == 1:
                merged.append(utterance)
                continue
            merged.append(utterance.model_copy(update={"text": " ".join(parts)}))
        return merged

    def _speaker_pattern(self) -> re.Pattern[str]:
        # Longest name first, so a name that is a prefix of another cannot shadow it.
        names = sorted(self.cast.names, key=len, reverse=True)
        alternatives = "|".join(re.escape(name) for name in names)
        cue_group = r"(?:\((?P<cue>[^)]{1,120})\)\s*)?" if self.capture_cues else ""
        return re.compile(
            rf"^[*_`]*\s*(?P<speaker>{alternatives})\s*[*_`]*\s*:\s*{cue_group}(?P<text>.*)$",
            re.IGNORECASE,
        )


def _excerpt(text: str, limit: int = 60) -> str:
    collapsed = " ".join(text.split())
    return collapsed if len(collapsed) <= limit else f"{collapsed[: limit - 1]}…"
