"""Roles and personas for STI/HIV consultations.

Personas describe *characters*, not voices. What a persona sounds like is decided later
by a :class:`~conversim.speech.voices.VoiceAssignment`, which is what allows the same
dialogue to be rendered with different speakers. Conflating the two -- as an earlier
version of this pipeline did, by giving actor configurations a ``voice`` field -- means
the recording's speaker assignment lives only in a filename.

Patient personas deliberately vary in how forthcoming they are. A dataset in which
every patient answers questions directly and completely teaches an extraction model
that information always arrives in the answer to the question that asked for it, which
is the opposite of what real consultations look like.
"""

from __future__ import annotations

from ...models import Persona, Role

__all__ = ["CLINICIAN", "CLINICIAN_PERSONAS", "PATIENT", "PATIENT_PERSONAS", "STI_ROLES"]

CLINICIAN = Role(
    id="clinician",
    label="Arzt:Ärztin",
    description=(
        "Führt die Anamnese. Geht respektvoll und nicht wertend mit sensiblen Themen um, "
        "ohne sie zu dramatisieren."
    ),
)
"""The clinician role."""

PATIENT = Role(
    id="patient",
    label="Patient:in",
    description="Sucht die Beratung auf und schildert die eigene Situation.",
)
"""The patient role."""

STI_ROLES: tuple[Role, ...] = (CLINICIAN, PATIENT)
"""Both roles, in speaking order."""


CLINICIAN_PERSONAS: tuple[Persona, ...] = (
    Persona(
        name="Lehmann",
        role="clinician",
        description=(
            "Ruhig, professionell und effizient. Wirkt erfahren, sachlich und nicht "
            "wertend. Geht respektvoll mit sensiblen Themen um, ohne sie zu dramatisieren."
        ),
    ),
    Persona(
        name="Yilmaz",
        role="clinician",
        description=(
            "Zugewandt, aufmerksam und respektvoll. Ermutigt zur freien Schilderung "
            "und geht sensibel mit Unsicherheiten um."
        ),
    ),
    Persona(
        name="Schneider",
        role="clinician",
        description=(
            "Entspannt, routiniert und einfühlsam. Verständnisvoll, verurteilt die "
            "Patient:in nicht."
        ),
    ),
    Persona(
        name="Müller",
        role="clinician",
        description=("Gründlich und systematisch. Erklärt viel und nimmt sich Zeit für Fragen."),
    ),
    Persona(
        name="Weber",
        role="clinician",
        description="Direkt aber freundlich. Hat einen lockeren Umgang mit sensiblen Themen.",
    ),
)
"""Clinician personas."""


PATIENT_PERSONAS: tuple[Persona, ...] = (
    Persona(
        name="Robin",
        role="patient",
        description=(
            "Beschreibt Dinge so, wie sie erlebt wurden — mit Umgangssprache, "
            "szenetypischen Ausdrücken und ohne Fachbegriffe."
        ),
    ),
    Persona(
        name="Mika",
        role="patient",
        description=(
            "Wirkt gelegentlich unsicher oder zögerlich. Gibt Informationen "
            "unstrukturiert und ohne Fachbegriffe preis, lässt sich aber gut führen."
        ),
    ),
    Persona(
        name="Luca",
        role="patient",
        description=(
            "Wirkt etwas nervös. Erzählt gerne den privaten Hintergrund und redet um "
            "sexuelle Themen herum, wenn Unsicherheit aufkommt."
        ),
    ),
    Persona(
        name="Charlie",
        role="patient",
        description=(
            "Offen und sex-positiv, spricht frei über die eigene Sexualität, mit Umgangssprache."
        ),
    ),
    Persona(
        name="Sascha",
        role="patient",
        description=(
            "Zurückhaltend bei sexuellen Themen. Gibt nur die nötigsten Informationen preis."
        ),
    ),
    Persona(
        name="Jessy",
        role="patient",
        description="Offen und sex-positiv, benutzt szenetypische Umgangssprache.",
    ),
    Persona(
        name="Alex",
        role="patient",
        description="Entspannt und pragmatisch. Beantwortet Fragen direkt ohne Umschweife.",
    ),
    Persona(
        name="Kim",
        role="patient",
        description="Anfangs unsicher, öffnet sich im Gesprächsverlauf zunehmend.",
    ),
)
"""Patient personas."""
