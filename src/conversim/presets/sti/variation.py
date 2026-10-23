"""Variation axes for STI/HIV consultations.

Seven categorical axes and three flags. Each axis contributes prompt text to *both*
generation stages, which is the point: telling only the dialogue stage that the patient
is hesitant produces a hesitant patient reciting a maximally complete record, because
the ground-truth stage never heard about it.

The style axis carries two pieces of metadata that other components read: a turn range
(so a "concise" consultation is actually short) and completeness guidance (so it also
has fewer facts to convey). Length and content are thus decided together rather than
fighting each other.
"""

from __future__ import annotations

from ...variation import CategoricalAxis, FlagAxis, VariationOption, VariationSpace

__all__ = [
    "CONVERSATION_TYPE",
    "DEMOGRAPHIC",
    "DIALECT",
    "EDUCATION",
    "HEALTH_LITERACY",
    "PATIENT_QUESTIONS",
    "SETTING",
    "SMALL_TALK",
    "STRUCTURE",
    "STYLE",
    "sti_variation_space",
]

CONVERSATION_TYPE = CategoricalAxis(
    name="conversation_type",
    label="Besuchsgrund",
    options=(
        VariationOption(
            value="routine_screening",
            instruction="Routineuntersuchung ohne akute Symptome",
        ),
        VariationOption(
            value="symptomatic", instruction="Patient:in berichtet von aktuellen Symptomen"
        ),
        VariationOption(
            value="post_exposure",
            instruction="Patient:in hatte kürzlich eine Risikoexposition",
        ),
        VariationOption(value="follow_up", instruction="Nachkontrolle nach vorheriger Diagnose"),
        VariationOption(
            value="partner_notification",
            instruction="Der Partner hat die Patient:in über eine mögliche Infektion informiert",
        ),
        VariationOption(
            value="first_time",
            instruction="Patient:in ist zum ersten Mal wegen eines STI-Anliegens da",
        ),
        VariationOption(
            value="prep_consultation",
            instruction="Beratung zu PrEP (Prä-Expositions-Prophylaxe)",
        ),
    ),
)

STYLE = CategoricalAxis(
    name="style",
    label="Gesprächsstil",
    options=(
        VariationOption(
            value="concise",
            instruction="knapp und effizient",
            metadata={
                "turn_range": (15, 25),
                "completeness": (
                    "Viele Felder sollten `not_discussed` sein — konzentriere dich auf "
                    "die wichtigsten 5-8 Themen (kurzes, effizientes Gespräch)."
                ),
            },
        ),
        VariationOption(
            value="detailed",
            instruction="ausführlich und gründlich",
            metadata={
                "turn_range": (30, 50),
                "completeness": (
                    "Die meisten Felder sollten ausgefüllt sein — ein gründliches, "
                    "ausführliches Gespräch (20-30 ausgefüllte Felder sind in Ordnung)."
                ),
            },
        ),
        VariationOption(
            value="hesitant_patient",
            instruction="Patient:in ist anfangs zögerlich und braucht Ermutigung",
            metadata={"turn_range": (20, 35)},
        ),
        VariationOption(
            value="open_patient",
            instruction="Patient:in ist sehr offen und teilt bereitwillig",
            metadata={"turn_range": (20, 35)},
        ),
        VariationOption(
            value="chaotic",
            instruction="Gespräch folgt keiner strengen Struktur, Patient:in schweift ab",
            metadata={
                "turn_range": (20, 35),
                "completeness": (
                    "Ausgewogene Mischung — manche Themen werden ausführlich besprochen, "
                    "andere übersprungen (10-15 ausgefüllte Felder)."
                ),
            },
        ),
        VariationOption(
            value="formal",
            instruction="sehr professionelle, strukturierte Gesprächsführung",
            metadata={"turn_range": (20, 35)},
        ),
        VariationOption(
            value="casual",
            instruction="lockere, vertrauensvolle Atmosphäre",
            metadata={"turn_range": (20, 35)},
        ),
    ),
)

DEMOGRAPHIC = CategoricalAxis(
    name="demographic",
    label="Patientenprofil",
    options=(
        VariationOption(value="msm", instruction="Mann, der Sex mit Männern hat (MSM)"),
        VariationOption(value="heterosexual", instruction="heterosexuell"),
        VariationOption(value="sex_worker", instruction="arbeitet in der Sexarbeit"),
        VariationOption(value="young_adult", instruction="junger Erwachsener (18-25 Jahre)"),
        VariationOption(value="middle_aged", instruction="mittleres Alter (26-50 Jahre)"),
        VariationOption(value="senior", instruction="älter (50+ Jahre)"),
        VariationOption(value="trans_person", instruction="trans Person"),
        VariationOption(value="non_binary", instruction="nicht-binäre Person"),
        VariationOption(value="migrant", instruction="Person mit Migrationshintergrund"),
    ),
)

HEALTH_LITERACY = CategoricalAxis(
    name="health_literacy",
    label="Gesundheitskompetenz",
    options=(
        VariationOption(
            value="low",
            instruction="gering — braucht einfache Erklärungen, keine Fachbegriffe",
        ),
        VariationOption(value="medium", instruction="durchschnittlich"),
        VariationOption(value="high", instruction="hoch — gut informiert, kennt Fachbegriffe"),
    ),
)

STRUCTURE = CategoricalAxis(
    name="structure",
    label="Gesprächsstruktur",
    options=(
        VariationOption(
            value="chronological",
            instruction="chronologisch (von den letzten Monaten in die Vergangenheit)",
        ),
        VariationOption(
            value="symptom_focused", instruction="beginnt mit und fokussiert auf Symptome"
        ),
        VariationOption(
            value="partner_focused",
            instruction="fokussiert auf Partnerhistorie und Sexualkontakte",
        ),
        VariationOption(
            value="risk_assessment", instruction="fokussiert auf Risikofaktoren und -abschätzung"
        ),
        VariationOption(
            value="patient_led",
            instruction="Patient:in führt das Gespräch, Arzt:Ärztin fragt ergänzend nach",
        ),
        VariationOption(
            value="doctor_led", instruction="Arzt:Ärztin führt systematisch durch die Anamnese"
        ),
    ),
)

SETTING = CategoricalAxis(
    name="setting",
    label="Setting",
    options=(
        VariationOption(
            value="sti_clinic",
            instruction="STI-Ambulanz / Gesundheitsamt, vertraulich und professionell",
            metadata={
                "location": "STI-Ambulanz / Gesundheitsamt",
                "atmosphere": "vertraulich und professionell",
            },
        ),
        VariationOption(
            value="gp_practice",
            instruction="Hausarztpraxis, vertraut und entspannt",
            metadata={"location": "Hausarztpraxis", "atmosphere": "vertraut und entspannt"},
        ),
        VariationOption(
            value="hospital",
            instruction="Krankenhaus-Ambulanz, routiniert und effizient",
            metadata={
                "location": "Krankenhaus-Ambulanz",
                "atmosphere": "routiniert und effizient",
            },
        ),
        VariationOption(
            value="checkpoint",
            instruction="Checkpoint (queere Gesundheitsberatung), offen und sex-positiv",
            metadata={
                "location": "Checkpoint (queere Gesundheitsberatung)",
                "atmosphere": "offen und sex-positiv",
            },
        ),
        VariationOption(
            value="university_clinic",
            instruction="Uniklinik / Spezialambulanz, akademisch und gründlich",
            metadata={
                "location": "Uniklinik / Spezialambulanz",
                "atmosphere": "akademisch und gründlich",
            },
        ),
    ),
)

DIALECT = CategoricalAxis(
    name="dialect",
    label="Sprachfärbung",
    options=(
        VariationOption(
            value="berlinerisch", instruction="Patient:in mit berlinerischem Einschlag"
        ),
        VariationOption(value="sueddeutsch", instruction="Patient:in mit süddeutschem Einschlag"),
        VariationOption(
            value="oesterreichisch", instruction="Patient:in mit österreichischem Einschlag"
        ),
    ),
    # Most consultations are in standard German; regional colour is the exception.
    skip_probability=0.5,
)

SMALL_TALK = FlagAxis(
    name="small_talk",
    instruction="Das Gespräch beginnt mit kurzem Small Talk.",
    probability=0.2,
)

EDUCATION = FlagAxis(
    name="education",
    instruction="Arzt:Ärztin erklärt STI-Risiken und Präventionsmöglichkeiten.",
    probability=0.3,
)

PATIENT_QUESTIONS = FlagAxis(
    name="patient_questions",
    instruction="Patient:in stellt eigene Fragen zu STIs, Tests oder Behandlungen.",
    probability=0.4,
)


def sti_variation_space() -> VariationSpace:
    """The full STI variation space.

    Returns:
        A space of seven categorical axes and three flags.

    Example:
        >>> space = sti_variation_space()
        >>> space.cardinality()
        1270080
    """
    return VariationSpace(
        categorical=(
            CONVERSATION_TYPE,
            STYLE,
            DEMOGRAPHIC,
            HEALTH_LITERACY,
            STRUCTURE,
            SETTING,
            DIALECT,
        ),
        flags=(SMALL_TALK, EDUCATION, PATIENT_QUESTIONS),
    )
