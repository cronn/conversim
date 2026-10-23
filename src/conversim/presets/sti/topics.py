"""Topic categories for STI/HIV consultations.

Each category names the schema fields it covers. Those paths are checked against the
schema when the planner is built, which is not a formality: in the pipeline this
library was extracted from, five categories referenced field names the schema did not
have. Four therefore never activated -- symptoms, STI history, test history and
prevention were silently absent from every topic plan across a 400-conversation
corpus -- and one compared a missing key against a sentinel string, so it activated
unconditionally. Nothing failed, nothing logged, and the guidance was simply wrong.

Note on three-valued fields: a value of ``no`` *does* activate its category, unlike
``not_discussed``. That distinction is the reason :class:`~conversim.presets.sti.schema.TriState`
exists -- "asked, and the answer was no" is content, and a conversation can convey it.
"""

from __future__ import annotations

from ...schema import SchemaSpec
from ...topics import PriorityRule, SchemaTopicPlanner, TopicCategory
from .schema import StiAnamnesis

__all__ = ["STI_PRIORITY_RULES", "STI_TOPIC_CATEGORIES", "sti_topic_planner"]

STI_TOPIC_CATEGORIES: tuple[TopicCategory, ...] = (
    TopicCategory(
        name="greeting",
        description="Vorstellung, Grund des Besuchs, Hinweis auf Vertraulichkeit",
        priority=1,
        position="early",
        always_include=True,
    ),
    TopicCategory(
        name="symptoms",
        description="Aktuelle Beschwerden: Art, Lokalisation, Beginn, Verlauf",
        field_paths=(
            "symptome.symptom_sites",
            "symptome.symptom_types",
            "symptome.symptom_onset_date_approx",
        ),
        priority=2,
    ),
    TopicCategory(
        name="social_context",
        description=(
            "Lebenskontext der Patient:in: Beziehungssituation, wie und wo Partner "
            "kennengelernt wurden, emotionale Äußerungen. Dient als natürliches "
            "Gesprächsrauschen und wird nicht als Ground Truth extrahiert"
        ),
        priority=5,
        always_include=True,
        # Not always present, and its position in the conversation genuinely varies:
        # this is the category that keeps dialogues from all sounding alike.
        inclusion_probability=0.7,
        priority_jitter=(3, 7),
    ),
    TopicCategory(
        name="patient_basics",
        description="Basisdaten: Alter bzw. Geburtsdatum, Geschlecht, Anatomie, Hormontherapie",
        field_paths=(
            "patient.date_of_birth",
            "patient.administrative_sex",
            "patient.sex_at_birth",
            "patient.gender_identity",
            "patient.anatomy_present",
            "patient.hormone_therapy_status",
            "patient.gender_affirming_surgery_status",
        ),
        priority=3,
    ),
    TopicCategory(
        name="recent_contacts",
        description="Letzte Sexualkontakte: wann, wie viele verschiedene Partner:innen",
        field_paths=(
            "partner_und_praktiken.last_sexual_contact_date_approx",
            "partner_und_praktiken.partners_last_3_months",
        ),
        priority=4,
    ),
    TopicCategory(
        name="practices",
        description="Sexualpraktiken (vaginal, anal, oral, Spielzeug) und Details dazu",
        field_paths=("partner_und_praktiken.practices_last_3_months",),
        priority=5,
    ),
    TopicCategory(
        name="condom_use",
        description="Schutzmaßnahmen bei den verschiedenen Praktiken",
        field_paths=(
            "partner_und_praktiken.condom_use_vaginal",
            "partner_und_praktiken.condom_use_anal",
            "partner_und_praktiken.condom_use_oral",
        ),
        priority=6,
    ),
    TopicCategory(
        name="partner_status",
        description="HIV-Status der Partner:innen, bekannte Infektionen im Umfeld",
        field_paths=(
            "partner_und_praktiken.partner_hiv_status_known",
            "partner_und_praktiken.recent_sti_in_partners",
        ),
        priority=6,
    ),
    TopicCategory(
        name="sti_history",
        description="Frühere STI-Diagnosen und deren Behandlung",
        field_paths=("sti_vorgeschichte.prior_sti_items",),
        priority=7,
    ),
    TopicCategory(
        name="test_history",
        description="Wann wurde zuletzt getestet, mit welchem Ergebnis",
        field_paths=("sti_vorgeschichte.last_sti_diagnosis_date_approx",),
        priority=7,
    ),
    TopicCategory(
        name="pregnancy",
        description="Schwangerschaftsmöglichkeit und aktueller Status",
        field_paths=("patient.pregnancy_possible", "patient.currently_pregnant"),
        priority=7,
    ),
    TopicCategory(
        name="prevention",
        description="Impfstatus (HPV, Hepatitis A/B), PrEP-Nutzung, Beratung zu Barrieremethoden",
        field_paths=(
            "praevention.vaccination_hepatitis_a",
            "praevention.vaccination_hepatitis_b",
            "praevention.vaccination_hpv",
            "praevention.prep_current_use",
            "praevention.prep_offered_or_discussed",
            "praevention.condom_counseling_done",
        ),
        priority=8,
    ),
    TopicCategory(
        name="substance_use",
        description="Alkohol- und Drogenkonsum im Zusammenhang mit Sexualverhalten",
        field_paths=(
            "substanzen.alcohol_use_frequency",
            "substanzen.recreational_drug_use_frequency",
            "substanzen.injection_drug_use",
        ),
        priority=8,
    ),
    TopicCategory(
        name="hiv_risk_window",
        description=(
            "HIV-Risikofenster: Exposition innerhalb 72h bzw. 6 Wochen, PEP, "
            "Hinweise auf eine akute Infektion"
        ),
        field_paths=(
            "hiv_risiko_fenster.relevant_hiv_exposure_suspected",
            "hiv_risiko_fenster.last_potential_hiv_exposure_date_approx",
            "hiv_risiko_fenster.exposure_within_72h",
            "hiv_risiko_fenster.pep_considered",
            "hiv_risiko_fenster.exposure_within_6_weeks",
            "hiv_risiko_fenster.acute_hiv_symptoms_present",
        ),
        priority=9,
    ),
    TopicCategory(
        name="planned_tests",
        description=(
            "Welche Tests sind geplant, welche Abstriche, wie werden Ergebnisse übermittelt"
        ),
        field_paths=(
            "diagnostik_und_kommunikation.tests_recommended",
            "diagnostik_und_kommunikation.sampling_sites",
            "diagnostik_und_kommunikation.result_delivery_method",
            "diagnostik_und_kommunikation.result_delivery_timeframe_days",
        ),
        priority=10,
        position="late",
    ),
)
"""The catalogue of topics an STI consultation may cover."""


STI_PRIORITY_RULES: tuple[PriorityRule, ...] = (
    # The reason for the visit dictates what gets addressed first.
    PriorityRule(axis="conversation_type", value="symptomatic", priorities={"symptoms": 1}),
    PriorityRule(
        axis="conversation_type",
        value="post_exposure",
        priorities={"hiv_risk_window": 2, "recent_contacts": 2},
    ),
    PriorityRule(
        axis="conversation_type",
        value="prep_consultation",
        priorities={"prevention": 3, "hiv_risk_window": 3},
    ),
    # Discourse structure reorders the middle of the consultation.
    PriorityRule(axis="structure", value="symptom_focused", priorities={"symptoms": 1}),
    PriorityRule(
        axis="structure",
        value="partner_focused",
        priorities={"recent_contacts": 2, "partner_status": 3},
    ),
    PriorityRule(
        axis="structure",
        value="risk_assessment",
        priorities={"practices": 3, "condom_use": 3, "sti_history": 4},
    ),
    PriorityRule(
        axis="structure",
        value="chronological",
        priorities={
            "recent_contacts": 2,
            "practices": 3,
            "sti_history": 7,
            "test_history": 8,
        },
    ),
)
"""Variation-dependent reorderings of the topic catalogue."""


def sti_topic_planner(schema: SchemaSpec | None = None) -> SchemaTopicPlanner:
    """Build the STI topic planner.

    Args:
        schema: Schema to validate field paths against. Defaults to
            :class:`~conversim.presets.sti.schema.StiAnamnesis`.

    Returns:
        A planner over the STI topic catalogue and priority rules.

    Raises:
        SchemaError: if a category references a field the schema lacks. Passing a
            modified schema here is therefore a cheap way to find out whether the
            topic catalogue still matches it.
    """
    return SchemaTopicPlanner(
        schema=schema or SchemaSpec(StiAnamnesis),
        categories=STI_TOPIC_CATEGORIES,
        priority_rules=STI_PRIORITY_RULES,
    )
