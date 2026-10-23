"""German prompt builders for STI/HIV consultations.

Ported from the pipeline this library was extracted from, with one behavioural change:
every random choice draws from the seeded per-sample generator rather than the module-level
``random``. The original picked example snippets and realism hints with global randomness,
which meant that even with a seed set, two runs produced different prompts -- so a
"reproducible" dataset was reproducible in its ground truth and not in its dialogues.

The prompt is organised around a single tension. Every substantive ground-truth value
must be recoverable from the conversation, and the conversation must not sound like a
form being filled in. The instruction that resolves it is the one telling the model that
``not_discussed`` fields must go *unmentioned*: without it, models achieve coverage by
saying "we did not discuss that", which is both unnatural and useless as training data.
"""

from __future__ import annotations

import random

from ...prompts import DialoguePromptContext, GroundTruthPromptContext
from ...topics import TopicPlan

__all__ = ["StiDialoguePrompt", "StiGroundTruthPrompt"]

_PROFILE_HINTS: tuple[str, ...] = (
    "Berücksichtige typische Patientenprofile und anamnestische Muster.",
    "Vergib auch atypische Feldwerte, solange keine widersprüchlichen Angaben entstehen.",
    "Variiere die Angaben, um unterschiedliche Risikoprofile und Lebensstile abzubilden.",
    "Fokussiere auf ein konsistentes, aber nicht stereotypes Patientenprofil.",
)

_SYMPTOM_HINTS: tuple[str, ...] = (
    "Die Patient:in hat keine aktiven Symptome und kommt nur zur Vorsorge.",
    "Die Patient:in berichtet von aktuellen Symptomen.",
    "Die Patient:in hatte früher Symptome, aktuell aber keine.",
    "Die Patient:in ist unsicher, ob die Symptome relevant sind.",
)

_RISK_HINTS: tuple[str, ...] = (
    "Das Risikoprofil sollte niedrig sein (wenige Partner, konsequenter Schutz).",
    "Das Risikoprofil sollte moderat sein.",
    "Das Risikoprofil sollte erhöht sein (Risikoexposition, mehrere Partner).",
    "Das Risikoprofil ist gemischt (manche Risikofaktoren, manche Schutzfaktoren).",
)

_DEFAULT_COMPLETENESS: tuple[str, ...] = (
    "Viele Felder sollten `not_discussed` sein (kurzes Gespräch mit 5-8 Themen).",
    "Die meisten Felder sollten ausgefüllt sein (gründliches Gespräch mit 20-30 Themen).",
    "Ausgewogene Mischung aus ausgefüllten und nicht besprochenen Feldern (12-18 Themen).",
)

_CLINICIAN_LOGIC: tuple[str, ...] = (
    "strukturierten klinischen",
    "einfühlsamen und flexiblen",
    "eher unsystematischen",
)

_PATIENT_DISCLOSURE: tuple[str, ...] = (
    "direkt und strukturiert",
    "eher zögerlich und fragmentiert",
    "ausschweifend mit vielen Details",
    "knapp und nur auf Nachfrage",
)

_UTTERANCE_LENGTH: tuple[str, ...] = (
    "kürzere, prägnante",
    "längere, ausführliche",
    "gemischte kurze und lange",
)

_BACKGROUND_FREQUENCY: tuple[str, ...] = ("nur gelegentlich", "häufig", "auf Nachfrage")


class StiGroundTruthPrompt:
    """Builds the German prompt that fills the STI schema.

    Args:
        include_field_guide: Restate every field description in the prompt. Off by
            default because Gemini's ``response_schema`` already carries them; worth
            turning on for providers that ignore schema descriptions.
        completeness_axis: Variation axis whose selected option may carry
            ``metadata["completeness"]`` -- guidance on how much of the schema to fill.
            This is what keeps a "concise" consultation from being handed thirty facts
            to cover in fifteen turns.
    """

    def __init__(
        self,
        *,
        include_field_guide: bool = False,
        completeness_axis: str = "style",
    ) -> None:
        self.include_field_guide = include_field_guide
        self.completeness_axis = completeness_axis

    def build(self, context: GroundTruthPromptContext) -> str:
        """Render the ground-truth prompt."""
        rng = context.rng
        scenario = context.scenario
        schema = context.schema

        hints: list[str] = [rng.choice(_PROFILE_HINTS)]
        if rng.random() < 0.8:
            hints.append(rng.choice(_SYMPTOM_HINTS))
        if rng.random() < 0.8:
            hints.append(rng.choice(_RISK_HINTS))
        hints.extend(context.variation.instructions)

        completeness = context.variation.metadata(self.completeness_axis, "completeness")
        hints.append(completeness or rng.choice(_DEFAULT_COMPLETENESS))

        sections = [
            f"Du generierst ein realistisch ausgefülltes JSON-Objekt als Ergebnis einer "
            f"medizinischen STI-/HIV-Anamnese ({schema.name}).",
            "",
            "WICHTIG: Das Schema dient als Ground Truth für ein realistisches "
            "Arzt-Patient-Gespräch.",
            "→ Fülle nur so viele Felder aus, wie in einem echten Gespräch dieser Art "
            "besprochen würden.",
            "→ Nutze `not_discussed` großzügig für Themen, die nicht zur Sprache kämen.",
            "",
            "SETTING",
            f"- Sprache: {scenario.language}",
            f"- Kontext: {scenario.context}",
            f"- Thema/Fokus: {scenario.topic}",
            "",
            "VORGABEN FÜR DIESEN FALL",
        ]
        sections += [f"- {hint}" for hint in hints]

        sections += [
            "",
            "LEITFADEN",
            "- Sei realistisch: nicht jedes Gespräch deckt alle Themen ab.",
            "- Keine Fantasielisten und keine unnötige Ausschmückung "
            "(also nicht 20 Partner auflisten).",
            "- Keine widersprüchlichen Angaben: Status-Flags müssen zu den Listen passen.",
            "- Variiere Partnerzahlen (1-5 ist realistisch), Kontaktarten und "
            "Präventionsmaßnahmen.",
            "- Datumsfelder: nutze für Geburtsdaten das ISO-Format, für ungefähre "
            "Zeitpunkte die Teilfelder (Jahr/Monat/Tag), soweit bekannt.",
        ]

        if schema.sentinels:
            sections += [
                "",
                "RESERVIERTE WERTE",
                schema.sentinel_guide(),
            ]

        if self.include_field_guide:
            guide = schema.field_guide()
            if guide:
                sections += ["", "FELDREFERENZ", guide]

        return "\n".join(sections)


class StiDialoguePrompt:
    """Builds the German prompt that writes the consultation.

    Args:
        clinician_role: Role id of the clinician.
        patient_role: Role id of the patient.
        confidentiality_probability: Chance of asking for an explicit confidentiality
            note at the start.
        encouragement_probability: Chance of asking the clinician to invite free
            narration.
    """

    def __init__(
        self,
        *,
        clinician_role: str = "clinician",
        patient_role: str = "patient",
        confidentiality_probability: float = 0.3,
        encouragement_probability: float = 0.4,
    ) -> None:
        self.clinician_role = clinician_role
        self.patient_role = patient_role
        self.confidentiality_probability = confidentiality_probability
        self.encouragement_probability = encouragement_probability

    def build(self, context: DialoguePromptContext) -> str:
        """Render the dialogue prompt."""
        rng = context.rng
        scenario = context.scenario
        schema = context.schema

        clinician = context.cast.by_role(self.clinician_role)
        patient = context.cast.by_role(self.patient_role)

        clinician_extras = ""
        if rng.random() < self.confidentiality_probability:
            clinician_extras += " Erklärt zu Beginn die Vertraulichkeit."
        if rng.random() < self.encouragement_probability:
            clinician_extras += " Ermutigt zur freien Schilderung."

        length = scenario.turns.describe()

        sections = [
            f"Du schreibst ein realistisches {scenario.language.lower()}sprachiges "
            f"Arzt-Patient-Gespräch zur STI-/HIV-Anamnese.",
            "",
            "WICHTIG: Das Gespräch dient als Trainingsdaten für ein KI-System, das aus "
            "Gesprächen strukturierte Daten extrahieren soll.",
            "",
            "OBERSTES PRINZIP: NATÜRLICHKEIT VOR VOLLSTÄNDIGKEIT",
            "- Ein realistisches Gespräch klingt natürlich, nicht wie eine Checkliste.",
            "- Informationen werden organisch im Gesprächsfluss integriert, nicht abgefragt.",
            f"- Zielbereich: {length} Dialogzeilen (nicht deutlich mehr).",
            "",
            "SETTING",
            f"- Kontext: {scenario.context}",
            f"- Fokus: {scenario.topic}",
            f"- Sprache: {scenario.language}",
        ]

        if context.variation.instructions:
            sections += ["", "SPEZIFISCHE VORGABEN FÜR DIESES GESPRÄCH"]
            sections += [f"- {line}" for line in context.variation.instructions]

        sections += [
            "",
            "PERSONEN",
            f"- Arzt:Ärztin: {clinician.name}",
            f"    {clinician.description or 'Arzt:Ärztin in einer STI-Ambulanz.'}"
            f"{clinician_extras}",
            f"- Patient:in: {patient.name}",
            f"    {patient.description or 'Patient:in, die eine STI-Anamnese benötigt.'}",
            "",
            "GROUND TRUTH (faktische Grundlage)",
            "Folgendes Objekt enthält die Fakten, die im Gespräch vorkommen müssen:",
            "```json",
            context.ground_truth_json(),
            "```",
            "",
            "UMGANG MIT DEM GROUND TRUTH",
            "- Alle Informationen, die NICHT `not_discussed` sind, müssen im Gespräch vorkommen.",
            "- Felder mit `not_discussed` dürfen NICHT erwähnt werden. Genau dadurch "
            "bleibt das Gespräch natürlich kurz.",
            "- Felder mit `unknown`: die Patient:in weiß es nicht oder ist unsicher, "
            "und sagt das auch.",
            "- Felder mit `not_applicable`: trifft nicht zu, darf erwähnt werden.",
            "- Listen: alle Einträge müssen vorkommen.",
            "- Datumsangaben müssen erkennbar sein, auch wenn die Patient:in sie "
            "ungefähr formuliert.",
            "- Informationen dürfen explizit oder implizit vermittelt werden, müssen "
            "aber eindeutig ableitbar bleiben.",
            "- Widerspreche dem Ground Truth niemals.",
            "- Hintergrundinformationen und Emotionen dürfen frei ergänzt werden.",
        ]

        topic_section = self._render_topic_plan(context.topic_plan)
        if topic_section:
            sections += ["", topic_section]

        sections += [
            "",
            "REALISMUS & NATÜRLICHKEIT",
            f"- Die Ärztin bzw. der Arzt folgt einer {rng.choice(_CLINICIAN_LOGIC)} "
            f"Gesprächslogik.",
            f"- Die Patient:in gibt Informationen {rng.choice(_PATIENT_DISCLOSURE)} preis.",
            f"- Bevorzuge {rng.choice(_UTTERANCE_LENGTH)} Äußerungen.",
            f"- Die Patient:in gibt {rng.choice(_BACKGROUND_FREQUENCY)} "
            f"Hintergrundinformationen zu Lebensstil und Beziehungen.",
            "- Vermeide wiederholte Ja/Nein-Sequenzen ohne Kontext.",
            "- Die Ärztin bzw. der Arzt vermeidet künstlich klingende Zusammenfassungen.",
            "",
            "BEISPIELE FÜR GUTE GESPRÄCHSZEILEN",
            self._example(rng, clinician.name, patient.name),
            "",
            "QUALITÄTSKONTROLLE",
            "Stelle vor der Ausgabe sicher:",
            "✓ Das Gespräch ist natürlich und realistisch (wichtigstes Kriterium).",
            f"✓ Die Länge liegt im Zielbereich ({length} Turns).",
            "✓ Die Themen folgen ungefähr der vorgegebenen Reihenfolge, aber organisch integriert.",
            "✓ Alle Felder außer `not_discussed` wurden behandelt.",
            "✓ `not_discussed`-Felder wurden NICHT erwähnt.",
            "✓ Informationen sind extrahierbar, aber nicht künstlich formuliert.",
            "✓ Keine Checklisten-Atmosphäre.",
            "",
            "FORMAT",
            f"- Eine Person pro Zeile: `{clinician.name}: ...` oder `{patient.name}: ...`",
            "- Verwende ausschließlich diese beiden Namen, exakt so geschrieben.",
        ]

        if context.include_cues:
            sections.append(
                "- Optional ein kurzer Sprech-/Emotionshinweis in Klammern vor der "
                f"Äußerung, z.B. `{clinician.name}: (ruhig) ...`. Nur wenn es die "
                "Interpretation wirklich verbessert."
            )

        sections += [
            "- Keine Metaerklärungen, keine Szenenanweisungen, kein JSON in der Ausgabe.",
            f"- Mindestens {scenario.turns.minimum} Dialogzeilen.",
            "",
            "Starte jetzt und gib nur die Dialogzeilen aus:",
        ]

        # Referenced for clarity in the sentinel instructions above; assert the
        # assumption rather than silently drifting from the schema's conventions.
        if "not_discussed" not in schema.sentinels:
            sections.insert(
                0,
                "HINWEIS: Dieses Schema verwendet nicht `not_discussed`. "
                f"Reservierte Werte sind: {', '.join(sorted(schema.sentinels))}.",
            )

        return "\n".join(sections)

    @staticmethod
    def _render_topic_plan(plan: TopicPlan) -> str:
        if not plan:
            return ""
        lines = [
            "THEMATISCHE REIHENFOLGE (Leitfaden, nicht strikt)",
            "Behandle die folgenden Themen ungefähr in dieser Reihenfolge, passe sie "
            "aber natürlich in den Gesprächsfluss ein:",
            "",
        ]
        lines += [
            f"{position}. {description}"
            for position, description in enumerate(plan.descriptions, 1)
        ]
        lines += [
            "",
            "→ Natürlicher Gesprächsfluss geht vor strikter Reihenfolge.",
            "→ Themen dürfen sich überschneiden oder organisch ineinander übergehen.",
        ]
        return "\n".join(lines)

    @staticmethod
    def _example(rng: random.Random, clinician: str, patient: str) -> str:
        """Pick one worked example, so the model has a concrete target register.

        Varying the example matters: a single fixed snippet visibly anchors the phrasing
        of every conversation in the dataset.
        """
        options = (
            (
                f"    {clinician}: Danke für die Offenheit. Es ist übrigens nicht "
                f"ungewöhnlich, dass erste Symptome verzögert oder gar nicht auftreten. "
                f"Aktuell leben Sie ohne feste Beziehung, und Ihre Sexualkontakte der "
                f"letzten Monate hatten ausschließlich Männer zum Partner?\n"
                f"    {patient}: Genau, ich bin homosexuell, hatte in den letzten zwölf "
                f"Monaten drei verschiedene Männer. Aber keiner davon hatte meines "
                f"Wissens nach bekannte Infektionen. Ich habe mich aber noch nie auf HIV "
                f"oder Hepatitis testen lassen und war auch nie geimpft."
            ),
            (
                f"    {patient}: Ich hatte letzten Monat ungeschützten Verkehr und mache "
                f"mir jetzt Sorgen. Es war auf einer Party, und wir waren beide ziemlich "
                f"betrunken.\n"
                f"    {clinician}: Ich verstehe Ihre Sorge. Können Sie mir sagen, wann "
                f"genau das war und ob Sie inzwischen irgendwelche Symptome bemerkt haben?"
            ),
            (
                f"    {clinician}: Haben Sie in den letzten Monaten Anzeichen wie "
                f"Ausfluss, Brennen beim Wasserlassen oder Hautveränderungen im "
                f"Intimbereich bemerkt?\n"
                f"    {patient}: Also, brennen tut es schon manchmal, aber ich dachte, "
                f"das ist normal? Ausfluss... ja, eigentlich schon, aber ich wusste "
                f"nicht, dass das wichtig ist."
            ),
        )
        return rng.choice(options)
