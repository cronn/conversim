"""German STI/HIV history-taking schema.

The structure and the field names are the ones used in the research scripts this library
was extracted from.

Three design decisions are worth restating, because they were arrived at by iteration
rather than by taste:

* **No free-text fields.** An earlier version had them. They are inconsistent to
  generate, and worse, near-impossible to score: two extractions can be equally
  correct and share no substring.
* **Coded enums over open values.** Every categorical field enumerates its options, so
  the schema itself teaches a model what answers exist. This is what makes small local
  models viable at extraction.
* **Medium granularity.** A more detailed version carrying per-field evidence pointers
  and confidence values proved too large for stable results: token usage rose and
  smaller models stopped following the structure reliably.

Every categorical field carries ``not_discussed``, and this is load-bearing rather than
defensive. It is the difference between "the patient has no symptoms" and "symptoms
never came up", and it is what lets a generated conversation stay plausibly short: the
dialogue prompt is instructed never to mention a field marked that way.
"""

from __future__ import annotations

from datetime import date
from enum import Enum

from pydantic import BaseModel, Field

__all__ = [
    "AdministrativeSex",
    "AnatomyPart",
    "ApproximateDate",
    "CondomUsePattern",
    "GenderIdentity",
    "HivPartnerStatus",
    "HivRiskWindow",
    "HormoneTherapyStatus",
    "PartnerAndExposure",
    "PatientBasics",
    "PracticeType",
    "Prevention",
    "ResultDeliveryMethod",
    "SampleSite",
    "SexAtBirth",
    "StiAnamnesis",
    "StiHistory",
    "StiHistoryItem",
    "SubstanceFrequency",
    "SubstanceUse",
    "SymptomSite",
    "SymptomType",
    "Symptoms",
    "TestRecommendation",
    "TestType",
    "TriState",
    "VaccineStatus",
]


class ApproximateDate(BaseModel):
    """A date that may be only partly known.

    Patients date events approximately ("some time last spring"), so a plain
    ``date`` would force the generator to invent precision that the conversation
    cannot support and an extractor cannot recover.
    """

    year: int | None = Field(default=None, description="Jahr, falls erkennbar.")
    month: int | None = Field(default=None, description="Monat (1-12), falls erkennbar.")
    day: int | None = Field(default=None, description="Tag (1-31), falls erkennbar.")

    def __str__(self) -> str:
        parts = [str(part) for part in (self.day, self.month, self.year) if part is not None]
        return ".".join(parts) if parts else "unbekannt"


class TriState(str, Enum):
    """Yes/no where "never came up" must stay distinguishable from "no"."""

    yes = "yes"
    no = "no"
    not_discussed = "not_discussed"


class AdministrativeSex(str, Enum):
    """Administrative sex, only when explicitly stated (billing, statistics)."""

    weiblich = "weiblich"
    maennlich = "maennlich"
    divers = "divers"
    not_discussed = "not_discussed"


class SexAtBirth(str, Enum):
    """Sex assigned at birth."""

    weiblich = "weiblich"
    maennlich = "maennlich"
    inter_divers = "inter/divers"
    not_discussed = "not_discussed"


class GenderIdentity(str, Enum):
    """Gender identity, for correct address and patient-centred care."""

    frau = "Frau"
    mann = "Mann"
    nicht_binaer = "nicht-binär"
    trans_frau = "trans Frau"
    trans_mann = "trans Mann"
    divers = "divers"
    not_discussed = "not_discussed"


class AnatomyPart(str, Enum):
    """Anatomy clinically relevant to STI screening and sampling site choice."""

    penis = "penis"
    scrotum = "scrotum"
    prostate = "prostate"
    vagina = "vagina"
    cervix = "cervix"
    uterus = "uterus"
    rectum = "rectum"
    neovagina = "neovagina"
    neopenis = "neopenis"
    not_discussed = "not_discussed"


class HormoneTherapyStatus(str, Enum):
    """Hormone therapy status."""

    no = "no"
    yes_current = "yes_current"
    yes_past = "yes_past"
    not_discussed = "not_discussed"


class PracticeType(str, Enum):
    """Sexual practices, coded. Determines which sites need sampling."""

    vaginal = "vaginal"
    anal_insertive = "anal_insertive"
    anal_receptive = "anal_receptive"
    oral_insertive = "oral_insertive"
    oral_receptive = "oral_receptive"
    sexspielzeug = "sexspielzeug"
    other_coded = "other_coded"


class CondomUsePattern(str, Enum):
    """Barrier-protection frequency."""

    always = "always"
    mostly = "mostly"
    sometimes = "sometimes"
    never = "never"
    not_discussed = "not_discussed"


class HivPartnerStatus(str, Enum):
    """Partner HIV status as known to the patient.

    ``positive_on_effective_art`` is kept distinct from
    ``positive_not_suppressed_or_unclear`` because the two imply very different
    transmission risk, and collapsing them would lose the clinically decisive part.
    """

    negative_known = "negative_known"
    positive_on_effective_art = "positive_on_effective_art"
    positive_not_suppressed_or_unclear = "positive_not_suppressed_or_unclear"
    unknown = "unknown"
    not_discussed = "not_discussed"


class SymptomSite(str, Enum):
    """Where symptoms are located."""

    mouth_throat = "mouth_throat"
    vagina = "vagina"
    penis_urethra = "penis_urethra"
    anal_rectal = "anal_rectal"
    skin = "skin"
    systemic = "systemic"
    not_discussed = "not_discussed"


class SymptomType(str, Enum):
    """Kind of symptom."""

    discharge = "discharge"
    pain_burning = "pain_burning"
    itch = "itch"
    ulcers_vesicles = "ulcers_vesicles"
    warts = "warts"
    rash = "rash"
    fever = "fever"
    lymph_nodes = "lymph_nodes"
    sore_throat = "sore_throat"
    dysuria = "dysuria"
    rectal_bleeding = "rectal_bleeding"
    not_discussed = "not_discussed"


class StiHistoryItem(str, Enum):
    """Previously diagnosed infections."""

    chlamydia = "chlamydia"
    gonorrhea = "gonorrhea"
    syphilis = "syphilis"
    trichomonas = "trichomonas"
    mycoplasma_genitalium = "mycoplasma_genitalium"
    hsv = "hsv"
    hpv_warts = "hpv_warts"
    hepatitis_a = "hepatitis_a"
    hepatitis_b = "hepatitis_b"
    hepatitis_c = "hepatitis_c"
    hiv = "hiv"
    not_discussed = "not_discussed"


class VaccineStatus(str, Enum):
    """Vaccination status."""

    complete = "complete"
    incomplete = "incomplete"
    unvaccinated = "unvaccinated"
    not_discussed = "not_discussed"


class SubstanceFrequency(str, Enum):
    """Coarse substance-use frequency."""

    never = "never"
    occasional = "occasional"
    regular = "regular"
    frequent = "frequent"
    not_discussed = "not_discussed"


class ResultDeliveryMethod(str, Enum):
    """Agreed channel for delivering results."""

    phone = "phone"
    sms = "sms"
    email = "email"
    patient_portal = "patient_portal"
    in_person = "in_person"
    letter = "letter"
    not_discussed = "not_discussed"


class TestType(str, Enum):
    """Tests that may be recommended."""

    hiv = "hiv"
    syphilis = "syphilis"
    chlamydia = "chlamydia"
    gonorrhea = "gonorrhea"
    hepatitis_a = "hepatitis_a"
    hepatitis_b = "hepatitis_b"
    hepatitis_c = "hepatitis_c"
    not_discussed = "not_discussed"


class SampleSite(str, Enum):
    """Planned sampling or swab sites."""

    blood = "blood"
    urine = "urine"
    vaginal_swab = "vaginal_swab"
    cervical_swab = "cervical_swab"
    urethral_swab = "urethral_swab"
    rectal_swab = "rectal_swab"
    throat_swab = "throat_swab"
    lesion_swab = "lesion_swab"
    not_discussed = "not_discussed"


class PatientBasics(BaseModel):
    """Demographics and anatomy relevant to screening."""

    date_of_birth: date | None = Field(
        default=None, description="Nur wenn genannt und sicher datierbar."
    )
    administrative_sex: AdministrativeSex = Field(
        default=AdministrativeSex.not_discussed,
        description="Rein administrativ/rechtlich, nur falls erwähnt.",
    )
    sex_at_birth: SexAtBirth = Field(
        default=SexAtBirth.not_discussed,
        description="Bei Geburt zugewiesen. Nur falls erwähnt.",
    )
    gender_identity: GenderIdentity = Field(
        default=GenderIdentity.not_discussed,
        description="Geschlechtsidentität (Ansprache/Patientenzentrierung).",
    )
    anatomy_present: list[AnatomyPart] = Field(
        default_factory=list,
        description="Welche Strukturen sind vorhanden? Nur befüllen, wenn erwähnt oder ableitbar.",
    )
    hormone_therapy_status: HormoneTherapyStatus = Field(
        default=HormoneTherapyStatus.not_discussed,
        description="Hormontherapie (relevant z.B. für Schleimhautzustand).",
    )
    gender_affirming_surgery_status: TriState = Field(
        default=TriState.not_discussed,
        description="Gab es geschlechtsangleichende Operationen? Nur wenn erwähnt.",
    )
    pregnancy_possible: TriState = Field(
        default=TriState.not_discussed,
        description=(
            "Nur setzen, wenn explizit thematisiert oder eindeutig aus der Anatomie genannt."
        ),
    )
    currently_pregnant: TriState = Field(
        default=TriState.not_discussed, description="Nur setzen, wenn genannt."
    )


class PartnerAndExposure(BaseModel):
    """Partners, practices, protection and partner status."""

    last_sexual_contact_date_approx: ApproximateDate | None = Field(
        default=None, description="Ungefähres Datum des letzten Sexualkontakts."
    )
    partners_last_3_months: int | None = Field(
        default=None,
        description=(
            "Anzahl verschiedener Sexualpartner:innen in den letzten 3 Monaten. "
            "Nur als Zahl setzen, wenn explizit genannt."
        ),
    )
    practices_last_3_months: list[PracticeType] = Field(
        default_factory=list, description="Praktiken der letzten 3 Monate (kodiert)."
    )
    condom_use_vaginal: CondomUsePattern = Field(
        default=CondomUsePattern.not_discussed, description="Kondomnutzung bei vaginalem Verkehr."
    )
    condom_use_anal: CondomUsePattern = Field(
        default=CondomUsePattern.not_discussed, description="Kondomnutzung bei analem Verkehr."
    )
    condom_use_oral: CondomUsePattern = Field(
        default=CondomUsePattern.not_discussed, description="Barrierenutzung bei oralem Verkehr."
    )
    partner_hiv_status_known: HivPartnerStatus = Field(
        default=HivPartnerStatus.not_discussed,
        description="HIV-Status der Partner:innen, soweit dem Patienten bekannt.",
    )
    recent_sti_in_partners: TriState = Field(
        default=TriState.not_discussed,
        description="Kürzliche STI-Diagnose bei einem Partner. Nur setzen, wenn erwähnt.",
    )


class Symptoms(BaseModel):
    """Current complaints, their location and onset."""

    symptom_onset_date_approx: ApproximateDate | None = Field(
        default=None, description="Ungefähres Datum des Symptombeginns. Nur wenn genannt."
    )
    symptom_sites: list[SymptomSite] = Field(
        default_factory=list, description="Lokalisation der Symptome (kodiert)."
    )
    symptom_types: list[SymptomType] = Field(
        default_factory=list, description="Art der Symptome (kodiert)."
    )


class StiHistory(BaseModel):
    """Previous diagnoses and their timing."""

    prior_sti_items: list[StiHistoryItem] = Field(
        default_factory=list, description="Welche STIs traten in der Anamnese auf (kodiert)."
    )
    last_sti_diagnosis_date_approx: ApproximateDate | None = Field(
        default=None, description="Ungefähres Datum der letzten STI-Diagnose, falls genannt."
    )


class Prevention(BaseModel):
    """Vaccination status, PrEP and counselling."""

    vaccination_hepatitis_a: VaccineStatus = Field(
        default=VaccineStatus.not_discussed, description="Impfstatus Hepatitis A."
    )
    vaccination_hepatitis_b: VaccineStatus = Field(
        default=VaccineStatus.not_discussed, description="Impfstatus Hepatitis B."
    )
    vaccination_hpv: VaccineStatus = Field(
        default=VaccineStatus.not_discussed, description="Impfstatus HPV."
    )
    prep_current_use: TriState = Field(
        default=TriState.not_discussed,
        description="Wird aktuell eine HIV-PrEP eingenommen? Nur wenn genannt.",
    )
    prep_offered_or_discussed: TriState = Field(
        default=TriState.not_discussed, description="Wurde PrEP angeboten oder besprochen?"
    )
    condom_counseling_done: TriState = Field(
        default=TriState.not_discussed, description="Erfolgte eine Kondom-/Barrierenberatung?"
    )


class SubstanceUse(BaseModel):
    """Alcohol and drug use, only where discussed."""

    alcohol_use_frequency: SubstanceFrequency = Field(
        default=SubstanceFrequency.not_discussed, description="Alkoholkonsum (grobe Frequenz)."
    )
    recreational_drug_use_frequency: SubstanceFrequency = Field(
        default=SubstanceFrequency.not_discussed,
        description="Konsum anderer Substanzen (grobe Frequenz).",
    )
    injection_drug_use: SubstanceFrequency = Field(
        default=SubstanceFrequency.not_discussed, description="Intravenöser Konsum."
    )


class HivRiskWindow(BaseModel):
    """Time-critical HIV assessment.

    The 72-hour and six-week windows are here because they change what happens next:
    post-exposure prophylaxis is only an option inside the former, and acute HIV
    infection is only a question inside the latter.
    """

    relevant_hiv_exposure_suspected: TriState = Field(
        default=TriState.not_discussed,
        description=(
            "Besteht laut Gespräch ein relevantes aktuelles HIV-Übertragungsrisiko? "
            "Nur bei expliziter Klärung setzen."
        ),
    )
    last_potential_hiv_exposure_date_approx: ApproximateDate | None = Field(
        default=None,
        description="Ungefähres Datum der letzten potentiell relevanten HIV-Exposition.",
    )
    exposure_within_72h: TriState = Field(
        default=TriState.not_discussed,
        description="Mögliche Exposition innerhalb der letzten 72h (PEP-Fenster)?",
    )
    pep_considered: TriState = Field(
        default=TriState.not_discussed, description="Wurde PEP erwogen oder angeboten?"
    )
    exposure_within_6_weeks: TriState = Field(
        default=TriState.not_discussed,
        description="Exposition innerhalb der letzten 6 Wochen (akute HIV-Frage relevant)?",
    )
    acute_hiv_symptoms_present: TriState = Field(
        default=TriState.not_discussed,
        description=(
            "Hinweise auf eine akute HIV-Infektion (z.B. Fieber, Exanthem, "
            "Lymphknotenschwellung, Pharyngitis)?"
        ),
    )


class TestRecommendation(BaseModel):
    """Planned diagnostics and how results are communicated."""

    tests_recommended: list[TestType] = Field(
        default_factory=list, description="Welche Tests werden empfohlen (kodiert)."
    )
    sampling_sites: list[SampleSite] = Field(
        default_factory=list,
        description="Welche Proben- bzw. Abstrichorte sind geplant (kodiert).",
    )
    result_delivery_method: ResultDeliveryMethod = Field(
        default=ResultDeliveryMethod.not_discussed,
        description="Vereinbarter Weg der Ergebnisübermittlung.",
    )
    result_delivery_timeframe_days: int | None = Field(
        default=None,
        description=(
            "Vereinbarter Zeithorizont bis zur Ergebnisrückmeldung in Tagen. "
            "Nur als Zahl, wenn genannt."
        ),
    )


class StiAnamnesis(BaseModel):
    """A complete STI/HIV history-taking record.

    Section names are German, as in the research scripts this library was extracted
    from.
    """

    patient: PatientBasics = Field(
        default_factory=PatientBasics, description="Basisdaten und Anatomie."
    )
    partner_und_praktiken: PartnerAndExposure = Field(
        default_factory=PartnerAndExposure,
        description="Partner:innen, Praktiken, Kondome, Partnerstatus.",
    )
    symptome: Symptoms = Field(
        default_factory=Symptoms, description="Symptome inklusive Lokalisation."
    )
    sti_vorgeschichte: StiHistory = Field(
        default_factory=StiHistory, description="STI-Vorgeschichte (kodiert)."
    )
    praevention: Prevention = Field(
        default_factory=Prevention, description="Prävention: Impfstatus, PrEP, Beratung."
    )
    substanzen: SubstanceUse = Field(
        default_factory=SubstanceUse, description="Alkohol und Drogen, nur wenn thematisiert."
    )
    hiv_risiko_fenster: HivRiskWindow = Field(
        default_factory=HivRiskWindow,
        description="HIV-Risikofenster inklusive PEP und Hinweisen auf akute Infektion.",
    )
    diagnostik_und_kommunikation: TestRecommendation = Field(
        default_factory=TestRecommendation,
        description="Geplante Diagnostik und Ergebnisübermittlung.",
    )
