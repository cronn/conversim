"""Scenario derivation for STI/HIV consultations."""

from __future__ import annotations

import random

from ...models import Scenario, TurnTarget
from ...variation import VariationConfig

__all__ = ["StiScenarioBuilder"]


class StiScenarioBuilder:
    """Derives the consultation's setting and length from the sampled variation.

    Both come from option metadata rather than from separate configuration: the
    ``setting`` axis carries a location and an atmosphere, and the ``style`` axis
    carries a turn range. Reading them here is what keeps the axes as the single place
    diversity is described, instead of having length and setting drift apart from the
    style they are supposed to belong to.

    Args:
        topic: Consultation focus, as stated in prompts.
        language: Target language name.
        default_turns: Fallback length when the style axis carries no range.
        setting_axis: Axis supplying ``metadata["location"]`` and
            ``metadata["atmosphere"]``.
        style_axis: Axis supplying ``metadata["turn_range"]``.
        fallback_context: Context used when the setting axis was skipped or unset.
    """

    def __init__(
        self,
        *,
        topic: str = "STI-/HIV-Anamnese",
        language: str = "Deutsch",
        default_turns: TurnTarget | None = None,
        setting_axis: str = "setting",
        style_axis: str = "style",
        fallback_context: str = "STI-Ambulanz, vertraulich und professionell.",
    ) -> None:
        self.topic = topic
        self.language = language
        self.default_turns = default_turns or TurnTarget(minimum=20, maximum=35)
        self.setting_axis = setting_axis
        self.style_axis = style_axis
        self.fallback_context = fallback_context

    def build(self, *, variation: VariationConfig, rng: random.Random) -> Scenario:
        """Build the scenario for one consultation."""
        location = variation.metadata(self.setting_axis, "location")
        atmosphere = variation.metadata(self.setting_axis, "atmosphere")
        if location:
            context = f"{location}." if not atmosphere else f"{location}. Atmosphäre: {atmosphere}."
        else:
            context = self.fallback_context

        turns = self.default_turns
        span = variation.metadata(self.style_axis, "turn_range")
        if span is not None:
            minimum, maximum = span
            turns = TurnTarget(minimum=int(minimum), maximum=int(maximum))

        return Scenario(
            topic=self.topic,
            context=context,
            language=self.language,
            turns=turns,
        )
