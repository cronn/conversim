"""Domain model invariants."""

from __future__ import annotations

import pytest

from conversim import Cast, Persona, Role, TokenUsage, TurnTarget, Utterance
from conversim.errors import ConfigurationError


def test_role_id_must_be_a_slug() -> None:
    Role(id="clinician-2", label="x")
    with pytest.raises(ValueError, match="slug"):
        Role(id="has space", label="x")
    with pytest.raises(ValueError, match="slug"):
        Role(id="", label="x")


def test_persona_name_cannot_contain_the_delimiter() -> None:
    with pytest.raises(ValueError, match="':'"):
        Persona(name="Dr: Meier", role="clinician")


def test_persona_name_must_be_single_line_and_non_empty() -> None:
    with pytest.raises(ValueError, match="single-line"):
        Persona(name="Two\nLines", role="clinician")
    with pytest.raises(ValueError, match="empty"):
        Persona(name="   ", role="clinician")


def test_persona_name_is_stripped() -> None:
    assert Persona(name="  Sam  ", role="agent").name == "Sam"


def test_one_persona_per_role() -> None:
    with pytest.raises(ValueError, match="exactly one persona"):
        Cast(
            personas=(
                Persona(name="A", role="agent"),
                Persona(name="B", role="agent"),
            )
        )


def test_persona_names_must_be_distinct_case_insensitively() -> None:
    with pytest.raises(ValueError, match="distinct"):
        Cast(personas=(Persona(name="Sam", role="a"), Persona(name="sam", role="b")))


def test_empty_cast_rejected() -> None:
    with pytest.raises(ValueError, match="at least one"):
        Cast(personas=())


def test_cast_lookup_by_role_and_name() -> None:
    cast = Cast(personas=(Persona(name="Sam", role="agent"), Persona(name="Kim", role="customer")))
    assert cast.by_role("agent").name == "Sam"
    assert cast.role_of("kim") == "customer"
    assert cast.role_of("Nobody") is None
    assert cast.names == ("Sam", "Kim")
    with pytest.raises(ConfigurationError, match="No persona plays"):
        cast.by_role("nonexistent")


def test_turn_target_must_be_ordered() -> None:
    assert TurnTarget(minimum=5, maximum=10).describe() == "5-10"
    assert TurnTarget(minimum=7, maximum=7).describe() == "7"
    with pytest.raises(ValueError, match=">= minimum"):
        TurnTarget(minimum=10, maximum=5)


def test_utterance_renders_back_to_script() -> None:
    assert (
        Utterance(index=0, speaker="Sam", role="a", text="Hi.", cue="warmly").to_script_line()
        == "Sam: (warmly) Hi."
    )
    assert Utterance(index=0, speaker="Sam", role="a", text="Hi.").to_script_line() == "Sam: Hi."


def test_token_usage_adds_up() -> None:
    total = TokenUsage(input_tokens=10, output_tokens=2) + TokenUsage(
        input_tokens=5, output_tokens=3
    )
    assert (total.input_tokens, total.output_tokens, total.total_tokens) == (15, 5, 20)
