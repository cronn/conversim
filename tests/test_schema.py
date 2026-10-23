"""Schema introspection: sentinels, substantive values, field paths."""

from __future__ import annotations

import pytest

from conversim import SchemaSpec
from conversim.errors import SchemaError

from .conftest import Ticket


def test_sentinels_are_not_substantive(ticket_schema: SchemaSpec) -> None:
    assert not ticket_schema.is_substantive("not_discussed")
    assert not ticket_schema.is_substantive("unknown")
    assert not ticket_schema.is_substantive("not_applicable")
    assert ticket_schema.is_substantive("printer offline")


@pytest.mark.parametrize(
    ("value", "expected"),
    [
        (None, False),
        ("", False),
        ("   ", False),
        ([], False),
        ({}, False),
        (False, False),
        (True, True),
        (0, True),  # a count of zero is a stated fact, unlike a bare "no"
        (0.0, True),
        (["not_discussed"], False),
        (["not_discussed", "real"], True),
        ({"a": None, "b": "not_discussed"}, False),
    ],
)
def test_substantive_rule(ticket_schema: SchemaSpec, value: object, expected: bool) -> None:
    assert ticket_schema.is_substantive(value) is expected


def test_default_instance_carries_no_content(ticket_schema: SchemaSpec) -> None:
    assert not ticket_schema.is_substantive(Ticket())


def test_valid_field_paths_pass(ticket_schema: SchemaSpec) -> None:
    for path in ("reference", "problem", "problem.summary", "resolution.actions"):
        ticket_schema.validate_field_path(path)


def test_invalid_field_path_names_alternatives(ticket_schema: SchemaSpec) -> None:
    with pytest.raises(SchemaError) as error:
        ticket_schema.validate_field_path("problem.sevrity")
    message = str(error.value)
    assert "sevrity" in message
    assert "severity" in message  # the fix is in the error itself


def test_field_path_cannot_descend_into_a_scalar(ticket_schema: SchemaSpec) -> None:
    with pytest.raises(SchemaError):
        ticket_schema.validate_field_path("reference.nope")


def test_empty_field_path_rejected(ticket_schema: SchemaSpec) -> None:
    with pytest.raises(SchemaError):
        ticket_schema.validate_field_path("")


def test_resolve_reads_nested_values(ticket_schema: SchemaSpec) -> None:
    data = {"problem": {"summary": "printer offline"}, "resolution": {"actions": ["a", "b"]}}
    assert ticket_schema.resolve(data, "problem.summary") == "printer offline"
    assert ticket_schema.resolve(data, "resolution.actions") == ["a", "b"]
    assert ticket_schema.resolve(data, "problem.missing") is None
    assert ticket_schema.resolve(data, "absent.path") is None


def test_resolve_accepts_a_model_instance(ticket_schema: SchemaSpec) -> None:
    instance = Ticket.model_validate({"problem": {"summary": "hello"}})
    assert ticket_schema.resolve(instance, "problem.summary") == "hello"


def test_has_content_combines_paths(ticket_schema: SchemaSpec) -> None:
    data = {"problem": {"summary": "not_discussed", "first_seen": "Tuesday"}}
    assert ticket_schema.has_content(data, ["problem.summary", "problem.first_seen"])
    assert not ticket_schema.has_content(data, ["problem.summary"])


def test_field_guide_lists_documented_fields(ticket_schema: SchemaSpec) -> None:
    guide = ticket_schema.field_guide()
    assert "- problem.summary: What is failing." in guide
    assert "- resolution.actions: Agreed next steps." in guide


def test_sentinel_guide_includes_documentation() -> None:
    spec = SchemaSpec(Ticket, sentinel_docs={"not_discussed": "Never came up."})
    assert "`not_discussed`: Never came up." in spec.sentinel_guide()


def test_documenting_a_non_sentinel_is_rejected() -> None:
    with pytest.raises(SchemaError):
        SchemaSpec(Ticket, sentinel_docs={"nonsense": "..."})


def test_non_model_schema_is_rejected() -> None:
    with pytest.raises(SchemaError):
        SchemaSpec(dict)  # type: ignore[arg-type]
