"""Dialogue parsing: the messy shapes models actually return."""

from __future__ import annotations

from conversim import Cast, DialogueParser

from .conftest import KIM, SAM


def parser(**kwargs: object) -> DialogueParser:
    return DialogueParser(cast=Cast(personas=(SAM, KIM)), **kwargs)  # type: ignore[arg-type]


def test_plain_script_parses() -> None:
    dialogue = parser().parse("Sam: Hello.\nKim: Hi there.")
    assert [(u.speaker, u.role, u.text) for u in dialogue.utterances] == [
        ("Sam", "agent", "Hello."),
        ("Kim", "customer", "Hi there."),
    ]
    assert dialogue.issues == ()


def test_indices_are_sequential() -> None:
    dialogue = parser().parse("Sam: a\nKim: b\nSam: c")
    assert [u.index for u in dialogue.utterances] == [0, 1, 2]


def test_cues_are_captured_and_stripped() -> None:
    dialogue = parser().parse("Sam: (warmly) Good morning.")
    assert dialogue.utterances[0].cue == "warmly"
    assert dialogue.utterances[0].text == "Good morning."


def test_cues_stay_inline_when_capture_is_off() -> None:
    dialogue = parser(capture_cues=False).parse("Sam: (warmly) Good morning.")
    assert dialogue.utterances[0].cue is None
    assert dialogue.utterances[0].text == "(warmly) Good morning."


def test_markdown_emphasis_and_bullets_are_tolerated() -> None:
    dialogue = parser().parse("**Sam:** Hello.\n- Kim: Hi.\n1. Sam: Again.")
    assert [u.speaker for u in dialogue.utterances] == ["Sam", "Kim", "Sam"]


def test_speaker_capitalisation_is_normalised() -> None:
    dialogue = parser().parse("SAM: Hello.\nkim: Hi.")
    assert [u.speaker for u in dialogue.utterances] == ["Sam", "Kim"]


def test_unknown_speakers_are_not_invented() -> None:
    """A generic 'word followed by colon' pattern would accept these as speakers."""
    dialogue = parser().parse(
        "GROUND TRUTH: do not treat me as a speaker\nSam: Hello.\nNote: neither me."
    )
    assert [u.speaker for u in dialogue.utterances] == ["Sam"]
    kinds = {issue.kind for issue in dialogue.issues}
    assert kinds == {"preamble", "unparsed_line"}


def test_preamble_is_discarded_with_an_issue() -> None:
    dialogue = parser().parse("Here is the conversation you asked for:\nSam: Hello.")
    assert len(dialogue.utterances) == 1
    assert dialogue.issues[0].kind == "preamble"
    assert dialogue.issues[0].line_number == 1


def test_trailing_pleasantry_is_dropped_by_default() -> None:
    """Appending it would put it in the transcript, poisoning the ASR target."""
    dialogue = parser().parse("Sam: Hello.\nLet me know if you'd like more!")
    assert dialogue.transcript() == "Hello."
    assert dialogue.issues[0].kind == "unparsed_line"


def test_continuation_lines_can_be_folded_in() -> None:
    dialogue = parser(continuation_policy="append").parse(
        "Sam: This is a long turn\nthat wrapped across lines.\nKim: Understood."
    )
    assert dialogue.utterances[0].text == "This is a long turn that wrapped across lines."
    assert len(dialogue.utterances) == 2


def test_empty_utterances_are_reported() -> None:
    dialogue = parser().parse("Sam:\nKim: Hi.")
    assert len(dialogue.utterances) == 1
    assert any(issue.kind == "empty_utterance" for issue in dialogue.issues)


def test_comments_and_blank_lines_are_ignored() -> None:
    dialogue = parser().parse("# a comment\n\nSam: Hello.\n\n")
    assert len(dialogue.utterances) == 1
    assert dialogue.issues == ()


def test_colons_inside_the_utterance_survive() -> None:
    dialogue = parser().parse("Sam: The error says: printer offline.")
    assert dialogue.utterances[0].text == "The error says: printer offline."


def test_round_trip_through_the_script_format() -> None:
    original = "Sam: (calmly) Hello.\nKim: Hi there."
    dialogue = parser().parse(original)
    assert dialogue.to_script() == original


def test_transcript_drops_labels_and_cues() -> None:
    dialogue = parser().parse("Sam: (calmly) Hello.\nKim: Hi.")
    assert dialogue.transcript() == "Hello. Hi."


def test_by_role_filters() -> None:
    dialogue = parser().parse("Sam: a\nKim: b\nSam: c")
    assert len(dialogue.by_role("agent")) == 2
    assert len(dialogue.by_role("customer")) == 1


def test_a_name_that_prefixes_another_is_not_shadowed() -> None:
    from conversim import Persona

    cast = Cast(
        personas=(
            Persona(name="Ann", role="agent"),
            Persona(name="Anna", role="customer"),
        )
    )
    dialogue = DialogueParser(cast=cast).parse("Anna: mine.\nAnn: also mine.")
    assert [u.speaker for u in dialogue.utterances] == ["Anna", "Ann"]
