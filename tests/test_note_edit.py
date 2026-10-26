from __future__ import annotations

from notesbro_bot.models import VoiceNote
from notesbro_bot.note_summary import (
    add_item,
    find_item_by_quote,
    replace_quoted_text,
)

NOTE = VoiceNote(
    note_title="Planning",
    list_title="Steps",
    items=["buy milk", "call mom", "file taxes"],
)


def test_quote_matches_exact_line() -> None:
    assert find_item_by_quote(NOTE, "call mom") == 1


def test_quote_ignores_bullet_and_case() -> None:
    assert find_item_by_quote(NOTE, "• CALL MOM") == 1


def test_quote_matches_unique_substring() -> None:
    assert find_item_by_quote(NOTE, "taxes") == 2


def test_ambiguous_or_empty_quote_matches_nothing() -> None:
    assert find_item_by_quote(NOTE, None) is None
    assert find_item_by_quote(NOTE, "   ") is None
    assert find_item_by_quote(NOTE, "a") is None


def test_quote_of_heading_is_not_an_item() -> None:
    assert find_item_by_quote(NOTE, "Planning") is None


def test_fragment_is_swapped_and_rest_of_line_survives() -> None:
    edited = replace_quoted_text(NOTE, "milk", "oat milk")
    assert edited is not None
    assert edited.items[0] == "buy oat milk"
    assert edited.items[1:] == ["call mom", "file taxes"]


def test_whole_line_replacement() -> None:
    edited = replace_quoted_text(NOTE, "call mom", "call dad")
    assert edited is not None
    assert edited.items[1] == "call dad"


def test_replacement_is_case_insensitive() -> None:
    edited = replace_quoted_text(NOTE, "FILE TAXES", "pay taxes")
    assert edited is not None
    assert edited.items[2] == "pay taxes"


def test_replace_without_quote_or_text_changes_nothing() -> None:
    assert replace_quoted_text(NOTE, None, "x") is None
    assert replace_quoted_text(NOTE, "milk", "   ") is None


def test_replace_unknown_quote_matches_nothing() -> None:
    assert replace_quoted_text(NOTE, "banana", "x") is None


def test_replace_same_text_is_a_noop() -> None:
    assert replace_quoted_text(NOTE, "call mom", "call mom") is None


def test_replace_truncates_overlong_item() -> None:
    edited = replace_quoted_text(NOTE, "milk", "m" * 500)
    assert edited is not None
    assert len(edited.items[0]) == 200


def test_add_item_appends_and_normalises_whitespace() -> None:
    edited = add_item(NOTE, "  wash   car  ")
    assert edited is not None
    assert edited.items[-1] == "wash car"
    assert len(edited.items) == 4


def test_add_item_rejects_empty_text() -> None:
    assert add_item(NOTE, "   ") is None


def test_add_item_respects_schema_limit() -> None:
    full = VoiceNote(
        note_title="T", list_title="L", items=[f"i{n}" for n in range(16)]
    )
    assert add_item(full, "one more") is None
