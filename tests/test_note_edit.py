from __future__ import annotations

from notesbro_bot.models import VoiceNote
from notesbro_bot.note_summary import apply_note_edits

NOTE = VoiceNote(
    note_title="Planning",
    list_title="Steps",
    items=["buy milk", "call mom", "file taxes"],
)


def test_replace_single_item_by_number() -> None:
    edited, applied = apply_note_edits(NOTE, "2 call dad")
    assert applied == 1
    assert edited.items == ["buy milk", "call dad", "file taxes"]
    assert edited.note_title == "Planning"
    assert edited.list_title == "Steps"


def test_multiple_edits_apply_in_order() -> None:
    edited, applied = apply_note_edits(NOTE, "1 buy bread\n-3\n+ walk dog")
    assert applied == 3
    assert edited.items == ["buy bread", "call mom", "walk dog"]


def test_trailing_item_can_be_deleted_by_number() -> None:
    edited, applied = apply_note_edits(NOTE, "-3")
    assert applied == 1
    assert edited.items == ["buy milk", "call mom"]


def test_headings_can_be_rewritten() -> None:
    edited, _ = apply_note_edits(NOTE, "t Weekend\ntitle Real Title")
    assert edited.note_title == "Real Title"
    edited, _ = apply_note_edits(NOTE, "l Shopping\nlist Errands")
    assert edited.list_title == "Errands"


def test_value_keeps_inner_spacing_and_extra_words() -> None:
    edited, _ = apply_note_edits(NOTE, "2   buy  oat milk  ")
    assert edited.items[1] == "buy  oat milk"


def test_out_of_range_and_garbage_lines_are_ignored() -> None:
    _, applied = apply_note_edits(NOTE, "9 nope\n0 nope\n-0\nx\n\n   ")
    assert applied == 0


def test_no_edits_reports_zero() -> None:
    assert apply_note_edits(NOTE, "")[1] == 0
    assert apply_note_edits(NOTE, "just a sentence")[1] == 0


def test_edits_are_capped_to_schema_limits() -> None:
    many = "\n".join(f"+ item {i}" for i in range(40))
    edited, applied = apply_note_edits(NOTE, many)
    assert applied == 40
    assert len(edited.items) == 16


def test_long_heading_is_truncated_to_schema_limit() -> None:
    edited, _ = apply_note_edits(NOTE, "t " + "T" * 500)
    assert len(edited.note_title) == 200


def test_noop_edit_changes_nothing() -> None:
    edited, applied = apply_note_edits(NOTE, "2 call mom")
    assert applied == 1
    assert edited == NOTE


def test_numbered_list_helper_matches_editing_numbers() -> None:
    from notesbro_bot.bot import _number_lines

    assert _number_lines(NOTE).splitlines()[1] == "2. call mom"
