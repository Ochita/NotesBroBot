from __future__ import annotations

from notesbro_bot.note_summary import parse_voice_note_from_message_text


def test_parse_voice_note_from_message_text_roundtrip_shape() -> None:
    text = "\n".join(
        [
            "📝 <b>Project kickoff</b>",
            "",
            "📋 <b>Next steps</b>",
            "• Schedule meeting",
            "• Prepare agenda",
        ]
    )
    note = parse_voice_note_from_message_text(text)
    assert note is not None
    assert note.note_title == "Project kickoff"
    assert note.list_title == "Next steps"
    assert note.items == ["Schedule meeting", "Prepare agenda"]

