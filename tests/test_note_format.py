from __future__ import annotations

from notesbro_bot.models import VoiceNote
from notesbro_bot.note_summary import voice_note_to_telegram_html


def test_voice_note_to_telegram_html_escapes_and_structure() -> None:
    note = VoiceNote(
        note_title='Shopping & <stuff>',
        list_title="Today's list",
        items=["Milk", "Bread & butter"],
    )
    html = voice_note_to_telegram_html(note)
    assert "<b>Shopping &amp; &lt;stuff&gt;</b>" in html
    assert "Today" in html and "list" in html
    assert "• Milk" in html
    assert "Bread &amp; butter" in html


def test_voice_note_to_telegram_html_truncates() -> None:
    note = VoiceNote(
        note_title="T",
        list_title="L",
        items=["x" * 5000],
    )
    html = voice_note_to_telegram_html(note, max_len=200)
    assert len(html) <= 200
    assert html.endswith("…")
