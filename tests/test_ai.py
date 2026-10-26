from __future__ import annotations

from types import SimpleNamespace

import pytest

from notesbro_bot.bot import transcribe_voice_sync
from notesbro_bot.models import VoiceNote
from notesbro_bot.note_summary import summarize_transcript_to_note_sync


def test_transcribe_voice_sync_uses_aitunnel_stt() -> None:
    seen: dict = {}

    def create(**kwargs):
        seen.update(kwargs)
        return SimpleNamespace(text="  привет  ")

    client = SimpleNamespace(
        audio=SimpleNamespace(transcriptions=SimpleNamespace(create=create))
    )
    text = transcribe_voice_sync(
        client, "voxtral-small-24b-2507-stt", b"abc", "audio/ogg"
    )
    assert text == "привет"
    assert seen["model"] == "voxtral-small-24b-2507-stt"
    assert "language" not in seen
    assert seen["file"] == ("voice.ogg", b"abc", "audio/ogg")


def test_summarize_uses_parsed_note() -> None:
    parsed = VoiceNote(note_title=" T ", list_title="L", items=[" a "])
    seen: dict = {}

    def parse(**kwargs):
        seen.update(kwargs)
        return SimpleNamespace(
            choices=[
                SimpleNamespace(
                    message=SimpleNamespace(parsed=parsed, content=None)
                )
            ]
        )

    client = SimpleNamespace(
        chat=SimpleNamespace(completions=SimpleNamespace(parse=parse))
    )
    note = summarize_transcript_to_note_sync(client, "chat-model", "hello")
    assert note.note_title == "T"
    assert note.items == ["a"]
    assert seen["model"] == "chat-model"
    assert seen["response_format"] is VoiceNote
    assert "hello" in seen["messages"][0]["content"]


def _client_with(message: SimpleNamespace, calls: list) -> SimpleNamespace:
    def parse(**kwargs):
        calls.append(kwargs)
        return SimpleNamespace(choices=[SimpleNamespace(message=message)])

    return SimpleNamespace(
        chat=SimpleNamespace(completions=SimpleNamespace(parse=parse))
    )


def test_note_falls_back_to_json_content() -> None:
    note = VoiceNote(note_title="T", list_title="L", items=["a"])
    message = SimpleNamespace(parsed=None, content=note.model_dump_json())
    client = _client_with(message, [])
    result = summarize_transcript_to_note_sync(client, "m", "hi")
    assert result == note


def test_note_retries_then_raises_on_empty_response(monkeypatch) -> None:
    from notesbro_bot import note_summary

    monkeypatch.setattr(note_summary.time, "sleep", lambda _s: None)
    calls: list = []
    message = SimpleNamespace(parsed=None, content="  ")
    client = _client_with(message, calls)
    with pytest.raises(ValueError, match="no structured note"):
        summarize_transcript_to_note_sync(client, "m", "hi")
    assert len(calls) == 3


def test_note_retries_transient_error_then_succeeds(monkeypatch) -> None:
    from notesbro_bot import note_summary

    monkeypatch.setattr(note_summary.time, "sleep", lambda _s: None)
    note = VoiceNote(note_title="T", list_title="L", items=["a"])
    attempts = {"n": 0}

    def parse(**kwargs):
        _ = kwargs
        attempts["n"] += 1
        if attempts["n"] == 1:
            raise RuntimeError("flaky")
        return SimpleNamespace(
            choices=[SimpleNamespace(message=SimpleNamespace(parsed=note))]
        )

    client = SimpleNamespace(
        chat=SimpleNamespace(completions=SimpleNamespace(parse=parse))
    )
    assert summarize_transcript_to_note_sync(client, "m", "hi") == note
    assert attempts["n"] == 2


def test_normalize_note_fills_empty_fields() -> None:
    from notesbro_bot.note_summary import _normalize_note

    note = VoiceNote(note_title="  ", list_title=" ", items=["  "])
    result = _normalize_note(note, "ru")
    assert result.note_title == "Заметка без названия"
    assert result.list_title == "Итоги"
    assert result.items == [
        "(Из расшифровки не удалось выделить отдельные пункты.)"
    ]


def test_merge_prompt_includes_previous_note() -> None:
    from notesbro_bot.note_summary import merge_transcript_into_note_sync

    seen: list = []
    note = VoiceNote(note_title="T", list_title="L", items=["a"])
    message = SimpleNamespace(parsed=note, content=None)
    merge_transcript_into_note_sync(
        _client_with(message, seen), "m", note, "new words", "ru"
    )
    prompt = seen[0]["messages"][0]["content"]
    assert "Merge an existing note" in prompt
    assert '"note_title":"T"' in prompt
    assert "new words" in prompt


def test_parse_note_returns_none_for_garbage() -> None:
    from notesbro_bot.note_summary import parse_voice_note_from_message_text

    assert parse_voice_note_from_message_text("") is None
    assert parse_voice_note_from_message_text("   \n  ") is None
    assert parse_voice_note_from_message_text("no bullets here") is None


def test_parse_note_without_emoji_uses_positional_fallbacks() -> None:
    from notesbro_bot.note_summary import parse_voice_note_from_message_text

    note = parse_voice_note_from_message_text("Title\nHeading\n• one")
    assert note is not None
    assert note.note_title == "Title"
    assert note.list_title == "Heading"
    assert note.items == ["one"]


def test_parse_note_without_any_heading_defaults_list_title() -> None:
    from notesbro_bot.note_summary import parse_voice_note_from_message_text

    note = parse_voice_note_from_message_text("Title\n• only")
    assert note is not None
    assert note.list_title == "Summary"
