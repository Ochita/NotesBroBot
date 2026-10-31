from __future__ import annotations

from types import SimpleNamespace

import pytest

from notesbro_bot.bot import transcribe_voice
from notesbro_bot.models import VoiceNote
from notesbro_bot.note_summary import summarize_text_to_note


async def _no_sleep(_seconds: float) -> None:
    """Skip the retry backoff so tests stay instant."""


async def test_transcribe_voice_uses_aitunnel_stt() -> None:
    seen: dict = {}

    async def create(**kwargs):
        seen.update(kwargs)
        return SimpleNamespace(text="  привет  ")

    client = SimpleNamespace(
        audio=SimpleNamespace(transcriptions=SimpleNamespace(create=create))
    )
    text = await transcribe_voice(
        client,
        "voxtral-small-24b-2507-stt",
        b"abc",
        "audio/ogg",
        "ru",
    )
    assert text == "привет"
    assert seen["model"] == "voxtral-small-24b-2507-stt"
    assert seen["language"] == "ru"
    assert seen["file"] == ("voice.ogg", b"abc", "audio/ogg")


async def test_summarize_uses_parsed_note() -> None:
    parsed = VoiceNote(note_title=" T ", list_title="L", items=[" a "])
    seen: dict = {}

    async def parse(**kwargs):
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
    note = await summarize_text_to_note(client, "chat-model", "hello")
    assert note.note_title == "T"
    assert note.items == ["a"]
    assert seen["model"] == "chat-model"
    assert seen["response_format"] is VoiceNote
    assert "hello" in seen["messages"][0]["content"]


def _client_with(message: SimpleNamespace, calls: list) -> SimpleNamespace:
    async def parse(**kwargs):
        calls.append(kwargs)
        return SimpleNamespace(choices=[SimpleNamespace(message=message)])

    return SimpleNamespace(
        chat=SimpleNamespace(completions=SimpleNamespace(parse=parse))
    )


async def test_note_falls_back_to_json_content() -> None:
    note = VoiceNote(note_title="T", list_title="L", items=["a"])
    message = SimpleNamespace(parsed=None, content=note.model_dump_json())
    client = _client_with(message, [])
    result = await summarize_text_to_note(client, "m", "hi")
    assert result == note


async def test_note_retries_then_raises_on_empty_response(monkeypatch) -> None:
    from notesbro_bot import note_summary

    monkeypatch.setattr(note_summary.asyncio, "sleep", _no_sleep)
    calls: list = []
    message = SimpleNamespace(parsed=None, content="  ")
    client = _client_with(message, calls)
    with pytest.raises(ValueError, match="no structured note"):
        await summarize_text_to_note(client, "m", "hi")
    assert len(calls) == 3


async def test_note_does_not_retry_transport_errors() -> None:
    note = VoiceNote(note_title="T", list_title="L", items=["a"])
    attempts = {"n": 0}

    async def parse(**kwargs):
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
    # transport-level retries live in the SDK (max_retries), not here
    with pytest.raises(RuntimeError, match="flaky"):
        await summarize_text_to_note(client, "m", "hi")
    assert attempts["n"] == 1


def test_normalize_note_fills_empty_fields() -> None:
    from notesbro_bot.note_summary import _normalize_note

    note = VoiceNote(note_title="  ", list_title=" ", items=["  "])
    result = _normalize_note(note, "ru")
    assert result.note_title == "Заметка без названия"
    assert result.list_title == "Итоги"
    assert result.items == [
        "(Из расшифровки не удалось выделить отдельные пункты.)"
    ]


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


@pytest.mark.parametrize(
    ("locale", "language"),
    [
        ("ru", "Russian"),
        ("de", "German"),
        ("uk", "Ukrainian"),
        ("es", "Spanish"),
        (None, "English"),
    ],
)
async def test_summary_prompt_pins_the_locale_language(
    locale, language
) -> None:
    seen: list = []
    note = VoiceNote(note_title="T", list_title="L", items=["a"])
    client = _client_with(SimpleNamespace(parsed=note, content=None), seen)
    await summarize_text_to_note(client, "m", "hi", locale)
    prompt = seen[0]["messages"][0]["content"]
    assert f"Write every field in {language}" in prompt
    # must not follow the transcript language any more
    assert "language of the transcript" not in prompt
