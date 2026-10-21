from __future__ import annotations

from types import SimpleNamespace

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
    assert seen["language"] == "ru"
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
