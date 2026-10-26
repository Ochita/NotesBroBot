from __future__ import annotations

import asyncio
import time
from types import SimpleNamespace

import pytest
from fakes import (
    FakeBot,
    FakeContext,
    FakeMessage,
    FakeRepository,
    FakeUser,
    FakeVoice,
    make_settings,
    make_update,
)

from notesbro_bot.bot import (
    _ADD_VOICE_TTL_SEC,
    _UD_ADD_PREVIOUS_NOTE_JSON,
    _UD_ADD_TARGET_MESSAGE_ID,
    _UD_AWAITING_ADD_VOICE,
    _locale_code,
    _post_init,
    _show_status,
    build_application,
    cmd_add,
    cmd_cancel,
    cmd_help,
    cmd_set_locale,
    cmd_start,
    on_voice,
    transcribe_voice_sync,
)
from notesbro_bot.models import VoiceNote
from notesbro_bot.note_summary import voice_note_to_telegram_html

NOTE = VoiceNote(note_title="T", list_title="L", items=["a"])
NOTE_TEXT = voice_note_to_telegram_html(NOTE)


class FakeClient:
    """Stands in for OpenAI: stt + chat calls with optional failure."""

    def __init__(self, transcript: str = "hello", fail: bool = False) -> None:
        self.transcript = transcript
        self.fail = fail
        self.stt_calls: list[dict] = []
        self.chat_calls: list[dict] = []
        self.audio = SimpleNamespace(
            transcriptions=SimpleNamespace(create=self._create)
        )
        self.chat = SimpleNamespace(
            completions=SimpleNamespace(parse=self._parse)
        )

    def _create(self, **kwargs):
        self.stt_calls.append(kwargs)
        if self.fail:
            raise RuntimeError("stt down")
        return SimpleNamespace(text=self.transcript)

    def _parse(self, **kwargs):
        self.chat_calls.append(kwargs)
        if self.fail:
            raise RuntimeError("chat down")
        return SimpleNamespace(
            choices=[SimpleNamespace(message=SimpleNamespace(parsed=NOTE))]
        )


async def test_start_replies_and_registers_user() -> None:
    repository = FakeRepository()
    context = FakeContext(repository=repository)
    message = FakeMessage()
    await cmd_start(make_update(message=message), context)
    assert repository.known_users == {42}
    assert "voice message" in message.replies[0]


async def test_start_without_message_still_registers() -> None:
    repository = FakeRepository()
    context = FakeContext(repository=repository)
    update = make_update(message=None)
    await cmd_start(update, context)
    assert repository.known_users == {42}


async def test_start_returns_on_missing_user() -> None:
    context = FakeContext()
    update = make_update(message=FakeMessage())
    update.effective_user = None
    await cmd_start(update, context)
    assert context.bot.sent == []


async def test_access_denied_for_unknown_user() -> None:
    settings = make_settings(allow_new_users=False)
    repository = FakeRepository()
    context = FakeContext(settings=settings, repository=repository)
    message = FakeMessage()
    await cmd_start(make_update(message=message), context)
    assert context.bot.sent[0]["text"].startswith("This bot is not accepting")
    assert repository.known_users == set()


async def test_access_denied_returns_true_without_user() -> None:
    settings = make_settings(allow_new_users=False)
    context = FakeContext(settings=settings)
    update = make_update(message=FakeMessage())
    update.effective_user = None
    await cmd_start(update, context)
    assert context.bot.sent == []


async def test_access_allowed_for_known_user() -> None:
    settings = make_settings(allow_new_users=False)
    repository = FakeRepository(known_users={42})
    context = FakeContext(settings=settings, repository=repository)
    message = FakeMessage()
    await cmd_start(make_update(message=message), context)
    assert "voice message" in message.replies[0]


async def test_start_replies_in_user_locale() -> None:
    repository = FakeRepository()
    repository.locales[42] = "ru"
    context = FakeContext(repository=repository)
    message = FakeMessage()
    await cmd_start(make_update(message=message), context)
    assert "голосовое" in message.replies[0]


async def test_start_uses_telegram_language_code() -> None:
    repository = FakeRepository()
    context = FakeContext(repository=repository)
    message = FakeMessage()
    update = make_update(message=message, user=FakeUser(language_code="de"))
    await cmd_start(update, context)
    assert "Sprachnachricht" in message.replies[0]


async def test_start_falls_back_to_default_locale() -> None:
    settings = make_settings(default_locale="it")
    repository = FakeRepository()
    context = FakeContext(settings=settings, repository=repository)
    message = FakeMessage()
    await cmd_start(make_update(message=message), context)
    assert "vocale" in message.replies[0]


async def test_locale_code_ignores_unknown_stored_locale() -> None:
    repository = FakeRepository()
    repository.locales[42] = "klingon"
    context = FakeContext(repository=repository)
    update = make_update(user=FakeUser(language_code="fr"))
    assert await _locale_code(update, context) == "fr"


async def test_help_sends_commands() -> None:
    context = FakeContext()
    await cmd_help(make_update(message=FakeMessage()), context)
    assert "/set_locale" in context.bot.sent[0]["text"]


async def test_set_locale_requires_argument() -> None:
    context = FakeContext()
    await cmd_set_locale(make_update(message=FakeMessage()), context)
    assert "language code" in context.bot.sent[0]["text"]


async def test_set_locale_rejects_unknown() -> None:
    context = FakeContext(args=["zh"])
    await cmd_set_locale(make_update(message=FakeMessage()), context)
    assert "Unknown language" in context.bot.sent[-1]["text"]


async def test_set_locale_saves_and_confirms_in_new_locale() -> None:
    repository = FakeRepository()
    context = FakeContext(repository=repository, args=["ru-RU"])
    await cmd_set_locale(make_update(message=FakeMessage()), context)
    assert repository.locales[42] == "ru"
    assert "Язык сохранен" in context.bot.sent[-1]["text"]


async def test_add_without_reply_prompts_usage() -> None:
    context = FakeContext()
    message = FakeMessage()
    await cmd_add(make_update(message=message), context)
    assert "Reply /add" in message.replies[0]


async def test_add_rejects_plain_text_target() -> None:
    context = FakeContext()
    message = FakeMessage(reply_to_message=FakeMessage(text="just chat"))
    await cmd_add(make_update(message=message), context)
    assert "couldn't parse" in message.replies[0]


async def test_add_arms_merge_and_stores_state() -> None:
    context = FakeContext()
    message = FakeMessage(reply_to_message=FakeMessage(text=NOTE_TEXT))
    await cmd_add(make_update(message=message), context)
    assert context.user_data[_UD_AWAITING_ADD_VOICE] is True
    assert context.user_data[_UD_ADD_TARGET_MESSAGE_ID] == 1
    assert _UD_ADD_PREVIOUS_NOTE_JSON in context.user_data
    assert str(int(_ADD_VOICE_TTL_SEC)) in message.replies[0]


async def test_add_returns_without_user() -> None:
    context = FakeContext()
    update = make_update(message=FakeMessage())
    update.effective_user = None
    await cmd_add(update, context)
    assert context.user_data == {}


async def test_cancel_reports_done_and_nothing() -> None:
    context = FakeContext()
    message = FakeMessage()
    await cmd_cancel(make_update(message=message), context)
    assert message.replies[-1] == "Nothing to cancel."
    context.user_data[_UD_AWAITING_ADD_VOICE] = True
    await cmd_cancel(make_update(message=message), context)
    assert message.replies[-1] == "Cancelled merge mode."
    assert _UD_AWAITING_ADD_VOICE not in context.user_data


async def test_post_init_initializes_repository() -> None:
    repository = FakeRepository()
    await _post_init(
        SimpleNamespace(bot_data={"repository": repository})
    )
    assert repository.initialized is True


async def test_show_status_prefers_reply_message() -> None:
    context = FakeContext()
    message = FakeMessage()
    await _show_status(
        context,
        chat_id=1,
        reply_message=message,
        edit_message_id=5,
        text="x",
    )
    assert message.edits == [("x", {})]
    assert context.bot.edited == []


async def test_show_status_edits_by_id_when_no_reply_message() -> None:
    context = FakeContext()
    await _show_status(
        context,
        chat_id=1,
        reply_message=None,
        edit_message_id=5,
        text="x",
    )
    assert context.bot.edited[0]["message_id"] == 5


async def test_show_status_swallows_edit_errors() -> None:
    bot = FakeBot(fail_edit=True)
    context = FakeContext(bot=bot)
    await _show_status(
        context,
        chat_id=1,
        reply_message=None,
        edit_message_id=5,
        text="x",
    )


async def test_on_voice_happy_path() -> None:
    client = FakeClient(transcript="привет")
    context = FakeContext(client=client)
    message = FakeMessage(voice=FakeVoice())
    await on_voice(make_update(message=message), context)
    assert message.replies[0] == "Transcribing…"
    assert message.edits[-1][0] == NOTE_TEXT
    assert message.edits[-1][1] == {"parse_mode": "HTML"}
    assert "language" not in client.stt_calls[0]
    assert "language of the transcript" in (
        client.chat_calls[0]["messages"][0]["content"]
    )


async def test_on_voice_ignores_non_voice_and_missing_user() -> None:
    context = FakeContext()
    await on_voice(make_update(message=FakeMessage()), context)
    update = make_update(message=FakeMessage(voice=FakeVoice()))
    update.effective_user = None
    await on_voice(update, context)
    assert context.bot.edited == []


async def test_on_voice_access_denied() -> None:
    settings = make_settings(allow_new_users=False)
    context = FakeContext(settings=settings)
    message = FakeMessage(voice=FakeVoice())
    await on_voice(make_update(message=message), context)
    assert message.replies == []
    assert context.bot.sent[0]["text"].startswith("This bot")


async def test_on_voice_reports_download_failure() -> None:
    bot = FakeBot(fail_download=True)
    context = FakeContext(bot=bot)
    message = FakeMessage(voice=FakeVoice())
    await on_voice(make_update(message=message), context)
    edits = message.edits
    assert "Could not download" in edits[-1][0]


async def test_on_voice_reports_transcription_failure() -> None:
    client = FakeClient(fail=True)
    context = FakeContext(client=client)
    message = FakeMessage(voice=FakeVoice())
    await on_voice(make_update(message=message), context)
    assert "Transcription failed" in message.edits[-1][0]


async def test_on_voice_reports_empty_transcript() -> None:
    client = FakeClient(transcript="   ")
    context = FakeContext(client=client)
    message = FakeMessage(voice=FakeVoice())
    await on_voice(make_update(message=message), context)
    assert "No speech recognized" in message.edits[-1][0]


async def test_on_voice_reports_note_failure() -> None:
    client = FakeClient()
    client._parse = lambda **kwargs: (_ for _ in ()).throw(
        RuntimeError("boom")
    )  # type: ignore[assignment]
    client.chat.completions.parse = client._parse  # type: ignore[union-attr]
    context = FakeContext(client=client)
    message = FakeMessage(voice=FakeVoice())
    await on_voice(make_update(message=message), context)
    assert "Could not build notes" in message.edits[-1][0]


async def test_on_voice_falls_back_to_plain_text_on_format_error() -> None:
    class EditFails(FakeMessage):
        async def edit_text(self, text: str, **kwargs):
            if kwargs.get("parse_mode"):
                raise RuntimeError("bad html")
            return await super().edit_text(text, **kwargs)

    bot = FakeBot()
    context = FakeContext(client=FakeClient(transcript="hi"), bot=bot)
    message = EditFails(voice=FakeVoice())
    await on_voice(make_update(message=message), context)
    assert "Could not send formatted notes" in message.edits[-1][0]
    assert "hi" in message.edits[-1][0]


async def _armed_context(**kwargs) -> FakeContext:
    context = FakeContext(**kwargs)
    context.user_data[_UD_AWAITING_ADD_VOICE] = True
    context.user_data[_UD_ADD_TARGET_MESSAGE_ID] = 55
    context.user_data[_UD_ADD_PREVIOUS_NOTE_JSON] = NOTE.model_dump_json()
    context.user_data["add_status_message_id"] = 99
    context.user_data["awaiting_add_deadline_mono"] = time.monotonic() + 60
    return context


async def test_on_voice_merges_into_original_message() -> None:
    client = FakeClient(transcript="more")
    context = await _armed_context(client=client)
    message = FakeMessage(voice=FakeVoice())
    await on_voice(make_update(message=message), context)
    assert context.bot.edited[0] == {
        "chat_id": 777,
        "message_id": 99,
        "text": "Transcribing add-on…",
    }
    assert context.bot.edited[-1]["message_id"] == 55
    assert context.bot.deleted[0]["message_id"] == 99
    assert _UD_AWAITING_ADD_VOICE not in context.user_data
    prompt = client.chat_calls[0]["messages"][0]["content"]
    assert "Merge an existing note" in prompt


async def test_on_voice_merge_without_status_id_uses_reply_status() -> None:
    client = FakeClient(transcript="more")
    context = await _armed_context(client=client)
    context.user_data.pop("add_status_message_id")
    context.user_data[_UD_ADD_TARGET_MESSAGE_ID] = 55
    message = FakeMessage(voice=FakeVoice())
    await on_voice(make_update(message=message), context)
    assert message.replies[0] == "Transcribing add-on…"
    # The reply-based progress message is cleaned up after the merge.
    assert context.bot.deleted == [{"chat_id": 777, "message_id": 2}]
    assert context.bot.edited[-1]["message_id"] == 55
    assert context.bot.edited[-1]["text"] == NOTE_TEXT


async def test_on_voice_merge_drops_corrupt_state() -> None:
    client = FakeClient(transcript="more")
    context = await _armed_context(client=client)
    context.user_data[_UD_ADD_PREVIOUS_NOTE_JSON] = "{not json"
    message = FakeMessage(voice=FakeVoice())
    await on_voice(make_update(message=message), context)
    assert message.replies[0] == "Transcribing…"
    assert _UD_AWAITING_ADD_VOICE not in context.user_data


async def test_on_voice_expired_arm_summarizes_instead_of_merging() -> None:
    client = FakeClient(transcript="more")
    context = await _armed_context(client=client)
    context.user_data["awaiting_add_deadline_mono"] = 0.0
    message = FakeMessage(voice=FakeVoice())
    await on_voice(make_update(message=message), context)
    assert message.replies[0] == "Transcribing…"


async def test_on_voice_merge_ignores_delete_failure() -> None:
    bot = FakeBot(fail_delete=True)
    context = await _armed_context(client=FakeClient(), bot=bot)
    message = FakeMessage(voice=FakeVoice())
    await on_voice(make_update(message=message), context)
    assert bot.edited[-1]["message_id"] == 55


def test_build_application_registers_handlers() -> None:
    application = build_application(make_settings())
    assert application.bot_data["settings"].model_name == "chat-model"
    assert application.bot_data["repository"] is not None
    assert application.bot_data["ai_client"] is not None


def test_transcribe_omits_language_hint_when_absent() -> None:
    client = FakeClient()
    assert transcribe_voice_sync(client, "m", b"a", "audio/ogg") == "hello"
    assert "language" not in client.stt_calls[0]


def test_transcribe_strips_and_defaults_mime() -> None:
    client = FakeClient(transcript="  text  ")
    assert transcribe_voice_sync(client, "m", b"a", "") == "text"
    assert client.stt_calls[0]["file"][0] == "voice.ogg"


@pytest.mark.parametrize("attempt", [1])
def test_transcribe_retries_then_raises(monkeypatch, attempt: int) -> None:
    from notesbro_bot import bot as bot_module

    monkeypatch.setattr(bot_module.time, "sleep", lambda _s: None)
    client = FakeClient(fail=True)
    with pytest.raises(RuntimeError, match="stt down"):
        transcribe_voice_sync(client, "m", b"a", "audio/mpeg")
    assert len(client.stt_calls) == 3


def test_audio_file_extension_mapping() -> None:
    from notesbro_bot.bot import _audio_file

    assert _audio_file(b"a", "audio/mpeg")[0] == "voice.mp3"
    assert _audio_file(b"a", "audio/AMR; codecs=x")[0] == "voice.amr"


async def test_repeated_flows_stay_independent() -> None:
    client = FakeClient(transcript="x")
    context = FakeContext(client=client)
    message = FakeMessage(voice=FakeVoice())
    await asyncio.gather(
        on_voice(make_update(message=message), context),
        on_voice(make_update(message=message), context),
    )
    assert message.replies == ["Transcribing…", "Transcribing…"]
    assert [text for text, _ in message.edits[-2:]] == [NOTE_TEXT, NOTE_TEXT]


async def test_arm_expired_without_deadline() -> None:
    from notesbro_bot.bot import _add_voice_arm_expired

    assert _add_voice_arm_expired({}) is True


async def test_locale_code_without_user() -> None:
    context = FakeContext()
    update = make_update()
    update.effective_user = None
    assert await _locale_code(update, context) == "en"


async def test_show_status_without_any_target() -> None:
    context = FakeContext()
    await _show_status(
        context,
        chat_id=1,
        reply_message=None,
        edit_message_id=None,
        text="x",
    )
    assert context.bot.edited == []


@pytest.mark.parametrize(
    "handler",
    [cmd_start, cmd_help, cmd_set_locale, cmd_add, cmd_cancel, on_voice],
)
async def test_handlers_noop_without_user(handler) -> None:
    context = FakeContext()
    update = make_update(message=FakeMessage(voice=FakeVoice()))
    update.effective_user = None
    update.effective_chat = None
    await handler(update, context)
    assert context.bot.sent == []
    assert context.bot.edited == []


async def test_set_locale_missing_user_data_still_works() -> None:
    context = FakeContext(args=["de"])
    context.user_data = None
    update = make_update(message=FakeMessage())
    await cmd_set_locale(update, context)
    assert "Sprache gespeichert" in context.bot.sent[-1]["text"]


async def test_cancel_without_user_data() -> None:
    context = FakeContext()
    context.user_data = None
    await cmd_cancel(make_update(message=FakeMessage()), context)
    assert context.bot.sent == []


async def test_add_without_user_data() -> None:
    context = FakeContext()
    context.user_data = None
    await cmd_add(make_update(message=FakeMessage()), context)
    assert context.bot.sent == []


async def test_add_without_message() -> None:
    context = FakeContext()
    await cmd_add(make_update(message=None), context)
    assert context.bot.sent == []


async def test_on_voice_merge_state_without_previous_json() -> None:
    client = FakeClient(transcript="more")
    context = await _armed_context(client=client)
    context.user_data.pop(_UD_ADD_PREVIOUS_NOTE_JSON)
    message = FakeMessage(voice=FakeVoice())
    await on_voice(make_update(message=message), context)
    assert message.replies[0] == "Transcribing…"


async def test_on_voice_merge_empty_status_text_keeps_message() -> None:
    bot = FakeBot()
    context = await _armed_context(client=FakeClient(), bot=bot)
    message = FakeMessage(voice=FakeVoice())
    await on_voice(make_update(message=message), context)
    assert [d["message_id"] for d in bot.deleted] == [99]
