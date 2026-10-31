from __future__ import annotations

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
    MODE_ADD,
    MODE_EDIT,
    _locale_code,
    _post_init,
    build_application,
    cmd_add,
    cmd_cancel,
    cmd_edit,
    cmd_help,
    cmd_set_locale,
    cmd_start,
    on_text,
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
        prompt = kwargs["messages"][0]["content"]
        if prompt.startswith("Split user text"):
            # model the split: one bullet per non-empty input line
            source = prompt.split("Text:\n", 1)[-1]
            items = [ln.strip() for ln in source.splitlines() if ln.strip()]
            note = NOTE.model_copy(update={"items": items or NOTE.items})
        else:
            note = NOTE
        return SimpleNamespace(
            choices=[SimpleNamespace(message=SimpleNamespace(parsed=note))]
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







async def test_post_init_initializes_repository() -> None:
    repository = FakeRepository()
    await _post_init(
        SimpleNamespace(bot_data={"repository": repository})
    )
    assert repository.initialized is True





async def test_on_voice_happy_path() -> None:
    client = FakeClient(transcript="привет")
    context = FakeContext(client=client)
    message = FakeMessage(voice=FakeVoice())
    await on_voice(make_update(message=message), context)
    assert message.replies[0] == "Transcribing…"
    assert message.edits[-1][0] == NOTE_TEXT
    assert message.edits[-1][1] == {"parse_mode": "HTML"}
    assert client.stt_calls[0]["language"] == "en"
    prompt = client.chat_calls[0]["messages"][0]["content"]
    assert "Write every field in English" in prompt


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



def test_build_application_registers_handlers() -> None:
    application = build_application(make_settings())
    assert application.bot_data["settings"].model_name == "chat-model"
    assert application.bot_data["repository"] is not None
    assert application.bot_data["ai_client"] is not None


def test_transcribe_omits_language_hint_when_absent() -> None:
    client = FakeClient()
    assert transcribe_voice_sync(client, "m", b"a", "audio/ogg") == "hello"
    assert "language" not in client.stt_calls[0]


async def test_stt_language_follows_user_locale() -> None:
    repository = FakeRepository()
    repository.locales[42] = "de"
    client = FakeClient()
    context = FakeContext(client=client, repository=repository)
    message = FakeMessage(voice=FakeVoice())
    await on_voice(make_update(message=message), context)
    assert client.stt_calls[0]["language"] == "de"


async def test_stt_language_follows_telegram_language_code() -> None:
    client = FakeClient()
    context = FakeContext(client=client)
    update = make_update(
        message=FakeMessage(voice=FakeVoice()),
        user=FakeUser(language_code="it-IT"),
    )
    await on_voice(update, context)
    assert client.stt_calls[0]["language"] == "it"


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



async def test_locale_code_without_user() -> None:
    context = FakeContext()
    update = make_update()
    update.effective_user = None
    assert await _locale_code(update, context) == "en"


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





async def _arm_edit(context: FakeContext) -> None:
    """Run /edit against the fake note so the next text is an edit batch."""
    command = FakeMessage(
        reply_to_message=FakeMessage(message_id=55, text=NOTE_TEXT)
    )
    await cmd_edit(make_update(message=command), context)























NOTE_TWO = VoiceNote(
    note_title="T", list_title="L", items=["first", "second"]
)
NOTE_TWO_TEXT = voice_note_to_telegram_html(NOTE_TWO)


def _ctx(**kwargs):
    return FakeContext(**kwargs)


def _reply(text: str = NOTE_TWO_TEXT, message_id: int = 55):
    return FakeMessage(reply_to_message=FakeMessage(
        message_id=message_id, text=text))


async def test_add_with_text_appends_immediately() -> None:
    context = _ctx(args=["buy", "bread"])
    message = _reply()
    await cmd_add(make_update(message=message), context)
    assert context.user_data == {}  # never armed
    edited = context.bot.edited[-1]
    assert "buy bread" in edited["text"]
    assert edited["message_id"] == 55
    assert context.bot.sent[-1]["text"] == "Note updated."


async def test_add_without_text_arms_for_one_minute() -> None:
    context = _ctx(args=[])
    message = _reply()
    await cmd_add(make_update(message=message), context)
    pending = context.user_data["pending_voice"]
    assert pending["mode"] == MODE_ADD
    assert pending["target_message_id"] == 55
    assert pending["status_message_id"] == 2
    assert context.bot.edited == []


async def test_edit_with_text_replaces_quoted_fragment() -> None:
    context = _ctx(args=["second", "wording"])
    message = _reply()
    message.quote = SimpleNamespace(text="second")
    await cmd_edit(make_update(message=message), context)
    edited = context.bot.edited[-1]["text"]
    assert "second wording" in edited
    assert "first" in edited


async def test_edit_without_text_arms_for_voice() -> None:
    context = _ctx(args=[])
    message = _reply()
    message.quote = SimpleNamespace(text="second")
    await cmd_edit(make_update(message=message), context)
    pending = context.user_data["pending_voice"]
    assert pending["mode"] == MODE_EDIT
    assert pending["quote"] == "second"


async def test_edit_without_quote_refuses() -> None:
    context = _ctx(args=["nope"])
    message = _reply()
    await cmd_edit(make_update(message=message), context)
    assert "quote" in message.replies[-1].lower()
    assert context.bot.edited == []


async def test_both_commands_require_reply_and_note() -> None:
    for handler in (cmd_add, cmd_edit):
        context = _ctx(args=["x"])
        message = FakeMessage()
        await handler(make_update(message=message), context)
        assert message.replies[-1].startswith("Reply to the note")
        message = FakeMessage(reply_to_message=FakeMessage(text="hi"))
        await handler(make_update(message=message), context)
        assert "couldn't parse" in message.replies[-1]


async def test_edit_quote_miss_reports_it() -> None:
    context = _ctx(args=["x"])
    message = _reply()
    message.quote = SimpleNamespace(text="nothing like this")
    await cmd_edit(make_update(message=message), context)
    assert "nothing like this" in context.bot.sent[-1]["text"]
    assert context.bot.edited == []


async def test_voice_applies_armed_add() -> None:
    # the model splits a dictated add-on into bullets
    context = _ctx(client=FakeClient(transcript="dictated point"))
    message = _reply()
    await cmd_add(make_update(message=message), context)  # arm
    await on_voice(
        make_update(message=FakeMessage(voice=FakeVoice())), context
    )
    assert "dictated point" in context.bot.edited[-1]["text"]
    assert context.user_data == {}


async def test_voice_applies_armed_edit_replacing_quote() -> None:
    context = _ctx(client=FakeClient(transcript="from voice"))
    message = _reply()
    message.quote = SimpleNamespace(text="second")
    await cmd_edit(make_update(message=message), context)  # arm
    await on_voice(
        make_update(message=FakeMessage(voice=FakeVoice())), context
    )
    edited = context.bot.edited[-1]["text"]
    assert "from voice" in edited
    assert "first" in edited


async def test_voice_without_arm_creates_new_note() -> None:
    client = FakeClient(transcript="hello")
    context = _ctx(client=client)
    message = FakeMessage(voice=FakeVoice())
    await on_voice(make_update(message=message), context)
    assert message.edits[-1][0] == NOTE_TEXT
    assert "Summarizing into notes…" in [e[0] for e in message.edits]


async def test_cancel_clears_armed_session() -> None:
    context = _ctx()
    await cmd_add(make_update(message=_reply()), context)
    message = FakeMessage()
    await cmd_cancel(make_update(message=message), context)
    assert message.replies[-1] == "Cancelled merge mode."
    assert context.user_data == {}
    await cmd_cancel(make_update(message=message), context)
    assert message.replies[-1] == "Nothing to cancel."


async def test_expired_session_voice_starts_a_new_note() -> None:
    context = _ctx(client=FakeClient(transcript="hello"))
    await cmd_add(make_update(message=_reply()), context)
    context.user_data["pending_voice"]["deadline_mono"] = 0.0
    message = FakeMessage(voice=FakeVoice())
    await on_voice(make_update(message=message), context)
    assert message.edits[-1][0] == NOTE_TEXT
    assert context.user_data == {}


async def test_armed_voice_with_lost_note_reports_failure() -> None:
    context = _ctx(client=FakeClient(transcript="hi"))
    await cmd_add(make_update(message=_reply()), context)
    context.user_data["pending_voice"]["note_json"] = "{broken"
    message = FakeMessage(voice=FakeVoice())
    await on_voice(make_update(message=message), context)
    assert "Could not update the note" in context.bot.edited[-1]["text"]


async def test_plain_text_creates_a_note_like_voice() -> None:
    context = _ctx(client=FakeClient(transcript="ignored"))
    message = FakeMessage(text="buy milk and call mom")
    await on_text(make_update(message=message), context)
    # same pipeline: progress reply, then the rendered note edited into it
    assert message.replies[0] == "Summarizing into notes…"
    assert message.edits[-1][0] == NOTE_TEXT
    assert context.bot.edited == []


async def test_plain_text_blank_or_missing_user_is_ignored() -> None:
    context = _ctx(client=FakeClient())
    await on_text(make_update(message=FakeMessage(text="   ")), context)
    assert context.bot.sent == []
    update = make_update(message=FakeMessage(text="hello"))
    update.effective_user = None
    await on_text(update, context)
    assert context.bot.sent == []


async def test_plain_text_note_failure_reports_in_status() -> None:
    context = _ctx(client=FakeClient(fail=True))
    message = FakeMessage(text="hi")
    await on_text(make_update(message=message), context)
    assert "Could not build notes" in message.edits[-1][0]


async def test_add_text_splits_several_points() -> None:
    context = _ctx(args=["buy", "milk", "and", "call", "mom"])
    message = _reply()
    # one line with both ideas: the model splits it into two bullets
    context.bot_data["ai_client"] = FakeClient()
    await cmd_add(make_update(message=message), context)
    edited = context.bot.edited[-1]["text"]
    assert "buy milk and call mom" in edited
    assert context.bot.sent[-1]["text"] == "Note updated."


async def test_add_split_falls_back_to_one_bullet_on_error() -> None:
    context = _ctx(args=["one", "point"])
    context.bot_data["ai_client"] = FakeClient(fail=True)
    message = _reply()
    await cmd_add(make_update(message=message), context)
    assert "one point" in context.bot.edited[-1]["text"]


async def test_edit_does_not_split_quoted_replacement() -> None:
    context = _ctx(args=["two", "words"])
    message = _reply()
    message.quote = SimpleNamespace(text="second")
    await cmd_edit(make_update(message=message), context)
    edited = context.bot.edited[-1]["text"]
    assert "two words" in edited
