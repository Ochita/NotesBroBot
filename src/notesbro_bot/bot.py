from __future__ import annotations

import logging
import time
from io import BytesIO

from openai import AsyncOpenAI
from telegram import Message, Update
from telegram.constants import ParseMode
from telegram.ext import (
    Application,
    CommandHandler,
    ContextTypes,
    MessageHandler,
    filters,
)

from notesbro_bot.config import Settings
from notesbro_bot.db import NoteRepository
from notesbro_bot.i18n import detect_locale, supported_locales, translate
from notesbro_bot.models import VoiceNote
from notesbro_bot.note_summary import (
    add_items,
    parse_voice_note_from_message_text,
    replace_quoted_text,
    split_into_items,
    summarize_text_to_note,
    voice_note_to_telegram_html,
)

LOGGER = logging.getLogger(__name__)

#: The OpenAI SDK retries network/5xx errors this many times on its own.
_API_RETRIES = 3

# One armed voice session shared by /add and /edit. All values live under
# a single key so /cancel clears the whole thing atomically.
_UD_PENDING = "pending_voice"

MODE_ADD = "add"
MODE_EDIT = "edit"

_VOICE_TTL_SEC = 60.0


def _arm(
    user_data: dict,
    mode: str,
    target_message_id: int,
    note: VoiceNote | None = None,
    quote: str | None = None,
    status_message_id: int | None = None,
    ttl_sec: float = _VOICE_TTL_SEC,
) -> None:
    user_data[_UD_PENDING] = {
        "mode": mode,
        "deadline_mono": time.monotonic() + ttl_sec,
        "target_message_id": int(target_message_id),
        "note_json": note.model_dump_json() if note else None,
        "quote": quote,
        "status_message_id": status_message_id,
    }


def _pending(user_data: dict) -> dict | None:
    """Armed session, or ``None`` when absent/expired (and then cleared)."""
    pending = user_data.get(_UD_PENDING)
    if not isinstance(pending, dict):
        return None
    if time.monotonic() > float(pending.get("deadline_mono", 0)):
        _disarm(user_data)
        return None
    return pending


def _disarm(user_data: dict) -> bool:
    """Drop the armed session. Returns whether one was actually armed."""
    return user_data.pop(_UD_PENDING, None) is not None


def _audio_file(audio: bytes, mime_type: str) -> tuple[str, bytes, str]:
    mime = (mime_type or "audio/ogg").split(";", 1)[0].strip().lower()
    if not mime:
        mime = "audio/ogg"
    ext = "mp3" if mime == "audio/mpeg" else mime.rsplit("/", 1)[-1]
    return (f"voice.{ext}", audio, mime)


async def transcribe_voice(
    client: AsyncOpenAI,
    model: str,
    audio: bytes,
    mime_type: str,
    language: str | None = None,
) -> str:
    """Transcribe audio, hinting the user's locale as the spoken language."""
    audio_file = _audio_file(audio, mime_type)
    kwargs: dict = {"model": model, "file": audio_file}
    if language:
        kwargs["language"] = language
    transcript = await client.audio.transcriptions.create(**kwargs)
    return (transcript.text or "").strip()


async def _guard(
    update: Update, context: ContextTypes.DEFAULT_TYPE
) -> str | None:
    """Resolve the locale and enforce the whitelist. ``None`` means stop."""
    user = update.effective_user
    chat = update.effective_chat
    if user is None or chat is None:
        return None
    settings: Settings = context.application.bot_data["settings"]
    repository: NoteRepository = context.application.bot_data["repository"]
    locale = await repository.get_user_locale(user.id)
    if locale not in supported_locales():
        locale = detect_locale(user.language_code, settings.default_locale)
    if settings.allow_new_users or await repository.user_exists(user.id):
        return locale
    await context.bot.send_message(
        chat_id=chat.id,
        text=translate(locale, "access_closed"),
    )
    return None


async def _post_init(application: Application) -> None:
    repository: NoteRepository = application.bot_data["repository"]
    await repository.init()
    LOGGER.info("Database initialized")


async def _post_shutdown(application: Application) -> None:
    client: AsyncOpenAI | None = application.bot_data.get("ai_client")
    if client is not None:
        await client.close()


async def cmd_start(
    update: Update, context: ContextTypes.DEFAULT_TYPE
) -> None:
    locale = await _guard(update, context)
    if locale is None:
        return
    repository: NoteRepository = context.application.bot_data["repository"]
    await repository.upsert_user(
        user_id=update.effective_user.id,
        chat_id=update.effective_chat.id,
    )
    if update.message:
        await update.message.reply_text(translate(locale, "start"))


async def cmd_help(update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
    locale = await _guard(update, context)
    if locale is None:
        return
    await context.bot.send_message(
        chat_id=update.effective_chat.id,
        text=translate(locale, "help"),
    )


async def cmd_set_locale(
    update: Update, context: ContextTypes.DEFAULT_TYPE
) -> None:
    locale = await _guard(update, context)
    if locale is None:
        return
    chat_id = update.effective_chat.id
    args = context.args or []
    if not args:
        await context.bot.send_message(
            chat_id=chat_id,
            text=translate(locale, "set_locale_missing"),
        )
        return
    chosen = args[0].strip().lower().split("-", maxsplit=1)[0]
    if chosen not in supported_locales():
        await context.bot.send_message(
            chat_id=chat_id,
            text=translate(locale, "set_locale_invalid"),
        )
        return
    repository: NoteRepository = context.application.bot_data["repository"]
    await repository.upsert_user(
        user_id=update.effective_user.id,
        chat_id=chat_id,
    )
    await repository.set_user_locale(update.effective_user.id, chosen)
    await context.bot.send_message(
        chat_id=chat_id,
        text=translate(chosen, "set_locale_saved", code=chosen),
    )


def _joined_args(context: ContextTypes.DEFAULT_TYPE) -> str:
    return " ".join(context.args or []).strip()


async def _arm_or_apply_text(
    update: Update,
    context: ContextTypes.DEFAULT_TYPE,
    *,
    locale: str,
    mode: str,
    typed: str,
) -> None:
    """Shared body of /add and /edit.

    Text given inline is applied immediately; without text the session is
    armed and the next voice message is applied the same way in one minute.
    """
    ud = context.user_data
    if ud is None or update.message is None:
        return
    chat_id = update.effective_chat.id  # type: ignore[union-attr]
    target = update.message.reply_to_message
    previous = parse_voice_note_from_message_text(
        (target.text if target else "") or ""
    )
    if target is None:
        _disarm(ud)
        await update.message.reply_text(translate(locale, "need_reply"))
        return
    if previous is None:
        _disarm(ud)
        await update.message.reply_text(translate(locale, "not_a_note"))
        return

    quote = _quote_text(update.message)
    if mode == MODE_EDIT and not quote:
        _disarm(ud)
        await update.message.reply_text(translate(locale, "edit_need_quote"))
        return

    if typed:
        _disarm(ud)
        await _apply_text(
            context,
            locale=locale,
            mode=mode,
            note=previous,
            quote=quote,
            text=typed,
            chat_id=chat_id,
            target_message_id=int(target.message_id),
            split=mode == MODE_ADD,
        )
        return

    status = await update.message.reply_text(
        translate(locale, f"{mode}_armed", ttl=f"{_VOICE_TTL_SEC:.0f}")
    )
    _arm(
        ud,
        mode,
        target_message_id=int(target.message_id),
        note=previous,
        quote=quote,
        status_message_id=int(status.message_id),
    )


async def _apply_text(
    context: ContextTypes.DEFAULT_TYPE,
    *,
    locale: str,
    mode: str,
    note: VoiceNote,
    quote: str | None,
    text: str,
    chat_id: int,
    target_message_id: int,
    split: bool = False,
) -> None:
    """Apply typed or transcribed text as an append or a quoted replace.

    ``split`` asks the model to break the text into several bullets first,
    which matters for /add because one message may carry many points.
    """
    if mode == MODE_EDIT:
        updated = replace_quoted_text(note, quote, text)
        if updated is None:
            await context.bot.send_message(
                chat_id=chat_id,
                text=translate(locale, "edit_quote_miss", quoted=quote or ""),
            )
            return
    else:
        items = await _bullets(context, locale, text, split=split)
        updated = add_items(note, items)
        if updated is None:
            await context.bot.send_message(
                chat_id=chat_id,
                text=translate(locale, "add_full", count=str(len(note.items))),
            )
            return
    if not await _push_note(context, chat_id, target_message_id, updated):
        await context.bot.send_message(
            chat_id=chat_id, text=translate(locale, "note_update_failed")
        )
        return
    await context.bot.send_message(
        chat_id=chat_id,
        text=translate(locale, "note_updated"),
    )


async def _bullets(
    context: ContextTypes.DEFAULT_TYPE,
    locale: str,
    text: str,
    *,
    split: bool,
) -> list[str]:
    """Bullets for /add: the model splits them, with a plain fallback."""
    if not split:
        return [text]
    settings: Settings = context.bot_data["settings"]
    client: AsyncOpenAI = context.bot_data["ai_client"]
    try:
        items = await split_into_items(
            client,
            settings.model_name,
            text,
            locale,
        )
    except Exception:
        LOGGER.exception("Splitting into bullets failed, using raw text")
        return [text]
    return items or [text]


async def cmd_add(update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
    locale = await _guard(update, context)
    if locale is None or update.message is None:
        return
    await _arm_or_apply_text(
        update,
        context,
        locale=locale,
        mode=MODE_ADD,
        typed=_joined_args(context),
    )


async def _push_note(
    context: ContextTypes.DEFAULT_TYPE,
    chat_id: int,
    message_id: int,
    note: VoiceNote,
) -> bool:
    """Rewrite a note message in place. Returns ``False`` on Telegram error."""
    try:
        await context.bot.edit_message_text(
            chat_id=chat_id,
            message_id=int(message_id),
            text=voice_note_to_telegram_html(note),
            parse_mode=ParseMode.HTML,
        )
        return True
    except Exception:
        LOGGER.exception(
            "Telegram HTML edit failed for message %s", message_id
        )
        return False


async def cmd_cancel(
    update: Update, context: ContextTypes.DEFAULT_TYPE
) -> None:
    locale = await _guard(update, context)
    if locale is None or update.message is None:
        return
    ud = context.user_data
    if ud is None:
        return
    key = "cancel_done" if _disarm(ud) else "cancel_nothing"
    await update.message.reply_text(translate(locale, key))


async def _status_update(
    context: ContextTypes.DEFAULT_TYPE,
    *,
    chat_id: int,
    status: Message | None,
    status_id: int | None,
    text: str,
    **kwargs,
) -> None:
    """Rewrite the progress message in place, or send a fresh one."""
    if status is not None:
        await status.edit_text(text, **kwargs)
    elif status_id is not None:
        try:
            await context.bot.edit_message_text(
                chat_id=chat_id,
                message_id=int(status_id),
                text=text,
                **kwargs,
            )
        except Exception:
            # The armed status message may have been deleted by the user.
            LOGGER.debug("Could not edit status message %s", status_id)
    else:
        await context.bot.send_message(chat_id=chat_id, text=text, **kwargs)


async def cmd_edit(update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
    """Replace the quoted fragment; without text the next voice does it."""
    locale = await _guard(update, context)
    if locale is None or update.message is None:
        return
    await _arm_or_apply_text(
        update,
        context,
        locale=locale,
        mode=MODE_EDIT,
        typed=_joined_args(context),
    )


def _quote_text(message: Message) -> str | None:
    """Selected fragment of the replied message, if the user quoted one."""
    quote = getattr(message, "quote", None)
    return getattr(quote, "text", None) if quote is not None else None


def _load_note(raw: object) -> VoiceNote | None:
    """Decode the note JSON stashed in ``user_data`` when arming a flow."""
    if not isinstance(raw, str) or not raw.strip():
        return None
    try:
        return VoiceNote.model_validate_json(raw)
    except Exception:
        return None


async def on_voice(update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
    if not update.message or not update.message.voice:
        return
    locale = await _guard(update, context)
    if locale is None:
        return

    settings: Settings = context.bot_data["settings"]
    client: AsyncOpenAI = context.bot_data["ai_client"]
    chat_id = update.effective_chat.id
    ud = context.user_data or {}

    armed = _pending(ud) is not None
    status_text = (
        "status_transcribing_addon" if armed else "status_transcribing"
    )
    pending_status_id = (ud.get(_UD_PENDING) or {}).get("status_message_id")
    if pending_status_id:
        await _status_update(
            context,
            chat_id=chat_id,
            status=None,
            status_id=pending_status_id,
            text=translate(locale, status_text),
        )
        status: Message | None = None
    else:
        status = await update.message.reply_text(
            translate(locale, status_text)
        )

    async def show(text_key: str, **kwargs: str) -> None:
        await _status_update(
            context,
            chat_id=chat_id,
            status=status,
            status_id=pending_status_id,
            text=translate(locale, text_key, **kwargs),
        )

    voice = update.message.voice
    mime = voice.mime_type or "audio/ogg"
    buf = BytesIO()
    try:
        tg_file = await context.bot.get_file(voice.file_id)
        await tg_file.download_to_memory(buf)
        audio = buf.getvalue()
    except Exception as e:
        LOGGER.exception("Telegram download failed")
        await show("err_download", error=str(e))
        return

    try:
        transcript = await transcribe_voice(
            client,
            settings.stt_model,
            audio,
            mime,
            locale,
        )
    except Exception as e:
        LOGGER.exception("Transcription failed")
        await show("err_transcribe", error=str(e))
        return
    if not transcript:
        await show("no_speech")
        return

    await show("status_merging" if armed else "status_summarizing")
    await _produce(
        update,
        context,
        locale=locale,
        text=transcript,
        status=status,
        status_id=pending_status_id,
    )
    if pending_status_id:
        try:
            await context.bot.delete_message(
                chat_id=chat_id, message_id=int(pending_status_id)
            )
        except Exception:
            LOGGER.debug(
                "Could not delete status message %s", pending_status_id
            )


async def on_text(update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
    """A plain text message creates a note, same pipeline as a voice one."""
    if not update.message or not update.message.text:
        return
    locale = await _guard(update, context)
    if locale is None:
        return
    text = update.message.text.strip()
    if not text:
        return
    status = await update.message.reply_text(
        translate(locale, "status_summarizing")
    )
    await _produce(update, context, locale=locale, text=text, status=status)


async def _produce(
    update: Update,
    context: ContextTypes.DEFAULT_TYPE,
    *,
    locale: str,
    text: str,
    status: Message | None,
    status_id: int | None = None,
) -> None:
    """Turn text into an armed note or a fresh one - typed or spoken alike."""
    settings: Settings = context.bot_data["settings"]
    client: AsyncOpenAI = context.bot_data["ai_client"]
    ud = context.user_data or {}
    chat_id = update.effective_chat.id  # type: ignore[union-attr]

    pending = _pending(ud)
    if pending is not None:
        _disarm(ud)
    armed = pending is not None

    async def fail(key: str, **kwargs: str) -> None:
        """Report an error in the progress message when we have one."""
        await _status_update(
            context,
            chat_id=chat_id,
            status=status,
            status_id=status_id,
            text=translate(locale, key, **kwargs),
        )

    if armed:
        note = _load_note(pending.get("note_json"))
        if note is None:
            await fail("note_update_failed")
            return
        await _apply_text(
            context,
            locale=locale,
            mode=str(pending.get("mode", MODE_ADD)),
            note=note,
            quote=pending.get("quote"),
            text=text,
            chat_id=chat_id,
            target_message_id=int(pending["target_message_id"]),
            split=not pending.get("quote"),
        )
        return

    try:
        note = await summarize_text_to_note(
            client,
            settings.model_name,
            text,
            locale,
        )
    except Exception as e:
        LOGGER.exception("Note step failed")
        await fail("err_notes", error=str(e))
        return

    html_body = voice_note_to_telegram_html(note)
    try:
        await _status_update(
            context,
            chat_id=chat_id,
            status=status,
            status_id=status_id,
            text=html_body,
            parse_mode=ParseMode.HTML,
        )
    except Exception as e:
        LOGGER.exception("Telegram HTML send failed, sending plain text")
        await fail("err_format", error=str(e), transcript=text)


def build_application(settings: Settings) -> Application:
    client = AsyncOpenAI(
        api_key=settings.api_key,
        base_url=settings.base_url,
        max_retries=_API_RETRIES,
    )
    repository = NoteRepository(settings.database_path)
    application = (
        Application.builder()
        .token(settings.telegram_bot_token)
        .post_init(_post_init)
        .post_shutdown(_post_shutdown)
        .concurrent_updates(True)
        .build()
    )
    application.bot_data["settings"] = settings
    application.bot_data["ai_client"] = client
    application.bot_data["repository"] = repository

    application.add_handler(CommandHandler("start", cmd_start))
    application.add_handler(CommandHandler("help", cmd_help))
    application.add_handler(CommandHandler("set_locale", cmd_set_locale))
    application.add_handler(CommandHandler("add", cmd_add))
    application.add_handler(CommandHandler("edit", cmd_edit))
    application.add_handler(CommandHandler("cancel", cmd_cancel))
    application.add_handler(MessageHandler(filters.VOICE, on_voice))
    application.add_handler(
        MessageHandler(filters.TEXT & ~filters.COMMAND, on_text)
    )

    return application
