from __future__ import annotations

import asyncio
import logging
import time
from io import BytesIO

from openai import OpenAI
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
    merge_transcript_into_note_sync,
    parse_voice_note_from_message_text,
    summarize_transcript_to_note_sync,
    voice_note_to_telegram_html,
)

LOGGER = logging.getLogger(__name__)

_API_RETRIES = 3

# user_data keys: next voice after /add is treated as merge.
# (Telegram has no reliable caption on push-to-talk voice.)
_UD_AWAITING_ADD_VOICE = "awaiting_add_voice"
_UD_AWAITING_ADD_DEADLINE_MONO = "awaiting_add_deadline_mono"
_UD_ADD_TARGET_MESSAGE_ID = "add_target_message_id"
_UD_ADD_PREVIOUS_NOTE_JSON = "add_previous_note_json"
_UD_ADD_STATUS_MESSAGE_ID = "add_status_message_id"

_ADD_VOICE_TTL_SEC = 120.0


def _disarm_add_voice(user_data: dict) -> None:
    user_data.pop(_UD_AWAITING_ADD_VOICE, None)
    user_data.pop(_UD_AWAITING_ADD_DEADLINE_MONO, None)
    user_data.pop(_UD_ADD_TARGET_MESSAGE_ID, None)
    user_data.pop(_UD_ADD_PREVIOUS_NOTE_JSON, None)
    user_data.pop(_UD_ADD_STATUS_MESSAGE_ID, None)


def _arm_add_voice(
    user_data: dict, ttl_sec: float = _ADD_VOICE_TTL_SEC
) -> None:
    user_data[_UD_AWAITING_ADD_VOICE] = True
    user_data[_UD_AWAITING_ADD_DEADLINE_MONO] = time.monotonic() + ttl_sec


def _add_voice_arm_expired(user_data: dict) -> bool:
    deadline = user_data.get(_UD_AWAITING_ADD_DEADLINE_MONO)
    if deadline is None:
        return True
    return time.monotonic() > float(deadline)


def _audio_file(audio: bytes, mime_type: str) -> tuple[str, bytes, str]:
    mime = (mime_type or "audio/ogg").split(";", 1)[0].strip().lower()
    if not mime:
        mime = "audio/ogg"
    ext = "mp3" if mime == "audio/mpeg" else mime.rsplit("/", 1)[-1]
    return (f"voice.{ext}", audio, mime)


def transcribe_voice_sync(
    client: OpenAI,
    model: str,
    audio: bytes,
    mime_type: str,
    language: str | None = None,
) -> str:
    """Transcribe audio, hinting the user's locale as the spoken language."""
    audio_file = _audio_file(audio, mime_type)
    for attempt in range(1, _API_RETRIES + 1):
        try:
            kwargs: dict = {"model": model, "file": audio_file}
            if language:
                kwargs["language"] = language
            transcript = client.audio.transcriptions.create(**kwargs)
            return (transcript.text or "").strip()
        except Exception:
            if attempt < _API_RETRIES:
                time.sleep(0.8 * attempt)
                continue
            raise


async def _locale_code(
    update: Update,
    context: ContextTypes.DEFAULT_TYPE,
) -> str:
    settings: Settings = context.application.bot_data["settings"]
    repository: NoteRepository = context.application.bot_data["repository"]
    user = update.effective_user
    if user is not None:
        stored = await repository.get_user_locale(user.id)
        if stored in supported_locales():
            return stored
    language_code = user.language_code if user is not None else None
    return detect_locale(language_code, settings.default_locale)


async def _is_access_denied(
    update: Update,
    context: ContextTypes.DEFAULT_TYPE,
    locale: str,
) -> bool:
    settings: Settings = context.application.bot_data["settings"]
    repository: NoteRepository = context.application.bot_data["repository"]
    if settings.allow_new_users:
        return False
    user = update.effective_user
    chat = update.effective_chat
    if user is None or chat is None:
        return True
    if await repository.user_exists(user.id):
        return False
    await context.bot.send_message(
        chat_id=chat.id,
        text=translate(locale, "access_closed"),
    )
    return True


async def _post_init(application: Application) -> None:
    repository: NoteRepository = application.bot_data["repository"]
    await repository.init()
    LOGGER.info("Database initialized")


async def cmd_start(
    update: Update, context: ContextTypes.DEFAULT_TYPE
) -> None:
    if update.effective_user is None or update.effective_chat is None:
        return
    locale = await _locale_code(update, context)
    if await _is_access_denied(update, context, locale):
        return
    repository: NoteRepository = context.application.bot_data["repository"]
    await repository.upsert_user(
        user_id=update.effective_user.id,
        chat_id=update.effective_chat.id,
    )
    if update.message:
        await update.message.reply_text(translate(locale, "start"))


async def cmd_help(update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
    if update.effective_user is None or update.effective_chat is None:
        return
    locale = await _locale_code(update, context)
    if await _is_access_denied(update, context, locale):
        return
    await context.bot.send_message(
        chat_id=update.effective_chat.id,
        text=translate(locale, "help"),
    )


async def cmd_set_locale(
    update: Update, context: ContextTypes.DEFAULT_TYPE
) -> None:
    if update.effective_user is None or update.effective_chat is None:
        return
    locale = await _locale_code(update, context)
    if await _is_access_denied(update, context, locale):
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


async def cmd_add(update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
    if (
        update.effective_user is None
        or update.effective_chat is None
        or update.message is None
    ):
        return
    locale = await _locale_code(update, context)
    if await _is_access_denied(update, context, locale):
        return
    ud = context.user_data
    if ud is None:
        return

    if update.message.reply_to_message is None:
        _disarm_add_voice(ud)
        await update.message.reply_text(translate(locale, "add_need_reply"))
        return

    target_message_id = update.message.reply_to_message.message_id
    prev_text = update.message.reply_to_message.text or ""
    previous = parse_voice_note_from_message_text(prev_text)
    if previous is None:
        _disarm_add_voice(ud)
        await update.message.reply_text(translate(locale, "add_not_a_note"))
        return

    _arm_add_voice(ud)
    ud[_UD_ADD_TARGET_MESSAGE_ID] = int(target_message_id)
    ud[_UD_ADD_PREVIOUS_NOTE_JSON] = previous.model_dump_json()
    status_msg = await update.message.reply_text(
        translate(locale, "add_armed", ttl=f"{_ADD_VOICE_TTL_SEC:.0f}")
    )
    ud[_UD_ADD_STATUS_MESSAGE_ID] = int(status_msg.message_id)


async def cmd_cancel(
    update: Update, context: ContextTypes.DEFAULT_TYPE
) -> None:
    if update.effective_user is None or update.message is None:
        return
    locale = await _locale_code(update, context)
    if await _is_access_denied(update, context, locale):
        return
    ud = context.user_data
    if ud is None:
        return
    had = bool(ud.get(_UD_AWAITING_ADD_VOICE))
    _disarm_add_voice(ud)
    key = "cancel_done" if had else "cancel_nothing"
    await update.message.reply_text(translate(locale, key))


async def _safe_delete_message(
    context: ContextTypes.DEFAULT_TYPE, *, chat_id: int, message_id: int
) -> None:
    try:
        await context.bot.delete_message(
            chat_id=chat_id, message_id=message_id
        )
    except Exception:
        LOGGER.debug(
            "Could not delete message %s in chat %s", message_id, chat_id
        )


async def _safe_edit_message(
    context: ContextTypes.DEFAULT_TYPE,
    *,
    chat_id: int,
    message_id: int,
    text: str,
) -> None:
    try:
        await context.bot.edit_message_text(
            chat_id=chat_id,
            message_id=message_id,
            text=text,
        )
    except Exception:
        # Message could be too old to edit or already deleted. No-op.
        LOGGER.debug(
            "Could not edit message %s in chat %s", message_id, chat_id
        )


async def _show_status(
    context: ContextTypes.DEFAULT_TYPE,
    *,
    chat_id: int,
    reply_message: Message | None,
    edit_message_id: int | None,
    text: str,
) -> None:
    """Update whichever progress message this flow is using."""
    if reply_message is not None:
        await reply_message.edit_text(text)
    elif edit_message_id is not None:
        await _safe_edit_message(
            context,
            chat_id=chat_id,
            message_id=int(edit_message_id),
            text=text,
        )


async def on_voice(update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
    if not update.message or not update.message.voice:
        return
    if update.effective_user is None or update.effective_chat is None:
        return
    locale = await _locale_code(update, context)
    if await _is_access_denied(update, context, locale):
        return

    settings: Settings = context.bot_data["settings"]
    client: OpenAI = context.bot_data["ai_client"]
    chat_id = update.effective_chat.id
    ud = context.user_data or {}

    merge = False
    previous = None
    target_message_id: int | None = None
    add_status_message_id: int | None = None
    if ud.get(_UD_AWAITING_ADD_VOICE):
        if _add_voice_arm_expired(ud):
            _disarm_add_voice(ud)
        else:
            target_message_id = ud.get(_UD_ADD_TARGET_MESSAGE_ID)
            add_status_message_id = ud.get(_UD_ADD_STATUS_MESSAGE_ID)
            prev_json = ud.get(_UD_ADD_PREVIOUS_NOTE_JSON)
            if isinstance(prev_json, str) and prev_json.strip():
                try:
                    previous = VoiceNote.model_validate_json(prev_json)
                except Exception:
                    previous = None
            _disarm_add_voice(ud)
            merge = previous is not None

    voice = update.message.voice
    mime = voice.mime_type or "audio/ogg"
    status: Message | None = None
    pending_text = (
        "status_transcribing_addon" if merge else "status_transcribing"
    )
    if merge and add_status_message_id is not None:
        await _show_status(
            context,
            chat_id=chat_id,
            reply_message=None,
            edit_message_id=add_status_message_id,
            text=translate(locale, pending_text),
        )
    else:
        status = await update.message.reply_text(
            translate(locale, pending_text)
        )

    async def show(text_key: str, **kwargs: str) -> None:
        await _show_status(
            context,
            chat_id=chat_id,
            reply_message=status,
            edit_message_id=add_status_message_id,
            text=translate(locale, text_key, **kwargs),
        )

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
        transcript = await asyncio.to_thread(
            transcribe_voice_sync,
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

    await show("status_merging" if merge else "status_summarizing")
    try:
        if merge and previous is not None:
            note = await asyncio.to_thread(
                merge_transcript_into_note_sync,
                client,
                settings.model_name,
                previous,
                transcript,
                locale,
            )
        else:
            note = await asyncio.to_thread(
                summarize_transcript_to_note_sync,
                client,
                settings.model_name,
                transcript,
                locale,
            )
    except Exception as e:
        LOGGER.exception("Note step failed")
        await show("err_notes", error=str(e))
        return

    html_body = voice_note_to_telegram_html(note)
    try:
        if merge and target_message_id is not None:
            # Update the original note message.
            await context.bot.edit_message_text(
                chat_id=chat_id,
                message_id=int(target_message_id),
                text=html_body,
                parse_mode=ParseMode.HTML,
            )
            # Clear helper/status message for a cleaner chat.
            if add_status_message_id is not None:
                await _safe_delete_message(
                    context,
                    chat_id=chat_id,
                    message_id=int(add_status_message_id),
                )
            if status is not None:
                await _safe_delete_message(
                    context,
                    chat_id=chat_id,
                    message_id=int(status.message_id),
                )
        else:
            if status is not None:
                await status.edit_text(html_body, parse_mode=ParseMode.HTML)
    except Exception as e:
        LOGGER.exception(
            "Telegram HTML edit failed, sending plain transcript fallback"
        )
        await show("err_format", error=str(e), transcript=transcript)


def build_application(settings: Settings) -> Application:
    client = OpenAI(
        api_key=settings.api_key,
        base_url=settings.base_url,
        max_retries=0,
    )
    repository = NoteRepository(settings.database_path)
    application = (
        Application.builder()
        .token(settings.telegram_bot_token)
        .post_init(_post_init)
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
    application.add_handler(CommandHandler("cancel", cmd_cancel))
    application.add_handler(MessageHandler(filters.VOICE, on_voice))
    return application
