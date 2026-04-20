from __future__ import annotations

import asyncio
import logging
import time
from io import BytesIO

from google import genai
from google.genai import types
from telegram import Update
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
from notesbro_bot.note_summary import (
    merge_transcript_into_note_sync,
    parse_voice_note_from_message_text,
    summarize_transcript_to_note_sync,
    voice_note_to_telegram_html,
)

LOGGER = logging.getLogger(__name__)

_GEMINI_RETRIES = 3

_ACCESS_CLOSED_TEXT = (
    "This bot is not accepting new users. If you should have access, ask the "
    "administrator to add your Telegram user id to the database."
)

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


def transcribe_voice_sync(
    client: genai.Client,
    model_name: str,
    audio: bytes,
    mime_type: str,
) -> str:
    import time as _time

    for attempt in range(1, _GEMINI_RETRIES + 1):
        try:
            response = client.models.generate_content(
                model=model_name,
                contents=[
                    (
                        "Transcribe this speech accurately. "
                        "Reply with only the transcribed words, "
                        "no labels or commentary."
                    ),
                    types.Part.from_bytes(data=audio, mime_type=mime_type),
                ],
            )
            return (response.text or "").strip()
        except Exception:
            if attempt < _GEMINI_RETRIES:
                _time.sleep(0.8 * attempt)
                continue
            raise


async def _is_access_denied(
    update: Update,
    context: ContextTypes.DEFAULT_TYPE,
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
        text=_ACCESS_CLOSED_TEXT,
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
    if await _is_access_denied(update, context):
        return
    repository: NoteRepository = context.application.bot_data["repository"]
    await repository.upsert_user(
        user_id=update.effective_user.id,
        chat_id=update.effective_chat.id,
    )
    if update.message:
        await update.message.reply_text(
            "Send a voice message. I will transcribe it with Gemini, then "
            "reply "
            "with summarized notes: a note title, a list heading, and bullet "
            "points.\n\n"
            "To extend a note: reply /add to the note message you want to "
            "update, then send your next voice message within two minutes. "
            "/cancel clears an armed /add. (No message ids are stored.)"
        )


async def cmd_add(update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
    if (
        update.effective_user is None
        or update.effective_chat is None
        or update.message is None
    ):
        return
    if await _is_access_denied(update, context):
        return
    ud = context.user_data
    if ud is None:
        return

    if update.message.reply_to_message is None:
        _disarm_add_voice(ud)
        await update.message.reply_text(
            "Reply /add to the note message you want to update."
        )
        return

    target_message_id = update.message.reply_to_message.message_id
    prev_text = update.message.reply_to_message.text or ""
    previous = parse_voice_note_from_message_text(prev_text)
    if previous is None:
        _disarm_add_voice(ud)
        await update.message.reply_text(
            "I couldn't parse that replied message as a note. "
            "Reply /add to a note message previously sent by this bot."
        )
        return

    _arm_add_voice(ud)
    ud[_UD_ADD_TARGET_MESSAGE_ID] = int(target_message_id)
    ud[_UD_ADD_PREVIOUS_NOTE_JSON] = previous.model_dump_json()
    status_msg = await update.message.reply_text(
        f"OK. Send 1 voice within {_ADD_VOICE_TTL_SEC:.0f}s to merge. "
        "Use /cancel to stop."
    )
    ud[_UD_ADD_STATUS_MESSAGE_ID] = int(status_msg.message_id)


async def cmd_cancel(
    update: Update, context: ContextTypes.DEFAULT_TYPE
) -> None:
    if update.effective_user is None or update.message is None:
        return
    if await _is_access_denied(update, context):
        return
    ud = context.user_data
    if ud is None:
        return
    had = bool(ud.get(_UD_AWAITING_ADD_VOICE))
    _disarm_add_voice(ud)
    if had:
        await update.message.reply_text("Cancelled merge mode.")
    else:
        await update.message.reply_text("Nothing to cancel.")

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


async def on_voice(update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
    if not update.message or not update.message.voice:
        return
    if update.effective_user is None or update.effective_chat is None:
        return
    if await _is_access_denied(update, context):
        return

    settings: Settings = context.bot_data["settings"]
    client: genai.Client = context.bot_data["genai_client"]
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
                    from notesbro_bot.models import VoiceNote

                    previous = VoiceNote.model_validate_json(prev_json)
                except Exception:
                    previous = None
            if previous is None:
                _disarm_add_voice(ud)
            else:
                _disarm_add_voice(ud)
                merge = True

    voice = update.message.voice
    mime = voice.mime_type or "audio/ogg"
    status = None
    if merge and add_status_message_id is not None:
        await _safe_edit_message(
            context,
            chat_id=chat_id,
            message_id=int(add_status_message_id),
            text="Transcribing add-on…",
        )
    else:
        status = await update.message.reply_text(
            "Transcribing…" if not merge else "Transcribing add-on…"
        )

    buf = BytesIO()
    try:
        tg_file = await context.bot.get_file(voice.file_id)
        await tg_file.download_to_memory(buf)
        audio = buf.getvalue()
    except Exception as e:
        LOGGER.exception("Telegram download failed")
        if status is not None:
            await status.edit_text(f"Could not download the voice file: {e}")
        elif merge and add_status_message_id is not None:
            await _safe_edit_message(
                context,
                chat_id=chat_id,
                message_id=int(add_status_message_id),
                text=f"Could not download the voice file: {e}",
            )
        return

    try:
        transcript = await asyncio.to_thread(
            transcribe_voice_sync,
            client,
            settings.model_name,
            audio,
            mime,
        )
    except Exception as e:
        LOGGER.exception("Gemini transcription failed")
        if status is not None:
            await status.edit_text(f"Transcription failed: {e}")
        elif merge and add_status_message_id is not None:
            await _safe_edit_message(
                context,
                chat_id=chat_id,
                message_id=int(add_status_message_id),
                text=f"Transcription failed: {e}",
            )
        return

    if not transcript:
        if status is not None:
            await status.edit_text(
                "(No speech recognized; try again or speak closer.)"
            )
        elif merge and add_status_message_id is not None:
            await _safe_edit_message(
                context,
                chat_id=chat_id,
                message_id=int(add_status_message_id),
                text="(No speech recognized; try again or speak closer.)",
            )
        return

    if status is not None:
        await status.edit_text(
            "Merging into your note…" if merge else "Summarizing into notes…"
        )
    elif merge and add_status_message_id is not None:
        await _safe_edit_message(
            context,
            chat_id=chat_id,
            message_id=int(add_status_message_id),
            text="Merging into your note…",
        )
    try:
        if merge:
            assert previous is not None
            note = await asyncio.to_thread(
                merge_transcript_into_note_sync,
                client,
                settings.model_name,
                previous,
                transcript,
            )
        else:
            note = await asyncio.to_thread(
                summarize_transcript_to_note_sync,
                client,
                settings.model_name,
                transcript,
            )
    except Exception as e:
        LOGGER.exception("Gemini note step failed")
        if status is not None:
            await status.edit_text(f"Could not build notes: {e}")
        elif merge and add_status_message_id is not None:
            await _safe_edit_message(
                context,
                chat_id=chat_id,
                message_id=int(add_status_message_id),
                text=f"Could not build notes: {e}",
            )
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
            assert status is not None
            await status.edit_text(html_body, parse_mode=ParseMode.HTML)
    except Exception as e:
        LOGGER.exception(
            "Telegram HTML edit failed, sending plain transcript fallback"
        )
        if status is not None:
            await status.edit_text(
                f"(Could not send formatted notes: {e})\n\n{transcript}"
            )
        elif merge and add_status_message_id is not None:
            await _safe_edit_message(
                context,
                chat_id=chat_id,
                message_id=int(add_status_message_id),
                text=f"(Could not send formatted notes: {e})\n\n{transcript}",
            )


def build_application(settings: Settings) -> Application:
    client = genai.Client(api_key=settings.api_key)
    repository = NoteRepository(settings.database_path)
    application = (
        Application.builder()
        .token(settings.telegram_bot_token)
        .post_init(_post_init)
        .concurrent_updates(True)
        .build()
    )
    application.bot_data["settings"] = settings
    application.bot_data["genai_client"] = client
    application.bot_data["repository"] = repository

    application.add_handler(CommandHandler("start", cmd_start))
    application.add_handler(CommandHandler("add", cmd_add))
    application.add_handler(CommandHandler("cancel", cmd_cancel))
    application.add_handler(MessageHandler(filters.VOICE, on_voice))
    return application
