from __future__ import annotations

import asyncio
import logging
import time
from io import BytesIO

from openai import OpenAI
from telegram import (
    InlineKeyboardButton,
    InlineKeyboardMarkup,
    Message,
    Update,
)
from telegram.constants import ParseMode
from telegram.ext import (
    Application,
    CallbackQueryHandler,
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
    _parse_item_number,
    apply_note_edits,
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

# user_data keys for /edit: next message is a set of line edits.
_UD_AWAITING_EDIT = "awaiting_edit"
_UD_EDIT_DEADLINE_MONO = "awaiting_edit_deadline_mono"
_UD_EDIT_TARGET_MESSAGE_ID = "edit_target_message_id"
_UD_EDIT_PREVIOUS_NOTE_JSON = "edit_previous_note_json"
_UD_EDIT_STATUS_MESSAGE_ID = "edit_status_message_id"
_UD_EDIT_SELECTED_ITEM = "edit_selected_item"

_EDIT_TTL_SEC = 300.0


def _disarm_add_voice(user_data: dict) -> None:
    user_data.pop(_UD_AWAITING_ADD_VOICE, None)
    user_data.pop(_UD_AWAITING_ADD_DEADLINE_MONO, None)
    user_data.pop(_UD_ADD_TARGET_MESSAGE_ID, None)
    user_data.pop(_UD_ADD_PREVIOUS_NOTE_JSON, None)
    user_data.pop(_UD_ADD_STATUS_MESSAGE_ID, None)


def _arm_edit(user_data: dict, ttl_sec: float = _EDIT_TTL_SEC) -> None:
    user_data[_UD_AWAITING_EDIT] = True
    user_data[_UD_EDIT_DEADLINE_MONO] = time.monotonic() + ttl_sec


def _disarm_edit(user_data: dict) -> None:
    user_data.pop(_UD_AWAITING_EDIT, None)
    user_data.pop(_UD_EDIT_DEADLINE_MONO, None)
    user_data.pop(_UD_EDIT_TARGET_MESSAGE_ID, None)
    user_data.pop(_UD_EDIT_PREVIOUS_NOTE_JSON, None)
    user_data.pop(_UD_EDIT_STATUS_MESSAGE_ID, None)
    user_data.pop(_UD_EDIT_SELECTED_ITEM, None)


def _edit_arm_expired(user_data: dict) -> bool:
    deadline = user_data.get(_UD_EDIT_DEADLINE_MONO)
    if deadline is None:
        return True
    return time.monotonic() > float(deadline)



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
    had_edit = bool(ud.get(_UD_AWAITING_EDIT))
    had_add = bool(ud.get(_UD_AWAITING_ADD_VOICE))
    _disarm_add_voice(ud)
    _disarm_edit(ud)
    if had_edit:
        key = "edit_cancelled"
    elif had_add:
        key = "cancel_done"
    else:
        key = "cancel_nothing"
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
    reply_markup: InlineKeyboardMarkup | None = None,
) -> None:
    kwargs = {"reply_markup": reply_markup} if reply_markup else {}
    try:
        await context.bot.edit_message_text(
            chat_id=chat_id,
            message_id=message_id,
            text=text,
            **kwargs,
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
    parse_mode: str | None = None,
) -> None:
    """Update whichever progress message this flow is using."""
    if reply_message is not None:
        kwargs = {"parse_mode": parse_mode} if parse_mode else {}
        await reply_message.edit_text(text, **kwargs)
    elif edit_message_id is not None:
        await _safe_edit_message(
            context,
            chat_id=chat_id,
            message_id=int(edit_message_id),
            text=text,
        )


async def cmd_edit(update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
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

    target = update.message.reply_to_message
    previous = parse_voice_note_from_message_text(
        (target.text if target else "") or ""
    )
    _disarm_add_voice(ud)
    if target is None:
        await update.message.reply_text(translate(locale, "edit_need_reply"))
        return
    if previous is None:
        await update.message.reply_text(translate(locale, "edit_not_a_note"))
        return

    _arm_edit(ud)
    ud[_UD_EDIT_TARGET_MESSAGE_ID] = int(target.message_id)
    ud[_UD_EDIT_PREVIOUS_NOTE_JSON] = previous.model_dump_json()
    status = await update.message.reply_text(
        _number_lines(previous) + "\n\n" + translate(locale, "edit_prompt"),
        reply_markup=_item_keyboard(previous),
    )
    ud[_UD_EDIT_STATUS_MESSAGE_ID] = int(status.message_id)


def _item_keyboard(note: VoiceNote) -> InlineKeyboardMarkup:
    """One button per item: tapping arms it for a plain-text replace."""
    buttons = [
        InlineKeyboardButton(str(number), callback_data=f"ed:{number}")
        for number in range(1, len(note.items) + 1)
    ]
    return InlineKeyboardMarkup(
        [buttons[i : i + 5] for i in range(0, len(buttons), 5)]
    )


async def on_edit_select(
    update: Update, context: ContextTypes.DEFAULT_TYPE
) -> None:
    """User tapped an item number: next plain text replaces just that item."""
    query = update.callback_query
    if query is None or query.data is None:
        return
    ud = context.user_data or {}
    locale = await _locale_code(update, context)
    if query.message is not None:
        await query.answer()
    if not ud.get(_UD_AWAITING_EDIT) or _edit_arm_expired(ud):
        if query.message is not None:
            await _safe_edit_message(
                context,
                chat_id=query.message.chat_id,
                message_id=query.message.message_id,
                text=translate(locale, "edit_expired"),
            )
        return
    previous = _load_note(ud.get(_UD_EDIT_PREVIOUS_NOTE_JSON))
    index = _parse_item_number(query.data.split(":", 1)[-1])
    if previous is None or index is None or index >= len(previous.items):
        if query.message is not None:
            await _safe_edit_message(
                context,
                chat_id=query.message.chat_id,
                message_id=query.message.message_id,
                text=translate(locale, "edit_nothing"),
            )
        return
    ud[_UD_EDIT_SELECTED_ITEM] = index
    if query.message is not None:
        await _safe_edit_message(
            context,
            chat_id=query.message.chat_id,
            message_id=query.message.message_id,
            text=translate(
                locale,
                "edit_selected",
                number=str(index + 1),
                current=previous.items[index],
            ),
            reply_markup=_item_keyboard(previous),
        )


async def on_edit_text(
    update: Update, context: ContextTypes.DEFAULT_TYPE
) -> None:
    """Apply ``2 fixed text`` style line edits to the armed note."""
    if not update.message or not update.message.text:
        return
    ud = context.user_data or {}
    if not ud.get(_UD_AWAITING_EDIT):
        return
    if _edit_arm_expired(ud):
        _disarm_edit(ud)
        return
    target_message_id = ud.get(_UD_EDIT_TARGET_MESSAGE_ID)
    status_message_id = ud.get(_UD_EDIT_STATUS_MESSAGE_ID)
    previous = _load_note(ud.get(_UD_EDIT_PREVIOUS_NOTE_JSON))
    # A tapped item number means the raw text is the replacement for it.
    selected = ud.get(_UD_EDIT_SELECTED_ITEM)
    _disarm_edit(ud)
    if target_message_id is None or previous is None:
        return

    locale = await _locale_code(update, context)
    if await _is_access_denied(update, context, locale):
        return
    chat_id = update.effective_chat.id
    # Only reuse a message we actually own and can edit; never post another
    # instruction block just because the user replied to something else.
    status = update.message.reply_to_message
    if status is None or status.message_id != status_message_id:
        status = None

    async def notify(key: str, **kwargs: str) -> None:
        text = translate(locale, key, **kwargs)
        if status is not None:
            await status.edit_text(text)
        else:
            await update.message.reply_text(text)

    raw = update.message.text
    if isinstance(selected, int) and not raw.lstrip().startswith(("+", "-")):
        # Tapped a button: the whole message is the new text for that item.
        if raw.strip():
            raw = f"{selected + 1} {raw.strip()}"

    edited, applied = apply_note_edits(previous, raw)
    if not applied:
        await notify("edit_nothing")
        return
    if edited == previous:
        # Telegram rejects a no-op edit with "message is not modified".
        await notify("edit_same")
        return

    html_body = voice_note_to_telegram_html(edited)
    try:
        await context.bot.edit_message_text(
            chat_id=chat_id,
            message_id=int(target_message_id),
            text=html_body,
            parse_mode=ParseMode.HTML,
        )
    except Exception as e:
        LOGGER.exception("Telegram HTML edit failed")
        await notify("err_format", error=str(e), transcript="")
        return
    # Clear the instruction block and the user's edit lines, then confirm.
    if status_message_id is not None:
        await _safe_delete_message(
            context, chat_id=chat_id, message_id=int(status_message_id)
        )
    await _safe_delete_message(
        context, chat_id=chat_id, message_id=update.message.message_id
    )
    await update.message.reply_text(
        translate(locale, "edit_done", count=str(applied))
    )


def _load_note(raw: object) -> VoiceNote | None:
    """Decode the note JSON stashed in ``user_data`` when arming a flow."""
    if not isinstance(raw, str) or not raw.strip():
        return None
    try:
        return VoiceNote.model_validate_json(raw)
    except Exception:
        return None


def _number_lines(note: VoiceNote) -> str:
    return "\n".join(
        f"{index}. {item}" for index, item in enumerate(note.items, start=1)
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
            previous = _load_note(ud.get(_UD_ADD_PREVIOUS_NOTE_JSON))
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
    application.add_handler(CommandHandler("edit", cmd_edit))
    application.add_handler(CommandHandler("cancel", cmd_cancel))
    application.add_handler(MessageHandler(filters.VOICE, on_voice))
    application.add_handler(
        CallbackQueryHandler(on_edit_select, pattern=r"^ed:\d+$")
    )
    application.add_handler(
        MessageHandler(filters.TEXT & ~filters.COMMAND, on_edit_text)
    )
    return application
