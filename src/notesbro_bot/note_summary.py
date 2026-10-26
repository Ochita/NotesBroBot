from __future__ import annotations

import html
import logging
import time
from collections.abc import Callable
from typing import TYPE_CHECKING, TypeVar

from notesbro_bot.i18n import translate
from notesbro_bot.models import VoiceNote

if TYPE_CHECKING:
    from openai import OpenAI

LOGGER = logging.getLogger(__name__)

T = TypeVar("T")

_API_RETRIES = 3
_MAX_ITEMS = 16
_MAX_ITEM_CHARS = 200

def _retry(call: Callable[[], T]) -> T:
    """Run ``call`` up to ``_API_RETRIES`` times, backing off between tries."""
    for attempt in range(1, _API_RETRIES + 1):
        try:
            return call()
        except Exception:
            if attempt == _API_RETRIES:
                raise
            time.sleep(0.8 * attempt)
    raise AssertionError("unreachable")


def _complete_note(
    client: "OpenAI", model: str, prompt: str, locale: str
) -> VoiceNote:
    def once() -> VoiceNote:
        response = client.chat.completions.parse(
            model=model,
            messages=[{"role": "user", "content": prompt}],
            response_format=VoiceNote,
        )
        message = response.choices[0].message
        parsed = message.parsed
        if isinstance(parsed, VoiceNote):
            return _normalize_note(parsed, locale)
        text = (message.content or "").strip()
        if text:
            return _normalize_note(
                VoiceNote.model_validate_json(text), locale
            )
        LOGGER.warning("Structured note response empty.")
        raise ValueError("Model returned no structured note.")

    # The empty-response ValueError is retried too, hence the inner function.
    return _retry(once)


_SUMMARY_INSTRUCTION = """Turn a raw speech transcript into structured notes.

Rules:
- note_title: a clear title for the whole note.
- list_title: one heading for the bullet list (not a repeat of note_title).
- items: 3 to 8 short bullets. One idea per line.
- Write every field in {language}, even when the transcript is in another
  language. Translate the content instead of copying the source wording.
- Do not invent facts; only summarize what is implied or stated.
"""


def summarize_transcript_to_note_sync(
    client: "OpenAI",
    model_name: str,
    transcript: str,
    locale: str | None = None,
) -> VoiceNote:
    language = _language_name(locale)
    prompt = (
        _SUMMARY_INSTRUCTION.format(language=language)
        + f"\n\nTranscript:\n{transcript}"
    )
    return _complete_note(client, model_name, prompt, locale or "en")


def _language_name(locale: str | None) -> str:
    """Human-readable name the model can follow, never a bare code."""
    return _LANGUAGE_NAMES.get(locale or "", "English")


_LANGUAGE_NAMES = {
    "en": "English",
    "ru": "Russian",
    "uk": "Ukrainian",
    "it": "Italian",
    "fr": "French",
    "de": "German",
    "es": "Spanish",
}


def _normalize_note(note: VoiceNote, locale: str | None = None) -> VoiceNote:
    title = note.note_title.strip() or translate(
        locale or "en", "note_untitled"
    )
    list_heading = note.list_title.strip() or translate(
        locale or "en", "note_summary_heading"
    )
    items = [s.strip() for s in note.items if s.strip()]
    if not items:
        items = [translate(locale or "en", "note_no_items")]
    return note.model_copy(
        update={
            "note_title": title,
            "list_title": list_heading,
            "items": items,
        }
    )


def voice_note_to_telegram_html(note: VoiceNote, max_len: int = 3900) -> str:
    """Build Telegram HTML (allowed tags only). Truncate if needed."""
    lines: list[str] = [
        f"📝 <b>{html.escape(note.note_title)}</b>",
        "",
        f"📋 <b>{html.escape(note.list_title)}</b>",
    ]
    for item in note.items:
        lines.append(f"• {html.escape(item)}")
    body = "\n".join(lines)
    if len(body) <= max_len:
        return body
    return body[: max_len - 1] + "…"


def add_item(note: VoiceNote, text: str) -> VoiceNote | None:
    """Append one bullet. Returns ``None`` when nothing can be added."""
    item = " ".join(text.split())
    if not item or len(note.items) >= _MAX_ITEMS:
        return None
    return note.model_copy(update={"items": [*note.items, item]})


def replace_quoted_text(
    note: VoiceNote, quote: str | None, new_text: str
) -> VoiceNote | None:
    """Swap the quoted fragment of one bullet for ``new_text``.

    The quote is what the user selected in Telegram, so only that part of the
    line changes and the rest of the wording survives. Returns ``None`` when
    the quote matches no bullet or nothing would change.
    """
    index = find_item_by_quote(note, quote)
    replacement = " ".join(new_text.split())
    if index is None or not replacement:
        return None
    item = note.items[index]
    needle = _clean_quoted_keep_case(quote or "")
    if not needle:
        return None
    updated = _replace_once(item, needle, replacement)
    if updated == item:
        return None
    items = list(note.items)
    items[index] = updated[:_MAX_ITEM_CHARS]
    return note.model_copy(update={"items": items})


def _replace_once(item: str, needle: str, replacement: str) -> str:
    position = item.find(needle)
    if position < 0:
        # Fall back to a case-insensitive search on the raw text.
        position = item.lower().find(needle.lower())
    if position < 0:
        return item
    return item[:position] + replacement + item[position + len(needle) :]


def _clean_quoted_keep_case(raw: str) -> str:
    text = raw.replace("•", " ").replace("📝", " ").replace("📋", " ")
    return " ".join(text.split())


def find_item_by_quote(note: VoiceNote, quote: str | None) -> int | None:
    """Locate the bullet a Telegram quote points at.

    Users select a line of the note and reply to it, so the quote carries the
    exact text. Match exactly first, then ignoring case/bullet decoration, and
    only fall back to a unique substring so an ambiguous quote edits nothing.
    """
    if not quote or not quote.strip():
        return None
    needle = _clean_quoted(quote)
    if not needle:
        return None
    for index, item in enumerate(note.items):
        if _clean_quoted(item) == needle:
            return index
    partial = [
        index
        for index, item in enumerate(note.items)
        if needle in _clean_quoted(item)
    ]
    return partial[0] if len(partial) == 1 else None


def _clean_quoted(raw: str) -> str:
    """Drop bullet/heading decoration and collapse whitespace for compare."""
    return _clean_quoted_keep_case(raw).casefold()


def parse_voice_note_from_message_text(text: str) -> VoiceNote | None:
    """Best-effort parse of a bot note message back into a VoiceNote.

    Works on the bot's own Telegram-HTML-ish output where bullets are lines
    starting with '• ' and the first two headings contain the title and list
    title.
    """
    if not text:
        return None
    # Remove the tags we emit, then decode the entities they escaped.
    scrubbed = html.unescape(text.replace("<b>", "").replace("</b>", ""))
    lines = [ln.strip() for ln in scrubbed.splitlines() if ln.strip()]
    if not lines:
        return None

    note_title: str | None = None
    list_title: str | None = None
    items: list[str] = []

    for ln in lines:
        if ln.startswith("📝"):
            note_title = ln.lstrip("📝").strip()
            continue
        if ln.startswith("📋"):
            list_title = ln.lstrip("📋").strip()
            continue
        if ln.startswith("•"):
            item = ln.lstrip("•").strip()
            if item:
                items.append(item)

    if note_title is None:
        # Fallback: first line might be the title even if emoji was removed.
        note_title = lines[0]
    if list_title is None:
        # Fallback: find first non-title non-bullet line.
        for ln in lines[1:]:
            if not ln.startswith("•"):
                list_title = ln
                break
    if list_title is None:
        list_title = "Summary"
    if not items:
        return None
    return VoiceNote(note_title=note_title, list_title=list_title, items=items)
