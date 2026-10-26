from __future__ import annotations

import html
import logging
import time
from typing import TYPE_CHECKING

from notesbro_bot.i18n import translate
from notesbro_bot.models import VoiceNote

if TYPE_CHECKING:
    from openai import OpenAI

LOGGER = logging.getLogger(__name__)

_API_RETRIES = 3

def _complete_note(
    client: "OpenAI", model: str, prompt: str, locale: str
) -> VoiceNote:
    for attempt in range(1, _API_RETRIES + 1):
        try:
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
        except Exception:
            if attempt < _API_RETRIES:
                time.sleep(0.8 * attempt)
                continue
            raise


_SUMMARY_INSTRUCTION = """Turn a raw speech transcript into structured notes.

Rules:
- note_title: a clear title for the whole note.
- list_title: one heading for the bullet list (not a repeat of note_title).
- items: 3 to 8 short bullets. One idea per line.
- Write every field in the language of the transcript. If the speaker
  switches languages, follow the dominant one. Never answer in English unless
  the transcript is English.
- Do not invent facts; only summarize what is implied or stated.
"""


_MERGE_INSTRUCTION = """Merge an existing note with new spoken content.

You will be given the existing note as JSON, then the new transcript in plain
text.

Return one updated VoiceNote: refine note_title if needed, one list_title,
and a merged items list.

Remove clear duplicates; keep facts from both the prior note and the new
transcript.
Write every field in the language of the new transcript, even if the existing
note is in another language.
Prefer roughly 4-14 bullets; stay within the schema max length.
"""


def merge_transcript_into_note_sync(
    client: "OpenAI",
    model_name: str,
    previous: VoiceNote,
    new_transcript: str,
    locale: str | None = None,
) -> VoiceNote:
    prompt = (
        _MERGE_INSTRUCTION
        + "\n\nExisting note (JSON):\n"
        + previous.model_dump_json()
        + "\n\nNew transcript:\n"
        + new_transcript
    )
    return _complete_note(client, model_name, prompt, locale or "en")


def summarize_transcript_to_note_sync(
    client: "OpenAI",
    model_name: str,
    transcript: str,
    locale: str | None = None,
) -> VoiceNote:
    prompt = f"{_SUMMARY_INSTRUCTION}\n\nTranscript:\n{transcript}"
    return _complete_note(client, model_name, prompt, locale or "en")


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
    return VoiceNote(
        note_title=title,
        list_title=list_heading,
        items=items,
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


def apply_note_edits(note: VoiceNote, text: str) -> tuple[VoiceNote, int]:
    """Apply line-level edits to a note.

    Each non-empty line targets one bullet so nothing has to be retyped:
    ``2 new text`` replaces, ``-2`` removes, ``+ text`` appends,
    ``t title`` / ``l heading`` rewrite the headings. Edits apply in order.
    Returns the note plus the number of applied edits (0 when nothing
    matched a valid command).
    """
    title, heading = note.note_title, note.list_title
    items = list(note.items)
    applied = 0
    for raw in text.splitlines():
        line = raw.strip()
        if not line:
            continue
        if line.startswith("+") and len(line) > 1:
            items.append(line[1:].strip())
            applied += 1
            continue
        if line.startswith("-") and len(line) > 1:
            index = _parse_item_number(line[1:].strip())
            if index is not None and index < len(items):
                del items[index]
                applied += 1
            continue
        command, _, value = line.partition(" ")
        value = value.strip()
        lowered = command.lower()
        if lowered in {"t", "title", "l", "list"} and value:
            if lowered in {"t", "title"}:
                title = value
            else:
                heading = value
            applied += 1
            continue
        index = _parse_item_number(command)
        if index is None or not value or index >= len(items):
            continue
        items[index] = value
        applied += 1
    note_out = VoiceNote(
        note_title=title[:200],
        list_title=heading[:200],
        items=items[:16],
    )
    return note_out, applied


def _parse_item_number(raw: str) -> int | None:
    """Map a 1-based user-facing bullet number to a 0-based index."""
    try:
        number = int(raw)
    except ValueError:
        return None
    return number - 1 if number >= 1 else None


def parse_voice_note_from_message_text(text: str) -> VoiceNote | None:
    """Best-effort parse of a bot note message back into a VoiceNote.

    Works on the bot's own Telegram-HTML-ish output where bullets are lines
    starting with '• ' and the first two headings contain the title and list
    title.
    """
    if not text:
        return None
    # Remove simple HTML tags we produce (<b>...</b>).
    scrubbed = (
        text.replace("<b>", "")
        .replace("</b>", "")
        .replace("&amp;", "&")
        .replace("&lt;", "<")
        .replace("&gt;", ">")
        .replace("&#x27;", "'")
        .replace("&quot;", "\"")
    )
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
