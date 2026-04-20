from __future__ import annotations

import html
import logging
import time
from typing import TYPE_CHECKING

from google.genai import types

from notesbro_bot.models import VoiceNote

if TYPE_CHECKING:
    from google import genai

LOGGER = logging.getLogger(__name__)

_GEMINI_RETRIES = 3

def _generate_with_retries(client: "genai.Client", *, model: str, contents, config):
    last_err: Exception | None = None
    for attempt in range(1, _GEMINI_RETRIES + 1):
        try:
            return client.models.generate_content(
                model=model,
                contents=contents,
                config=config,
            )
        except Exception as e:
            last_err = e
            if attempt < _GEMINI_RETRIES:
                time.sleep(0.8 * attempt)
                continue
            raise

_SUMMARY_INSTRUCTION = """You turn a raw speech transcript into short structured notes.

Rules:
- note_title: a clear, concise title for the whole note (like naming a notebook page).
- list_title: one thematic heading for the bullet list (not a repeat of note_title).
- items: 3 to 8 short bullets summarizing the transcript. Each bullet one idea, one line.
- Use the same language as the transcript when possible.
- Do not invent facts; only summarize what is implied or stated in the transcript.
"""


_MERGE_INSTRUCTION = """You merge an existing structured note with new spoken content.

You will be given the existing note as JSON, then the new transcript in plain text.

Return one updated VoiceNote: refine note_title if needed, one list_title, and a merged
items list. Remove clear duplicates; keep facts from both the prior note and the new
transcript. Prefer roughly 4–14 bullets; stay within the schema max length."""


def merge_transcript_into_note_sync(
    client: "genai.Client",
    model_name: str,
    previous: VoiceNote,
    new_transcript: str,
) -> VoiceNote:
    existing = previous.model_dump_json()
    prompt = (
        _MERGE_INSTRUCTION
        + "\n\nExisting note (JSON):\n"
        + existing
        + "\n\nNew transcript:\n"
        + new_transcript
    )
    config = types.GenerateContentConfig(
        response_mime_type="application/json",
        response_schema=VoiceNote,
    )
    response = _generate_with_retries(
        client, model=model_name, contents=[prompt], config=config
    )
    parsed = response.parsed
    if isinstance(parsed, VoiceNote):
        return _normalize_note(parsed)
    if isinstance(parsed, dict):
        return _normalize_note(VoiceNote.model_validate(parsed))
    text = (response.text or "").strip()
    if text:
        return _normalize_note(VoiceNote.model_validate_json(text))
    LOGGER.warning("Merge response empty.")
    raise ValueError("Model returned no merged note.")


def summarize_transcript_to_note_sync(
    client: "genai.Client",
    model_name: str,
    transcript: str,
) -> VoiceNote:
    config = types.GenerateContentConfig(
        response_mime_type="application/json",
        response_schema=VoiceNote,
    )
    response = _generate_with_retries(
        client,
        model=model_name,
        contents=[
            _SUMMARY_INSTRUCTION,
            f"Transcript:\n{transcript}",
        ],
        config=config,
    )
    parsed = response.parsed
    if isinstance(parsed, VoiceNote):
        return _normalize_note(parsed)
    if isinstance(parsed, dict):
        return _normalize_note(VoiceNote.model_validate(parsed))
    text = (response.text or "").strip()
    if text:
        return _normalize_note(VoiceNote.model_validate_json(text))
    LOGGER.warning("Structured note response empty; raw candidates may be blocked.")
    raise ValueError("Model returned no structured note.")


def _normalize_note(note: VoiceNote) -> VoiceNote:
    title = note.note_title.strip() or "Untitled note"
    list_heading = note.list_title.strip() or "Summary"
    items = [s.strip() for s in note.items if s.strip()]
    if not items:
        items = ["(No distinct points extracted from the transcript.)"]
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


def parse_voice_note_from_message_text(text: str) -> VoiceNote | None:
    """Best-effort parse of a bot note message back into a VoiceNote.

    Works on the bot's own Telegram-HTML-ish output where bullets are lines starting
    with '• ' and the first two headings contain the title and list title.
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
