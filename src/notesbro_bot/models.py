from __future__ import annotations

from pydantic import BaseModel, Field


class VoiceNote(BaseModel):
    """Structured summary returned by Gemini for a voice transcript."""

    note_title: str = Field(
        ...,
        min_length=1,
        max_length=200,
        description="Short descriptive name for the whole note.",
    )
    list_title: str = Field(
        ...,
        min_length=1,
        max_length=200,
        description=(
            "Heading for the bullet list (e.g. Key points, Next steps)."
        ),
    )
    items: list[str] = Field(
        ...,
        min_length=1,
        max_length=16,
        description="Concise bullet points derived from the transcript.",
    )
