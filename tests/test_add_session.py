from __future__ import annotations

import time

from notesbro_bot.bot import (
    _VOICE_TTL_SEC,
    MODE_ADD,
    MODE_EDIT,
    _arm,
    _disarm,
    _pending,
)
from notesbro_bot.models import VoiceNote

NOTE = VoiceNote(note_title="T", list_title="L", items=["a"])


def test_arm_then_pending_roundtrip() -> None:
    ud: dict = {}
    _arm(ud, MODE_ADD, target_message_id=5, note=NOTE, ttl_sec=60.0)
    pending = _pending(ud)
    assert pending is not None
    assert pending["mode"] == MODE_ADD
    assert pending["target_message_id"] == 5
    assert _disarm(ud) is True
    assert _pending(ud) is None


def test_pending_expires_and_clears_itself() -> None:
    ud: dict = {}
    _arm(ud, MODE_EDIT, target_message_id=5, note=NOTE, ttl_sec=0.01)
    time.sleep(0.03)
    assert _pending(ud) is None
    assert ud == {}


def test_disarm_when_nothing_armed() -> None:
    assert _disarm({}) is False


def test_pending_ignores_garbage_state() -> None:
    assert _pending({"pending_voice": "nope"}) is None


def test_default_ttl_is_one_minute() -> None:
    assert _VOICE_TTL_SEC == 60.0


def test_quote_is_stored_for_edit_mode() -> None:
    ud: dict = {}
    _arm(ud, MODE_EDIT, target_message_id=7, note=NOTE, quote="a")
    pending = _pending(ud)
    assert pending is not None
    assert pending["quote"] == "a"
