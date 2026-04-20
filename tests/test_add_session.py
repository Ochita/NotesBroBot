from __future__ import annotations

import time

from notesbro_bot.bot import (
    _ADD_VOICE_TTL_SEC,
    _UD_AWAITING_ADD_DEADLINE_MONO,
    _UD_AWAITING_ADD_VOICE,
    _add_voice_arm_expired,
    _arm_add_voice,
    _disarm_add_voice,
)


def test_arm_disarm_roundtrip() -> None:
    ud: dict = {}
    _arm_add_voice(ud, ttl_sec=60.0)
    assert ud.get(_UD_AWAITING_ADD_VOICE) is True
    assert _UD_AWAITING_ADD_DEADLINE_MONO in ud
    _disarm_add_voice(ud)
    assert _UD_AWAITING_ADD_VOICE not in ud
    assert _UD_AWAITING_ADD_DEADLINE_MONO not in ud


def test_arm_expires() -> None:
    ud: dict = {}
    _arm_add_voice(ud, ttl_sec=0.01)
    time.sleep(0.05)
    assert _add_voice_arm_expired(ud) is True


def test_default_ttl_positive() -> None:
    assert _ADD_VOICE_TTL_SEC > 0
