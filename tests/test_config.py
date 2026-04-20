from __future__ import annotations

from pathlib import Path

import pytest

from notesbro_bot.config import Settings, load_settings


def test_load_settings_minimal(tmp_path: Path) -> None:
    p = tmp_path / "settings.yaml"
    p.write_text(
        'telegram_bot_token: "tok"\n'
        'api_key: "key"\n'
        'model_name: "gemini-2.0-flash"\n',
        encoding="utf-8",
    )
    s = load_settings(str(p))
    assert isinstance(s, Settings)
    assert s.telegram_bot_token == "tok"
    assert s.api_key == "key"
    assert s.model_name == "gemini-2.0-flash"
    assert s.database_path == "notesbro.sqlite3"
    assert s.allow_new_users is True


def test_load_settings_default_model(tmp_path: Path) -> None:
    p = tmp_path / "settings.yaml"
    p.write_text(
        'telegram_bot_token: "t"\n'
        'api_key: "k"\n',
        encoding="utf-8",
    )
    s = load_settings(str(p))
    assert s.model_name == "gemini-2.0-flash"
    assert s.database_path == "notesbro.sqlite3"
    assert s.allow_new_users is True


def test_load_settings_allow_new_users_false(tmp_path: Path) -> None:
    p = tmp_path / "settings.yaml"
    p.write_text(
        'telegram_bot_token: "t"\n'
        'api_key: "k"\n'
        "allow_new_users: false\n"
        'database_path: "custom.sqlite3"\n',
        encoding="utf-8",
    )
    s = load_settings(str(p))
    assert s.allow_new_users is False
    assert s.database_path == "custom.sqlite3"


def test_load_settings_allow_new_users_invalid(tmp_path: Path) -> None:
    p = tmp_path / "settings.yaml"
    p.write_text(
        'telegram_bot_token: "t"\n'
        'api_key: "k"\n'
        'allow_new_users: "maybe"\n',
        encoding="utf-8",
    )
    with pytest.raises(ValueError, match="allow_new_users"):
        load_settings(str(p))
