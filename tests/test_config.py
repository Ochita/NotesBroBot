from __future__ import annotations

from pathlib import Path

import pytest

from notesbro_bot.config import Settings, load_settings


def test_load_settings_minimal(tmp_path: Path) -> None:
    p = tmp_path / "settings.yaml"
    p.write_text(
        'telegram_bot_token: "tok"\n'
        'api_key: "key"\n'
        'model_name: "chat-model"\n',
        encoding="utf-8",
    )
    s = load_settings(str(p))
    assert isinstance(s, Settings)
    assert s.telegram_bot_token == "tok"
    assert s.api_key == "key"
    assert s.base_url == "https://api.aitunnel.ru/v1/"
    assert s.stt_model == "voxtral-small-24b-2507-stt"
    assert s.model_name == "chat-model"
    assert s.database_path == "notesbro.sqlite3"
    assert s.allow_new_users is True


def test_load_settings_custom_base_url(tmp_path: Path) -> None:
    p = tmp_path / "settings.yaml"
    p.write_text(
        'telegram_bot_token: "t"\n'
        'api_key: "k"\n'
        'base_url: "https://example.test/v1/"\n'
        'stt_model: "whisper-1"\n'
        'model_name: "chat-model"\n',
        encoding="utf-8",
    )
    s = load_settings(str(p))
    assert s.base_url == "https://example.test/v1/"
    assert s.stt_model == "whisper-1"


def test_load_settings_default_model(tmp_path: Path) -> None:
    p = tmp_path / "settings.yaml"
    p.write_text(
        'telegram_bot_token: "t"\n'
        'api_key: "k"\n',
        encoding="utf-8",
    )
    s = load_settings(str(p))
    assert s.stt_model == "voxtral-small-24b-2507-stt"
    assert s.model_name == "deepseek-v4-flash"
    assert s.base_url == "https://api.aitunnel.ru/v1/"


def test_load_settings_allow_new_users_false(tmp_path: Path) -> None:
    p = tmp_path / "settings.yaml"
    p.write_text(
        'telegram_bot_token: "t"\n'
        'api_key: "k"\n'
        'model_name: "chat-model"\n'
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
        'model_name: "chat-model"\n'
        'allow_new_users: "maybe"\n',
        encoding="utf-8",
    )
    with pytest.raises(ValueError, match="allow_new_users"):
        load_settings(str(p))


def test_missing_config_file(tmp_path: Path) -> None:
    with pytest.raises(ValueError, match="Missing config file"):
        load_settings(str(tmp_path / "nope.yaml"))


def test_config_must_be_mapping(tmp_path: Path) -> None:
    p = tmp_path / "settings.yaml"
    p.write_text("- a\n- b\n", encoding="utf-8")
    with pytest.raises(ValueError, match="mapping"):
        load_settings(str(p))


@pytest.mark.parametrize(
    "body, message",
    [
        ("api_key: k\n", "telegram_bot_token"),
        ("telegram_bot_token: t\n", "api_key"),
        ("telegram_bot_token: t\napi_key: k\ndatabase_path: ' '\n",
         "database_path"),
        ("telegram_bot_token: t\napi_key: k\nbase_url: ' '\n", "base_url"),
        ("telegram_bot_token: t\napi_key: k\nmodel_name: []\n", "model_name"),
        ("telegram_bot_token: t\napi_key: k\nstt_model: ' '\n", "stt_model"),
        ("telegram_bot_token: t\napi_key: k\ndefault_locale: zz\n",
         "default_locale"),
    ],
)
def test_invalid_values(tmp_path: Path, body: str, message: str) -> None:
    p = tmp_path / "settings.yaml"
    p.write_text(body, encoding="utf-8")
    with pytest.raises(ValueError, match=message):
        load_settings(str(p))


def test_default_locale_normalized(tmp_path: Path) -> None:
    p = tmp_path / "settings.yaml"
    p.write_text(
        'telegram_bot_token: "t"\n'
        'api_key: "k"\n'
        'default_locale: "RU-ru"\n',
        encoding="utf-8",
    )
    assert load_settings(str(p)).default_locale == "ru"


def test_default_locale_defaults_to_english(tmp_path: Path) -> None:
    p = tmp_path / "settings.yaml"
    p.write_text(
        'telegram_bot_token: "t"\napi_key: "k"\n', encoding="utf-8"
    )
    assert load_settings(str(p)).default_locale == "en"
