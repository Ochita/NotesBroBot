from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import yaml

from notesbro_bot.i18n import supported_locales

_DEFAULT_BASE_URL = "https://api.aitunnel.ru/v1/"
_DEFAULT_STT_MODEL = "voxtral-small-24b-2507-stt"
_DEFAULT_MODEL_NAME = "deepseek-v4-flash"


@dataclass(frozen=True, slots=True)
class Settings:
    telegram_bot_token: str
    api_key: str
    #: OpenAI-compatible API root, for example AITunnel.
    base_url: str
    #: Speech-to-text model id on that endpoint.
    stt_model: str
    #: Chat model id on that endpoint. Turns a transcript into a note.
    model_name: str
    database_path: str
    #: Fallback UI language when a user has none stored or set on Telegram.
    default_locale: str
    #: When false, only users already present in the users table may use it.
    allow_new_users: bool


def load_settings(config_path: str = "config/settings.yaml") -> Settings:
    path = Path(config_path)
    if not path.exists():
        raise ValueError(
            f"Missing config file: {path}. "
            "Copy config/settings.example.yaml to config/settings.yaml."
        )

    with path.open("r", encoding="utf-8") as config_file:
        data = yaml.safe_load(config_file) or {}

    if not isinstance(data, dict):
        raise ValueError("Config must be a YAML mapping at the top level.")

    token = _required(data, "telegram_bot_token")
    api_key = _required(data, "api_key")
    if not token:
        raise ValueError("Missing telegram_bot_token in YAML config.")
    if not api_key:
        raise ValueError("Missing api_key in YAML config.")

    stt_model = _model_id(data, "stt_model", _DEFAULT_STT_MODEL)
    model_name = _model_id(data, "model_name", _DEFAULT_MODEL_NAME)
    base_url = _base_url(data)
    database_path = str(data.get("database_path", "notesbro.sqlite3")).strip()
    if not database_path:
        raise ValueError("database_path cannot be empty in YAML config.")

    default_locale = _default_locale(data)

    allow_new_users = data.get("allow_new_users", True)
    if not isinstance(allow_new_users, bool):
        raise ValueError(
            "allow_new_users must be true or false in YAML config."
        )

    return Settings(
        telegram_bot_token=token,
        api_key=api_key,
        base_url=base_url,
        stt_model=stt_model,
        model_name=model_name,
        database_path=database_path,
        default_locale=default_locale,
        allow_new_users=allow_new_users,
    )


def _required(data: dict, key: str) -> str:
    value = data.get(key, "")
    return str(value).strip()


def _base_url(data: dict) -> str:
    raw = data.get("base_url", _DEFAULT_BASE_URL)
    url = str(raw).strip()
    if not url:
        raise ValueError("base_url cannot be empty in YAML config.")
    return url


def _model_id(data: dict, key: str, default: str) -> str:
    raw = data.get(key, default)
    if isinstance(raw, list):
        raise ValueError(f"{key} must be a single string, not a list.")
    model = str(raw).strip()
    if not model:
        raise ValueError(f"Missing {key} in YAML config.")
    return model


def _default_locale(data: dict) -> str:
    raw = data.get("default_locale", "en")
    locale = str(raw).strip().lower().split("-", maxsplit=1)[0]
    if locale not in supported_locales():
        known = ", ".join(sorted(supported_locales()))
        raise ValueError(f"default_locale must be one of: {known}.")
    return locale
