from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import yaml


@dataclass(frozen=True, slots=True)
class Settings:
    telegram_bot_token: str
    api_key: str
    model_name: str
    database_path: str
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

    model_name = _model_name(data)
    database_path = str(data.get("database_path", "notesbro.sqlite3")).strip()
    if not database_path:
        raise ValueError("database_path cannot be empty in YAML config.")

    allow_new_users = data.get("allow_new_users", True)
    if not isinstance(allow_new_users, bool):
        raise ValueError(
            "allow_new_users must be true or false in YAML config."
        )

    return Settings(
        telegram_bot_token=token,
        api_key=api_key,
        model_name=model_name,
        database_path=database_path,
        allow_new_users=allow_new_users,
    )


def _required(data: dict, key: str) -> str:
    value = data.get(key, "")
    return str(value).strip()


def _model_name(data: dict) -> str:
    raw = data.get("model_name")
    if raw is None:
        raw = "gemini-2.0-flash"
    if isinstance(raw, list):
        raise ValueError("model_name must be a single string, not a list.")
    model = str(raw).strip()
    if not model:
        raise ValueError("Missing model_name in YAML config.")
    return model
