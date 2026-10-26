from __future__ import annotations

from pathlib import Path

import yaml

_MESSAGES_PATH = Path(__file__).with_name("messages.yaml")


def _load_messages() -> dict[str, dict[str, str]]:
    with _MESSAGES_PATH.open("r", encoding="utf-8") as file:
        data = yaml.safe_load(file) or {}
    return {
        str(locale): {str(key): str(value) for key, value in values.items()}
        for locale, values in data.items()
        if isinstance(values, dict)
    }


_MESSAGES = _load_messages()


def translate(locale: str, key: str, **kwargs: str) -> str:
    messages = _MESSAGES.get(locale, _MESSAGES["en"])
    template = messages.get(key, _MESSAGES["en"][key])
    return template.format(**kwargs)


def supported_locales() -> frozenset[str]:
    return frozenset(_MESSAGES)


def detect_locale(
    user_language_code: str | None,
    default_locale: str,
) -> str:
    if user_language_code:
        base = user_language_code.lower().split("-", maxsplit=1)[0]
        if base in _MESSAGES:
            return base
    return default_locale
