from __future__ import annotations

import string

import pytest

from notesbro_bot.i18n import (
    _MESSAGES,
    detect_locale,
    supported_locales,
    translate,
)


def test_translate_unknown_locale_falls_back_to_en() -> None:
    assert translate("xx", "cancel_nothing") == "Nothing to cancel."


def test_translate_unknown_key_raises() -> None:
    with pytest.raises(KeyError):
        translate("ru", "no_such_key")


def test_translate_formats_placeholders() -> None:
    assert translate("en", "add_armed", ttl="120").startswith("OK.")


def test_detect_locale_from_telegram_language_code() -> None:
    assert detect_locale("it-IT", "en") == "it"
    assert detect_locale("RU", "en") == "ru"
    assert detect_locale(None, "en") == "en"
    assert detect_locale("zh", "ru") == "ru"
    assert detect_locale("", "de") == "de"


def test_supported_locales() -> None:
    assert supported_locales() == {"en", "ru", "uk", "it", "fr", "de", "es"}


def test_every_locale_has_every_key() -> None:
    expected = set(_MESSAGES["en"])
    for locale, messages in _MESSAGES.items():
        assert set(messages) == expected, locale


@pytest.mark.parametrize("locale", sorted(supported_locales()))
def test_every_locale_renders_every_key(locale: str) -> None:
    values = "x"
    for key, template in _MESSAGES[locale].items():
        names = {
            name
            for _, name, _, _ in string.Formatter().parse(template)
            if name
        }
        assert translate(locale, key, **{n: values for n in names})


def test_russian_locale_is_translated() -> None:
    assert translate("ru", "cancel_done") == "Режим дополнения отменен."
