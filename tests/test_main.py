from __future__ import annotations

import logging
from types import SimpleNamespace

import pytest

from notesbro_bot.main import configure_logging, main


class FakeApp:
    def __init__(self) -> None:
        self.polled = False

    def run_polling(self, allowed_updates: object) -> None:
        _ = allowed_updates
        self.polled = True


def test_configure_logging_quietens_httpx() -> None:
    configure_logging()
    assert logging.getLogger("httpx").level == logging.WARNING


def test_main_passes_config_path(monkeypatch) -> None:
    fake_app = FakeApp()
    seen: dict = {}

    def fake_load_settings(config_path: str):
        seen["path"] = config_path
        return SimpleNamespace(stt_model="stt", model_name="chat")

    monkeypatch.setattr("notesbro_bot.main.load_settings", fake_load_settings)
    monkeypatch.setattr(
        "notesbro_bot.main.build_application", lambda _s: fake_app
    )
    monkeypatch.setattr("sys.argv", ["prog", "--config", "custom.yaml"])

    main()
    assert seen["path"] == "custom.yaml"
    assert fake_app.polled is True


def test_main_exits_on_bad_config(monkeypatch) -> None:
    def fake_load_settings(config_path: str):
        _ = config_path
        raise ValueError("boom")

    monkeypatch.setattr("notesbro_bot.main.load_settings", fake_load_settings)
    monkeypatch.setattr("sys.argv", ["prog"])

    with pytest.raises(SystemExit) as exit_info:
        main()
    assert exit_info.value.code == 1
