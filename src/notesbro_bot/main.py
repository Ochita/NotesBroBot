from __future__ import annotations

import argparse
import logging
import sys

from telegram import Update

from notesbro_bot.bot import build_application
from notesbro_bot.config import load_settings


def configure_logging() -> None:
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s | %(levelname)s | %(name)s | %(message)s",
    )
    logging.getLogger("httpx").setLevel(logging.WARNING)


def main() -> None:
    configure_logging()
    log = logging.getLogger(__name__)

    parser = argparse.ArgumentParser(
        description="Run NotesBro Telegram bot (voice → Gemini transcript).",
    )
    parser.add_argument(
        "--config",
        default="config/settings.yaml",
        help="Path to YAML configuration file.",
    )
    args = parser.parse_args()

    try:
        settings = load_settings(config_path=args.config)
    except ValueError as e:
        log.error("%s", e)
        sys.exit(1)

    application = build_application(settings)
    log.info("Starting bot (model=%s)", settings.model_name)
    application.run_polling(allowed_updates=Update.ALL_TYPES)


if __name__ == "__main__":
    main()
