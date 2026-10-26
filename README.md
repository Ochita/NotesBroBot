# NotesBroBot

Telegram bot that accepts voice messages and turns them into summarized, formatted
notes using an OpenAI-compatible endpoint (speech-to-text + structured notes).

## What it does

- User subscribes with `/start` (optional whitelist/registration blocking supported)
- User sends a voice message
- Bot transcribes audio via `/audio/transcriptions` (`stt_model`, language
  auto-detected so notes answer in the language you spoke)
- Bot converts the transcript into a structured note:
  - `note_title`
  - `list_title`
  - bullet `items`
- Bot replies with a formatted note message (Telegram HTML)
- All bot messages and errors are localized (`en`, `ru`, `uk`, `it`, `fr`,
  `de`, `es`); `/set_locale` switches the UI language per user
- User can update an existing note:
  - reply `/add` to the note message
  - send the next voice message
  - bot merges the new transcript into the note and edits the original message

## Project structure

```text
src/notesbro_bot/
  __main__.py
  main.py
  config.py
  db.py
  i18n.py
  messages.yaml
  models.py
  note_summary.py
  bot.py
```

## Setup

1. Create and activate venv

```bash
python3 -m venv .venv
source .venv/bin/activate
```

2. Install dependencies

```bash
pip install -e .
```

3. Configure YAML settings

```bash
cp config/settings.example.yaml config/settings.yaml
```

Edit `config/settings.yaml` with your secrets:
- `telegram_bot_token`
- `api_key` (key for the OpenAI-compatible endpoint)
- `base_url` (optional, default `https://api.aitunnel.ru/v1/`)
- `stt_model` (speech model; default `voxtral-small-24b-2507-stt`)
- `model_name` (chat model; default `deepseek-v4-flash`, turns the transcript into a note)
- `default_locale` (optional, default `en`) — fallback UI language used when
  the user has not chosen one and Telegram reports no supported language

## Run

```bash
python -m notesbro_bot
```

Or use a custom config path:

```bash
python -m notesbro_bot --config /run/secrets/notesbro/settings.yaml
```

## Run with Docker

1. Prepare Docker config file:

```bash
cp config/settings.docker.example.yaml config/settings.yaml
```

Edit `config/settings.yaml` and set:
- `telegram_bot_token`
- `api_key`
- `stt_model` and `model_name`
- optionally `allow_new_users`

2. Build image:

```bash
docker build -t notesbro-bot:latest .
```

3. Run container:

```bash
docker run -d \
  --name notesbro-bot \
  --restart unless-stopped \
  -v "$(pwd)/config/settings.yaml:/run/secrets/notesbro/settings.yaml:ro" \
  -v notesbro_data:/app/data \
  notesbro-bot:latest
```

The SQLite database is persisted in Docker volume `notesbro_data`.

## Run with Docker Compose

```bash
docker compose up -d --build
```

Stop:

```bash
docker compose down
```

## Commands

- `/start` subscribe user (and pass whitelist check if enabled)
- `/help` show the commands
- `/set_locale <code>` set the UI language (`en`, `ru`, `uk`, `it`, `fr`, `de`, `es`)
- `/add` reply to a note message, then send next voice to merge into it
- `/cancel` cancel a pending `/add` merge

## Notes

- Config is read from YAML via `yaml.safe_load` (no `.env` loading).
- The bot uses SQLite only for user whitelist/registration (`allow_new_users`)
  and the chosen locale.
- UI text lives in `src/notesbro_bot/messages.yaml`; notes themselves always
  follow the language of the voice message.
- API requests retry up to 3 times for transient failures.
- `/add` works without storing message ids in the database:
  it parses the replied note message and edits that message via Telegram API.

## Development checks

Install dev tools:

```bash
pip install -e ".[dev]"
```

Run tests (coverage floor is 91%, enforced by `--cov-fail-under`):

```bash
pytest
ruff check src tests
```

