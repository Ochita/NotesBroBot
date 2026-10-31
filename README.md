# NotesBroBot

Telegram bot that turns voice messages into structured, formatted notes via an
OpenAI-compatible endpoint (speech-to-text + structured chat completion).

## What it does

- `/start` subscribes the user (optional whitelist via `allow_new_users`)
- User sends a voice message **or plain text** — both take the same pipeline
- For voice, the bot transcribes it through `/audio/transcriptions`, passing
  the user's current locale as the `language` hint
- The bot turns that text into a structured note: `note_title`,
  `list_title`, and bullet `items`
- Bot replies with the note as Telegram HTML
- Notes and all bot messages are written in the user's locale (`en`, `ru`,
  `uk`, `it`, `fr`, `de`, `es`); `/set_locale` switches it per user. The
  locale is also passed to the speech-to-text model as a `language` hint, and
  the summarize prompt pins the output language explicitly

### Editing a note afterwards

`/add` and `/edit` work the same way — text is applied immediately, no text
arms a 60-second voice session:

| Command | With text | Without text |
| --- | --- | --- |
| `/add` | `/add buy milk` appends the point(s), split apart if there are several | waits 60s for a voice, appends it |
| `/edit` | `/edit oat milk` replaces the quoted fragment | waits 60s for a voice, replaces with it |
| `/cancel` | — | stops the waiting session |

Both are replies to the bot's note message. `/edit` replaces **only the
fragment you quote**, so the rest of the line is never retyped:

```
note line:  buy milk
you quote:        milk
/edit oat milk
result:     buy oat milk
```

Typed text and speech share one code path: `/add` sends the text through a
split step so one message can contribute several bullets, while `/edit` swaps
only the quoted fragment.

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
- `model_name` (chat model; default `deepseek-v4-flash`, turns the transcript
  into a note)
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

- `/start` subscribe and show short usage help
- `/help` full command list
- `/set_locale <code>` UI language: `en`, `ru`, `uk`, `it`, `fr`, `de`, `es`
- `/add [text]` reply to a note; appends the text, or a voice within 60s
- `/edit [text]` reply quoting a fragment; replaces it with the text or voice
- `/cancel` stop waiting for the armed voice message

## Notes

- Config is read from YAML via `yaml.safe_load` (no `.env` loading).
- SQLite stores only the user whitelist (`allow_new_users`) and the chosen
  locale — never note contents.
- UI text lives in `src/notesbro_bot/messages.yaml`; notes use the user's
  locale regardless of the language actually spoken.
- API requests retry up to 3 times for transient failures.
- Notes are not stored anywhere: `/add` and `/edit` re-parse the replied
  message text and rewrite that message through the Telegram API.

## Development checks

```bash
pip install -e ".[dev]"
pytest          # coverage floor is 91% via --cov-fail-under
ruff check src tests
```
