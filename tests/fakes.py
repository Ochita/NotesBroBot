from __future__ import annotations

from dataclasses import dataclass, field
from types import SimpleNamespace

from notesbro_bot.config import Settings

CHAT_ID = 777
USER_ID = 42


@dataclass
class FakeUser:
    id: int = USER_ID
    language_code: str | None = None


@dataclass
class FakeChat:
    id: int = CHAT_ID


@dataclass
class FakeVoice:
    file_id: str = "file-1"
    mime_type: str | None = "audio/ogg"


@dataclass
class FakeCallbackQuery:
    data: str | None = None
    message: FakeMessage | None = None
    answered: bool = False

    async def answer(self, **kwargs) -> None:
        _ = kwargs
        self.answered = True


@dataclass
class FakeMessage:
    message_id: int = 1
    chat_id: int = CHAT_ID
    text: str | None = None
    voice: FakeVoice | None = None
    reply_to_message: FakeMessage | None = None
    quote: object | None = None
    replies: list[str] = field(default_factory=list)
    edits: list[tuple[str, dict]] = field(default_factory=list)
    markups: list[object] = field(default_factory=list)

    async def reply_text(self, text: str, **kwargs):
        self.replies.append(text)
        if kwargs.get("reply_markup") is not None:
            self.markups.append(kwargs["reply_markup"])
        return type(self)(
            message_id=self.message_id + 1,
            replies=self.replies,
            edits=self.edits,
            markups=self.markups,
        )

    async def edit_text(self, text: str, **kwargs):
        self.edits.append((text, kwargs))


@dataclass
class FakeBot:
    sent: list[dict] = field(default_factory=list)
    edited: list[dict] = field(default_factory=list)
    deleted: list[dict] = field(default_factory=list)
    audio: bytes = b"audio-bytes"
    fail_download: bool = False
    fail_edit: bool = False
    fail_delete: bool = False

    async def send_message(self, **kwargs):
        self.sent.append(kwargs)
        return FakeMessage(message_id=900)

    async def edit_message_text(self, **kwargs):
        if self.fail_edit:
            raise RuntimeError("edit failed")
        self.edited.append(kwargs)

    async def delete_message(self, **kwargs):
        if self.fail_delete:
            raise RuntimeError("delete failed")
        self.deleted.append(kwargs)

    async def get_file(self, file_id: str):
        _ = file_id

        async def download_to_memory(buffer) -> None:
            if self.fail_download:
                raise RuntimeError("download failed")
            buffer.write(self.audio)

        return SimpleNamespace(download_to_memory=download_to_memory)


class FakeRepository:
    def __init__(self, known_users: set[int] | None = None) -> None:
        self.known_users = known_users or set()
        self.locales: dict[int, str] = {}
        self.initialized = False

    async def init(self) -> None:
        self.initialized = True

    async def user_exists(self, user_id: int) -> bool:
        return user_id in self.known_users

    async def upsert_user(self, user_id: int, chat_id: int) -> None:
        _ = chat_id
        self.known_users.add(user_id)

    async def set_user_locale(self, user_id: int, locale: str) -> None:
        self.locales[user_id] = locale

    async def get_user_locale(self, user_id: int) -> str | None:
        return self.locales.get(user_id)


class FakeContext:
    def __init__(
        self,
        *,
        settings: Settings | None = None,
        repository: FakeRepository | None = None,
        client: object | None = None,
        user_data: dict | None = None,
        args: list[str] | None = None,
        bot: FakeBot | None = None,
    ) -> None:
        self.bot = bot or FakeBot()
        self.user_data = user_data if user_data is not None else {}
        self.args = args or []
        self.bot_data = {
            "settings": settings or make_settings(),
            "repository": repository or FakeRepository(),
            "ai_client": client,
        }
        self.application = SimpleNamespace(bot_data=self.bot_data)


def make_settings(**overrides) -> Settings:
    values = {
        "telegram_bot_token": "tok",
        "api_key": "key",
        "base_url": "https://example.test/v1/",
        "stt_model": "stt-model",
        "model_name": "chat-model",
        "database_path": "test.sqlite3",
        "default_locale": "en",
        "allow_new_users": True,
    }
    values.update(overrides)
    return Settings(**values)


def make_update(
    *,
    message: FakeMessage | None = None,
    user: FakeUser | None = None,
) -> SimpleNamespace:
    return SimpleNamespace(
        effective_user=user if user is not None else FakeUser(),
        effective_chat=FakeChat(),
        message=message,
        effective_message=message,
    )
