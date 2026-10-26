from __future__ import annotations

from pathlib import Path

import pytest

from notesbro_bot.db import NoteRepository


@pytest.mark.asyncio
async def test_repository_user_whitelist_roundtrip(tmp_path: Path) -> None:
    db_path = tmp_path / "t.sqlite3"
    repo = NoteRepository(str(db_path))
    await repo.init()
    assert await repo.user_exists(42) is False

    await repo.upsert_user(42, 42000)
    assert await repo.user_exists(42) is True


@pytest.mark.asyncio
async def test_locale_roundtrip(tmp_path: Path) -> None:
    repo = NoteRepository(str(tmp_path / "l.sqlite3"))
    await repo.init()
    await repo.upsert_user(7, 70)
    assert await repo.get_user_locale(7) is None
    await repo.set_user_locale(7, "ru")
    assert await repo.get_user_locale(7) == "ru"


@pytest.mark.asyncio
async def test_get_user_locale_for_missing_user(tmp_path: Path) -> None:
    repo = NoteRepository(str(tmp_path / "m.sqlite3"))
    await repo.init()
    assert await repo.get_user_locale(999) is None


@pytest.mark.asyncio
async def test_init_migrates_legacy_table_without_locale(
    tmp_path: Path,
) -> None:
    import aiosqlite

    db_path = tmp_path / "legacy.sqlite3"
    async with aiosqlite.connect(str(db_path)) as conn:
        await conn.execute(
            "CREATE TABLE users ("
            "user_id INTEGER PRIMARY KEY, "
            "chat_id INTEGER NOT NULL, "
            "created_at_utc TEXT NOT NULL DEFAULT (datetime('now')));"
        )
        await conn.execute(
            "INSERT INTO users(user_id, chat_id) VALUES (1, 1);"
        )
        await conn.commit()

    repo = NoteRepository(str(db_path))
    await repo.init()  # idempotent migration adds the locale column
    await repo.set_user_locale(1, "de")
    assert await repo.get_user_locale(1) == "de"
    await repo.init()  # running twice must stay safe
