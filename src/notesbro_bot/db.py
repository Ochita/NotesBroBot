from __future__ import annotations

import aiosqlite


class NoteRepository:
    """SQLite persistence for users (whitelist)."""

    def __init__(self, database_path: str) -> None:
        self._database_path = database_path

    async def init(self) -> None:
        async with aiosqlite.connect(self._database_path) as conn:
            await conn.execute("PRAGMA journal_mode=WAL;")
            await conn.execute(
                """
                CREATE TABLE IF NOT EXISTS users (
                    user_id INTEGER PRIMARY KEY,
                    chat_id INTEGER NOT NULL,
                    created_at_utc TEXT NOT NULL DEFAULT (datetime('now'))
                );
                """
            )
            await conn.commit()

    async def user_exists(self, user_id: int) -> bool:
        async with aiosqlite.connect(self._database_path) as conn:
            async with conn.execute(
                "SELECT 1 FROM users WHERE user_id = ? LIMIT 1;",
                (user_id,),
            ) as cursor:
                row = await cursor.fetchone()
        return row is not None

    async def upsert_user(self, user_id: int, chat_id: int) -> None:
        async with aiosqlite.connect(self._database_path) as conn:
            await conn.execute(
                """
                INSERT INTO users(user_id, chat_id)
                VALUES (?, ?)
                ON CONFLICT(user_id) DO UPDATE SET
                    chat_id = excluded.chat_id;
                """,
                (user_id, chat_id),
            )
            await conn.commit()
