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
