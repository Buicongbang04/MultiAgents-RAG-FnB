import os
import subprocess
import sys
from pathlib import Path

import pytest
import pytest_asyncio
from sqlalchemy.ext.asyncio import AsyncSession, create_async_engine
from sqlalchemy.pool import NullPool

ROOT = Path(__file__).resolve().parents[2]
TEST_DATABASE_URL = os.getenv(
    "TEST_DATABASE_URL", "postgresql+asyncpg://fnb:fnb@localhost:5434/fnb_test"
)


@pytest.fixture(scope="session")
def migrated_test_db() -> str:
    """Run `alembic upgrade head` once against the TEST database (never the dev one)."""
    subprocess.run(
        [sys.executable, "-m", "alembic", "-x", f"url={TEST_DATABASE_URL}", "upgrade", "head"],
        cwd=ROOT,
        check=True,
    )
    return TEST_DATABASE_URL


@pytest_asyncio.fixture
async def db_session(migrated_test_db):
    """Session inside an outer transaction that is always rolled back."""
    engine = create_async_engine(migrated_test_db, poolclass=NullPool)
    async with engine.connect() as conn:
        trans = await conn.begin()
        session = AsyncSession(
            bind=conn, expire_on_commit=False, join_transaction_mode="create_savepoint"
        )
        try:
            yield session
        finally:
            await session.close()
            await trans.rollback()
    await engine.dispose()
