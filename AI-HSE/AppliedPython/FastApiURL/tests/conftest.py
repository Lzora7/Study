import asyncio
import os

import pytest
from httpx import AsyncClient, ASGITransport
from sqlalchemy.ext.asyncio import AsyncSession, async_sessionmaker, create_async_engine

# Ensure cache is disabled for tests (no Redis needed)
os.environ.setdefault("DISABLE_CACHE", "1")


@pytest.fixture(scope="session")
def event_loop():
    loop = asyncio.new_event_loop()
    yield loop
    loop.close()


@pytest.fixture(scope="session")
async def test_engine(tmp_path_factory):
    # Use a file-based SQLite DB to share across async connections
    db_path = tmp_path_factory.mktemp("db") / "test.db"
    engine = create_async_engine(f"sqlite+aiosqlite:///{db_path}")
    yield engine
    await engine.dispose()


@pytest.fixture(scope="session")
async def prepare_db(test_engine):
    # Create all tables for links and auth on the test engine
    from links.models import metadata as links_metadata
    from auth.db import Base as AuthBase

    async with test_engine.begin() as conn:
        await conn.run_sync(links_metadata.create_all)
        await conn.run_sync(AuthBase.metadata.create_all)


@pytest.fixture(scope="session")
async def session_maker(test_engine, prepare_db):
    return async_sessionmaker(test_engine, expire_on_commit=False, class_=AsyncSession)


@pytest.fixture(scope="session")
async def app(session_maker, test_engine):
    # Override database dependency and globals to use the test engine/session
    import database as db_module
    from main import app as fastapi_app

    db_module.engine = test_engine
    db_module.async_session_maker = session_maker

    async def override_get_async_session():
        async with session_maker() as session:
            yield session

    fastapi_app.dependency_overrides[db_module.get_async_session] = override_get_async_session
    return fastapi_app


@pytest.fixture()
async def db_session(session_maker):
    async with session_maker() as session:
        yield session


@pytest.fixture()
async def client(app):
    transport = ASGITransport(app=app)
    async with AsyncClient(transport=transport, base_url="http://testserver") as ac:
        yield ac


