from datetime import datetime, timezone, timedelta

import pytest

from links.router import (
    create_short_link,
    redirect_short_link,
    redirect_short_link_root,
    get_link_stats,
    delete_short_link,
    update_short_link,
)
from links.schemas import LinkCreate, LinkUpdate


@pytest.mark.asyncio
async def test_create_with_tz_expires(db_session):
    # создание ссылки с временем действия
    expires = datetime.now(timezone.utc) + timedelta(days=1)
    link = LinkCreate(original_url="https://timezone.example", custom_alias="timezonecheck", expires_at=expires)
    out = await create_short_link(link, session=db_session)
    assert out.short_code == "timezonecheck"
    assert out.expires_at is not None
    # ensure naive datetime stored (tzinfo removed)
    assert out.expires_at.tzinfo is None


@pytest.mark.asyncio
async def test_redirect_not_found(db_session):
    # проверка на несуществующую ссылку
    with pytest.raises(Exception):
        await redirect_short_link(short_code="nope", session=db_session)


@pytest.mark.asyncio
async def test_root_reserved(db_session):
    # проверка на зарезервированный код
    with pytest.raises(Exception):
        await redirect_short_link_root(short_code="docs", session=db_session)


@pytest.mark.asyncio
async def test_update_no_changes_and_with_tz(db_session):
    # создание ссылки
    link = LinkCreate(original_url="https://up.example", custom_alias="up0")
    out = await create_short_link(link, session=db_session)

    # проверка на обновление без изменений
    out2 = await update_short_link("up0", LinkUpdate(), session=db_session)
    assert out2.short_code == out.short_code
    assert out2.original_url == out.original_url

    # обновление срока действия с учетом часового пояса
    new_exp = datetime.now(timezone.utc) + timedelta(days=2)
    out3 = await update_short_link("up0", LinkUpdate(expires_at=new_exp), session=db_session)
    assert out3.expires_at is not None
    assert out3.expires_at.tzinfo is None


@pytest.mark.asyncio
async def test_delete_and_stats_404(db_session):
    # удаление несуществующей ссылки
    with pytest.raises(Exception):
        await delete_short_link("unknown", session=db_session)

    # stats 404
    with pytest.raises(Exception):
        await get_link_stats("unknown", session=db_session)


