from datetime import datetime, timedelta, timezone

import pytest


@pytest.mark.asyncio
async def test_reserved_alias(client):
    # проверка на бан код
    resp = await client.post("/links/shorten", json={
        "original_url": "https://ok.ru", 
        "custom_alias": "search"
        }
        )
    assert resp.status_code == 400


@pytest.mark.asyncio
async def test_duplicate_alias(client):
    # проверка на дубликат кода
    await client.post("/links/shorten", json={"original_url": "https://duplicated.ru", "custom_alias": "dup"})
    resp = await client.post("/links/shorten", json={"original_url": "https://duplicated.ru/2", "custom_alias": "dup"})
    assert resp.status_code == 409


@pytest.mark.asyncio
async def test_invalid_url_validation(client):
    # проверка на невалидный url
    resp = await client.post("/links/shorten", json={"original_url": "not-a-url"})
    assert resp.status_code == 422


@pytest.mark.asyncio
async def test_expired_link_auto_delete(client):
    # проверка на срок действия ссылки
    expires_at = (datetime.now(timezone.utc) - timedelta(minutes=1)).isoformat()
    create = await client.post("/links/shorten", json={"original_url": "https://expired.ru", "custom_alias": "exp", "expires_at": expires_at})
    assert create.status_code == 200

    # проверка на удаление и возврат 404
    r = await client.get("/exp")
    assert r.status_code == 404
    assert r.json()["detail"] in {"Short link expired", "Not found"}


