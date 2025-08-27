from datetime import datetime, timedelta, timezone

import pytest


@pytest.mark.asyncio
async def test_create_and_get_stats(client):
    # проверка, что ссылка прошла
    payload = {
        "original_url": "https://example.com/long",
        "custom_alias": "customcode",
    }
    resp = await client.post("/links/shorten", json=payload)
    assert resp.status_code == 200, resp.text
    data = resp.json()
    assert data["short_code"] == "customcode"
    assert data["original_url"] == "https://example.com/long"

    stats = await client.get(f"/links/{data['short_code']}/stats")
    assert stats.status_code == 200
    s = stats.json()
    assert s["short_code"] == data["short_code"]
    assert s["click_count"] == 0


@pytest.mark.asyncio
async def test_redirect_and_click_count(client):
    # автоматический код
    resp = await client.post("/links/shorten", json={"original_url": "https://x.org"})
    assert resp.status_code == 200
    sc = resp.json()["short_code"]

    r1 = await client.get(f"/links/{sc}")  # via /links scope
    assert r1.status_code == 200
    rroot = await client.get(f"/{sc}")  # via open root route
    assert rroot.status_code == 200

    stats = await client.get(f"/links/{sc}/stats")
    assert stats.status_code == 200
    assert stats.json()["click_count"] == 2


@pytest.mark.asyncio
async def test_update_and_delete(client):
    # создание ссылки
    resp = await client.post("/links/shorten", json={"original_url": "https://site.org/a"})
    assert resp.status_code == 200
    sc = resp.json()["short_code"]

    # обновление ссылки
    upd = await client.put(
        f"/links/{sc}",
        json={
            "new_short_code": "newcode123",
            "new_original_url": "https://site.org/b",
        },
    )
    assert upd.status_code == 200
    assert upd.json()["short_code"] == "newcode123"

    # удаление ссылки
    dele = await client.delete("/links/newcode123")
    assert dele.status_code == 204

    # проверка на 404
    stats = await client.get("/links/newcode123/stats")
    assert stats.status_code == 404


@pytest.mark.asyncio
async def test_search(client):
    # создание ссылок
    await client.post("/links/shorten", json={"original_url": "https://foo.bar/x", "custom_alias": "a1"})
    await client.post("/links/shorten", json={"original_url": "https://foo.bar/x", "custom_alias": "a2"})

    res = await client.get("/links/search", params={"original_url": "https://foo.bar/x"})
    assert res.status_code == 200
    codes = set(res.json()["short_codes"])
    assert {"a1", "a2"}.issubset(codes)


