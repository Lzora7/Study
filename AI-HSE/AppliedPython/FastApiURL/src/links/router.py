from datetime import datetime, timezone
from typing import Optional
import string
import secrets
import re

from fastapi import APIRouter, Depends, HTTPException, Path, Response
from sqlalchemy import select, insert, delete, update
from sqlalchemy.ext.asyncio import AsyncSession

from database import get_async_session
from links.models import links
from links.schemas import LinkCreate, LinkOut, LinkUpdate, LinkStats

router = APIRouter(prefix="/links", tags=["links"])
open_router = APIRouter()



def _generate_short_code() -> str:
    alphabet = string.ascii_letters + string.digits
    return "".join(secrets.choice(alphabet) for _ in range(7))


async def _ensure_unique_short_code(session: AsyncSession, desired: Optional[str]) -> str:
    # Reserved words that cannot be used as short codes
    reserved = {"search", "shorten", "docs", "openapi.json", "redoc", "auth", "links", "report", "protected-route", "unprotected-route", "favicon.ico"}
    
    if desired:
        if desired in reserved:
            raise HTTPException(status_code=400, detail="custom_alias cannot be a reserved word")
        query = select(links.c.id).where(links.c.short_code == desired)
        existing = await session.execute(query)
        if existing.scalar_one_or_none() is not None:
            raise HTTPException(status_code=409, detail="custom_alias already exists")
        return desired
    # generate unique
    for _ in range(10):
        candidate = _generate_short_code()
        if candidate not in reserved:
            query = select(links.c.id).where(links.c.short_code == candidate)
            existing = await session.execute(query)
            if existing.scalar_one_or_none() is None:
                return candidate
    raise HTTPException(status_code=500, detail="Failed to generate unique short code")


@router.post("/shorten", response_model=LinkOut)
async def create_short_link(payload: LinkCreate, session: AsyncSession = Depends(get_async_session)):
    short_code = await _ensure_unique_short_code(session, payload.custom_alias)
    now = datetime.now(timezone.utc).replace(tzinfo=None)
    
    # Convert expires_at to naive datetime if it's timezone-aware
    expires_at = payload.expires_at
    if expires_at and expires_at.tzinfo is not None:
        expires_at = expires_at.replace(tzinfo=None)
    
    stmt = insert(links).values(
        short_code=short_code,
        original_url=str(payload.original_url),
        created_at=now,
        expires_at=expires_at,
        click_count=0,
    ).returning(links.c.short_code, links.c.original_url, links.c.created_at, links.c.expires_at)
    row = (await session.execute(stmt)).first()
    await session.commit()
    return LinkOut(
        short_code=row.short_code,
        original_url=row.original_url,
        created_at=row.created_at,
        expires_at=row.expires_at,
    )


@router.get("/search")
async def search_by_original_url(original_url: str, session: AsyncSession = Depends(get_async_session)):
    query = select(links.c.short_code).where(links.c.original_url == original_url)
    result = await session.execute(query)
    codes = [row[0] for row in result.all()]
    return {"short_codes": codes}


# removed: /links/info/{short_code}








@router.get("/{short_code}")
async def redirect_short_link(
    short_code: str = Path(pattern=r"^[A-Za-z0-9_-]{3,64}$"),
    session: AsyncSession = Depends(get_async_session),
):
    # Prevent conflicts with system routes
    if short_code in ["search", "shorten"]:
        raise HTTPException(status_code=404, detail="Short link not found")
    
    now = datetime.now(timezone.utc).replace(tzinfo=None)
    query = select(links).where(links.c.short_code == short_code)
    result = await session.execute(query)
    row = result.mappings().first()
    if row is None:
        raise HTTPException(status_code=404, detail="Short link not found")
    if row["expires_at"] is not None and row["expires_at"] <= now:
        # auto delete expired
        await session.execute(delete(links).where(links.c.id == row["id"]))
        await session.commit()
        raise HTTPException(status_code=404, detail="Short link expired")

    # increment click_count and update last_accessed_at
    await session.execute(
        update(links)
        .where(links.c.id == row["id"]) 
        .values(click_count=links.c.click_count + 1, last_accessed_at=now)
    )
    await session.commit()
    return {
        "short_code": short_code,
        "original_url": row["original_url"],
        "message": "redirect suppressed for Swagger/CORS; use this URL in browser",
    }


@open_router.get("/{short_code}")
async def redirect_short_link_root(
    short_code: str = Path(...),
    session: AsyncSession = Depends(get_async_session),
):
    # Prevent conflicts with FastAPI system routes - reject anything with specific reserved words
    reserved_routes = ["docs", "openapi.json", "redoc", "auth", "report", "links"]
    if short_code in reserved_routes or short_code in ["protected-route", "unprotected-route"]:
        raise HTTPException(status_code=404, detail="Not found")

    # Only accept short codes with specific pattern (letters, digits, underscore, dash)
    if not re.match(r"^[A-Za-z0-9_-]{3,64}$", short_code):
        raise HTTPException(status_code=404, detail="Not found")

    now = datetime.now(timezone.utc).replace(tzinfo=None)
    query = select(links).where(links.c.short_code == short_code)
    result = await session.execute(query)
    row = result.mappings().first()
    if row is None:
        raise HTTPException(status_code=404, detail="Short link not found")
    if row["expires_at"] is not None and row["expires_at"] <= now:
        await session.execute(delete(links).where(links.c.id == row["id"]))
        await session.commit()
        raise HTTPException(status_code=404, detail="Short link expired")

    # update stats but do not actually redirect — return JSON instead
    await session.execute(
        update(links)
        .where(links.c.id == row["id"]) 
        .values(click_count=links.c.click_count + 1, last_accessed_at=now)
    )
    await session.commit()
    return {
        "short_code": short_code,
        "original_url": row["original_url"],
        "message": "redirect suppressed for Swagger/CORS; use this URL in browser",
    }


# removed: /links/safe-preview/{short_code}


@router.get("/{short_code}/stats", response_model=LinkStats)
async def get_link_stats(short_code: str, session: AsyncSession = Depends(get_async_session)):
    query = select(links).where(links.c.short_code == short_code)
    result = await session.execute(query)
    row = result.mappings().first()
    if row is None:
        raise HTTPException(status_code=404, detail="Short link not found")
    return LinkStats(
        short_code=row["short_code"],
        original_url=row["original_url"],
        created_at=row["created_at"],
        expires_at=row["expires_at"],
        click_count=row["click_count"],
        last_accessed_at=row["last_accessed_at"],
    )


@router.delete("/{short_code}", status_code=204)
async def delete_short_link(short_code: str, session: AsyncSession = Depends(get_async_session)):
    res = await session.execute(delete(links).where(links.c.short_code == short_code))
    if res.rowcount == 0:
        raise HTTPException(status_code=404, detail="Short link not found")
    await session.commit()
    return Response(status_code=204)


@router.put("/{short_code}", response_model=LinkOut)
async def update_short_link(short_code: str, payload: LinkUpdate, session: AsyncSession = Depends(get_async_session)):
    # fetch existing
    result = await session.execute(select(links).where(links.c.short_code == short_code))
    existing = result.mappings().first()
    if not existing:
        raise HTTPException(status_code=404, detail="Short link not found")

    values = {}
    if payload.new_short_code and payload.new_short_code != existing["short_code"]:
        # ensure uniqueness
        _ = await _ensure_unique_short_code(session, payload.new_short_code)
        values["short_code"] = payload.new_short_code
    if payload.new_original_url:
        values["original_url"] = str(payload.new_original_url)
    if payload.expires_at is not None:
        # Convert expires_at to naive datetime if it's timezone-aware
        expires_at = payload.expires_at
        if expires_at.tzinfo is not None:
            expires_at = expires_at.replace(tzinfo=None)
        values["expires_at"] = expires_at

    if not values:
        # nothing to update
        return LinkOut(
            short_code=existing["short_code"],
            original_url=existing["original_url"],
            created_at=existing["created_at"],
            expires_at=existing["expires_at"],
        )

    stmt = (
        update(links)
        .where(links.c.id == existing["id"])
        .values(**values)
        .returning(links.c.short_code, links.c.original_url, links.c.created_at, links.c.expires_at)
    )
    row = (await session.execute(stmt)).first()
    await session.commit()
    return LinkOut(
        short_code=row.short_code,
        original_url=row.original_url,
        created_at=row.created_at,
        expires_at=row.expires_at,
    )


# removed: /links/preview-url