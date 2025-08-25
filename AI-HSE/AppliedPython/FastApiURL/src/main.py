from fastapi import FastAPI, Depends
from fastapi.middleware.cors import CORSMiddleware
from collections.abc import AsyncIterator
from contextlib import asynccontextmanager
from auth.users import auth_backend, current_active_user, fastapi_users
from auth.schemas import UserCreate, UserRead #, UserUpdate
from auth.db import User, create_db_and_tables
from tasks.router import router as tasks_router
import os
from redis import asyncio as aioredis
from fastapi_cache import FastAPICache
from links.router import router as links_router, open_router as links_open_router

import uvicorn

# инициализация Redis для кэширования
@asynccontextmanager
async def lifespan(_: FastAPI) -> AsyncIterator[None]:
    disable_cache = os.getenv("DISABLE_CACHE", "0").lower() in {"1", "true", "yes"}
    if not disable_cache:
        # Import Redis backend lazily to avoid importing aioredis during tests
        from fastapi_cache.backends.redis import RedisBackend
        redis_host = os.getenv("REDIS_HOST", "localhost")
        redis_port = os.getenv("REDIS_PORT", "6379")
        redis = aioredis.from_url(f"redis://{redis_host}:{redis_port}")
        FastAPICache.init(RedisBackend(redis), prefix="fastapi-cache")
    # await create_db_and_tables()
    yield

# создание FastAPI приложения
app = FastAPI(lifespan=lifespan)

# настройка CORS для работы с Swagger UI
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # В продакшене указать конкретные домены
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# подключение роутеров
app.include_router(
    fastapi_users.get_auth_router(auth_backend), prefix="/auth/jwt", tags=["auth"]
)
app.include_router(
    fastapi_users.get_register_router(UserRead, UserCreate),
    prefix="/auth",
    tags=["auth"],
)

app.include_router(tasks_router)
app.include_router(links_router)

# защищенный эндпоинт
@app.get("/protected-route")
def protected_route(user: User = Depends(current_active_user)):
    return f"Hello, {user.email}"

# незащищенный эндпоинт
@app.get("/unprotected-route")
def unprotected_route():
    return f"Hello, anonym"

# открытый роутер должен быть ПОСЛЕ специфичных роутеров
app.include_router(links_open_router)


if __name__ == "__main__":
    uvicorn.run("main:app", reload=True, host="0.0.0.0", log_level="info")
