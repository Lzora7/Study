from fastapi import Depends
from fastapi_users.db import SQLAlchemyBaseUserTableUUID, SQLAlchemyUserDatabase
from sqlalchemy.ext.asyncio import AsyncSession
from sqlalchemy.orm import DeclarativeBase
from database import engine, get_async_session

# базовый declarative класс для fastapi-users.
class Base(DeclarativeBase):
    pass

# ORM модель пользователя
class User(SQLAlchemyBaseUserTableUUID, Base):
    pass

# создание таблиц в БД
async def create_db_and_tables():
    async with engine.begin() as conn:
        await conn.run_sync(Base.metadata.create_all)

# получение пользователя из БД
async def get_user_db(session: AsyncSession = Depends(get_async_session)):
    yield SQLAlchemyUserDatabase(session, User)