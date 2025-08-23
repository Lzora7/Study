import os
import smtplib
from email.message import EmailMessage

from celery import Celery
from datetime import datetime, timezone
from sqlalchemy import delete
from sqlalchemy.ext.asyncio import create_async_engine, async_sessionmaker
from config import DB_HOST, DB_NAME, DB_PASS, DB_PORT, DB_USER, SMTP_PASSWORD, SMTP_USER
from celery.schedules import crontab
from links.models import links

SMTP_HOST = "smtp.gmail.com"
SMTP_PORT = 465


_redis_host = os.getenv("REDIS_HOST", "localhost")
_redis_port = os.getenv("REDIS_PORT", "6379")
celery = Celery('tasks', broker=f'redis://{_redis_host}:{_redis_port}')
celery.conf.beat_schedule = {
    "purge-expired-links-every-minute": {
        "task": "tasks.tasks.purge_expired_links",
        "schedule": crontab(),  # каждую минуту
    }
}

# функция для получения шаблона email
def get_template_email(username: str):
    email = EmailMessage()
    email['Subject'] = 'Привет'
    email['From'] = SMTP_USER
    email['To'] = SMTP_USER
    email.set_content(
        '<div>'
        f'<h1 style="color: red;">Здравствуйте, {username}</h1>'
        '</div>',
        subtype='html'
    )
    return email

# функция для отправки email
@celery.task(default_retry_delay=5, max_retries=3)
def send_email(username: str):
    email = get_template_email(username)
    with smtplib.SMTP_SSL(SMTP_HOST, SMTP_PORT) as server:
        print(server.login(SMTP_USER, SMTP_PASSWORD))
        try:
            server.send_message(email)
        except:
            send_email.retry()

# функция для получения url базы данных
def _get_database_url() -> str:
    return f"postgresql+asyncpg://{DB_USER}:{DB_PASS}@{DB_HOST}:{DB_PORT}/{DB_NAME}"

# функция для удаления просроченных ссылок
@celery.task
def purge_expired_links():
    """Удаление просроченных ссылок в фоновом режиме. Запускается в Celery worker."""
    import asyncio

    async def _purge():
        engine = create_async_engine(_get_database_url())
        async_session = async_sessionmaker(engine, expire_on_commit=False)
        async with async_session() as session:
            now = datetime.now(timezone.utc).replace(tzinfo=None)
            await session.execute(delete(links).where((links.c.expires_at.is_not(None)) & (links.c.expires_at <= now)))
            await session.commit()

    asyncio.run(_purge())
