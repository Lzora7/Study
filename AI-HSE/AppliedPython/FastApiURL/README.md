# URL Shortener Service

Сервис для сокращения длинных ссылок с аналитикой и управлением. Пользователи могут создавать короткие ссылки, получать статистику переходов и управлять временем жизни ссылок.

## Функционал

- ✅ Создание коротких ссылок (автогенерация или кастомный alias)
- ✅ Редирект по коротким ссылкам
- ✅ Статистика переходов (количество кликов, даты)
- ✅ Обновление и удаление ссылок
- ✅ Поиск по оригинальному URL
- ✅ Время жизни ссылок с автоудалением
- ✅ Аутентификация пользователей (JWT)
- ✅ Фоновая очистка просроченных ссылок

## API Endpoints

### Короткие ссылки

#### POST `/links/shorten` - Создание короткой ссылки
```json
{
  "original_url": "https://example.com/very/long/path",
  "custom_alias": "demo",
  "expires_at": "2026-12-31T23:59:00"
}
```

**Ответ:**
```json
{
  "short_code": "demo",
  "original_url": "https://example.com/very/long/path",
  "created_at": "2025-08-12T12:00:00.123456",
  "expires_at": "2026-12-31T23:59:00"
}
```

#### GET `/{short_code}` - Редирект по короткой ссылке
Возвращает HTTP 307 с заголовком `Location` на оригинальный URL.

#### GET `/links/{short_code}/stats` - Статистика ссылки
```json
{
  "short_code": "demo",
  "original_url": "https://example.com/very/long/path",
  "created_at": "2025-08-12T12:00:00.123456",
  "expires_at": "2026-12-31T23:59:00",
  "click_count": 42,
  "last_accessed_at": "2025-08-12T15:30:00.987654"
}
```

#### PUT `/links/{short_code}` - Обновление ссылки
```json
{
  "new_short_code": "new_demo",
  "new_original_url": "https://example.org/updated",
  "expires_at": "2027-01-01T00:00:00"
}
```

#### DELETE `/links/{short_code}` - Удаление ссылки
Возвращает HTTP 204 (No Content).

#### GET `/links/search?original_url={url}` - Поиск по URL
```json
{
  "short_codes": ["demo1", "demo2"]
}
```

### Аутентификация

#### POST `/auth/register` - Регистрация
```json
{
  "email": "user@example.com",
  "password": "secure_password"
}
```

#### POST `/auth/jwt/login` - Вход
```json
{
  "username": "user@example.com",
  "password": "secure_password"
}
```

**Ответ:**
```json
{
  "access_token": "eyJ0eXAiOiJKV1QiLCJhbGciOiJIUzI1NiJ9...",
  "token_type": "bearer"
}
```

## Примеры запросов

### Создание и использование короткой ссылки

```bash
# 1. Создать короткую ссылку
curl -X POST http://localhost:9999/links/shorten \
  -H "Content-Type: application/json" \
  -d '{
    "original_url": "https://example.com/very/long/path",
    "custom_alias": "mylink",
    "expires_at": "2026-12-31T23:59:00"
  }'

# 2. Перейти по ссылке (в браузере или через curl)
curl -I http://localhost:9999/mylink

# 3. Посмотреть статистику
curl http://localhost:9999/links/mylink/stats

# 4. Обновить ссылку
curl -X PUT http://localhost:9999/links/mylink \
  -H "Content-Type: application/json" \
  -d '{
    "new_short_code": "newlink",
    "new_original_url": "https://example.org/new"
  }'

# 5. Удалить ссылку
curl -X DELETE http://localhost:9999/links/newlink
```

### Работа с аутентификацией

```bash
# Регистрация
curl -X POST http://localhost:9999/auth/register \
  -H "Content-Type: application/json" \
  -d '{
    "email": "user@example.com",
    "password": "secure_password"
  }'

# Вход
curl -X POST http://localhost:9999/auth/jwt/login \
  -H "Content-Type: application/x-www-form-urlencoded" \
  -d "username=user@example.com&password=secure_password"

# Использование токена
curl http://localhost:9999/protected-route \
  -H "Authorization: Bearer YOUR_TOKEN_HERE"
```

## Инструкция по запуску

### Требования
- Docker и Docker Compose
- Свободные порты: 9999 (API), 8888 (Flower)

### Запуск через Docker (рекомендуется)

1. **Клонируйте репозиторий:**
```bash
git clone <repository-url>
cd FastApi-URL
```

2. **Создайте файл окружения `.env`:**
```env
DB_NAME=postgres
DB_USER=postgres
DB_PASS=postgres
SMTP_USER=your-email@gmail.com
SMTP_PASSWORD=your-app-password
SECRET=your-secret-key-here
```

3. **Запустите сервисы:**
```bash
docker compose up -d
```

4. **Проверьте работу:**
- API: http://localhost:9999/docs (Swagger UI)
- Flower (мониторинг задач): http://localhost:8888

### Локальный запуск

1. **Установите зависимости:**
```bash
python -m venv .venv
source .venv/bin/activate  # Linux/Mac
# .venv\Scripts\activate   # Windows
pip install -r requirements.txt
```

2. **Настройте окружение:**
- Запустите PostgreSQL и Redis
- Установите переменные окружения или создайте `.env`

3. **Примените миграции:**
```bash
alembic upgrade head
```

4. **Запустите приложение:**
```bash
# FastAPI
uvicorn src.main:app --reload --host 0.0.0.0 --port 9999

# Celery (в отдельном терминале)
cd src && celery --app=tasks.tasks:celery worker -B -l INFO
```

### Остановка
```bash
docker compose down
```

## База данных

### Схема БД

#### Таблица `links`
```sql
CREATE TABLE links (
    id SERIAL PRIMARY KEY,
    short_code VARCHAR(64) UNIQUE NOT NULL,
    original_url TEXT NOT NULL,
    created_at TIMESTAMP NOT NULL,
    expires_at TIMESTAMP NULL,
    click_count INTEGER NOT NULL DEFAULT 0,
    last_accessed_at TIMESTAMP NULL
);

CREATE UNIQUE INDEX ix_links_short_code ON links (short_code);
```

#### Таблица `user` (аутентификация)
```sql
CREATE TABLE user (
    id UUID PRIMARY KEY,
    email VARCHAR NOT NULL,
    hashed_password VARCHAR NOT NULL,
    registered_at TIMESTAMP DEFAULT NOW(),
    is_active BOOLEAN NOT NULL DEFAULT TRUE,
    is_superuser BOOLEAN NOT NULL DEFAULT FALSE,
    is_verified BOOLEAN NOT NULL DEFAULT FALSE
);
```

### Особенности
- **PostgreSQL** с асинхронными соединениями (`asyncpg`)
- **Alembic** для миграций
- **Redis** для кэширования и брокера Celery
- **SQLAlchemy Core** для работы с БД

### Автоматическая очистка
Celery-задача `purge_expired_links` запускается каждую минуту и удаляет просроченные ссылки (`expires_at <= NOW()`).

## Архитектура

```
FastAPI App
├── src/
│   ├── main.py           # Точка входа, сборка роутеров
│   ├── config.py         # Конфигурация (env переменные)
│   ├── database.py       # Подключение к БД
│   ├── auth/             # Аутентификация (fastapi-users)
│   ├── links/            # Сервис коротких ссылок
│   └── tasks/            # Celery задачи
├── migrations/           # Миграции Alembic
└── docker/              # Скрипты запуска контейнеров
```

## Технологии

- **FastAPI** - веб-фреймворк
- **PostgreSQL** - основная БД
- **Redis** - кэш и брокер задач
- **Celery** - фоновые задачи
- **Alembic** - миграции БД
- **SQLAlchemy** - ORM/Core
- **Pydantic** - валидация данных
- **fastapi-users** - аутентификация
- **Docker** - контейнеризация

## Разработка

### Добавление новых эндпоинтов
1. Создайте роутер в соответствующем модуле
2. Добавьте схемы Pydantic для валидации
3. Подключите роутер в `main.py`

### Изменения БД
1. Измените модели в `models.py`
2. Создайте миграцию: `alembic revision --autogenerate -m "описание"`
3. Примените: `alembic upgrade head`

### Фоновые задачи
Добавляйте новые задачи в `src/tasks/tasks.py` с декоратором `@celery.task`.
