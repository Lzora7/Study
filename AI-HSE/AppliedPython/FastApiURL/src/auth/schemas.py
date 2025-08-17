import uuid

from fastapi_users import schemas

# pydantic модель для чтения пользователя
class UserRead(schemas.BaseUser[uuid.UUID]):
    pass

# pydantic модель для создания пользователя
class UserCreate(schemas.BaseUserCreate):
    pass
