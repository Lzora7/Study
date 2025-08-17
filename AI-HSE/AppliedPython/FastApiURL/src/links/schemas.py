from datetime import datetime
from typing import Optional

from pydantic import BaseModel, AnyUrl, Field

# схемы валидации 
class LinkCreate(BaseModel):
    original_url: AnyUrl
    custom_alias: Optional[str] = Field(default=None, max_length=64)
    expires_at: Optional[datetime] = None

class LinkUpdate(BaseModel):
    # Updating the short code (alias) or the original URL
    new_short_code: Optional[str] = Field(default=None, max_length=64)
    new_original_url: Optional[AnyUrl] = None
    expires_at: Optional[datetime] = None

class LinkOut(BaseModel):
    short_code: str
    original_url: AnyUrl
    created_at: datetime
    expires_at: Optional[datetime]

class LinkStats(BaseModel):
    short_code: str
    original_url: AnyUrl
    created_at: datetime
    expires_at: Optional[datetime]
    click_count: int
    last_accessed_at: Optional[datetime]


