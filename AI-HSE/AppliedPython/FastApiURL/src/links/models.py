from sqlalchemy import Table, Column, Integer, DateTime, MetaData, String

metadata = MetaData()

links = Table(
    "links",
    metadata,
    Column("id", Integer, primary_key=True),
    Column("short_code", String(64), unique=True, index=True, nullable=False),
    Column("original_url", String, nullable=False),
    Column("created_at", DateTime, nullable=False),
    Column("expires_at", DateTime, nullable=True),
    Column("click_count", Integer, nullable=False, default=0),
    Column("last_accessed_at", DateTime, nullable=True),
)


