import os
from sqlalchemy.ext.asyncio import (
    AsyncEngine,
    AsyncSession,
    create_async_engine,
    async_sessionmaker,
)
from sqlalchemy.orm import DeclarativeBase

DATABASE_URL = os.getenv("DATABASE_URL", "postgresql://user:pass@host:port/db")
ASYNC_DATABASE_URL = DATABASE_URL.replace("postgresql://", "postgresql+asyncpg://")

class Base(DeclarativeBase):
    pass

# Echo=False for prod; flip to True while debugging SQL
async_engine: AsyncEngine = create_async_engine(ASYNC_DATABASE_URL, echo=False, future=True)

# Use expire_on_commit=False so objects keep attributes after commit
AsyncSessionLocal = async_sessionmaker(
    bind=async_engine,
    class_=AsyncSession,
    expire_on_commit=False,
)