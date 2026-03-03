import asyncio
import logging

from sqlalchemy.ext.asyncio import create_async_engine, async_sessionmaker, AsyncSession
from sqlalchemy.orm import DeclarativeBase
from app.config import settings

logger = logging.getLogger(__name__)

engine = create_async_engine(settings.DATABASE_URL, echo=settings.DEBUG)
async_session = async_sessionmaker(engine, expire_on_commit=False)


class Base(DeclarativeBase):
    pass


async def get_db():
    """FastAPI dependency: yield an async database session."""
    async with async_session() as session:
        yield session


async def init_db(retries: int = 10, delay: float = 3.0):
    """Create all tables on startup, with retry."""
    for attempt in range(1, retries + 1):
        try:
            async with engine.begin() as conn:
                await conn.run_sync(Base.metadata.create_all)
            logger.info("Database connected successfully")
            return
        except Exception as e:
            logger.warning(f"DB connection attempt {attempt}/{retries} failed: {e}")
            if attempt == retries:
                raise
            await asyncio.sleep(delay)
