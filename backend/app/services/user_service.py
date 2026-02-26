import random
from datetime import datetime, timezone

from sqlalchemy import select, func
from sqlalchemy.ext.asyncio import AsyncSession

from app.models.db_models import User, UserRole


async def get_or_create_user(db: AsyncSession, openid: str) -> tuple[User, bool]:
    """Find existing user by openid, or create a new one.

    Returns (user, created) where created is True if a new user was inserted.
    """
    result = await db.execute(select(User).where(User.openid == openid))
    user = result.scalar_one_or_none()
    if user:
        user.last_login_at = datetime.now(timezone.utc)
        await db.commit()
        return user, False

    tag = random.randint(1000, 9999)
    user = User(openid=openid, nickname=f"球友#{tag}")
    db.add(user)
    await db.commit()
    await db.refresh(user)
    return user, True


async def update_profile(db: AsyncSession, user: User, nickname: str | None, avatar_url: str | None) -> User:
    if nickname is not None:
        user.nickname = nickname
    if avatar_url is not None:
        user.avatar_url = avatar_url
    await db.commit()
    await db.refresh(user)
    return user


async def list_users(db: AsyncSession, page: int = 1, size: int = 20) -> dict:
    offset = (page - 1) * size
    total_result = await db.execute(select(func.count(User.id)))
    total = total_result.scalar()
    result = await db.execute(select(User).offset(offset).limit(size).order_by(User.created_at.desc()))
    users = result.scalars().all()
    return {"total": total, "page": page, "size": size, "items": users}


async def update_role(db: AsyncSession, user_id: int, role: UserRole) -> User:
    result = await db.execute(select(User).where(User.id == user_id))
    user = result.scalar_one_or_none()
    if user is None:
        raise ValueError(f"User {user_id} not found")
    user.role = role
    await db.commit()
    await db.refresh(user)
    return user
