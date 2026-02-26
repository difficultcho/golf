from typing import Annotated

from fastapi import APIRouter, Depends, HTTPException, status
from pydantic import BaseModel
from sqlalchemy.ext.asyncio import AsyncSession

from app.auth import (
    wechat_code2session,
    create_token,
    get_current_user,
    require_role,
)
from app.database import get_db
from app.models.db_models import User, UserRole
from app.services import user_service

router = APIRouter(prefix="/api/auth", tags=["auth"])


# --- Request / Response schemas ---

class LoginRequest(BaseModel):
    code: str  # wx.login() code


class LoginResponse(BaseModel):
    token: str
    user_id: int
    nickname: str
    role: str
    is_new_user: bool


class UserInfo(BaseModel):
    id: int
    nickname: str
    avatar_url: str | None
    role: str
    created_at: str

    @classmethod
    def from_orm_user(cls, u: User) -> "UserInfo":
        return cls(
            id=u.id,
            nickname=u.nickname,
            avatar_url=u.avatar_url,
            role=u.role.value,
            created_at=u.created_at.isoformat(),
        )


class UpdateProfileRequest(BaseModel):
    nickname: str | None = None
    avatar_url: str | None = None


class UpdateRoleRequest(BaseModel):
    role: UserRole


class UserListResponse(BaseModel):
    total: int
    page: int
    size: int
    items: list[UserInfo]


# --- Endpoints ---

@router.post("/login", response_model=LoginResponse)
async def login(body: LoginRequest, db: Annotated[AsyncSession, Depends(get_db)]):
    """WeChat Mini Program login: exchange code for JWT."""
    openid = await wechat_code2session(body.code)
    user, is_new = await user_service.get_or_create_user(db, openid)
    token = create_token(user.id, user.role.value)
    return LoginResponse(
        token=token,
        user_id=user.id,
        nickname=user.nickname,
        role=user.role.value,
        is_new_user=is_new,
    )


@router.get("/me", response_model=UserInfo)
async def get_me(current_user: Annotated[User, Depends(get_current_user)]):
    """Get current user profile."""
    return UserInfo.from_orm_user(current_user)


@router.put("/me", response_model=UserInfo)
async def update_me(
    body: UpdateProfileRequest,
    current_user: Annotated[User, Depends(get_current_user)],
    db: Annotated[AsyncSession, Depends(get_db)],
):
    """Update current user's nickname / avatar."""
    user = await user_service.update_profile(db, current_user, body.nickname, body.avatar_url)
    return UserInfo.from_orm_user(user)


@router.get("/users", response_model=UserListResponse)
async def list_users(
    page: int = 1,
    size: int = 20,
    _admin: User = Depends(require_role(UserRole.admin)),
    db: AsyncSession = Depends(get_db),
):
    """Admin only: list all users."""
    data = await user_service.list_users(db, page, size)
    return UserListResponse(
        total=data["total"],
        page=data["page"],
        size=data["size"],
        items=[UserInfo.from_orm_user(u) for u in data["items"]],
    )


@router.put("/users/{user_id}/role", response_model=UserInfo)
async def change_role(
    user_id: int,
    body: UpdateRoleRequest,
    _admin: User = Depends(require_role(UserRole.admin)),
    db: AsyncSession = Depends(get_db),
):
    """Admin only: change a user's role."""
    try:
        user = await user_service.update_role(db, user_id, body.role)
    except ValueError as e:
        raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail=str(e))
    return UserInfo.from_orm_user(user)
