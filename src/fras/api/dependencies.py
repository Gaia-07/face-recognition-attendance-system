"""FastAPI dependency providers for authentication, authorization, and database sessions."""

from collections.abc import Callable
from typing import Annotated

import jwt
from fastapi import Depends, HTTPException, status
from fastapi.security import OAuth2PasswordBearer
from sqlalchemy import select
from sqlalchemy.orm import Session

from fras.db.models import User
from fras.db.session import get_db
from fras.security.auth import decode_access_token, has_permission

oauth2_scheme = OAuth2PasswordBearer(tokenUrl="/api/auth/token")


def get_current_user(
    token: Annotated[str, Depends(oauth2_scheme)],
    db: Annotated[Session, Depends(get_db)],
) -> User:
    """Validates JWT bearer token and resolves the authenticated User entity."""
    credentials_exception = HTTPException(
        status_code=status.HTTP_401_UNAUTHORIZED,
        detail="Could not validate credentials",
        headers={"WWW-Authenticate": "Bearer"},
    )
    try:
        payload = decode_access_token(token)
        username: str | None = payload.get("sub")
        if username is None:
            raise credentials_exception
    except jwt.PyJWTError:
        raise credentials_exception from None

    user = db.scalars(select(User).where(User.username == username)).first()
    if user is None:
        raise credentials_exception
    return user


def require_role(required_role: str) -> Callable[[User], User]:
    """Factory creating an authorization dependency enforcing RBAC role checks."""

    def role_checker(current_user: Annotated[User, Depends(get_current_user)]) -> User:
        if not has_permission(current_user.role, required_role):
            raise HTTPException(
                status_code=status.HTTP_403_FORBIDDEN,
                detail=f"Operation requires '{required_role}' privileges. Current role: '{current_user.role}'.",
            )
        return current_user

    return role_checker
