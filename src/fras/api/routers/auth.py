"""Authentication router handling user login and JWT bearer token issuance."""

from typing import Annotated

from fastapi import APIRouter, Depends, HTTPException, status
from fastapi.security import OAuth2PasswordRequestForm
from sqlalchemy import select
from sqlalchemy.orm import Session

from fras.api.schemas import Token
from fras.db.models import User
from fras.db.session import get_db
from fras.security.audit import AuditService
from fras.security.auth import create_access_token, verify_password

router = APIRouter(prefix="/api/auth", tags=["Authentication"])
audit_service = AuditService()


@router.post("/token", response_model=Token)
def login_for_access_token(
    form_data: Annotated[OAuth2PasswordRequestForm, Depends()],
    db: Annotated[Session, Depends(get_db)],
) -> Token:
    """Authenticates username and password; returns JWT access token."""
    user = db.scalars(select(User).where(User.username == form_data.username)).first()

    if not user or not verify_password(form_data.password, user.password_hash):
        audit_service.log_login_attempt(form_data.username, success=False, db=db)
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Incorrect username or password",
            headers={"WWW-Authenticate": "Bearer"},
        )

    audit_service.log_login_attempt(user.username, success=True, db=db)
    access_token = create_access_token(data={"sub": user.username, "role": user.role})
    return Token(
        access_token=access_token,
        role=user.role,
        username=user.username,
    )
