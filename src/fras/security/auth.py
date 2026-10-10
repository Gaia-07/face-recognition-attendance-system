"""Authentication and role-based access control (RBAC) utilities."""

import os
from datetime import UTC, datetime, timedelta
from enum import StrEnum
from typing import Any

import bcrypt
import jwt

SECRET_KEY = os.getenv("JWT_SECRET_KEY", "dev-secret-gaia-07-biometric-jwt-token-key-2026")
ALGORITHM = "HS256"
DEFAULT_TOKEN_EXPIRE_HOURS = 8


class UserRole(StrEnum):
    ADMIN = "admin"
    FACULTY = "faculty"
    AUDITOR = "auditor"


ROLE_HIERARCHY: dict[UserRole, set[str]] = {
    UserRole.ADMIN: {"admin", "faculty", "auditor"},
    UserRole.FACULTY: {"faculty"},
    UserRole.AUDITOR: {"auditor"},
}


def hash_password(password: str) -> str:
    """Hashes a plaintext password with bcrypt and automatic salt."""
    salt = bcrypt.gensalt(rounds=12)
    hashed = bcrypt.hashpw(password.encode("utf-8"), salt)
    return hashed.decode("utf-8")


def verify_password(plain_password: str, hashed_password: str) -> bool:
    """Verifies a plaintext password against a stored bcrypt hash."""
    return bcrypt.checkpw(plain_password.encode("utf-8"), hashed_password.encode("utf-8"))


def create_access_token(
    data: dict[str, Any],
    expires_delta: timedelta | None = None,
) -> str:
    """Generates a signed JWT bearer token containing claims and expiration."""
    to_encode = data.copy()
    expire = datetime.now(UTC) + (expires_delta or timedelta(hours=DEFAULT_TOKEN_EXPIRE_HOURS))
    to_encode.update({"exp": expire, "iat": datetime.now(UTC)})
    encoded_jwt = jwt.encode(to_encode, SECRET_KEY, algorithm=ALGORITHM)
    return str(encoded_jwt)


def decode_access_token(token: str) -> dict[str, Any]:
    """Decodes and validates a JWT bearer token."""
    payload: dict[str, Any] = jwt.decode(token, SECRET_KEY, algorithms=[ALGORITHM])
    return payload


def has_permission(user_role: str, target_role: str) -> bool:
    """Checks whether the user's role satisfies the target role's permission level."""
    try:
        role_enum = UserRole(user_role)
    except ValueError:
        return False
    return target_role in ROLE_HIERARCHY.get(role_enum, set())
