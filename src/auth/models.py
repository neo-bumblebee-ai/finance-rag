"""
RBAC models: roles, permissions, and user/token schemas.

Role hierarchy:
  ADMIN   — full access: ingest, query any ticker, manage users, view all traces
  ANALYST — query tickers assigned to them, view own traces
  VIEWER  — query only (no ingestion, no trace access)
"""

from __future__ import annotations

from enum import Enum
from typing import Literal

from pydantic import BaseModel, Field


# ---------------------------------------------------------------------------
# Roles & permissions
# ---------------------------------------------------------------------------

class Role(str, Enum):
    ADMIN   = "admin"
    ANALYST = "analyst"
    VIEWER  = "viewer"


# Permissions each role holds
ROLE_PERMISSIONS: dict[Role, set[str]] = {
    Role.ADMIN: {
        "query:any",       # query any ticker
        "ingest:run",      # trigger ingestion pipeline
        "users:read",      # list users
        "users:write",     # create / update users
        "traces:read_all", # view all LangSmith / Langfuse traces
    },
    Role.ANALYST: {
        "query:assigned",  # query only tickers in user.allowed_tickers
        "traces:read_own", # view own traces
    },
    Role.VIEWER: {
        "query:assigned",  # same ticker restriction as analyst
    },
}


def has_permission(role: Role, permission: str) -> bool:
    return permission in ROLE_PERMISSIONS.get(role, set())


# ---------------------------------------------------------------------------
# User model (in production: replace with DB-backed UserRepository)
# ---------------------------------------------------------------------------

class User(BaseModel):
    user_id: str
    username: str
    role: Role
    # Tickers this user may query. Empty list = unrestricted (ADMIN only).
    allowed_tickers: list[str] = Field(default_factory=list)
    disabled: bool = False


# ---------------------------------------------------------------------------
# Token schemas
# ---------------------------------------------------------------------------

class TokenPayload(BaseModel):
    sub: str            # user_id
    username: str
    role: Role
    allowed_tickers: list[str]
    exp: int            # Unix timestamp


class TokenResponse(BaseModel):
    access_token: str
    token_type: Literal["bearer"] = "bearer"
    expires_in: int     # seconds


class LoginRequest(BaseModel):
    username: str
    password: str


# ---------------------------------------------------------------------------
# Seeded in-memory user store (replace with DB + bcrypt in production)
# ---------------------------------------------------------------------------
# Passwords here are plaintext for demo clarity.
# In production use: passlib CryptContext with bcrypt rounds=12.

_USER_STORE: dict[str, dict] = {
    "admin": {
        "user_id": "u-001",
        "username": "admin",
        "password": "admin-secret",          # bcrypt hash in prod
        "role": Role.ADMIN,
        "allowed_tickers": [],               # unrestricted
        "disabled": False,
    },
    "alice": {
        "user_id": "u-002",
        "username": "alice",
        "password": "alice-secret",
        "role": Role.ANALYST,
        "allowed_tickers": ["AAPL", "MSFT", "NVDA"],
        "disabled": False,
    },
    "bob": {
        "user_id": "u-003",
        "username": "bob",
        "password": "bob-secret",
        "role": Role.VIEWER,
        "allowed_tickers": ["AAPL"],
        "disabled": False,
    },
}


def get_user_by_username(username: str) -> User | None:
    record = _USER_STORE.get(username)
    if not record:
        return None
    return User(**{k: v for k, v in record.items() if k != "password"})


def verify_password(username: str, password: str) -> bool:
    """Plaintext check for demo. Replace with bcrypt.checkpw() in production."""
    record = _USER_STORE.get(username)
    if not record:
        return False
    return record["password"] == password
