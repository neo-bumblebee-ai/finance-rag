"""
JWT-based RBAC for Finance RAG.

Flow:
  POST /auth/token  →  verify credentials  →  issue JWT
  Any protected endpoint  →  Bearer token  →  decode JWT  →  inject current_user

Token algorithm: HS256 (symmetric).
  Production upgrade path: switch to RS256 with a rotating key pair (private key
  on the auth server, public key on API replicas) — no shared secret needed.

Usage in endpoint:
    @router.get("/ask")
    async def ask(current_user: User = Depends(require_permission("query:any"))):
        ...
"""

from __future__ import annotations

import os
from datetime import datetime, timezone, timedelta
from functools import wraps

from fastapi import Depends, HTTPException, status
from fastapi.security import OAuth2PasswordBearer
from jose import JWTError, jwt

from src.auth.models import (
    Role,
    TokenPayload,
    TokenResponse,
    User,
    get_user_by_username,
    has_permission,
    verify_password,
)

# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------

SECRET_KEY: str = os.environ.get("JWT_SECRET_KEY", "change-me-in-production-use-32-chars-min")
ALGORITHM: str  = "HS256"
ACCESS_TOKEN_EXPIRE_MINUTES: int = int(os.environ.get("JWT_EXPIRE_MINUTES", "60"))

oauth2_scheme = OAuth2PasswordBearer(tokenUrl="/auth/token")


# ---------------------------------------------------------------------------
# Token creation
# ---------------------------------------------------------------------------

def create_access_token(user: User) -> TokenResponse:
    expire = datetime.now(timezone.utc) + timedelta(minutes=ACCESS_TOKEN_EXPIRE_MINUTES)
    payload = {
        "sub": user.user_id,
        "username": user.username,
        "role": user.role.value,
        "allowed_tickers": user.allowed_tickers,
        "exp": int(expire.timestamp()),
    }
    token = jwt.encode(payload, SECRET_KEY, algorithm=ALGORITHM)
    return TokenResponse(
        access_token=token,
        expires_in=ACCESS_TOKEN_EXPIRE_MINUTES * 60,
    )


def authenticate_user(username: str, password: str) -> User:
    """Verify credentials and return User, or raise 401."""
    if not verify_password(username, password):
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Invalid username or password",
            headers={"WWW-Authenticate": "Bearer"},
        )
    user = get_user_by_username(username)
    if user is None or user.disabled:
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="User not found or disabled",
            headers={"WWW-Authenticate": "Bearer"},
        )
    return user


# ---------------------------------------------------------------------------
# FastAPI dependencies
# ---------------------------------------------------------------------------

async def get_current_user(token: str = Depends(oauth2_scheme)) -> User:
    """Decode JWT and return the current User. Raises 401 on any failure."""
    credentials_exception = HTTPException(
        status_code=status.HTTP_401_UNAUTHORIZED,
        detail="Could not validate credentials",
        headers={"WWW-Authenticate": "Bearer"},
    )
    try:
        payload = jwt.decode(token, SECRET_KEY, algorithms=[ALGORITHM])
        user_id: str = payload.get("sub")
        if user_id is None:
            raise credentials_exception
        token_data = TokenPayload(**payload)
    except JWTError:
        raise credentials_exception

    user = get_user_by_username(token_data.username)
    if user is None or user.disabled:
        raise credentials_exception
    return user


def require_permission(permission: str):
    """
    Dependency factory. Returns a FastAPI Depends that checks the user's role
    has the given permission string.

    Usage:
        @router.post("/ingest")
        async def ingest(user: User = Depends(require_permission("ingest:run"))):
            ...
    """
    async def _check(current_user: User = Depends(get_current_user)) -> User:
        if not has_permission(current_user.role, permission):
            raise HTTPException(
                status_code=status.HTTP_403_FORBIDDEN,
                detail=f"Role '{current_user.role}' does not have permission '{permission}'",
            )
        return current_user
    return _check


def require_ticker_access(ticker: str, current_user: User) -> None:
    """
    Raise 403 if the user's role restricts ticker access and the requested
    ticker is not in their allowed_tickers list.
    Called explicitly inside endpoint handlers where the ticker is known.
    """
    if current_user.role == Role.ADMIN:
        return  # unrestricted
    if ticker.upper() not in [t.upper() for t in current_user.allowed_tickers]:
        raise HTTPException(
            status_code=status.HTTP_403_FORBIDDEN,
            detail=f"You are not authorized to query ticker '{ticker}'",
        )
