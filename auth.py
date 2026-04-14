"""
DataPilot — Auth & Tier Middleware
===================================
Drop-in auth layer using JWT tokens + Supabase (or any Postgres).
Supports Free / Pro / Team tiers with per-endpoint enforcement.

Swap SUPABASE_URL / SUPABASE_KEY with your own, or replace
`verify_token()` with any JWT library you prefer.
"""

import os
import time
import hashlib
from functools import wraps
from typing import Optional
from datetime import datetime, timedelta, timezone

import jwt
from fastapi import HTTPException, Request, Depends
from fastapi.security import HTTPBearer, HTTPAuthorizationCredentials
from pydantic import BaseModel

# ── Config ─────────────────────────────────────────────
JWT_SECRET   = os.environ.get("JWT_SECRET", "change-me-in-production-please")
JWT_ALGO     = "HS256"
JWT_EXPIRE_H = 72   # hours

# ── Tier definitions ────────────────────────────────────
TIERS = {
    "free": {
        "label":          "Free",
        "price_monthly":  0,
        "max_rows":       500,
        "max_datasets":   1,
        "max_cols":       20,
        "ml_allowed":     False,
        "forecast_allowed": False,
        "ai_query_allowed": False,
        "export_watermark": True,
        "api_calls_per_day": 20,
        "max_file_mb":    2,
        "pdf_reports":    False,
        "team_seats":     1,
    },
    "pro": {
        "label":          "Pro",
        "price_monthly":  19,
        "max_rows":       500_000,
        "max_datasets":   50,
        "max_cols":       500,
        "ml_allowed":     True,
        "forecast_allowed": True,
        "ai_query_allowed": True,
        "export_watermark": False,
        "api_calls_per_day": 2000,
        "max_file_mb":    50,
        "pdf_reports":    True,
        "team_seats":     1,
    },
    "team": {
        "label":          "Team",
        "price_monthly":  50,
        "max_rows":       5_000_000,
        "max_datasets":   500,
        "max_cols":       2000,
        "ml_allowed":     True,
        "forecast_allowed": True,
        "ai_query_allowed": True,
        "export_watermark": False,
        "api_calls_per_day": 20_000,
        "max_file_mb":    500,
        "pdf_reports":    True,
        "team_seats":     20,
    },
}

# ── In-memory rate limiter (replace with Redis in production) ──
_rate_store: dict[str, list[float]] = {}

def check_rate_limit(user_id: str, tier: str) -> bool:
    """Returns True if request is allowed, False if rate-limited."""
    limit = TIERS[tier]["api_calls_per_day"]
    now   = time.time()
    window = 86400  # 24h window

    calls = _rate_store.get(user_id, [])
    # Drop calls older than window
    calls = [t for t in calls if now - t < window]
    if len(calls) >= limit:
        return False
    calls.append(now)
    _rate_store[user_id] = calls
    return True


# ── JWT helpers ─────────────────────────────────────────
def create_token(user_id: str, email: str, tier: str = "free") -> str:
    payload = {
        "sub":   user_id,
        "email": email,
        "tier":  tier,
        "iat":   datetime.now(timezone.utc),
        "exp":   datetime.now(timezone.utc) + timedelta(hours=JWT_EXPIRE_H),
    }
    return jwt.encode(payload, JWT_SECRET, algorithm=JWT_ALGO)


def decode_token(token: str) -> dict:
    try:
        return jwt.decode(token, JWT_SECRET, algorithms=[JWT_ALGO])
    except jwt.ExpiredSignatureError:
        raise HTTPException(401, "Token expired — please log in again")
    except jwt.InvalidTokenError:
        raise HTTPException(401, "Invalid token")


# ── FastAPI dependency ──────────────────────────────────
security = HTTPBearer(auto_error=False)

class CurrentUser(BaseModel):
    user_id: str
    email:   str
    tier:    str

    @property
    def limits(self) -> dict:
        return TIERS[self.tier]


def get_current_user(
    request: Request,
    credentials: Optional[HTTPAuthorizationCredentials] = Depends(security),
) -> CurrentUser:
    """
    FastAPI dependency — extracts and validates JWT.
    Attach to any route: `user: CurrentUser = Depends(get_current_user)`
    """
    token = None

    # 1. Check Authorization header
    if credentials:
        token = credentials.credentials

    # 2. Check cookie fallback (web UI sends this)
    if not token:
        token = request.cookies.get("dp_token")

    if not token:
        raise HTTPException(401, "Authentication required. Please log in.")

    payload = decode_token(token)
    user = CurrentUser(
        user_id=payload["sub"],
        email=payload["email"],
        tier=payload.get("tier", "free"),
    )

    # Rate limit check
    if not check_rate_limit(user.user_id, user.tier):
        limit = TIERS[user.tier]["api_calls_per_day"]
        raise HTTPException(
            429,
            f"Daily API limit reached ({limit} calls/day on {user.tier} plan). "
            "Upgrade for higher limits."
        )

    return user


def require_tier(*allowed_tiers: str):
    """
    Decorator factory for tier-gating endpoints.

    Usage:
        @app.post("/train-model")
        async def train(req: TrainReq, user: CurrentUser = Depends(require_tier("pro","team"))):
            ...
    """
    def dependency(user: CurrentUser = Depends(get_current_user)) -> CurrentUser:
        if user.tier not in allowed_tiers:
            needed = " or ".join(t.title() for t in allowed_tiers)
            raise HTTPException(
                403,
                f"This feature requires a {needed} plan. "
                f"You're on {user.tier.title()}. Upgrade at /pricing"
            )
        return user
    return Depends(dependency)


def check_row_limit(user: CurrentUser, row_count: int):
    """Raises 403 if uploaded dataset exceeds tier row limit."""
    limit = user.limits["max_rows"]
    if row_count > limit:
        raise HTTPException(
            403,
            f"Your {user.tier.title()} plan supports up to {limit:,} rows. "
            f"This file has {row_count:,} rows. Upgrade to Pro for up to 500,000 rows."
        )


def check_file_size(user: CurrentUser, size_bytes: int):
    """Raises 403 if file exceeds tier size limit."""
    limit_mb = user.limits["max_file_mb"]
    size_mb  = size_bytes / 1_000_000
    if size_mb > limit_mb:
        raise HTTPException(
            403,
            f"File is {size_mb:.1f} MB. Your {user.tier.title()} plan supports "
            f"up to {limit_mb} MB per file. Upgrade to Pro for up to 50 MB."
        )
