import os
from typing import Any, Dict

import httpx
from fastapi import HTTPException, Request
from fastapi.responses import JSONResponse
from fastapi.security import HTTPAuthorizationCredentials, HTTPBearer

SUPABASE_URL = os.environ.get("SUPABASE_URL", "").rstrip("/")
SUPABASE_ANON_KEY = os.environ.get("SUPABASE_ANON_KEY", "")

_bearer = HTTPBearer(auto_error=False)

API_PREFIX = "/fhe-vault"
PUBLIC_PATHS = frozenset(
    {
        f"{API_PREFIX}/health",
        f"{API_PREFIX}/openapi.json",
        f"{API_PREFIX}/docs",
        f"{API_PREFIX}/redoc",
    }
)


def _forbidden(detail: str = "Invalid or missing Supabase token") -> HTTPException:
    return HTTPException(status_code=403, detail=detail)


def validate_supabase_access_token(token: str) -> Dict[str, Any]:
    if not SUPABASE_URL or not SUPABASE_ANON_KEY:
        raise _forbidden("Supabase auth is not configured")

    url = f"{SUPABASE_URL}/auth/v1/user"
    headers = {
        "apikey": SUPABASE_ANON_KEY,
        "Authorization": f"Bearer {token}",
    }
    with httpx.Client(timeout=15.0) as client:
        response = client.get(url, headers=headers)

    if response.status_code != 200:
        raise _forbidden()

    user = response.json()
    if not user or not user.get("id"):
        raise _forbidden()

    return user


async def resolve_supabase_user(request: Request) -> Dict[str, Any]:
    cached = getattr(request.state, "supabase_user", None)
    if cached is not None:
        return cached

    credentials: HTTPAuthorizationCredentials | None = await _bearer(request)
    if credentials is None or credentials.scheme.lower() != "bearer":
        raise _forbidden("Missing or invalid Authorization header")

    user = validate_supabase_access_token(credentials.credentials)
    request.state.supabase_user = user
    return user


async def supabase_auth_middleware(request: Request, call_next):
    if request.url.path in PUBLIC_PATHS or request.method == "OPTIONS":
        return await call_next(request)

    try:
        await resolve_supabase_user(request)
    except HTTPException as exc:
        return JSONResponse(status_code=exc.status_code, content={"detail": exc.detail})

    return await call_next(request)
