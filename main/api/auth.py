"""Workspace authentication shared by browser sessions and API clients."""

import ipaddress
import os
import secrets
import threading
import time

from fastapi import APIRouter, HTTPException, Request, Response
from fastapi.responses import JSONResponse
from pydantic import BaseModel

from core.config import settings

router = APIRouter(prefix="/api/auth")
AUTH_DEV_MODE = os.getenv("APP_DEV_MODE", "false").strip().lower() in {"true", "1", "yes"}
AUTH_SECRET_PASSWORD = os.getenv("APP_AUTH_PASSWORD", "")
_COOKIE = "researchdesk_session"
_SESSION_SECONDS = 12 * 60 * 60
_sessions: dict[str, float] = {}
_session_lock = threading.Lock()
_public_routes = {
    ("GET", "/api/health"),
    ("GET", "/api/auth/status"),
    ("POST", "/api/auth/login"),
    ("POST", "/api/auth/logout"),
}


class LoginRequest(BaseModel):
    password: str = ""


def _is_loopback(host: str | None) -> bool:
    if not host:
        return False
    if host.lower() == "localhost":
        return True
    try:
        return ipaddress.ip_address(host).is_loopback
    except ValueError:
        return False


def _local_dev_access(request: Request) -> bool:
    return (
        AUTH_DEV_MODE
        and not AUTH_SECRET_PASSWORD
        and request.client is not None
        and _is_loopback(request.client.host)
        and _is_loopback(request.url.hostname)
    )


def _credential(request: Request) -> tuple[str | None, bool]:
    header = request.headers.get("authorization")
    if header is None:
        return request.cookies.get(_COOKIE), False
    scheme, _, token = header.partition(" ")
    return (token.strip() or None, True) if scheme.lower() == "bearer" else (None, True)


def _valid_session(token: str | None) -> bool:
    if not token:
        return False
    with _session_lock:
        expiry = _sessions.get(token)
        if expiry is None:
            return False
        if expiry <= time.monotonic():
            _sessions.pop(token)
            return False
        return True


def _same_origin(request: Request) -> bool:
    origin = request.headers.get("origin")
    return not origin or origin == f"{request.url.scheme}://{request.url.netloc}"


def check_api_access(request: Request) -> Response | None:
    path = request.url.path
    method = request.method
    if (method, path) in _public_routes:
        return None
    if method == "OPTIONS" and request.headers.get("origin") and request.headers.get("access-control-request-method"):
        return None

    token, bearer = _credential(request)
    valid = _valid_session(token)
    if not valid and not _local_dev_access(request):
        return JSONResponse(
            {"detail": "Workspace authentication required."},
            status_code=401,
            headers={"WWW-Authenticate": "Bearer"},
        )
    if not (valid and bearer) and method not in {"GET", "HEAD", "OPTIONS"} and not _same_origin(request):
        return JSONResponse({"detail": "Cross-origin workspace writes are forbidden."}, status_code=403)
    return None


@router.get("/status")
def auth_status(request: Request):
    local_dev = _local_dev_access(request)
    authenticated = local_dev or _valid_session(_credential(request)[0])
    return {
        "authenticated": authenticated,
        "configured": bool(AUTH_SECRET_PASSWORD) or local_dev,
        "write_access": authenticated,
        "mode": settings.EXECUTION_MODE if authenticated else None,
        "dev_mode": local_dev,
    }


@router.post("/login")
def auth_login(request: Request, response: Response, req: LoginRequest | None = None):
    if not _same_origin(request):
        raise HTTPException(status_code=403, detail="Cross-origin login is forbidden.")
    if not AUTH_SECRET_PASSWORD:
        raise HTTPException(status_code=503, detail="Workspace password is not configured.")
    password = req.password if req else ""
    if not secrets.compare_digest(password.encode("utf-8"), AUTH_SECRET_PASSWORD.encode("utf-8")):
        raise HTTPException(status_code=401, detail="Invalid authentication credentials.")

    token = secrets.token_urlsafe(32)
    now = time.monotonic()
    previous, _ = _credential(request)
    with _session_lock:
        for expired in [key for key, expiry in _sessions.items() if expiry <= now]:
            _sessions.pop(expired)
        if previous:
            _sessions.pop(previous, None)
        _sessions[token] = now + _SESSION_SECONDS
    response.set_cookie(
        _COOKIE, token, max_age=_SESSION_SECONDS, path="/api",
        httponly=True, secure=request.url.scheme == "https", samesite="strict",
    )
    return {"authenticated": True, "token": token, "write_access": True, "dev_mode": False}


@router.post("/logout")
def auth_logout(request: Request, response: Response):
    if not _same_origin(request):
        raise HTTPException(status_code=403, detail="Cross-origin logout is forbidden.")
    token, _ = _credential(request)
    with _session_lock:
        for value in (token, request.cookies.get(_COOKIE)):
            if value:
                _sessions.pop(value, None)
    response.delete_cookie(
        _COOKIE, path="/api", httponly=True, samesite="strict",
        secure=request.url.scheme == "https",
    )
    return {"authenticated": False}
