"""Simple shared-password auth for nekozuki's admin (non-search) surface.

The public surface (health, RAG query, coarse trick search + detail, technique
browsing, login) needs no auth. Everything else — summarization, writeup
ingestion, reprocess, build-index, previews, writeup↔trick browsing — requires a
valid session.

Sessions are stateless HMAC-signed tokens: ``<expiry-epoch>.<hmac-sha256>``
signed with the ``AUTH_PASSWORD`` secret. The token travels in the ``nekozuki_auth``
cookie (HttpOnly) for browsers and is also accepted as ``Authorization: Bearer``
for scripts. When ``AUTH_PASSWORD`` is empty, auth is disabled (all open).
"""

import hashlib
import hmac
import time

from fastapi import Request

from src.config import settings

AUTH_COOKIE = "nekozuki_auth"
#: Session lifetime (seconds).
AUTH_TOKEN_TTL = 7 * 24 * 3600  # 7 days


def auth_enabled() -> bool:
    """Whether login is required at all (AUTH_PASSWORD set)."""
    return bool(settings.auth_password)


def issue_token(now: float | None = None) -> str:
    """Issue a stateless signed token valid for ``AUTH_TOKEN_TTL``."""
    exp = int((now if now is not None else time.time()) + AUTH_TOKEN_TTL)
    payload = str(exp)
    sig = hmac.new(
        settings.auth_password.encode(), payload.encode(), hashlib.sha256
    ).hexdigest()
    return f"{payload}.{sig}"


def verify_token(token: str) -> bool:
    """Return True if ``token`` is signed with the password and not expired."""
    if not token:
        return False
    try:
        exp_str, sig = token.split(".", 1)
        expected = hmac.new(
            settings.auth_password.encode(), exp_str.encode(), hashlib.sha256
        ).hexdigest()
        if not hmac.compare_digest(sig, expected):
            return False
        return time.time() < float(exp_str)
    except (ValueError, TypeError):
        return False


def is_authenticated(request: Request) -> bool:
    """Whether the request carries a valid session (or auth is disabled)."""
    if not auth_enabled():
        return True
    token = request.cookies.get(AUTH_COOKIE, "")
    if not token:
        auth = request.headers.get("Authorization", "")
        if auth.startswith("Bearer "):
            token = auth[7:].strip()
    return verify_token(token)
