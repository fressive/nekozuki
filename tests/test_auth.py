"""Tests for the admin auth gate (public search vs protected endpoints/pages)."""

import sys
import time
from pathlib import Path

import pytest

# Ensure project src is importable when running pytest from project root
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import httpx

from src.api import auth
from src.app import app
from src.config import settings


def test_issue_and_verify_token(monkeypatch):
    monkeypatch.setattr(settings, "auth_password", "secret")
    token = auth.issue_token()
    assert auth.verify_token(token)

    # tampered signature is rejected
    assert not auth.verify_token(token[:-3] + "zzz")
    # expired token is rejected
    old = auth.issue_token(now=time.time() - auth.AUTH_TOKEN_TTL - 10)
    assert not auth.verify_token(old)
    # empty / garbage is rejected
    assert not auth.verify_token("")
    assert not auth.verify_token("garbage")


@pytest.mark.asyncio
async def test_auth_gate_public_and_protected(monkeypatch):
    monkeypatch.setattr(settings, "auth_password", "testpass")
    transport = httpx.ASGITransport(app=app)
    async with httpx.AsyncClient(transport=transport, base_url="http://t", follow_redirects=False) as c:
        # public surface: health + search + technique browsing need no login
        assert (await c.get("/api/health")).status_code == 200
        assert (await c.get("/api/techniques")).status_code == 200
        assert (await c.get("/search")).status_code == 200

        # protected surface without login: API 401, page 302 -> /login
        assert (await c.get("/api/summarize/status")).status_code == 401
        r = await c.get("/ingest-url")
        assert r.status_code == 302 and r.headers["location"] == "/login"

        # wrong password rejected
        assert (await c.post("/api/login", json={"password": "wrong"})).status_code == 401

        # correct password -> cookie set, protected surface now works
        r = await c.post("/api/login", json={"password": "testpass"})
        assert r.status_code == 200
        assert "nekozuki_auth" in r.cookies
        assert (await c.get("/api/summarize/status")).status_code == 200
        assert (await c.get("/ingest-url")).status_code == 200


@pytest.mark.asyncio
async def test_bearer_token_accepted(monkeypatch):
    monkeypatch.setattr(settings, "auth_password", "testpass")
    token = auth.issue_token()
    transport = httpx.ASGITransport(app=app)
    async with httpx.AsyncClient(transport=transport, base_url="http://t") as c:
        assert (await c.get("/api/summarize/status")).status_code == 401
        r = await c.get(
            "/api/summarize/status", headers={"Authorization": f"Bearer {token}"}
        )
        assert r.status_code == 200


@pytest.mark.asyncio
async def test_auth_disabled_when_no_password(monkeypatch):
    monkeypatch.setattr(settings, "auth_password", "")
    transport = httpx.ASGITransport(app=app)
    async with httpx.AsyncClient(transport=transport, base_url="http://t") as c:
        # everything open when AUTH_PASSWORD is unset
        assert (await c.get("/api/summarize/status")).status_code == 200
        assert (await c.get("/ingest-url")).status_code == 200
