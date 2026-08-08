"""FastAPI web application for nekozuki."""

import logging
from pathlib import Path

from fastapi import FastAPI, Request
from fastapi.responses import JSONResponse, RedirectResponse
from fastapi.staticfiles import StaticFiles
from fastapi.templating import Jinja2Templates

from src.api.auth import is_authenticated
from src.api.routes import router as api_router

logger = logging.getLogger(__name__)

# Locate templates relative to this file (src/ui/templates)
UI_DIR = Path(__file__).parent / "ui"

app = FastAPI(
    title="Nekozuki",
    description="CTF Writeup Summarization & RAG System",
    version="0.1.0",
)

templates = Jinja2Templates(directory=str(UI_DIR / "templates"))
app.mount("/static", StaticFiles(directory=str(UI_DIR / "static")), name="static")

app.include_router(api_router)


# ---- Auth gate ------------------------------------------------------------
# Public surface: health, login/logout, RAG query, coarse trick search/detail,
# technique browsing, and the public pages. Everything else under /api/ and the
# admin pages require a valid session (see src/api/auth.py). When AUTH_PASSWORD
# is unset, auth is disabled and everything is allowed.

_PUBLIC_API_PREFIXES = (
    "/api/health",
    "/api/login",
    "/api/logout",
    "/api/rag/query",
    "/api/techniques",
    "/api/technique/",
    "/api/tricks/",
)
_PUBLIC_PAGE_PREFIXES = ("/static/", "/search", "/trick/", "/technique/", "/login")


@app.middleware("http")
async def auth_gate(request: Request, call_next):
    path = request.url.path

    if path.startswith("/api/"):
        if any(path.startswith(p) for p in _PUBLIC_API_PREFIXES):
            return await call_next(request)
        if not is_authenticated(request):
            return JSONResponse({"detail": "Authentication required"}, status_code=401)
        return await call_next(request)

    # Non-API page: admin pages require login, public pages don't.
    if path == "/" or any(path.startswith(p) for p in _PUBLIC_PAGE_PREFIXES):
        return await call_next(request)
    if not is_authenticated(request):
        return RedirectResponse(url="/login", status_code=302)
    return await call_next(request)


# ---- Page routes (server-rendered HTML) ----

@app.get("/login", include_in_schema=False)
async def login_page(request: Request):
    return templates.TemplateResponse(request, "login.html", {"authed": False})


@app.get("/", include_in_schema=False)
async def index(request: Request):
    return templates.TemplateResponse(request, "index.html", {"authed": is_authenticated(request)})


@app.get("/progress", include_in_schema=False)
async def progress_page(request: Request):
    return templates.TemplateResponse(request, "progress.html", {"authed": True})


@app.get("/preview", include_in_schema=False)
async def preview_page(request: Request):
    return templates.TemplateResponse(request, "preview.html", {"authed": True})


@app.get("/search", include_in_schema=False)
async def search_page(request: Request):
    return templates.TemplateResponse(
        request, "search.html", {"authed": is_authenticated(request)}
    )


@app.get("/trick/{trick_id}", include_in_schema=False)
async def trick_page(request: Request, trick_id: str):
    """Server-rendered view of a single trick (by id, from coarse search)."""
    from fastapi import HTTPException

    from src.api.routes import get_trick

    try:
        detail = await get_trick(trick_id)
    except HTTPException:
        return templates.TemplateResponse(
            request, "trick_detail.html",
            {"detail": None, "error": "No trick with this id.", "authed": is_authenticated(request)},
            status_code=404,
        )
    return templates.TemplateResponse(
        request, "trick_detail.html",
        {"detail": detail, "authed": is_authenticated(request)},
    )


@app.get("/writeups", include_in_schema=False)
async def writeups_page(request: Request):
    return templates.TemplateResponse(request, "writeups.html", {"authed": True})


@app.get("/ingest-url", include_in_schema=False)
async def ingest_url_page(request: Request):
    return templates.TemplateResponse(request, "ingest_url.html", {"authed": True})


@app.get("/embed-preview", include_in_schema=False)
async def embed_preview_page(request: Request):
    return templates.TemplateResponse(request, "embed_preview.html", {"authed": True})


@app.get("/technique/{name}", include_in_schema=False)
async def technique_page(request: Request, name: str):
    """Server-rendered view of a single technique file."""
    from pathlib import Path

    from src.config import settings

    file_path = Path(settings.output_dir) / f"{name}.md"
    if not file_path.exists():
        return templates.TemplateResponse(
            request, "technique.html",
            {"name": name, "content": "Technique not found.", "authed": is_authenticated(request)},
            status_code=404,
        )
    content = file_path.read_text(encoding="utf-8")
    return templates.TemplateResponse(
        request, "technique.html",
        {"name": name, "content": content, "authed": is_authenticated(request)},
    )