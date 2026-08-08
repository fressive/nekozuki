"""API routes for summarization control, RAG query, and writeup preview."""

import asyncio
import hashlib
import json
import logging
import re
from collections import defaultdict
from functools import lru_cache
from pathlib import Path

from fastapi import APIRouter, File, HTTPException, Request, UploadFile
from fastapi.responses import JSONResponse, StreamingResponse

from src.config import settings
from src.models import AddWriteupRequest, ProgressEvent, QueryRequest, QueryResponse

logger = logging.getLogger(__name__)

router = APIRouter(prefix="/api")


class SummarizeJob:
    """Tracks the state of a running summarization job."""

    def __init__(self):
        self.task: asyncio.Task | None = None
        self.extractor = None
        self.pause_requested = False
        self.last_event: dict = {}
        self.subscribers: list[asyncio.Queue] = []

    @property
    def is_running(self) -> bool:
        return self.task is not None and not self.task.done()

    async def broadcast(self, event: dict) -> None:
        """Send an event to all SSE subscribers."""
        self.last_event = event
        for queue in self.subscribers:
            await queue.put(event)

    def add_subscriber(self) -> asyncio.Queue:
        queue = asyncio.Queue(maxsize=100)
        self.subscribers.append(queue)
        return queue


# Module-level singleton job manager
job = SummarizeJob()


@router.get("/health")
async def health() -> dict:
    """Health check endpoint."""
    return {"status": "ok"}


@router.post("/summarize/start")
async def start_summarization() -> dict:
    """Start (or resume) the summarization pipeline."""
    if job.is_running:
        return {"status": "already_running", "message": "Summarization is already running"}

    from src.summarization.extractor import TrickExtractor

    try:
        job.extractor = TrickExtractor()
    except ValueError as e:
        return JSONResponse(status_code=500, content={"status": "error", "message": str(e)})

    job.pause_requested = False

    async def _run():
        try:
            async for event in job.extractor.extract_all():
                await job.broadcast(event.model_dump())
                if event.status in ("paused", "completed", "failed"):
                    break
        except asyncio.CancelledError:
            logger.info("Summarization task cancelled")
        except Exception as e:
            logger.exception("Summarization task failed: %s", e)  # noqa: TRY401
            await job.broadcast(ProgressEvent(
                status="failed",
                message=str(e),
            ).model_dump())

    job.task = asyncio.create_task(_run())

    return {
        "status": "started",
        "message": "Summarization started",
    }


@router.post("/summarize/pause")
async def pause_summarization() -> dict:
    """Request a pause after the current batch completes."""
    if not job.is_running:
        return {"status": "not_running", "message": "No summarization running"}

    job.pause_requested = True
    await job.extractor.request_pause()

    return {"status": "pause_requested", "message": "Pausing after current batch"}


@router.get("/summarize/status")
async def summarization_status() -> dict:
    """Get the current summarization progress."""
    from src.summarization.checkpoint import CheckpointManager

    checkpoint = CheckpointManager().load()

    progress_pct = (
        checkpoint.batch_index / checkpoint.total_batches * 100
        if checkpoint.total_batches > 0 else 0
    )

    return {
        "status": checkpoint.status,
        "batch_index": checkpoint.batch_index,
        "total_batches": checkpoint.total_batches,
        "tricks_extracted": checkpoint.total_tricks_extracted,
        "tokens_used": checkpoint.total_tokens_used,
        "progress_pct": round(progress_pct, 1),
        "is_running": job.is_running,
    }


@router.get("/summarize/stream")
async def stream_progress(request: Request) -> StreamingResponse:
    """SSE endpoint for real-time progress updates."""
    queue = job.add_subscriber()

    async def event_generator():
        try:
            # Send the last known event immediately
            if job.last_event:
                yield f"event: progress\ndata: {json.dumps(job.last_event)}\n\n"

            while True:
                if await request.is_disconnected():
                    break

                # Timeout after 15s to send a heartbeat and re-check disconnect
                try:
                    event = await asyncio.wait_for(queue.get(), timeout=15.0)
                except TimeoutError:
                    # Heartbeat to keep connection alive
                    yield ": heartbeat\n\n"
                    continue

                yield f"event: progress\ndata: {json.dumps(event)}\n\n"
        finally:
            if queue in job.subscribers:
                job.subscribers.remove(queue)

    return StreamingResponse(
        event_generator(),
        media_type="text/event-stream",
        headers={"Cache-Control": "no-cache", "X-Accel-Buffering": "no"},
    )


@router.get("/techniques")
async def list_techniques() -> dict:
    """List all extracted technique files with trick counts."""
    output_dir = Path(settings.output_dir)
    if not output_dir.exists():
        return {"techniques": []}

    techniques = []
    for file_path in sorted(output_dir.glob("*.md")):
        content = file_path.read_text(encoding="utf-8")
        trick_count = content.count("\n## ")
        techniques.append({
            "name": file_path.stem,
            "tricks": trick_count,
            "size_bytes": file_path.stat().st_size,
        })

    return {"techniques": techniques}


@router.get("/technique/{name}")
async def get_technique(name: str) -> dict:
    """Get the full content of a technique file."""
    file_path = Path(settings.output_dir) / f"{name}.md"
    if not file_path.exists():
        raise HTTPException(status_code=404, detail=f"Technique '{name}' not found")

    return {
        "name": name,
        "content": file_path.read_text(encoding="utf-8"),
    }


# ---- Coarse trick search + detail -----------------------------------------
# A "trick" is one H2 section in an output/*.md technique file. The coarse
# search scans title + description (fast, no rerank) and returns lightweight
# results (id, title, description); the detail endpoint returns the full trick
# by id. Parsing and id assignment are deterministic so ids stay stable across
# re-renders (as long as the title doesn't change).


def parse_trick_section(technique: str, title: str, section: str) -> dict:
    """Parse one ``## title`` section of a rendered technique file.

    Matches the format written by ``TrickDeduplicator._render_trick``:
    Description / Conditions / Implementation (bullets) / Key code/payload /
    Example (fenced) / Detection signs (bullets) / Example challenge.
    """
    result: dict = {
        "technique_name": technique,
        "title": title,
        "description": "",
        "conditions": [],
        "implementation_steps": [],
        "key_code": None,
        "example": None,
        "detection_signs": [],
        "example_challenge": None,
    }
    lines = section.splitlines()
    current: str | None = None  # bullet/fence field we're filling
    fence = False
    fence_content: list[str] = []
    i = 1  # skip the "## title" line
    while i < len(lines):
        line = lines[i]
        stripped = line.strip()
        if line.startswith("```"):
            if fence:
                result[current] = "\n".join(fence_content).strip() or None
                fence_content = []
                current = None
                fence = False
            else:
                fence = True
            i += 1
            continue
        if fence:
            fence_content.append(line)
            i += 1
            continue
        if stripped.startswith("Description:"):
            result["description"] = stripped[len("Description:"):].strip()
            current = None
        elif stripped.startswith("Conditions:"):
            raw = stripped[len("Conditions:"):].strip().rstrip(";")
            result["conditions"] = [c.strip() for c in raw.split(";") if c.strip()]
            current = None
        elif stripped == "Implementation:":
            current = "implementation"
        elif stripped == "Key code/payload:":
            current = "key_code"
        elif stripped == "Example:":
            current = "example"
        elif stripped == "Detection signs:":
            current = "detection"
        elif stripped.startswith("Example challenge:"):
            result["example_challenge"] = stripped[len("Example challenge:"):].strip()
            current = None
        elif current == "implementation" and stripped.startswith("- "):
            result["implementation_steps"].append(stripped[2:].strip())
        elif current == "detection" and stripped.startswith("- "):
            result["detection_signs"].append(stripped[2:].strip())
        i += 1
    return result


def _trick_id(technique: str, title: str) -> str:
    """Stable trick id: technique name + short hash of the title."""
    digest = hashlib.sha1(title.encode("utf-8")).hexdigest()[:10]
    return f"{technique}::{digest}"


@lru_cache(maxsize=1)
def _load_tricks_index() -> list[dict]:
    """Parse every rendered trick (output/*.md H2 section) for search/detail.

    Each entry carries the parsed fields plus internal ``_title_tokens`` /
    ``_content_tokens`` frozensets used by :func:`search_tricks_in`.
    Cached for the lifetime of the process.
    """
    output_dir = Path(settings.output_dir)
    if not output_dir.exists():
        return []
    tricks: list[dict] = []
    for file_path in sorted(output_dir.glob("*.md")):
        technique = file_path.stem
        content = file_path.read_text(encoding="utf-8")
        for section in re.split(r"\n(?=## )", content):
            if not section.startswith("## "):
                continue
            title = section.split("\n", 1)[0][3:].strip()
            trick = parse_trick_section(technique, title, section)
            trick["id"] = _trick_id(technique, title)
            trick["content"] = section.strip()
            trick["_title_tokens"] = frozenset(re.findall(r"\w+", title.lower()))
            trick["_content_tokens"] = frozenset(re.findall(r"\w+", section.lower()))
            tricks.append(trick)
    logger.info("Loaded %d tricks for coarse search from %s", len(tricks), output_dir)
    return tricks


def search_tricks_in(tricks: list[dict], q: str, limit: int = 20) -> list[dict]:
    """Coarse keyword search over parsed tricks (title + content).

    Scores each trick by query-token overlap (title weighted higher, plus a
    bonus when the whole query is a title substring) and returns the top
    ``limit`` results as ``{id, technique_name, title, description}``.
    """
    tokens = set(re.findall(r"\w+", q.lower())) if q else set()
    q_lower = q.lower().strip() if q else ""
    if not tokens and not q_lower:
        return []

    scored: list[tuple[float, dict]] = []
    for trick in tricks:
        score = 0.0
        for tok in tokens:
            if tok in trick["_title_tokens"]:
                score += 3.0
            elif tok in trick["_content_tokens"]:
                score += 1.0
        if q_lower and q_lower in trick["title"].lower():
            score += 5.0
        if score > 0:
            scored.append((score, trick))
    scored.sort(key=lambda st: -st[0])

    results = []
    for _score, trick in scored[:limit]:
        results.append({
            "id": trick["id"],
            "technique_name": trick["technique_name"],
            "title": trick["title"],
            "description": (trick.get("description") or "")[:300],
        })
    return results


@lru_cache(maxsize=1)
def _trick_sources_map() -> dict[tuple[str, str], list[str]]:
    """(technique, title) -> union of source writeup URLs from tricks_all.json.

    Rendered output/*.md does not record source URLs, so the detail endpoint
    looks them up here (best effort) to link back to the writeups.
    """
    path = Path(settings.tricks_dir) / "tricks_all.json"
    if not path.exists():
        return {}
    try:
        with open(path, "r", encoding="utf-8") as f:
            tricks = json.load(f)
    except (json.JSONDecodeError, OSError):
        return {}
    m: dict[tuple[str, str], list[str]] = defaultdict(list)
    for t in tricks:
        key = (t.get("technique_name", ""), t.get("title", ""))
        if key[0] and key[1]:
            m[key].extend(t.get("source_writeups", []))
    return {k: list(dict.fromkeys(v)) for k, v in m.items()}


@router.get("/tricks/search")
async def search_tricks(q: str = "", limit: int = 20) -> dict:
    """Coarse search over rendered tricks; returns id/title/description.

    Fast keyword scan (no rerank) — the lightweight result list the UI can show
    before fetching full details via ``GET /api/tricks/{id}``.
    """
    if not q.strip():
        raise HTTPException(status_code=422, detail="q parameter is required")
    results = search_tricks_in(_load_tricks_index(), q, limit=min(max(limit, 1), 100))
    return {"query": q, "results": results, "total": len(results)}


@router.get("/tricks/{trick_id}")
async def get_trick(trick_id: str) -> dict:
    """Full detail of one trick, looked up by its id."""
    tricks = _load_tricks_index()
    for trick in tricks:
        if trick["id"] == trick_id:
            detail = {
                key: trick[key]
                for key in (
                    "id", "technique_name", "title", "description", "conditions",
                    "implementation_steps", "key_code", "example", "example_challenge",
                    "detection_signs",
                )
            }
            detail["source_writeups"] = _trick_sources_map().get(
                (trick["technique_name"], trick["title"]), []
            )
            return detail
    raise HTTPException(status_code=404, detail=f"Trick '{trick_id}' not found")


@router.post("/rag/query")
async def rag_query(request: QueryRequest) -> QueryResponse:
    """Run a RAG query over the technique index."""
    if not request.query.strip():
        raise HTTPException(status_code=422, detail="Query must not be empty")

    from src.retrieval.index import load_or_build_index

    searcher = load_or_build_index()
    if searcher is None:
        raise HTTPException(
            status_code=503,
            detail="Index not built. Run `nekozuki build-index` first.",
        )

    results = await searcher.search(request.query, top_k=request.top_k)

    return QueryResponse(
        results=results,
        query=request.query,
        time_ms=0.0,
    )


# ---- Writeup upload & preview ----

_preview_jobs: dict[str, dict] = {}


# ---- Writeup ↔ trick lookup ----

@lru_cache(maxsize=1)
def _build_writeup_trick_index() -> dict[str, list[dict]]:
    """Build an inverted index: writeup URL → list of tricks extracted from it.

    Cached indefinitely (the cache is invalidated on server restart).
    """
    path = Path(settings.tricks_dir) / "tricks_all.json"
    if not path.exists():
        return {}
    try:
        with open(path, "r", encoding="utf-8") as f:
            tricks = json.load(f)
    except (json.JSONDecodeError, OSError):
        return {}

    index: dict[str, list[dict]] = defaultdict(list)
    for trick in tricks:
        for url in trick.get("source_writeups", []):
            index[url].append(trick)
    return dict(index)


@lru_cache(maxsize=1)
def _build_writeup_metadata() -> dict[str, tuple[str, str]]:
    """Build a mapping from writeup URL to (challenge_title, challenge_source).

    Uses the same cache lookup as the deduplicator's _load_url_map.
    """
    result: dict[str, tuple[str, str]] = {}
    # Prefer the cleaned cache
    cache_path = Path(settings.cleaned_writeups_path)
    if cache_path.exists():
        try:
            with open(cache_path, "r", encoding="utf-8") as f:
                for line in f:
                    line = line.strip()
                    if not line:
                        continue
                    item = json.loads(line)
                    url = item.get("url", "")
                    title = item.get("challenge_title", item.get("challenge_name", ""))
                    source = item.get("challenge_source", "")
                    if url and title:
                        result[url] = (title, source)
            return result
        except (json.JSONDecodeError, OSError):
            pass
    # Fallback to raw data.json
    data_path = Path(settings.data_path)
    if data_path.exists():
        try:
            with open(data_path, "r", encoding="utf-8") as f:
                for item in json.load(f):
                    url = item.get("url", "")
                    title = item.get("challenge_title", item.get("challenge_name", ""))
                    source = item.get("challenge_source", "")
                    if url and title:
                        result[url] = (title, source)
        except (json.JSONDecodeError, OSError):
            pass
    return result


@router.get("/writeups/search")
async def search_writeups(q: str = "", limit: int = 50) -> dict:
    """Search writeups by challenge name, showing which have tricks."""
    meta = _build_writeup_metadata()
    index = _build_writeup_trick_index()

    results = []
    q_lower = q.lower().strip() if q else ""

    for url, (title, source) in sorted(meta.items(), key=lambda kv: kv[1][0]):
        if q_lower and q_lower not in title.lower() and q_lower not in source.lower():
            continue
        trick_count = len(index.get(url, []))
        results.append({
            "url": url,
            "title": title,
            "source": source,
            "trick_count": trick_count,
        })
        if len(results) >= limit:
            break

    return {"results": results, "total": len(results)}


@router.get("/writeup/tricks")
async def get_writeup_tricks(url: str = "") -> dict:
    """Get all tricks extracted from a specific writeup URL."""
    if not url:
        raise HTTPException(status_code=422, detail="url parameter is required")

    index = _build_writeup_trick_index()
    tricks = index.get(url, [])

    meta = _build_writeup_metadata()
    info = meta.get(url, ("", ""))

    # Summarize each trick for display
    summarized = []
    for t in tricks:
        summarized.append({
            "technique_name": t.get("technique_name", ""),
            "title": t.get("title", ""),
            "category": t.get("category", ""),
            "description": t.get("description", "")[:200],
            "confidence": t.get("confidence", 0),
            "key_code": t.get("key_code", ""),
            "example": t.get("example", ""),
        })

    return {
        "url": url,
        "challenge_title": info[0],
        "challenge_source": info[1],
        "tricks": summarized,
        "total": len(summarized),
    }


@router.post("/summarize/upload")
async def upload_writeup(file: UploadFile = File(...)) -> dict:  # noqa: B008 (FastAPI injects UploadFile)
    """Upload a markdown writeup and preview its summarization.

    The writeup is processed by the extraction pipeline (single writeup),
    and the resulting tricks are returned for preview.
    """
    import uuid

    content = (await file.read()).decode("utf-8", errors="replace")
    if not content.strip():
        raise HTTPException(status_code=422, detail="Empty file content")

    job_id = str(uuid.uuid4())
    _preview_jobs[job_id] = {"status": "processing", "content": content}

    async def _process():
        try:
            from src.llm import LLMClient
            from src.models import Writeup
            from src.processing.batch import format_batch_for_prompt
            from src.processing.clean import clean_html_content
            from src.summarization.prompts import build_extraction_prompt

            writeup = Writeup(
                url="upload://preview",
                challenge_title=file.filename or "uploaded writeup",
                challenge_name="uploaded-writeup",
                challenge_source="user upload",
                content=content,
                cleaned_content=clean_html_content(content),
            )

            # Direct LLM call — no TrickExtractor overhead (signal handlers,
            # checkpoint, semaphore) that could hang in the server context.
            llm = LLMClient()
            writeup_text = format_batch_for_prompt([writeup])
            system_prompt, user_message = build_extraction_prompt(writeup_text)
            response = await llm.create_message(
                system_prompt=system_prompt,
                user_message=user_message,
                cache_system=True,
            )

            if isinstance(response, list):
                tricks = response
            elif isinstance(response, dict):
                tricks = response.get("tricks", [])
            else:
                tricks = []

            _preview_jobs[job_id] = {"status": "completed", "tricks": tricks}
        except Exception as e:
            logger.exception("Preview processing failed: %s", e)  # noqa: TRY401
            _preview_jobs[job_id] = {"status": "failed", "error": str(e)}

    asyncio.create_task(_process())
    return {"job_id": job_id, "status": "processing"}


@router.get("/summarize/preview/{job_id}")
async def get_preview(job_id: str) -> dict:
    """Get the summarization result for a preview job."""
    if job_id not in _preview_jobs:
        raise HTTPException(status_code=404, detail="Preview job not found")

    result = _preview_jobs[job_id]
    if result["status"] == "processing":
        return {"status": "processing"}
    if result["status"] == "failed":
        return {"status": "failed", "error": result["error"]}

    return {"status": "completed", "tricks": result["tricks"]}


# ---- Writeup ingestion via the pipeline (URL or pasted content) ----

_url_ingestion_jobs: dict[str, dict] = {}


async def _process_writeup_job(
    job_id: str,
    url: str = "",
    content: str = "",
    title: str = "",
    source: str = "",
) -> None:
    """Extract tricks from a writeup (URL or pasted content) and persist them
    to the pipeline (tricks.jsonl / tricks_all.json).

    This intentionally does NOT re-run dedup/embed automatically — those are
    heavy steps the user triggers when ready (e.g. ``nekozuki dedup-tricks`` /
    the rebuild-index button), so adding a single writeup stays fast. Runs as a
    background task; progress is polled via
    ``GET /api/writeup/ingest-status/{job_id}``.
    """
    try:
        if content.strip():
            from src.ingestion import ingest_writeup_from_content_async

            tricks = await ingest_writeup_from_content_async(
                content=content,
                challenge_title=title,
                challenge_source=source,
                url=url,
                persist=True,
            )
        else:
            from src.ingestion import ingest_writeup_from_url_async

            tricks = await ingest_writeup_from_url_async(url, persist=True)

        _url_ingestion_jobs[job_id] = {
            "status": "completed",
            "tricks": [t.model_dump() for t in tricks],
            "total": len(tricks),
        }
    except Exception as e:
        logger.exception("Writeup ingestion failed: %s", e)  # noqa: TRY401
        _url_ingestion_jobs[job_id] = {"status": "failed", "error": str(e)}


@router.post("/writeup/add")
async def add_writeup(request: AddWriteupRequest) -> dict:
    """Add a writeup to the pipeline: fetch a URL or accept pasted content.

    Runs the writeup through fetch/clean → LLM trick extraction → persist to
    tricks.jsonl/tricks_all.json. Dedup/embed are NOT run automatically (see
    :func:`_process_writeup_job`). One of ``url`` or ``content`` is required;
    an optional ``url`` is recorded on the extracted tricks as the source link.
    Returns a job ID to poll.
    """
    import uuid

    if not request.url.strip() and not request.content.strip():
        raise HTTPException(
            status_code=422, detail="Provide a url or content (or both)"
        )

    job_id = str(uuid.uuid4())
    _url_ingestion_jobs[job_id] = {"status": "queued", "url": request.url}
    asyncio.create_task(_process_writeup_job(
        job_id,
        url=request.url,
        content=request.content,
        title=request.challenge_title,
        source=request.challenge_source,
    ))
    return {"job_id": job_id, "status": "queued"}


@router.post("/writeup/from-url")
async def ingest_writeup_from_url(url: str = "") -> dict:
    """Fetch a writeup from a URL, extract tricks, and persist to the pipeline.

    Convenience wrapper over ``POST /api/writeup/add`` for URL-only ingestion.
    Returns a job ID pollable via GET /api/writeup/ingest-status/{job_id}.
    """
    import uuid

    if not url.strip():
        raise HTTPException(status_code=422, detail="url parameter is required")

    job_id = str(uuid.uuid4())
    _url_ingestion_jobs[job_id] = {"status": "queued", "url": url}
    asyncio.create_task(_process_writeup_job(job_id, url=url))
    return {"job_id": job_id, "status": "queued"}


@router.get("/writeup/ingest-status/{job_id}")
async def get_ingest_status(job_id: str) -> dict:
    """Get the status of a writeup-ingestion job."""
    if job_id not in _url_ingestion_jobs:
        raise HTTPException(status_code=404, detail="Ingestion job not found")
    return _url_ingestion_jobs[job_id]


# ---- BM25 index rebuild (background) ----

_build_index_jobs: dict[str, dict] = {}


async def _run_build_index_job(job_id: str, force: bool) -> None:
    """Split output/*.md into chunks and rebuild the tantivy BM25 index.

    Runs off the event loop (BM25 indexing is CPU-bound). Afterward the running
    server's cached ``HybridSearcher`` is evicted so the next query loads the
    fresh index (no restart needed).
    """
    try:
        from src.retrieval.bm25_index import BM25Index

        bm25 = BM25Index()
        built = await asyncio.to_thread(bm25.build_from_output_dir, force=force)
        if built is None:
            _build_index_jobs[job_id] = {
                "status": "failed",
                "error": "No technique files found in output/",
            }
            return

        # Evict the process-level searcher cache so queries use the new index.
        try:
            from src.retrieval.index import load_or_build_index
            load_or_build_index.cache_clear()
            logger.info("Cleared HybridSearcher cache after BM25 rebuild")
        except Exception:  # cache eviction is best-effort
            logger.warning("Failed to clear HybridSearcher cache", exc_info=True)

        _build_index_jobs[job_id] = {
            "status": "completed",
            "chunks": len(built.chunks),
            "index_path": str(built.index_path),
        }
    except Exception as e:
        logger.exception("BM25 index rebuild failed: %s", e)  # noqa: TRY401
        _build_index_jobs[job_id] = {"status": "failed", "error": str(e)}


@router.post("/build-index")
async def build_index(force: bool = True) -> dict:
    """Rebuild the BM25 search index from output/*.md, as a background job.

    Splits every technique file into chunks and rebuilds the tantivy BM25
    index. Much faster than ``nekozuki embed`` (no embedding/Chroma/question
    generation). The running server's cached searcher is refreshed
    automatically. Returns a job ID pollable via
    ``GET /api/build-index/status/{job_id}``.
    """
    import uuid

    job_id = str(uuid.uuid4())
    _build_index_jobs[job_id] = {"status": "queued", "force": force}
    asyncio.create_task(_run_build_index_job(job_id, force))
    return {"job_id": job_id, "status": "queued", "force": force}


@router.get("/build-index/status/{job_id}")
async def build_index_status(job_id: str) -> dict:
    """Get the status of a BM25 index rebuild job."""
    if job_id not in _build_index_jobs:
        raise HTTPException(status_code=404, detail="Build-index job not found")
    return _build_index_jobs[job_id]


# ---- Full pipeline reprocess (dedup → embed → BM25), background ----

_reprocess_jobs: dict[str, dict] = {}


async def _run_reprocess_job(job_id: str, generate_questions: bool = False) -> None:
    """Re-run the whole post-ingestion pipeline at once.

    Steps: (1) re-dedup + re-render output/*.md, (2) incrementally embed the
    changed chunks into Chroma, (3) rebuild the BM25 keyword index, (4) evict
    the running server's cached searcher. Use after adding one or more writeups
    via ``/api/writeup/add`` to process all accumulated tricks together instead
    of running these heavy steps on every single add.
    """
    try:
        # 1. Re-dedup + re-render output/*.md (CPU-bound, run off the loop).
        from src.summarization.deduplicator import run_deduplication

        written = await asyncio.to_thread(run_deduplication)

        # 2. Incrementally embed changed chunks into Chroma (hash-based diff).
        embed = {}
        try:
            from src.embedding.engine import EmbeddingEngine

            engine = EmbeddingEngine()
            embed = await engine.generate_all(
                force_reset=False, generate_questions=generate_questions
            )
        except ValueError as e:
            logger.warning("Skipping embed (no embedding key): %s", e)
        except Exception as e:  # noqa: BLE001 (embed is best-effort; keep BM25)
            logger.warning("Embed failed (%s); continuing with BM25", e)

        # 3. Rebuild the BM25 keyword index from the re-rendered output.
        from src.retrieval.bm25_index import BM25Index

        bm25 = BM25Index()
        built = await asyncio.to_thread(bm25.build_from_output_dir, force=True)
        bm25_chunks = len(built.chunks) if built is not None else 0

        # 4. Evict the running server's cached searcher so queries use the new index.
        try:
            from src.retrieval.index import load_or_build_index
            load_or_build_index.cache_clear()
        except Exception:  # cache eviction is best-effort
            logger.warning("Failed to clear HybridSearcher cache", exc_info=True)

        _reprocess_jobs[job_id] = {
            "status": "completed",
            "dedup_wrote": len(written),
            "embed_chunks": int(embed.get("chunks", 0)),
            "bm25_chunks": bm25_chunks,
        }
    except Exception as e:
        logger.exception("Pipeline reprocess failed: %s", e)  # noqa: TRY401
        _reprocess_jobs[job_id] = {"status": "failed", "error": str(e)}


@router.post("/reprocess")
async def reprocess(questions: bool = False) -> dict:
    """Re-run the full pipeline: dedup + re-render, incremental embed, BM25 rebuild.

    Use after adding one or more writeups via ``/api/writeup/add`` to process
    all accumulated tricks at once. ``questions`` optionally enables
    question generation/embedding (off by default). Returns a job ID polled via
    ``GET /api/reprocess/status/{job_id}``.
    """
    import uuid

    job_id = str(uuid.uuid4())
    _reprocess_jobs[job_id] = {"status": "queued", "questions": questions}
    asyncio.create_task(_run_reprocess_job(job_id, generate_questions=questions))
    return {"job_id": job_id, "status": "queued", "questions": questions}


@router.get("/reprocess/status/{job_id}")
async def reprocess_status(job_id: str) -> dict:
    """Get the status of a full-pipeline reprocess job."""
    if job_id not in _reprocess_jobs:
        raise HTTPException(status_code=404, detail="Reprocess job not found")
    return _reprocess_jobs[job_id]


# ---- Embedding pipeline preview (extract → split → questions) ----

_embed_preview_jobs: dict[str, dict] = {}


@router.post("/embed/preview")
async def embed_preview(file: UploadFile = File(...)) -> dict:  # noqa: B008 (FastAPI injects UploadFile)
    """Preview the full embedding pipeline for a writeup.

    Runs the writeup through the same steps the summarization + embedding
    pipeline would: LLM trick extraction → technique-file formatting → chunk
    splitting → preset-question generation.  Returns the intermediate results
    (tricks, formatted markdown, chunks, questions) so the UI can show what the
    embed step would produce.

    As with the upload preview, this does NOT persist anything to the pipeline.
    """
    import uuid

    content = (await file.read()).decode("utf-8", errors="replace")
    if not content.strip():
        raise HTTPException(status_code=422, detail="Empty file content")

    job_id = str(uuid.uuid4())
    _embed_preview_jobs[job_id] = {"status": "processing"}

    async def _process():
        try:
            from src.embedding.questions import QuestionGenerator
            from src.embedding.splitter import MarkdownAwareTextSplitter
            from src.llm import LLMClient
            from src.models import Writeup
            from src.processing.batch import format_batch_for_prompt
            from src.processing.clean import clean_html_content
            from src.summarization.prompts import build_extraction_prompt

            # 1. Extract tricks (same as the upload preview)
            writeup = Writeup(
                url="upload://preview",
                challenge_title=file.filename or "uploaded writeup",
                challenge_name="uploaded-writeup",
                challenge_source="user upload",
                content=content,
                cleaned_content=clean_html_content(content),
            )
            llm = LLMClient()
            system_prompt, user_message = build_extraction_prompt(
                format_batch_for_prompt([writeup])
            )
            response = await llm.create_message(
                system_prompt=system_prompt,
                user_message=user_message,
                cache_system=True,
            )
            tricks = (
                response if isinstance(response, list) else response.get("tricks", [])
            )

            # 2. Format as a technique file (frontmatter + H2 trick sections)
            technique_md = _format_tricks_as_technique_file(tricks)
            technique_name = "preview_technique"

            # 3. Split into chunks
            splitter = MarkdownAwareTextSplitter()
            chunks = splitter.split_technique_file(technique_md, technique_name)
            chunk_preview = [
                {
                    "section_title": c.section_title,
                    "content": c.content,
                    "token_count": c.token_count,
                    "chunk_id": c.chunk_id,
                }
                for c in chunks
            ]

            # 4. Generate preset questions
            questions = []
            try:
                qgen = QuestionGenerator()
                questions = await qgen.generate_for_file(technique_md, technique_name)
            except Exception as e:  # noqa: BLE001 (question gen is best-effort)
                logger.warning("Question generation failed: %s", e)

            _embed_preview_jobs[job_id] = {
                "status": "completed",
                "tricks": tricks,
                "technique_md": technique_md,
                "chunks": chunk_preview,
                "questions": questions,
            }
        except Exception as e:
            logger.exception("Embed preview failed: %s", e)  # noqa: TRY401
            _embed_preview_jobs[job_id] = {"status": "failed", "error": str(e)}

    asyncio.create_task(_process())
    return {"job_id": job_id, "status": "processing"}


@router.get("/embed/preview/{job_id}")
async def get_embed_preview(job_id: str) -> dict:
    """Get the result of an embed-preview job."""
    if job_id not in _embed_preview_jobs:
        raise HTTPException(status_code=404, detail="Embed preview job not found")
    return _embed_preview_jobs[job_id]


def _format_tricks_as_technique_file(tricks: list[dict]) -> str:
    """Format a list of tricks into a technique-file markdown string (in memory).

    Mirrors the deduplicator's output format so the splitter preview is
    representative of what ends up in the real ``output/*.md`` files.
    """
    if not tricks:
        return "---\ncategory: uncategorized\n---\n# Preview\n\n_No tricks extracted._"

    category = tricks[0].get("category", "uncategorized") or "uncategorized"
    sections = []
    for trick in tricks:
        title = trick.get("title") or trick.get("technique_name") or "Untitled"
        lines = [f"## {title}"]
        if trick.get("description"):
            lines.append(f"\nDescription: {trick['description']}")
        if trick.get("conditions"):
            lines.append(f"\nConditions: {'; '.join(trick['conditions'])}")
        if trick.get("implementation_steps"):
            lines.append(f"\nImplementation: {'; '.join(trick['implementation_steps'])}")
        if trick.get("example"):
            lines.append(f"\nExample:\n```\n{trick['example']}\n```")
        if trick.get("detection_signs"):
            lines.append(f"\nDetection signs: {'; '.join(trick['detection_signs'])}")
        sections.append("\n".join(lines))

    return f"---\ncategory: {category}\n---\n# Preview\n\n" + "\n\n".join(sections)