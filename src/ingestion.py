"""Ingest a CTF writeup directly from a URL.

Fetches the page, converts it to clean markdown, runs the same LLM trick
extraction used by the batch pipeline, and persists the resulting tricks to
the reconcilable pipeline files (tricks.jsonl + tricks_all.json) so a normal
`nekozuki summarize` / `nekozuki embed` run picks them up.

Both the CLI (`nekozuki add-url`) and the web API (`POST /api/writeup/from-url`)
call into this module, so the fetch→clean→extract→persist flow stays in one place.
"""

import json
import logging
from pathlib import Path

from src.config import settings
from src.llm import LLMClient
from src.models import Trick, Writeup
from src.processing.batch import format_batch_for_prompt
from src.processing.clean import clean_html_content, is_content_worthwhile
from src.summarization.prompts import build_extraction_prompt

logger = logging.getLogger(__name__)

# Default per-page fetch guardrails.
FETCH_TIMEOUT = 30.0
# Cap the body we feed to the LLM so a single page cannot blow the context.
MAX_INPUT_CHARS = 20000


def fetch_url_content(url: str) -> str:
    """Fetch a URL and return its raw body text.

    Uses a browser-like User-Agent because several writeup hosts (Medium,
    HackTheBox, some self-hosted blogs) return a firewall/403 page otherwise.
    Raises on network errors or non-2xx status codes.
    """
    import httpx

    headers = {
        "User-Agent": (
            "Mozilla/5.0 (X11; Linux x86_64) AppleWebKit/537.36 "
            "(KHTML, like Gecko) Chrome/124.0 Safari/537.36"
        ),
        "Accept-Language": "en-US,en;q=0.9",
    }
    with httpx.Client(
        timeout=FETCH_TIMEOUT, follow_redirects=True, headers=headers
    ) as client:
        resp = client.get(url)
    resp.raise_for_status()

    # If the server returned HTML, return it as-is; clean_html_content handles
    # the stripping. If it is already plain text/markdown, return it verbatim.
    ctype = resp.headers.get("content-type", "")
    if "text/html" in ctype or "application/xhtml" in ctype:
        return resp.text

    text = resp.text
    # Some hosts serve markdown/plain text that is fully usable as-is.
    if text.strip():
        return text
    return ""


def _infer_title(url: str, cleaned: str) -> str:
    """Best-effort title from the URL path or the first non-empty heading."""
    from urllib.parse import urlparse

    path = urlparse(url).path.strip("/")
    if path:
        last = path.split("/")[-1]
        if last and last != "index":
            return last.replace("-", " ").replace("_", " ").title()
    for line in cleaned.splitlines():
        stripped = line.strip().lstrip("#").strip()
        if stripped and len(stripped) < 120:
            return stripped
    return url or "untitled writeup"


def _clean_and_check(content: str, source_label: str) -> str:
    """Clean raw content and enforce the pipeline's length/quality guards."""
    cleaned = clean_html_content(content)
    if not cleaned.strip():
        raise ValueError(
            f"No readable content in {source_label}. The page may be empty, "
            "paywalled, or require JavaScript to render."
        )
    if not is_content_worthwhile(cleaned, min_length=100):
        raise ValueError(
            f"Content in {source_label} is too short ({len(cleaned)} chars) to "
            "be a useful writeup."
        )
    if len(cleaned) > MAX_INPUT_CHARS:
        logger.warning(
            "Truncating content from %d to %d chars", len(cleaned), MAX_INPUT_CHARS
        )
        cleaned = cleaned[:MAX_INPUT_CHARS] + "\n...[truncated for length]"
    return cleaned


async def _extract_tricks_from_writeup(
    writeup: Writeup, llm: LLMClient | None
) -> list[Trick]:
    """Run one writeup through LLM trick extraction (shared by all ingest paths)."""
    llm = llm or LLMClient()
    system_prompt, user_message = build_extraction_prompt(
        format_batch_for_prompt([writeup])
    )
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

    parsed = [Trick(**t) for t in tricks] if isinstance(tricks, list) else []
    # Record the source URL on each trick so the writeup↔trick index links back.
    if writeup.url and writeup.url.startswith("http"):
        for t in parsed:
            if writeup.url not in t.source_writeups:
                t.source_writeups.append(writeup.url)
    return parsed


def _append_trick_to_pipeline(trick: Trick) -> None:
    """Append a single trick to tricks.jsonl and rebuild tricks_all.json."""
    tricks_dir = Path(settings.tricks_dir)
    tricks_dir.mkdir(parents=True, exist_ok=True)

    jsonl_path = tricks_dir / "tricks.jsonl"
    with open(jsonl_path, "a", encoding="utf-8") as f:
        f.write(trick.model_dump_json() + "\n")
    logger.info("Appended trick '%s' to %s", trick.technique_name, jsonl_path)

    # Rebuild the consolidated JSON so downstream tools (dedup, writeup↔trick
    # index) see the new trick without a full re-run.
    all_path = tricks_dir / "tricks_all.json"
    tricks: list[dict] = []
    if jsonl_path.exists():
        with open(jsonl_path, "r", encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                try:
                    tricks.append(json.loads(line))
                except json.JSONDecodeError:
                    continue
    with open(all_path, "w", encoding="utf-8") as f:
        json.dump(tricks, f, indent=2, ensure_ascii=False)
    logger.info("Rebuilt %s with %d tricks", all_path, len(tricks))


def ingest_writeup_from_url(
    url: str,
    persist: bool = True,
    llm: LLMClient | None = None,
) -> list[Trick]:
    """Fetch a writeup URL, extract tricks with the LLM, and persist them.

    This is a convenience wrapper around ingest_writeup_from_url_async for
    synchronous callers.  `persist` and `llm` are passed through unchanged.

    Raises:
        ValueError: if the URL is empty or the page has no usable content.
        httpx.HTTPError: on fetch/network failures.
    """
    import asyncio
    return asyncio.run(ingest_writeup_from_url_async(url, persist=persist, llm=llm))


async def ingest_writeup_from_url_async(
    url: str,
    persist: bool = True,
    llm: LLMClient | None = None,
) -> list[Trick]:
    """Async version of ingest_writeup_from_url (the LLM call is async).

    See ingest_writeup_from_url for the full contract.
    """
    if not url or not url.strip():
        raise ValueError("A non-empty URL is required.")

    logger.info("Fetching writeup from %s", url)
    raw = await _fetch_url_content_async(url)
    cleaned = _clean_and_check(raw, f"at {url}")
    title = _infer_title(url, cleaned)
    writeup = Writeup(
        source="url",
        url=url,
        challenge_title=title,
        challenge_name=title,
        challenge_source="url ingestion",
        content=raw,
        cleaned_content=cleaned,
    )

    parsed = await _extract_tricks_from_writeup(writeup, llm)
    if persist:
        for t in parsed:
            _append_trick_to_pipeline(t)

    logger.info("Extracted %d trick(s) from %s", len(parsed), url)
    return parsed


async def ingest_writeup_from_content_async(
    content: str,
    challenge_title: str = "pasted writeup",
    challenge_source: str = "manual",
    url: str = "",
    persist: bool = True,
    llm: LLMClient | None = None,
) -> list[Trick]:
    """Extract tricks from pasted writeup content and persist to the pipeline.

    Mirrors :func:`ingest_writeup_from_url_async` but takes the writeup text
    directly (markdown/HTML) instead of fetching a URL. An optional ``url`` is
    recorded on the extracted tricks as the source link. ``challenge_title`` /
    ``challenge_source`` let the caller label the writeup.

    Raises:
        ValueError: if content is empty or too short to be a useful writeup.
    """
    if not content or not content.strip():
        raise ValueError("Writeup content must not be empty.")

    cleaned = _clean_and_check(content, "pasted content")
    title = challenge_title.strip() or _infer_title(url, cleaned)
    writeup = Writeup(
        source="content",
        url=url,
        challenge_title=title,
        challenge_name=title,
        challenge_source=challenge_source.strip() or "manual",
        content=content,
        cleaned_content=cleaned,
    )

    parsed = await _extract_tricks_from_writeup(writeup, llm)
    if persist:
        for t in parsed:
            _append_trick_to_pipeline(t)

    logger.info("Extracted %d trick(s) from pasted writeup '%s'", len(parsed), title)
    return parsed


async def _fetch_url_content_async(url: str) -> str:
    """Async fetch identical to fetch_url_content (uses httpx.AsyncClient)."""
    import httpx

    headers = {
        "User-Agent": (
            "Mozilla/5.0 (X11; Linux x86_64) AppleWebKit/537.36 "
            "(KHTML, like Gecko) Chrome/124.0 Safari/537.36"
        ),
        "Accept-Language": "en-US,en;q=0.9",
    }
    async with httpx.AsyncClient(
        timeout=FETCH_TIMEOUT, follow_redirects=True, headers=headers
    ) as client:
        resp = await client.get(url)
    resp.raise_for_status()

    ctype = resp.headers.get("content-type", "")
    if "text/html" in ctype or "application/xhtml" in ctype:
        return resp.text
    return resp.text