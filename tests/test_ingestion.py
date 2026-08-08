"""Tests for URL-based writeup ingestion (src/ingestion.py)."""

import asyncio
import json
import sys
from pathlib import Path

import pytest

# Ensure project src is importable when running pytest from project root
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.config import settings
from src.ingestion import _infer_title, ingest_writeup_from_url_async


class MockLLMClient:
    """Replacement for src.llm.LLMClient returning canned tricks."""

    def __init__(self, tricks):
        self.tricks = tricks

    async def create_message(self, system_prompt, user_message, cache_system=True):
        return self.tricks


SAMPLE_TRICKS = [
    {
        "technique_name": "blind_sql_injection",
        "title": "Time-Based Blind SQL Injection",
        "category": "web",
        "description": "Use SLEEP() to exfiltrate data via timing when no data is returned.",
        "conditions": ["no data returned", "sql injection entrance"],
        "implementation_steps": ["Inject SLEEP(10)", "Measure response time"],
        "key_code": "SLEEP(10)",
        "detection_signs": ["timing differs on boolean conditions"],
        "confidence": 0.9,
    }
]

HTML_BODY = (
    "<html><body><h1>Test Challenge</h1><p>We used a time-based blind SQL injection "
    "to extract the flag character by character. The SLEEP() function caused a delay "
    "when the condition was true, letting us infer each byte.</p></body></html>"
)


async def _mock_fetch(url: str) -> str:
    """Mock for _fetch_url_content_async — returns the sample HTML."""
    return HTML_BODY


async def _mock_fetch_empty(url: str) -> str:
    """Mock that returns empty content."""
    return ""


def test_infer_title_from_url_path():
    assert _infer_title("https://example.com/writeups/htb-forest", "") == "Htb Forest"
    assert _infer_title("https://example.com/page", "") == "Page"


def test_infer_title_from_heading():
    cleaned = "# My Awesome Writeup\n\nsome content"
    # Use a URL with no meaningful path segment so the heading fallback is used.
    assert _infer_title("https://example.com/", cleaned) == "My Awesome Writeup"


@pytest.mark.asyncio
async def test_ingest_persists_to_pipeline(tmp_path, monkeypatch):
    """Fetch + extract + persist should write tricks.jsonl & tricks_all.json."""
    monkeypatch.setattr(settings, "tricks_dir", tmp_path / "tricks")
    monkeypatch.setattr(
        "src.ingestion._fetch_url_content_async", _mock_fetch
    )

    tricks = await ingest_writeup_from_url_async(
        "https://example.com/writeup/2",
        persist=True,
        llm=MockLLMClient(SAMPLE_TRICKS),
    )

    assert len(tricks) == 1
    assert tricks[0].technique_name == "blind_sql_injection"
    # The source URL is recorded on the trick so the writeup↔trick index links back.
    assert "https://example.com/writeup/2" in tricks[0].source_writeups

    jsonl = tmp_path / "tricks" / "tricks.jsonl"
    assert jsonl.exists()
    lines = [json.loads(l) for l in jsonl.read_text().splitlines() if l.strip()]
    assert len(lines) == 1
    assert lines[0]["technique_name"] == "blind_sql_injection"

    all_path = tmp_path / "tricks" / "tricks_all.json"
    assert all_path.exists()
    all_tricks = json.loads(all_path.read_text())
    assert len(all_tricks) == 1


@pytest.mark.asyncio
async def test_ingest_no_persist_returns_only(tmp_path, monkeypatch):
    """persist=False should return tricks without writing pipeline files."""
    monkeypatch.setattr(settings, "tricks_dir", tmp_path / "tricks")
    monkeypatch.setattr(
        "src.ingestion._fetch_url_content_async", _mock_fetch
    )

    tricks = await ingest_writeup_from_url_async(
        "https://example.com/writeup/3",
        persist=False,
        llm=MockLLMClient(SAMPLE_TRICKS),
    )

    assert len(tricks) == 1
    assert not (tmp_path / "tricks" / "tricks.jsonl").exists()
    assert not (tmp_path / "tricks" / "tricks_all.json").exists()


@pytest.mark.asyncio
async def test_ingest_too_short_raises(tmp_path, monkeypatch):
    """A page with no readable content should raise ValueError."""
    monkeypatch.setattr("src.ingestion._fetch_url_content_async", _mock_fetch_empty)
    with pytest.raises(ValueError):
        await ingest_writeup_from_url_async(
            "https://example.com/empty", persist=True, llm=MockLLMClient(SAMPLE_TRICKS)
        )