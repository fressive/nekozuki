"""Tests for adding a writeup to the pipeline (ingestion + /api/writeup/add)."""

import sys
from pathlib import Path

import pytest

# Ensure project src is importable when running pytest from project root
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.ingestion import (
    _extract_tricks_from_writeup,
    ingest_writeup_from_content_async,
)
from src.models import Trick, Writeup


class FakeLLM:
    """LLM stand-in returning canned tricks from create_message."""

    def __init__(self, tricks: list[dict]):
        self.tricks = tricks

    async def create_message(self, system_prompt, user_message, cache_system=True):
        return {"tricks": self.tricks}


@pytest.mark.asyncio
async def test_extract_tricks_records_source_url():
    llm = FakeLLM([{
        "technique_name": "sql_injection",
        "title": "Tautology login bypass",
        "description": "Inject OR 1=1.",
    }])
    writeup = Writeup(
        url="https://example.com/w",
        challenge_title="Challenge",
        challenge_name="Challenge",
        cleaned_content="Some writeup content.",
    )
    tricks = await _extract_tricks_from_writeup(writeup, llm)
    assert len(tricks) == 1
    assert tricks[0].technique_name == "sql_injection"
    assert "https://example.com/w" in tricks[0].source_writeups


@pytest.mark.asyncio
async def test_ingest_writeup_from_content(monkeypatch):
    from src import ingestion

    captured = {}

    async def fake_extract(writeup, llm):
        captured["writeup"] = writeup
        t = Trick(
            technique_name="ssrf",
            title="Fetch internal metadata",
            description="A test trick.",
            source_writeups=[],
        )
        return [t]

    monkeypatch.setattr(ingestion, "_extract_tricks_from_writeup", fake_extract)
    tricks = await ingest_writeup_from_content_async(
        content="# My Writeup\n\nWe hit the internal metadata endpoint on the "
                "SSRF target and leaked the cloud provider's instance profile "
                "credentials, which gave us access to the storage bucket.",
        challenge_title="My Challenge",
        url="https://example.com/w",
        persist=False,
    )
    assert len(tricks) == 1
    assert tricks[0].technique_name == "ssrf"
    assert captured["writeup"].challenge_title == "My Challenge"
    assert captured["writeup"].source == "content"


@pytest.mark.asyncio
async def test_ingest_writeup_from_content_rejects_empty(monkeypatch):
    from src.ingestion import ingest_writeup_from_content_async

    with pytest.raises(ValueError):
        await ingest_writeup_from_content_async(content="   ", persist=False)


@pytest.mark.asyncio
async def test_process_writeup_job_completes_with_dedup(monkeypatch):
    """The job extracts tricks, persists, re-dedups, embeds, and rebuilds BM25."""
    from unittest.mock import AsyncMock, patch

    from src.api import routes

    extracted = [Trick(technique_name="xss", title="Stored XSS", description="d")]
    calls = {}

    async def fake_ingest(content="", **kwargs):
        calls["persist"] = kwargs.get("persist")
        return extracted

    def fake_dedup():
        return [Path("output/xss.md")]

    monkeypatch.setattr(routes, "_url_ingestion_jobs", {})
    monkeypatch.setattr("src.ingestion.ingest_writeup_from_content_async", fake_ingest)
    monkeypatch.setattr("src.summarization.deduplicator.run_deduplication", fake_dedup)
    monkeypatch.setattr("src.retrieval.index.load_or_build_index.cache_clear", lambda: None)

    with patch("src.embedding.engine.EmbeddingEngine") as MockEngine:
        instance = MockEngine.return_value
        instance.generate_all = AsyncMock(return_value={"chunks": 5, "status": "completed"})

        with patch("src.retrieval.bm25_index.BM25Index") as MockBM25:
            bm25_instance = MockBM25.return_value
            bm25_instance.build_from_output_dir.return_value = bm25_instance
            bm25_instance.chunks = [1, 2, 3]

            job_id = "job1"
            await routes._process_writeup_job(job_id, content="some writeup content")

    job = routes._url_ingestion_jobs[job_id]
    assert job["status"] == "completed"
    assert job["total"] == 1
    assert job["dedup_wrote"] == 1
    assert job["embed_chunks"] == 5
    assert job["bm25_chunks"] == 3
    assert calls["persist"] is True
