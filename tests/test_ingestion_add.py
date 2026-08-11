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
async def test_process_writeup_job_persists_only(monkeypatch):
    """The job extracts tricks and persists them — no dedup/embed auto-run."""
    from src.api import routes

    extracted = [Trick(technique_name="xss", title="Stored XSS", description="d")]
    calls = {}

    async def fake_ingest(content="", **kwargs):
        calls["persist"] = kwargs.get("persist")
        return extracted

    monkeypatch.setattr(routes, "_url_ingestion_jobs", {})
    monkeypatch.setattr("src.ingestion.ingest_writeup_from_content_async", fake_ingest)

    job_id = "job1"
    await routes._process_writeup_job(job_id, content="some writeup content")
    job = routes._url_ingestion_jobs[job_id]
    assert job["status"] == "completed"
    assert job["total"] == 1
    assert calls["persist"] is True
    # No dedup/embed/BM25 fields — those are not auto-run.
    assert "dedup_wrote" not in job
    assert "embed_chunks" not in job
    assert "bm25_chunks" not in job


@pytest.mark.asyncio
async def test_reprocess_job_runs_full_pipeline(monkeypatch):
    """The reprocess job runs dedup → embed → BM25 and reports all stats."""
    from unittest.mock import AsyncMock, patch

    from src.api import routes

    monkeypatch.setattr(routes, "_reprocess_jobs", {})

    with patch("src.summarization.deduplicator.run_deduplication") as mock_dedup:
        mock_dedup.return_value = [Path("output/xss.md"), Path("output/sql.md")]
        with patch("src.embedding.engine.EmbeddingEngine") as MockEngine:
            instance = MockEngine.return_value
            instance.generate_all = AsyncMock(return_value={"chunks": 7})
            with patch("src.retrieval.bm25_index.BM25Index") as MockBM25:
                b = MockBM25.return_value
                b.build_from_output_dir.return_value = b
                b.chunks = [1, 2, 3, 4]

                await routes._run_reprocess_job("job_r", generate_questions=False)

    job = routes._reprocess_jobs["job_r"]
    assert job["status"] == "completed"
    assert job["dedup_wrote"] == 2
    assert job["embed_chunks"] == 7
    assert job["bm25_chunks"] == 4
    # questions stay off by default
    assert instance.generate_all.await_args.kwargs["generate_questions"] is False


@pytest.mark.asyncio
async def test_resummarize_job_replaces_tricks_and_runs_dedup(monkeypatch, tmp_path):
    """The resummarize job loads the writeup, extracts tricks, replaces old
    tricks in the pipeline, re-runs dedup, and evicts in-memory caches."""
    from src.api import routes

    monkeypatch.setattr(routes, "_resummarize_jobs", {})

    # 1. Fake the writeup lookup.
    fake_writeup = Writeup(
        url="https://example.com/old-writeup",
        challenge_title="Old Challenge",
        challenge_name="old-challenge",
        cleaned_content="Some writeup content about exploiting a vulnerability.",
    )
    monkeypatch.setattr(routes, "_load_writeup_by_url", lambda url: (fake_writeup, "test"))

    # 2. Capture the trick that extraction produces.
    class FakeLLM:
        async def create_message(self, **kwargs):
            return {"tricks": [{
                "technique_name": "sqli",
                "title": "Tautology login bypass",
                "category": "web",
                "description": "Inject OR 1=1.",
                "source_writeups": [],
            }]}

    fake_llm = FakeLLM()
    from src.ingestion import _extract_tricks_from_writeup

    async def fake_extract(writeup, llm):
        return await _extract_tricks_from_writeup(writeup, fake_llm)

    monkeypatch.setattr("src.ingestion._extract_tricks_from_writeup", fake_extract)

    # 3. Fake dedup to return a controlled value.
    from unittest.mock import patch

    with patch("src.summarization.deduplicator.run_deduplication") as mock_dedup:
        mock_dedup.return_value = [Path("output/sqli.md")]

        # 4. Run the resummarize job.
        await routes._run_resummarize_job("job_rs", "https://example.com/old-writeup")

    # 5. Verify the result.
    job = routes._resummarize_jobs["job_rs"]
    assert job["status"] == "completed"
    assert job["total"] == 1
    assert job["dedup_wrote"] == 1
    assert job["tricks"][0]["technique_name"] == "sqli"
    assert job["url"] == "https://example.com/old-writeup"


@pytest.mark.asyncio
async def test_resummarize_job_fails_on_missing_url(monkeypatch):
    """The resummarize job reports failure when the writeup URL is not found."""
    from src.api import routes

    monkeypatch.setattr(routes, "_resummarize_jobs", {})

    def raise_keyerror(url):
        msg = f"Writeup URL not found in the data: {url}"
        raise KeyError(msg)

    monkeypatch.setattr(routes, "_load_writeup_by_url", raise_keyerror)

    await routes._run_resummarize_job("job_missing", "https://example.com/nonexistent")

    job = routes._resummarize_jobs["job_missing"]
    assert job["status"] == "failed"
    assert "not found" in job["error"]


@pytest.mark.asyncio
async def test_remove_tricks_for_url_preserves_multi_source_tricks(monkeypatch, tmp_path):
    """_remove_tricks_for_url only removes tricks whose sole source is the URL;
    tricks with multiple sources keep the other sources after the URL is removed."""
    from src.api import routes
    from src.config import settings

    # Create a temporary tricks directory and file.
    tricks_dir = tmp_path / "tricks"
    tricks_dir.mkdir()
    monkeypatch.setattr("src.config.settings.tricks_dir", tricks_dir)

    jsonl_path = tricks_dir / "tricks.jsonl"
    with open(jsonl_path, "w") as f:
        # Trick 1: sole source = "https://example.com/target" → should be removed
        f.write('{"title":"T1","source_writeups":["https://example.com/target"]}\n')
        # Trick 2: multi-source including target → target removed, trick kept
        f.write('{"title":"T2","source_writeups":["https://example.com/target","https://example.com/other"]}\n')
        # Trick 3: no target → kept as-is
        f.write('{"title":"T3","source_writeups":["https://example.com/unrelated"]}\n')
        # Trick 4: empty source_writeups → kept
        f.write('{"title":"T4","source_writeups":[]}\n')
        # Trick 5: malformed → kept (but becomes a parse error that's skipped)
        f.write('not valid json\n')

    routes._remove_tricks_for_url("https://example.com/target")

    # Re-read the file and check what's left.
    with open(jsonl_path) as f:
        lines = [l.strip() for l in f if l.strip()]

    # T1 should be gone (only source was target).
    assert not any("T1" in l for l in lines), "T1 should have been removed"

    # T2 should remain with target removed from its source_writeups.
    t2_line = [l for l in lines if "T2" in l]
    assert len(t2_line) == 1
    import json
    t2 = json.loads(t2_line[0])
    assert "https://example.com/target" not in t2["source_writeups"]
    assert "https://example.com/other" in t2["source_writeups"]

    # T3 and T4 should be untouched.
    assert any("T3" in l for l in lines)
    assert any("T4" in l for l in lines)
