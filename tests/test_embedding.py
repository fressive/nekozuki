"""Tests for the embedding pipeline: chunking, question generation, resume."""

import hashlib
import json
import sys
from pathlib import Path

import numpy as np
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.config import settings


class MockAsyncOpenAI:
    """Deterministic hash-based embeddings (no network calls)."""

    def __init__(self, **kwargs):
        self.embeddings = self.Embeddings()

    class Embeddings:
        async def create(self, model, input, dimensions):
            class R:
                pass

            r = R()
            r.data = []
            for text in input:
                digest = hashlib.sha256(text.encode()).digest()
                rng = np.random.default_rng(int.from_bytes(digest[:8], "big"))
                r.data.append(type("D", (), {"embedding": rng.normal(size=dimensions).tolist()})())
            return r


class MockQuestionGenerator:
    """Returns one canned question per technique (no LLM call)."""

    async def generate_for_file(self, content, name):
        return [f"How to exploit {name}?"]


def _write_technique_files(tmp_path: Path, names: list[str]) -> Path:
    """Create a few technique .md files in tmp_path/output."""
    out_dir = tmp_path / "output"
    out_dir.mkdir(parents=True, exist_ok=True)
    templates = {
        "sql_injection": "---\ncategory: web\n---\n# SQL Injection\n\n## Blind SQLi\n\nDescription: use SLEEP to extract data.\n\nExample:\n```\nSLEEP(10)\n```\n",
        "xss": "---\ncategory: web\n---\n# XSS\n\n## Stored XSS\n\nDescription: store a malicious script.\n\n```\n<img src=x onerror=alert(1)>\n```\n",
        "ssti": "---\ncategory: web\n---\n# SSTI\n\n## Jinja2 RCE\n\nDescription: inject template expressions.\n",
    }
    for name in names:
        (out_dir / f"{name}.md").write_text(templates[name])
    return out_dir


@pytest.mark.asyncio
async def test_embedding_pipeline_resume(tmp_path, monkeypatch):
    """Embedding run 1 + new technique + run 2 should accumulate, not wipe."""
    monkeypatch.setattr(settings, "output_dir", tmp_path / "output")
    monkeypatch.setattr(settings, "vectors_dir", tmp_path / "vectors")
    monkeypatch.setattr(settings, "chroma_dir", tmp_path / "chroma")
    monkeypatch.setattr(settings, "chunks_dir", tmp_path / "chunks")
    monkeypatch.setattr(settings, "checkpoint_dir", tmp_path / "checkpoints")
    monkeypatch.setattr(settings, "embedding_api_key", "mock-key")

    _write_technique_files(tmp_path, ["sql_injection", "xss"])

    import src.embedding.engine as engine_module
    from src.embedding.engine import EmbeddingEngine

    # Swap in mocks (the engine's __init__ will construct MockAsyncOpenAI)
    engine_module.QuestionGenerator = MockQuestionGenerator
    engine_module.AsyncOpenAI = MockAsyncOpenAI

    engine = EmbeddingEngine()

    result1 = await engine.generate_all()
    assert result1["status"] == "completed"
    assert result1["chunks"] == 4  # 2 files, 2 sections each

    # Add a new technique and resume
    _write_technique_files(tmp_path, ["ssti"])
    result2 = await engine.generate_all()
    assert result2["status"] == "completed"
    assert result2["chunks"] == 6  # accumulated: 4 + 2

    # Vector store should have ALL chunks, not just the new ones
    from src.retrieval.vector_store import VectorStore

    vs = VectorStore(tmp_path / "chroma")
    assert vs.load()
    assert len(vs.chunks) == 6, f"Expected 6 chunks, got {len(vs.chunks)}"
    assert len(vs.question_meta) == 3, f"Expected 3 questions, got {len(vs.question_meta)}"
    assert sorted({c.technique_name for c in vs.chunks}) == ["sql_injection", "ssti", "xss"]


def test_vector_store_question_scores(tmp_path):
    """Question-based technique boosting works with Chroma."""
    from src.embedding.splitter import MarkdownAwareTextSplitter
    from src.retrieval.vector_store import VectorStore

    chunks = MarkdownAwareTextSplitter().split_technique_file(
        "## Trick A\n\ndescription text about sql injection.",
        "sql_injection",
    )

    # Build Chroma collections in a temp dir
    chroma_dir = tmp_path / "chroma"
    chunks_col, questions_col = VectorStore.connect_collections(chroma_dir, reset=True)

    # Insert chunks with deterministic embeddings
    rng = np.random.RandomState(0)
    chunk_embs = [rng.normal(size=8).tolist() for _ in chunks]
    VectorStore.upsert_chunks(chunks_col, chunks, chunk_embs)

    # Insert questions
    q_meta = [
        ("sql_injection", "How to use sql injection?"),
        ("sql_injection", "When is UNION useful?"),
    ]
    rng2 = np.random.RandomState(1)
    q_embs = [rng2.normal(size=8).tolist() for _ in q_meta]
    VectorStore.upsert_questions(questions_col, q_meta, q_embs, start_index=0)

    # Load the same Chroma dir via VectorStore
    vs = VectorStore(chroma_dir)
    assert vs.load()
    scores = vs.question_technique_scores(
        np.random.RandomState(2).normal(size=8).astype(np.float32)
    )
    assert "sql_injection" in scores
    assert isinstance(scores["sql_injection"], float)