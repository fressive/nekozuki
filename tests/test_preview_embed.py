"""End-to-end tests: preview extraction of tricks → embed (preset questions + splitter).

These tests exercise the full writeup → trick → technique-file → chunk → question
→ embedding flow, mirroring what the `/api/summarize/upload` preview endpoint does
for the extraction half and the embedding pipeline for the storage half.  All LLM
and embedding calls are mocked.
"""

import hashlib
import sys
from pathlib import Path

import numpy as np
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.config import settings


# ---------------------------------------------------------------------------
# Mocks
# ---------------------------------------------------------------------------

class MockLLMClient:
    """Replacement for src.llm.LLMClient returning canned tricks (preview)."""

    async def create_message(self, system_prompt, user_message, cache_system=True):
        return [
            {
                "technique_name": "blind_sql_injection",
                "title": "Time-Based Blind SQL Injection",
                "category": "web",
                "description": "Use SLEEP() to exfiltrate data via timing when no data is returned.",
                "conditions": ["no data returned", "sql injection entrance"],
                "implementation_steps": ["Inject SLEEP(10)", "Measure response time"],
                "key_code": "SLEEP(10)",
                "example": "?id=1 OR SLEEP(10)",
                "detection_signs": ["timing differs on boolean conditions"],
                "confidence": 0.9,
                "source_indexes": [1],
            },
            {
                "technique_name": "union_sqli",
                "title": "Union Based SQL Injection",
                "category": "web",
                "description": "Use UNION SELECT to combine result sets and extract data.",
                "conditions": ["column count matches"],
                "implementation_steps": ["Find column count", "Inject UNION SELECT"],
                "key_code": "UNION SELECT 1,2,3",
                "example": "?id=0 UNION SELECT 1,2,3",
                "detection_signs": ["extra columns in output"],
                "confidence": 0.92,
                "source_indexes": [1],
            },
        ]


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
        return [f"How to exploit {name} with SLEEP()?"]


SAMPLE_WRITEUP = (
    "We exploited a time-based blind SQL injection in the login form. The query was "
    "SELECT * FROM users WHERE username='admin' AND password='$password' directly. "
    "Because the input was unsanitized, we injected a SLEEP(10) payload that caused "
    "a delay when the condition was true, letting us infer each byte of the flag. "
    "Once we knew the column count matched, we used a UNION SELECT to dump the "
    "flags table."
)


# ---------------------------------------------------------------------------
# Preview: extract tricks from a single writeup
# ---------------------------------------------------------------------------

@pytest.mark.asyncio
async def test_preview_extracts_tricks(monkeypatch):
    """The preview extraction produces structured tricks with the expected fields."""
    from src.models import Writeup
    from src.processing.batch import format_batch_for_prompt
    from src.processing.clean import clean_html_content
    from src.summarization.prompts import build_extraction_prompt

    writeup = Writeup(
        url="upload://preview",
        challenge_title="Test Challenge",
        challenge_name="test-challenge",
        challenge_source="user upload",
        content=SAMPLE_WRITEUP,
        cleaned_content=clean_html_content(SAMPLE_WRITEUP),
    )

    # Run the *same* logic the /api/summarize/upload preview endpoint uses.
    llm = MockLLMClient()
    writeup_text = format_batch_for_prompt([writeup])
    system_prompt, user_message = build_extraction_prompt(writeup_text)
    response = await llm.create_message(system_prompt, user_message, cache_system=True)

    if isinstance(response, list):
        tricks = response
    else:
        tricks = response.get("tricks", [])

    assert len(tricks) == 2
    assert tricks[0]["technique_name"] == "blind_sql_injection"
    assert tricks[0]["title"] == "Time-Based Blind SQL Injection"
    assert tricks[0]["category"] == "web"
    assert tricks[0]["confidence"] == 0.9
    # The prompt is built from the static, cacheable system prompt
    assert "technique extraction" in system_prompt.lower()


# ---------------------------------------------------------------------------
# End-to-end: extract → write technique file → split → questions → embed
# ---------------------------------------------------------------------------

def _write_tricks_into_technique_file(tmp_path: Path, tricks: list[dict]) -> Path:
    """Write tricks into a technique .md file in the trick format (H2 sections)."""
    out_dir = tmp_path / "output"
    out_dir.mkdir(parents=True, exist_ok=True)
    path = out_dir / "sql_injection.md"

    sections = []
    for trick in tricks:
        sections.append(
            f"## {trick['title'] or trick['technique_name']}\n\n"
            f"Description: {trick['description']}\n\n"
            f"Conditions: {'; '.join(trick.get('conditions', []))}\n\n"
            f"Implementation: {'; '.join(trick.get('implementation_steps', []))}\n\n"
            f"Example:\n```\n{trick.get('example', '')}\n```\n\n"
            f"Detection signs: {'; '.join(trick.get('detection_signs', []))}"
        )
    md = "---\ncategory: web\n---\n# SQL Injection\n\n" + "\n\n".join(sections)
    path.write_text(md)
    return path


def test_splitter_handles_trick_format(tmp_path):
    """The trick format's H2 sections and fenced code blocks are split correctly."""
    from src.embedding.splitter import MarkdownAwareTextSplitter

    tricks = [
        {"title": "Time-Based Blind SQL", "technique_name": "sql_injection",
         "description": "Use SLEEP() to time-extract data.",
         "conditions": ["no data returned"], "implementation_steps": ["Inject SLEEP(10)"],
         "example": "?id=1 OR SLEEP(10)", "detection_signs": ["timing differs"]},
        {"title": "Union Based SQL", "technique_name": "sql_injection",
         "description": "Use UNION SELECT to dump data.",
         "conditions": ["column count matches"], "implementation_steps": ["Find columns"],
         "example": "?id=0 UNION SELECT 1,2,3", "detection_signs": ["extra columns"]},
    ]
    md_path = _write_tricks_into_technique_file(tmp_path, tricks)

    splitter = MarkdownAwareTextSplitter()
    chunks = splitter.split_file(md_path)

    # One chunk per H2 section (the "overview" preamble is not a trick)
    section_titles = [c.section_title for c in chunks
                      if c.section_title != "overview"]
    assert "Time-Based Blind SQL" in section_titles
    assert "Union Based SQL" in section_titles

    # Find the code-block chunk and confirm it is intact (not split)
    for c in chunks:
        if c.section_title == "Time-Based Blind SQL":
            assert "SLEEP(10)" in c.content
        if c.section_title == "Union Based SQL":
            assert "UNION SELECT 1,2,3" in c.content


@pytest.mark.asyncio
async def test_preview_to_embed_end_to_end(tmp_path, monkeypatch):
    """Full flow: preview tricks → technique file → split → questions → embed.

    Verifies the Chroma store ends up with the right chunks AND the preset
    questions generated for those tricks.
    """
    monkeypatch.setattr(settings, "output_dir", tmp_path / "output")
    monkeypatch.setattr(settings, "vectors_dir", tmp_path / "vectors")
    monkeypatch.setattr(settings, "chroma_dir", tmp_path / "chroma")
    monkeypatch.setattr(settings, "chunks_dir", tmp_path / "chunks")
    monkeypatch.setattr(settings, "checkpoint_dir", tmp_path / "checkpoints")
    monkeypatch.setattr(settings, "embedding_api_key", "mock-key")

    # 1. Extract tricks (preview)
    from src.models import Writeup
    from src.processing.batch import format_batch_for_prompt
    from src.processing.clean import clean_html_content
    from src.summarization.prompts import build_extraction_prompt

    writeup = Writeup(
        url="upload://preview",
        challenge_title="Test Challenge",
        challenge_name="test-challenge",
        challenge_source="user upload",
        content=SAMPLE_WRITEUP,
        cleaned_content=clean_html_content(SAMPLE_WRITEUP),
    )
    llm = MockLLMClient()
    system_prompt, user_message = build_extraction_prompt(
        format_batch_for_prompt([writeup])
    )
    response = await llm.create_message(system_prompt, user_message, cache_system=True)
    tricks = response if isinstance(response, list) else response.get("tricks", [])
    assert len(tricks) == 2

    # 2. Write the technique file (this is what dedup would produce)
    _write_tricks_into_technique_file(tmp_path, tricks)

    # 3. Run the embedding pipeline with preset questions
    import src.embedding.engine as engine_module
    from src.embedding.engine import EmbeddingEngine

    engine_module.QuestionGenerator = MockQuestionGenerator
    engine_module.AsyncOpenAI = MockAsyncOpenAI

    engine = EmbeddingEngine()
    result = await engine.generate_all()
    assert result["status"] == "completed"
    assert result["chunks"] == 3        # 2 tricks + 1 overview = 3 H2 sections
    assert result["questions"] == 1     # 1 preset question per technique file

    # 4. Verify the Chroma store holds both chunks and preset questions
    from src.retrieval.vector_store import VectorStore

    vs = VectorStore(tmp_path / "chroma")
    assert vs.load()
    assert len(vs.chunks) == 3
    assert len(vs.question_meta) == 1
    assert vs.question_meta[0][0] == "sql_injection"
    assert "SLEEP()" in vs.question_meta[0][1]

    # 5. Chunk content is intact (code blocks preserved)
    contents = {c.section_title: c.content for c in vs.chunks}
    assert "Time-Based Blind SQL Injection" in contents
    assert "SLEEP(10)" in contents["Time-Based Blind SQL Injection"]
    assert "UNION SELECT 1,2,3" in contents["Union Based SQL Injection"]