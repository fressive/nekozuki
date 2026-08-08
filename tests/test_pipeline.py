"""Integration test for the full summarize pipeline using a mocked LLM."""

import json
import sys
from pathlib import Path

import pytest

# Ensure project src is importable when running pytest from project root
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.config import settings
from src.summarization.checkpoint import CheckpointManager
from src.summarization.deduplicator import run_deduplication
from src.summarization.extractor import TrickExtractor


class MockLLMClient:
    """Replacement for src.llm.LLMClient that returns canned tricks."""

    async def create_message(self, system_prompt, user_message, cache_system=True):
        return [
            {
                "technique_name": "blind_sql_injection",
                "title": "Time-Based Blind SQL Injection",
                "category": "web",
                "description": "Use SLEEP() to create delays and extract data when no data is returned.",
                "conditions": ["no data returned", "sql injection entrance"],
                "implementation_steps": ["Inject SLEEP(10)", "Measure response time"],
                "key_code": "SLEEP(10)",
                "example": "?id=1 OR SLEEP(10)",
                "detection_signs": ["timing differs on boolean conditions"],
                "confidence": 0.9,
            },
            {
                "technique_name": "time_based_sqli",
                "title": "Blind SQLi via Time Delay",
                "category": "web",
                "description": "Utilize SLEEP/BENCHMARK functions to infer data through time cost.",
                "conditions": ["no data returning"],
                "implementation_steps": ["Inject SLEEP(10)", "Compare timing"],
                "key_code": "SLEEP(10)",
                "example": "?id=0 OR SLEEP(5)",
                "detection_signs": ["response latency"],
                "confidence": 0.85,
            },
            {
                "technique_name": "union_sqli",
                "title": "Union Based SQL Injection",
                "category": "web",
                "description": "Use UNION SELECT to combine result sets and extract data.",
                "conditions": ["column count matches"],
                "implementation_steps": ["Find column count", "Inject UNION SELECT"],
                "key_code": "UNION SELECT 1,2,3",
                "detection_signs": ["extra columns in output"],
                "confidence": 0.92,
            },
        ]


def _write_sample_data(tmp_path: Path) -> Path:
    """Create a small sample data.json with a few writeups."""
    long_body = (
        "<p>We exploited a SQL injection in the login form. The server executed the query "
        "SELECT * FROM users WHERE username='admin' AND password='$password' directly. "
        "Because the input was unsanitized, we injected a time-based blind SQL injection "
        "payload to extract the flag character by character. We used the SLEEP() function "
        "so that a true condition causes a 10 second delay, letting us infer each byte of "
        "the flag through timing. We also had to account for the WAF blocking the equals "
        "sign, so we used BETWEEN and less-than/greater-than comparisons instead. Finally "
        "we used a UNION SELECT to dump the contents of the flags table once we knew the "
        "number of columns matched.</p>"
    )
    writeups = [
        {
            "source": "ctftime",
            "url": f"https://ctftime.org/writeup/{i}",
            "challenge_title": f"Test Challenge {i}",
            "challenge_name": f"test-challenge-{i}",
            "challenge_category": ["web"],
            "challenge_source": "TestCTF 2024",
            "content": f"<html><body><h1>Test {i}</h1>{long_body}</body></html>",
        }
        for i in range(3)
    ]
    data_path = tmp_path / "data.json"
    data_path.write_text(json.dumps(writeups))
    return data_path


@pytest.mark.asyncio
async def test_full_summarize_pipeline(tmp_path, monkeypatch):
    """Test extract → checkpoint → dedup → write output files."""
    data_path = _write_sample_data(tmp_path)

    # Point settings at the temp dir
    monkeypatch.setattr(settings, "data_path", data_path)
    monkeypatch.setattr(settings, "output_dir", tmp_path / "output")
    monkeypatch.setattr(settings, "tricks_dir", tmp_path / "tricks")
    monkeypatch.setattr(settings, "checkpoint_dir", tmp_path / "checkpoints")
    monkeypatch.setattr(settings, "technique_mapping_path", tmp_path / "technique_mapping.yaml")
    monkeypatch.setattr(settings, "cleaned_writeups_path", tmp_path / "processed" / "writeups_clean.jsonl")

    # Patch the LLM client
    monkeypatch.setattr("src.summarization.extractor.LLMClient", MockLLMClient)

    checkpoint_mgr = CheckpointManager(tmp_path / "checkpoints" / "test.json")
    extractor = TrickExtractor(checkpoint_mgr)
    extractor.llm = MockLLMClient()  # replace the client directly

    events = []
    async for event in extractor.extract_all(batch_limit=2):
        events.append(event)

    # Extraction should complete
    assert events[-1].status == "completed"
    assert events[-1].tricks_extracted > 0

    # Checkpoint should be written
    assert checkpoint_mgr.checkpoint_path.exists()
    checkpoint = checkpoint_mgr.load()
    assert checkpoint.status == "completed"

    # Tricks file should exist
    tricks_file = tmp_path / "tricks" / "tricks.jsonl"
    assert tricks_file.exists()

    # Run deduplication and write output
    written = run_deduplication(tricks_path=tricks_file, output_dir=tmp_path / "output")

    # Should produce a sql_injection.md file
    sql_file = tmp_path / "output" / "sql_injection.md"
    assert sql_file.exists(), f"Expected sql_injection.md, got {written}"

    content = sql_file.read_text()
    # Dedup should merge the two SLEEP tricks into one
    assert content.count("## ") == 2, f"Expected 2 tricks after dedup, content:\n{content}"
    assert "Time-Based Blind SQL Injection" in content
    assert "Union Based SQL Injection" in content
    # Frontmatter present
    assert content.startswith("---\n")
    assert "category: web" in content


@pytest.mark.asyncio
async def test_save_all_tricks_merges_existing_accumulator(tmp_path, monkeypatch):
    """_save_all_tricks rebuilds from tricks.jsonl so resumed runs don't drop
    tricks extracted before a pause."""
    tricks_dir = tmp_path / "tricks"
    tricks_dir.mkdir()
    monkeypatch.setattr(settings, "tricks_dir", tricks_dir)
    monkeypatch.setattr(settings, "checkpoint_dir", tmp_path / "checkpoints")

    # Simulate a previous session that appended tricks to the accumulator
    old_tricks = [
        {"technique_name": "sql_injection", "title": "Old trick 1",
         "source_writeups": ["https://ctftime.org/writeup/1"]},
        {"technique_name": "xss", "title": "Old trick 2",
         "source_writeups": ["https://ctftime.org/writeup/2"]},
    ]
    with open(tricks_dir / "tricks.jsonl", "w") as f:
        f.writelines(json.dumps(t) + "\n" for t in old_tricks)

    extractor = TrickExtractor()
    # The resumed session appends its new tricks to the JSONL accumulator...
    new_tricks = [
        {"technique_name": "sql_injection", "title": "New trick",
         "source_writeups": ["https://ctftime.org/writeup/3"]},
    ]
    extractor._save_tricks_batch(0, new_tricks, 0)
    # ...and its in-memory accumulator only holds this session's tricks.
    # _save_all_tricks must rebuild from the full JSONL, not just this list.
    extractor._save_all_tricks(new_tricks)

    saved = json.loads((tricks_dir / "tricks_all.json").read_text())
    titles = {t["title"] for t in saved}
    assert len(saved) == 3, f"Expected old + new merged, got {saved}"
    assert "Old trick 1" in titles
    assert "Old trick 2" in titles
    assert "New trick" in titles


def test_missing_writeups_command(tmp_path, monkeypatch, capsys):
    """missing-writeups reports and optionally saves the URL list."""
    from argparse import Namespace

    from src.main import run_missing_writeups
    from src.models import Writeup

    fake = [
        Writeup(url="https://ctftime.org/writeup/1", challenge_title="Alpha"),
        Writeup(url="https://ctftime.org/writeup/2", challenge_title="Beta"),
    ]
    monkeypatch.setattr("src.main._load_missing_writeups", lambda: fake)

    save_path = tmp_path / "missing.txt"
    rc = run_missing_writeups(Namespace(sample=1, save=str(save_path)))
    assert rc == 0
    out = capsys.readouterr().out
    assert "2 writeups have no tricks yet" in out
    assert save_path.read_text().splitlines() == [
        "https://ctftime.org/writeup/1",
        "https://ctftime.org/writeup/2",
    ]


def test_load_or_build_index_is_cached(monkeypatch):
    """HybridSearcher is built once per process, not per query."""
    from src.retrieval import index as index_mod

    index_mod.load_or_build_index.cache_clear()
    created = []

    class _Fake:
        loaded = True

    def _fake_builder():
        created.append(1)
        return _Fake()

    monkeypatch.setattr(index_mod, "HybridSearcher", _fake_builder)
    a = index_mod.load_or_build_index()
    b = index_mod.load_or_build_index()
    assert a is b
    assert len(created) == 1
    index_mod.load_or_build_index.cache_clear()


@pytest.mark.asyncio
async def test_question_boost_retries_after_failure(monkeypatch):
    """A failed question boost is retried after cooldown, not disabled forever."""
    import time
    from types import SimpleNamespace

    from src.retrieval.hybrid import HybridSearcher

    # Skip the Chroma load in __init__ (weight 0); we wire up the boost manually.
    monkeypatch.setattr(settings, "hybrid_question_weight", 0)
    searcher = HybridSearcher()

    class FakeEmbedder:
        async def embed(self, q):
            return [0.1] * settings.embedding_dimensions

    searcher._embedder = FakeEmbedder()
    ok = {"sql_injection": 1.0}
    state = {"fail": True}

    def score(qe):
        if state["fail"]:
            raise RuntimeError("chroma down")
        return ok

    searcher._vector_store = SimpleNamespace(question_technique_scores=score)
    searcher._boost_retry_at = 0.0

    # First call fails -> cooldown armed, this call returns no boost
    assert await searcher._question_boosts("q") == {}
    assert searcher._boost_retry_at > time.monotonic()

    # After the failure clears, the boost works again (not permanently disabled)
    state["fail"] = False
    searcher._boost_retry_at = 0.0
    assert await searcher._question_boosts("q") == ok


def test_normalizer_mapping(tmp_path, monkeypatch):
    """Test that variant technique names map to canonical files."""
    from src.summarization.normalizer import TechniqueNormalizer

    # Isolate from the real mapping file so unknown names aren't persisted
    monkeypatch.setattr(settings, "technique_mapping_path", tmp_path / "mapping.yaml")

    n = TechniqueNormalizer()
    assert n.normalize("blind_sql_injection") == "sql_injection"
    assert n.normalize("time_based_sqli") == "sql_injection"
    assert n.normalize("union_select") == "sql_injection"
    assert n.normalize("return_oriented_programming") == "buffer_overflow"
    assert n.normalize("ssti") == "server_side_template_injection"


def test_normalizer_verbose_names(tmp_path, monkeypatch):
    """Long chatty technique names embed a known keyword -> canonical file."""
    from src.summarization.normalizer import TechniqueNormalizer

    monkeypatch.setattr(settings, "technique_mapping_path", tmp_path / "mapping.yaml")

    n = TechniqueNormalizer()
    assert n.normalize(
        "Time-based blind SQL injection with SLEEP() for character-by-character extraction"
    ) == "sql_injection"
    assert n.normalize(
        "Exploiting a format string vulnerability to leak the stack and overwrite GOT"
    ) == "format_string"


def test_extractor_source_indexes_mapping():
    """source_indexes from LLM are correctly mapped to writeup URLs."""
    from src.models import Writeup

    writeups = [
        Writeup(url="https://ctftime.org/writeup/1"),
        Writeup(url="https://ctftime.org/writeup/2"),
        Writeup(url="https://ctftime.org/writeup/3"),
    ]

    fake_tricks = [
        {"technique_name": "sql_injection", "title": "T1", "confidence": 0.9,
         "source_indexes": [1, 3]},
        {"technique_name": "xss", "title": "T2", "confidence": 0.8,
         "source_indexes": [2]},
        {"technique_name": "rce", "title": "T3", "confidence": 0.7,
         "source_indexes": None},  # no source_indexes → fallback to all
        {"technique_name": "ssrf", "title": "T4", "confidence": 0.7,
         "source_indexes": [99]},  # out of range → skip
    ]

    # Simulate what _process_batch does with source_indexes
    for trick in fake_tricks:
        idxs = trick.pop("source_indexes", None)
        if isinstance(idxs, list) and idxs:
            urls = []
            for i in idxs:
                try:
                    w = writeups[int(i) - 1]
                    urls.append(w.url)
                except (ValueError, IndexError):
                    continue
            trick["source_writeups"] = urls or []
        else:
            trick["source_writeups"] = [w.url for w in writeups]

    assert fake_tricks[0]["source_writeups"] == ["https://ctftime.org/writeup/1", "https://ctftime.org/writeup/3"]
    assert fake_tricks[1]["source_writeups"] == ["https://ctftime.org/writeup/2"]
    assert fake_tricks[2]["source_writeups"] == ["https://ctftime.org/writeup/1", "https://ctftime.org/writeup/2", "https://ctftime.org/writeup/3"]
    assert fake_tricks[3]["source_writeups"] == []  # out-of-range → empty


def test_splitter_never_splits_code_blocks(tmp_path):
    """Test that the splitter keeps code blocks intact."""
    from src.embedding.splitter import MarkdownAwareTextSplitter

    markdown = """---
category: pwn
---
# Test

## First Trick

Description text.

```
## fake heading inside code block
SELECT * FROM users WHERE id=1
```

## Second Trick

More text.
"""
    splitter = MarkdownAwareTextSplitter()
    chunks = splitter.split_technique_file(markdown, "test_technique")

    # Should be 2 sections (first + second), plus overview
    sections = [c.section_title for c in chunks]
    assert "First Trick" in sections
    assert "Second Trick" in sections
    assert "fake heading inside code block" not in sections

    # The code block should be intact in the First Trick chunk
    first = next(c for c in chunks if c.section_title == "First Trick")
    assert "SELECT * FROM users" in first.content


class FlakyMockLLMClient:
    """Fails the first call per key, then succeeds, to test retries."""

    def __init__(self, always_fail: bool = False):
        self.call_count = 0
        self.always_fail = always_fail

    async def create_message(self, system_prompt, user_message, cache_system=True):
        self.call_count += 1
        if self.always_fail or self.call_count == 1:
            raise RuntimeError("simulated gateway failure")
        return [
            {
                "technique_name": "sql_injection",
                "title": "Union Based SQL Injection",
                "category": "web",
                "description": "Use UNION SELECT to combine result sets.",
                "conditions": ["column count matches"],
                "implementation_steps": ["Find column count", "Inject UNION SELECT"],
                "key_code": "UNION SELECT 1,2,3",
                "confidence": 0.9,
            }
        ]


def _setup(tmp_path, monkeypatch):
    """Shared setup: temp dirs + small sample data."""
    data_path = _write_sample_data(tmp_path)
    monkeypatch.setattr(settings, "data_path", data_path)
    monkeypatch.setattr(settings, "output_dir", tmp_path / "output")
    monkeypatch.setattr(settings, "tricks_dir", tmp_path / "tricks")
    monkeypatch.setattr(settings, "checkpoint_dir", tmp_path / "checkpoints")
    monkeypatch.setattr(settings, "technique_mapping_path", tmp_path / "technique_mapping.yaml")
    monkeypatch.setattr(settings, "cleaned_writeups_path", tmp_path / "processed" / "writeups_clean.jsonl")
    monkeypatch.setattr("src.summarization.extractor.LLMClient", FlakyMockLLMClient)
    return CheckpointManager(tmp_path / "checkpoints" / "test.json")


@pytest.mark.asyncio
async def test_transient_failure_is_retried(tmp_path, monkeypatch):
    """A batch that fails once is retried and the run still completes."""
    checkpoint_mgr = _setup(tmp_path, monkeypatch)
    monkeypatch.setattr(settings, "llm_max_retries", 3)

    extractor = TrickExtractor(checkpoint_mgr)
    extractor.llm = FlakyMockLLMClient(always_fail=False)

    events = []
    async for event in extractor.extract_all(batch_limit=2):
        events.append(event)

    # After retry, the batch succeeds -> run completes
    assert events[-1].status == "completed"
    # The flaky mock was called more than once (initial attempt + retry)
    assert extractor.llm.call_count > 1, f"Expected a retry, got {extractor.llm.call_count} calls"
    checkpoint = checkpoint_mgr.load()
    assert checkpoint.status == "completed"
    assert checkpoint.failed_batches == []  # no permanent failures


@pytest.mark.asyncio
async def test_permanent_failure_is_not_success(tmp_path, monkeypatch):
    """A batch that always fails is NOT marked as a successful completion."""
    checkpoint_mgr = _setup(tmp_path, monkeypatch)
    monkeypatch.setattr(settings, "llm_max_retries", 2)

    extractor = TrickExtractor(checkpoint_mgr)
    extractor.llm = FlakyMockLLMClient(always_fail=True)

    events = []
    async for event in extractor.extract_all(batch_limit=2):
        events.append(event)

    # Final status must NOT be "completed" -> it's "failed"
    assert events[-1].status == "failed", f"Expected failed, got {events[-1].status}"
    checkpoint = checkpoint_mgr.load()
    assert checkpoint.status == "failed"
    # Permanent failures recorded in the checkpoint
    assert len(checkpoint.failed_batches) > 0
    # The failed batches contributed 0 tricks to the accumulator
    assert checkpoint.total_tricks_extracted == 0