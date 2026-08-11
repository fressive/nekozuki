"""Pydantic models for nekozuki data types."""

from datetime import UTC, datetime

from pydantic import BaseModel, Field, field_validator


class Writeup(BaseModel):
    """A single CTF writeup entry."""
    source: str = "ctftime"
    url: str = ""
    challenge_title: str = ""
    challenge_name: str = ""
    challenge_category: list[str] = Field(default_factory=list)
    challenge_source: str = ""
    content: str = ""
    cleaned_content: str = ""


class Trick(BaseModel):
    """A single extracted trick/technique from a writeup."""
    technique_name: str = ""
    title: str = ""
    category: str = ""
    description: str = ""
    conditions: list[str] = Field(default_factory=list)
    implementation_steps: list[str] = Field(default_factory=list)
    key_code: str | None = None
    example: str | None = None
    example_challenge: str | None = None
    detection_signs: list[str] = Field(default_factory=list)
    confidence: float = 0.0
    source_writeups: list[str] = Field(default_factory=list)
    original_terms: list[str] = Field(default_factory=list)
    created_at: str = Field(default_factory=lambda: datetime.now(UTC).isoformat())

    @field_validator("conditions", "implementation_steps", "detection_signs", mode="before")
    @classmethod
    def _coerce_none_to_empty(cls, v: object) -> object:
        """The LLM occasionally emits ``null`` for list fields; coerce to ``[]``."""
        if v is None:
            return []
        return v


class TrickBatch(BaseModel):
    """LLM response for a batch of writeups."""
    tricks: list[Trick] = Field(default_factory=list)


class TechniqueFile(BaseModel):
    """Represents a single technique markdown file."""
    technique_name: str = ""
    canonical_name: str = ""
    category: str = ""
    description: str = ""
    tricks: list[Trick] = Field(default_factory=list)


class Chunk(BaseModel):
    """A text chunk from a technique file for embedding."""
    chunk_id: str = ""
    technique_name: str = ""
    section_title: str = ""
    content: str = ""
    token_count: int = 0
    embedding: list[float] | None = None
    generated_questions: list[str] = Field(default_factory=list)


class SummarizationCheckpoint(BaseModel):
    """Checkpoint state for pause/resume."""
    phase: str = "summarize"
    batch_index: int = 0
    total_batches: int = 0
    processed_writeup_urls: list[str] = Field(default_factory=list)
    total_tricks_extracted: int = 0
    total_tokens_used: int = 0
    started_at: str = Field(default_factory=lambda: datetime.now(UTC).isoformat())
    updated_at: str = Field(default_factory=lambda: datetime.now(UTC).isoformat())
    status: str = "running"  # running | paused | completed | failed
    failed_batches: list[dict] = Field(default_factory=list)


class EmbeddingCheckpoint(BaseModel):
    """Checkpoint state for embedding."""
    phase: str = "embedding"
    processed_files: list[str] = Field(default_factory=list)
    chunk_hashes: dict[str, str] = Field(default_factory=dict)  # chunk_id → content_sha256
    total_chunks: int = 0
    total_questions: int = 0
    started_at: str = Field(default_factory=lambda: datetime.now(UTC).isoformat())
    updated_at: str = Field(default_factory=lambda: datetime.now(UTC).isoformat())
    status: str = "running"  # running | paused | completed | failed


class ProgressEvent(BaseModel):
    """Progress event emitted during processing."""
    batch_index: int = 0
    total_batches: int = 0
    tricks_extracted: int = 0
    tokens_used: int = 0
    progress_pct: float = 0.0
    status: str = "running"
    message: str = ""
    completed_count: int = 0  # number of successfully completed batches


class QueryRequest(BaseModel):
    """RAG query request."""
    query: str = ""
    top_k: int = 10


class QueryResult(BaseModel):
    """Single RAG query result."""
    chunk_id: str = ""
    technique_name: str = ""
    section_title: str = ""
    content: str = ""
    bm25_score: float = 0.0
    embedding_score: float = 0.0
    rrf_score: float = 0.0
    rerank_score: float = 0.0
    question_boost: float = 0.0
    rank: int = 0


class QueryResponse(BaseModel):
    """RAG query response."""
    results: list[QueryResult] = Field(default_factory=list)
    query: str = ""
    time_ms: float = 0.0


class AddWriteupRequest(BaseModel):
    """Body for POST /api/writeup/add: add a writeup via the pipeline.

    At least one of ``url`` or ``content`` must be provided. ``content`` is the
    writeup markdown/text pasted directly; ``url`` is fetched if given (and also
    recorded on the extracted tricks as the source link).
    """
    url: str = ""
    content: str = ""
    challenge_title: str = ""
    challenge_source: str = ""


class LoginRequest(BaseModel):
    """Body for POST /api/login."""
    password: str = ""


class ResummarizeRequest(BaseModel):
    """Body for POST /api/writeup/resummarize: re-extract tricks from a writeup.

    ``url`` is the writeup URL identifying which writeup to re-summarize.
    """
    url: str = ""