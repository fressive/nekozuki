"""Central configuration for nekozuki."""

import os
from pathlib import Path

from dotenv import load_dotenv
from pydantic_settings import BaseSettings

load_dotenv()


class Settings(BaseSettings):
    # Paths
    data_path: Path = Path("data.json")
    output_dir: Path = Path("output")
    checkpoint_dir: Path = Path("data/checkpoints")
    tricks_dir: Path = Path("data/tricks")
    chunks_dir: Path = Path("data/chunks")
    vectors_dir: Path = Path("data/vectors")
    chroma_dir: Path = Path("data/chroma")
    technique_mapping_path: Path = Path("data/technique_mapping.yaml")
    cleaned_writeups_path: Path = Path("data/processed/writeups_clean.jsonl")

    # LLM (Anthropic)
    llm_api_key: str = os.getenv("ANTHROPIC_API_KEY", "")
    llm_base_url: str = os.getenv("BASE_URL", "")
    llm_model: str = os.getenv("LLM_MODEL", "claude-sonnet-4-20250514")
    llm_max_tokens: int = int(os.getenv("LLM_MAX_TOKENS", "4096"))
    llm_temperature: float = float(os.getenv("LLM_TEMPERATURE", "0.3"))
    llm_max_concurrency: int = int(os.getenv("LLM_MAX_CONCURRENCY", "8"))
    llm_batch_size: int = int(os.getenv("LLM_BATCH_SIZE", "20"))
    llm_max_batch_tokens: int = int(os.getenv("LLM_MAX_BATCH_TOKENS", "25000"))
    llm_timeout: float = float(os.getenv("LLM_TIMEOUT", "120"))
    llm_max_retries: int = int(os.getenv("LLM_MAX_RETRIES", "3"))

    # Embedding (OpenAI)
    embedding_api_key: str = os.getenv("OPENAI_API_KEY", "")
    embedding_base_url: str = os.getenv("EMBEDDING_BASE_URL", "")
    embedding_model: str = os.getenv("EMBEDDING_MODEL", "text-embedding-3-small")
    embedding_dimensions: int = int(os.getenv("EMBEDDING_DIMENSIONS", "512"))
    embedding_batch_size: int = int(os.getenv("EMBEDDING_BATCH_SIZE", "45"))
    embedding_timeout: float = float(os.getenv("EMBEDDING_TIMEOUT", "120"))
    embedding_max_concurrency: int = int(os.getenv("EMBEDDING_MAX_CONCURRENCY", "8"))

    # Chunking
    chunk_size: int = int(os.getenv("CHUNK_SIZE", "500"))
    chunk_overlap: int = int(os.getenv("CHUNK_OVERLAP", "50"))
    # Max chars of a technique file passed to the question generator.  Multi-trick
    # files (e.g. encoding.md, 38 tricks) have their later tricks cut off at
    # small limits, so their retrieval questions never cover the deep tricks.
    question_source_char_limit: int = int(os.getenv("QUESTION_SOURCE_CHAR_LIMIT", "32000"))

    # Retrieval
    hybrid_bm25_weight: float = float(os.getenv("HYBRID_BM25_WEIGHT", "0.3"))
    hybrid_embedding_weight: float = float(os.getenv("HYBRID_EMBEDDING_WEIGHT", "0.7"))
    top_k: int = int(os.getenv("TOP_K", "10"))
    # Per-technique boost from the pre-generated questions collection, blended
    # additively into the rerank score: final = rerank + weight * question_boost.
    # Set to 0 to disable (BM25 + rerank only).
    hybrid_question_weight: float = float(os.getenv("HYBRID_QUESTION_WEIGHT", "0.3"))

    # Online cross-encoder reranker (BM25 recall → rerank → top_k)
    reranker_model: str = os.getenv("RERANK_MODEL") or os.getenv("RERANKER_MODEL") or "bge-reranker-v2-m3"
    reranker_base_url: str = os.getenv("RERANK_BASE_URL", "")
    reranker_api_key: str = os.getenv("RERANK_API_KEY", "")
    reranker_recall_k: int = int(os.getenv("RERANKER_RECALL_K", "100"))
    # How many scored results the rerank API returns. Default 100 = score every
    # BM25 candidate; lower values make the API cheaper but leave the tail
    # unscored (0.0 fallback).
    reranker_top_n: int = int(os.getenv("RERANKER_TOP_N", "100"))

    # Filtering
    min_content_length: int = int(os.getenv("MIN_CONTENT_LENGTH", "500"))

    # Technique-level dedup (`nekozuki dedup-techniques`)
    # Two canonical techniques merge on content when they share this many
    # identical trick-title token sets, or on name when fuzzy ratio >= the
    # name threshold AND word-Dice >= 0.7 / token containment.
    technique_name_sim_threshold: int = int(os.getenv("TECHNIQUE_NAME_SIM_THRESHOLD", "85"))
    technique_content_shared_titles: int = int(os.getenv("TECHNIQUE_CONTENT_SHARED_TITLES", "2"))
    # Min title-token Jaccard for a name-similar pair to be LLM-judged.
    technique_llm_content_jaccard: float = float(os.getenv("TECHNIQUE_LLM_CONTENT_JACCARD", "0.1"))

    model_config = {"arbitrary_types_allowed": True}


settings = Settings()