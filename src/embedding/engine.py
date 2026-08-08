"""Embedding generation pipeline.

Splits technique files into chunks, generates pre-retrieval questions,
embeds everything via OpenAI, and persists vectors + metadata to a Chroma vector
database (replacing the previous numpy ``.npz`` flat-file storage).
"""

import asyncio
import json
import logging
from pathlib import Path

import numpy as np
from openai import AsyncOpenAI

from src.config import settings
from src.embedding.checkpoint import EmbeddingCheckpointManager
from src.embedding.questions import QuestionGenerator
from src.embedding.splitter import MarkdownAwareTextSplitter
from src.models import Chunk
from src.retrieval.vector_store import VectorStore

logger = logging.getLogger(__name__)


class EmbeddingEngine:
    """Generates and stores embeddings for technique chunks."""

    def __init__(self, checkpoint_manager: EmbeddingCheckpointManager | None = None):
        if not settings.embedding_api_key:
            raise ValueError(
                "OpenAI API key is required for embeddings. "
                "Set OPENAI_API_KEY in .env or environment."
            )
        # EMBEDDING_BASE_URL lets you route through a gateway/proxy. Pass None
        # (the SDK default) when unset so an empty string never breaks the client.
        self.client = AsyncOpenAI(
            api_key=settings.embedding_api_key,
            base_url=settings.embedding_base_url or None,
            timeout=settings.embedding_timeout,
        )
        self.model = settings.embedding_model
        self.batch_size = settings.embedding_batch_size
        self.checkpoint_mgr = checkpoint_manager or EmbeddingCheckpointManager()
        self.splitter = MarkdownAwareTextSplitter()

    async def embed_texts(self, texts: list[str]) -> list[list[float]]:
        """Embed a batch of texts via the OpenAI API.

        Handles rate limits with simple retry on transient failures.
        """
        embeddings: list[list[float]] = []
        for i in range(0, len(texts), self.batch_size):
            batch = texts[i : i + self.batch_size]
            for attempt in range(3):
                try:
                    response = await self.client.embeddings.create(
                        model=self.model,
                        input=batch,
                        dimensions=settings.embedding_dimensions,
                    )
                    # Keep response order aligned with input order
                    batch_embeddings = [d.embedding for d in response.data]
                    embeddings.extend(batch_embeddings)
                    break
                except Exception as e:
                    if attempt == 2:
                        logger.error("Failed to embed batch after 3 attempts: %s", e)
                        raise
                    logger.warning("Embedding retry %d: %s", attempt + 1, e)
                    import asyncio
                    await asyncio.sleep(1.5 * (attempt + 1))
        return embeddings

    async def embed(self, text: str) -> list[float]:
        """Embed a single text."""
        result = await self.embed_texts([text])
        return result[0]

    def load_chunks(self) -> list[Chunk]:
        """Load chunks from the chunks directory (or split fresh)."""
        chunks_path = settings.chunks_dir / "chunks.json"
        if chunks_path.exists():
            try:
                with open(chunks_path, "r") as f:
                    data = json.load(f)
                chunks = [Chunk(**item) for item in data]
                logger.info("Loaded %d chunks from disk", len(chunks))
                return chunks
            except (json.JSONDecodeError, TypeError) as e:
                logger.warning("Failed to load chunks, resplitting: %s", e)

        # Split fresh from output dir
        output_dir = settings.output_dir
        chunks = []
        if output_dir.exists():
            for file_path in sorted(output_dir.glob("*.md")):
                file_chunks = self.splitter.split_file(file_path)
                chunks.extend(file_chunks)

        logger.info("Split %d chunks from technique files", len(chunks))
        return chunks

    def save_chunks(self, chunks: list[Chunk]) -> None:
        """Persist chunks to disk for reuse."""
        settings.chunks_dir.mkdir(parents=True, exist_ok=True)
        with open(settings.chunks_dir / "chunks.json", "w") as f:
            json.dump([c.model_dump() for c in chunks], f, indent=2, ensure_ascii=False)
        logger.info("Saved %d chunks to disk", len(chunks))

    async def generate_all(
        self, force_reset: bool = False, generate_questions: bool = False
    ) -> dict:
        """Run the embedding pipeline with per-chunk (per-trick) resume.

        Technique files are split into chunks; each chunk's content hash is
        tracked in the checkpoint.  On a subsequent run, only chunks whose
        content changed (or are new) are re-embedded — an unchanged trick is
        left untouched even if other tricks in the same file changed.

        Multiple files are processed concurrently (controlled by
        ``EMBEDDING_MAX_CONCURRENCY``) to saturate the API endpoints.

        Returns:
            dict with summary stats.
        """
        import hashlib

        if force_reset:
            self.checkpoint_mgr.reset()

        output_dir = Path(settings.output_dir)
        if not output_dir.exists() or not list(output_dir.glob("*.md")):
            logger.warning(
                "No technique files found in %s. Run `nekozuki summarize` first.",
                output_dir,
            )
            return {"status": "no_chunks", "chunks": 0}

        # Connect to Chroma (fresh collections on force_reset).
        chunks_col, questions_col = VectorStore.connect_collections(
            settings.chroma_dir, reset=force_reset
        )

        checkpoint = self.checkpoint_mgr.load()
        known_hashes = dict(checkpoint.chunk_hashes) if not force_reset else {}

        # ---- Split ALL files and compute per-chunk hashes ----
        technique_files = sorted(output_dir.glob("*.md"))
        # Group chunks by technique (file) for question generation co-location.
        chunks_by_technique: dict[str, list[Chunk]] = {}
        hashes_by_chunk: dict[str, str] = {}  # chunk_id → sha256 of content
        for file_path in technique_files:
            technique = file_path.stem
            chunks = self.splitter.split_file(file_path)
            if not chunks:
                continue
            chunks_by_technique[technique] = chunks
            for c in chunks:
                hashes_by_chunk[c.chunk_id] = hashlib.sha256(c.content.encode()).hexdigest()

        # ---- Determine new / changed / deleted chunks ----
        current_ids = set(hashes_by_chunk.keys())
        changed_ids = {
            cid for cid, h in hashes_by_chunk.items()
            if known_hashes.get(cid) != h
        }
        deleted_ids = set(known_hashes.keys()) - current_ids

        if not changed_ids and not deleted_ids:
            logger.info("No chunk changes detected — nothing to re-embed")
            return {
                "status": "completed",
                "chunks": len(current_ids),
                "questions": checkpoint.total_questions,
                "dimensions": settings.embedding_dimensions,
            }

        logger.info(
            "Chunk diff: %d changed, %d new, %d deleted",
            len(changed_ids & current_ids),
            len(changed_ids - (changed_ids & current_ids)),
            len(deleted_ids),
        )

        # Delete removed chunks from Chroma
        if deleted_ids:
            chunks_col.delete(ids=list(deleted_ids))
            logger.info("Deleted %d removed chunks from Chroma", len(deleted_ids))

        # Which techniques have any change? (for question regeneration)
        changed_techniques = {
            technique
            for technique, chunks in chunks_by_technique.items()
            if any(c.chunk_id in changed_ids for c in chunks)
        }

        question_gen = QuestionGenerator()

        from tqdm import tqdm

        target_techniques = [t for t in technique_files if t.stem in changed_techniques]
        bar = tqdm(
            total=len(target_techniques),
            desc="Embedding changed techniques",
            unit="file",
            bar_format="{l_bar}{bar}| {n_fmt}/{total_fmt} files "
                       "[{elapsed}<{remaining}] {postfix}",
        )

        lock = asyncio.Lock()
        sem = asyncio.Semaphore(settings.embedding_max_concurrency)
        _total_chunks = len(current_ids)
        _total_questions = checkpoint.total_questions
        _next_q_id = checkpoint.total_questions
        _embedded = 0

        async def _process_one(file_path: Path) -> None:
            """Embed the changed/new chunks of one technique file."""
            nonlocal _total_questions, _next_q_id, _embedded
            technique = file_path.stem
            chunks = chunks_by_technique[technique]

            # Only embed chunks that changed or are new
            to_embed = [c for c in chunks if c.chunk_id in changed_ids]
            if not to_embed:
                bar.update(1)
                return

            async with sem:
                content = file_path.read_text(encoding="utf-8")
                questions: list[str] = []
                if generate_questions and technique in changed_techniques:
                    questions = await question_gen.generate_for_file(content, technique)

                # Embed changed chunks + (regenerated) questions
                texts = [c.content for c in to_embed]
                q_indices = list(range(len(texts), len(texts) + len(questions)))
                texts.extend(questions)

                logger.debug(
                    "Embedding '%s': %d changed chunk(s)%s",
                    technique,
                    len(to_embed),
                    f" + {len(questions)} questions" if questions else "",
                )
                embeddings = np.asarray(await self.embed_texts(texts), dtype=np.float32)

                new_embeddings = [embeddings[i].tolist() for i in range(len(to_embed))]

                async with lock:
                    # Replace changed chunks (upsert by same chunk_id)
                    VectorStore.upsert_chunks(chunks_col, to_embed, new_embeddings)

                    # Update question metadata for this technique
                    if questions:
                        # Delete old questions for this technique.  Filter by
                        # technique_name via `where` instead of scanning the
                        # whole collection: an unfiltered get() binds one SQL
                        # variable per record and exceeds SQLite's variable
                        # limit on large collections ("too many SQL variables").
                        q_data = questions_col.get(
                            where={"technique_name": technique},
                            include=["metadatas"],
                        )
                        old_q_ids = q_data.get("ids", [])
                        if old_q_ids:
                            questions_col.delete(ids=old_q_ids)

                        # Add new questions with unique IDs
                        nonlocal _next_q_id
                        q_ids = [f"q_{_next_q_id + i}" for i in range(len(questions))]
                        _next_q_id += len(questions)
                        q_meta = [(technique, q) for q in questions]
                        q_embs = [embeddings[i].tolist() for i in q_indices]
                        questions_col.add(
                            ids=q_ids,
                            embeddings=q_embs,
                            documents=[q for _, q in q_meta],
                            metadatas=[
                                {"technique_name": technique, "question_text": q}
                                for _, q in q_meta
                            ],
                        )
                        _total_questions = _total_questions - len(old_q_ids) + len(questions)

                    # Update checkpoint hashes
                    for c in to_embed:
                        checkpoint.chunk_hashes[c.chunk_id] = hashlib.sha256(
                            c.content.encode()
                        ).hexdigest()
                    checkpoint.total_chunks = _total_chunks
                    checkpoint.total_questions = _total_questions
                    checkpoint.status = "running"
                    self.checkpoint_mgr.save(checkpoint)

                    _embedded += len(to_embed)
                    bar.update(1)
                    bar.set_postfix(technique=technique, chunks=len(to_embed))

        async with asyncio.TaskGroup() as tg:
            for fpath in target_techniques:
                tg.create_task(_process_one(fpath))

        bar.close()

        checkpoint.processed_files = [str(p) for p in technique_files]
        checkpoint.total_chunks = _total_chunks
        checkpoint.total_questions = _total_questions
        checkpoint.status = "completed"
        self.checkpoint_mgr.save(checkpoint)

        logger.info(
            "Embedding complete: %d chunks (%d embedded), %d questions, dim=%d",
            _total_chunks,
            _embedded,
            _total_questions,
            settings.embedding_dimensions,
        )

        return {
            "status": "completed",
            "chunks": _total_chunks,
            "questions": _total_questions,
            "dimensions": settings.embedding_dimensions,
        }


async def run_embedding_pipeline(
    force_reset: bool = False,
    generate_questions: bool = False,
    concurrency: int = 0,
    batch_size: int = 0,
) -> int:
    """CLI entry point for the embedding pipeline."""
    # Apply CLI overrides (0 = use ENV / default) before building the engine,
    # since __init__ reads batch_size.
    prev_concurrency = settings.embedding_max_concurrency
    prev_batch_size = settings.embedding_batch_size
    if concurrency > 0:
        settings.embedding_max_concurrency = concurrency
    if batch_size > 0:
        settings.embedding_batch_size = batch_size

    try:
        engine = EmbeddingEngine()
    except ValueError as e:
        settings.embedding_max_concurrency = prev_concurrency
        settings.embedding_batch_size = prev_batch_size
        logger.error("%s", e)
        return 1

    try:
        result = await engine.generate_all(
            force_reset=force_reset, generate_questions=generate_questions
        )
    finally:
        # Restore original values
        settings.embedding_max_concurrency = prev_concurrency
        settings.embedding_batch_size = prev_batch_size

    if result.get("status") == "completed":
        logger.info("=== EMBEDDING COMPLETE ===")
        return 0
    elif result.get("status") == "no_chunks":
        logger.error("No technique files found. Run `nekozuki summarize` first.")
        return 2
    return 1