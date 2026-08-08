"""Hybrid retrieval: BM25 coarse recall → cross-encoder re-rank → question boost.

    BM25 recall (top-100) → cross-encoder reranker → top-10
                                    ↘ per-technique question boost

The cross-encoder reads the full (query, chunk) text together, so it catches
semantic and CTF-specific term relationships that BM25 alone misses, without
requiring a separate dense embedding model or RRF tuning.  The optional
per-technique boost from the pre-generated questions collection nudges
techniques whose retrieval questions match the query (``HYBRID_QUESTION_WEIGHT``
controls the blend; 0 disables it).
"""

import logging
import time

from src.config import settings
from src.models import QueryResult
from src.retrieval.bm25_index import BM25Index
from src.retrieval.reranker import CrossEncoderReranker

logger = logging.getLogger(__name__)

# Seconds to wait before retrying the question boost after a failure.  The
# boost is an optional signal and the searcher is a process-level singleton, so
# a single transient failure must not disable it permanently — but we also must
# not hammer a broken Chroma/embedder on every query.
_BOOST_RETRY_COOLDOWN = 120


class HybridSearcher:
    """BM25 coarse recall → cross-encoder re-rank → optional question boost.

    BM25 provides fast, exact keyword recall over every chunk; the reranker
    scores only the top candidates, so the cost is bounded.  The per-technique
    question boost (if enabled) adds a small additive signal from the Chroma
    questions collection.
    """

    def __init__(self, bm25_index: BM25Index | None = None):
        self.bm25_index = bm25_index or BM25Index()
        self.reranker = CrossEncoderReranker()
        self._loaded = self.bm25_index.load()
        self.recall_k = settings.reranker_recall_k
        self.question_weight = settings.hybrid_question_weight

        # Optional per-technique question boost: ``final = rerank + weight *
        # question_boost``.  The boost comes from querying the Chroma questions
        # collection (pre-generated retrieval questions) for the technique that
        # best matches the query.  Disabled when the weight is 0, Chroma is
        # unavailable, or the questions collection is empty.
        self._vector_store = None
        self._embedder = None  # lazy EmbeddingEngine for query embedding
        # Monotonic time before which the question boost is skipped (see
        # _BOOST_RETRY_COOLDOWN).  Lets a failed boost be retried later instead
        # of being disabled for the rest of the process.
        self._boost_retry_at = 0.0
        if self.question_weight > 0:
            try:
                from src.retrieval.vector_store import VectorStore
                vs = VectorStore()
                if vs.load(questions_only=True) and len(vs.question_meta):
                    self._vector_store = vs
                    logger.info(
                        "Question boost enabled: %d questions, weight=%.2f",
                        len(vs.question_meta),
                        self.question_weight,
                    )
            except Exception as e:  # noqa: BLE001 (Chroma optional)
                logger.warning("Question boost unavailable: %s", e)
                # _vector_store stays None; _question_boosts retries after cooldown

        if self._loaded:
            logger.info(
                "BM25 + reranker ready: %d chunks, recall_k=%d",
                len(self.bm25_index.chunks),
                self.recall_k,
            )

    @property
    def loaded(self) -> bool:
        return self._loaded

    async def search(
        self, query: str, top_k: int | None = None
    ) -> list[QueryResult]:
        """Run BM25 + reranker (+ optional question-boost) search.

        Returns results sorted by ``rerank + weight * question_boost``
        descending.
        """
        start = time.monotonic()
        if top_k is None:
            top_k = settings.top_k

        n_chunks = len(self.bm25_index.chunks) if self._loaded else 0
        if n_chunks == 0:
            return []

        # ---- BM25 recall (tantivy top-k) ----
        candidates = self.bm25_index.search(query, top_k=self.recall_k)
        # candidates is list[(chunk_index, bm25_score)]
        # Build a score lookup for the final result display
        bm25_lookup = {idx: score for idx, score in candidates}
        if not candidates:
            elapsed = (time.monotonic() - start) * 1000
            logger.debug("BM25 returned no results in %.1f ms", elapsed)
            return []

        # ---- Cross-encoder re-rank ----
        chunk_pairs = [(i, self.bm25_index.chunks[i]) for i, _ in candidates]
        ranked = self.reranker.rerank(query, chunk_pairs)

        # ---- Optional per-technique question boost ----
        boosts: dict[str, float] = {}
        if self.question_weight > 0 and time.monotonic() >= self._boost_retry_at:
            boosts = await self._question_boosts(query)

        # Blend question boost into the rerank score and re-sort.
        scored: list[tuple[int, float, float, float]] = []
        for idx, rerank_score in ranked:
            boost = boosts.get(self.bm25_index.chunks[idx].technique_name, 0.0)
            final = rerank_score + self.question_weight * boost
            scored.append((idx, rerank_score, final, boost))
        scored.sort(key=lambda t: t[2], reverse=True)

        # Build results
        results = []
        for rank, (idx, rerank_score, _final, boost) in enumerate(scored[:top_k]):
            chunk = self.bm25_index.chunks[idx]
            results.append(QueryResult(
                chunk_id=chunk.chunk_id,
                technique_name=chunk.technique_name,
                section_title=chunk.section_title,
                content=chunk.content,
                bm25_score=float(bm25_lookup.get(idx, 0.0)),
                rerank_score=float(rerank_score),
                question_boost=float(boost),
                rrf_score=float(rerank_score),  # keep for backward compat
                rank=rank + 1,
            ))

        elapsed = (time.monotonic() - start) * 1000
        logger.debug(
            "Reranker search returned %d results in %.1f ms",
            len(results), elapsed,
        )
        return results

    async def _question_boosts(self, query: str) -> dict[str, float]:
        """Per-technique boost scores from the pre-generated questions.

        Embeds the query with the same model used to embed the questions, then
        queries the Chroma questions collection for the best-matching technique.
        On failure the boost is skipped and retried after a cooldown rather than
        disabled for the rest of the process.
        """
        try:
            if self._vector_store is None:
                from src.retrieval.vector_store import VectorStore
                vs = VectorStore()
                if not (vs.load(questions_only=True) and len(vs.question_meta)):
                    logger.warning("Question boost: no questions available")
                    self._boost_retry_at = time.monotonic() + _BOOST_RETRY_COOLDOWN
                    return {}
                self._vector_store = vs
            if self._embedder is None:
                from src.embedding.engine import EmbeddingEngine
                self._embedder = EmbeddingEngine()
            import numpy as np
            query_embedding = np.asarray(
                await self._embedder.embed(query), dtype=np.float32
            )
            return self._vector_store.question_technique_scores(query_embedding)
        except Exception as e:  # noqa: BLE001 (optional signal)
            logger.warning("Question boost failed (%s) — rerank order only", e)
            self._boost_retry_at = time.monotonic() + _BOOST_RETRY_COOLDOWN
            return {}


def load_or_build_index() -> HybridSearcher | None:
    """Load the BM25 index + reranker, or return None if not built."""
    searcher = HybridSearcher()
    if not searcher.loaded:
        logger.error(
            "BM25 index not found. Run `nekozuki build-index` first."
        )
        return None
    return searcher