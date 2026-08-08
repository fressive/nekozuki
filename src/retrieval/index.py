"""High-level index loading helper.

Ties together the BM25 index and vector store that were persisted during the
embedding phase, and exposes a single `load_or_build_index()` entry point.
"""

import logging
from functools import lru_cache

from src.retrieval.hybrid import HybridSearcher

logger = logging.getLogger(__name__)


@lru_cache(maxsize=1)
def load_or_build_index() -> HybridSearcher | None:
    """Load the persisted hybrid index, cached per process.

    The HybridSearcher (BM25 index + reranker + optional question boost) is
    read-only for the lifetime of the process; building it once and reusing it
    avoids re-reading every chunk from disk on each API request.  If you
    rebuild the index (``nekozuki embed`` / ``build-index``), restart the
    server so the cache picks up the new index.
    """
    searcher = HybridSearcher()
    if not searcher.loaded:
        logger.error(
            "Index not found. Run `nekozuki embed` to build the retrieval index."
        )
        return None
    return searcher