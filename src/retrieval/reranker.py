"""Online cross-encoder reranker backed by an OpenAI-compatible rerank API.

BM25 (in ``bm25_index.py``) does coarse keyword recall over all chunks; this
reranker re-ranks the top candidates by sending the (query, documents) pair to
a remote ``/rerank`` endpoint.  This removes the local ``transformers`` /
``torch`` dependency — the model lives on the server.

The endpoint expects the OpenAI-style rerank request format:

    POST {base_url}/rerank
    {
      "model": "<rerank model>",
      "query": "<search query>",
      "documents": ["<doc 1>", "<doc 2>", ...],
      "top_n": <number>
    }

and returns ``results`` with ``index`` + ``relevance_score`` per document.
"""

import logging

import httpx

from src.config import settings
from src.models import Chunk

logger = logging.getLogger(__name__)

# Default rerank model name (override with RERANKER_MODEL).
DEFAULT_RERANKER_MODEL = "bge-reranker-v2-m3"


class CrossEncoderReranker:
    """Re-ranks a set of candidate chunks via an online rerank API.

    Parameters
    ----------
    base_url : str, optional
        Base URL of the OpenAI-compatible API.  Defaults to ``RERANK_BASE_URL``,
        then ``EMBEDDING_BASE_URL``, then the OpenAI default.
    api_key : str, optional
        API key.  Defaults to ``RERANK_API_KEY``, then ``OPENAI_API_KEY``.
    model : str, optional
        Rerank model name.  Defaults to ``RERANK_MODEL`` (``RERANKER_MODEL``
        also accepted for compatibility).
    top_n : int, optional
        Number of top results to return from the server.
    """

    def __init__(
        self,
        base_url: str | None = None,
        api_key: str | None = None,
        model: str | None = None,
        top_n: int | None = None,
    ):
        self.base_url = (base_url or settings.reranker_base_url
                         or settings.embedding_base_url
                         or "https://api.openai.com/v1").rstrip("/")
        self.api_key = api_key or settings.reranker_api_key or settings.embedding_api_key
        self.model = model or settings.reranker_model or DEFAULT_RERANKER_MODEL
        self.top_n = top_n or settings.reranker_top_n

        # The two field names some gateways use for the documents list.
        self._timeout = settings.embedding_timeout

    def rerank(
        self, query: str, candidates: list[tuple[int, Chunk]]
    ) -> list[tuple[int, float]]:
        """Score each candidate against the query via the API and re-sort.

        Parameters
        ----------
        query : str
            The user's search query.
        candidates : list of (chunk_index, Chunk)
            The coarse-recall candidates (e.g. BM25 top-100).

        Returns
        -------
        list of (chunk_index, relevance_score)
            Candidates sorted by relevance score descending.
            On API failure, returns candidates in their original order with a
            score of 0.0 so the BM25 ranking is preserved as a fallback.
        """
        if not candidates:
            return []
        if not self.api_key:
            logger.warning("No API key for reranker — returning BM25 order")
            return [(idx, 0.0) for idx, _ in candidates]

        documents = [c.content for _, c in candidates]
        payload = {
            "model": self.model,
            "query": query,
            "documents": documents,
            "top_n": self.top_n,
        }

        headers = {
            "Authorization": f"Bearer {self.api_key}",
            "Content-Type": "application/json",
        }

        try:
            with httpx.Client(timeout=self._timeout) as client:
                resp = client.post(
                    f"{self.base_url}/rerank", json=payload, headers=headers
                )
                resp.raise_for_status()
                data = resp.json()
        except (httpx.HTTPError, ValueError) as e:
            logger.warning("Reranker API failed (%s) — falling back to BM25 order", e)
            return [(idx, 0.0) for idx, _ in candidates]

        # Parse results: [{index, relevance_score}, ...] → score per candidate idx.
        results = data.get("results", [])
        if not results:
            logger.warning("Reranker returned no results — falling back to BM25 order")
            return [(idx, 0.0) for idx, _ in candidates]

        # The API returns ``index`` as the position in the documents array
        # (== the candidates list order), not the chunk's global index.
        # Map position → chunk index so scores key by the candidates we hold.
        score_by_candidate: dict[int, float] = {}
        for r in results:
            pos = r.get("index", -1)
            if 0 <= pos < len(candidates):
                chunk_idx = candidates[pos][0]
                score_by_candidate[chunk_idx] = float(r.get("relevance_score", 0.0))

        ranked = sorted(
            candidates, key=lambda ic: score_by_candidate.get(ic[0], 0.0), reverse=True
        )
        return [(idx, score_by_candidate.get(idx, 0.0)) for idx, _ in ranked]