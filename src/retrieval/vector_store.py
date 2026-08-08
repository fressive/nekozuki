"""Vector store for chunk embeddings, backed by ChromaDB.

Replaces the previous numpy + JSON flat-file storage with a Chroma persistent
client.  Two collections are maintained:

  - ``nekozuki_chunks``     — one document per technique chunk.
  - ``nekozuki_questions``  — one document per pre-generated retrieval question.

Both use cosine distance (``hnsw:space=cosine``).  Embeddings are always
provided explicitly — Chroma never auto-embeds text.
"""

import logging
from pathlib import Path

import numpy as np

from src.config import settings
from src.models import Chunk

try:
    import chromadb
    from chromadb import EmbeddingFunction
except ImportError:
    chromadb = None  # type: ignore[assignment]
    EmbeddingFunction = object

logger = logging.getLogger(__name__)

_CHUNKS_COLLECTION = "nekozuki_chunks"
_QUESTIONS_COLLECTION = "nekozuki_questions"


class _NoOpEmbeddingFunction(EmbeddingFunction):
    """Pass-through embedding function — we always supply vectors explicitly."""

    def __init__(self):
        super().__init__()

    def __call__(self, input):
        # Chroma only calls this when auto-embedding, which we never do.
        # Return a dummy so collection creation doesn't fail.
        return np.zeros((len(input), 1), dtype=np.float32)

    @staticmethod
    def name() -> str:
        return "no_op"


class VectorStore:
    """Chroma-backed vector store for chunk and question embeddings.

    Parameters
    ----------
    vectors_dir : str or Path, optional
        Ignored for Chroma (we use ``chroma_dir`` from settings).  The
        parameter is kept for backward compatibility with callers that pass
        a path.
    """

    def __init__(self, vectors_dir: str | Path | None = None):
        self._chroma_path = Path(vectors_dir) if vectors_dir else Path(settings.chroma_dir)
        self._client: chromadb.PersistentClient | None = None
        self._chunks_collection = None
        self._questions_collection = None
        self._chunks: list[Chunk] = []
        self._chunk_id_map: dict[str, int] = {}  # chunk_id → position in self._chunks
        self._chunk_embeddings: np.ndarray = np.empty((0, 0), dtype=np.float32)
        self._question_embeddings: np.ndarray = np.empty((0, 0), dtype=np.float32)
        self._question_meta: list[tuple[str, str]] = []
        self._loaded = False

    # ---- public API (matches the previous interface) ----

    @property
    def loaded(self) -> bool:
        return self._loaded

    @property
    def chunks(self) -> list[Chunk]:
        return self._chunks

    @property
    def question_meta(self) -> list[tuple[str, str]]:
        return self._question_meta

    def load(self, questions_only: bool = False) -> bool:
        """Connect to Chroma and load collection metadata / embeddings.

        Parameters
        ----------
        questions_only : bool
            When True, only connect to the questions collection and load its
            metadata — skipping the (much heavier) reconstruction of all chunk
            embeddings.  This is the light path used by HybridSearcher for the
            per-technique question boost, where chunk vectors are not needed.
        """
        if chromadb is None:
            logger.error("chromadb is not installed — run `pip install chromadb`")
            return False

        try:
            self._client = chromadb.PersistentClient(path=str(self._chroma_path))
            if questions_only:
                return self._load_questions_only()
            try:
                self._chunks_collection = self._client.get_collection(
                    _CHUNKS_COLLECTION,
                    embedding_function=_NoOpEmbeddingFunction(),
                )
            except ValueError:
                logger.info("Chroma collection '%s' not found", _CHUNKS_COLLECTION)
                return False

            self._questions_collection = self._client.get_or_create_collection(
                _QUESTIONS_COLLECTION,
                embedding_function=_NoOpEmbeddingFunction(),
                metadata={"hnsw:space": "cosine"},
            )

            # Reconstruct chunks and embeddings from Chroma
            self._reconstruct_from_chroma()
            self._loaded = True
            logger.info(
                "Chroma store ready: %d chunks, %d questions",
                len(self._chunks),
                len(self._question_meta),
            )
            return True
        except Exception as e:
            logger.error("Failed to load Chroma vector store: %s", e)
            return False

    def _load_questions_only(self) -> bool:
        """Lightweight load of just the questions collection (no chunk data)."""
        try:
            self._questions_collection = self._client.get_collection(
                _QUESTIONS_COLLECTION,
                embedding_function=_NoOpEmbeddingFunction(),
            )
        except ValueError:
            logger.info("Chroma questions collection '%s' not found", _QUESTIONS_COLLECTION)
            return False

        q_data = self._get_all(self._questions_collection, include=["metadatas"])
        self._question_meta = [
            (m.get("technique_name", ""), m.get("question_text", ""))
            for m in q_data.get("metadatas", [])
        ]
        self._loaded = True
        logger.info(
            "Chroma questions-only ready: %d questions",
            len(self._question_meta),
        )
        return True

    def search(
        self, query_embedding: np.ndarray, top_k: int = 10
    ) -> list[tuple[int, float]]:
        """ANN search via Chroma, returning ``(chunk_index, cosine_similarity)``.

        Parameters
        ----------
        query_embedding : np.ndarray
            Query vector (1D, shape ``(dim,)``).
        top_k : int
            Number of nearest neighbours to return.

        Returns
        -------
        list of (int, float)
            Each tuple is ``(index into ``self.chunks``, cosine similarity)``,
            sorted by similarity descending.
        """
        if not self._loaded or self._chunks_collection is None:
            return []

        results = self._chunks_collection.query(
            query_embeddings=[query_embedding.tolist()],
            n_results=min(top_k, len(self._chunks) or 1),
            include=["distances", "metadatas"],
        )
        if not results["ids"] or not results["ids"][0]:
            return []

        pairs: list[tuple[int, float]] = []
        for dist, meta in zip(results["distances"][0], results["metadatas"][0]):
            cid = meta.get("chunk_id", "")
            idx = self._chunk_id_map.get(cid, -1)
            if idx < 0:
                continue
            # Cosine distance -> cosine similarity: sim = 1 - dist
            sim = 1.0 - dist
            pairs.append((idx, sim))
        return pairs

    def search_scores(self, query_embedding: np.ndarray) -> np.ndarray:
        """Exhaustive cosine similarity scores for *all* chunks.

        This is a fallback for HybridSearcher's RRF fusion.  Returns a vector
        of length ``len(self.chunks)`` where entry ``i`` is the cosine
        similarity between the query and chunk ``i``.
        """
        if not self._loaded or self._chunk_embeddings.size == 0:
            return np.zeros(0)

        query_vec = np.asarray(query_embedding, dtype=np.float32).reshape(1, -1)
        norms = np.linalg.norm(query_vec)
        if norms > 0:
            query_vec = query_vec / norms
        return (self._chunk_embeddings @ query_vec.T).ravel()

    def question_technique_scores(
        self, query_embedding: np.ndarray
    ) -> dict[str, float]:
        """Per-technique boost scores from pre-retrieval questions.

        Queries the Chroma questions collection and aggregates the top
        question scores by technique name.
        """
        if not self._loaded or self._questions_collection is None:
            return {}

        n = len(self._question_meta)
        if n == 0:
            return {}

        results = self._questions_collection.query(
            query_embeddings=[query_embedding.tolist()],
            n_results=min(n, 50),
            include=["distances", "metadatas"],
        )
        if not results["ids"] or not results["ids"][0]:
            return {}

        technique_scores: dict[str, float] = {}
        for dist, meta in zip(results["distances"][0], results["metadatas"][0]):
            sim = 1.0 - dist
            technique = meta.get("technique_name", "")
            if technique and sim > 0:
                technique_scores[technique] = max(
                    technique_scores.get(technique, 0.0), sim
                )
        return technique_scores

    # ---- Chroma helpers ----

    @staticmethod
    def _get_all(
        collection: "chromadb.Collection", include: list[str]
    ) -> dict[str, list]:
        """Fetch every record from a collection using limit/offset paging.

        A single unfiltered ``get()`` binds one SQL variable per record, which
        exceeds SQLite's variable limit (default 32,766) on large collections
        and fails with "too many SQL variables".  Paging keeps each query well
        under the cap.
        """
        page_size = 1000
        merged: dict[str, list] = {"ids": []}
        for key in ("embeddings", "metadatas", "documents"):
            if key in include:
                merged[key] = []
        offset = 0
        while True:
            page = collection.get(limit=page_size, offset=offset, include=include)
            ids = page["ids"]
            if not ids:
                break
            merged["ids"].extend(ids)
            for key in merged:
                if key != "ids":
                    merged[key].extend(page[key])
            offset += len(ids)
            if len(ids) < page_size:
                break
        return merged

    def _reconstruct_from_chroma(self) -> None:
        """Reload chunks, embeddings, and question metadata from Chroma."""
        # Load all chunk data
        all_data = self._get_all(
            self._chunks_collection, include=["embeddings", "metadatas"]
        )
        chunk_ids = all_data["ids"]
        embs = all_data["embeddings"]
        metas = all_data["metadatas"]

        # Rebuild chunks list preserving insertion order
        self._chunks = []
        self._chunk_id_map = {}
        chunk_emb_list: list[np.ndarray] = []
        for cid, emb, meta in zip(chunk_ids, embs, metas):
            chunk = Chunk(
                chunk_id=cid,
                technique_name=meta.get("technique_name", ""),
                section_title=meta.get("section_title", ""),
                content=meta.get("content", ""),
                token_count=meta.get("token_count", 0),
            )
            self._chunk_id_map[cid] = len(self._chunks)
            self._chunks.append(chunk)
            chunk_emb_list.append(np.asarray(emb, dtype=np.float32))

        if chunk_emb_list:
            self._chunk_embeddings = np.stack(chunk_emb_list)
        else:
            self._chunk_embeddings = np.empty((0, 0), dtype=np.float32)

        # Load question data
        try:
            q_data = self._get_all(
                self._questions_collection, include=["embeddings", "metadatas"]
            )
            q_embs = q_data["embeddings"]
            q_metas = q_data["metadatas"]
            self._question_meta = [
                (m.get("technique_name", ""), m.get("question_text", ""))
                for m in q_metas
            ]
            if q_embs is not None and len(q_embs) > 0:
                self._question_embeddings = np.asarray(
                    list(q_embs) if isinstance(q_embs, np.ndarray) else q_embs,
                    dtype=np.float32,
                )
            else:
                self._question_embeddings = np.empty((0, 0), dtype=np.float32)
        except ValueError:
            self._question_meta = []
            self._question_embeddings = np.empty((0, 0), dtype=np.float32)

    # ---- used by EmbeddingEngine during build ----

    @classmethod
    def connect_collections(
        cls, chroma_path: str | Path, reset: bool = False
    ) -> tuple["chromadb.Collection", "chromadb.Collection"]:
        """Connect to Chroma, optionally resetting (delete + recreate) collections.

        Returns a ``(chunks_collection, questions_collection)`` tuple.
        """
        client = chromadb.PersistentClient(path=str(chroma_path))
        if reset:
            for name in (_CHUNKS_COLLECTION, _QUESTIONS_COLLECTION):
                try:
                    client.delete_collection(name)
                except (ValueError, chromadb.errors.NotFoundError):
                    pass
        chunks_col = client.get_or_create_collection(
            _CHUNKS_COLLECTION,
            embedding_function=_NoOpEmbeddingFunction(),
            metadata={"hnsw:space": "cosine"},
        )
        questions_col = client.get_or_create_collection(
            _QUESTIONS_COLLECTION,
            embedding_function=_NoOpEmbeddingFunction(),
            metadata={"hnsw:space": "cosine"},
        )
        return chunks_col, questions_col

    @classmethod
    def load_chunks_from(
        cls, collection: "chromadb.Collection"
    ) -> tuple[set[str], list[Chunk]]:
        """Load existing chunk IDs and ``Chunk`` objects from a Chroma collection.

        Returns ``(chunk_ids_set, chunks_in_insertion_order)``.
        """
        data = cls._get_all(collection, include=["metadatas"])
        if not data["ids"]:
            return set(), []

        chunks = []
        for cid, meta in zip(data["ids"], data["metadatas"]):
            chunk = Chunk(
                chunk_id=cid,
                technique_name=meta.get("technique_name", ""),
                section_title=meta.get("section_title", ""),
                content=meta.get("content", ""),
                token_count=meta.get("token_count", 0),
            )
            chunks.append(chunk)

        chunk_ids = {c.chunk_id for c in chunks}
        return chunk_ids, chunks

    @classmethod
    def load_questions_from(
        cls, collection: "chromadb.Collection"
    ) -> list[tuple[str, str]]:
        """Load ``(technique_name, question_text)`` pairs from the questions collection."""
        try:
            data = cls._get_all(collection, include=["metadatas"])
        except ValueError:
            return []
        if not data["metadatas"]:
            return []
        return [
            (m.get("technique_name", ""), m.get("question_text", ""))
            for m in data["metadatas"]
        ]

    @staticmethod
    def upsert_chunks(
        collection: "chromadb.Collection",
        chunks: list[Chunk],
        embeddings: list[list[float]],
    ) -> None:
        """Upsert a batch of chunks into the Chroma collection.

        Parameters
        ----------
        collection :
            The Chroma chunks collection.
        chunks :
            Chunk objects to insert.
        embeddings :
            Embedding vectors for each chunk (same order).
        """
        ids = [c.chunk_id for c in chunks]
        documents = [c.content for c in chunks]
        metadatas = [
            {
                "chunk_id": c.chunk_id,
                "technique_name": c.technique_name,
                "section_title": c.section_title,
                "token_count": c.token_count,
                "content": c.content,
            }
            for c in chunks
        ]
        # Delete first so we can always use add() — avoids Chroma bugs where
        # upsert() can fail on IDs that already exist in the same batch.
        try:
            collection.delete(ids=ids)
        except Exception:
            pass  # IDs may not exist yet (first run, or new chunks)
        collection.add(
            ids=ids,
            embeddings=embeddings,
            documents=documents,
            metadatas=metadatas,
        )

    @staticmethod
    def upsert_questions(
        collection: "chromadb.Collection",
        question_meta: list[tuple[str, str]],
        embeddings: list[list[float]],
        start_index: int = 0,
    ) -> None:
        """Upsert question embeddings into the Chroma questions collection.

        Parameters
        ----------
        collection :
            The Chroma questions collection.
        question_meta :
            List of ``(technique_name, question_text)`` pairs.
        embeddings :
            Embedding vectors for each question (same order).
        start_index :
            Global offset for the question IDs so batches don't collide.
        """
        ids = [f"q_{start_index + i}" for i in range(len(question_meta))]
        documents = [q for _, q in question_meta]
        metadatas = [
            {"technique_name": t, "question_text": q}
            for t, q in question_meta
        ]
        collection.add(
            ids=ids,
            embeddings=embeddings,
            documents=documents,
            metadatas=metadatas,
        )

    @staticmethod
    def count_chunks() -> int:
        """Return the number of chunks in the Chroma collection, or 0."""
        try:
            client = chromadb.PersistentClient(path=str(settings.chroma_dir))
            col = client.get_collection(
                _CHUNKS_COLLECTION, embedding_function=_NoOpEmbeddingFunction()
            )
            return col.count()
        except (ValueError, Exception):
            return 0