"""BM25 index backed by tantivy (Rust), supporting incremental updates.

Replaces the previous ``rank_bm25`` + pickle approach with a persistent on-disk
tantivy index.  New chunks can be added incrementally without rebuilding the
entire index.
"""

import logging
import tempfile
from pathlib import Path

import tantivy

from src.config import settings
from src.models import Chunk

logger = logging.getLogger(__name__)

# Field name constants
_FIELD_INDEX = "idx"  # integer position (0-based) in the global chunks list
_FIELD_CONTENT = "content"
_FIELD_CHUNK_ID = "chunk_id"
_FIELD_TECHNIQUE = "technique"
_FIELD_SECTION = "section_title"
_FIELD_TOKENS = "token_count"

# Tokenizer name for the content field
_CONTENT_TOKENIZER = "fts"


def _build_schema() -> tantivy.Schema:
    """Return the tantivy schema for the BM25 index."""
    b = tantivy.SchemaBuilder()
    b.add_integer_field(_FIELD_INDEX, stored=True)
    b.add_text_field(_FIELD_CONTENT, stored=True, tokenizer_name=_CONTENT_TOKENIZER)
    b.add_text_field(_FIELD_CHUNK_ID, stored=True)
    b.add_text_field(_FIELD_TECHNIQUE, stored=True)
    b.add_text_field(_FIELD_SECTION, stored=True)
    b.add_integer_field(_FIELD_TOKENS, stored=True)
    return b.build()


def _register_tokenizer(index: tantivy.Index) -> None:
    """Register the simple tokenizer for the content field."""
    analyzer = tantivy.TextAnalyzerBuilder(tantivy.Tokenizer.simple()).build()
    index.register_tokenizer(_CONTENT_TOKENIZER, analyzer)


def _chunk_to_doc(chunk: Chunk, idx: int) -> tantivy.Document:
    """Convert a Chunk + index to a tantivy Document."""
    return tantivy.Document(
        idx=[idx],
        content=[chunk.content],
        chunk_id=[chunk.chunk_id],
        technique=[chunk.technique_name],
        section_title=[chunk.section_title],
        token_count=[chunk.token_count],
    )


def _doc_to_chunk(doc: tantivy.Document, idx: int) -> Chunk:
    """Reconstruct a Chunk from a tantivy Document (used on load)."""
    return Chunk(
        chunk_id=doc.get_first(_FIELD_CHUNK_ID) or "",
        technique_name=doc.get_first(_FIELD_TECHNIQUE) or "",
        section_title=doc.get_first(_FIELD_SECTION) or "",
        content=doc.get_first(_FIELD_CONTENT) or "",
        token_count=doc.get_first(_FIELD_TOKENS) or 0,
    )


class BM25Index:
    """Persistent BM25 index backed by tantivy.

    Parameters
    ----------
    index_path : str or Path, optional
        Directory for the tantivy index.  Defaults to ``data/vectors/tantivy/``.
    """

    def __init__(self, index_path: str | Path | None = None):
        if index_path is None:
            index_path = settings.vectors_dir / "tantivy"
        self.index_path = Path(index_path)
        self._schema = _build_schema()
        self._index: tantivy.Index | None = None
        self.chunks: list[Chunk] = []

    def build(self, chunks: list[Chunk]) -> "BM25Index":
        """Build the index from a list of chunks (replaces any existing index)."""
        # Remove existing index if present
        if self.index_path.exists():
            import shutil
            shutil.rmtree(self.index_path)
        self.index_path.mkdir(parents=True, exist_ok=True)

        self._index = tantivy.Index(self._schema, path=str(self.index_path))
        _register_tokenizer(self._index)

        writer = self._index.writer()
        for i, chunk in enumerate(chunks):
            writer.add_document(_chunk_to_doc(chunk, i))
        writer.commit()
        # Release the writer lock
        del writer
        self._index.reload()

        self.chunks = list(chunks)
        logger.info(
            "Built tantivy index with %d chunks -> %s",
            len(chunks), self.index_path,
        )
        return self

    def add_chunks(self, chunks: list[Chunk]) -> None:
        """Incrementally add chunks to an existing index."""
        if self._index is None:
            self.load()
        if self._index is None:
            logger.warning("No index to add to — call build() first")
            return

        writer = self._index.writer()
        start = len(self.chunks)
        for i, chunk in enumerate(chunks, start=start):
            writer.add_document(_chunk_to_doc(chunk, i))
        writer.commit()
        del writer
        self._index.reload()

        self.chunks.extend(chunks)
        logger.info("Added %d chunks -> total %d", len(chunks), len(self.chunks))

    def load(self) -> bool:
        """Load an existing tantivy index from disk."""
        if not self.index_path.exists():
            logger.debug("tantivy index not found at %s", self.index_path)
            return False

        try:
            self._index = tantivy.Index(self._schema, path=str(self.index_path))
            _register_tokenizer(self._index)
            searcher = self._index.searcher()

            # Reconstruct chunks list from the stored field values
            indexed: list[tuple[int, Chunk]] = []
            # Use a dummy query to get all documents
            # We iterate by reading all docs from the index
            dummy_query = self._index.parse_query("*", [_FIELD_CONTENT])
            hits = searcher.search(dummy_query, searcher.num_docs).hits
            for _score, addr in hits:
                doc = searcher.doc(addr)
                idx = doc.get_first(_FIELD_INDEX)
                if idx is not None:
                    indexed.append((int(idx), _doc_to_chunk(doc, int(idx))))

            indexed.sort(key=lambda x: x[0])
            self.chunks = [c for _, c in indexed]

            logger.info(
                "Loaded tantivy index: %d chunks from %s",
                len(self.chunks), self.index_path,
            )
            return True
        except Exception as e:
            logger.warning("Failed to load tantivy index: %s", e)
            self._index = None
            self.chunks = []
            return False

    def search(self, query: str, top_k: int = 10) -> list[tuple[int, float]]:
        """Return ``(chunk_index, bm25_score)`` pairs for the query.

        Uses tantivy's BM25 scoring.
        """
        if self._index is None:
            return []
        query_str = query.strip()
        if not query_str:
            return []

        searcher = self._index.searcher()
        try:
            q = self._index.parse_query(query_str, [_FIELD_CONTENT])
        except Exception as e:
            logger.debug("Query parse error: %s", e)
            return []

        hits = searcher.search(q, min(top_k, searcher.num_docs)).hits
        results = []
        for score, addr in hits:
            doc = searcher.doc(addr)
            idx = doc.get_first(_FIELD_INDEX)
            if idx is not None and score > 0:
                results.append((int(idx), float(score)))
        return results

    def search_scores(self, query: str) -> list[float]:
        """Return BM25 scores for *all* chunks.

        This is a convenience wrapper used by the CLI for display.  For
        production queries, use ``search()`` instead which is faster.
        """
        if not self.chunks:
            return []
        scored = self.search(query, top_k=len(self.chunks))
        # Build a full array indexed by chunk position
        scores = [0.0] * len(self.chunks)
        for idx, score in scored:
            scores[idx] = score
        return scores

    def build_from_output_dir(self, force: bool = False) -> "BM25Index | None":
        """Build the index directly from technique markdown files."""
        if not force and self.load():
            return self

        from src.embedding.splitter import split_all_technique_files
        chunks = split_all_technique_files()
        if not chunks:
            logger.warning("No technique files found in %s", settings.output_dir)
            return None
        return self.build(chunks)