"""Profile where RAG search time goes: BM25 vs rerank vs question boost."""
import asyncio
import time

from src.retrieval.hybrid import HybridSearcher


async def main():
    searcher = HybridSearcher()
    if not searcher.loaded:
        print("index not loaded")
        return
    n_chunks = len(searcher.bm25_index.chunks)
    print(f"chunks: {n_chunks}, recall_k: {searcher.recall_k}, question_weight: {searcher.question_weight}")
    print(f"reranker: base_url={searcher.reranker.base_url[:50]}... model={searcher.reranker.model}, top_n={searcher.reranker.top_n}")
    print()

    queries = [
        "how to bypass the CSP of the main website which only allows same-origin",
        "sql injection authentication bypass",
    ]
    for q in queries:
        print(f"=== query: {q[:60]} ===")
        # 1. BM25 alone
        t0 = time.perf_counter()
        candidates = searcher.bm25_index.search(q, top_k=searcher.recall_k)
        t_bm25 = time.perf_counter() - t0
        print(f"  BM25 recall top-{searcher.recall_k}: {t_bm25*1000:.0f} ms ({len(candidates)} hits)")

        # 2. Rerank alone
        chunk_pairs = [(i, searcher.bm25_index.chunks[i]) for i, _ in candidates]
        t0 = time.perf_counter()
        ranked = searcher.reranker.rerank(q, chunk_pairs)
        t_rerank = time.perf_counter() - t0
        print(f"  rerank API ({len(chunk_pairs)} docs): {t_rerank*1000:.0f} ms")

        # 3. Question boost alone (embedding + chroma)
        t0 = time.perf_counter()
        boosts = await searcher._question_boosts(q)
        t_boost = time.perf_counter() - t0
        print(f"  question boost: {t_boost*1000:.0f} ms ({len(boosts)} techniques)")

        # 4. Full search
        t0 = time.perf_counter()
        results = await searcher.search(q, top_k=10)
        t_total = time.perf_counter() - t0
        print(f"  FULL search: {t_total*1000:.0f} ms")
        print()


asyncio.run(main())
