---
name: nekozuki
description: Query the nekozuki CTF writeup knowledge base — a RAG over 39k CTF writeups distilled into technique files. Use when the user asks a natural-language question about CTF techniques, exploits, detection, or attack vectors (e.g. "how do I exploit PHP type juggling?", "what are detection signs for SSTI?").
argument-hint: [query] [-n N]
allowed-tools: [Bash, Read]
---

# nekozuki RAG query

Search the nekozuki knowledge base (distilled CTF tricks + hybrid BM25/embedding
retrieval) for techniques relevant to the user's question.

## Arguments

- `query` — the natural-language question, quoted if it contains spaces.
- `-n N` — number of results to return (default 10).

## Steps

1. Run the query script, passing the query through. Use the default result
   count unless the user asked for more/fewer:

   ```
   bash "$CLAUDE_PLUGIN_ROOT/scripts/query.py" "<query>" -n N
   ```

   If running from inside the repo, the script also works as
   `uv run .../scripts/query.py`. The script prefers a live nekozuki API
   server (fast coarse search) and falls back to the full CLI RAG query when
   the server is not running.

2. Present the results as a concise numbered list. For each hit show:
   - the technique name and trick title,
   - the rerank/bm25 scores (CLI mode) or the coarse match (API mode),
   - a short content preview.

3. If the user wants the full trick (conditions, implementation, key code,
   example, detection signs), advise they can fetch it via
   `GET /api/tricks/{id}` on the running server, or ask to open the matching
   `output/*.md` technique file.

## Notes

- If no results come back, suggest the user may need to run
  `uv run nekozuki embed` (the RAG API returns 503 until the index is built)
  or start the server with `uv run nekozuki serve`.
- The coarse API search is fast (~20-50ms) but does no reranking; the full
  CLI query (`uv run nekozuki query`) reranks and is slower.