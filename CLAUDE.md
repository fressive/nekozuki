# CLAUDE.md

## Project Overview

Nekozuki is a CTF Writeup Summarization & RAG system. It distills 39,125 CTF
writeups (in `data.json`) into reusable, technique-organized markdown files, then
indexes them for hybrid (BM25 + embedding) semantic search.

## Commands

- `uv run nekozuki test [--no-connectivity]` — run configuration/connectivity diagnostics (exit 0 = all required checks pass)
- `uv run nekozuki summarize [--batch-limit N] [--force]` — extract tricks from writeups
- `uv run nekozuki summarize --fill-gaps` — extract tricks only for writeups that have none yet (incremental fill; uses a temp checkpoint)
- `uv run nekozuki missing-writeups [--sample N] [--save FILE]` — list writeups with no tricks yet (the same set `--fill-gaps` targets; `--save` writes the URLs)
- `uv run nekozuki dedup-techniques [--dry-run] [--no-llm] [--name-threshold X] [--content-threshold Y] [--llm-content-jaccard Z] [--out DIR]` — merge near-duplicate canonical techniques (spelling/synonym variants) into one file; a real run backs up, rewrites `data/tricks` technique names, appends merged aliases to `data/technique_mapping.yaml`, and re-renders `output/*.md`
- `uv run nekozuki dedup-tricks [--dry-run] [--no-embed] [--no-llm] [--embed-threshold X] [--llm-gray-lo Y] [--out DIR]` — Stage 2 trick dedup: lexical (free, deterministic) → embedding cosine (opt-in, ~$0.05) → LLM judge for gray-zone pairs (opt-in); a real run prunes merged-away tricks from `data/tricks` and re-renders `output/*.md`
- `uv run nekozuki embed [--force]` — split technique files, generate questions, embed, build index
- `uv run nekozuki query "natural language query"` — CLI RAG query
- `uv run nekozuki serve --port 8000` — start the web UI + API

## Architecture

```
data.json → clean → batch → LLM extract → normalize → dedup → output/*.md
                                                            ↓
                    splitter → embed → Chroma → BM25 + reranker → RAG API
```

- `src/processing/` — HTML cleaning (`clean.py`), writeup batching (`batch.py`)
- `src/summarization/` — LLM extraction (`extractor.py`), prompts (`prompts.py`),
  canonical technique mapping (`normalizer.py`; use `build_deterministic_mapping`/`normalize_batch`
  for stable grouping), technique-level merger (`technique_merger.py`), trick-level
  three-tier dedup (`trick_deduper.py`), dedup + file writer (`deduplicator.py`),
  pause/resume (`checkpoint.py`)
- `src/embedding/` — markdown-aware splitter (`splitter.py`), question generation
  (`questions.py`), OpenAI embedding (`engine.py`)
- `src/retrieval/` — BM25 (tantivy-backed `bm25_index.py`), Chroma vector store (`vector_store.py`),
  online cross-encoder reranker (`reranker.py`), hybrid search (`hybrid.py`)
- `src/api/` + `src/ui/` — FastAPI app, SSE progress, upload preview, embed preview.
  Coarse trick search: `GET /api/tricks/search?q=&limit=` returns lightweight
  `{id, technique_name, title, description}` over rendered `output/*.md` H2
  sections (fast in-memory scan, ~20-50ms, no rerank); `GET /api/tricks/{id}`
  returns the full trick (description/conditions/steps/key_code/example/signs +
  source writeup URLs) — id is `{technique}::{sha1(title)[:10]}`, stable across
  re-renders. Add a writeup via the pipeline: `POST /api/writeup/add` (JSON
  `{url?, content?, challenge_title?, challenge_source?}`; URL or pasted content)
  fetches/cleans → LLM extracts → persists to `data/tricks` → re-runs dedup to
  re-render `output/*.md`; background job polled via
  `GET /api/writeup/ingest-status/{job_id}`. The `/ingest-url` WebUI page
  supports both URL and pasted content.
- `src/diagnostics.py` — `nekozuki test` config/connectivity checks

## Key Conventions

- **English only** — all code output, logs, comments, and LLM prompts/output are English.
- **Prompt caching** — the summarize and question-generation system prompts are static
  (never interpolated with batch data) so Anthropic's ephemeral cache hits on every call.
  Keep them ≥1024 tokens and identical across calls.
- **Streaming LLM calls** — `src/llm.py` always streams and walks the raw SSE events
  manually (NOT the SDK's `messages.stream()` accumulator, which crashes on gateways
  that emit `thinking` blocks). Text deltas are accumulated; thinking blocks are skipped.
  `LLM_MAX_TOKENS` is the per-request OUTPUT budget and is clamped to 64k.
- **Canonical technique mapping** — `src/summarization/normalizer.py` maps variants to a
  single file (e.g. blind/time/union/error sqli → `sql_injection`). New names are
  auto-discovered and saved to `data/technique_mapping.yaml`.
- **Pause/resume** — `data/checkpoints/summarization_state.json` and
  `data/checkpoints/embedding_state.json` track progress; `Ctrl+C` or the API pauses
  cleanly after the in-flight batch.
- **Trick format** — each trick is an H2 in `output/*.md` with `Description:`,
  `Conditions:`, `Implementation:`, `Example:` (fenced), and `Detection signs:`.
- **API keys** — `ANTHROPIC_API_KEY` (LLM) and `OPENAI_API_KEY` (embeddings) read from `.env`.
- **Custom endpoints** — `BASE_URL` and `EMBEDDING_BASE_URL` (both optional, empty = provider default)
  route LLM/embedding requests through a gateway or proxy (e.g. an internal inference server).

## Testing

- `uv run pytest tests/ -q` — integration test with a mocked LLM client
- `tests/test_pipeline.py` covers: full summarize flow, normalizer mapping, splitter code-block handling

## Notes

- `data.json` is 512 KB / 379k lines / 39,125 writeups; ~13k are filtered by the
  `min_content_length` (500 chars) clean step.
- Embeddings are stored as numpy `.npz`; BM25 as a pickle. Both in `data/vectors/`.
- The RAG API returns 503 until `nekozuki embed` has been run at least once.