# nekozuki — Claude Code plugin

Query the nekozuki CTF writeup knowledge base directly from Claude Code.

nekozuki distills 39k CTF writeups into technique-organized markdown and indexes
them for hybrid (BM25 + embedding) semantic search. This plugin exposes that
search as a slash command so Claude can answer CTF technique questions from the
knowledge base.

## Contents

- `.claude-plugin/plugin.json` — plugin manifest
- `skills/nekozuki/SKILL.md` — the `/nekozuki` slash command (user-invoked)
- `skills/nekozuki-rag/SKILL.md` — **auto-trigger skill**: Claude automatically queries the knowledge base when you ask a CTF/security question and it would benefit from the curated writeup data
- `skills/nekozuki-url/SKILL.md` — the `/nekozuki-url` slash command (set/clear the persistent server URL)
- `scripts/query.py` — query helper (API-first, CLI fallback)
- `scripts/config.py` — persistent server URL config

## Install

### As a local plugin (recommended)

From the project root, install the plugin directory directly:

```
claude plugin install ./plugin
```

This registers the plugin for the current project (or `--user` for all projects)
and enables the `/nekozuki` command.

### As a local marketplace

Or add this directory as a marketplace and enable the plugin:

```
claude marketplace add ./plugin
claude plugin install nekozuki@local
```

## Usage

```
/nekozuki how do I exploit PHP type juggling?
/nekozuki detection signs for SSTI -n 5
```

The command runs `scripts/query.py`, which:

1. Tries a running nekozuki API server (fast coarse search, ~20-50ms, no auth).
   Point it elsewhere with `NEKOZUKI_URL` (default `http://localhost:8000`).
2. Falls back to `uv run nekozuki query "<query>"` (full RAG with reranking,
   slower — loads the whole index).

If nothing comes back, start the server (`uv run nekozuki serve`) or build the
index (`uv run nekozuki embed`); the RAG API returns 503 until the index exists.

## Development

Reload the plugin after editing a command/script:

```
claude plugin reload
```

## License

MIT