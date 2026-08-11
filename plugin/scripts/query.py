#!/usr/bin/env python3
"""Query the nekozuki RAG knowledge base.

Usage:
    python query.py "natural language query" [-n N] [--server URL]

Tries a running nekozuki API server first (fast coarse search, no auth needed).
Falls back to `uv run nekozuki query` (full RAG with rerank, slower).

Set NEKOZUKI_URL env var to point to a running server (default http://localhost:8000).
"""

import json
import os
import subprocess
import sys

DEFAULT_SERVER = os.environ.get("NEKOZUKI_URL", "http://localhost:8000")
DEFAULT_TOP_K = 10


def query_api(query: str, top_k: int, server: str) -> list[dict] | None:
    """Try coarse search via the live API. Returns list of results or None."""
    import urllib.error
    import urllib.parse
    import urllib.request

    url = f"{server.rstrip('/')}/api/tricks/search?q={urllib.parse.quote(query)}&limit={top_k}"
    try:
        req = urllib.request.Request(url, method="GET")
        with urllib.request.urlopen(req, timeout=5) as resp:
            data = json.loads(resp.read().decode())
        return data.get("results", [])
    except (urllib.error.URLError, urllib.error.HTTPError, OSError, json.JSONDecodeError):
        return None


def query_cli(query: str, top_k: int) -> list[dict] | None:
    """Fall back to `uv run nekozuki query`. Returns list of results or None."""
    try:
        result = subprocess.run(
            ["uv", "run", "nekozuki", "query", query, "--top-k", str(top_k)],
            capture_output=True,
            text=True,
            timeout=120,
            cwd=_find_project_root(),
        )
        if result.returncode != 0:
            return None
        # Parse the CLI output (it prints formatted text, not JSON)
        return _parse_cli_output(result.stdout, query)
    except (subprocess.TimeoutExpired, subprocess.CalledProcessError, FileNotFoundError):
        return None


def _find_project_root() -> str | None:
    """Walk up from the script dir to find pyproject.toml (project root)."""
    script_dir = os.path.dirname(os.path.abspath(__file__))
    for _ in range(5):
        if os.path.exists(os.path.join(script_dir, "pyproject.toml")):
            return script_dir
        parent = os.path.dirname(script_dir)
        if parent == script_dir:
            return None
        script_dir = parent
    return None


def _parse_cli_output(stdout: str, _query: str) -> list[dict]:
    """Parse the plain-text CLI output into structured results."""
    results = []
    current: dict | None = None
    for line in stdout.splitlines():
        line = line.strip()
        if line.startswith("[") and "]" in line and "—" in line:
            if current:
                results.append(current)
            rank = len(results) + 1
            # e.g. "[1] sql_injection — Blind SQL Injection via Timing"
            rest = line[line.index("]") + 1:].strip()
            parts = rest.split(" — ", 1)
            technique = parts[0] if len(parts) > 0 else ""
            title = parts[1] if len(parts) > 1 else ""
            current = {"rank": rank, "technique_name": technique, "title": title, "content": ""}
        elif "Rerank:" in line and current:
            # e.g. "Rerank: 0.923 (bm25: 1.234)"
            for part in line.split():
                if "Rerank:" in part:
                    try:
                        current["rerank_score"] = float(part.split(":")[1].rstrip(","))
                    except (ValueError, IndexError):
                        pass
        elif current and line.startswith("http"):
            current["content"] = line[:200]
        elif current and not line.startswith("=") and not line.startswith("Query:"):
            current["content"] = line[:200]
    if current:
        results.append(current)
    return results


def format_results_api(results: list[dict], query: str) -> str:
    """Format API results (coarse search, no scores)."""
    lines = [f"Query: {query}", "", f"Results ({len(results)}):", "=" * 60]
    for i, r in enumerate(results, 1):
        technique = r.get("technique_name", "")
        title = r.get("title", "")
        desc = r.get("description", "")[:180]
        lines.append(f"\n[{i}] {technique} — {title}")
        if desc:
            lines.append(f"    {desc}")
    return "\n".join(lines)


def format_results_cli(results: list[dict], query: str) -> str:
    """Format CLI results (full RAG with scores)."""
    if not results:
        return "No results."
    lines = [f"Query: {query}", "", "Results (RAG + rerank):", "=" * 60]
    for r in results:
        score = r.get("rerank_score", "?")
        technique = r.get("technique_name", "")
        title = r.get("title", "")
        content = r.get("content", "")[:200]
        lines.append(f"\n[{r.get('rank', '?')}] {technique} — {title}")
        lines.append(f"    Rerank: {score}")
        if content:
            lines.append(f"    {content}")
    return "\n".join(lines)


def main() -> int:
    args = sys.argv[1:]
    query = ""
    top_k = DEFAULT_TOP_K
    server = DEFAULT_SERVER
    force_cli = False

    i = 0
    while i < len(args):
        if args[i] in ("-n", "--top-k", "--top_k"):
            i += 1
            if i < len(args):
                top_k = int(args[i])
        elif args[i] == "--server":
            i += 1
            if i < len(args):
                server = args[i]
        elif args[i] == "--cli":
            force_cli = True
        elif not query:
            query = args[i]
        i += 1

    if not query:
        print("Usage: query.py <query> [-n N] [--server URL] [--cli]", file=sys.stderr)
        return 1

    # Try API first, then CLI
    if not force_cli:
        results = query_api(query, top_k, server)
        if results is not None:
            print(format_results_api(results, query))
            return 0
        # API not available, fall through to CLI

    results = query_cli(query, top_k)
    if results is not None:
        print(format_results_cli(results, query))
        return 0

    print(
        "nekozuki query failed: no API server running and CLI query errored.\n"
        "Start the server with `uv run nekozuki serve` or run `nekozuki embed` first.",
        file=sys.stderr,
    )
    return 1


if __name__ == "__main__":
    sys.exit(main())