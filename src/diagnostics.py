"""Configuration & connectivity diagnostics for nekozuki.

`nekozuki test` runs a battery of checks against the current configuration
(.env + environment) and the on-disk artifacts, then reports a pass/fail
report and a process exit code:

    0 = all required checks pass
    1 = at least one required check failed
"""

import asyncio
import json
import logging
from dataclasses import dataclass
from pathlib import Path

from src.config import settings

logger = logging.getLogger(__name__)

# Statuses
OK = "ok"
WARN = "warn"
FAIL = "fail"
SKIP = "skip"


@dataclass
class CheckResult:
    """A single diagnostic check outcome."""
    name: str
    status: str  # ok | warn | fail | skip
    detail: str


def _mask(value: str) -> str:
    """Mask an API key for safe display: sk-***abcd."""
    value = value.strip()
    if not value:
        return "(empty)"
    if len(value) <= 10:
        return "***"
    return f"{value[:4]}...{value[-4:]}"


def _dir_writable(path: Path) -> bool:
    """Check whether a directory exists (or can be created) and is writable."""
    try:
        path.mkdir(parents=True, exist_ok=True)
        test = path / ".write_test"
        test.write_text("x")
        test.unlink()
        return True
    except OSError:
        return False


def check_llm_key() -> CheckResult:
    """Anthropic API key present."""
    if settings.llm_api_key:
        return CheckResult(
            "Anthropic API key", OK,
            f"set ({_mask(settings.llm_api_key)})",
        )
    return CheckResult(
        "Anthropic API key", FAIL,
        "missing — set ANTHROPIC_API_KEY in .env",
    )


def check_embedding_key() -> CheckResult:
    """OpenAI embedding API key present."""
    if settings.embedding_api_key:
        return CheckResult(
            "OpenAI API key", OK,
            f"set ({_mask(settings.embedding_api_key)})",
        )
    return CheckResult(
        "OpenAI API key", FAIL,
        "missing — set OPENAI_API_KEY in .env",
    )


def check_base_url() -> CheckResult:
    """Custom LLM endpoint (BASE_URL) configured."""
    if settings.llm_base_url:
        return CheckResult(
            "LLM base URL", OK, settings.llm_base_url,
        )
    return CheckResult(
        "LLM base URL", WARN, "unset — using Anthropic default endpoint",
    )


def check_embedding_base_url() -> CheckResult:
    """Custom embedding endpoint (EMBEDDING_BASE_URL) configured."""
    if settings.embedding_base_url:
        return CheckResult(
            "Embedding base URL", OK, settings.embedding_base_url,
        )
    return CheckResult(
        "Embedding base URL", WARN, "unset — using OpenAI default endpoint",
    )


def check_llm_max_tokens() -> CheckResult:
    """LLM_MAX_TOKENS is a sane output budget (not the context window)."""
    max_tokens = settings.llm_max_tokens
    if max_tokens > 64000:
        return CheckResult(
            "LLM max_tokens", WARN,
            f"{max_tokens:,} is the OUTPUT budget — huge values force streaming "
            "and exceed gateway caps; the client clamps to 64k",
        )
    return CheckResult(
        "LLM max_tokens", OK,
        f"{max_tokens:,} output tokens (reasonable)",
    )


def check_data_file() -> CheckResult:
    """data.json exists and parses as a writeup list."""
    path = Path(settings.data_path)
    if not path.exists():
        return CheckResult("Data file", FAIL, f"not found at {path}")

    try:
        with open(path, "r", encoding="utf-8") as f:
            data = json.load(f)
        size_kb = path.stat().st_size / 1024
        if isinstance(data, list):
            return CheckResult(
                "Data file", OK,
                f"{path} — {len(data):,} writeups, {size_kb:.0f} KB",
            )
        return CheckResult(
            "Data file", FAIL, f"{path} — expected a JSON list, got {type(data).__name__}",
        )
    except json.JSONDecodeError as e:
        return CheckResult("Data file", FAIL, f"{path} — invalid JSON: {e}")


def check_directories() -> CheckResult:
    """Output / checkpoint / vector directories are writable."""
    dirs = {
        "output": settings.output_dir,
        "checkpoints": settings.checkpoint_dir,
        "vectors": settings.vectors_dir,
    }
    problems = []
    for name, path in dirs.items():
        if not _dir_writable(path):
            problems.append(f"{name} ({path}) not writable")
    if problems:
        return CheckResult("Directories", FAIL, "; ".join(problems))
    return CheckResult(
        "Directories", OK,
        "output/ checkpoints/ vectors/ writable",
    )


def check_summarize_checkpoint() -> CheckResult:
    """Summarization checkpoint state (pause/resume)."""
    path = Path(settings.checkpoint_dir) / "summarization_state.json"
    if not path.exists():
        return CheckResult(
            "Summarize checkpoint", SKIP,
            "none found — summarize has not been started",
        )
    try:
        from src.summarization.checkpoint import CheckpointManager
        cp = CheckpointManager().load()
        pct = (cp.batch_index / cp.total_batches * 100) if cp.total_batches else 0
        return CheckResult(
            "Summarize checkpoint", OK,
            f"{cp.status} — batch {cp.batch_index}/{cp.total_batches} "
            f"({pct:.1f}%), {cp.total_tricks_extracted:,} tricks",
        )
    except Exception as e:  # noqa: BLE001 (corrupt checkpoint)
        return CheckResult("Summarize checkpoint", WARN, f"unreadable: {e}")


def check_technique_files() -> CheckResult:
    """Generated technique files present in output/."""
    output_dir = Path(settings.output_dir)
    if not output_dir.exists():
        return CheckResult("Technique files", SKIP, "output/ does not exist yet")
    files = sorted(output_dir.glob("*.md"))
    if not files:
        return CheckResult(
            "Technique files", SKIP,
            "none yet — run `nekozuki summarize`",
        )
    tricks = sum(f.read_text(encoding="utf-8").count("\n## ") for f in files)
    return CheckResult(
        "Technique files", OK,
        f"{len(files)} files, {tricks} tricks in {output_dir}",
    )


def check_embedding_index() -> CheckResult:
    """Embedding vectors + metadata + BM25 index present."""
    vectors_dir = Path(settings.vectors_dir)
    npz = vectors_dir / "embeddings.npz"
    meta = vectors_dir / "chunk_metadata.json"
    bm25 = vectors_dir / "bm25_index.pkl"

    if not npz.exists():
        return CheckResult(
            "Embedding index", SKIP,
            "not built — run `nekozuki embed`",
        )

    missing = [str(p.name) for p in (meta, bm25) if not p.exists()]
    try:
        import numpy as np
        with np.load(npz, allow_pickle=True) as data:
            n_chunks = int(data["chunk_embeddings"].shape[0])
            dim = int(data["chunk_embeddings"].shape[1]) if n_chunks else 0
    except Exception as e:  # noqa: BLE001 (bad npz)
        return CheckResult("Embedding index", FAIL, f"unreadable: {e}")

    detail = f"{n_chunks:,} chunk vectors, {dim}d"
    if missing:
        detail += f"; missing: {', '.join(missing)}"
        return CheckResult("Embedding index", WARN, detail)
    return CheckResult("Embedding index", OK, detail)


def check_prompt_cache() -> CheckResult:
    """System prompts exceed the 1024-token cache threshold."""
    import tiktoken

    from src.summarization.prompts import QUESTIONS_SYSTEM_PROMPT, SYSTEM_PROMPT

    enc = tiktoken.get_encoding("cl100k_base")
    tokens = {
        "summarize": len(enc.encode(SYSTEM_PROMPT)),
        "questions": len(enc.encode(QUESTIONS_SYSTEM_PROMPT)),
    }
    below = [name for name, n in tokens.items() if n < 1024]
    detail = " · ".join(f"{name} {n} tok" for name, n in tokens.items())
    if below:
        return CheckResult(
            "Prompt cache threshold", WARN,
            f"{detail} — {'/'.join(below)} below 1024 tokens",
        )
    return CheckResult(
        "Prompt cache threshold", OK,
        f"{detail} (>=1024 tokens, cacheable)",
    )


def check_llm_connectivity() -> CheckResult:
    """Reach the Anthropic endpoint with a tiny test call."""
    if not settings.llm_api_key:
        return CheckResult("LLM connectivity", SKIP, "no API key")

    async def _probe() -> str:
        from anthropic import AsyncAnthropic
        client = AsyncAnthropic(
            api_key=settings.llm_api_key,
            base_url=settings.llm_base_url or None,
        )
        try:
            resp = await client.messages.create(
                model=settings.llm_model,
                max_tokens=1,
                messages=[{"role": "user", "content": "ping"}],
            )
            return resp.model if resp.model else "ok"
        finally:
            # Close the underlying httpx client so no background task survives
            # the asyncio.run() event loop teardown.
            await client.close()

    try:
        model = asyncio.run(_probe())
        return CheckResult("LLM connectivity", OK, f"reachable ({model})")
    except Exception as e:  # noqa: BLE001 (probe failure)
        return CheckResult("LLM connectivity", FAIL, f"{type(e).__name__}: {e}")


def check_embedding_connectivity() -> CheckResult:
    """Reach the embedding endpoint with a tiny test call."""
    if not settings.embedding_api_key:
        return CheckResult("Embedding connectivity", SKIP, "no API key")

    async def _probe() -> int:
        from openai import AsyncOpenAI
        client = AsyncOpenAI(
            api_key=settings.embedding_api_key,
            base_url=settings.embedding_base_url or None,
        )
        try:
            resp = await client.embeddings.create(
                model=settings.embedding_model,
                input="ping",
                dimensions=settings.embedding_dimensions,
            )
            return len(resp.data[0].embedding)
        finally:
            await client.close()

    try:
        dim = asyncio.run(_probe())
        return CheckResult(
            "Embedding connectivity", OK,
            f"reachable ({dim}d vectors)",
        )
    except Exception as e:  # noqa: BLE001 (probe failure)
        return CheckResult(
            "Embedding connectivity", FAIL,
            f"{type(e).__name__}: {e}",
        )


def run_checks(include_connectivity: bool = True) -> tuple[list[CheckResult], int]:
    """Run the full diagnostic battery.

    Returns:
        (results, exit_code) — exit 0 if all required checks pass, else 1.
    """
    results = [
        check_llm_key(),
        check_embedding_key(),
        check_base_url(),
        check_embedding_base_url(),
        check_llm_max_tokens(),
        check_data_file(),
        check_directories(),
        check_summarize_checkpoint(),
        check_technique_files(),
        check_embedding_index(),
        check_prompt_cache(),
    ]

    if include_connectivity:
        results.append(check_llm_connectivity())
        results.append(check_embedding_connectivity())

    has_fail = any(r.status == FAIL for r in results)
    return results, 1 if has_fail else 0


def render_report(results: list[CheckResult]) -> str:
    """Format the check results as a human-readable report."""
    symbols = {OK: "[ OK ]", WARN: "[WARN]", FAIL: "[FAIL]", SKIP: "[ SKIP]"}
    lines = [
        "nekozuki configuration check",
        "============================",
        "",
    ]
    for r in results:
        lines.append(f"{symbols[r.status]} {r.name}: {r.detail}")

    n_ok = sum(1 for r in results if r.status == OK)
    n_fail = sum(1 for r in results if r.status == FAIL)
    n_warn = sum(1 for r in results if r.status == WARN)
    lines += [
        "",
        f"{n_ok} ok · {n_warn} warn · {n_fail} failed · {len(results) - n_ok - n_warn - n_fail} skipped",
    ]
    return "\n".join(lines)


def run_test_command(include_connectivity: bool = True) -> int:
    """CLI entry point for `nekozuki test`."""
    results, exit_code = run_checks(include_connectivity=include_connectivity)
    print(render_report(results))
    return exit_code