"""Writeup batching logic for efficient LLM processing."""

import json
import logging
from pathlib import Path

from tqdm import tqdm

from src.config import settings
from src.models import Writeup
from src.processing.clean import clean_html_content, is_content_worthwhile

logger = logging.getLogger(__name__)


def estimate_tokens(text: str) -> int:
    """Estimate the number of tokens in a text string (~4 chars/token).

    Character-based (not tiktoken) because batching calls this once per writeup
    across ~25k writeups — encoding every formatted writeup was a multi-minute
    bottleneck with no accuracy benefit for batch sizing.
    """
    return max(1, len(text) // 4)


def load_writeups(data_path: str | Path | None = None, use_cache: bool = True) -> list[Writeup]:
    """Load writeups from data.json and clean them.

    The cleaned result is cached to disk so that pause/resume runs do not
    re-clean all 39k writeups (~90s of HTML parsing).
    """
    if data_path is None:
        data_path = settings.data_path

    data_path = Path(data_path)
    if not data_path.exists():
        logger.error("Data file not found: %s", data_path)
        return []

    cache_path = settings.cleaned_writeups_path

    # Load from cache if available and fresh
    if use_cache and cache_path.exists():
        # Invalidate the cache if the source data changed
        try:
            if cache_path.stat().st_mtime >= data_path.stat().st_mtime:
                return _load_writeups_from_cache(cache_path)
            logger.info("Source data changed, refreshing writeup cache")
        except OSError:
            pass

    logger.info("Loading writeups from %s", data_path)
    with open(data_path, "r", encoding="utf-8") as f:
        raw_data = json.load(f)

    writeups = []
    skipped = 0
    for item in tqdm(raw_data, desc="Cleaning writeups", unit="writeup", mininterval=0.5):
        writeup = Writeup(**item)
        writeup.cleaned_content = clean_html_content(writeup.content)

        if not is_content_worthwhile(writeup.cleaned_content):
            skipped += 1
            continue

        writeups.append(writeup)

    # Save to cache for fast resume
    if use_cache and writeups:
        _save_writeups_to_cache(cache_path, writeups)

    logger.info(
        "Loaded %d writeups, skipped %d (too short or empty)",
        len(writeups),
        skipped,
    )
    return writeups


def _save_writeups_to_cache(path: Path, writeups: list[Writeup]) -> None:
    """Write cleaned writeups to a JSONL cache file."""
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        f.writelines(w.model_dump_json() + "\n" for w in writeups)
    logger.info("Cached %d cleaned writeups to %s", len(writeups), path)


def _load_writeups_from_cache(path: Path) -> list[Writeup]:
    """Load cleaned writeups from the JSONL cache."""
    writeups = []
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            try:
                writeups.append(Writeup.model_validate_json(line))
            except (ValueError, TypeError):
                continue
    logger.info("Loaded %d writeups from cache %s", len(writeups), path)
    return writeups


def categorize_writeup(writeup: Writeup) -> str:
    """Determine the primary category for a writeup.

    Uses challenge_category if available, otherwise falls back to
    content-based keyword hints.
    """
    if writeup.challenge_category:
        # Return the first meaningful category
        non_empty = [c for c in writeup.challenge_category if c.strip()]
        if non_empty:
            return non_empty[0].lower().strip()

    # Fallback: extract from content
    from src.processing.clean import extract_technique_hints
    hints = extract_technique_hints(writeup.cleaned_content)
    if hints:
        return hints[0]

    return "uncategorized"


def create_writeup_batches(
    writeups: list[Writeup] | None = None,
    batch_size: int | None = None,
    max_batch_tokens: int | None = None,
) -> list[list[Writeup]]:
    """Create batches optimized for LLM context and cache efficiency.

    Strategy:
    1. Group by category (similar content → similar techniques)
    2. Sort by challenge_source within each group
    3. Fill batches respecting token limits
    4. Long writeups (>max_batch_tokens) are truncated
    """
    if writeups is None:
        writeups = load_writeups()
    if batch_size is None:
        batch_size = settings.llm_batch_size
    if max_batch_tokens is None:
        max_batch_tokens = settings.llm_max_batch_tokens

    # Categorize each writeup (shows progress; hint extraction is regex-heavy)
    for w in tqdm(writeups, desc="Categorizing writeups", unit="writeup", mininterval=0.5):
        w.challenge_category = w.challenge_category or [categorize_writeup(w)]

    # Group by primary category
    groups: dict[str, list[Writeup]] = {}
    for w in writeups:
        cat = w.challenge_category[0] if w.challenge_category else "uncategorized"
        cat = cat.lower().strip()
        if cat not in groups:
            groups[cat] = []
        groups[cat].append(w)

    # Sort groups by size (largest first) for checkpoint progress visibility
    sorted_groups = sorted(groups.items(), key=lambda x: len(x[1]), reverse=True)

    # Create batches
    batches = []
    for cat, group in tqdm(
        sorted_groups, desc="Building batches", unit="category", leave=False
    ):
        # Sort by challenge_source for context coherence
        group.sort(key=lambda w: w.challenge_source)

        current_batch: list[Writeup] = []
        current_tokens = 0

        for w in group:
            # Truncate very long content to fit within max_batch_tokens
            content = w.cleaned_content
            if len(content) > 10000:
                content = content[:10000] + "\n...[truncated for length]"

            w_tokens = estimate_tokens(
                _format_writeup_for_batch(w, content)
            )

            # If single writeup exceeds max, it gets its own batch
            if w_tokens >= max_batch_tokens:
                if current_batch:
                    batches.append(current_batch)
                batches.append([w])
                current_batch = []
                current_tokens = 0
                continue

            # If adding this writeup exceeds limit, start new batch
            if current_tokens + w_tokens > max_batch_tokens and current_batch:
                batches.append(current_batch)
                current_batch = []
                current_tokens = 0

            current_batch.append(w)
            current_tokens += w_tokens

            if len(current_batch) >= batch_size:
                batches.append(current_batch)
                current_batch = []
                current_tokens = 0

        if current_batch:
            batches.append(current_batch)

    logger.info("Created %d batches from %d writeups", len(batches), len(writeups))
    return batches


def _format_writeup_for_batch(writeup: Writeup, content: str | None = None) -> str:
    """Format a single writeup for inclusion in a batch prompt."""
    if content is None:
        content = writeup.cleaned_content
        if len(content) > 10000:
            content = content[:10000] + "\n...[truncated]"

    parts = [
        f"Source: {writeup.challenge_source}",
        f"Challenge: {writeup.challenge_title}",
        f"Category: {', '.join(writeup.challenge_category) if writeup.challenge_category else 'unknown'}",
        f"Content:\n{content}",
    ]
    return "\n".join(parts)


def format_batch_for_prompt(batch: list[Writeup]) -> str:
    """Format a batch of writeups as a single prompt block."""
    parts = []
    for i, w in enumerate(batch):
        content = w.cleaned_content
        if len(content) > 10000:
            content = content[:10000] + "\n...[truncated]"
        parts.append(
            f"<writeup_{i+1}>\n"
            f"Source: {w.challenge_source}\n"
            f"Challenge: {w.challenge_title}\n"
            f"Category: {', '.join(w.challenge_category) if w.challenge_category else 'unknown'}\n"
            f"Content:\n{content}\n"
            f"</writeup_{i+1}>"
        )
    return "\n\n".join(parts)