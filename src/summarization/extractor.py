"""Async LLM extraction loop for batch processing writeups."""

import asyncio
import json
import logging
import signal
from collections.abc import AsyncGenerator
from datetime import UTC, datetime

from src.config import settings
from src.llm import LLMClient
from src.models import ProgressEvent, Writeup
from src.processing.batch import (
    create_writeup_batches,
    format_batch_for_prompt,
    load_writeups,
)
from src.summarization.checkpoint import CheckpointManager
from src.summarization.prompts import build_extraction_prompt

logger = logging.getLogger(__name__)


class TrickExtractor:
    """Orchestrates async batch extraction of tricks from writeups."""

    def __init__(
        self,
        checkpoint_manager: CheckpointManager | None = None,
        concurrency: int = 0,
    ):
        self.llm = LLMClient()
        self.checkpoint_mgr = checkpoint_manager or CheckpointManager()
        # Batch concurrency: CLI override wins, otherwise LLM_MAX_CONCURRENCY.
        self.max_concurrency = concurrency or settings.llm_max_concurrency
        self._paused = asyncio.Event()
        self._running = True
        self._active_tasks: set[asyncio.Task] = set()
        self._semaphore = asyncio.Semaphore(self.max_concurrency)

        # Register signal handlers for graceful pause
        try:
            loop = asyncio.get_event_loop()
            for sig in (signal.SIGINT, signal.SIGTERM):
                loop.add_signal_handler(sig, self._handle_signal)
        except (NotImplementedError, ValueError, RuntimeError):
            # Windows, testing, or server context (uvicorn) without signal support
            pass

    def _handle_signal(self) -> None:
        """Handle termination signals by requesting a pause."""
        logger.info("Received interrupt signal, pausing after current batch...")
        self._paused.set()

    async def request_pause(self) -> None:
        """Request a pause after the current batch completes."""
        logger.info("Pause requested, waiting for in-flight batches...")
        self._paused.set()

    async def extract_all(
        self,
        batch_limit: int = 0,
        force_reset: bool = False,
        writeups: list[Writeup] | None = None,
    ) -> AsyncGenerator[ProgressEvent, None]:
        """Main extraction loop with async concurrency and checkpointing.

        Args:
            batch_limit: Only process this many batches (0 = all).
            force_reset: Ignore existing checkpoint and start fresh.
            writeups: Optional pre-filtered writeup list. If None, loads all
                writeups from data.json (used by `--fill-gaps` to only process
                writeups that have no tricks yet).

        Yields:
            ProgressEvent after each batch.
        """
        if force_reset:
            self.checkpoint_mgr.reset()
            if writeups is None:
                # Full re-extraction (writeups is None = whole corpus, not
                # --fill-gaps). Clear the trick accumulator so a forced run
                # produces a clean, single-copy trick set instead of appending
                # new tricks to prior runs' tricks.jsonl (which made --force
                # runs accumulate old+new tricks).
                self._clear_tricks_files()

        # Load checkpoint
        checkpoint = self.checkpoint_mgr.load()

        if checkpoint.status == "completed":
            yield ProgressEvent(
                status="completed",
                message="All batches already processed",
                progress_pct=100.0,
            )
            return

        # Load and batch writeups
        if writeups is None:
            writeups = load_writeups()
        if not writeups:
            yield ProgressEvent(
                status="failed",
                message="No writeups loaded",
                progress_pct=0.0,
            )
            return

        logger.info("Loaded %d writeups, creating batches...", len(writeups))
        batches = create_writeup_batches(writeups)
        total_batches = min(batch_limit, len(batches)) if batch_limit > 0 else len(batches)
        logger.info("Created %d batches, starting extraction...", total_batches)

        if checkpoint.total_batches == 0:
            checkpoint.total_batches = total_batches
            checkpoint.batch_index = 0
            checkpoint.status = "running"
            checkpoint.started_at = datetime.now(UTC).isoformat()

        start_idx = self.checkpoint_mgr.resume_from(checkpoint)
        if start_idx >= total_batches:
            yield ProgressEvent(
                status="completed",
                message="All batches already processed",
                tricks_extracted=checkpoint.total_tricks_extracted,
                tokens_used=checkpoint.total_tokens_used,
                progress_pct=100.0,
            )
            return

        logger.info(
            "Starting extraction from batch %d/%d",
            start_idx + 1,
            total_batches,
        )

        # Yield immediately so callers can show a progress bar with the total
        # known, without waiting for the first (slow) LLM batch to complete.
        yield ProgressEvent(
            status="running",
            message=f"Extracting from batch {start_idx + 1}/{total_batches}",
            batch_index=start_idx,
            total_batches=total_batches,
            tricks_extracted=checkpoint.total_tricks_extracted,
            tokens_used=checkpoint.total_tokens_used,
            progress_pct=round((start_idx / total_batches) * 100, 1) if total_batches else 0.0,
        )

        # Process batches with concurrency
        tricks_accumulator: list[dict] = []
        pending_set: set[asyncio.Task] = set()
        batch_queue = list(range(start_idx, total_batches))
        # Re-queue any permanently-failed batches from a previous run so they
        # get retried.  Without this, the linear range(start_idx, total_batches)
        # skips every failed batch whose index is before start_idx.
        for fb in list(checkpoint.failed_batches):
            idx = fb["batch_index"]
            if idx < start_idx and idx not in batch_queue:
                batch_queue.append(idx)
        # Clear the old failure records so we start fresh for this resume
        checkpoint.failed_batches = []
        completed_count = start_idx
        failed_count = 0

        try:
            while batch_queue or pending_set:
                # Fill up to the concurrency limit
                while batch_queue and len(pending_set) < self.max_concurrency:
                    batch_idx = batch_queue.pop(0)
                    task = asyncio.create_task(
                        self._process_batch(batch_idx, batches[batch_idx])
                    )
                    pending_set.add(task)

                # Wait for at least one to complete
                if pending_set:
                    done, pending_set = await asyncio.wait(
                        pending_set, return_when=asyncio.FIRST_COMPLETED
                    )
                else:
                    break

                for task in done:
                    batch_idx, batch_tricks, batch_tokens, unresolved_urls = task.result()

                    if unresolved_urls:
                        failed_count += 1
                        checkpoint.failed_batches.append({
                            "batch_index": batch_idx,
                            "unresolved_urls": unresolved_urls,
                        })
                        self.checkpoint_mgr.save(checkpoint)
                        logger.error(
                            "Batch %d: %d writeup(s) unresolved (%s), partial tricks saved",
                            batch_idx, len(unresolved_urls), unresolved_urls,
                        )
                        # Still add partial tricks from the successful writeups
                        if batch_tricks:
                            tricks_accumulator.extend(batch_tricks)
                            checkpoint.total_tricks_extracted += len(batch_tricks)
                            checkpoint.total_tokens_used += batch_tokens
                            checkpoint.processed_writeup_urls.extend(
                                w.url for w in batches[batch_idx]
                                if w.url not in unresolved_urls
                            )
                            self.checkpoint_mgr.save(checkpoint)
                            self._save_tricks_batch(batch_tricks)
                        yield ProgressEvent(
                            batch_index=batch_idx,
                            total_batches=total_batches,
                            tricks_extracted=checkpoint.total_tricks_extracted,
                            tokens_used=checkpoint.total_tokens_used,
                            completed_count=completed_count,
                            progress_pct=round((completed_count / total_batches) * 100, 1) if total_batches > 0 else 0,
                            status="running",
                            message=f"Batch {batch_idx + 1} had {len(unresolved_urls)} unresolved writeup(s)",
                        )
                        continue  # no re-queue; internal retries already exhausted

                    # ---- Successful batch (no unresolved writeups) ----
                    completed_count += 1

                    # Extend tricks accumulator
                    tricks_accumulator.extend(batch_tricks)

                    # Update checkpoint
                    checkpoint.batch_index = batch_idx
                    checkpoint.total_tricks_extracted += len(batch_tricks)
                    checkpoint.total_tokens_used += batch_tokens
                    checkpoint.processed_writeup_urls.extend(
                        w.url for w in batches[batch_idx]
                    )
                    self.checkpoint_mgr.save(checkpoint)

                    # Save tricks incrementally
                    self._save_tricks_batch(batch_tricks)

                    yield ProgressEvent(
                        batch_index=batch_idx,
                        total_batches=total_batches,
                        tricks_extracted=checkpoint.total_tricks_extracted,
                        tokens_used=checkpoint.total_tokens_used,
                                completed_count=completed_count,
                        progress_pct=round((completed_count / total_batches) * 100, 1) if total_batches > 0 else 0,
                        status="running",
                        message=f"Processed batch {batch_idx + 1}/{total_batches}",
                    )

                # Check if pause was requested
                if self._paused.is_set():
                    logger.info("Pausing extraction at batch %d", completed_count)
                    checkpoint.status = "paused"
                    self.checkpoint_mgr.save(checkpoint)

                    # Cancel remaining tasks
                    for t in pending_set:
                        t.cancel()

                    await asyncio.gather(*pending_set, return_exceptions=True)
                    pending_set.clear()

                    yield ProgressEvent(
                        status="paused",
                        message=f"Paused at batch {completed_count}/{total_batches}",
                        batch_index=completed_count - 1,
                        total_batches=total_batches,
                        tricks_extracted=checkpoint.total_tricks_extracted,
                        tokens_used=checkpoint.total_tokens_used,
                                completed_count=completed_count,
                        progress_pct=round((completed_count / total_batches) * 100, 1) if total_batches > 0 else 0,
                    )
                    return

            # All done — determine final status
            if failed_count > 0:
                checkpoint.status = "failed"
                self.checkpoint_mgr.save(checkpoint)
                msg = (
                    f"{completed_count} batches processed, {failed_count} failed permanently"
                )
                final_status = "failed"
            else:
                checkpoint.status = "completed"
                self.checkpoint_mgr.save(checkpoint)
                msg = (
                    f"All {total_batches} batches processed, "
                    f"{checkpoint.total_tricks_extracted} tricks extracted"
                )
                final_status = "completed"

            # Save all tricks to a combined file
            self._save_all_tricks(tricks_accumulator)

            yield ProgressEvent(
                status=final_status,
                message=msg,
                batch_index=total_batches - 1,
                total_batches=total_batches,
                tricks_extracted=checkpoint.total_tricks_extracted,
                tokens_used=checkpoint.total_tokens_used,
                                completed_count=completed_count,
                progress_pct=100.0,
            )

        except Exception as e:
            logger.exception("Extraction failed: %s", e)  # noqa: TRY401
            checkpoint.status = "failed"
            self.checkpoint_mgr.save(checkpoint)
            yield ProgressEvent(
                status="failed",
                message=str(e),
                batch_index=completed_count,
                total_batches=total_batches,
                tricks_extracted=checkpoint.total_tricks_extracted,
                tokens_used=checkpoint.total_tokens_used,
                progress_pct=round((completed_count / total_batches) * 100, 1) if total_batches > 0 else 0,
            )

    async def _attempt_batch(
        self, writeups: list[Writeup]
    ) -> tuple[list[dict], int]:
        """Run one LLM extraction attempt on ``writeups``.

        Raises on LLM/gateway errors; returns (tricks, tokens_used) on success.
        """
        writeup_text = format_batch_for_prompt(writeups)
        system_prompt, user_message = build_extraction_prompt(writeup_text)

        response = await self.llm.create_message(
            system_prompt=system_prompt,
            user_message=user_message,
            cache_system=True,
        )

        # If the LLM returned a bare list (common: the prompt asks for
        # a JSON array), use it directly.
        if isinstance(response, list):
            tricks = response
        elif isinstance(response, dict):
            tricks = response.get("tricks", [])
            if not tricks and "raw_content" in response:
                # Parsing failed; try to recover tricks from raw text
                tricks = self._extract_tricks_from_raw(response["raw_content"])
        else:
            tricks = []

        # Attach source writeup URLs. The LLM reports source_indexes
        # (writeup numbers within the batch, 1-based); only those writeups
        # are credited with having exhibited the trick. If the field is
        # missing/invalid, fall back to the whole batch (legacy behaviour).
        for trick in tricks:
            if not isinstance(trick, dict):
                continue
            idxs = trick.pop("source_indexes", None)
            if isinstance(idxs, list) and idxs:
                urls = []
                for i in idxs:
                    try:
                        w = writeups[int(i) - 1]
                        urls.append(w.url)
                    except (ValueError, IndexError):
                        continue
                if urls:
                    trick["source_writeups"] = urls
                else:
                    trick["source_writeups"] = []
            else:
                trick["source_writeups"] = [w.url for w in writeups]
            trick["original_terms"] = [trick.get("technique_name", "")]

        # Estimate tokens used
        tokens_used = sum(len(w.cleaned_content) // 4 for w in writeups)

        return tricks, tokens_used

    async def _process_batch(
        self, batch_idx: int, writeups: list[Writeup]
    ) -> tuple[int, list[dict], int, list[str]]:
        """Process a batch of writeups through the LLM with failure recovery.

        Tiered strategy:
        1. Retry the full batch up to ``llm_max_retries`` times (transient errors).
        2. If still failing, reduce batch size: split in half and recurse.
        3. A single isolated writeup that still fails is truncated to
           ``max_single_truncate_chars`` (0.5M) as a last resort.

        Returns:
            (batch_idx, tricks, tokens_used, unresolved_urls)
            ``unresolved_urls`` is empty when every writeup was successfully
            extracted; non-empty means some writeups permanently failed.
        """
        async with self._semaphore:
            return await self._process_batch_resolve(batch_idx, writeups)

    async def _process_batch_resolve(
        self, batch_idx: int, writeups: list[Writeup]
    ) -> tuple[int, list[dict], int, list[str]]:
        """Internal resolver: retry → halve → single-writeup truncation."""
        # Tier 1: retry the full batch up to llm_max_retries
        for attempt in range(1, settings.llm_max_retries + 1):
            try:
                tricks, tokens = await self._attempt_batch(writeups)
                return batch_idx, tricks, tokens, []
            except Exception:
                if attempt < settings.llm_max_retries:
                    logger.warning(
                        "Batch %d failed (attempt %d/%d), retrying",
                        batch_idx, attempt, settings.llm_max_retries,
                    )
                else:
                    logger.warning(
                        "Batch %d exhausted retries, reducing batch size",
                        batch_idx,
                    )

        # Tier 2: reduce batch size by halving
        if len(writeups) > 1:
            mid = len(writeups) // 2
            left = await self._process_batch_resolve(batch_idx, writeups[:mid])
            right = await self._process_batch_resolve(batch_idx, writeups[mid:])
            return (
                batch_idx,
                left[1] + right[1],
                left[2] + right[2],
                left[3] + right[3],
            )

        # Tier 3: single writeup — truncate to max_single_truncate_chars
        w = writeups[0]
        limit = settings.max_single_truncate_chars
        if limit > 0 and len(w.cleaned_content) > limit:
            logger.warning(
                "Single writeup %s still failing, truncating to %d chars",
                w.url, limit,
            )
            w.cleaned_content = (
                w.cleaned_content[:limit] + "\n...[truncated for length]"
            )
            try:
                tricks, tokens = await self._attempt_batch(writeups)
                return batch_idx, tricks, tokens, []
            except Exception:
                pass

        unresolved = [w.url for w in writeups if w.url]
        logger.error(
            "Writeup(s) %s failed permanently", unresolved,
        )
        return batch_idx, [], 0, [u for u in unresolved if u]

    def _extract_tricks_from_raw(self, raw: str) -> list[dict]:
        """Fallback: try to extract trick objects from raw text."""
        import re

        tricks = []
        # Try to find JSON arrays
        for match in re.finditer(r"\[[\s\S]*?\]", raw):
            try:
                data = json.loads(match.group())
                if isinstance(data, list):
                    for item in data:
                        if isinstance(item, dict) and "technique_name" in item:
                            tricks.append(item)
                    if tricks:
                        break
            except (json.JSONDecodeError, ValueError):
                continue
        return tricks

    def _clear_tricks_files(self) -> None:
        """Delete the trick accumulator files for a clean forced re-extraction.

        The JSONL accumulator is opened in append mode during extraction, so a
        --force run without clearing would append a second copy of every trick
        to the existing file. Remove both the JSONL accumulator and the combined
        JSON array so deduplication sees only the freshly extracted tricks.
        """
        for name in ("tricks.jsonl", "tricks_all.json"):
            path = settings.tricks_dir / name
            if path.exists():
                path.unlink()
                logger.info("Removed %s for fresh --force re-extraction", path)

    def _save_tricks_batch(
        self, tricks: list[dict]
    ) -> None:
        """Save tricks from a batch to the JSONL accumulator file."""
        tricks_dir = settings.tricks_dir
        tricks_dir.mkdir(parents=True, exist_ok=True)

        # Accumulate into a single JSONL file
        output_file = tricks_dir / "tricks.jsonl"
        with open(output_file, "a") as f:
            f.writelines(json.dumps(trick) + "\n" for trick in tricks)

    def _save_all_tricks(self, tricks: list[dict]) -> None:
        """Save all extracted tricks to a single combined file.

        Rebuilds from the JSONL accumulator (tricks.jsonl) rather than the
        in-memory list, because the accumulator records every batch across
        sessions: a resumed run starts with an empty in-memory accumulator, so
        writing it directly would drop the tricks extracted before the pause.
        Falls back to the in-memory list if the accumulator is missing.
        """
        output_file = settings.tricks_dir / "tricks_all.json"

        jsonl_path = settings.tricks_dir / "tricks.jsonl"
        if jsonl_path.exists():
            merged = []
            with open(jsonl_path, "r", encoding="utf-8") as f:
                for line in f:
                    line = line.strip()
                    if not line:
                        continue
                    try:
                        merged.append(json.loads(line))
                    except json.JSONDecodeError:
                        continue
            tricks = merged

        with open(output_file, "w") as f:
            json.dump(tricks, f, indent=2)
        logger.info("Saved %d tricks to %s", len(tricks), output_file)