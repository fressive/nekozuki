"""Checkpoint system for summarization pause/resume."""

import json
import logging
from pathlib import Path

from src.config import settings
from src.models import SummarizationCheckpoint

logger = logging.getLogger(__name__)


class CheckpointManager:
    """Manages save/load/resume of summarization pipeline state."""

    def __init__(self, checkpoint_path: str | Path | None = None):
        if checkpoint_path is None:
            checkpoint_path = settings.checkpoint_dir / "summarization_state.json"
        self.checkpoint_path = Path(checkpoint_path)
        self.checkpoint_path.parent.mkdir(parents=True, exist_ok=True)

    def load(self) -> SummarizationCheckpoint:
        """Load checkpoint from disk, or return a fresh one."""
        if not self.checkpoint_path.exists():
            logger.info("No checkpoint found, starting fresh")
            return SummarizationCheckpoint()

        try:
            with open(self.checkpoint_path, "r") as f:
                data = json.load(f)
            checkpoint = SummarizationCheckpoint(**data)
            logger.info(
                "Loaded checkpoint: batch %d/%d, status=%s, tricks=%d",
                checkpoint.batch_index,
                checkpoint.total_batches,
                checkpoint.status,
                checkpoint.total_tricks_extracted,
            )
            return checkpoint
        except (json.JSONDecodeError, KeyError) as e:
            logger.warning("Corrupted checkpoint, starting fresh: %s", e)
            return SummarizationCheckpoint()

    def save(self, checkpoint: SummarizationCheckpoint) -> None:
        """Save checkpoint to disk."""
        checkpoint.updated_at = __import__("datetime").datetime.now(
            __import__("datetime").timezone.utc
        ).isoformat()
        with open(self.checkpoint_path, "w") as f:
            f.write(checkpoint.model_dump_json(indent=2))
        logger.debug("Checkpoint saved: batch %d/%d", checkpoint.batch_index, checkpoint.total_batches)

    def resume_from(self, checkpoint: SummarizationCheckpoint) -> int:
        """Determine the next batch index to process."""
        if checkpoint.status in ("paused", "failed"):
            return checkpoint.batch_index + 1
        if checkpoint.status == "running" and checkpoint.batch_index > 0:
            return checkpoint.batch_index
        return 0  # fresh start

    def can_resume(self) -> bool:
        """Check if a valid checkpoint exists for resume."""
        if not self.checkpoint_path.exists():
            return False
        try:
            cp = self.load()
            return cp.status in ("paused", "running") and cp.batch_index > 0
        except Exception:  # noqa: BLE001 (corrupt checkpoint -> can't resume)
            return False

    def reset(self) -> None:
        """Delete the checkpoint file for a fresh start."""
        if self.checkpoint_path.exists():
            self.checkpoint_path.unlink()
            logger.info("Checkpoint reset")