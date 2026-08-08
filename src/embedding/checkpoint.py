"""Embedding pipeline checkpoint for pause/resume."""

import json
import logging
from datetime import UTC
from pathlib import Path

from src.config import settings
from src.models import EmbeddingCheckpoint

logger = logging.getLogger(__name__)


class EmbeddingCheckpointManager:
    """Manages save/load/resume of the embedding pipeline."""

    def __init__(self, checkpoint_path: str | Path | None = None):
        if checkpoint_path is None:
            checkpoint_path = settings.checkpoint_dir / "embedding_state.json"
        self.checkpoint_path = Path(checkpoint_path)
        self.checkpoint_path.parent.mkdir(parents=True, exist_ok=True)

    def load(self) -> EmbeddingCheckpoint:
        """Load embedding checkpoint from disk."""
        if not self.checkpoint_path.exists():
            logger.info("No embedding checkpoint found, starting fresh")
            return EmbeddingCheckpoint()

        try:
            with open(self.checkpoint_path, "r") as f:
                data = json.load(f)
            checkpoint = EmbeddingCheckpoint(**data)
            logger.info(
                "Loaded embedding checkpoint: %d files processed, %d chunks",
                len(checkpoint.processed_files),
                checkpoint.total_chunks,
            )
            return checkpoint
        except (json.JSONDecodeError, KeyError, TypeError) as e:
            logger.warning("Corrupted embedding checkpoint: %s", e)
            return EmbeddingCheckpoint()

    def save(self, checkpoint: EmbeddingCheckpoint) -> None:
        """Save the embedding checkpoint."""
        from datetime import datetime

        checkpoint.updated_at = datetime.now(UTC).isoformat()
        with open(self.checkpoint_path, "w") as f:
            f.write(checkpoint.model_dump_json(indent=2))
        logger.debug("Embedding checkpoint saved")

    def reset(self) -> None:
        """Delete the embedding checkpoint for a fresh start."""
        if self.checkpoint_path.exists():
            self.checkpoint_path.unlink()
            logger.info("Embedding checkpoint reset")