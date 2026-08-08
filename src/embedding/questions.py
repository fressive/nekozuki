"""Pre-generated questions per technique for retrieval augmentation.

Each technique file gets a set of natural-language questions (generated once via
LLM) that are embedded alongside the chunks. At query time these question
embeddings broaden the semantic coverage of the index.
"""

import json
import logging
from pathlib import Path

from src.config import settings
from src.llm import LLMClient
from src.summarization.prompts import QUESTIONS_SYSTEM_PROMPT, QUESTIONS_USER_TEMPLATE

logger = logging.getLogger(__name__)


class QuestionGenerator:
    """Generates pre-retrieval questions for each technique file."""

    def __init__(self, llm: LLMClient | None = None):
        self.llm = llm or LLMClient()

    async def generate_for_file(self, content: str, technique_name: str) -> list[str]:
        """Generate retrieval questions for a single technique file.

        Args:
            content: The full markdown content of the technique file.
            technique_name: The canonical technique name (e.g. sql_injection).

        Returns:
            A list of natural-language question strings.
        """
        try:
            # Cap the source text so very large multi-trick files stay within a
            # sane LLM budget, but keep the window generous enough that tricks
            # deep in a file are still represented (see
            # QUESTION_SOURCE_CHAR_LIMIT).
            user_message = QUESTIONS_USER_TEMPLATE.format(
                content=content[: settings.question_source_char_limit]
            )
            response = await self.llm.create_message(
                system_prompt=QUESTIONS_SYSTEM_PROMPT,
                user_message=user_message,
                cache_system=True,
            )

            # The response should be a list of strings
            if isinstance(response, list):
                questions = [str(q) for q in response]
            elif isinstance(response, dict) and "questions" in response:
                questions = [str(q) for q in response["questions"]]
            else:
                questions = []

            # Filter out malformed entries
            questions = [q for q in questions if len(q) > 8]
            logger.debug(
                "Generated %d questions for '%s'",
                len(questions),
                technique_name,
            )
            return questions
        except Exception as e:  # noqa: BLE001 (LLM failure -> return no questions)
            logger.error("Failed to generate questions for '%s': %s", technique_name, e)
            return []

    async def generate_for_all(self, chunks_dir: str | Path | None = None) -> dict[str, list[str]]:
        """Generate questions for all technique files.

        Returns:
            dict mapping technique_name -> list of questions.
        """
        chunks_dir = Path(chunks_dir or settings.chunks_dir)
        questions_path = settings.vectors_dir / "questions.json"

        # Load existing questions if present (for resume)
        existing = {}
        if questions_path.exists():
            try:
                existing = json.loads(questions_path.read_text())
                logger.info("Loaded %d existing question sets", len(existing))
            except (json.JSONDecodeError, OSError):
                pass

        # Iterate over technique files
        output_dir = Path(settings.output_dir)
        if not output_dir.exists():
            logger.warning("No output dir found at %s", output_dir)
            return existing

        for file_path in sorted(output_dir.glob("*.md")):
            technique_name = file_path.stem
            if technique_name in existing:
                continue

            content = file_path.read_text(encoding="utf-8")
            questions = await self.generate_for_file(content, technique_name)
            if questions:
                existing[technique_name] = questions
                # Save incrementally for crash recovery
                self._save(existing, questions_path)

        return existing

    def _save(self, questions: dict[str, list[str]], path: Path) -> None:
        """Save questions incrementally."""
        path.parent.mkdir(parents=True, exist_ok=True)
        with open(path, "w") as f:
            json.dump(questions, f, indent=2, ensure_ascii=False)