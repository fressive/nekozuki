"""Markdown-aware text splitter for embedding chunking.

Rules:
1. Each H2 section (##) is a chunk boundary.
2. Never split inside a fenced code block (```...```).
3. Never split inside an inline code span.
4. If an H2 section is too long, split at paragraph boundaries with overlap.
"""

import re
from pathlib import Path

from src.config import settings
from src.models import Chunk


class MarkdownAwareTextSplitter:
    """Splits technique markdown files into chunks for embedding."""

    # Approximate chars per token for English text
    CHAR_TO_TOKEN_RATIO = 4.0

    def __init__(self, chunk_size: int | None = None, chunk_overlap: int | None = None):
        self.chunk_size = chunk_size or settings.chunk_size
        self.chunk_overlap = chunk_overlap or settings.chunk_overlap
        self.char_limit = int(self.chunk_size * self.CHAR_TO_TOKEN_RATIO)
        self.overlap_chars = int(self.chunk_overlap * self.CHAR_TO_TOKEN_RATIO)

    def split_technique_file(self, content: str, technique_name: str) -> list[Chunk]:
        """Split a single technique .md file into chunks.

        Each trick (H2 section) becomes one or more chunks.
        A global counter ensures every chunk in the file has a unique ID,
        even when the same section title appears multiple times.
        """
        sections = self._extract_h2_sections(content)

        chunks = []
        chunk_counter = 0
        for section_title, section_content in sections:
            section_chunks = self._split_section(
                section_content, technique_name, section_title, chunk_counter
            )
            chunk_counter += len(section_chunks)
            chunks.extend(section_chunks)

        return chunks

    def split_file(self, file_path: str | Path) -> list[Chunk]:
        """Split a technique markdown file on disk."""
        file_path = Path(file_path)
        content = file_path.read_text(encoding="utf-8")
        technique_name = file_path.stem
        return self.split_technique_file(content, technique_name)

    def _extract_h2_sections(self, content: str) -> list[tuple[str, str]]:
        """Split markdown content into H2-delimited sections.

        Code blocks are protected (replaced with placeholders) so that
        H2 markers inside code blocks don't create false boundaries.
        """
        protected, code_blocks = self._protect_code_blocks(content)

        sections = []
        lines = protected.split("\n")
        current_title = "overview"
        current_lines = []

        for line in lines:
            if line.startswith("## "):
                if current_lines:
                    sections.append((current_title, "\n".join(current_lines)))
                current_title = line[3:].strip()
                current_lines = [line]
            else:
                current_lines.append(line)

        if current_lines:
            sections.append((current_title, "\n".join(current_lines)))

        # Restore code blocks
        restored = []
        for title, sec_content in sections:
            for placeholder, block in code_blocks.items():
                sec_content = sec_content.replace(placeholder, block)
            restored.append((title, sec_content))

        # Skip the frontmatter/overview preamble (title + intro) if it has no
        # substantive content, but keep it if it contains useful description.
        return restored

    def _protect_code_blocks(self, content: str) -> tuple[str, dict[str, str]]:
        """Replace fenced code blocks with unique placeholders."""
        blocks = {}
        counter = [0]

        def replacer(match):
            placeholder = f"__CODEBLOCK_{counter[0]}__"
            counter[0] += 1
            blocks[placeholder] = match.group(0)
            return placeholder

        protected = re.sub(r"```[\s\S]*?```", replacer, content)
        return protected, blocks

    def _split_section(
        self, content: str, technique_name: str, section_title: str, start_idx: int = 0
    ) -> list[Chunk]:
        """Split a single H2 section into chunks (splitting at paragraph boundaries).

        ``start_idx`` is the global chunk counter so IDs stay unique file-wide.
        """
        if len(content) <= self.char_limit:
            return [
                Chunk(
                    chunk_id=self._make_chunk_id(technique_name, start_idx),
                    technique_name=technique_name,
                    section_title=section_title,
                    content=content.strip(),
                    token_count=self._estimate_tokens(content),
                )
            ]

        # Split at paragraph boundaries
        paragraphs = content.split("\n\n")
        chunks_text: list[str] = []
        current_chunk = ""

        for para in paragraphs:
            # If a single paragraph is longer than the limit, split on sentences
            if len(para) > self.char_limit:
                if current_chunk:
                    chunks_text.append(current_chunk)
                chunks_text.extend(self._split_long_paragraph(para))
                current_chunk = ""
                continue

            if len(current_chunk) + len(para) > self.char_limit and current_chunk:
                chunks_text.append(current_chunk)
                # Start new chunk with overlap from previous
                current_chunk = self._get_overlap(current_chunk) + "\n\n" + para
            else:
                current_chunk += ("\n\n" + para) if current_chunk else para

        if current_chunk:
            chunks_text.append(current_chunk)

        return [
            Chunk(
                chunk_id=self._make_chunk_id(technique_name, start_idx + i),
                technique_name=technique_name,
                section_title=section_title,
                content=chunk.strip(),
                token_count=self._estimate_tokens(chunk),
            )
            for i, chunk in enumerate(chunks_text)
        ]

    def _split_long_paragraph(self, para: str) -> list[str]:
        """Split an over-long paragraph at sentence boundaries."""
        sentences = re.split(r"(?<=[.!?])\s+", para)
        parts = []
        current = ""

        for sentence in sentences:
            if len(current) + len(sentence) > self.char_limit and current:
                parts.append(current)
                current = self._get_overlap(current) + " " + sentence
            else:
                current += (" " + sentence) if current else sentence

        if current:
            parts.append(current)

        return parts

    def _get_overlap(self, text: str) -> str:
        """Get the tail of text as overlap for the next chunk, breaking at a line."""
        if len(text) <= self.overlap_chars:
            return text

        cutoff = len(text) - self.overlap_chars
        # Prefer breaking at a newline or space
        last_newline = text.rfind("\n", cutoff)
        if last_newline > 0:
            return text[last_newline + 1 :]
        last_space = text.rfind(" ", cutoff)
        if last_space > 0:
            return text[last_space + 1 :]
        return text[-self.overlap_chars :]

    def _make_chunk_id(self, technique: str, chunk_idx: int) -> str:
        """Build a unique chunk ID using a file-wide chunk counter."""
        return f"{technique}__c{chunk_idx}"

    def _estimate_tokens(self, text: str) -> int:
        """Estimate token count from character count."""
        return max(1, len(text) // int(self.CHAR_TO_TOKEN_RATIO))


def split_all_technique_files(output_dir: str | Path | None = None) -> list[Chunk]:
    """Split all technique .md files in the output dir into chunks."""
    output_dir = Path(output_dir or settings.output_dir)
    if not output_dir.exists():
        return []

    splitter = MarkdownAwareTextSplitter()
    all_chunks = []
    for file_path in sorted(output_dir.glob("*.md")):
        chunks = splitter.split_file(file_path)
        all_chunks.extend(chunks)

    return all_chunks