"""Anthropic LLM client wrapper with prompt caching and streaming.

Requests stream (stream=True) and are consumed MANUALLY (iterating the raw
event stream) rather than via the SDK's `messages.stream()` context manager.
Two reasons:

1. The SDK raises "Streaming is required for operations that may take longer
   than 10 minutes" for non-streaming requests whose max_tokens exceed ~21k
   (or whose model is unknown to it), so we always stream.
2. The SDK's `messages.stream()` accumulator crashes with
   `AttributeError: 'NoneType' object has no attribute 'append'` on gateways
   (e.g. reasoning models) that emit `thinking` content blocks, so we walk the
   raw SSE events ourselves and keep only the text deltas.
"""

import json
import logging
import re

from anthropic import AsyncAnthropic

from src.config import settings

logger = logging.getLogger(__name__)


class LLMClient:
    """Async Anthropic client with prompt caching and manual streaming."""

    def __init__(self, api_key: str | None = None):
        self.api_key = api_key or settings.llm_api_key
        if not self.api_key:
            raise ValueError(
                "Anthropic API key is required. "
                "Set ANTHROPIC_API_KEY in .env or environment."
            )
        # BASE_URL lets you route requests through a gateway/proxy. Pass None
        # (the SDK default) when unset so an empty string never breaks the client.
        # The timeout guards against a hung gateway leaving a batch stuck forever.
        self.client = AsyncAnthropic(
            api_key=self.api_key,
            base_url=settings.llm_base_url or None,
            timeout=settings.llm_timeout,
        )
        self.model = settings.llm_model
        # max_tokens is the OUTPUT budget, not the context window. Guard against
        # absurd values (e.g. LLM_MAX_TOKENS=1000000) which make the Anthropic
        # SDK demand streaming for huge outputs AND exceed gateway output caps.
        self.max_tokens = min(settings.llm_max_tokens, 64000)
        if self.max_tokens != settings.llm_max_tokens:
            logger.warning(
                "Clamped LLM_MAX_TOKENS %d -> %d (output budget too large)",
                settings.llm_max_tokens, self.max_tokens,
            )
        self.temperature = settings.llm_temperature

    def _system_block(self, system_prompt: str, cache_system: bool) -> list[dict]:
        """Build the system block, optionally marked cacheable."""
        block = {"type": "text", "text": system_prompt}
        if cache_system:
            block["cache_control"] = {"type": "ephemeral"}
        return [block]

    async def _stream(self, system_block: list[dict], user_message: str) -> str:
        """Stream the response and accumulate text deltas (skipping thinking)."""
        stream = await self.client.messages.create(
            model=self.model,
            max_tokens=self.max_tokens,
            temperature=self.temperature,
            system=system_block,
            messages=[{"role": "user", "content": user_message}],
            stream=True,
        )
        chunks: list[str] = []
        async for event in stream:
            if event.type == "content_block_delta":
                delta = event.delta
                if getattr(delta, "type", "") == "text_delta":
                    chunks.append(delta.text)
        return "".join(chunks)

    async def _nonstream(self, system_block: list[dict], user_message: str) -> str:
        """Non-streaming fallback, extracting text from text content blocks.

        Only used if streaming is unavailable; also tolerates `thinking` blocks.
        """
        response = await self.client.messages.create(
            model=self.model,
            max_tokens=self.max_tokens,
            temperature=self.temperature,
            system=system_block,
            messages=[{"role": "user", "content": user_message}],
        )
        return "".join(
            block.text
            for block in response.content
            if getattr(block, "type", "") in ("text", "text_delta")
        )

    async def _complete(self, system_block: list[dict], user_message: str) -> str:
        """Stream a completion, falling back to non-streaming on failure."""
        try:
            return await self._stream(system_block, user_message)
        except Exception as e:  # noqa: BLE001 (fallback for unsupported streaming)
            logger.warning("Streaming failed, falling back to non-streaming: %s", e)
            return await self._nonstream(system_block, user_message)

    async def create_message(
        self,
        system_prompt: str,
        user_message: str,
        cache_system: bool = True,
    ) -> dict:
        """Create a message with prompt caching and return parsed JSON.

        Args:
            system_prompt: The system prompt text.
            user_message: The user message content.
            cache_system: Whether to mark the system prompt as cacheable.

        Returns:
            Parsed JSON response content. On parse failure, returns
            {"raw_content": <text>, "tricks": []} so callers can recover.
        """
        content = await self._complete(
            self._system_block(system_prompt, cache_system), user_message
        )
        return self._parse_json(content)

    async def create_message_raw(
        self,
        system_prompt: str,
        user_message: str,
        cache_system: bool = True,
    ) -> str:
        """Create a message and return raw text (no JSON parsing)."""
        return await self._complete(
            self._system_block(system_prompt, cache_system), user_message
        )

    @staticmethod
    def _parse_json(content: str) -> dict:
        """Parse JSON from LLM output, tolerating markdown fences and prose.

        Tries (in order): direct parse → fenced code block → bare array →
        bare object. Falls back to a raw_content dict on failure.
        """
        content = content.strip()
        if not content:
            return {"raw_content": "", "tricks": []}

        # Direct parse
        try:
            return json.loads(content)
        except json.JSONDecodeError:
            pass

        # Fenced code block
        fenced = re.search(r"```(?:json)?\s*\n?([\s\S]*?)\n?```", content)
        if fenced:
            try:
                return json.loads(fenced.group(1))
            except json.JSONDecodeError:
                pass

        # Bare JSON array or object anywhere in the text
        for pattern in (r"(\[[\s\S]*\])", r"(\{[\s\S]*\})"):
            match = re.search(pattern, content)
            if match:
                try:
                    return json.loads(match.group(1))
                except json.JSONDecodeError:
                    continue

        logger.warning("Failed to parse LLM response as JSON: %s", content[:200])
        return {"raw_content": content, "tricks": []}

    @property
    def token_count(self) -> int:
        """Return the approximate token usage.
        Note: This is a simplified estimate; actual tracking is done per-request.
        """
        return 0