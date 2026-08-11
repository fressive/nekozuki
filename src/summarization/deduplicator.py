"""Post-extraction trick deduplication.

Collects all tricks grouped by canonical technique, merges near-duplicates,
and renders the final markdown technique files.
"""

import json
import logging
import re
from collections import defaultdict
from pathlib import Path
from typing import ClassVar

from pydantic import ValidationError

from src.config import settings
from src.models import Trick
from src.summarization.normalizer import TechniqueNormalizer

logger = logging.getLogger(__name__)

# Trick fields that must be lists. The LLM occasionally emits `null` for these
# (instead of an empty list); coerce to [] on load so Pydantic validation
# doesn't reject the whole trick.
_LIST_FIELDS = ("conditions", "implementation_steps", "detection_signs")


def _coerce_trick_dict(item: dict) -> dict:
    """Return a copy of ``item`` with list fields defaulted from ``None``."""
    for key in _LIST_FIELDS:
        if item.get(key) is None:
            item[key] = []
    return item


class TrickDeduplicator:
    """Deduplicate tricks and write final technique markdown files."""

    # Similarity threshold above which two tricks are considered duplicates
    SIMILARITY_THRESHOLD = 0.72
    # Two very short tricks only merge if their combined similarity is above
    # this (near-identical). Below it, token overlap on short text is likely
    # coincidental, so we keep them separate.
    SHORT_TEXT_STRONG_SIM = 0.95
    # Total (title+desc+steps+code) tokens below which a trick is "short".
    MIN_CONTENT_TOKENS = 12

    def __init__(self, normalizer: TechniqueNormalizer | None = None):
        self.normalizer = normalizer or TechniqueNormalizer()

    def load_tricks(self, tricks_path: str | Path | None = None) -> list[Trick]:
        """Load tricks from the accumulator JSONL file."""
        if tricks_path is None:
            tricks_path = settings.tricks_dir / "tricks.jsonl"

        tricks_path = Path(tricks_path)
        if not tricks_path.exists():
            logger.warning("No tricks file found at %s", tricks_path)
            return []

        tricks = []
        with open(tricks_path, "r") as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                try:
                    data = json.loads(line)
                    trick = Trick(**_coerce_trick_dict(data))
                    tricks.append(trick)
                except (json.JSONDecodeError, ValidationError, TypeError) as e:
                    logger.warning("Skipping malformed trick line: %s", e)

        logger.info("Loaded %d raw tricks", len(tricks))
        return tricks

    def load_tricks_from_file(self, tricks_path: str | Path | None = None) -> list[Trick]:
        """Load tricks from a tricks file (JSONL accumulator or JSON array).

        Handles both the JSONL accumulator written incrementally during
        extraction and the single JSON array written at completion.
        """
        if tricks_path is None:
            tricks_path = settings.tricks_dir / "tricks_all.json"

        tricks_path = Path(tricks_path)
        if not tricks_path.exists():
            return self.load_tricks(tricks_path)

        content = tricks_path.read_text(encoding="utf-8").strip()
        if not content:
            return []

        # Try as a single JSON array first
        try:
            data = json.loads(content)
            tricks = []
            for item in data:
                try:
                    tricks.append(Trick(**_coerce_trick_dict(item)))
                except (json.JSONDecodeError, ValidationError, TypeError) as e:
                    logger.warning("Skipping malformed trick: %s", e)
            logger.info("Loaded %d raw tricks from %s", len(tricks), tricks_path)
            return tricks
        except (json.JSONDecodeError, TypeError):
            pass  # Not a single JSON doc — fall through to line-by-line

        # Fallback: parse line by line (JSONL format)
        tricks = []
        for line in content.splitlines():
            line = line.strip()
            if not line:
                continue
            try:
                tricks.append(Trick(**_coerce_trick_dict(json.loads(line))))
            except (json.JSONDecodeError, ValidationError, TypeError) as e:
                logger.warning("Skipping malformed trick line: %s", e)
        logger.info("Loaded %d raw tricks from %s (JSONL)", len(tricks), tricks_path)
        return tricks

    def group_by_technique(self, tricks: list[Trick]) -> dict[str, list[Trick]]:
        """Group tricks by canonical technique name.

        Uses the deterministic batch mapping (sorted, frozen) instead of
        per-trick :meth:`TechniqueNormalizer.normalize`, which is order-
        dependent and can map the same name to different canonicals depending
        on processing order (see [[normalizer-order-dependence]]).
        """
        mapping = self.normalizer.normalize_batch(
            t.technique_name for t in tricks
        )
        groups: dict[str, list[Trick]] = defaultdict(list)
        for trick in tricks:
            canonical = mapping[trick.technique_name]
            groups[canonical].append(trick)
        return dict(groups)

    def deduplicate_group(self, tricks: list[Trick]) -> list[Trick]:
        """Deduplicate tricks within a single technique group (lexical tier).

        Clusters deterministically (see :meth:`cluster_group`) and merges each
        cluster into a representative trick.
        """
        if len(tricks) <= 1:
            return tricks
        clusters = self.cluster_group(tricks)
        logger.info(
            "Deduplicated %d tricks into %d clusters (threshold %.2f)",
            len(tricks),
            len(clusters),
            self.SIMILARITY_THRESHOLD,
        )
        return [self._merge_cluster([tricks[i] for i in cluster]) for cluster in clusters]

    def cluster_group(self, tricks: list[Trick]) -> list[list[int]]:
        """Greedy lexical clustering within a group, returning clusters of indices.

        Deterministic and order-independent:
        1. Sort by (confidence desc, content size desc, title, created_at) so
           the same input always yields the same clusters.
        2. Pre-compute token sets once per trick (avoids O(n²) re-tokenization).
        3. Each trick seeds a cluster; any trick sufficiently similar to an
           existing cluster's representative joins it.

        Similarity is containment-aware (a shorter trick whose text is a subset
        of a longer one merges upward), with a short-text guard so coincidental
        token overlap on tiny tricks does not merge distinct tricks.
        """
        if not tricks:
            return []
        if len(tricks) == 1:
            return [[0]]

        def sort_key(i: int) -> tuple:
            toks = self._tokenize_trick(tricks[i])
            return (
                -tricks[i].confidence,
                -sum(len(v) for v in toks.values()),
                tricks[i].title,
                tricks[i].created_at,
            )

        # Deterministic processing order (ties broken by title/created_at).
        order = sorted(range(len(tricks)), key=sort_key)
        cached = [self._tokenize_trick(tricks[i]) for i in order]

        clusters: list[list[int]] = [[order[0]]]
        # rep_positions: index into `order` of each cluster's representative.
        # rep_cluster[rep_pos] -> cluster index (mirrors rep_positions).
        rep_positions = [0]
        rep_cluster = {0: 0}

        # Token -> rep positions that contain it. A Jaccard-based merge needs
        # at least one shared token, so a candidate only has to be compared
        # against reps sharing one of its own tokens. This is EXACT (a rep with
        # zero token overlap can never reach the 0.72 threshold) and turns the
        # per-group O(n^2) into ~O(n * shared_rep_buckets).
        token_reps: dict[str, set[int]] = defaultdict(set)
        for token in self._all_tokens(cached[0]):
            token_reps[token].add(0)

        for pos in range(1, len(order)):
            i = order[pos]
            candidate_tokens = self._all_tokens(cached[pos])
            candidate_reps = set()
            for token in candidate_tokens:
                candidate_reps.update(token_reps.get(token, ()))
            placed = False
            for rep_pos in sorted(candidate_reps):
                if self._should_merge(cached[pos], cached[rep_pos]):
                    clusters[rep_cluster[rep_pos]].append(i)
                    placed = True
                    break
            if not placed:
                clusters.append([i])
                rep_positions.append(pos)
                rep_cluster[pos] = len(clusters) - 1
                for token in candidate_tokens:
                    token_reps[token].add(pos)

        return clusters

    @staticmethod
    def _all_tokens(tok: dict) -> set[str]:
        """Union of every token in a pre-tokenized trick (for the rep index)."""
        out: set[str] = set()
        for field in tok.values():
            out.update(field)
        return out

    @staticmethod
    def _content_len(trick: Trick) -> int:
        """Rough informative-content length (title + desc + steps)."""
        return (
            len(trick.title)
            + len(trick.description)
            + sum(len(s) for s in trick.implementation_steps)
        )

    def _should_merge(self, a: dict, b: dict) -> bool:
        """Whether two pre-tokenized tricks are duplicates (lexical tier)."""
        # Identical title+description text is the same trick no matter the
        # threshold. (Two prose-only tricks cap at 0.70 combined — title 1.0 +
        # desc 1.0 — which sits below SIMILARITY_THRESHOLD, so without this
        # short-circuit exact text duplicates with no steps/code never merge.)
        if (a["title"] or a["desc"]) and a["title"] == b["title"] and a["desc"] == b["desc"]:
            return True
        sim = self._cached_similarity(a, b)
        if sim < self.SIMILARITY_THRESHOLD:
            return False
        # A strong payload match overrides the short-text guard.
        if self._jaccard(a["code"], b["code"]) >= 0.6:
            return True
        a_len = sum(len(v) for v in a.values())
        b_len = sum(len(v) for v in b.values())
        return not (
            a_len < self.MIN_CONTENT_TOKENS
            and b_len < self.MIN_CONTENT_TOKENS
            and sim < self.SHORT_TEXT_STRONG_SIM
        )

    @staticmethod
    def _tokenize_trick(trick: Trick) -> dict:
        """Pre-compute token sets for a single trick.

        Returns a dict of frozensets for efficient similarity computation.
        """
        return {
            "title": frozenset(re.findall(r"\w+", trick.title.lower())),
            "desc": frozenset(re.findall(r"\w+", trick.description.lower())),
            "steps": frozenset(
                w for s in trick.implementation_steps
                for w in re.findall(r"\w+", s.lower())
            ),
            "code": frozenset(
                re.findall(r"\w+", (trick.key_code or "").lower())
            ),
        }

    @staticmethod
    def _jaccard(a: frozenset, b: frozenset) -> float:
        """Jaccard similarity between two sets."""
        if not a or not b:
            return 0.0
        return len(a & b) / len(a | b)

    @staticmethod
    def _containment(a: frozenset, b: frozenset) -> float:
        """How much of the smaller set is present in the larger (1.0 = subset).

        Kept for callers that need it, but NOT used in the merge similarity:
        containment over-scores short text (a one-token payload is a subset of
        every payload containing that token), so it over-merges.
        """
        if not a or not b:
            return 0.0
        return len(a & b) / min(len(a), len(b))

    def _cached_similarity(self, a: dict, b: dict) -> float:
        """Compute similarity from pre-computed token sets (no re-tokenization).

        Plain Jaccard on all fields. Containment-based scoring was tried and
        abandoned: a long representative's token set contains many short
        tricks' tokens, and a short key_code is a subset of many longer ones,
        so containment over-merges distinct tricks.
        """
        code_sim = self._jaccard(a["code"], b["code"])
        title_sim = self._jaccard(a["title"], b["title"])
        desc_sim = self._jaccard(a["desc"], b["desc"])
        steps_sim = self._jaccard(a["steps"], b["steps"])

        combined = 0.35 * title_sim + 0.35 * desc_sim + 0.2 * steps_sim + 0.1 * code_sim

        # If the payloads clearly match, treat as duplicates regardless of wording
        if code_sim >= 0.6:
            return max(code_sim, combined)

        return combined

    def _compute_similarity(self, a: Trick, b: Trick) -> float:
        """Compute a combined similarity score between two tricks.
        (Kept for backward compatibility — delegates to cached version.)
        """
        return self._cached_similarity(self._tokenize_trick(a), self._tokenize_trick(b))

    def _text_similarity(self, a: str, b: str) -> float:
        """Jaccard similarity on word tokens."""
        if not a and not b:
            return 1.0
        if not a or not b:
            return 0.0

        tokens_a = set(re.findall(r"\w+", a.lower()))
        tokens_b = set(re.findall(r"\w+", b.lower()))

        if not tokens_a or not tokens_b:
            return 0.0

        intersection = len(tokens_a & tokens_b)
        union = len(tokens_a | tokens_b)
        return intersection / union

    def _list_similarity(self, a: list[str], b: list[str]) -> float:
        """Jaccard similarity over list items (word-level)."""
        if not a and not b:
            return 1.0
        if not a or not b:
            return 0.0

        set_a = set()
        set_b = set()
        for item in a:
            set_a.update(re.findall(r"\w+", item.lower()))
        for item in b:
            set_b.update(re.findall(r"\w+", item.lower()))

        if not set_a or not set_b:
            return 0.0

        intersection = len(set_a & set_b)
        union = len(set_a | set_b)
        return intersection / union

    def _merge_cluster(self, cluster: list[Trick]) -> Trick:
        """Merge a cluster of similar tricks into a single representative.

        The representative carries the title/description/steps of the most
        informative member — highest confidence, then most content, then most
        sources — while conditions/codes/examples/signs/sources are unioned.
        """
        best = max(
            cluster,
            key=lambda t: (t.confidence, self._content_len(t), len(t.source_writeups)),
        )

        # Union of sources
        all_sources = list(dict.fromkeys(
            url for t in cluster for url in t.source_writeups
        ))

        # Union of code snippets
        all_codes = list(dict.fromkeys(
            t.key_code for t in cluster if t.key_code
        ))

        # Union of conditions
        all_conditions = list(dict.fromkeys(
            cond for t in cluster for cond in t.conditions
        ))

        # Union of detection signs
        all_signs = list(dict.fromkeys(
            sign for t in cluster for sign in t.detection_signs
        ))

        # Union of examples
        all_examples = list(dict.fromkeys(
            t.example for t in cluster if t.example
        ))

        # Union of original terms
        all_terms = list(dict.fromkeys(
            term for t in cluster for term in t.original_terms
        ))

        # Average confidence
        avg_confidence = sum(t.confidence for t in cluster) / len(cluster)

        return Trick(
            technique_name=best.technique_name,
            title=best.title,
            category=best.category,
            description=best.description,
            conditions=all_conditions,
            implementation_steps=best.implementation_steps,
            key_code="\n\n---\n\n".join(all_codes) if all_codes else None,
            example=all_examples[0] if all_examples else None,
            example_challenge=best.example_challenge,
            detection_signs=all_signs,
            confidence=avg_confidence,
            source_writeups=all_sources,
            original_terms=list(dict.fromkeys([best.technique_name] + all_terms)),
            created_at=min(t.created_at for t in cluster),
        )

    def deduplicate_all(
        self, tricks: list[Trick] | None = None, tricks_path: str | Path | None = None
    ) -> dict[str, list[Trick]]:
        """Deduplicate all tricks and return grouped by canonical technique."""
        if tricks is None:
            # Try the all-tricks file first, then the JSONL accumulator
            tricks = self.load_tricks_from_file(tricks_path)
            if not tricks:
                tricks = self.load_tricks(tricks_path)

        if not tricks:
            return {}

        groups = self.group_by_technique(tricks)
        result = {}
        for technique, group in groups.items():
            deduped = self.deduplicate_group(group)
            result[technique] = deduped
            logger.info(
                "Technique '%s': %d tricks after dedup (%d raw)",
                technique,
                len(deduped),
                len(group),
            )

        return result


class TechniqueFileWriter:
    """Writes deduplicated techniques to markdown files in the output dir."""

    # Canonical technique -> primary category (read-only lookup table)
    CATEGORY_MAP: ClassVar[dict[str, str]] = {
        # Canonical technique -> primary category
        "sql_injection": "web",
        "xss": "web",
        "server_side_template_injection": "web",
        "command_injection": "web",
        "path_traversal": "web",
        "ssrf": "web",
        "xxe": "web",
        "insecure_deserialization": "web",
        "type_juggling": "web",
        "jwt": "web",
        "prototype_pollution": "web",
        "race_condition": "web",
        "http_smuggling": "web",
        "nosql_injection": "web",
        "ldap_injection": "web",
        "file_upload": "web",
        "open_redirect": "web",
        "clickjacking": "web",
        "oauth": "web",
        "cors": "web",
        "csrf": "web",
        "graphql": "web",
        "websocket": "web",
        "php": "web",
        "buffer_overflow": "pwn",
        "heap_exploitation": "pwn",
        "format_string": "pwn",
        "shellcode": "pwn",
        "pwn": "pwn",
        "sandbox_escape": "misc",
        "reverse_engineering": "rev",
        "symbolic_execution": "rev",
        "crypto": "crypto",
        "rsa_attacks": "crypto",
        "padding_oracle": "crypto",
        "timing_attack": "crypto",
        "xor_cipher": "crypto",
        "hash_length_extension": "crypto",
        "stegano": "forensics",
        "forensics": "forensics",
        "memory_forensics": "forensics",
        "pcap_analysis": "forensics",
        "osint": "osint",
        "fuzzing": "pwn",
        "misc": "misc",
        "cloud": "cloud",
        "docker": "misc",
        "kubernetes": "misc",
        "mobile": "mobile",
        "browser": "pwn",
        "kernel": "pwn",
        "windows": "pwn",
        "linux": "pwn",
    }

    def __init__(self, output_dir: str | Path | None = None):
        self.output_dir = Path(output_dir or settings.output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)
        self._url_to_challenge: dict[str, tuple[str, str]] | None = None

    def _load_url_map(self) -> dict[str, tuple[str, str]]:
        """Build a mapping from writeup URL to (challenge_title, challenge_source).

        Used to derive example_challenge from source_writeups when the LLM
        didn't provide one (e.g. old data). Prefers the cleaned writeup cache
        (much smaller than the raw data.json) and falls back to data.json.
        """
        if self._url_to_challenge is not None:
            return self._url_to_challenge

        self._url_to_challenge = {}

        # Prefer the cleaned cache (small, already-parsed) over the 347MB raw file
        cache_path = Path(settings.cleaned_writeups_path)
        if cache_path.exists():
            try:
                with open(cache_path, "r", encoding="utf-8") as f:
                    for line in f:
                        line = line.strip()
                        if not line:
                            continue
                        item = json.loads(line)
                        url = item.get("url", "")
                        title = item.get("challenge_title", item.get("challenge_name", ""))
                        source = item.get("challenge_source", "")
                        if url and title:
                            self._url_to_challenge[url] = (title, source)
                return self._url_to_challenge
            except Exception:  # noqa: BLE001 (fall back to raw data)
                logger.warning("Failed to load URL map from cache %s", cache_path)

        data_path = Path(settings.data_path)
        if not data_path.exists():
            return self._url_to_challenge

        try:
            with open(data_path, "r", encoding="utf-8") as f:
                data = json.load(f)
            for item in data:
                url = item.get("url", "")
                title = item.get("challenge_title", item.get("challenge_name", ""))
                source = item.get("challenge_source", "")
                if url and title:
                    self._url_to_challenge[url] = (title, source)
        except Exception:  # noqa: BLE001 (non-critical, best-effort fallback)
            logger.warning("Failed to load URL→challenge map from %s", data_path)

        return self._url_to_challenge

    def _example_challenge_from_url(self, url: str) -> str | None:
        """Derive a human-readable example challenge name from a writeup URL."""
        mapping = self._load_url_map()
        pair = mapping.get(url)
        if pair:
            title, source = pair
            return f"{title} ({source})" if source else title
        return None

    def write_technique_file(self, technique_name: str, tricks: list[Trick]) -> Path:
        """Write a single technique file to the output directory."""
        if not tricks:
            return None

        # Sort tricks by confidence descending
        tricks_sorted = sorted(tricks, key=lambda t: t.confidence, reverse=True)

        # Determine category from the most common category in tricks
        categories = [t.category for t in tricks if t.category]
        category = self.CATEGORY_MAP.get(
            technique_name,
            max(set(categories), key=categories.count) if categories else "misc",
        )

        # Build description from the highest-confidence trick
        best = tricks_sorted[0]
        description = best.description or f"Techniques related to {technique_name.replace('_', ' ')}."

        # Build the markdown content
        lines = [
            "---",
            f"category: {category}",
            f"description: {description}",
            "---",
            f"# {self._to_title_case(technique_name)}",
            "",
        ]

        for trick in tricks_sorted:
            lines.extend(self._render_trick(trick))
            lines.append("")

        # Deduplicate consecutive blank lines
        content = "\n".join(lines)
        content = re.sub(r"\n{3,}", "\n\n", content).strip() + "\n"

        output_file = self.output_dir / f"{technique_name}.md"
        output_file.write_text(content, encoding="utf-8")
        logger.info("Wrote %d tricks to %s", len(tricks), output_file)
        return output_file

    def _render_trick(self, trick: Trick) -> list[str]:
        """Render a single trick as markdown H2 section.

        Follows the canonical output format:
            ## Title

            Description: ...
            Conditions: ...; ...;

            <body/prose>

            Example:
            ```
            ...
            ```
        """
        lines = [f"## {trick.title}"]

        if trick.description:
            lines.append("")
            lines.append(f"Description: {trick.description}")

        if trick.conditions:
            lines.append("")
            lines.append(f"Conditions: {'; '.join(trick.conditions)};")

        if trick.implementation_steps:
            lines.append("")
            lines.append("Implementation:")
            for step in trick.implementation_steps:
                lines.append(f"- {step}")

        if trick.key_code:
            lines.append("")
            lines.append("Key code/payload:")
            lines.append("")
            lines.append("```")
            lines.append(trick.key_code)
            lines.append("```")

        if trick.example:
            lines.append("")
            lines.append("Example:")
            lines.append("")
            lines.append("```")
            lines.append(trick.example)
            lines.append("```")

        if trick.detection_signs:
            lines.append("")
            lines.append("Detection signs:")
            for sign in trick.detection_signs:
                lines.append(f"- {sign}")

        # Derive example_challenge from source_writeups if the LLM didn't
        # provide one (e.g. old data).  Pick the first writeup URL and look up
        # the challenge name from data.json.
        example_challenge = trick.example_challenge
        if not example_challenge and trick.source_writeups:
            example_challenge = self._example_challenge_from_url(trick.source_writeups[0])

        if example_challenge:
            lines.append("")
            lines.append(f"Example challenge: {example_challenge}")

        lines.append("")
        return lines

    def _to_title_case(self, name: str) -> str:
        """Convert snake_case to Title Case, preserving common acronyms."""
        acronyms = {
            "sql", "xss", "ssti", "rsa", "xxe", "ssrf", "lfi", "rfi",
            "idor", "jwt", "cors", "csrf", "xpath", "ldap", "http",
            "dns", "xml", "cve", "osint", "lsb", "aes", "api", "ftp",
            "smtp", "tcp", "udp", "url", "cli", "gui", "json", "yaml",
            "wasm", "v8", "ptrace", "gdb", "ida", "ntfs", "iis",
        }
        words = []
        for word in name.split("_"):
            if word.lower() in acronyms:
                words.append(word.upper())
            else:
                words.append(word.capitalize())
        return " ".join(words)

    def write_all(self, grouped: dict[str, list[Trick]]) -> list[Path]:
        """Write all technique files and return the written paths."""
        written = []
        for technique_name, tricks in grouped.items():
            path = self.write_technique_file(technique_name, tricks)
            if path:
                written.append(path)
        logger.info("Wrote %d technique files", len(written))
        return written


def run_deduplication(tricks_path: str | Path | None = None, output_dir: str | Path | None = None) -> list[Path]:
    """Convenience function: load, deduplicate, and write technique files.

    Groups tricks deterministically (order-independent), merges near-duplicate
    canonical techniques (content + name-similarity tiers, no LLM), then
    deduplicates tricks within each merged group.
    """
    normalizer = TechniqueNormalizer()
    dedup = TrickDeduplicator(normalizer)
    writer = TechniqueFileWriter(output_dir)

    tricks = dedup.load_tricks_from_file(tricks_path)
    if not tricks:
        tricks = dedup.load_tricks(tricks_path)

    groups = dedup.group_by_technique(tricks)

    # Deterministic technique-level merging (no LLM in the hot path; richer
    # merges come from `nekozuki dedup-techniques`, which persists aliases that
    # are loaded back here via the mapping file).
    from src.summarization.technique_merger import TechniqueMerger, merged_groups

    merger = TechniqueMerger(use_llm=False)
    result = merger.build_merged_mapping(groups)
    groups = merged_groups(groups, result.merged_mapping)

    deduped = {}
    for technique, group in groups.items():
        deduped[technique] = dedup.deduplicate_group(group)
    return writer.write_all(deduped)