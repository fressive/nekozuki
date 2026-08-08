"""Three-tier trick deduplication: lexical -> embedding -> LLM.

Tier 1 (lexical) is deterministic and free: tricks are greedily clustered
within each technique group using token Jaccard, exactly as the pipeline's
``TrickDeduplicator`` did, but with a deterministic sort and a short-text guard.
This is deliberately conservative — containment-based scoring over-merges
(a long representative's token set contains many short tricks' tokens), so it
is avoided.

Tiers 2-3 are opt-in and cost API tokens:
- Tier 2 (embedding): candidate pairs are embedded (title + description) and
  merged when cosine >= ``embed_threshold`` (0.90). Candidate pairs are bounded
  to same-source-within-group, high title-Jaccard within group, and identical
  cross-group titles — a full pairwise comparison would be 600k+ pairs.
- Tier 3 (LLM): pairs whose cosine falls in the gray zone (0.75-0.90) are
  judged by an LLM (static cacheable system prompt, pair data in the user
  message), so genuinely-same tricks merge while distinct ones stay separate.

The expensive tiers run only via ``nekozuki dedup-tricks``; the summarize path
(`run_deduplication`) uses tier 1 alone.
"""

import asyncio
import json
import logging
import re
from collections import Counter, defaultdict
from dataclasses import dataclass, field

import numpy as np

from src.config import settings
from src.models import Trick
from src.summarization.deduplicator import TrickDeduplicator

logger = logging.getLogger(__name__)


@dataclass
class TrickDedupResult:
    """Flattened view of the corpus with lexical clusters and candidates."""

    #: id -> trick (flattened across all technique groups)
    tricks: list[Trick] = field(default_factory=list)
    #: id -> canonical technique group
    group_of: dict[int, str] = field(default_factory=dict)
    #: lexical clusters (lists of trick ids) — tier 1 output
    lexical_clusters: list[list[int]] = field(default_factory=list)
    #: bounded candidate pairs for the embedding tier
    candidates: list[tuple[int, int]] = field(default_factory=list)
    stats: dict = field(default_factory=dict)


class TrickDeduper:
    """Stage 2 trick dedup across the whole corpus (within merged groups)."""

    EMBED_DUP_THRESHOLD = 0.90
    EMBED_GRAY_LO = 0.75
    TITLE_JACCARD_CANDIDATE = 0.5
    MIN_CANDIDATE_TOKENS = 5
    # Only block on title tokens this rare (globally) — specific words like
    # "tcache" or "spectrogram" signal related tricks; "bypass"/"injection"
    # appear in thousands of tricks and would make the pair space explode.
    TITLE_TOKEN_DF_CAP = 100
    LLM_BATCH_SIZE = 20
    # Embedding gateway (EMBEDDING_BASE_URL) rejects requests whose serialized
    # payload exceeds ~4MB; with EMBEDDING_DIMENSIONS=4096 that is ~65 texts, so
    # chunk well under it and fan out with limited concurrency.
    EMBED_CHUNK_SIZE = 40
    EMBED_CONCURRENCY = 4

    def __init__(
        self,
        use_embed: bool = True,
        use_llm: bool = True,
        embed_threshold: float | None = None,
        gray_lo: float | None = None,
        embedder=None,
        llm_client=None,
    ):
        self.lexical = TrickDeduplicator()
        self.use_embed = use_embed
        self.use_llm = use_llm
        self.embed_threshold = (
            embed_threshold if embed_threshold is not None else self.EMBED_DUP_THRESHOLD
        )
        self.gray_lo = gray_lo if gray_lo is not None else self.EMBED_GRAY_LO
        self.embedder = embedder  # async embed_texts(texts) -> list[list[float]]
        self.llm_client = llm_client  # create_message(system, user, cache_system)
        self._llm_kept: list[tuple[int, int]] = []

    # -- tier 1 ---------------------------------------------------------------

    def build(
        self, groups: dict[str, list[Trick]], progress=None
    ) -> TrickDedupResult:
        """Flatten groups, run lexical clustering, and generate candidates.

        ``progress`` (optional) is a callable ``progress(increment)`` invoked
        once per technique group, so a UI can show tier-1 progress.
        """
        tricks: list[Trick] = []
        group_of: dict[int, str] = {}
        clusters: list[list[int]] = []
        cluster_of: dict[int, int] = {}

        for name, ts in groups.items():
            base = len(tricks)
            tricks.extend(ts)
            for k in range(len(ts)):
                group_of[base + k] = name
            for gc in self.lexical.cluster_group(ts):  # indices within group
                cid = [base + i for i in gc]
                clusters.append(cid)
                for i in cid:
                    cluster_of[i] = len(clusters) - 1
            if progress:
                progress(1)

        result = TrickDedupResult(
            tricks=tricks,
            group_of=group_of,
            lexical_clusters=clusters,
        )
        result.candidates = self._find_candidates(result, cluster_of)
        result.stats = {
            "tricks": len(tricks),
            "lexical_clusters": len(clusters),
            "lexical_merges": len(tricks) - len(clusters),
            "candidates": len(result.candidates),
        }
        return result

    def _find_candidates(
        self, result: TrickDedupResult, cluster_of: dict[int, int]
    ) -> list[tuple[int, int]]:
        """Bounded candidate pairs for the embedding tier.

        Three sources, all cheap:
        1. same source writeup + same group (a writeup often yields the same
           trick twice under slightly different wording)
        2. title+description Jaccard >= 0.5 within a group (cross-writeup dupes)
        3. identical title token-sets across DIFFERENT groups

        Pairs already in the same lexical cluster are skipped.
        """
        tricks = result.tricks
        cand: set[tuple[int, int]] = set()

        def _in_different_cluster(a: int, b: int) -> bool:
            return cluster_of.get(a) != cluster_of.get(b)

        # 1. same-source within group
        src_idx: dict[tuple[str, str], list[int]] = defaultdict(list)
        for i, t in enumerate(tricks):
            for url in t.source_writeups:
                src_idx[(result.group_of[i], url)].append(i)
        for idxs in src_idx.values():
            for a in range(len(idxs)):
                for b in range(a + 1, len(idxs)):
                    if _in_different_cluster(idxs[a], idxs[b]):
                        cand.add((min(idxs[a], idxs[b]), max(idxs[a], idxs[b])))

        # 2. high title-token Jaccard within group — blocked on RARE title
        #    tokens only, then Jaccard-filtered (cross-writeup near-dups)
        title_df = Counter()
        for t in tricks:
            title_df.update(self._significant_title_tokens(t.title))
        title_idx: dict[tuple[str, str], list[int]] = defaultdict(list)
        for i, t in enumerate(tricks):
            for w in self._significant_title_tokens(t.title):
                if title_df[w] <= self.TITLE_TOKEN_DF_CAP:
                    title_idx[(result.group_of[i], w)].append(i)
        seen: set[tuple[int, int]] = set()
        for idxs in title_idx.values():
            for a in range(len(idxs)):
                for b in range(a + 1, len(idxs)):
                    ia, ib = idxs[a], idxs[b]
                    if not _in_different_cluster(ia, ib):
                        continue
                    pair = (min(ia, ib), max(ia, ib))
                    if pair in seen:
                        continue
                    seen.add(pair)
                    if self._title_jaccard(tricks, ia, ib) >= self.TITLE_JACCARD_CANDIDATE:
                        cand.add(pair)

        # 3. identical title token-sets across different groups
        title_set: dict[frozenset, list[int]] = defaultdict(list)
        for i, t in enumerate(tricks):
            title_set[frozenset(re.findall(r"\w+", t.title.lower()))].append(i)
        for idxs in title_set.values():
            for a in range(len(idxs)):
                for b in range(a + 1, len(idxs)):
                    if (
                        result.group_of[idxs[a]] != result.group_of[idxs[b]]
                        and _in_different_cluster(idxs[a], idxs[b])
                    ):
                        cand.add((min(idxs[a], idxs[b]), max(idxs[a], idxs[b])))

        return sorted(cand)

    @staticmethod
    def _significant_title_tokens(title: str) -> set[str]:
        """Title words long enough to be meaningful as a blocking key."""
        return {w for w in re.findall(r"\w+", title.lower()) if len(w) >= 4}

    @staticmethod
    def _title_jaccard(tricks: list[Trick], a: int, b: int) -> float:
        """Jaccard over TITLE tokens only (the strongest same-trick signal).

        Descriptions are excluded: two variants of one trick often share a
        title but word their descriptions differently, and mixing desc in
        dilutes the title overlap below the candidate gate.
        """
        ta = frozenset(re.findall(r"\w+", tricks[a].title.lower()))
        tb = frozenset(re.findall(r"\w+", tricks[b].title.lower()))
        if len(ta) < 3 or len(tb) < 3:
            return 0.0
        if not ta or not tb:
            return 0.0
        return len(ta & tb) / len(ta | tb)

    @staticmethod
    def _embed_text(trick: Trick) -> str:
        """The text embedded for semantic similarity (title + description)."""
        text = f"{trick.title}. {trick.description}".strip()
        return text[:800]

    # -- tier 2: embedding ----------------------------------------------------

    async def run_embed_tier(
        self, result: TrickDedupResult, progress=None
    ) -> tuple[set[frozenset], list[tuple[int, int]]]:
        """Embed candidate pairs; return (merge_pairs, gray_zone_pairs).

        Merge pairs have cosine >= threshold; gray-zone pairs (cosine in
        [gray_lo, threshold)) go to the LLM tier; everything else stays.
        ``progress`` (optional) is invoked once per completed request chunk.
        """
        if not self.use_embed or not self.embedder or not result.candidates:
            gray = list(result.candidates) if self.use_llm else []
            return set(), gray

        unique = sorted({i for pair in result.candidates for i in pair})
        texts = [self._embed_text(result.tricks[i]) for i in unique]
        embedder = self.embedder

        # Chunk requests (the embedding gateway's ~4MB limit at 4096 dims is
        # ~65 texts) and fan out concurrently; store as float32 to keep memory
        # sane (15k x 4096 Python floats would be ~500MB).
        sem = asyncio.Semaphore(self.EMBED_CONCURRENCY)

        async def _embed_chunk(chunk: list[str]) -> list[list[float]]:
            async with sem:
                return await embedder.embed_texts(chunk)

        tasks = [
            _embed_chunk(texts[i : i + self.EMBED_CHUNK_SIZE])
            for i in range(0, len(texts), self.EMBED_CHUNK_SIZE)
        ]
        chunk_results: list[list[list[float]]] = []
        for fut in asyncio.as_completed(tasks):
            chunk_results.append(await fut)
            if progress:
                progress(1)

        flat: list[list[float]] = [v for r in chunk_results for v in r]
        matrix = np.asarray(flat, dtype=np.float32)
        norms = np.linalg.norm(matrix, axis=1)
        emb_by_id = {tid: i for i, tid in enumerate(unique)}

        merge: set[frozenset] = set()
        gray: list[tuple[int, int]] = []
        for a, b in result.candidates:
            ia, ib = emb_by_id[a], emb_by_id[b]
            denom = norms[ia] * norms[ib]
            c = float(matrix[ia] @ matrix[ib] / denom) if denom else 0.0
            if c >= self.embed_threshold:
                merge.add(frozenset((a, b)))
            elif c >= self.gray_lo:
                gray.append((a, b))
        logger.info(
            "Embedding tier: %d candidate pairs -> %d merge, %d gray, %d keep",
            len(result.candidates), len(merge), len(gray),
            len(result.candidates) - len(merge) - len(gray),
        )
        return merge, gray

    # -- tier 3: LLM ----------------------------------------------------------

    async def run_llm_tier(
        self, gray_pairs: list[tuple[int, int]], result: TrickDedupResult, progress=None
    ) -> set[frozenset]:
        """LLM judge for gray-zone pairs; returns the merge set.

        Batches run with bounded concurrency (``llm_max_concurrency``).
        ``progress`` (optional) is invoked once per completed batch.
        """
        if not self.use_llm or not self.llm_client or not gray_pairs:
            return set()

        llm_client = self.llm_client
        accepted: set[frozenset] = set()
        self._llm_kept = []
        batches = [
            gray_pairs[i : i + self.LLM_BATCH_SIZE]
            for i in range(0, len(gray_pairs), self.LLM_BATCH_SIZE)
        ]
        sem = asyncio.Semaphore(settings.llm_max_concurrency)

        async def _judge(batch: list[tuple[int, int]]) -> tuple[set[frozenset], list]:
            async with sem:
                user_message = self._build_judge_message(batch, result)
                try:
                    parsed = await llm_client.create_message(
                        _TRICK_JUDGE_PROMPT, user_message, cache_system=True
                    )
                except Exception as e:  # noqa: BLE001 (LLM failure -> keep separate)
                    logger.warning("LLM trick-judge batch failed, keeping separate: %s", e)
                    return set(), list(batch)
                verdicts = self._extract_verdicts(parsed)
                ok: set[frozenset] = set()
                kept: list = []
                for pid, (a, b) in enumerate(batch):
                    if verdicts.get(pid) == "merge":
                        ok.add(frozenset((a, b)))
                    else:
                        kept.append((a, b))
                return ok, kept

        tasks = [_judge(b) for b in batches]
        for fut in asyncio.as_completed(tasks):
            ok, kept = await fut
            accepted.update(ok)
            self._llm_kept.extend(kept)
            if progress:
                progress(1)

        logger.info("LLM tier: %d gray pairs -> %d merge, %d keep",
                    len(gray_pairs), len(accepted), len(self._llm_kept))
        return accepted

    def _build_judge_message(
        self, batch: list[tuple[int, int]], result: TrickDedupResult
    ) -> str:
        payload = []
        for pid, (a, b) in enumerate(batch):
            payload.append(
                {
                    "pair_id": pid,
                    "trick_a": self._trick_evidence(result.tricks[a]),
                    "trick_b": self._trick_evidence(result.tricks[b]),
                }
            )
        return json.dumps(payload)

    @staticmethod
    def _trick_evidence(trick: Trick) -> dict:
        return {
            "technique": trick.technique_name,
            "title": trick.title,
            "description": (trick.description or "")[:300],
            "key_code": (trick.key_code or "")[:200],
        }

    @staticmethod
    def _extract_verdicts(parsed) -> dict[int, str]:
        data = parsed
        if isinstance(parsed, dict):
            for key in ("verdicts", "results", "pairs"):
                if isinstance(parsed.get(key), list):
                    data = parsed[key]
                    break
            else:
                return {}
        if not isinstance(data, list):
            return {}
        verdicts: dict[int, str] = {}
        for item in data:
            if not isinstance(item, dict):
                continue
            pid = item.get("pair_id", item.get("id"))
            verdict = str(item.get("verdict", "")).lower()
            if isinstance(pid, int) and verdict in ("merge", "keep"):
                verdicts[pid] = verdict
        return verdicts

    # -- apply ----------------------------------------------------------------

    def final_merge(
        self, result: TrickDedupResult, extra_merges: set[frozenset]
    ) -> dict[str, list[Trick]]:
        """Union lexical clusters + extra merges -> merged per-group trick lists.

        Each merged component becomes one representative trick (union of
        sources/codes/conditions/signs), routed to the group of its best member.
        """
        parent = {i: i for i in range(len(result.tricks))}

        def find(x: int) -> int:
            while parent[x] != x:
                parent[x] = parent[parent[x]]
                x = parent[x]
            return x

        def union(a: int, b: int) -> None:
            ra, rb = find(a), find(b)
            if ra != rb:
                parent[rb] = ra

        for cluster in result.lexical_clusters:
            for i in cluster[1:]:
                union(cluster[0], i)
        for fs in extra_merges:
            a, b = tuple(fs)
            union(a, b)

        components: dict[int, list[int]] = defaultdict(list)
        for i in range(len(result.tricks)):
            components[find(i)].append(i)

        merged: dict[str, list[Trick]] = defaultdict(list)
        for comp in components.values():
            best = max(
                comp,
                key=lambda i: (
                    result.tricks[i].confidence,
                    self.lexical._content_len(result.tricks[i]),
                    len(result.tricks[i].source_writeups),
                ),
            )
            rep = self.lexical._merge_cluster([result.tricks[i] for i in comp])
            merged[result.group_of[best]].append(rep)
        return dict(merged)


# Static, >= 1024 tokens so every batch hits the same prompt-cache entry.
_TRICK_JUDGE_PROMPT = """\
You are an expert CTF (Capture The Flag) trick deduplication judge. Your job is
to decide, for each pair of TRICKS, whether they are the SAME trick (and should
be merged into one entry) or DISTINCT tricks (and should both stay).

CONTEXT — how tricks are produced and why we dedup:
A trick is a short reusable note describing ONE concrete attack move: its title,
a description of how and when to use it, and optionally a payload. Tricks are
extracted from ~40k challenge writeups by an LLM, so the SAME trick frequently
appears several times under slightly different wording — the same SQL injection
login bypass might be written once as "Authentication Bypass via Tautology" and
again as "SQLi OR 1=1 login bypass". These duplicates pollute retrieval: the
search index fills with near-identical chunks and a query returns the same
technique five times. Merging identical tricks fixes that. But two tricks that
are genuinely DIFFERENT attack moves must stay separate even when they share
words or belong to the same broad technique.

MERGE when the pair is the same attack move:
- Same technique, same payload, only the wording differs: "Authentication Bypass
  via Tautology SQL Injection (' or 1=1)" vs "Login bypass with OR 1=1".
- The same trick described at different lengths: a terse one-liner vs the same
  move explained in detail.
- Trivial rephrasing, e.g. "Use After Free overwrite" vs "UAF overwrite", or
  "Heap tcache poisoning via double free" vs "Double free -> tcache poisoning".
- The same trick with the same key_code/payload string, even if prose differs.

KEEP SEPARATE when the two tricks are different moves, even if closely related:
- Different attack vectors: "Boolean blind SQLi via response time" vs "UNION
  based SQLi"; "tcache poisoning" vs "fastbin attack" (different heap bins).
- Different stages of an exploit: "leaking a libc address" vs "overwriting
  __free_hook" (both heap exploitation, different moves).
- Different targets: "XSS in attribute context" vs "XSS in script context".
- Same technique, different payload/purpose: "SQLi filter bypass via comments"
  vs "SQLi data exfiltration via UNION".
- A general note vs a specific instance that deserves to stay its own entry:
  "command injection detection" vs "command injection via mail header".

SIGNALS, strongest first:
1. Identical or near-identical key_code payloads are very strong evidence of the
   same trick — unless the payload is a generic one-liner used by many tricks
   (e.g. "cat /etc/passwd"), in which case weigh the prose more.
2. Highly overlapping title AND description with the same technique name is
   strong evidence.
3. Same technique name alone is weak evidence — a technique group legitimately
   contains many distinct tricks.
4. If only the technique name and a generic word ("bypass", "injection",
   "exploit") overlap, keep them separate.

The input is a JSON array. Each element has an integer "pair_id", and two
objects "trick_a" and "trick_b", each with fields: "technique", "title",
"description" (truncated), and "key_code" (truncated).

Output STRICTLY as a single JSON object, never anything else:
{"verdicts": [{"pair_id": 0, "verdict": "merge"|"keep", "reason": "short justification"}]}
- "verdict" must be exactly the lowercase string "merge" or "keep".
- "pair_id" must match the input id, and every input pair must appear exactly
  once.
- "reason" should be 1-2 short clauses naming the deciding signal, e.g. "same
  payload, same technique" or "different attack vectors (boolean vs union)".
- Do not output markdown fences, prose, or anything outside the JSON.
"""
