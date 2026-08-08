"""Tests for Stage 2 trick dedup (lexical -> embedding -> LLM tiers)."""

import asyncio
import hashlib
import json
import re
import sys
from collections import defaultdict
from pathlib import Path

# Ensure project src is importable when running pytest from project root
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.models import Trick
from src.summarization.normalizer import build_deterministic_mapping
from src.summarization.trick_deduper import TrickDeduper


def _trick(technique, title, desc="", code=None, conf=0.9, sources=("https://w/1",)):
    return Trick(
        technique_name=technique,
        title=title,
        description=desc,
        key_code=code,
        confidence=conf,
        source_writeups=list(sources),
    )


def _groups(*tricks) -> dict[str, list[Trick]]:
    mapping = build_deterministic_mapping(t.technique_name for t in tricks)
    groups: dict[str, list[Trick]] = defaultdict(list)
    for t in tricks:
        groups[mapping[t.technique_name]].append(t)
    return dict(groups)


class FakeEmbedder:
    """Token-hash embeddings of the TITLE only.

    Cosine therefore tracks title-token overlap: identical titles -> 1.0
    (embed tier merges what the lexical tier missed because descriptions
    differ), near-identical titles land in the gray zone, unrelated titles ~0.
    """

    DIM = 24

    async def embed_texts(self, texts):
        vecs = []
        for text in texts:
            v = [0.0] * self.DIM
            # texts are "title. description"; embed the title only
            title = text.split(".", 1)[0]
            for tok in re.findall(r"\w+", title.lower()):
                h = int(hashlib.md5(tok.encode()).hexdigest()[:8], 16)
                v[h % self.DIM] += 1.0
            vecs.append(v)
        return vecs


class MockJudgeClient:
    """LLM stand-in returning canned verdicts per pair_id."""

    def __init__(self, verdicts: dict[int, str]):
        self.verdicts = verdicts

    async def create_message(self, system_prompt, user_message, cache_system=True):
        payload = json.loads(user_message)
        return {
            "verdicts": [
                {"pair_id": it["pair_id"], "verdict": self.verdicts.get(it["pair_id"], "keep"), "reason": "mock"}
                for it in payload
            ]
        }


# ---------------------------------------------------------------------------
# Tier 1: lexical
# ---------------------------------------------------------------------------


def test_lexical_cluster_is_deterministic_and_merges_near_dups():
    tricks = [
        _trick("sql_injection", "Authentication Bypass via Tautology SQL Injection",
               desc="Use OR 1=1 to bypass the login query's WHERE clause.", conf=0.9),
        # identical title+description token-sets (word-order variant) -> exact dup
        _trick("sql_injection", "Authentication Bypass via Tautology SQL Injection",
               desc="Login query's WHERE clause use OR 1=1 to bypass the", conf=0.85),
        _trick("sql_injection", "Blind SQLi via time delay",
               desc="Use SLEEP to infer data when no rows are returned."),
    ]
    deduper = TrickDeduper()
    result = deduper.build(_groups(*tricks))
    assert result.stats["lexical_clusters"] == 2  # two identical-text tricks merged

    # deterministic regardless of input order
    result2 = deduper.build(_groups(*reversed(tricks)))
    assert result.stats["lexical_clusters"] == result2.stats["lexical_clusters"]
    assert result.stats["lexical_merges"] == result2.stats["lexical_merges"]


def test_lexical_keeps_distinct_tricks_in_group():
    tricks = [
        _trick("heap_exploitation", "tcache poisoning via double free",
               desc="Overwrite the tcache entry to allocate an arbitrary address."),
        _trick("heap_exploitation", "fastbin attack to malloc_hook",
               desc="Corrupt the fastbin list to return a target chunk."),
    ]
    result = TrickDeduper().build(_groups(*tricks))
    assert result.stats["lexical_clusters"] == 2


# ---------------------------------------------------------------------------
# Candidate generation
# ---------------------------------------------------------------------------


def test_same_source_and_high_title_candidates():
    tricks = [
        # same writeup, same group -> candidate
        _trick("xss", "Stored XSS via comment field", desc="Persist script in a comment.",
               sources=("https://w/dup",)),
        _trick("xss", "Comment stored cross site scripting", desc="Stored XSS persists a payload in comments.",
               sources=("https://w/dup",)),
        # different writeups, high-overlap titles sharing a rare token -> candidate
        _trick("rsa_attacks", "Common modulus attack via GCD", desc="Factor n from shared modulus.",
               sources=("https://w/a",)),
        _trick("rsa_attacks", "Common modulus attack using GCD", desc="Recover private key via GCD of two messages.",
               sources=("https://w/b",)),
        # unrelated -> NOT a candidate
        _trick("crypto", "XOR single-byte key brute force", desc="Try 256 keys.", sources=("https://w/c",)),
    ]
    result = TrickDeduper().build(_groups(*tricks))
    ids = {t.title: i for i, t in enumerate(result.tricks)}
    cand = set(result.candidates)
    assert (ids["Stored XSS via comment field"], ids["Comment stored cross site scripting"]) in cand
    assert (ids["Common modulus attack via GCD"], ids["Common modulus attack using GCD"]) in cand
    assert len(cand) >= 2


# ---------------------------------------------------------------------------
# Tiers 2-3: embedding + LLM
# ---------------------------------------------------------------------------


def test_embed_tier_merges_same_title_different_desc():
    tricks = [
        _trick("xss", "Stored XSS via comment field",
               desc="Persist a script in the comment field so it runs for every visitor."),
        # identical title, different description: lexical combined < 0.72, so only
        # the embedding tier (title cosine 1.0) merges them
        _trick("xss", "Stored XSS via comment field",
               desc="A persistent script in comments executes whenever the page loads."),
        _trick("xss", "Reflected XSS via search parameter",
               desc="The search query is echoed into the page without escaping."),
    ]
    deduper = TrickDeduper(use_embed=True, use_llm=False, embedder=FakeEmbedder())
    result = deduper.build(_groups(*tricks))
    assert result.candidates, "expected same-title tricks to be candidates"
    merge, gray = asyncio.run(deduper.run_embed_tier(result))
    assert merge, "expected title-identical pair to merge via embedding"
    assert not gray
    final = deduper.final_merge(result, merge)
    titles = [t.title for group in final.values() for t in group]
    assert len(titles) == 2, f"expected stored pair merged, got {titles}"
    assert "Reflected XSS via search parameter" in titles


def test_llm_tier_judges_gray_zone():
    tricks = [
        _trick("rsa_attacks", "Common modulus attack via GCD",
               desc="Two messages share a modulus; GCD recovers the prime."),
        _trick("rsa_attacks", "Common modulus attack using GCD",
               desc="Factor the shared modulus from two related messages."),
        _trick("rsa_attacks", "Hastad broadcast attack",
               desc="Same message sent to many users with small exponent."),
    ]
    # FakeEmbedder title-cosine for the two near-identical common-modulus titles
    # lands in the gray zone (5/6 shared tokens -> ~0.83), not >= 0.90.
    deduper = TrickDeduper(use_embed=True, use_llm=True, embedder=FakeEmbedder(),
                           llm_client=MockJudgeClient({0: "merge"}))
    result = deduper.build(_groups(*tricks))
    merge_pairs, gray = asyncio.run(deduper.run_embed_tier(result))
    assert gray, "expected the common-modulus pair to be gray-zone"
    assert not merge_pairs
    llm_merges = asyncio.run(deduper.run_llm_tier(gray, result))
    final = deduper.final_merge(result, llm_merges)
    titles = sorted(t.title for group in final.values() for t in group)
    assert len(titles) == 2, f"expected LLM merge of the common-modulus pair, got {titles}"
    assert "Hastad broadcast attack" in titles


def test_final_merge_unions_sources():
    tricks = [
        _trick("path_traversal", "Dot-dot-slash file read", desc="Read /etc/passwd via ../..",
               code="../../etc/passwd", sources=("https://w/a",)),
        _trick("path_traversal", "Dot dot slash traversal", desc="Traverse to read arbitrary files.",
               code="../../etc/passwd", sources=("https://w/b",)),
    ]
    deduper = TrickDeduper(use_embed=False, use_llm=False)
    result = deduper.build(_groups(*tricks))
    assert result.stats["lexical_clusters"] == 1  # same key_code merges lexically
    final = deduper.final_merge(result, set())
    reps = [t for group in final.values() for t in group]
    assert len(reps) == 1
    assert sorted(reps[0].source_writeups) == ["https://w/a", "https://w/b"]
    assert reps[0].key_code == "../../etc/passwd"
