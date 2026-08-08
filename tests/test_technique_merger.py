"""Tests for technique-level dedup (Stage 1: technique_merger + normalizer)."""

import asyncio
import json
import random
import sys
from collections import defaultdict
from pathlib import Path

# Ensure project src is importable when running pytest from project root
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.models import Trick
from src.summarization.deduplicator import run_deduplication
from src.summarization.normalizer import build_deterministic_mapping
from src.summarization.technique_merger import (
    TechniqueMerger,
    run_llm_merges,
)


def _trick(technique, title, conf=0.9, desc="", key_code=None):
    return Trick(
        technique_name=technique,
        title=title,
        description=desc,
        key_code=key_code,
        confidence=conf,
        source_writeups=[f"https://example.com/{technique}"],
    )


def _group(tricks: list[Trick]) -> dict[str, list[Trick]]:
    """Group tricks by the deterministic canonical mapping."""
    mapping = build_deterministic_mapping(t.technique_name for t in tricks)
    groups: dict[str, list[Trick]] = defaultdict(list)
    for t in tricks:
        groups[mapping[t.technique_name]].append(t)
    return groups


def _build(groups, use_llm=False):
    merger = TechniqueMerger(use_llm=use_llm)
    result = merger.build_merged_mapping(groups)
    return merger, result


# ---------------------------------------------------------------------------
# Stage 1A: deterministic canonicalization
# ---------------------------------------------------------------------------


def test_deterministic_mapping_is_order_independent():
    names = [
        "blind_sql_injection",
        "sql_injection",
        "json_injection",
        "jsonp_injection",
        "php_disable_functions_bypass",
        "php_disabled_functions_bypass",
    ]
    m1 = build_deterministic_mapping(names)
    rng = random.Random(0)
    shuffled = names[:]
    rng.shuffle(shuffled)
    m2 = build_deterministic_mapping(shuffled)
    assert m1 == m2

    # Curated aliases still collapse ...
    assert m1["blind_sql_injection"] == "sql_injection"
    # ... but the live normalizer's fuzzy garbage (json_injection ->
    # sql_injection) is NOT reproduced by the deterministic map.
    assert m1["json_injection"] == "json_injection"
    assert m1["jsonp_injection"] == "jsonp_injection"


# ---------------------------------------------------------------------------
# Stage 1B: name-similarity auto-merge
# ---------------------------------------------------------------------------


def test_auto_merge_name_variants():
    tricks = [
        _trick("php_disable_functions_bypass", "Bypass disable_functions via regex", conf=0.9),
        _trick("php_disable_functions_bypass", "Bypass disable_functions via iconv", conf=0.9),
        _trick("php_disable_functions_bypass", "Bypass disable_functions via LD_PRELOAD", conf=0.9),
        _trick("php_disabled_functions_bypass", "Bypass disabled functions", conf=0.8),
        _trick("disable_functions_bypass", "Bypass disable_functions", conf=0.7),
    ]
    groups = _group(tricks)
    _, result = _build(groups)
    fm = result.merged_mapping
    # Representative is the most common spelling.
    assert fm["php_disabled_functions_bypass"] == "php_disable_functions_bypass"
    assert fm["disable_functions_bypass"] == "php_disable_functions_bypass"
    assert fm["php_disable_functions_bypass"] == "php_disable_functions_bypass"


# ---------------------------------------------------------------------------
# Stage 1B: content-merge of name-different synonyms
# ---------------------------------------------------------------------------


def test_content_merge_different_names():
    tricks = [
        _trick("type_juggling", "Magic hash comparison bypass via 0e"),
        _trick("type_juggling", "Loose comparison strict vs loose"),
        _trick("php_loose_comparison", "Magic hash comparison bypass via 0e"),
        _trick("php_loose_comparison", "Loose comparison strict vs loose"),
    ]
    groups = _group(tricks)
    _, result = _build(groups)
    fm = result.merged_mapping
    assert fm["php_loose_comparison"] == "type_juggling"


def test_content_merge_requires_two_shared_titles():
    # A single coincidental shared title must NOT merge unrelated techniques.
    tricks = [
        _trick("sql_injection", "Extract data via boolean oracle"),
        _trick("path_traversal", "Extract data via boolean oracle"),
    ]
    groups = _group(tricks)
    _, result = _build(groups)
    fm = result.merged_mapping
    assert fm["sql_injection"] == "sql_injection"
    assert fm["path_traversal"] == "path_traversal"


# ---------------------------------------------------------------------------
# Stage 1B: conservative keep-separate
# ---------------------------------------------------------------------------


def test_keep_distinct_similar_names():
    tricks = [
        _trick("aslr_bypass", "ASLR bypass via libc info leak"),
        _trick("kaslr_bypass", "KASLR bypass via kernel info leak"),
        _trick("json_injection", "JSON injection in object keys"),
        _trick("jsonp_injection", "JSONP callback injection"),
    ]
    groups = _group(tricks)
    _, result = _build(groups, use_llm=False)
    fm = result.merged_mapping
    assert fm["aslr_bypass"] == "aslr_bypass"
    assert fm["kaslr_bypass"] == "kaslr_bypass"
    assert fm["json_injection"] == "json_injection"
    assert fm["jsonp_injection"] == "jsonp_injection"


def test_antipodal_swap_never_auto_merges():
    # Overflow vs underflow share 3/4 tokens (Dice 0.75) but are distinct
    # vulnerabilities — must not be merged by name similarity alone.
    tricks = [
        _trick("smart_contract_integer_overflow", "Integer overflow in arithmetic"),
        _trick("smart_contract_integer_underflow", "Integer underflow in arithmetic"),
    ]
    groups = _group(tricks)
    _, result = _build(groups, use_llm=False)
    fm = result.merged_mapping
    assert fm["smart_contract_integer_overflow"] == "smart_contract_integer_overflow"
    assert fm["smart_contract_integer_underflow"] == "smart_contract_integer_underflow"


# ---------------------------------------------------------------------------
# Stage 1B: LLM judge
# ---------------------------------------------------------------------------


class MockJudgeClient:
    """LLM stand-in: returns canned verdicts per pair_id."""

    def __init__(self, verdicts: dict[int, str]):
        self.verdicts = verdicts

    async def create_message(self, system_prompt, user_message, cache_system=True):
        payload = json.loads(user_message)
        verdicts = [
            {
                "pair_id": item["pair_id"],
                "verdict": self.verdicts.get(item["pair_id"], "keep"),
                "reason": "mock",
            }
            for item in payload
        ]
        return {"verdicts": verdicts}


def test_llm_judge_merges_one_and_keeps_one():
    tricks = [
        _trick("python_code_injection", "Python code injection RCE"),
        _trick("python_code_injection", "Code injection in python eval"),
        _trick("python_bytecode_injection", "Python bytecode injection RCE"),
        _trick("aes_cbc_oracle", "AES CBC padding oracle decrypt"),
        _trick("aes_ecb_oracle", "AES ECB padding oracle decrypt"),
    ]
    groups = _group(tricks)
    merger = TechniqueMerger(use_llm=True, llm_client=MockJudgeClient({1: "merge"}))
    result = merger.build_merged_mapping(groups)
    assert result.llm_candidates, "expected name-similar pairs to reach the LLM tier"
    # Candidates are sorted tuples: pair_id 0 = (aes_cbc, aes_ecb), 1 = (python_bytecode, python_code).
    assert result.llm_candidates[0] == ("aes_cbc_oracle", "aes_ecb_oracle")
    assert result.llm_candidates[1] == ("python_bytecode_injection", "python_code_injection")

    final_mapping = asyncio.run(run_llm_merges(merger, result))
    # pair_id 1 -> merge: python_bytecode folds into python_code (more tricks).
    assert final_mapping["python_bytecode_injection"] == "python_code_injection"
    # pair_id 0 -> keep: CBC and ECB stay separate.
    assert final_mapping["aes_cbc_oracle"] == "aes_cbc_oracle"
    assert final_mapping["aes_ecb_oracle"] == "aes_ecb_oracle"


def test_merger_output_is_deterministic():
    tricks = [
        _trick("type_juggling", "Magic hash comparison bypass via 0e"),
        _trick("php_loose_comparison", "Magic hash comparison bypass via 0e"),
        _trick("php_disable_functions_bypass", "Bypass disable_functions via regex"),
        _trick("php_disabled_functions_bypass", "Bypass disabled functions"),
        _trick("aslr_bypass", "ASLR bypass via leak"),
        _trick("kaslr_bypass", "KASLR bypass via leak"),
    ]
    m1 = build_deterministic_mapping(t.technique_name for t in tricks)
    rng = random.Random(1)
    shuffled = tricks[:]
    rng.shuffle(shuffled)
    m2 = build_deterministic_mapping(t.technique_name for t in shuffled)
    assert m1 == m2
    _, r1 = _build(_group(tricks))
    _, r2 = _build(_group(shuffled))
    assert r1.merged_mapping == r2.merged_mapping


# ---------------------------------------------------------------------------
# Stage 1C: end-to-end render merges technique variants into one file
# ---------------------------------------------------------------------------


def test_run_deduplication_merges_technique_variants(tmp_path):
    tricks = [
        _trick("php_disable_functions_bypass", "Bypass disable_functions via regex"),
        _trick("php_disabled_functions_bypass", "Bypass disabled functions"),
        _trick("sql_injection", "Union based SQL injection"),
    ]
    tricks_file = tmp_path / "tricks.jsonl"
    tricks_file.write_text(
        "\n".join(json.dumps(t.model_dump()) for t in tricks), encoding="utf-8"
    )
    written = run_deduplication(tricks_path=tricks_file, output_dir=tmp_path / "output")
    names = {p.name for p in written}
    assert "php_disable_functions_bypass.md" in names
    assert "php_disabled_functions_bypass.md" not in names
    assert "sql_injection.md" in names
