"""Preview the trick-dedup embed tier on real data WITHOUT writing anything.

Reports merge/gray/keep counts and samples the actual merged pairs so we can
eyeball quality before pruning data/tricks.
"""
import asyncio
import json
import time
from collections import defaultdict

from src.embedding.engine import EmbeddingEngine
from src.models import Trick
from src.summarization.normalizer import build_deterministic_mapping
from src.summarization.technique_merger import TechniqueMerger, merged_groups
from src.summarization.trick_deduper import TrickDeduper


async def main():
    t0 = time.perf_counter()
    with open("data/tricks/tricks_all.json") as f:
        d = json.load(f)
    tricks = [Trick(**t) for t in d]
    mapping = build_deterministic_mapping(t.technique_name for t in tricks)
    groups = defaultdict(list)
    for t in tricks:
        groups[mapping[t.technique_name]].append(t)
    merger = TechniqueMerger(use_llm=False)
    groups = merged_groups(groups, merger.build_merged_mapping(groups).merged_mapping)
    print(f"groups={len(groups)} tricks={sum(len(v) for v in groups.values())} ({time.perf_counter()-t0:.1f}s)")

    deduper = TrickDeduper(use_embed=True, use_llm=True, embedder=EmbeddingEngine())
    result = deduper.build(groups)
    print(f"tier1: {result.stats['lexical_clusters']} clusters, {result.stats['candidates']} candidates")

    merge_pairs, gray = await deduper.run_embed_tier(result)
    print(f"embed tier: {len(merge_pairs)} merge, {len(gray)} gray, "
          f"{len(result.candidates) - len(merge_pairs) - len(gray)} keep")

    # sample merged pairs for eyeballing
    by_title = {}
    for i, t in enumerate(result.tricks):
        by_title.setdefault(t.title, []).append(i)
    print("\n--- sample of embed-merged pairs (title A | title B) ---")
    shown = 0
    for fs in sorted(merge_pairs, key=lambda s: sorted(s)[0])[:25]:
        a, b = sorted(fs)
        ta, tb = result.tricks[a].title, result.tricks[b].title
        if ta == tb:
            continue
        print(f"  {ta[:55]}")
        print(f"    == {tb[:55]}")
        shown += 1
    print(f"\n... {shown} shown of {len(merge_pairs)} embed merges")
    print(f"gray-zone pairs (would go to LLM): {len(gray)}")
    print(f"preview done ({time.perf_counter()-t0:.1f}s) — no files changed")


asyncio.run(main())
