"""CLI entry point for nekozuki."""

import argparse
import asyncio
import logging
import sys

from src.config import settings

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s %(levelname)-8s %(name)s:%(lineno)d %(message)s",
    datefmt="%H:%M:%S",
)

# Quiet the HTTP/API client libraries: their per-request INFO lines (one per
# LLM/embedding call) drown out the pipeline's own output.
for _lib in (
    "httpx",
    "httpcore",
    "openai",
    "anthropic",
    "chromadb",
    "urllib3",
    "urllib3.connectionpool",
):
    logging.getLogger(_lib).setLevel(logging.WARNING)
logger = logging.getLogger("nekozuki")


async def run_summarize(args: argparse.Namespace) -> int:
    """Run the summarization pipeline."""
    from src.summarization.checkpoint import CheckpointManager
    from src.summarization.extractor import TrickExtractor

    # --- --fill-gaps: only process writeups that have no tricks yet ---
    extra_kw = {}
    if args.fill_gaps:
        # Use a temp checkpoint so we don't disturb the real one
        checkpoint_mgr = CheckpointManager(
            settings.checkpoint_dir / "fill_gaps.tmp.json"
        )
        missing = _load_missing_writeups()
        if not missing:
            logger.info("All %d writeups already have tricks — nothing to do", len(missing))
            return 0
        logger.info("Fill-gaps: %d writeups still need tricks", len(missing))
        extra_kw["writeups"] = missing
    else:
        checkpoint_mgr = CheckpointManager()

    try:
        extractor = TrickExtractor(checkpoint_mgr, concurrency=args.concurrency)
    except ValueError as e:
        logger.error("%s", e)
        return 3

    if args.force:
        checkpoint_mgr.reset()

    from tqdm import tqdm

    bar = tqdm(
        total=0,
        desc="Summarizing writeups",
        unit="batch",
        bar_format="{l_bar}{bar}| {n_fmt}/{total_fmt} batches "
                   "[{elapsed}<{remaining}] {postfix}",
    )

    # Local counter that only advances forward (retried batches don't regress)
    completed = 0

    # Handle pause requests from keyboard
    async for event in extractor.extract_all(
        batch_limit=args.batch_limit, force_reset=args.force, **extra_kw
    ):
        # Update the progress bar from the yielded event
        if event.total_batches and event.total_batches != bar.total:
            bar.total = event.total_batches
        # Only advance forward — retried or permanent-failure batches
        # (whose batch_index may be old) must not regress the bar.
        completed = max(completed, event.batch_index + 1)
        bar.n = completed
        bar.set_postfix(
            tricks=event.tricks_extracted,
            tokens=f"{event.tokens_used:,}",
        )
        bar.refresh()

        if event.status in ("paused", "completed", "failed"):
            bar.set_description(f"[{event.status}] {event.message}")
            bar.refresh()
            bar.close()
            break

    if event.status == "completed":
        logger.info("=== SUMMARIZATION COMPLETE ===")
        # extractor._save_all_tricks already rebuilt tricks_all.json from the
        # full JSONL accumulator (old + new, incl. --fill-gaps), so dedup here
        # sees every trick.
        logger.info("Running deduplication...")
        from src.summarization.deduplicator import run_deduplication
        written = run_deduplication()
        logger.info("Wrote %d technique files to %s", len(written), settings.output_dir)
        return 0
    elif event.status == "paused":
        logger.info("=== PAUSED — run `nekozuki summarize` again to resume ===")
        return 1
    elif event.status == "failed":
        logger.error("=== SUMMARIZATION FAILED: %s ===", event.message)
        # Still run dedup on partial tricks so the user can see what was extracted
        if event.tricks_extracted > 0:
            logger.info("Running deduplication on partial results (%d tricks)...", event.tricks_extracted)
            from src.summarization.deduplicator import run_deduplication
            written = run_deduplication()
            logger.info("Wrote %d technique files to %s", len(written), settings.output_dir)
        return 2
    else:
        logger.error("=== SUMMARIZATION FAILED: %s ===", event.message)
        return 2


def _load_missing_writeups() -> list:
    """Return writeups that have no tricks extracted yet.

    Reads the existing trick source URLs from tricks_all.json (or tricks.jsonl)
    and filters writeups from data.json to only those whose URLs are absent.
    """
    from pathlib import Path

    from src.processing.batch import load_writeups

    # 1. Collect all writeup URLs that already have tricks
    tricks_path = settings.tricks_dir / "tricks_all.json"
    tricks_path_alt = settings.tricks_dir / "tricks.jsonl"
    existing_urls: set[str] = set()

    if tricks_path.exists():
        import json
        with open(tricks_path, "r", encoding="utf-8") as f:
            for trick in json.load(f):
                existing_urls.update(trick.get("source_writeups", []))
    elif tricks_path_alt.exists():
        import json
        with open(tricks_path_alt, "r", encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                trick = json.loads(line)
                existing_urls.update(trick.get("source_writeups", []))

    logger.info("Found %d writeup URLs with existing tricks", len(existing_urls))

    if not existing_urls:
        logger.warning("No tricks exist yet; use `nekozuki summarize` without --fill-gaps")
        return []

    # 2. Load all writeups (from cache) and filter
    data_path = Path(settings.data_path)
    if not data_path.exists():
        return []

    # Load from cache if available, else raw data.json
    all_writeups = load_writeups()
    if not all_writeups:
        return []

    # 3. Keep only writeups whose URL is NOT in the existing set
    missing = [w for w in all_writeups if w.url not in existing_urls]
    logger.info("Missing writeups: %d (of %d total)", len(missing), len(all_writeups))
    return missing


def run_missing_writeups(args: argparse.Namespace) -> int:
    """Report writeups that have no extracted tricks yet.

    Recomputes the "missing" set from the current tricks store and writeup set
    (same logic `--fill-gaps` uses) without extracting anything.  Shows a
    sample and optionally writes the full URL list to a file.
    """
    missing = _load_missing_writeups()

    if not missing:
        print("All writeups already have tricks.")
        return 0

    print(f"{len(missing)} writeups have no tricks yet:")
    for w in missing[: args.sample]:
        title = (
            getattr(w, "challenge_title", None)
            or getattr(w, "challenge_name", None)
            or "(no title)"
        )
        print(f"  {w.url}  {title}")

    if args.save:
        with open(args.save, "w", encoding="utf-8") as f:
            for w in missing:
                f.write(w.url + "\n")
        print(f"Saved {len(missing)} URLs to {args.save}")

    return 0


def run_embed(args: argparse.Namespace) -> int:
    """Run the embedding pipeline."""
    from src.embedding.engine import run_embedding_pipeline
    return asyncio.run(run_embedding_pipeline(
        force_reset=args.force,
        generate_questions=not args.no_questions,
        concurrency=args.concurrency,
        batch_size=args.batch_size,
    ))


def run_add_url(args: argparse.Namespace) -> int:
    """Ingest a writeup from a URL and extract tricks."""
    from src.ingestion import ingest_writeup_from_url

    try:
        tricks = ingest_writeup_from_url(args.url, persist=True)
    except Exception as e:  # noqa: BLE001 (ingestion errors are surfaced below)
        logger.error("Failed to ingest writeup from %s: %s", args.url, e)
        return 2

    if not tricks:
        logger.info("No tricks extracted from %s", args.url)
        return 0

    logger.info("Extracted %d trick(s) from %s:", len(tricks), args.url)
    for t in tricks:
        logger.info("  • %s (%s) — conf %.2f", t.title, t.technique_name, t.confidence)

    # Re-run dedup so the output markdown files reflect the new tricks.
    if not args.no_dedup:
        logger.info("Running deduplication to update output files...")
        from src.summarization.deduplicator import run_deduplication
        written = run_deduplication()
        logger.info("Wrote %d technique files to %s", len(written), settings.output_dir)

    return 0


def run_dedup_techniques(args: argparse.Namespace) -> int:
    """Merge near-duplicate canonical techniques and rewrite data/tricks.

    Stage 1 of the dedup work: consolidate the thousands of deterministic
    canonical technique groups into genuinely-distinct files (spelling variants,
    synonyms, chatty rephrasings collapse into one), so trick-level dedup and
    retrieval operate on clean groups. Optionally LLM-judges ambiguous
    name-similar pairs. Rewrites ``data/tricks/*`` to the merged technique names
    (backed up first) and re-renders ``output/*.md``.
    """
    import asyncio
    import json
    import shutil
    from collections import defaultdict
    from datetime import UTC, datetime
    from pathlib import Path

    from src.models import Trick
    from src.summarization.deduplicator import TechniqueFileWriter, TrickDeduplicator
    from src.summarization.normalizer import build_deterministic_mapping
    from src.summarization.technique_merger import (
        TechniqueMerger,
        merged_groups,
        run_llm_merges,
    )

    # 1. Load tricks as raw dicts (rewrite preserves the exact original schema).
    tricks_path = settings.tricks_dir / "tricks_all.json"
    if not tricks_path.exists():
        logger.error("No tricks file at %s", tricks_path)
        return 2
    raw_tricks = json.loads(tricks_path.read_text(encoding="utf-8"))
    if not raw_tricks:
        logger.error("tricks_all.json is empty — nothing to dedup")
        return 2
    tricks = [Trick(**d) for d in raw_tricks]
    logger.info("Loaded %d tricks from %s", len(tricks), tricks_path)

    # 2. Deterministic canonical groups (order-independent).
    canonical = build_deterministic_mapping(t.technique_name for t in tricks)
    groups: dict[str, list[Trick]] = defaultdict(list)
    for t in tricks:
        groups[canonical[t.technique_name]].append(t)

    # 3. Technique-level merging (content + name-similarity tiers).
    merger = TechniqueMerger(
        name_threshold=args.name_threshold,
        content_shared_titles=args.content_threshold,
        llm_content_jaccard=args.llm_content_jaccard,
        use_llm=not args.no_llm,
    )
    result = merger.build_merged_mapping(groups)
    st = result.stats
    logger.info(
        "Technique merge: %d groups -> %d files (content %d, name-auto %d, "
        "name-similar pending %d, kept %d)",
        st["groups"], st["merged_files"], st["content_merges"],
        st["auto_merges"], st["llm_pending"], st["kept"],
    )

    final_mapping = result.merged_mapping
    llm_merges = 0
    # The LLM judge is skipped on --dry-run (it would spend API tokens on a
    # preview). A dry-run reports how many pairs WOULD be judged.
    if not args.no_llm and not args.dry_run and result.llm_candidates:
        try:
            from src.llm import LLMClient
            merger.llm_client = LLMClient()
        except ValueError as e:
            logger.warning("LLM judge disabled: %s", e)
        if merger.llm_client:
            logger.info("LLM-judging %d name-similar pairs...", len(result.llm_candidates))
            final_mapping = asyncio.run(run_llm_merges(merger, result))
            llm_merges = len(result.llm_candidates) - len(merger.llm_kept)
            logger.info("LLM merged %d pairs, kept %d separate", llm_merges, len(merger.llm_kept))

    # 4. Report.
    final_groups = merged_groups(groups, final_mapping)
    print("\nTechnique dedup summary")
    print(f"  raw tricks          : {len(tricks)}")
    print(f"  canonical groups    : {st['groups']}")
    print(f"  merged by content   : {st['content_merges']}")
    print(f"  merged by name      : {st['auto_merges']}")
    if llm_merges:
        print(f"  merged by LLM       : {llm_merges} (of {st['llm_pending']} judged)")
    elif args.no_llm:
        print(f"  name-similar kept   : {st['kept']} (LLM disabled)")
    else:
        print(f"  would LLM-judge     : {st['llm_pending']} name-similar pairs")
    print(f"  final technique files: {len(final_groups)}")
    print(f"  tricks preserved    : {sum(len(v) for v in final_groups.values())}")
    print()
    print("Merge clusters (representative <- members):")
    shown = 0
    for comp in result.components:
        if len(comp) > 1:
            print(f"  {comp[0]}  <-  {', '.join(comp[1:])}")
            shown += 1
            if shown >= 60:
                print(f"  ... and {sum(1 for c in result.components if len(c) > 1) - 60} more")
                break
    if args.dry_run:
        print("\nDry run — no files were changed.")
        return 0

    # 5. Backup, then rewrite data/tricks to merged technique names.
    ts = datetime.now(UTC).strftime("%Y%m%d_%H%M%S")
    backup_dir = Path("backups") / f"tricks_{ts}"
    backup_dir.mkdir(parents=True, exist_ok=True)
    for fname in ("tricks_all.json", "tricks.jsonl"):
        src = settings.tricks_dir / fname
        if src.exists():
            shutil.copy2(src, backup_dir / fname)
    out_backup = Path("backups") / f"output_{ts}"
    if settings.output_dir.exists():
        shutil.copytree(settings.output_dir, out_backup)
    logger.info("Backed up data/tricks -> %s, output -> %s", backup_dir, out_backup)

    for d, t in zip(raw_tricks, tricks):
        d["technique_name"] = final_mapping.get(canonical[t.technique_name], canonical[t.technique_name])
    with open(tricks_path, "w", encoding="utf-8") as f:
        json.dump(raw_tricks, f, indent=2, ensure_ascii=False)
    with open(settings.tricks_dir / "tricks.jsonl", "w", encoding="utf-8") as f:
        f.writelines(json.dumps(d, ensure_ascii=False) + "\n" for d in raw_tricks)
    logger.info("Rewrote %s and tricks.jsonl with merged technique names", tricks_path)

    _merge_aliases_into_mapping(final_mapping)

    # 6. Re-render output with per-group trick dedup on the merged groups.
    deduper = TrickDeduplicator()
    output_dir = Path(args.out) if args.out else settings.output_dir
    output_dir.mkdir(parents=True, exist_ok=True)
    deduped = {name: deduper.deduplicate_group(ts) for name, ts in final_groups.items()}
    written = TechniqueFileWriter(output_dir).write_all(deduped)
    _remove_stale_technique_files(written, output_dir)
    logger.info("Wrote %d technique files to %s", len(written), output_dir)
    return 0


def _merge_aliases_into_mapping(final_mapping: dict[str, str]) -> None:
    """Persist merged canonical aliases to data/technique_mapping.yaml.

    Future extractions that produce an old (pre-merge) technique name land
    directly in the merged file. Existing user-curated aliases are not
    overwritten.
    """
    import yaml

    path = settings.technique_mapping_path
    custom = {}
    if path.exists():
        try:
            custom = yaml.safe_load(path.read_text(encoding="utf-8")) or {}
        except yaml.YAMLError as e:
            logger.warning("Skipping mapping update (bad YAML): %s", e)
            return
    cm = custom.get("technique_mapping", {})
    added = 0
    for old, merged in sorted(final_mapping.items()):
        if old != merged and old not in cm:
            cm[old] = merged
            added += 1
    if not added:
        logger.info("No new technique aliases to persist")
        return
    custom["technique_mapping"] = cm
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        yaml.dump(custom, f, default_flow_style=False, sort_keys=True)
    logger.info("Added %d technique aliases to %s", added, path)


def _remove_stale_technique_files(written: list, output_dir) -> None:
    """Delete output/*.md files no longer produced by the current merge."""
    from pathlib import Path

    written_names = {Path(p).name for p in written}
    removed = 0
    for p in Path(output_dir).glob("*.md"):
        if p.name not in written_names:
            p.unlink()
            removed += 1
    if removed:
        logger.info("Removed %d stale technique files from %s", removed, output_dir)


def run_dedup_tricks(args: argparse.Namespace) -> int:
    """Stage 2: three-tier trick dedup (lexical -> embedding -> LLM).

    Clusters tricks deterministically (tier 1), then optionally merges
    semantically-duplicate pairs via embedding cosine (tier 2) and an LLM judge
    for the gray zone (tier 3). A real run prunes the merged-away tricks from
    ``data/tricks`` (backed up first) and re-renders ``output/*.md``.
    """
    import asyncio
    import json
    import shutil
    from collections import defaultdict
    from datetime import UTC, datetime
    from pathlib import Path

    from src.models import Trick
    from src.summarization.deduplicator import TechniqueFileWriter
    from src.summarization.normalizer import build_deterministic_mapping
    from src.summarization.technique_merger import TechniqueMerger, merged_groups
    from src.summarization.trick_deduper import TrickDeduper

    # 1. Load tricks as raw dicts (rewrite preserves the exact schema).
    tricks_path = settings.tricks_dir / "tricks_all.json"
    if not tricks_path.exists():
        logger.error("No tricks file at %s", tricks_path)
        return 2
    raw_tricks = json.loads(tricks_path.read_text(encoding="utf-8"))
    if not raw_tricks:
        logger.error("tricks_all.json is empty — nothing to dedup")
        return 2
    tricks = [Trick(**d) for d in raw_tricks]

    # 2. Deterministic technique grouping + Stage 1 technique merge.
    canonical = build_deterministic_mapping(t.technique_name for t in tricks)
    groups: dict[str, list[Trick]] = defaultdict(list)
    for t in tricks:
        groups[canonical[t.technique_name]].append(t)
    merger = TechniqueMerger(use_llm=False)
    groups = merged_groups(groups, merger.build_merged_mapping(groups).merged_mapping)

    # 3. Tier 1 (lexical) + candidate generation.
    from tqdm import tqdm

    deduper = TrickDeduper(
        use_embed=not args.no_embed,
        use_llm=not args.no_llm,
        embed_threshold=args.embed_threshold,
        gray_lo=args.llm_gray_lo,
    )
    bar = tqdm(total=len(groups), desc="Tier 1 lexical clustering", unit="group", ncols=100)
    result = deduper.build(groups, progress=bar.update)
    bar.close()
    st = result.stats
    print("\nTrick dedup (stage 2) — tier 1 (lexical)")
    print(f"  tricks              : {st['tricks']}")
    print(f"  technique groups    : {len(groups)}")
    print(f"  lexical clusters    : {st['lexical_clusters']} (merged {st['lexical_merges']})")
    print(f"  embedding candidates: {st['candidates']}")

    # 4. Tiers 2-3 (embedding + LLM) — skipped on dry-run (free preview).
    extra_merges: set[frozenset] = set()
    if not args.dry_run and not args.no_embed:
        try:
            from src.embedding.engine import EmbeddingEngine
            deduper.embedder = EmbeddingEngine()
        except ValueError as e:
            logger.warning("Embedding tier disabled: %s", e)
        if deduper.embedder:
            if not args.no_llm:
                try:
                    from src.llm import LLMClient
                    deduper.llm_client = LLMClient()
                except ValueError as e:
                    logger.warning("LLM judge disabled: %s", e)

            import math

            unique_count = len({i for pair in result.candidates for i in pair})

            async def _run_tiers() -> set[frozenset]:
                emb_total = math.ceil(unique_count / deduper.EMBED_CHUNK_SIZE)
                emb_bar = tqdm(total=emb_total, desc="Tier 2 embedding", unit="req", ncols=100)
                merge_pairs, gray = await deduper.run_embed_tier(result, progress=emb_bar.update)
                emb_bar.close()
                if gray and deduper.llm_client:
                    llm_total = math.ceil(len(gray) / deduper.LLM_BATCH_SIZE)
                    llm_bar = tqdm(total=llm_total, desc="Tier 3 LLM judge", unit="batch", ncols=100)
                    llm_merges = await deduper.run_llm_tier(gray, result, progress=llm_bar.update)
                    llm_bar.close()
                else:
                    llm_merges = set()
                return merge_pairs | llm_merges

            logger.info("Running embedding/LLM tiers...")
            extra_merges = asyncio.run(_run_tiers())
        else:
            logger.warning("No embedding engine available — falling back to lexical only")

    # 5. Report projected final state.
    merged_groups_out = deduper.final_merge(result, extra_merges)
    print("\nFinal:")
    print(f"  tricks after dedup  : {sum(len(v) for v in merged_groups_out.values())}")
    print(f"  technique files     : {len(merged_groups_out)}")
    if args.dry_run:
        if not args.no_embed:
            print(f"  (dry-run: embedding/LLM tiers skipped — would judge {st['candidates']} candidate pairs)")
        print("\nDry run — no files were changed.")
        return 0

    # 6. Backup, prune data/tricks to representatives, re-render output.
    ts = datetime.now(UTC).strftime("%Y%m%d_%H%M%S")
    backup_dir = Path("backups") / f"tricks_{ts}"
    backup_dir.mkdir(parents=True, exist_ok=True)
    for fname in ("tricks_all.json", "tricks.jsonl"):
        src = settings.tricks_dir / fname
        if src.exists():
            shutil.copy2(src, backup_dir / fname)
    out_backup = Path("backups") / f"output_{ts}"
    if settings.output_dir.exists():
        shutil.copytree(settings.output_dir, out_backup)
    logger.info("Backed up data/tricks -> %s, output -> %s", backup_dir, out_backup)

    rep_dicts = [t.model_dump() for group in merged_groups_out.values() for t in group]
    with open(tricks_path, "w", encoding="utf-8") as f:
        json.dump(rep_dicts, f, indent=2, ensure_ascii=False)
    with open(settings.tricks_dir / "tricks.jsonl", "w", encoding="utf-8") as f:
        f.writelines(json.dumps(d, ensure_ascii=False) + "\n" for d in rep_dicts)
    logger.info(
        "Pruned data/tricks: %d -> %d representative tricks",
        len(raw_tricks), len(rep_dicts),
    )

    output_dir = Path(args.out) if args.out else settings.output_dir
    output_dir.mkdir(parents=True, exist_ok=True)
    written = TechniqueFileWriter(output_dir).write_all(merged_groups_out)
    _remove_stale_technique_files(written, output_dir)
    logger.info("Wrote %d technique files to %s", len(written), output_dir)
    return 0


def run_query(args: argparse.Namespace) -> int:
    """Run a RAG query from the CLI."""
    import asyncio

    async def _search():
        from src.retrieval.index import load_or_build_index

        searcher = load_or_build_index()
        if searcher is None:
            logger.error(
                "No index found. Run `nekozuki build-index` first."
            )
            return None

        return await searcher.search(args.query, top_k=args.top_k)

    results = asyncio.run(_search())
    if results is None:
        return 1

    print(f"\nQuery: {args.query}\n")
    print("Results:")
    print("=" * 60)
    for result in results:
        print(f"\n[{result.rank}] {result.technique_name} — {result.section_title}")
        print(f"    Rerank: {result.rerank_score:.3f} (bm25: {result.bm25_score:.3f})")
        # Print first 150 chars of content
        content_preview = result.content.replace("\n", " ")[:200]
        print(f"    {content_preview}...")
    return 0


def run_build_index(args: argparse.Namespace) -> int:
    """Build the search index without embedding the whole corpus."""
    from src.retrieval.bm25_index import BM25Index

    bm25 = BM25Index()
    built = bm25.build_from_output_dir(force=args.force)
    if built is None:
        logger.error("Failed to build BM25 index")
        return 1
    logger.info("BM25 index built")
    return 0


def run_serve(args: argparse.Namespace) -> int:
    """Start the web server."""
    import uvicorn
    logger.info("Starting nekozuki web UI on %s:%s", args.host, args.port)
    uvicorn.run(
        "src.app:app",
        host=args.host,
        port=args.port,
        reload=False,
        log_level="info",
    )
    return 0


def main(argv: list[str] | None = None) -> int:
    """Main entry point."""
    parser = argparse.ArgumentParser(
        prog="nekozuki",
        description="CTF Writeup Summarization & RAG System",
    )
    subparsers = parser.add_subparsers(dest="command")

    # summarize subcommand
    sum_parser = subparsers.add_parser(
        "summarize", help="Extract tricks from writeups into technique files"
    )
    sum_parser.add_argument(
        "--batch-limit",
        type=int,
        default=0,
        help="Only process this many batches (0 = all)",
    )
    sum_parser.add_argument(
        "--force",
        action="store_true",
        help="Ignore existing checkpoint and start fresh",
    )
    sum_parser.add_argument(
        "--fill-gaps",
        action="store_true",
        help="Only process writeups that have no tricks yet (incremental fill)",
    )
    sum_parser.add_argument(
        "--concurrency",
        type=int,
        default=0,
        help="Number of writeup batches to process concurrently (default: %(default)s, uses ENV LLM_MAX_CONCURRENCY)",
    )
    sum_parser.set_defaults(func=lambda args: asyncio.run(run_summarize(args)))

    # embed subcommand
    embed_parser = subparsers.add_parser(
        "embed", help="Generate embeddings for technique files"
    )
    embed_parser.add_argument(
        "--force",
        action="store_true",
        help="Ignore existing embedding checkpoint and re-embed everything",
    )
    embed_parser.add_argument(
        "--no-questions",
        action="store_true",
        help="Skip pre-retrieval question generation and embedding (chunks only)",
    )
    embed_parser.add_argument(
        "--concurrency",
        type=int,
        default=0,
        help="Number of files to process concurrently (default: %(default)s, uses ENV EMBEDDING_MAX_CONCURRENCY)",
    )
    embed_parser.add_argument(
        "--batch-size",
        type=int,
        default=0,
        help="Texts per embedding API call (default: %(default)s, uses ENV EMBBEDING_BATCH_SIZE)",
    )
    embed_parser.set_defaults(func=lambda args: run_embed(args))

    # query subcommand
    query_parser = subparsers.add_parser(
        "query", help="Query the RAG system with natural language"
    )
    query_parser.add_argument("query", type=str, help="Natural language query")
    query_parser.add_argument(
        "--top-k", type=int, default=settings.top_k, help="Number of results to return"
    )
    query_parser.set_defaults(func=lambda args: run_query(args))

    # add-url subcommand
    add_url_parser = subparsers.add_parser(
        "add-url",
        help="Fetch a writeup from a URL and extract tricks into the pipeline",
    )
    add_url_parser.add_argument("url", type=str, help="URL of the writeup to ingest")
    add_url_parser.add_argument(
        "--no-dedup",
        action="store_true",
        help="Skip re-running deduplication after extraction",
    )
    add_url_parser.set_defaults(func=lambda args: run_add_url(args))

    # missing-writeups subcommand
    missing_parser = subparsers.add_parser(
        "missing-writeups",
        help="List writeups that have no extracted tricks yet",
    )
    missing_parser.add_argument(
        "--save",
        type=str,
        default=None,
        metavar="FILE",
        help="Write the full list of writeup URLs to FILE (one per line)",
    )
    missing_parser.add_argument(
        "--sample",
        type=int,
        default=10,
        help="Show this many example URLs (default: %(default)s)",
    )
    missing_parser.set_defaults(func=lambda args: run_missing_writeups(args))

    # dedup-techniques subcommand
    dedup_parser = subparsers.add_parser(
        "dedup-techniques",
        help="Merge near-duplicate canonical techniques; rewrite data/tricks + output",
    )
    dedup_parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Report merge clusters and projected file counts without changing anything",
    )
    dedup_parser.add_argument(
        "--no-llm",
        action="store_true",
        help="Skip the LLM judge for ambiguous name-similar pairs (keep them separate)",
    )
    dedup_parser.add_argument(
        "--name-threshold",
        type=int,
        default=None,
        help="Fuzzy ratio threshold for name-similar candidates (default: 85)",
    )
    dedup_parser.add_argument(
        "--content-threshold",
        type=int,
        default=None,
        help="Shared identical trick titles required for a content merge (default: 2)",
    )
    dedup_parser.add_argument(
        "--llm-content-jaccard",
        type=float,
        default=None,
        help="Min title-token Jaccard for a name-similar pair to be LLM-judged (default: 0.1)",
    )
    dedup_parser.add_argument(
        "--out",
        type=str,
        default=None,
        help="Output directory for re-rendered files (default: settings.output_dir)",
    )
    dedup_parser.set_defaults(func=lambda args: run_dedup_techniques(args))

    # dedup-tricks subcommand (Stage 2: lexical -> embedding -> LLM)
    tricks_parser = subparsers.add_parser(
        "dedup-tricks",
        help="Three-tier trick dedup: lexical, embedding, LLM; prune data/tricks",
    )
    tricks_parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Run tier 1 only and report projected counts without changing anything",
    )
    tricks_parser.add_argument(
        "--no-embed",
        action="store_true",
        help="Skip the embedding tier (lexical only)",
    )
    tricks_parser.add_argument(
        "--no-llm",
        action="store_true",
        help="Skip the LLM judge for gray-zone pairs",
    )
    tricks_parser.add_argument(
        "--embed-threshold",
        type=float,
        default=None,
        help="Cosine threshold to merge embedding candidates (default: 0.90)",
    )
    tricks_parser.add_argument(
        "--llm-gray-lo",
        type=float,
        default=None,
        help="Lower bound of the cosine gray zone sent to the LLM (default: 0.75)",
    )
    tricks_parser.add_argument(
        "--out",
        type=str,
        default=None,
        help="Output directory for re-rendered files (default: settings.output_dir)",
    )
    tricks_parser.set_defaults(func=lambda args: run_dedup_tricks(args))

    # build-index subcommand
    index_parser = subparsers.add_parser(
        "build-index", help="Build the BM25 search index"
    )
    index_parser.add_argument(
        "--force", action="store_true", help="Rebuild index from scratch"
    )
    index_parser.set_defaults(func=lambda args: run_build_index(args))

    # serve subcommand
    serve_parser = subparsers.add_parser(
        "serve", help="Start the web UI server"
    )
    serve_parser.add_argument("--host", type=str, default="127.0.0.1")
    serve_parser.add_argument("--port", type=int, default=8000)
    serve_parser.set_defaults(func=lambda args: run_serve(args))

    # test subcommand
    test_parser = subparsers.add_parser(
        "test", help="Check configuration, artifacts, and API connectivity"
    )
    test_parser.add_argument(
        "--no-connectivity",
        action="store_true",
        help="Skip live API connectivity checks (offline check only)",
    )
    test_parser.set_defaults(func=lambda args: run_test(args))

    args = parser.parse_args(argv)
    if not args.command:
        parser.print_help()
        return 0

    return args.func(args)


def run_test(args: argparse.Namespace) -> int:
    """Run the configuration/connectivity diagnostics."""
    from src.diagnostics import run_test_command
    return run_test_command(include_connectivity=not args.no_connectivity)


if __name__ == "__main__":
    sys.exit(main())