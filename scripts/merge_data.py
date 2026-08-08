#!/usr/bin/env python3
"""Merge two data.json files, deduplicating by writeup URL.

Both files must be JSON arrays of writeup objects with a ``url`` field.
Writeups in the second file whose URL already exists in the first file are
skipped.  The output is the first file's content followed by the new writeups.

Usage::

    # Write to a new file
    uv run python scripts/merge_data.py data.json NEW_data.json -o merged.json

    # Overwrite the first file (in-place append)
    uv run python scripts/merge_data.py data.json NEW_data.json --inplace

    # Stats only, no write
    uv run python scripts/merge_data.py data.json NEW_data.json --dry-run

After merging, run ``nekozuki summarize --fill-gaps`` so only the new writeups
get LLM-extracted.  The clean cache (``data/processed/writeups_clean.jsonl``)
auto-refreshes when the merged ``data.json`` mtime is newer than the cache.
"""

import argparse
import json
import os
import sys
from pathlib import Path


def load_writeups(path: str | Path) -> list[dict]:
    """Load a data.json file.  Exits on error with a clear message."""
    p = Path(path)
    if not p.exists():
        print(f"Error: file not found: {p}", file=sys.stderr)
        sys.exit(1)
    if not p.is_file():
        print(f"Error: not a file: {p}", file=sys.stderr)
        sys.exit(1)

    try:
        data = json.loads(p.read_text(encoding="utf-8"))
    except json.JSONDecodeError as e:
        print(f"Error: invalid JSON in {p}: {e}", file=sys.stderr)
        sys.exit(1)

    if not isinstance(data, list):
        print(f"Error: {p} must be a JSON array (got {type(data).__name__})", file=sys.stderr)
        sys.exit(1)

    return data


def merge(
    base: list[dict],
    other: list[dict],
    *,
    verbose: bool = False,
) -> list[dict]:
    """Merge *other* into *base*, deduplicating by ``url``.

    Returns a new list (base + new writeups from other).  Writeups in *other*
    whose ``url`` is already in *base* are skipped.
    """
    # Build a fast lookup of existing URLs (skip entries without a URL)
    seen: set[str] = set()
    for w in base:
        url = w.get("url", "")
        if url:
            seen.add(url)

    existing_count = len(base)
    dup_count = 0
    no_url_count = 0
    new_writeups: list[dict] = []

    for w in other:
        url = w.get("url", "")
        if not url:
            no_url_count += 1
            if verbose:
                # Print a snippet for debugging
                title = w.get("challenge_title", w.get("challenge_name", "(no title)"))
                print(f"  [warn] skip: no url field — {title}")
            continue
        if url in seen:
            dup_count += 1
            if verbose:
                print(f"  [skip] {url} — already exists in base")
            continue
        seen.add(url)
        new_writeups.append(w)

    if verbose:
        print(f"  base writeups:          {existing_count:>8}")
        print(f"  other writeups:         {len(other):>8}")
        print(f"    duplicates skipped:    {dup_count:>8}")
        print(f"    no-url entries skipped: {no_url_count:>8}")
        print(f"    new writeups:          {len(new_writeups):>8}")
        print(f"  merged total:           {existing_count + len(new_writeups):>8}")

    return base + new_writeups


def main() -> None:
    parser = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument("base", help="Base data.json file (will be kept as-is)")
    parser.add_argument("other", help="Second data.json file (new writeups)")
    parser.add_argument(
        "-o", "--output",
        help="Write merged result to this path (mutually exclusive with --inplace)",
    )
    parser.add_argument(
        "--inplace",
        action="store_true",
        help="Overwrite the base file with the merged result",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Show stats without writing any file",
    )
    parser.add_argument(
        "-v", "--verbose",
        action="store_true",
        help="Show per-writeup skip reasons",
    )

    args = parser.parse_args()

    # Validate output mode
    if args.inplace and args.output:
        print("Error: --inplace and --output are mutually exclusive", file=sys.stderr)
        sys.exit(1)
    if not args.inplace and not args.output and not args.dry_run:
        print("Error: specify --output, --inplace, or --dry-run", file=sys.stderr)
        sys.exit(1)

    # Load both files
    if args.verbose:
        print(f"Loading base:  {args.base}")
        print(f"Loading other: {args.other}")
    base = load_writeups(args.base)
    other = load_writeups(args.other)

    if args.verbose:
        print(f"Base:  {len(base)} writeups")
        print(f"Other: {len(other)} writeups")
        print(f"Merging…")

    merged = merge(base, other, verbose=args.verbose)

    if args.dry_run:
        print(f"\nDry run — no files written.")
        return

    # Determine output path
    output_path: str
    if args.inplace:
        output_path = args.base
    else:
        output_path = args.output  # type: ignore[assignment]

    if args.verbose:
        print(f"Writing {len(merged)} writeups to {output_path} …")

    Path(output_path).write_text(
        json.dumps(merged, ensure_ascii=False, indent=1),
        encoding="utf-8",
    )

    file_size_mb = os.path.getsize(output_path) / 1_000_000
    print(f"\nDone: {len(merged)} writeups written to {output_path} ({file_size_mb:.1f} MB)")

    if args.inplace or output_path == args.base:
        cache = Path("data/processed/writeups_clean.jsonl")
        if cache.exists():
            # The merged data.json mtime is newer than the cache, so
            # load_writeups() will auto-refresh on the next run.  No action
            # needed, but let the user know.
            print(f"  Note: cache {cache} will auto-refresh on next run (merged file is newer).")


if __name__ == "__main__":
    main()