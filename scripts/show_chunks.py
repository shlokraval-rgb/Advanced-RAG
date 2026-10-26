"""Look inside the index: list chunks of a document, optionally filtered by text or type.

Usage:
    python scripts/show_chunks.py --doc gep --grep "South Asia" --limit 5
    python scripts/show_chunks.py --doc imac --type chart
    python scripts/show_chunks.py --doc gpo --stats
"""
import argparse
from collections import Counter

import _bootstrap  # noqa: F401
from advanced_rag.config import DOC_IDS
from advanced_rag.store import load_chunks


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--doc", required=True, choices=DOC_IDS)
    ap.add_argument("--grep", help="only chunks containing this text (case-insensitive)")
    ap.add_argument("--type", help="only this element type (paragraph, table, chart, callout, footnote...)")
    ap.add_argument("--limit", type=int, default=10)
    ap.add_argument("--stats", action="store_true", help="only print counts per type and section")
    args = ap.parse_args()

    chunks = load_chunks(args.doc)
    if args.stats:
        print(f"{len(chunks)} chunks")
        print("by type:", dict(Counter(c.element_type for c in chunks)))
        for path, n in Counter(c.section_path for c in chunks).most_common(25):
            print(f"  {n:>3}  {path}")
        return
    shown = 0
    for c in chunks:
        if args.type and c.element_type != args.type:
            continue
        if args.grep and args.grep.lower() not in c.text.lower():
            continue
        print(f"\n--- {c.chunk_id} | PDF p. {c.pages} | {c.element_type} | {c.section_path}")
        print(f"    metadata: {c.metadata}")
        print(c.text)
        shown += 1
        if shown >= args.limit:
            break
    print(f"\n({shown} chunk(s) shown)")


if __name__ == "__main__":
    main()
