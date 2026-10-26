"""Parse -> chunk -> embed -> store one or all documents.

Usage:
    python scripts/ingest.py --doc gpo
    python scripts/ingest.py --doc all
"""
import argparse

import _bootstrap  # noqa: F401
from advanced_rag import ollama_client
from advanced_rag.config import DOC_IDS, is_offline
from advanced_rag.pipeline import ingest


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--doc", required=True, choices=[*DOC_IDS, "all"])
    args = ap.parse_args()

    if is_offline():
        print("NOTE: OFFLINE test mode (RAG_OFFLINE=1) -> simple hashing embeddings, no Ollama.\n")
    else:
        ollama_client.check_ready()
        print("Using local Ollama embeddings (free). This can take a few minutes on a CPU.\n")
    for doc_id in (DOC_IDS if args.doc == "all" else [args.doc]):
        print(f"Ingesting '{doc_id}' ...")
        stats = ingest(doc_id)
        for line in stats["structure_log"]:
            print(f"  check: {line}")
        print(f"  {stats['chunks']} chunks  {stats['by_type']}  embedder={stats['embedder']}  ({stats['seconds']}s)")
        print(f"  chunk file: data/processed/{doc_id}_chunks.jsonl\n")


if __name__ == "__main__":
    main()
