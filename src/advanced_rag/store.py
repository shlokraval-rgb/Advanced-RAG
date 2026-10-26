"""Storage: chunk files (JSONL), the ChromaDB vector store and the BM25 keyword index."""
from __future__ import annotations

import json
import re

import chromadb
from chromadb.config import Settings
from rank_bm25 import BM25Okapi

from .config import path
from .models import Chunk

# ---------------------------------------------------------------- chunk files

def chunks_file(doc_id: str):
    return path("processed_dir") / f"{doc_id}_chunks.jsonl"


def save_chunks(doc_id: str, chunks: list[Chunk]) -> None:
    with open(chunks_file(doc_id), "w", encoding="utf-8") as f:
        for c in chunks:
            f.write(json.dumps(c.to_dict(), ensure_ascii=False) + "\n")


def load_chunks(doc_id: str) -> list[Chunk]:
    p = chunks_file(doc_id)
    if not p.exists():
        raise SystemExit(f"No index for '{doc_id}' yet. Run:  python scripts/ingest.py --doc {doc_id}")
    with open(p, encoding="utf-8") as f:
        return [Chunk.from_dict(json.loads(line)) for line in f if line.strip()]


# ---------------------------------------------------------------- vector store (ChromaDB)

class VectorStore:
    def __init__(self, embedder_name: str):
        db_dir = path("index_dir") / embedder_name   # separate folders for ollama / offline vectors
        self.client = chromadb.PersistentClient(path=str(db_dir), settings=Settings(anonymized_telemetry=False))

    def rebuild(self, doc_id: str, chunks: list[Chunk], vectors: list[list[float]]) -> None:
        try:
            self.client.delete_collection(doc_id)       # re-ingesting always starts clean
        except Exception:
            pass
        col = self.client.create_collection(doc_id, metadata={"hnsw:space": "cosine"})
        for i in range(0, len(chunks), 500):
            batch = chunks[i:i + 500]
            col.add(
                ids=[c.chunk_id for c in batch],
                documents=[c.text for c in batch],
                metadatas=[c.chroma_metadata() for c in batch],
                embeddings=vectors[i:i + 500],
            )

    def query(self, doc_id: str, vector: list[float], n: int, where: dict | None = None) -> list[tuple[str, float]]:
        """Return [(chunk_id, cosine_similarity)] best first."""
        col = self.client.get_collection(doc_id)
        n = min(n, col.count())
        res = col.query(query_embeddings=[vector], n_results=n, where=where or None)
        return [(cid, 1.0 - dist) for cid, dist in zip(res["ids"][0], res["distances"][0])]


# ---------------------------------------------------------------- keyword index (BM25)

_STOP = set("""a an and are as at be by for from has have how in is it its of on or that the this to was were
what when where which who will with does do did according much many say says about""".split())


def tokenize(text: str) -> list[str]:
    text = text.lower().replace(",", "")          # "802,834" -> "802834"
    return [t for t in re.findall(r"[a-z0-9]+(?:\.[0-9]+)?", text) if t not in _STOP]


class KeywordIndex:
    """BM25 finds exact words and numbers ("FDsys", "ENERGY STAR", "Table 2.9") that embeddings can blur."""

    def __init__(self, chunks: list[Chunk]):
        self.chunks = chunks
        self.bm25 = BM25Okapi([tokenize(c.text) for c in chunks])

    def query(self, question: str, n: int, allowed: set[str] | None = None) -> list[tuple[str, float]]:
        scores = self.bm25.get_scores(tokenize(question))
        ranked = sorted(zip(self.chunks, scores), key=lambda x: -x[1])
        out = [(c.chunk_id, s) for c, s in ranked if s > 0 and (allowed is None or c.chunk_id in allowed)]
        return out[:n]
