"""Hybrid retrieval: vector search + BM25 keyword search, merged with Reciprocal Rank Fusion (RRF).

Metadata filters (e.g. region="South Asia", chapter_num=3) come from the document profile's
query analysis and are applied to BOTH searches, so chunks from other regions/chapters can't leak in.
"""
from __future__ import annotations

from dataclasses import dataclass, field

from .config import load_config
from .embeddings import get_embedder
from .models import Chunk
from .store import KeywordIndex, VectorStore, load_chunks


@dataclass
class QueryPlan:
    """What the profile learned from the question."""
    search_text: str                       # question + synonym expansions
    filters: dict = field(default_factory=dict)   # exact-match metadata filters
    notes: list[str] = field(default_factory=list)  # human-readable explanation (printed with --verbose)
    # Chunks containing one of these exact phrases are always retrieved and put first.
    # Used sparingly by a profile when it KNOWS which passage answers a question type.
    pin_phrases: list[str] = field(default_factory=list)


@dataclass
class Hit:
    chunk: Chunk
    score: float          # fused RRF score
    similarity: float     # cosine similarity from the vector search (0 if only BM25 found it)


def _matches(chunk: Chunk, filters: dict) -> bool:
    meta = chunk.chroma_metadata()
    return all(meta.get(k) == v for k, v in filters.items())


def _chroma_where(filters: dict) -> dict | None:
    if not filters:
        return None
    clauses = [{k: v} for k, v in filters.items()]
    return clauses[0] if len(clauses) == 1 else {"$and": clauses}


class Retriever:
    def __init__(self, doc_id: str):
        self.doc_id = doc_id
        self.cfg = load_config()["retrieval"]
        self.chunks = load_chunks(doc_id)
        self.by_id = {c.chunk_id: c for c in self.chunks}
        self.embedder = get_embedder()
        self.vectors = VectorStore(self.embedder.name)
        self.keywords = KeywordIndex(self.chunks)

    def retrieve(self, plan: QueryPlan, top_k: int | None = None) -> list[Hit]:
        top_k = top_k or self.cfg["top_k"]
        n = self.cfg["candidates"]
        filters = dict(plan.filters)

        allowed = {c.chunk_id for c in self.chunks if _matches(c, filters)} if filters else None
        if filters and not allowed:
            plan.notes.append(f"No chunks match filters {filters} -> nothing to retrieve in that section.")
            return []

        vec = self.embedder.embed([plan.search_text], kind="query")[0]
        vec_hits = self.vectors.query(self.doc_id, vec, n, _chroma_where(filters))
        kw_hits = self.keywords.query(plan.search_text, n, allowed)

        k = self.cfg["rrf_k"]
        fused: dict[str, float] = {}
        for ranked in (vec_hits, kw_hits):
            for rank, (cid, _) in enumerate(ranked):
                fused[cid] = fused.get(cid, 0.0) + 1.0 / (k + rank + 1)
        sims = dict(vec_hits)

        ordered = sorted(fused, key=lambda cid: -fused[cid])
        if plan.pin_phrases:
            pinned = []
            for phrase in plan.pin_phrases:              # the first chunk that contains each phrase
                cid = next((c.chunk_id for c in self.chunks if phrase in c.text
                            and (allowed is None or c.chunk_id in allowed)), None)
                if cid and cid not in pinned:
                    pinned.append(cid)
            if pinned:
                plan.notes.append(f"pinned {len(pinned)} chunk(s) containing: {', '.join(plan.pin_phrases)}")
            top = max(fused.values(), default=0.0)
            for cid in pinned:
                fused[cid] = max(fused.get(cid, 0.0), top) + 1e-6          # keep pinned chunks on top
            ordered = pinned + [cid for cid in ordered if cid not in pinned]
        ordered = ordered[:top_k]
        return [Hit(self.by_id[cid], fused[cid], sims.get(cid, 0.0)) for cid in ordered]
