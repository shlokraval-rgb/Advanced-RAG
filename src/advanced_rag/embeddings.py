"""Embedding models.

- OllamaEmbedder: the real one - nomic-embed-text running locally in Ollama (free, no API key).
- HashingEmbedder: a tiny offline stand-in used by the unit tests (RAG_OFFLINE=1). It is NOT semantic
  (it only captures shared words), so its retrieval quality is lower.
"""
from __future__ import annotations

import math
import re
import zlib

from . import ollama_client
from .config import is_offline, load_config


class OllamaEmbedder:
    name = "ollama"

    def __init__(self):
        cfg = load_config()["ollama"]
        # nomic-embed-text works best when it is told whether a text is a document or a question
        self.doc_prefix = cfg.get("document_prefix", "")
        self.query_prefix = cfg.get("query_prefix", "")

    def embed(self, texts: list[str], kind: str = "document") -> list[list[float]]:
        prefix = self.query_prefix if kind == "query" else self.doc_prefix
        return ollama_client.embed(texts, prefix)


class HashingEmbedder:
    name = "offline"
    dims = 1024

    def embed(self, texts: list[str], kind: str = "document") -> list[list[float]]:
        out = []
        for t in texts:
            tokens = re.findall(r"[a-z0-9]+", t.lower())
            feats = tokens + [f"{a}_{b}" for a, b in zip(tokens, tokens[1:])]
            vec = [0.0] * self.dims
            for f in feats:
                vec[zlib.crc32(f.encode()) % self.dims] += 1.0
            vec = [math.log1p(v) for v in vec]
            norm = math.sqrt(sum(v * v for v in vec)) or 1.0
            out.append([v / norm for v in vec])
        return out


def get_embedder():
    return HashingEmbedder() if is_offline() else OllamaEmbedder()
