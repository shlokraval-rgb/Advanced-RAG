"""Base class for document profiles.

A *profile* holds everything that is specific to ONE document:
  - how to parse it (layout, headers/footers, tables, footnotes...)
  - how to chunk it
  - how to understand questions about it (filters, synonyms)
Everything else (embeddings, vector store, retrieval, LLM) is shared.
"""
from __future__ import annotations

import re

import pymupdf

from ..config import doc_config, path
from ..models import Chunk
from ..retriever import QueryPlan


class BaseProfile:
    doc_id = ""

    def __init__(self):
        self.cfg = doc_config(self.doc_id)
        self.pdf_path = path("raw_dir") / self.cfg["file"]
        if not self.pdf_path.exists():
            raise SystemExit(f"PDF not found: {self.pdf_path}")

    def open(self) -> pymupdf.Document:
        return pymupdf.open(self.pdf_path)

    def build_chunks(self) -> list[Chunk]:
        raise NotImplementedError

    def analyze_query(self, question: str) -> QueryPlan:
        return self.expand_vocabulary(QueryPlan(search_text=question), question)

    def postprocess_answer(self, question: str, answer: str, hits: list) -> tuple[str, list[str]]:
        """Document-specific, source-grounded checks of the LLM answer. Default: no change.
        Returns (answer, notes about what was changed)."""
        return answer, []

    def expand_vocabulary(self, plan: QueryPlan, question: str) -> QueryPlan:
        """Add the document's own words for words people use in questions (config: `synonyms`).

        Example: the GPO report says "full-scale deployment" where a question says "fully deployed".
        """
        extra = []
        for phrase, alternatives in (self.cfg.get("synonyms") or {}).items():
            if re.search(rf"\b{re.escape(phrase)}\b", question.lower()):
                extra += [a for a in alternatives if a.lower() not in question.lower()]
        if extra:
            plan.search_text += " " + " ".join(extra)
            plan.notes.append(f"added document vocabulary: {', '.join(extra)}")
        return plan
