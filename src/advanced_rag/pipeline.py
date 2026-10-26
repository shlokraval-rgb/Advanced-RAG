"""The two top-level operations: ingest(doc_id) and ask(doc_id, question)."""
from __future__ import annotations

import time
from collections import Counter
from dataclasses import dataclass, field

from .config import doc_config, is_offline, load_config
from .embeddings import get_embedder
from .generator import (NOT_FOUND, add_missing_citations, build_context, generate_answer,
                        remove_duplicate_statements, repair_citations, unsupported_numbers)
from .profiles import get_profile
from .retriever import Hit, Retriever
from .store import VectorStore, save_chunks


def ingest(doc_id: str) -> dict:
    t0 = time.time()
    profile = get_profile(doc_id)
    chunks = profile.build_chunks()                     # parse + chunk (document-specific)
    save_chunks(doc_id, chunks)
    embedder = get_embedder()
    vectors = embedder.embed([c.text for c in chunks])  # embeddings
    VectorStore(embedder.name).rebuild(doc_id, chunks, vectors)  # vector storage
    return {
        "structure_log": getattr(profile, "log", []),   # e.g. World Bank bookmark verification
        "doc_id": doc_id,
        "chunks": len(chunks),
        "by_type": dict(Counter(c.element_type for c in chunks)),
        "embedder": embedder.name,
        "seconds": round(time.time() - t0, 1),
    }


@dataclass
class Answer:
    question: str
    answer: str
    hits: list[Hit]
    plan_notes: list[str] = field(default_factory=list)
    filters: dict = field(default_factory=dict)
    warnings: list[str] = field(default_factory=list)
    offline: bool = False


_retrievers: dict[str, Retriever] = {}


def ask(doc_id: str, question: str, use_filters: bool = True) -> Answer:
    """use_filters=False switches off the document-aware query analysis (for comparison only)."""
    cfg = doc_config(doc_id)
    profile = get_profile(doc_id)
    plan = profile.analyze_query(question)              # document-aware query understanding
    if not use_filters:
        plan.filters, plan.notes = {}, ["(filters disabled for comparison)"]
    key = f"{doc_id}:{'offline' if is_offline() else 'ollama'}"
    if key not in _retrievers:
        _retrievers[key] = Retriever(doc_id)
    hits = _retrievers[key].retrieve(plan)              # hybrid retrieval (+ filters)
    result = Answer(question, "", hits, plan.notes, plan.filters, offline=is_offline())

    if not hits:
        result.answer = NOT_FOUND
        return result
    if result.offline:
        result.answer = "(offline mode - no LLM answer; see the retrieved evidence below)"
        return result

    min_sim = load_config()["retrieval"]["min_similarity"]
    if max(h.similarity for h in hits) < min_sim:
        result.answer = NOT_FOUND
        result.warnings.append(f"Best similarity {max(h.similarity for h in hits):.2f} < {min_sim}: treated as out-of-document.")
        return result

    result.answer = generate_answer(question, hits, cfg["title"])  # grounded generation (local LLM)
    result.answer, n_dup = remove_duplicate_statements(result.answer)
    if n_dup:
        result.warnings.append(f"Removed {n_dup} repeated statement(s) from the answer.")
    result.answer, notes = profile.postprocess_answer(question, result.answer, hits)  # source-grounded checks
    result.warnings += notes
    result.answer, n_added = add_missing_citations(result.answer, hits)
    if n_added:
        result.warnings.append(f"{n_added} page citation(s) were added automatically by matching the "
                               "sentence to the excerpt it came from.")
    result.answer, fixed = repair_citations(result.answer, hits)
    if fixed:
        result.warnings.append("Citation check: corrected page(s) that did not support the statement: "
                               + "; ".join(fixed))
    bad = unsupported_numbers(result.answer, build_context(hits) + " " + question)
    if bad and NOT_FOUND not in result.answer:
        result.warnings.append("Numbers not found in the retrieved context (check them!): " + ", ".join(bad))
    return result
