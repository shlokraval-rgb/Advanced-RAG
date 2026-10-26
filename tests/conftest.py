"""Tests run without Ollama and without any API key.

They use RAG_OFFLINE=1 (simple hashing embeddings) and write their index into a TEMPORARY folder,
so running the tests never touches your real index in data/index/.
"""
import os
import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
os.environ["RAG_OFFLINE"] = "1"


@pytest.fixture(scope="session", autouse=True)
def temporary_data_dirs(tmp_path_factory):
    from advanced_rag.config import load_config
    paths = load_config()["paths"]
    saved = dict(paths)
    tmp = tmp_path_factory.mktemp("rag_data")
    paths["processed_dir"] = str(tmp / "processed")   # absolute paths win over PROJECT_ROOT
    paths["index_dir"] = str(tmp / "index")
    yield
    paths.update(saved)


@pytest.fixture(scope="session")
def chunks():
    """Parse + chunk all three documents once (no embeddings needed for most tests)."""
    from advanced_rag.profiles import get_profile
    return {doc: get_profile(doc).build_chunks() for doc in ("gpo", "gep", "imac")}


@pytest.fixture(scope="session")
def ingested():
    """Full offline ingest (hashing embeddings + ChromaDB) for the retrieval tests."""
    from advanced_rag.pipeline import ingest
    for doc in ("gpo", "gep", "imac"):
        ingest(doc)
    return True
