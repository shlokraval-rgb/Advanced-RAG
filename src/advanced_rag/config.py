"""Loads config.yaml and the optional .env file, and decides normal (Ollama) vs offline test mode."""
from __future__ import annotations

import os
from functools import lru_cache
from pathlib import Path

import yaml
from dotenv import load_dotenv

PROJECT_ROOT = Path(__file__).resolve().parents[2]
DOC_IDS = ("gpo", "gep", "imac")


@lru_cache(maxsize=1)
def load_config() -> dict:
    load_dotenv(PROJECT_ROOT / ".env")
    with open(PROJECT_ROOT / "config.yaml", encoding="utf-8") as f:
        return yaml.safe_load(f)


def doc_config(doc_id: str) -> dict:
    cfg = load_config()["documents"]
    if doc_id not in cfg:
        raise SystemExit(f"Unknown document '{doc_id}'. Choose one of: {', '.join(DOC_IDS)}")
    return {"doc_id": doc_id, **cfg[doc_id]}


def path(key: str) -> Path:
    p = PROJECT_ROOT / load_config()["paths"][key]
    p.mkdir(parents=True, exist_ok=True)
    return p


def is_offline() -> bool:
    """Offline test mode = no model calls at all (hashing embeddings, no LLM answer).

    Normal mode uses the free local Ollama models and needs NO API key.
    Offline mode is only switched on with RAG_OFFLINE=1 (the unit tests do this).
    """
    load_config()  # makes sure .env (optional) is loaded
    return os.getenv("RAG_OFFLINE", "").strip().lower() in ("1", "true", "yes")
