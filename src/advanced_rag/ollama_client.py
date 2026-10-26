"""Tiny client for a LOCAL Ollama server (free, runs on your own PC, no API key).

Ollama runs in the background after installation and listens on http://localhost:11434.
We only use two of its endpoints:
  POST /api/embed  -> turn texts into vectors        (model: nomic-embed-text)
  POST /api/chat   -> generate the grounded answer   (model: qwen2.5:3b)
"""
from __future__ import annotations

import httpx

from .config import load_config


class OllamaError(SystemExit):
    """Stops the program with a beginner-friendly message (no long traceback)."""


def _cfg() -> dict:
    return load_config()["ollama"]


def _post(endpoint: str, payload: dict) -> dict:
    cfg = _cfg()
    url = cfg["base_url"].rstrip("/") + endpoint
    try:
        # trust_env=False: never send local Ollama traffic through a company proxy
        resp = httpx.post(url, json=payload, timeout=cfg.get("timeout_seconds", 600), trust_env=False)
    except httpx.ConnectError:
        raise OllamaError(
            f"\nCannot reach Ollama at {cfg['base_url']}.\n"
            "Fix: start the Ollama app (llama icon near the clock), or run `ollama serve` in another terminal, "
            "then try again.")
    except httpx.TimeoutException:
        raise OllamaError("\nOllama took too long to answer. Try again, or raise `timeout_seconds` in config.yaml.")
    if resp.status_code == 404 and "not found" in resp.text.lower():
        model = payload.get("model")
        raise OllamaError(f"\nThe Ollama model '{model}' is not installed.\nFix: run   ollama pull {model}")
    if resp.status_code != 200:
        raise OllamaError(f"\nOllama returned an error ({resp.status_code}): {resp.text[:300]}")
    return resp.json()


def embed(texts: list[str], prefix: str = "") -> list[list[float]]:
    """Embed many texts (in batches). `prefix` is the task prefix nomic-embed-text expects."""
    cfg = _cfg()
    vectors: list[list[float]] = []
    batch_size = cfg.get("embed_batch_size", 32)
    for i in range(0, len(texts), batch_size):
        batch = [prefix + t.replace("\n", " ") for t in texts[i:i + batch_size]]
        data = _post("/api/embed", {"model": cfg["embedding_model"], "input": batch, "truncate": True})
        vectors += data["embeddings"]
    return vectors


def chat(messages: list[dict]) -> str:
    cfg = _cfg()
    data = _post("/api/chat", {
        "model": cfg["chat_model"],
        "messages": messages,
        "stream": False,
        "options": {
            "temperature": cfg.get("temperature", 0),
            "seed": 42,                                  # same question -> same answer
            "num_ctx": cfg.get("num_ctx", 8192),         # room for the retrieved context
        },
    })
    return data["message"]["content"].strip()


def check_ready() -> None:
    """Fail early with a clear message if Ollama is not running or a model is missing."""
    cfg = _cfg()
    try:
        resp = httpx.get(cfg["base_url"].rstrip("/") + "/api/tags", timeout=10, trust_env=False)
        installed = {m["name"] for m in resp.json().get("models", [])}
    except (httpx.HTTPError, ValueError):
        raise OllamaError(
            f"\nCannot reach Ollama at {cfg['base_url']}.\n"
            "Fix: start the Ollama app (llama icon near the clock), or run `ollama serve` in another terminal.")
    short = {n.split(":")[0] for n in installed} | {n.removesuffix(":latest") for n in installed}
    for model in (cfg["embedding_model"], cfg["chat_model"]):
        if model not in installed and model not in short:
            raise OllamaError(f"\nThe Ollama model '{model}' is not installed.\nFix: run   ollama pull {model}")
