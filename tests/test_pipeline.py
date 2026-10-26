"""End-to-end tests of retrieval (offline) and of the local-LLM answer step with a FAKE Ollama server."""
import json
import re
import threading
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path

import pytest

from advanced_rag import generator, pipeline
from advanced_rag.config import load_config
from advanced_rag.embeddings import HashingEmbedder


def test_retrieval_finds_official_answers(ingested):
    cases = [
        ("gpo", "How many depository libraries did GPO staff visit in FY 2018?", "166 depository libraries"),
        ("gep", "What is the growth outlook for South Asia in this report?", "6.3 percent"),
        ("imac", "How much less energy does iMac use compared to the ENERGY STAR requirement?", "58 percent"),
    ]
    for doc, q, expected in cases:
        res = pipeline.ask(doc, q)
        assert any(expected in h.chunk.text for h in res.hits), (doc, q)


def test_section_filter_applied_and_respected(ingested):
    res = pipeline.ask("gep", "What is the growth outlook for South Asia in this report?")
    assert res.filters == {"region": "South Asia"}
    assert all(h.chunk.metadata.get("region") == "South Asia" for h in res.hits)


def test_nonexistent_chapter_returns_not_found(ingested):
    res = pipeline.ask("gep", "What does Chapter 6 say about climate migration?")
    assert res.hits == [] and res.answer == generator.NOT_FOUND


# ---------------------------------------------------------------- local LLM path (fake Ollama server)
# A tiny HTTP server that speaks the same API as Ollama, so the REAL client code (httpx requests,
# prefixes, options, error messages) is exercised without installing Ollama.

class FakeOllama(BaseHTTPRequestHandler):
    answer = ""
    requests: list = []
    models = ["nomic-embed-text:latest", "qwen2.5:3b"]

    def log_message(self, *args):          # keep test output quiet
        pass

    def _send(self, code, payload):
        body = json.dumps(payload).encode()
        self.send_response(code)
        self.send_header("Content-Type", "application/json")
        self.send_header("Content-Length", str(len(body)))
        self.end_headers()
        self.wfile.write(body)

    def do_GET(self):
        self._send(200, {"models": [{"name": n} for n in FakeOllama.models]})

    def do_POST(self):
        payload = json.loads(self.rfile.read(int(self.headers["Content-Length"])))
        FakeOllama.requests.append((self.path, payload))
        if self.path == "/api/embed":
            self._send(200, {"embeddings": HashingEmbedder().embed(payload["input"])})
        elif self.path == "/api/chat":
            self._send(200, {"message": {"role": "assistant", "content": FakeOllama.answer}})
        else:
            self._send(404, {"error": "not found"})


@pytest.fixture
def fake_ollama(monkeypatch, tmp_path):
    server = ThreadingHTTPServer(("127.0.0.1", 0), FakeOllama)
    threading.Thread(target=server.serve_forever, daemon=True).start()
    cfg = load_config()
    monkeypatch.setitem(cfg["ollama"], "base_url", f"http://127.0.0.1:{server.server_port}")
    monkeypatch.setitem(cfg["paths"], "index_dir", str(tmp_path / "index"))
    monkeypatch.setenv("RAG_OFFLINE", "0")
    FakeOllama.requests = []
    pipeline._retrievers.clear()
    pipeline.ingest("gpo")                       # embeds through the (fake) Ollama /api/embed
    yield FakeOllama
    server.shutdown()
    pipeline._retrievers.clear()


def test_no_api_key_needed_and_prefixes_used(fake_ollama, monkeypatch):
    monkeypatch.delenv("OPENAI_API_KEY", raising=False)
    fake_ollama.answer = "GPO staff visited 166 depository libraries in 25 states in FY 2018 (p. 5)."
    res = pipeline.ask("gpo", "How many depository libraries did GPO staff visit in FY 2018?")
    embeds = [p for path, p in fake_ollama.requests if path == "/api/embed"]
    assert embeds[0]["model"] == "nomic-embed-text"
    assert embeds[0]["input"][0].startswith("search_document: ")     # documents at ingest
    assert embeds[-1]["input"][0].startswith("search_query: ")       # the question at query time
    assert res.answer.startswith("GPO staff visited 166") and res.warnings == []


def test_prompt_contains_grounding_rules_and_page_labels(fake_ollama):
    fake_ollama.answer = "166 (p. 5)"
    pipeline.ask("gpo", "How many depository libraries did GPO staff visit in FY 2018?")
    path, chat = fake_ollama.requests[-1]
    assert path == "/api/chat" and chat["model"] == "qwen2.5:3b"
    assert chat["options"]["temperature"] == 0 and chat["options"]["num_ctx"] >= 8192
    system, user = chat["messages"][0]["content"], chat["messages"][1]["content"]
    assert "ONLY" in system and generator.NOT_FOUND in system
    assert "(p. " in user and "Question:" in user


def test_invented_number_is_flagged(fake_ollama):
    fake_ollama.answer = "GPO staff visited 999 depository libraries (p. 5)."
    res = pipeline.ask("gpo", "How many depository libraries did GPO staff visit in FY 2018?")
    assert res.warnings and "999" in res.warnings[0]


def test_end_to_end_wrong_govinfo_date_is_corrected(fake_ollama):
    """The exact wrong answer seen with qwen2.5:3b goes through the whole pipeline and comes out fixed."""
    fake_ollama.answer = "- govinfo replaced FDsys (p. 4).\n- govinfo was fully deployed in December 2018 (p. 4)."
    res = pipeline.ask("gpo", "What new digital system replaced FDsys, and when was it fully deployed?")
    assert "fully deployed in December 2018" not in res.answer
    assert "during FY 2018 (p. 4)" in res.answer and "December 2018 is the date for retiring FDsys" in res.answer
    assert any(w.startswith("Date check") for w in res.warnings)
    assert not any(w.startswith("Numbers not found") for w in res.warnings)


def test_friendly_errors(fake_ollama, monkeypatch):
    from advanced_rag import ollama_client
    monkeypatch.setattr(FakeOllama, "models", ["nomic-embed-text:latest"])
    with pytest.raises(SystemExit, match="ollama pull qwen2.5:3b"):
        ollama_client.check_ready()
    monkeypatch.setitem(load_config()["ollama"], "base_url", "http://127.0.0.1:9")
    with pytest.raises(SystemExit, match="Cannot reach Ollama"):
        ollama_client.check_ready()


def test_no_paid_api_in_normal_pipeline():
    """The normal pipeline (src/) must never import or call OpenAI."""
    src = Path(__file__).resolve().parents[1] / "src"
    for f in src.rglob("*.py"):
        text = f.read_text(encoding="utf-8").lower()
        assert "import openai" not in text and "from openai" not in text, f
        assert "api.openai.com" not in text, f


def test_end_to_end_wrong_page_citation_is_corrected(fake_ollama):
    """gpo-3: the real local-model answer cited p. 16 for a fact that is on p. 5."""
    fake_ollama.answer = ("- GPO procured more than $375 million worth of printing and other information "
                          "products in FY 2018 (p. 16).")
    res = pipeline.ask("gpo", "How much did GPO procure in printing and information products in FY 2018 "
                              "on behalf of federal agencies?")
    assert "(p. 16)" not in res.answer and re.search(r"\(pp?\. 5", res.answer)
    assert any(w.startswith("Citation check") for w in res.warnings)
