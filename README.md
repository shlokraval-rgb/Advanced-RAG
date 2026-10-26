# Advanced-RAG

Three **document-aware** Retrieval-Augmented Generation (RAG) pipelines in one small Python project,
one for each use case of the *Advanced RAG Hands-On* assignment:

| # | Document | What makes it hard | What this project does about it |
|---|----------|--------------------|---------------------------------|
| 1 | **GPO 2018 Annual Report** (82 p.) | Plain narrative text, two columns, photos, a "By the Numbers" table, plus a separate financial report | Column-aware reading order, header/footer removal, paragraph-aware chunks, captions kept apart, stats kept as `label: value` lines |
| 2 | **World Bank – Global Economic Prospects, June 2026** (200 p.) | Long; chapters → sections → subsections, 6 regions, boxes, tables | Hierarchy from PDF bookmarks **verified** against real headings, chapter/section/region/box metadata on every chunk, question analysis → metadata filters, tables rebuilt row by row |
| 3 | **Apple iMac Product Environmental Report** (12 p.) | Facts sit in a chart, small tables, icon call-outs and footnotes | Chart rebuilt from label positions (only labelled values), tables rebuilt row by row, call-outs as own chunks, footnotes linked to the claims that cite them, numbers validated against the PDF text |

All three share the same core: **PDF → document-specific parsing → document-specific chunking → metadata →
embeddings → ChromaDB → hybrid retrieval (vectors + BM25, with filters) → context → LLM → grounded answer with page references.**

**100% free and local:** embeddings and answers come from two free models running on your own PC with
[Ollama](https://ollama.com) – `nomic-embed-text` (embeddings) and `qwen2.5:3b` (answers).
No API key, no internet calls, no cost. (Designed for a normal Windows PC with 16 GB RAM; no GPU needed.)

---

## 1. Setup (one time)

You need **Python 3.10 or newer** and **Ollama**.

```bash
# 1. Install Ollama from https://ollama.com/download (free, no account), then download the two models:
ollama pull nomic-embed-text
ollama pull qwen2.5:3b

# 2. open a terminal in the Advanced-RAG folder, then create a virtual environment
python -m venv .venv

# 3. activate it
.venv\Scripts\activate          # Windows (PowerShell / cmd)
source .venv/bin/activate       # macOS / Linux

# 4. install the libraries
pip install -r requirements.txt
```

No `.env` file and no API key are needed. (`.env.example` only shows optional settings.)
Ollama must be running – after installation it starts automatically (llama icon near the clock).

## 2. Run it

```bash
# Build the indexes (parse + chunk + embed + store). Free; a few minutes on a normal CPU.
python scripts/ingest.py --doc all

# Ask questions (answers include page references). The first answer can take ~30-60 s while the
# model loads; later answers are faster.
python scripts/ask.py --doc gpo  "How many depository libraries did GPO staff visit in FY 2018?"
python scripts/ask.py --doc gep  "What is the growth outlook for South Asia in this report?"
python scripts/ask.py --doc imac "What is the total carbon footprint of the iMac, and how much comes from product use versus manufacturing?"

# Interactive mode (type questions, empty line to quit); --show-context prints retrieved text
python scripts/ask.py --doc gep --show-context

# Evaluate on the official sample questions + extra + negative questions (~10-20 min on a CPU)
python scripts/evaluate.py --verbose

# Unit tests (no Ollama needed; they use a temporary folder, your real index is not touched)
python -m pytest -q
```

Useful extras:

```bash
python scripts/ask.py --doc gpo "..." --show-prompt          # print the exact excerpts sent to the model
python scripts/show_chunks.py --doc gep --stats              # what was indexed, per section
python scripts/show_chunks.py --doc imac --type chart        # look at a specific chunk type
python scripts/show_chunks.py --doc gep --grep "TABLE 2.9"
python scripts/evaluate.py --doc gep --no-filters            # see what happens WITHOUT section-aware filters

```

### Offline test mode (no Ollama)

With `RAG_OFFLINE=1` (in `.env` or the terminal) everything runs without Ollama, using a simple "hashing"
embedding and **no LLM**: `ask.py` shows the retrieved evidence instead of an answer, and `evaluate.py`
reports retrieval metrics only. The unit tests use this mode. Offline and Ollama vectors are stored in
separate folders (`data/index/offline`, `data/index/ollama`).

---

## 3. What happens to a question

```
question ──► profile.analyze_query()       e.g. "South Asia" → filter region = South Asia
         ──► hybrid retrieval               vector search (meaning)  +  BM25 (exact words/numbers)
                                            merged with Reciprocal Rank Fusion, filters applied to both
         ──► top 6 chunks → context         each excerpt labelled "(p. 96 | Ch. 2 … > South Asia > Outlook | paragraph)"
         ──► qwen2.5:3b via Ollama (temp 0) "use ONLY the excerpts, cite pages, say 'not found' otherwise"
         ──► checks                         - refuse if nothing relevant was retrieved
                                            - repeated statements are removed (overlapping chunks are also
                                              de-duplicated before they reach the model)
                                            - every (p. N) citation is checked: if no excerpt on that page
                                              supports the sentence, the page of the excerpt that does is used
                                            - a sentence without a (p. N) citation gets the page of the
                                              excerpt it matches (shared numbers/words); reported as a note
                                            - warn if the answer contains a number that is not in the context
```

## 4. How each use case is document-aware

### Use case 1 – GPO (baseline, `src/advanced_rag/profiles/gpo.py`)
* Pages 1–30 are a two-column magazine layout → blocks are read **left column, then right column**
  (naive extraction zig-zags across columns and scrambles sentences).
* Running headers/footers are removed but used to learn the **printed page number** and **financial section** (Section I–IV).
* A paragraph cut by a column/page break is re-joined ("…over 35" + "million title retrievals…").
* Photo captions become separate small chunks so they don't interrupt paragraphs.
* Question words are mapped to the report's own words (`synonyms` in `config.yaml`), e.g. "fully deployed" →
  "full-scale deployment" – so the FDsys question retrieves p. 4 (govinfo's full-scale deployment during FY 2018)
  and not only the pages about FDsys's retirement in December 2018.
* **govinfo/FDsys date guard** (narrow, source-grounded): for questions about govinfo replacing FDsys, the p. 4
  passage "…during FY 2018 was the full-scale deployment of govinfo" is always retrieved first, and if the
  local model still gives govinfo's deployment the FDsys *retirement* date (December 2018), that sentence is
  replaced by both facts quoted from the retrieved report text with their pages. The change is reported under the answer.
* "GPO By the Numbers FY 2018" becomes clean `label: value` lines (e.g. `Tangible copies distributed to depository libraries: 802,834`).
* **Paragraph-aware chunking**: whole paragraphs packed to ~220 words, one paragraph of overlap, never crossing a section.

### Use case 2 – World Bank GEP (structure-aware, `profiles/gep.py`)
* Hierarchy from the **PDF bookmarks**, then **verified** against the real headings (font size/weight).
  The ingest log shows the result – e.g. `Region South Asia  bookmark p.95  CORRECTED -> heading is on p.96`.
* Every chunk carries `chapter`, `chapter_num`, `section`, `subsection`, `region`, `box_id`, page numbers,
  and a short header such as `[World Bank GEP June 2026 | Ch. 2 Regional Outlooks > South Asia > Outlook | PDF p. 96 | region: South Asia]`.
* Chunks never cross a section boundary → South Asia text is never mixed with Sub-Saharan Africa text.
* **Query analysis → metadata filters**: region names/abbreviations (South Asia, SAR, …), "Chapter 3",
  "executive summary", "Box 1.1", "statistical annex". Retrieval then only searches that part of the report.
  If two regions are mentioned, no region filter is used.
* Tables (e.g. *TABLE 2.9 South Asia forecast summary*) are rebuilt **row by row from word positions**;
  long tables are split with the column headers repeated.
* Removed noise: chart axis ticks/legends, contents pages, abbreviations, reference lists.
* **Regional outlook completeness**: if the answer to a regional outlook question leaves out one of the region's
  headline forecasts (e.g. South Asia: 6.3 percent in 2026 and about 7 percent per year in 2027–28, p. 96),
  that sentence is added, quoted from the retrieved Outlook section with its page.
  Fixed text: broken "Th" ligature ("Te" → "The"), "ﬁ" ligatures, hyphenation.

### Use case 3 – iMac (visual / scattered facts, `profiles/imac.py`)
* **Chart (p. 2)**: rebuilt from the positions of its labels → three bars (445 kg baseline, –99 kg reductions,
  346 kg iMac) with their segment names. The chunk states that **only the totals are labelled** and the segments
  have no numbers – so the LLM must not estimate them. Arithmetic check: 445 – 99 = 346.
* **Tables** (p. 10 carbon footprint, p. 12 power consumption) rebuilt row by row: `• Product use | 45%`, `Sleep | 0.56W | 0.57W | 0.59W`.
* **Call-outs / icons** ("iMac uses 100% recycled aluminum in the stand", "76% recycled content in fiber packaging") become their own chunks.
* **Footnotes**: superscript markers are kept as `[^N]`; every chunk gets the text of the footnotes it cites,
  e.g. "58 percent less energy than … ENERGY STAR.[^18]" + endnote 18. Each endnote is also its own chunk.
  The p. 12 power table is recognised as part of endnote 18.
* **Validation**: every number in the chart/table descriptions is checked against the PDF text layer (see ingest log).
* **Answer checks against the retrieved excerpts**: a percentage given for a part (e.g. the stand) must come
  from a sentence about that part (100% recycled aluminum in the stand, not the device-wide 15%); a value for
  an unlabelled chart segment is replaced by what the report states (segments are not labelled); and an
  emissions value in kg CO2e is never reported as the iMac's weight (the report gives no weight).
* Small vocabulary bridge: questions saying "manufacturing" also search for "production" (the table's word),
  and the carbon table carries the report's own definitions of its life-cycle phases (p. 9: *Production*
  "includes … the manufacture, transport, and assembly of all parts"), so "manufacturing" questions get the 50% share.

## 5. About the vision model (not needed)

Every value in the iMac report (chart totals, tables, call-outs) also exists in the PDF's text layer, so the
profile rebuilds them **without** a vision model – deterministic, free and verifiable (see the `check:` lines
printed by `ingest.py`). `scripts/extract_imac_visuals.py` is an **optional, paid** extra (OpenAI vision) for
purely graphical content such as logos. It is **not** part of the normal pipeline and is not needed for the
assignment; it only runs if you explicitly start it with an OpenAI key.

## 6. Evaluation

`eval/questions.yaml` contains:
* the **9 official sample questions** from the assignment (3 per document),
* **7 extra questions** (other regions, Box 1.1, table cells, footnotes, chart values),
* **7 negative questions** (not in the documents, a non-existent chapter, and a chart "trap" asking for an unlabelled value).

| Metric | Meaning |
|--------|---------|
| `retrieval_hit` / rank | a retrieved chunk covers a gold page (rank 1 = best) |
| `evidence_found` | every expected fact (e.g. "166", "6.3", "45%") is in the retrieved context |
| `section_purity` | share of retrieved chunks from the expected region/chapter/section (World Bank) |
| `answer_correct` | the LLM answer contains every expected fact *(local LLM)* |
| `cites_gold_page` | the answer cites a correct page *(local LLM)* |
| `refused` | negative questions are answered "not found" *(local LLM)* |

Results from the offline run (retrieval only), all documents:

```
Passed 17/17 scored questions      gpo 3/3   gep 7/7   imac 7/7   (all official questions: answer page at rank 1)
With --no-filters (plain retrieval) gep drops to 5/6: South Asia purity 0.75, executive summary purity 0.50
```

Run `python scripts/evaluate.py --verbose` (with Ollama running) to get the full answer-level scores.
Detailed results are written to `eval/results/` (git-ignored).

## 7. Project structure

```
Advanced-RAG/
├── README.md  requirements.txt  config.yaml  .env.example  .gitignore
├── data/
│   ├── raw/            the 3 PDFs (+ the assignment document)
│   ├── extracted/      optional, reviewed artefacts (iMac vision cache)
│   ├── processed/      chunk files *.jsonl            (generated, git-ignored)
│   └── index/          ChromaDB vector store           (generated, git-ignored)
├── src/advanced_rag/
│   ├── config.py       config.yaml + .env, online/offline mode
│   ├── models.py       Element and Chunk data classes
│   ├── pdf_utils.py    PyMuPDF helpers: blocks, column order, table rows, text clean-up
│   ├── chunking.py     paragraph-aware packing, continuation merging
│   ├── ollama_client.py  talks to the local Ollama server (embeddings + chat)
│   ├── embeddings.py   Ollama embeddings (+ offline hashing stand-in for tests)
│   ├── store.py        chunk files, ChromaDB, BM25
│   ├── retriever.py    hybrid retrieval with metadata filters (RRF fusion)
│   ├── generator.py    context building, grounded prompt, number check
│   ├── pipeline.py     ingest() and ask()
│   └── profiles/       base.py, gpo.py, gep.py, imac.py  ← the document-specific parts
├── scripts/            ingest.py, ask.py, evaluate.py, show_chunks.py, extract_imac_visuals.py
├── eval/questions.yaml
└── tests/              pytest unit tests (offline)
```

## 8. Design choices (kept deliberately simple)

* **Free local models** via Ollama: `nomic-embed-text` (embeddings, with its `search_document:` /
  `search_query:` prefixes) and `qwen2.5:3b` (answers, temperature 0, fixed seed, 8k context).
  A 3B model is less fluent than big cloud models; the grounding rules, page labels and the
  "numbers must exist in the context" check keep its answers tied to the documents.
* **No LangChain / LlamaIndex** – every step is plain, readable Python.
* **Hybrid retrieval instead of a reranker model**: BM25 catches exact terms and numbers ("FDsys", "Table 2.9",
  "802,834"); vectors catch meaning. Together with metadata filters this already puts every official answer at
  rank 1, so a separate reranking model (extra download, extra latency) was not added.
* **One collection per document** – each use case is its own Q&A system, as the assignment describes.
* Settings (chunk size, top-k, models, similarity threshold) live in `config.yaml`.

## 9. Troubleshooting

| Problem | Fix |
|---------|-----|
| `Cannot reach Ollama at http://localhost:11434` | start the Ollama app (or run `ollama serve` in another terminal) |
| `The Ollama model 'qwen2.5:3b' is not installed` | run `ollama pull qwen2.5:3b` (same for `nomic-embed-text`) |
| `No index for 'gpo' yet` | run `python scripts/ingest.py --doc all` |
| Answers are slow | normal on a CPU; close other heavy programs. The first question also loads the model |
| `Ollama took too long to answer` | raise `timeout_seconds` in `config.yaml` |
| Want to start fresh | delete `data/index/` and `data/processed/`, then ingest again |
