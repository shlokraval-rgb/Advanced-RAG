"""Evaluate the RAG system on eval/questions.yaml.

Usage:
    python scripts/evaluate.py                 # all documents
    python scripts/evaluate.py --doc gep       # one document
    python scripts/evaluate.py --verbose       # also print answers

Metrics (per question):
  retrieval_hit    a retrieved chunk covers one of the gold pages
  first_hit_rank   rank of the first such chunk (1 = best)
  evidence_found   every expected fact (must_contain) is present in the retrieved context
  section_purity   share of retrieved chunks that have the expected metadata (World Bank)
  --- only in normal mode (local Ollama LLM) ---
  answer_correct   every expected fact is present in the generated answer
  cites_gold_page  the answer cites at least one gold page
  refused          negative questions: the answer is "not found" (that is the correct behaviour)
  unsupported_nums numbers in the answer that are not in the retrieved context (should be empty)
"""
import argparse
import json
import re
import time

import yaml

import _bootstrap  # noqa: F401
from advanced_rag import ollama_client
from advanced_rag.config import DOC_IDS, PROJECT_ROOT, is_offline
from advanced_rag.generator import NOT_FOUND, build_context
from advanced_rag.pipeline import ask


# Ways a (small, local) model says "the document doesn't say this". The prompt asks for the exact
# NOT_FOUND sentence, but a 3B model sometimes rephrases it slightly.
REFUSAL_PHRASES = [NOT_FOUND[:30], "could not find", "cannot find", "can't find", "not found in the",
                   "does not contain", "does not mention", "no information", "not mentioned", "not provided"]


def contains_all(text: str, groups: list[list[str]]) -> bool:
    low = text.lower()
    return all(any(alt.lower() in low for alt in group) for group in groups)


def sentences(text: str) -> list[str]:
    """Split an answer into sentences/bullets (never inside a citation like "(p. 5)")."""
    out = []
    for line in text.split("\n"):
        out += [s for s in re.split(r"(?<!\(p\.)(?<!pp\.)(?<! p\.)(?<=[.!?])\s+(?=[A-Z\-•*])", line) if s.strip()]
    return out


def check_answer(answer: str, q: dict) -> tuple[bool, list[str]]:
    """(correct?, wrong statements). Correct = not a give-up, every expected fact present, and no
    forbidden statement: `must_not_contain` (plain text) or `must_not_match` (regex, checked per sentence)."""
    low = answer.lower()
    gave_up = any(p in low for p in (NOT_FOUND.lower()[:30], "could not find"))
    wrong = [w for w in q.get("must_not_contain", []) if w.lower() in low]
    for pattern in q.get("must_not_match", []):
        wrong += [s for s in sentences(answer) if re.search(pattern, s, re.I)]
    return (not gave_up) and contains_all(answer, q["must_contain"]) and not wrong, wrong


def cited_pages(answer: str) -> set[int]:
    pages = set()
    for m in re.finditer(r"pp?\.\s*([\d\s,\-–and]+)", answer):
        for a, b in re.findall(r"(\d+)(?:\s*[-–]\s*(\d+))?", m.group(1)):
            lo, hi = int(a), int(b or a)
            pages.update(range(lo, min(hi, lo + 30) + 1))
    return pages


def evaluate_one(q: dict, use_filters: bool = True) -> dict:
    res = ask(q["doc"], q["question"], use_filters=use_filters)
    context = build_context(res.hits) if res.hits else ""
    row = {"id": q["id"], "doc": q["doc"], "question": q["question"], "negative": bool(q.get("negative")),
           "official": bool(q.get("official")), "filters": res.filters, "answer": res.answer,
           "warnings": res.warnings,
           "retrieved": [{"pages": h.chunk.pages, "type": h.chunk.element_type, "section": h.chunk.section_path}
                         for h in res.hits]}
    if not row["negative"]:
        gold = set(q["gold_pages"])
        ranks = [i for i, h in enumerate(res.hits, 1)
                 if gold & set(range(h.chunk.page_start, h.chunk.page_end + 1))]
        row["retrieval_hit"] = bool(ranks)
        row["first_hit_rank"] = ranks[0] if ranks else None
        row["evidence_found"] = contains_all(context, q["must_contain"])
        if q.get("expect_meta"):
            ok = [all(h.chunk.chroma_metadata().get(k) == v for k, v in q["expect_meta"].items()) for h in res.hits]
            row["section_purity"] = round(sum(ok) / len(ok), 2) if ok else 0.0
    if res.offline and row["negative"] and not res.hits:
        row["refused"] = True          # nothing retrievable (e.g. a chapter that doesn't exist)
    if not res.offline:
        # "not found", or (for trap questions) an explicit statement that the value is not given
        phrases = [p.lower() for p in REFUSAL_PHRASES + q.get("accept_refusals", [])]
        refused = any(p in res.answer.lower() for p in phrases)
        row["refused"] = refused
        if not row["negative"]:
            row["answer_correct"], row["wrong_statements"] = check_answer(res.answer, q)
            row["cites_gold_page"] = bool(cited_pages(res.answer) & set(q["gold_pages"]))
        row["unsupported_nums"] = [w for w in res.warnings if w.startswith("Numbers")]
    return row


def passed(row: dict, offline: bool) -> bool:
    if row["negative"]:
        return row.get("refused") if not offline else (True if row.get("refused") else None)
    ok = row["retrieval_hit"] and row["evidence_found"] and row.get("section_purity", 1.0) >= 0.75
    if not offline:
        ok = ok and row["answer_correct"] and row["cites_gold_page"]
    return ok


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--doc", choices=[*DOC_IDS, "all"], default="all")
    ap.add_argument("--verbose", action="store_true")
    ap.add_argument("--no-filters", action="store_true",
                    help="switch off section-aware filters to see what plain retrieval would do")
    args = ap.parse_args()

    offline = is_offline()
    if not offline:
        ollama_client.check_ready()
    questions = yaml.safe_load(open(PROJECT_ROOT / "eval" / "questions.yaml", encoding="utf-8"))
    questions = [q for q in questions if args.doc == "all" or q["doc"] == args.doc]
    mode = "OFFLINE test mode (retrieval metrics only)" if offline else "LOCAL MODELS (Ollama, free)"
    if args.no_filters:
        mode += " | section-aware filters DISABLED (comparison run)"
    print(f"Evaluating {len(questions)} questions - mode: {mode}\n")

    rows = []
    for q in questions:
        row = evaluate_one(q, use_filters=not args.no_filters)
        row["passed"] = passed(row, offline)
        rows.append(row)
        status = {True: "PASS", False: "FAIL", None: "n/a "}[row["passed"]]
        detail = []
        if not row["negative"]:
            detail.append(f"hit={'Y' if row['retrieval_hit'] else 'N'}@{row['first_hit_rank'] or '-'}")
            detail.append(f"evidence={'Y' if row['evidence_found'] else 'N'}")
            if "section_purity" in row:
                detail.append(f"purity={row['section_purity']:.2f}")
            if not offline:
                detail.append(f"answer={'Y' if row['answer_correct'] else 'N'}")
                detail.append(f"cites={'Y' if row['cites_gold_page'] else 'N'}")
        elif not offline or row.get("refused"):
            detail.append(f"refused={'Y' if row['refused'] else 'N'}")
        else:
            detail.append("needs LLM")
        kind = "NEG" if row["negative"] else ("OFF" if row["official"] else "EXT")
        print(f"[{status}] {row['id']:<12} {kind}  {' '.join(detail):<48} {row['question'][:70]}")
        if args.verbose and not offline:
            print("        " + row["answer"].replace("\n", "\n        "))
            for w in row["warnings"]:
                print(f"        [!] {w}")

    scored = [r for r in rows if r["passed"] is not None]
    n_pass = sum(bool(r["passed"]) for r in scored)
    print(f"\nPassed {n_pass}/{len(scored)} scored questions  (OFF = official sample, EXT = extra, NEG = negative)")
    for doc in DOC_IDS:
        d = [r for r in scored if r["doc"] == doc]
        if d:
            print(f"  {doc:<5} {sum(bool(r['passed']) for r in d)}/{len(d)}")

    out_dir = PROJECT_ROOT / "eval" / "results"
    out_dir.mkdir(parents=True, exist_ok=True)
    stamp = time.strftime("%Y%m%d-%H%M%S")
    out = out_dir / f"eval_{'offline' if offline else 'online'}_{stamp}.json"
    out.write_text(json.dumps(rows, indent=2, ensure_ascii=False), encoding="utf-8")
    print(f"\nDetailed results: {out.relative_to(PROJECT_ROOT)}")


if __name__ == "__main__":
    main()
