"""Ask a question about one document.

Usage:
    python scripts/ask.py --doc gpo "How many depository libraries did GPO staff visit in FY 2018?"
    python scripts/ask.py --doc gep "What is the growth outlook for South Asia?" --show-context
    python scripts/ask.py --doc imac            (interactive mode: type questions, empty line to quit)
"""
import argparse

import _bootstrap  # noqa: F401
from advanced_rag import ollama_client
from advanced_rag.config import DOC_IDS, is_offline
from advanced_rag.generator import build_context
from advanced_rag.pipeline import ask


def print_answer(doc_id: str, question: str, show_context: bool, show_prompt: bool = False) -> None:
    res = ask(doc_id, question)
    if show_prompt and res.hits:
        print("\n" + "#" * 80 + "\nEXACT CONTEXT SENT TO THE LOCAL MODEL:\n" + "#" * 80)
        print(build_context(res.hits))
    print("\n" + "=" * 80)
    print(f"Q: {question}")
    if res.filters:
        print(f"   (section-aware filter applied: {res.filters})")
    for note in res.plan_notes:
        print(f"   note: {note}")
    print("-" * 80)
    print(res.answer)
    for w in res.warnings:
        print(f"\n[!] {w}")
    print("\nSources (retrieved, best first):")
    for i, h in enumerate(res.hits, 1):
        c = h.chunk
        print(f"  {i}. PDF p. {c.pages:<7} {c.element_type:<10} {c.section_path}")
        if show_context or res.offline:
            body = c.text.split("\n", 1)[-1].replace("\n", " ")
            print(f"       {body[:400]}{'...' if len(body) > 400 else ''}")
    print("=" * 80)


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--doc", required=True, choices=DOC_IDS)
    ap.add_argument("question", nargs="?", help="leave empty for interactive mode")
    ap.add_argument("--show-context", action="store_true", help="print the retrieved text too")
    ap.add_argument("--show-prompt", action="store_true", help="print the exact excerpts sent to the model")
    args = ap.parse_args()
    if not is_offline():
        ollama_client.check_ready()

    if args.question:
        print_answer(args.doc, args.question, args.show_context, args.show_prompt)
        return
    print(f"Interactive mode for '{args.doc}'. Empty line to quit.")
    while True:
        q = input("\nQuestion> ").strip()
        if not q:
            break
        print_answer(args.doc, q, args.show_context, args.show_prompt)


if __name__ == "__main__":
    main()
