"""OPTIONAL and PAID - NOT part of the normal pipeline. You do NOT need to run this.

Describes visual-only content of the iMac report with an OpenAI vision model (needs OPENAI_API_KEY
with credits). The normal pipeline already gets every iMac value from the PDF text layer for free.

Why optional?  Every number in this PDF (chart totals, tables, call-outs) also exists in the PDF's
text layer, and the iMac profile already rebuilds them from text positions - deterministically and
for free. What the text layer does NOT contain is purely graphical content (e.g. which logos are shown
on page 1). This script captures that, ONCE, and caches the result.

Safety: every statement the model returns is checked - if it contains a number that does not exist
in that page's text layer, the statement is marked validated=false and is NOT indexed.

Usage:
    python scripts/extract_imac_visuals.py              # default pages 1 2 3 6
    python scripts/extract_imac_visuals.py --pages 1 2
    python scripts/ingest.py --doc imac                 # re-ingest to include the validated descriptions
Output: data/extracted/imac_vision.json  (review it - you can edit or delete entries by hand)
"""
import argparse
import base64
import json
import os

import _bootstrap  # noqa: F401
import pymupdf
from advanced_rag.config import doc_config, is_offline, load_config, path  # noqa: E402
from advanced_rag.profiles.imac import validate_numbers

PROMPT = """You are extracting information from page {page} of Apple's iMac Product Environmental Report.
Describe ONLY content that is visual (charts, icons, logos, badges, call-out graphics, product images).
Rules:
- Copy numbers exactly as printed. Never estimate a value that is not printed (e.g. unlabelled bar segments).
- If a chart segment has no printed number, say "not labelled".
- One short factual statement per item.
Return JSON: {{"items": [{{"type": "chart|icon|logo|callout|image", "text": "..."}}]}}"""


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--pages", type=int, nargs="+", default=[1, 2, 3, 6])
    args = ap.parse_args()
    if not os.getenv("OPENAI_API_KEY", "").startswith("sk-") or is_offline():
        raise SystemExit("This OPTIONAL, paid step needs OPENAI_API_KEY in .env. It is not needed for the "
                         "assignment - the normal pipeline is free and uses local Ollama models.")

    from openai import OpenAI
    client = OpenAI()
    model = load_config()["openai"]["vision_model"]
    doc = pymupdf.open(path("raw_dir") / doc_config("imac")["file"])

    items = []
    for pn in args.pages:
        page = doc[pn - 1]
        png = page.get_pixmap(dpi=150).tobytes("png")
        resp = client.chat.completions.create(
            model=model, temperature=0, response_format={"type": "json_object"},
            messages=[{"role": "user", "content": [
                {"type": "text", "text": PROMPT.format(page=pn)},
                {"type": "image_url", "image_url": {"url": "data:image/png;base64," + base64.b64encode(png).decode()}},
            ]}],
        )
        found = json.loads(resp.choices[0].message.content).get("items", [])
        page_text = page.get_text()
        for it in found:
            bad = validate_numbers(it.get("text", ""), page_text)
            items.append({"page": pn, "type": it.get("type"), "text": it.get("text", ""),
                          "validated": not bad, "unverified_numbers": bad})
            mark = "OK " if not bad else "REJECTED (numbers not in PDF text: %s)" % bad
            print(f"p.{pn} [{mark}] {it.get('text', '')}")

    out = path("extracted_dir") / "imac_vision.json"
    out.write_text(json.dumps({"model": model, "items": items}, indent=2, ensure_ascii=False), encoding="utf-8")
    print(f"\nSaved {len(items)} items ({sum(i['validated'] for i in items)} validated) to {out}")
    print("Now re-run:  python scripts/ingest.py --doc imac")


if __name__ == "__main__":
    main()
