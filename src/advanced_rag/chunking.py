"""Shared chunking helpers. Each profile decides how to GROUP elements; this file packs them."""
from __future__ import annotations

import re

from .models import Chunk, Element
from .pdf_utils import word_count


def merge_continuations(elements: list[Element]) -> list[Element]:
    """Join a paragraph that was cut by a column or page break with its continuation.

    Example: "...experienced over 35" (end of p. 4) + "million title retrievals each month." (p. 5)
    """
    out: list[Element] = []
    last_par: dict = {}        # section -> index of its last paragraph (captions/boxes may sit in between)
    for el in elements:
        key = el.meta.get("section_key", el.meta.get("section"))
        idx = last_par.get(key)
        prev = out[idx] if idx is not None else None
        if (prev and el.element_type == "paragraph"
                and el.page - prev.meta.get("page_end", prev.page) <= 3
                and not re.search(r"[.!?:;\"”)\]]$", prev.text) and el.text[:1].islower()):
            joiner = "" if prev.text.endswith("-") and prev.text[-2:-1].isalpha() else " "
            text = (prev.text[:-1] if joiner == "" else prev.text) + joiner + el.text
            out[idx] = Element(text, prev.page, "paragraph", {**prev.meta, "page_end": el.page})
            continue
        out.append(el)
        if el.element_type == "paragraph":
            last_par[key] = len(out) - 1
    return out


def split_long_text(text: str, max_words: int) -> list[str]:
    """Split an over-long paragraph on sentence boundaries."""
    sentences = re.split(r"(?<=[.!?])\s+(?=[A-Z(\"“])", text)
    parts, current = [], ""
    for s in sentences:
        if current and word_count(current) + word_count(s) > max_words:
            parts.append(current)
            current = s
        else:
            current = f"{current} {s}".strip()
    if current:
        parts.append(current)
    return parts


def pack_elements(elements: list[Element], max_words: int, overlap: int = 1) -> list[list[Element]]:
    """Paragraph-aware packing: whole paragraphs are added until the chunk is ~max_words.

    - A paragraph is never cut in the middle (unless it alone is far too long).
    - `overlap` paragraphs from the end of one chunk are repeated at the start of the next one,
      so a fact that sits at a chunk border is still found with its context.
    """
    # 1) break up very long single elements
    items: list[Element] = []
    for el in elements:
        if word_count(el.text) > max_words * 1.5 and el.element_type in ("paragraph", "footnote"):
            for part in split_long_text(el.text, max_words):
                items.append(Element(part, el.page, el.element_type, dict(el.meta)))
            # (a split piece keeps the page range of the original paragraph)
        else:
            items.append(el)

    # 2) pack
    groups: list[list[Element]] = []
    current: list[Element] = []
    carried = 0                      # how many elements at the start of `current` are overlap copies
    for el in items:
        size = sum(word_count(e.text) for e in current)
        if current and size + word_count(el.text) > max_words:
            if len(current) > carried:           # don't emit a chunk that is only overlap
                groups.append(current)
                tail = current[-overlap:] if overlap else []
                # only carry short paragraphs as overlap (keeps chunks from growing)
                current = [e for e in tail if word_count(e.text) <= max_words // 2]
            else:
                current = []
            carried = len(current)
        current.append(el)
    if current and len(current) > carried:
        groups.append(current)
    return groups


def make_chunk(doc_id: str, index: int, elements: list[Element], header: str, section_path: str,
               element_type: str, metadata: dict | None = None, body: str | None = None) -> Chunk:
    """Create a Chunk. The short `header` is prepended so the embedding 'knows' where the text is from."""
    pages = [e.page for e in elements] + [e.meta["page_end"] for e in elements if e.meta.get("page_end")]
    text_body = body if body is not None else "\n".join(e.text for e in elements)
    return Chunk(
        chunk_id=f"{doc_id}-{index:04d}",
        doc_id=doc_id,
        text=f"{header}\n{text_body}".strip(),
        page_start=min(pages),
        page_end=max(pages),
        element_type=element_type,
        section_path=section_path,
        metadata=metadata or {},
    )
