"""Small, reusable PDF helpers built on PyMuPDF.

The profiles (gpo.py, gep.py, imac.py) decide WHAT to do with a document;
these helpers only know HOW to read text blocks, fix text and order columns.
"""
from __future__ import annotations

import re
import unicodedata
from collections import Counter

import pymupdf

# Some PDF fonts lose the "Th" ligature, so "The" is extracted as "Te" (seen in the World Bank PDF).
_BROKEN_TH = {
    "Te": "The", "Tis": "This", "Tese": "These", "Tey": "They", "Tat": "That",
    "Teir": "Their", "Tere": "There", "Tem": "Them", "Tose": "Those", "Tus": "Thus",
    "Terefore": "Therefore", "Tird": "Third", "Trough": "Through", "Tough": "Though",
}
_BROKEN_TH_RE = re.compile(r"\b(" + "|".join(_BROKEN_TH) + r")\b")

# When a line ends with a hyphen we normally join the word ("sustainabil-" + "ity").
# If the next part is one of these, it is a real compound word ("low-" + "carbon").
_COMPOUND_SECOND_PARTS = {
    "carbon", "income", "term", "based", "run", "level", "related", "led", "oriented", "specific",
    "free", "year", "owned", "making", "scale", "wide", "time", "growth", "market", "border",
    "price", "intensive", "risk", "air", "end", "life", "by", "to", "of", "in", "up",
}


def clean_text(text: str) -> str:
    text = unicodedata.normalize("NFKC", text)          # "ﬁ" -> "fi", non-breaking spaces, etc.
    text = _BROKEN_TH_RE.sub(lambda m: _BROKEN_TH[m.group(1)], text)
    text = re.sub(r"CO\s?2\s?e\b", "CO2e", text)       # "CO 2 e" (subscript) -> "CO2e"
    text = re.sub(r"\(\s*continued\s*\)", "(continued)", text)
    text = text.replace("\t", " ")
    text = re.sub(r"[  ]+", " ", text)
    return text.strip()


def _join_lines(lines: list[str]) -> str:
    out = ""
    for line in lines:
        line = line.strip()
        if not line:
            continue
        if out.endswith("-") and line[:1].islower() and len(out) > 1 and out[-2].isalpha():
            m = re.match(r"[a-z]+", line)
            first = m.group(0) if m else ""
            if first in _COMPOUND_SECOND_PARTS:
                out += line                      # keep hyphen: "low-carbon"
            else:
                out = out[:-1] + line            # drop hyphen: "sustainability"
        else:
            out = f"{out} {line}" if out else line
    return out


def page_blocks(page: pymupdf.Page, mark_footnotes: bool = False) -> list[dict]:
    """Return the text blocks of a page with position and dominant font information.

    If mark_footnotes=True, superscript numbers (footnote markers) become "[^N]".
    """
    blocks = []
    for b in page.get_text("dict")["blocks"]:
        if b.get("type") != 0:
            continue
        lines, sizes, fonts = [], Counter(), Counter()
        for line in b["lines"]:
            parts = []
            for s in line["spans"]:
                t = s["text"]
                if not t.strip():
                    parts.append(t)
                    continue
                is_sup = bool(s["flags"] & 1)
                if mark_footnotes and is_sup and re.fullmatch(r"\s*\d{1,2}\s*", t):
                    parts.append(f"[^{t.strip()}]")
                    continue
                parts.append(t)
                sizes[round(s["size"], 1)] += len(t)
                fonts[s["font"]] += len(t)
            lines.append("".join(parts))
        text = clean_text(_join_lines(lines))
        if not text:
            continue
        # A footnote marker should stick to the word before it: "audits. [^12]" -> "audits.[^12]"
        text = re.sub(r"\s+\[\^", "[^", text)
        blocks.append({
            "text": text,
            "bbox": tuple(b["bbox"]),
            "size": sizes.most_common(1)[0][0] if sizes else 0.0,
            "font": fonts.most_common(1)[0][0] if fonts else "",
            "n_lines": len(b["lines"]),
        })
    return blocks


def order_blocks(blocks: list[dict], tolerance: float = 15.0) -> list[dict]:
    """Put blocks in natural reading order for 1- or 2-column pages.

    Full-width blocks (titles, wide tables) act as separators. Between two separators we read the
    whole LEFT column first and then the RIGHT column - instead of zig-zagging across columns,
    which is what naive extraction does and what scrambles sentences.
    """
    if not blocks:
        return []
    x0 = min(b["bbox"][0] for b in blocks)
    x1 = max(b["bbox"][2] for b in blocks)
    mid = (x0 + x1) / 2

    left, right, full = [], [], []
    for b in blocks:
        bx0, _, bx1, _ = b["bbox"]
        if bx1 <= mid + tolerance:
            left.append(b)
        elif bx0 >= mid - tolerance:
            right.append(b)
        else:
            full.append(b)

    def by_y(items):
        return sorted(items, key=lambda b: (b["bbox"][1], b["bbox"][0]))

    left, right, full = by_y(left), by_y(right), by_y(full)
    ordered = []
    for sep in full:
        y = sep["bbox"][1]
        ordered += [b for b in left if b["bbox"][1] < y] + [b for b in right if b["bbox"][1] < y]
        left = [b for b in left if b["bbox"][1] >= y]
        right = [b for b in right if b["bbox"][1] >= y]
        ordered.append(sep)
    return ordered + left + right


def table_rows(page: pymupdf.Page, rect: pymupdf.Rect | None = None, y_tol: float = 2.5,
               gap: float = 9.0) -> list[str]:
    """Rebuild table rows from individual words: words with the same vertical position form one row.

    Columns are separated by " | " when the horizontal gap between words is large.
    Used for World Bank forecast tables, where block extraction merges unrelated cells.
    """
    words = page.get_text("words", clip=rect) if rect else page.get_text("words")
    rows: list[list] = []
    for w in sorted(words, key=lambda w: ((w[1] + w[3]) / 2, w[0])):
        yc = (w[1] + w[3]) / 2
        if rows and abs(rows[-1][0] - yc) <= y_tol:
            rows[-1][1].append(w)
        else:
            rows.append([yc, [w]])
    out = []
    for _, ws in rows:
        ws.sort(key=lambda w: w[0])
        line, prev_x1 = "", None
        for w in ws:
            sep = "" if prev_x1 is None else (" | " if w[0] - prev_x1 > gap else " ")
            line += sep + w[4]
            prev_x1 = w[2]
        line = clean_text(line)
        if line:
            out.append(line)
    return out


def word_count(text: str) -> int:
    return len(text.split())
