"""USE CASE 2 - World Bank Global Economic Prospects (June 2026): advanced, structure-aware RAG.

What is document-specific here (and why):
  * 200 pages with chapters -> sections -> subsections, regional sections, boxes and tables.
  * The hierarchy comes from the PDF bookmarks, but every bookmark is VERIFIED against the real
    heading on the page (e.g. the "South Asia" bookmark points to p. 95, while the heading is on p. 96).
  * Headings are recognised by font: level-1 = bold Helvetica >= 14pt, level-2 = regular Helvetica ~11pt.
  * Every chunk stores chapter / section / subsection / region / box metadata, and never crosses a
    section boundary, so "South Asia" text can never be mixed with "Sub-Saharan Africa" text.
  * Questions are analysed: "South Asia" / "SAR" -> region filter, "Chapter 3" -> chapter filter,
    "executive summary" -> section filter. Retrieval only searches inside that part of the report.
  * Tables (e.g. "TABLE 2.9 South Asia forecast summary") are rebuilt row by row from word positions.
  * Chart internals (axis ticks, legends, tiny source notes) are dropped; figure titles + captions kept.
  * Reference lists, contents pages, abbreviations are not indexed (lots of words, no answers).
"""
from __future__ import annotations

import re

import pymupdf

from ..chunking import make_chunk, merge_continuations, pack_elements
from ..generator import page_label
from ..models import Chunk, Element
from ..pdf_utils import order_blocks, page_blocks, table_rows
from ..retriever import QueryPlan
from .base import BaseProfile


def _norm(s: str) -> str:
    return re.sub(r"\s+", " ", s.replace("--", " ")).strip()


def _is_helvetica(font: str) -> bool:
    return font.startswith("Helvetica")


class GEPProfile(BaseProfile):
    doc_id = "gep"

    def __init__(self):
        super().__init__()
        self.regions = {name: [a.lower() for a in aliases] for name, aliases in self.cfg["regions"].items()}
        self.log: list[str] = []      # what the structure verification found (printed by ingest)

    # ================================================================== structure (bookmarks + checks)
    def chapter_ranges(self, doc: pymupdf.Document) -> list[dict]:
        """Top-level parts of the report from the bookmarks: Executive Summary, Ch. 1-4, Statistical Annex."""
        tops = []
        for _lvl, title, page in doc.get_toc():
            title = _norm(title)
            m = re.match(r"Ch\. (\d) (.*)", title)
            if m:
                tops.append({"start": page, "num": int(m.group(1)), "title": f"Ch. {m.group(1)} {m.group(2)}"})
            elif title in ("Summary of Contents", "Contents", "Executive Summary", "Statistical Annex"):
                tops.append({"start": page, "num": 0, "title": title})
        tops.sort(key=lambda t: t["start"])
        for t, nxt in zip(tops, tops[1:] + [None]):
            t["end"] = (nxt["start"] - 1) if nxt else doc.page_count
        # verify chapter starts: the opening page must say "CHAPTER N"
        for t in tops:
            if t["num"]:
                text = doc[t["start"] - 1].get_text()
                ok = f"CHAPTER {t['num']}" in text
                self.log.append(f"{t['title'][:45]:<45} bookmark p.{t['start']:<4} "
                                f"{'verified (CHAPTER heading found)' if ok else 'WARNING: heading not found'}")
        return tops

    def verify_region_starts(self, doc: pymupdf.Document) -> dict[str, int]:
        """Region bookmarks are sometimes off by a page; find the real heading near the bookmark."""
        found = {}
        for _lvl, title, page in doc.get_toc():
            title = _norm(title)
            if title not in self.regions:
                continue
            actual = None
            for p in range(max(1, page - 1), min(doc.page_count, page + 2) + 1):
                for b in page_blocks(doc[p - 1]):
                    if _is_helvetica(b["font"]) and b["size"] >= 14 and _norm(b["text"]) == title:
                        actual = p
                        break
                if actual:
                    break
            found[title] = actual or page
            note = "verified" if actual == page else (f"CORRECTED -> heading is on p.{actual}" if actual else
                                                     "WARNING: heading not found, using bookmark")
            self.log.append(f"Region {title[:38]:<38} bookmark p.{page:<4} {note}")
        return found

    # ================================================================== parsing
    def parse(self) -> list[Element]:
        doc = self.open()
        tops = self.chapter_ranges(doc)
        self.verify_region_starts(doc)
        table_pages = set(self.cfg["table_pages"])
        skip = {_norm(s) for s in self.cfg["skip_sections"]}

        elements: list[Element] = []
        section, subsection, region = "", "", None
        current_top, front_title = None, None

        for i, page in enumerate(doc):
            pn = i + 1
            top = next((t for t in tops if t["start"] <= pn <= t["end"]), None)
            chapter = top["title"] if top else "Front matter"
            chapter_num = top["num"] if top else 0
            if top is not current_top:                       # a new chapter resets the hierarchy
                current_top, section, subsection, region = top, "", "", None
                front_title = None
            if front_title:                                  # e.g. "Foreword" inside the contents range
                chapter = front_title

            blocks = page_blocks(page)
            # running header: "CHAPTER 2 GLOBAL ECONOMIC PROSPECTS | JUNE 2026 74"
            printed = None
            body = []
            for b in blocks:
                if b["bbox"][1] < 55 and b["size"] <= 7.5:
                    m = re.search(r"(\d+)\s*$", b["text"]) or re.match(r"^(\d+)\s", b["text"])
                    printed = m.group(1) if m else None
                    continue
                if b["bbox"][1] > 735 and b["size"] <= 8.5 and len(b["text"]) <= 6:   # "xvii" footers
                    printed = b["text"]
                    continue
                body.append(b)
            body = order_blocks(body)

            # Box pages: a "BOX 1.1 ..." title at the top makes the whole page part of that box.
            box_id, box_title = None, None
            for b in body:
                m = re.match(r"BOX (\d+\.\d+)\s+(.*)", b["text"])
                if m and b["bbox"][1] < 130:
                    box_id = m.group(1)
                    box_title = re.sub(r"\s*\(continued\)\s*$", "", m.group(2)).strip()
                    break

            base = {"chapter": chapter, "chapter_num": chapter_num, "printed_page": printed}

            # ----- Statistical annex: whole page is a table
            if pn in table_pages:
                rows = [r for r in table_rows(page) if "STATISTICAL APPENDIX" not in r]   # drop running header
                title = next((r for r in rows if "Real GDP growth" in r), "Statistical Annex table")
                elements.append(Element("\n".join(rows), pn, "table",
                                        {**base, "section": "Statistical Annex", "subsection": title,
                                         "region": None, "box_id": None, "title": title}))
                continue

            consumed: set[int] = set()
            box_sub = ""
            for bi, b in enumerate(body):
                if bi in consumed:
                    continue
                t, size, font = b["text"], b["size"], b["font"]
                words = len(t.split())

                # ---------- headings
                if font.startswith("HelveticaLTStd-C") and size >= 20:          # level 0: Foreword, Executive Summary...
                    section, subsection, region = _norm(t), "", None
                    if top is None or top["title"] in ("Summary of Contents", "Contents"):
                        front_title = chapter = _norm(t)
                        base["chapter"] = chapter
                    continue
                if box_id and re.match(r"BOX \d+\.\d+", t):
                    continue
                is_h1 = _is_helvetica(font) and size >= 13.5 and words <= 14 and not re.match(r"(FIGURE|TABLE|BOX)", t)
                is_h2 = _is_helvetica(font) and 10.5 <= size < 13.5 and words <= 14 \
                    and not re.match(r"(FIGURE|TABLE|BOX)", t) and not t.endswith(".")
                if is_h1 and not box_id:
                    section, subsection = _norm(t), ""
                    region = _norm(t) if _norm(t) in self.regions else None
                    continue
                if is_h2:
                    if box_id:
                        box_sub = _norm(t)
                    else:
                        subsection = _norm(t)
                    continue

                if _norm(section) in skip:
                    continue

                if box_id:
                    meta = {**base, "section": f"Box {box_id}: {box_title}", "subsection": box_sub,
                            "region": None, "box_id": box_id}
                else:
                    meta = {**base, "section": section or chapter, "subsection": subsection,
                            "region": region, "box_id": None}
                meta["section_key"] = (chapter, meta["section"], meta["subsection"])

                # ---------- tables: rebuild rows from word positions
                if re.match(r"TABLE \d+\.\d+", t):
                    x0, y0, x1, _ = b["bbox"]
                    y_end = page.rect.height - 40
                    for bj, other in enumerate(body):
                        if bj != bi and other["bbox"][1] > y0 and other["text"].startswith("Source"):
                            y_end = min(y_end, other["bbox"][1])
                    rect = pymupdf.Rect(min(x0, 30), y0, page.rect.width - 30, y_end)
                    for bj, other in enumerate(body):
                        if pymupdf.Rect(other["bbox"]).intersects(rect) and bj != bi:
                            if pymupdf.Rect(other["bbox"]).y0 >= y0 - 1:
                                consumed.add(bj)
                    rows = table_rows(page, rect)
                    meta["title"] = t
                    elements.append(Element("\n".join(rows), pn, "table", meta))
                    continue

                # ---------- figures: keep title + descriptive caption, drop chart internals
                if re.match(r"FIGURE [B]?\d", t):
                    # the descriptive caption is the small Helvetica text right under the title
                    for bj, other in enumerate(body):
                        dy = other["bbox"][1] - b["bbox"][3]
                        if bj not in consumed and bj != bi and -2 <= dy <= 30 and 7.5 <= other["size"] <= 8.5 \
                                and _is_helvetica(other["font"]) and abs(other["bbox"][0] - b["bbox"][0]) < 40:
                            t += " - " + other["text"]
                            consumed.add(bj)
                            break
                    elements.append(Element(t, pn, "figure", {**meta, "title": t}))
                    continue

                if size < 7.5:          # axis ticks, legends, sources/notes of charts
                    continue
                if size >= 20:          # chapter-opener display text ("CHAPTER 3 / A RISING CHALLENGE")
                    continue
                elements.append(Element(t, pn, "paragraph", meta))
        return elements

    # ================================================================== chunking
    def build_chunks(self) -> list[Chunk]:
        elements = merge_continuations(self.parse())
        chunks: list[Chunk] = []
        max_words = self.cfg["chunk_words"]

        def path_of(e: Element) -> str:
            parts = [e.meta["chapter"]]
            if e.meta["section"] and e.meta["section"] != e.meta["chapter"]:
                parts.append(e.meta["section"])
            if e.meta.get("subsection"):
                parts.append(e.meta["subsection"])
            return " > ".join(parts)

        groups: list[list[Element]] = []
        for e in elements:
            key = (path_of(e), e.element_type)
            if groups and (path_of(groups[-1][0]), groups[-1][0].element_type) == key \
                    and e.element_type != "table":
                groups[-1].append(e)
            else:
                groups.append([e])

        for group in groups:
            first = group[0]
            spath = path_of(first)
            if first.element_type == "table":
                # one table per chunk; long tables are split by rows (title repeated)
                rows = first.text.split("\n")
                # rows up to the column-header line (e.g. "2023 | 2024 | 2025e ...") are repeated in every part
                head_n = next((k + 1 for k, r in enumerate(rows[:8]) if re.search(r"\b20\d\d[ef]?\s*\|", r)), 1)
                header_rows = rows[:head_n]
                packs, cur = [], []
                for r in rows:
                    if len(cur) > head_n and len(" ".join(cur).split()) + len(r.split()) > max_words * 2:
                        packs.append(cur)
                        cur = list(header_rows)
                    cur.append(r)
                packs.append(cur)
                bodies = ["\n".join(p) for p in packs]
                pack_list = [[first]] * len(bodies)
            else:
                pack_list = pack_elements(group, max_words, overlap=1)
                bodies = [None] * len(pack_list)

            for pack, body in zip(pack_list, bodies):
                pages = sorted({e.page for e in pack} | {e.meta["page_end"] for e in pack if e.meta.get("page_end")})
                region = first.meta.get("region")
                header = (f"[World Bank GEP June 2026 | {spath} | PDF p. {pages[0]}"
                          + (f"-{pages[-1]}" if len(pages) > 1 else "")
                          + (f" | region: {region}" if region else "") + "]")
                meta = {
                    "chapter": first.meta["chapter"],
                    "chapter_num": first.meta["chapter_num"],
                    "section": first.meta["section"],
                    "subsection": first.meta.get("subsection") or None,
                    "region": region,
                    "box_id": first.meta.get("box_id"),
                    "printed_page": first.meta.get("printed_page"),
                }
                chunks.append(make_chunk("gep", len(chunks), pack, header, spath, first.element_type, meta, body))
        return chunks

    # ================================================================== query understanding
    def analyze_query(self, question: str) -> QueryPlan:
        plan = QueryPlan(search_text=question)
        q = question.lower()

        # region (longest alias first so "north africa" wins over "africa")
        hits = set()
        for name, aliases in self.regions.items():
            for a in aliases:
                if len(a) <= 4:   # abbreviations like SAR / EAP must be written in capitals
                    if re.search(rf"\b{re.escape(a.upper())}\b", question):
                        hits.add(name)
                elif re.search(rf"\b{re.escape(a)}\b", q):
                    hits.add(name)
        if len(hits) == 1:
            plan.filters["region"] = hits.pop()
            plan.notes.append(f"region detected -> only search the '{plan.filters['region']}' section")
        elif len(hits) > 1:
            plan.notes.append(f"several regions mentioned ({', '.join(sorted(hits))}) -> no region filter")

        m = re.search(r"\b(?:chapter|ch\.?)\s*(\d+)\b", q)
        if m:
            plan.filters["chapter_num"] = int(m.group(1))
            plan.notes.append(f"chapter detected -> only search Chapter {m.group(1)}")
        if "executive summary" in q:
            plan.filters["chapter"] = "Executive Summary"
            plan.notes.append("'executive summary' -> only search the Executive Summary")
        if re.search(r"statistical (annex|appendix)", q):
            plan.filters["chapter"] = "Statistical Annex"
        m = re.search(r"\bbox\s*(\d+\.\d+)", q)
        if m:
            plan.filters["box_id"] = m.group(1)
        return self.expand_vocabulary(plan, question)

    # ================================================================== answer completeness (regional outlook)
    # Each regional "Outlook" section opens with the headline forecasts, e.g. (p. 96):
    #   "Growth in SAR is expected to soften to 6.3 percent in 2026 ..."
    #   "... growth in SAR is set to pick up to an average rate of 7 percent per year in 2027-28."
    # A small model sometimes drops one of them. If the answer misses a headline percentage, the
    # sentence is added, quoted from the retrieved section, with its page.

    REGION_ABBR = {"East Asia and Pacific": "EAP", "Europe and Central Asia": "ECA",
                   "Latin America and the Caribbean": "LAC",
                   "Middle East, North Africa, Afghanistan, and Pakistan": "MNA",
                   "South Asia": "SAR", "Sub-Saharan Africa": "SSA"}

    @staticmethod
    def _percents(text: str) -> set[str]:
        return set(re.findall(r"(\d+(?:\.\d+)?)\s*(?:percent|%)", text))

    def headline_forecasts(self, region: str, hits: list) -> list[tuple[str, str]]:
        """Sentences in the region's Outlook section that state the region-wide growth forecast."""
        abbr = self.REGION_ABBR.get(region)
        found = []
        chunks = sorted((h.chunk for h in hits if h.chunk.metadata.get("region") == region
                         and h.chunk.element_type == "paragraph"
                         and h.chunk.metadata.get("subsection") == "Outlook"),
                        key=lambda c: (c.page_start, c.chunk_id))
        for c in chunks:
            body = c.text.split("\n", 1)[-1].replace("\n", " ")
            for sentence in re.split(r"(?<=[.!?])\s+(?=[A-Z])", body):
                if (abbr and re.search(rf"\bgrowth in {abbr}\b", sentence, re.I)
                        and self._percents(sentence) and re.search(r"\b20\d\d", sentence)
                        and sentence not in [f[0] for f in found]):
                    found.append((sentence.strip(), c.pages))
        return found[:3]

    def postprocess_answer(self, question: str, answer: str, hits: list) -> tuple[str, list[str]]:
        regions = {h.chunk.metadata.get("region") for h in hits}
        if len(regions) != 1 or None in regions:
            return answer, []
        if not re.search(r"outlook|forecast|growth|prospect|project", question, re.I):
            return answer, []
        region = regions.pop()
        have = self._percents(answer)
        missing = [(s, pages) for s, pages in self.headline_forecasts(region, hits)
                   if not self._percents(s) <= have]
        if not missing:
            return answer, []
        extra = [f'- {s.rstrip(".")} ({page_label(pages)}).' for s, pages in missing]
        return answer.rstrip() + "\n" + "\n".join(extra), [
            f"Completeness check: added {len(extra)} headline {region} forecast(s) quoted from the "
            "Outlook section that the answer had left out."]
