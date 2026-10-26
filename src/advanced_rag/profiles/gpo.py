"""USE CASE 1 - GPO 2018 Annual Report: baseline RAG over narrative text.

What is document-specific here (and why):
  * The PDF holds TWO documents: a magazine-style narrative report (pp. 1-30, two columns,
    photos + captions) and a formal financial report (pp. 31-82, single column, Sections I-IV).
  * Two-column pages are read column by column (pdf_utils.order_blocks) so sentences are not scrambled.
  * Running headers/footers are removed, but used to learn the printed page number / section name.
  * Photo captions become their own small chunks (they would otherwise interrupt paragraphs).
  * "GPO By the Numbers FY 2018" (pp. 14-15) is a label/value table -> kept as clean "label: value" lines.
  * Chunking is paragraph-aware: whole paragraphs packed to ~220 words, never crossing a section.
"""
from __future__ import annotations

import re

from ..chunking import make_chunk, merge_continuations, pack_elements
from ..generator import page_label
from ..models import Chunk, Element
from ..pdf_utils import order_blocks, page_blocks
from ..retriever import QueryPlan
from .base import BaseProfile

FIN_SECTIONS = {
    "I": "Section I: Management Discussion and Analysis (Unaudited)",
    "II": "Section II: Inspector General's Report",
    "III": "Section III: Independent Auditors' Report",
    "IV": "Section IV: Financial Statements",
}
_VALUE_AT_END = re.compile(r"^(.*?)\s+((?:Over\s+)?\$?[\d,.]+(?:\s+million)?)$")


def _is_spaced_caps(text: str) -> bool:
    """Letter-spaced display text such as "U . S . G O V E R N M E N T" (decoration, not content)."""
    tokens = text.split()
    return len(tokens) > 5 and sum(len(t) == 1 for t in tokens) / len(tokens) > 0.7


def _is_toc_page(blocks) -> bool:
    return sum("....." in b["text"] or "……" in b["text"] for b in blocks) >= 3


class GPOProfile(BaseProfile):
    doc_id = "gpo"

    # ------------------------------------------------------------------ parsing
    def parse(self) -> list[Element]:
        doc = self.open()
        n0, n1 = self.cfg["narrative_pages"]
        btn_pages = set(range(self.cfg["by_the_numbers_pages"][0], self.cfg["by_the_numbers_pages"][1] + 1))
        elements: list[Element] = []
        section, subsection = "Front matter", ""

        for i, page in enumerate(doc):
            pn = i + 1
            part = "Narrative report" if n0 <= pn <= n1 else "Financial report"
            blocks = page_blocks(page)
            if _is_toc_page(blocks):
                continue
            height = page.rect.height
            printed = None
            body = []
            for b in blocks:
                t, y0, size = b["text"], b["bbox"][1], b["size"]
                # --- running footers / headers (remove, but learn page number + section)
                if part == "Narrative report" and y0 > height * 0.93:
                    m = re.match(r"^(\d{1,3})\s+(?:2 0 1 8|U \. S)", t) \
                        or re.search(r"R E P O R T\s+(\d{1,3})$", t)     # "12 2 0 1 8 A N N U A L ..."
                    if m:
                        printed = int(m.group(1))
                    continue
                if part == "Financial report":
                    if y0 > 700 and re.fullmatch(r"\d{1,3}", t):
                        printed = int(t)
                        continue
                    if y0 > 700 and "Financial Year In Review" in t:
                        continue
                    if y0 < 55 and t.startswith("Section "):
                        section = t
                        continue
                    m = re.fullmatch(r"SECTION (I{1,3}|IV)", t)
                    if m:
                        section, subsection = FIN_SECTIONS[m.group(1)], ""
                        continue
                if size >= 20 or _is_spaced_caps(t):   # cover-page / decorative display text
                    continue
                body.append(b)

            # Reading order: two columns in the narrative part, top-to-bottom in the financial part.
            body = order_blocks(body) if part == "Narrative report" else sorted(body, key=lambda b: b["bbox"][1])

            if pn in btn_pages:
                section = "GPO By the Numbers FY 2018"
                rows = []
                for b in body:
                    if b["size"] >= 14:                # the page title itself
                        continue
                    m = _VALUE_AT_END.match(b["text"])
                    if m:
                        rows.append(f"{m.group(1)}: {m.group(2)}")
                elements.append(Element("\n".join(rows), pn, "table",
                                        {"part": part, "section": section, "subsection": "", "printed_page": printed}))
                continue

            small = [b for b in body if b["size"] < 6]      # e.g. organisational chart labels
            dropcap = ""
            for b in body:
                t, size, font = b["text"], b["size"], b["font"]
                if size < 6:
                    continue
                if len(t) == 1 and size > 30:              # decorative drop-cap letter ("T" + "he ...")
                    dropcap = t
                    continue
                if dropcap:
                    t, dropcap = dropcap + t, ""
                meta = {"part": part, "printed_page": printed}
                etype = "paragraph"
                if part == "Narrative report":
                    if "HelveticaNeue" in font and (9.9 <= size <= 10.1 or size >= 14) and len(t.split()) <= 10 \
                                and ("-M" in font or "-L" in font):
                        section, subsection = t, ""
                        continue
                    if "HelveticaNeue" in font and 8.1 <= size <= 8.3:
                        etype = "caption"
                else:
                    is_title = size >= 13.5 or "TimesNewRomanPS-Bold" in font or re.match(r"^([A-Z]|\d{1,2})\.\s", t)
                    if "Bold" in font and is_title and len(t.split()) <= 12 and not re.search(r"[\d:]$", t):
                        # "(Dollars in Thousands)" / the agency name repeated above each statement are not titles
                        if not t.startswith("(") and t != "U.S. Government Publishing Office":
                            subsection = t
                        continue
                meta.update(section=section, subsection=subsection)
                elements.append(Element(t, pn, etype, meta))
            if len(small) >= 10:
                txt = "; ".join(b["text"] for b in sorted(small, key=lambda b: (b["bbox"][1], b["bbox"][0])))
                elements.append(Element(f"Chart labels: {txt}", pn, "chart",
                                        {"part": part, "section": section, "subsection": subsection,
                                         "printed_page": printed}))
        return elements

    # ------------------------------------------------------------------ chunking
    def build_chunks(self) -> list[Chunk]:
        elements = merge_continuations(self.parse())
        chunks: list[Chunk] = []

        def path_of(e: Element) -> str:
            return " > ".join(p for p in (e.meta["part"], e.meta["section"], e.meta.get("subsection")) if p)

        # group consecutive elements of the same section and same kind (captions/charts/tables apart)
        groups: list[list[Element]] = []
        for e in elements:
            key = (path_of(e), e.element_type)
            if groups and (path_of(groups[-1][0]), groups[-1][0].element_type) == key:
                groups[-1].append(e)
            else:
                groups.append([e])

        for group in groups:
            first = group[0]
            spath = path_of(first)
            packs = pack_elements(group, self.cfg["chunk_words"], self.cfg["overlap_paragraphs"]) \
                if first.element_type == "paragraph" else [group]
            for pack in packs:
                printed = sorted({e.meta.get("printed_page") for e in pack if e.meta.get("printed_page")})
                pages = sorted({e.page for e in pack} | {e.meta['page_end'] for e in pack if e.meta.get('page_end')})
                header = f"[GPO 2018 Annual Report | {spath} | PDF p. {pages[0]}" + \
                         (f"-{pages[-1]}" if len(pages) > 1 else "") + "]"
                meta = {
                    "part": first.meta["part"],
                    "section": first.meta["section"],
                    "printed_pages": ",".join(map(str, printed)) or None,
                }
                chunks.append(make_chunk("gpo", len(chunks), pack, header, spath, first.element_type, meta))
        return chunks

    # ------------------------------------------------------------------ govinfo / FDsys date guard
    # The report (p. 4) says: "One of the most important developments at GPO during FY 2018 was the
    # full-scale deployment of govinfo ... in order to meet GPO's goal of retiring FDsys in December 2018."
    # Many other pages repeat "FDsys ... retired/replaced ... December 2018". A small local model tends to
    # attach December 2018 (FDsys's RETIREMENT date) to govinfo's DEPLOYMENT. Two narrow fixes:
    #   1. retrieval: the passage with "full-scale deployment of govinfo" is always retrieved first;
    #   2. answer check: a sentence that gives govinfo's deployment the retirement date is replaced by
    #      the two facts quoted from the retrieved source passages, each with its page.

    DEPLOYMENT_PHRASE = "full-scale deployment of govinfo"

    @staticmethod
    def _about_govinfo_deployment(question: str) -> bool:
        q = question.lower()
        return bool(re.search(r"fdsys|govinfo", q) and re.search(r"deploy|launch|roll|replac|successor", q))

    def analyze_query(self, question: str) -> QueryPlan:
        plan = super().analyze_query(question)
        if self._about_govinfo_deployment(question):
            plan.pin_phrases.append(self.DEPLOYMENT_PHRASE)
        return plan

    @staticmethod
    def _find_sentence(hits: list, test) -> tuple[str, str] | None:
        """First sentence in the retrieved chunks (document order) that passes `test` -> (sentence, pages)."""
        for h in sorted(hits, key=lambda h: (h.chunk.page_start, h.chunk.chunk_id)):
            body = h.chunk.text.split("\n", 1)[-1]
            for sentence in re.split(r"(?<=[.!?])\s+(?=[A-Z])", body.replace("\n", " ")):
                if test(sentence):
                    return sentence.strip(), h.chunk.pages
        return None

    def postprocess_answer(self, question: str, answer: str, hits: list) -> tuple[str, list[str]]:
        months = r"(?:January|February|March|April|May|June|July|August|September|October|November|December) \d{4}"
        deploy = self._find_sentence(hits, lambda s: self.DEPLOYMENT_PHRASE in s)
        retire = self._find_sentence(hits, lambda s: "FDsys" in s and re.search(r"retir", s) and re.search(months, s))
        if not (deploy and retire):
            return answer, []                    # only act when BOTH facts are in the retrieved sources
        period = re.search(r"\b(FY \d{4})", deploy[0])
        retire_date = re.search(months, retire[0]).group(0)
        if not period:
            return answer, []
        period = period.group(1)

        # 1) remove statements that give govinfo's deployment the FDsys retirement date
        deploy_words = r"deploy|launch|roll(?:ed)? out|went live|introduc"
        kept_lines, removed = [], []
        for line in answer.split("\n"):
            parts = re.split(r"(?<!\(p\.)(?<!pp\.)(?<! p\.)(?<=[.!?])\s+(?=[A-Z])", line)
            good = []
            for part in parts:
                wrong = (retire_date in part and re.search(deploy_words, part, re.I)
                         and not re.search(r"retir|decommission", part, re.I))
                (removed if wrong else good).append(part)
            new_line = " ".join(good).strip()
            if new_line and new_line not in ("-", "*", "•"):
                kept_lines.append(new_line)

        has_period = re.search(rf"{re.escape(period)}|fiscal year {period[3:]}", "\n".join(kept_lines), re.I)
        if not removed and (has_period or not self._about_govinfo_deployment(question)):
            return answer, []

        # 2) add the two facts, quoted from the retrieved source passages
        deploy_quote = deploy[0].split(", our public-facing")[0]
        facts = [
            f"- govinfo's full-scale deployment took place during {period} ({page_label(deploy[1])}). "
            f'Source: "{deploy_quote}."',
            f"- {retire_date} is the date for retiring FDsys, not govinfo's deployment date "
            f'({page_label(retire[1])}). Source: "{retire[0]}"',
        ]
        notes = []
        if removed:
            notes.append(f"Date check: removed a statement that gave govinfo's deployment the FDsys retirement "
                         f"date ({retire_date}): {' '.join(removed)!r}")
        notes.append(f"Date check: added the deployment period ({period}) and the separate FDsys retirement date "
                     f"({retire_date}), quoted from the retrieved report text.")
        return "\n".join(kept_lines + facts), notes
