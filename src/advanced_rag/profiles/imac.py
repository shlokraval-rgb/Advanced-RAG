"""USE CASE 3 - Apple iMac Product Environmental Report: document-aware RAG for visual content.

The important facts in this 12-page report are NOT in normal paragraphs. They sit in:
  * a stacked bar CHART (p. 2)              -> rebuilt from the positions of its labels
  * small TABLES (p. 10 carbon footprint, p. 12 power consumption)  -> rebuilt row by row
  * CALL-OUTS / icons ("100% recycled aluminum in the stand", "76% recycled content")
  * FOOTNOTES ([^1]..[^19] on pp. 11-12) that qualify the claims

What this profile does:
  * Superscript footnote markers are kept as [^N] and every chunk that cites a footnote gets the
    footnote's text appended -> a claim and its condition are always retrieved together.
  * Each call-out (label + its text, or a big number + its description) becomes its own small chunk.
  * The chart is described ONLY with the values that are labelled in the PDF (445 kg, -99 kg, 346 kg).
    The coloured segments have no numbers, and the chunk says so explicitly - the LLM must not guess.
  * Every number in the chart/table descriptions is VALIDATED against the PDF text layer.
  * Optional: scripts/extract_imac_visuals.py asks a vision model to describe pages (e.g. logos);
    its output is cached in data/extracted/imac_vision.json and also validated before use.
"""
from __future__ import annotations

import json
import re

import pymupdf

from ..chunking import make_chunk, pack_elements
from ..config import path
from ..generator import NOT_FOUND, page_label, select_context
from ..models import Chunk, Element
from ..pdf_utils import clean_text, page_blocks, table_rows
from .base import BaseProfile

FOOTNOTE_RE = re.compile(r"\[\^(\d+)\]")
TABLE_VALUE_RE = re.compile(r"(\d[\d.,]*\s?(W|%|kg CO2e)|<1%|↓\s?>\s?\d+%)$")
LIFECYCLE = ("Design and Source", "Make", "Package and Ship", "Use", "Recover")


def numbers_in(text: str) -> set[str]:
    return {n.replace(",", "") for n in re.findall(r"\d[\d,]*(?:\.\d+)?", FOOTNOTE_RE.sub(" ", text))}


def validate_numbers(text: str, page_text: str) -> list[str]:
    """Return numbers in `text` that do NOT appear in the PDF text layer of the page (should be empty)."""
    return sorted(numbers_in(text) - numbers_in(page_text))


def _x_overlap(a, b) -> float:
    lo, hi = max(a[0], b[0]), min(a[2], b[2])
    return max(0.0, hi - lo) / max(1.0, min(a[2] - a[0], b[2] - b[0]))


class IMacProfile(BaseProfile):
    doc_id = "imac"

    def __init__(self):
        super().__init__()
        self.log: list[str] = []

    # ================================================================== chart (p. 2)
    def parse_chart(self, page: pymupdf.Page, top_y: float) -> Element | None:
        words = [w for w in page.get_text("words") if w[1] >= top_y - 2 and w[1] < 735]
        # value labels: "445" "kg", "–99" "kg", "346" "kg"
        values = []
        for a, b in zip(words, words[1:]):
            if re.fullmatch(r"[–-]?\d+", a[4]) and b[4] == "kg" and abs(a[1] - b[1]) < 3:
                values.append(((a[0] + b[2]) / 2, f"{a[4]} kg", a))
        if len(values) < 2:
            return None
        values.sort()
        centers = [v[0] for v in values]
        bars: list[list] = [[] for _ in values]
        other: list = []
        for w in words:
            xc = (w[0] + w[2]) / 2
            dist, idx = min((abs(xc - c), i) for i, c in enumerate(centers))
            (bars[idx] if dist <= 40 else other).append(w)

        def labels(ws) -> list[str]:
            """Group words into labels: same line -> join; small vertical gap -> same label."""
            ws = sorted(ws, key=lambda w: (round((w[1] + w[3]) / 2), w[0]))
            out, last_y, last_bottom = [], None, None
            for w in ws:
                # a footnote number glued to a word ("emissions7") -> "emissions[^7]"
                txt = re.sub(r"(?<=[a-z])(1[0-9]|[1-9])$", r"[^\1]", w[4])
                is_sup = re.fullmatch(r"\d{1,2}", txt) and (w[3] - w[1]) < 6
                if is_sup and out:
                    out[-1] += f"[^{txt}]"
                    continue
                same_line = last_y is not None and abs(w[1] - last_y) <= 2
                next_line = last_bottom is not None and 0 <= w[1] - last_bottom <= 3   # tight line spacing
                if out and (same_line or next_line):
                    out[-1] += " " + txt
                else:
                    out.append(txt)
                last_y, last_bottom = w[1], w[3]
            return [clean_text(o) for o in out]

        # labels below the bottom of the bars (e.g. "Renewable energy emissions") are annotations
        bottom = max((w[3] for w in words if w[4] == "Transportation"), default=10_000)
        other += [w for ws in bars for w in ws if w[1] > bottom + 5]
        bars = [[w for w in ws if w[1] <= bottom + 5] for ws in bars]

        lines = []
        for (_, value, _w), ws in zip(values, bars):
            lab = labels(ws)
            head = lab[0] if lab else value
            segs = lab[1:]
            lines.append(f"- Bar '{head}'. Stacked segments from top to bottom: {'; '.join(segs) or 'none'}.")
        extra = labels(other)
        total_labels = ", ".join(v[1] for v in values)
        nums = [int(v[1].split()[0].replace("–", "-")) for v in values]
        arithmetic = ""
        if len(nums) == 3 and nums[0] + nums[1] == nums[2]:
            arithmetic = f" The labelled totals are consistent: {nums[0]} kg + ({nums[1]} kg) = {nums[2]} kg."
        text = (
            "CHART: stacked bar chart of iMac carbon emissions (kg CO2e), from baseline emissions, through "
            "emissions reductions, to the final iMac emissions.\n" + "\n".join(lines) +
            (f"\nOther chart label: {'; '.join(extra)}." if extra else "") +
            f"\nIMPORTANT: only the bar totals are labelled with numbers ({total_labels}).{arithmetic} "
            "The individual coloured segments (materials and process, electricity for manufacturing, "
            "electricity for product use, transportation, etc.) have NO numeric labels in the chart, so "
            "their sizes are not stated by the chart. The percentage split by life-cycle phase is given in "
            "the Carbon Footprint table (page 10)."
        )
        bad = validate_numbers(text.replace("page 10", "").replace("CO2e", ""), page.get_text())
        self.log.append(f"Chart p.{page.number + 1}: labelled values {total_labels}; "
                        f"{'all numbers found in the PDF text layer' if not bad else 'UNVERIFIED numbers: ' + str(bad)}"
                        f"{'; arithmetic check OK' if arithmetic else ''}")
        return Element(text, page.number + 1, "chart", {"title": "Carbon emissions chart"})

    # ================================================================== tables (pp. 10, 12)
    def find_tables(self, page: pymupdf.Page, blocks: list[dict]) -> list[tuple[pymupdf.Rect, list[int]]]:
        """Table rows are tiny (<=7pt) blocks ending with a value (W, %, kg CO2e) + their bold headers."""
        idx = [i for i, b in enumerate(blocks)
               if b["size"] <= 7.0 and (TABLE_VALUE_RE.search(FOOTNOTE_RE.sub("", b["text"]))
                                        or ("bold" in b["font"].lower() and len(b["text"].split()) <= 8))]
        idx.sort(key=lambda i: blocks[i]["bbox"][1])
        groups: list[list[int]] = []
        for i in idx:
            if groups and blocks[i]["bbox"][1] - max(blocks[j]["bbox"][3] for j in groups[-1]) <= 14:
                groups[-1].append(i)
            else:
                groups.append([i])
        tables = []
        for g in groups:
            if sum(bool(TABLE_VALUE_RE.search(FOOTNOTE_RE.sub("", blocks[i]["text"]))) for i in g) < 1:
                continue
            rect = pymupdf.Rect(blocks[g[0]]["bbox"])
            for i in g:
                rect |= pymupdf.Rect(blocks[i]["bbox"])
            tables.append((rect, g))
        return tables

    def table_element(self, page: pymupdf.Page, rect: pymupdf.Rect, blocks: list[dict], used: list[int]) -> Element:
        rows = table_rows(page, rect + (-2, -2, 2, 2), y_tol=5)
        # put back footnote markers that the word-based rebuild glues to words ("achieved7" -> "achieved[^7]")
        refs = sorted({n for i in used for n in FOOTNOTE_RE.findall(blocks[i]["text"])}, key=int)
        for r in refs:
            rows = [re.sub(rf"(?<=[a-z]){r}(?=\s*(\||$))", f"[^{r}]", row) for row in rows]
        text = "TABLE (page %d):\n" % (page.number + 1) + "\n".join(rows)
        bad = validate_numbers(text.replace("page %d" % (page.number + 1), ""), page.get_text())
        self.log.append(f"Table p.{page.number + 1}: {len(rows)} rows; "
                        f"{'all numbers found in the PDF text layer' if not bad else 'UNVERIFIED: ' + str(bad)}")
        return Element(text, page.number + 1, "table", {"title": rows[0] if rows else "table"})

    # ================================================================== footnotes (pp. 11-12)
    @staticmethod
    def parse_footnotes(blocks: list[dict], pn: int, footnotes: dict, tables: list) -> None:
        current = None
        for b in sorted(blocks, key=lambda b: b["bbox"][1]):
            t = b["text"]
            # a table that sits inside a footnote (power table inside endnote 18) belongs to that footnote
            m = re.match(r"^\[\^(\d+)\]\s*(.*)", t) or (re.match(r"^(\d{1,2})\s+(.*)", t) if b["bbox"][0] < 212 else None)
            if m and b["bbox"][0] < 214:
                current = int(m.group(1))
                footnotes[current] = {"text": m.group(2).strip(), "page": pn, "y": b["bbox"][1]}
            elif current is not None:
                footnotes[current]["text"] += " " + t
        for fn in footnotes.values():
            fn["text"] = clean_text(fn["text"])

    # ================================================================== everything else: call-outs + text
    @staticmethod
    def is_label(b: dict, pn: int) -> bool:
        words = len(b["text"].split())
        bold = any(k in b["font"] for k in ("Semibold", "Bold", "GreenMarker"))
        stat = b["size"] >= 15 and re.fullmatch(r"[<>~]?\d[\d.,]*%?", b["text"])
        return bool(stat) or (bold and 7.0 <= b["size"] < 20 and words <= 9 and not b["text"].endswith("."))

    def group_callouts(self, blocks: list[dict], pn: int) -> list[tuple[str, str]]:
        """Return [(element_type, text)] where call-outs = label + the text under it."""
        blocks = sorted(blocks, key=lambda b: (b["bbox"][1], b["bbox"][0]))
        used, out = set(), []
        for i, b in enumerate(blocks):
            if i in used or not self.is_label(b, pn):
                continue
            used.add(i)
            label, body, last = b["text"], [], b["bbox"]
            stat = b["size"] >= 15
            changed = True
            while changed:
                changed = False
                for j, o in enumerate(blocks):
                    if j in used:
                        continue
                    gap = o["bbox"][1] - last[3]
                    if -3 <= gap <= 25 and _x_overlap(o["bbox"], last) >= 0.3:
                        if self.is_label(o, pn) and not body:     # heading split over 2 lines
                            label += " " + o["text"]
                        elif self.is_label(o, pn):
                            continue
                        else:
                            body.append(o["text"])
                        used.add(j)
                        last = o["bbox"]
                        changed = True
                        break
            sep = " " if stat else ": "
            out.append(("callout", label + (sep + " ".join(body) if body else "")))
        for i, b in enumerate(blocks):
            if i in used:
                continue
            t = b["text"]
            # bold lead-in + text next to an icon/photo, e.g. "Aluminum. We use 100 percent recycled ..."
            if re.match(r"^[A-Z][\w ]{2,25}\.\s", t) and len(t.split()) <= 60:
                out.append(("callout", t))
            elif re.fullmatch(r"(\[\^\d+\])+", t):                 # a lone footnote marker next to a logo
                out.append(("callout", f"Logo/badge on this page{t}"))
            else:
                out.append(("paragraph", t))
        return out

    # ================================================================== parse everything
    def parse(self) -> tuple[list[Element], dict]:
        doc = self.open()
        elements: list[Element] = []
        footnotes: dict[int, dict] = {}
        for page in doc:
            pn = page.number + 1
            blocks = []
            for b in page_blocks(page, mark_footnotes=True):
                t = b["text"]
                if t.startswith("iMac | Product Environmental Report") or t.startswith("©"):
                    continue
                if "GreenMarker" in b["font"] and pn != 3:           # small navigation wheel on pp. 4-8
                    continue
                if re.fullmatch(r"\d{1,2}", t) and b["size"] < 6:    # lone footnote marker next to a logo
                    b["text"] = f"[^{t}]"
                blocks.append(b)
            title_blocks = [b for b in blocks if b["size"] >= 20]
            title = clean_text(" ".join(b["text"] for b in title_blocks)) or "Product Environmental Report"
            blocks = [b for b in blocks if b["size"] < 20]
            meta = {"page_title": title, "lifecycle_stage": title if title in LIFECYCLE else None}

            # tables
            tables = self.find_tables(page, blocks)
            table_idx = {i for _, g in tables for i in g}
            for rect, g in tables:
                el = self.table_element(page, rect, blocks, g)
                note = next((b["text"] for b in blocks if b["text"].startswith("Note:")
                             and 0 <= b["bbox"][1] - rect.y1 <= 25), None)
                if note:
                    el.text += "\n" + note
                el.meta.update(meta)
                el.meta["y"] = rect.y0
                elements.append(el)
            blocks = [b for i, b in enumerate(blocks) if i not in table_idx and not b["text"].startswith("Note:")]

            # footnotes
            if title == "Endnotes":
                self.parse_footnotes(blocks, pn, footnotes, tables)
                continue

            # chart: everything below the first "<number> kg" label
            chart_top = next((b["bbox"][1] for b in blocks if re.fullmatch(r"[–-]?\d+ kg", b["text"])), None)
            if chart_top is not None:
                chart = self.parse_chart(page, chart_top)
                if chart:
                    chart.meta.update(meta)
                    elements.append(chart)
                    blocks = [b for b in blocks if b["bbox"][1] < chart_top - 2]

            for etype, text in self.group_callouts(blocks, pn):
                elements.append(Element(text, pn, etype, dict(meta)))

        # tables that sit inside an endnote (p. 12 power table is part of endnote 18)
        for el in elements:
            if el.element_type == "table" and el.meta.get("page_title") == "Endnotes":
                owner = max((n for n, f in footnotes.items() if f["page"] == el.page and f["y"] < el.meta["y"]),
                            default=None)
                if owner:
                    el.text = f"(This table is part of endnote [^{owner}])\n" + el.text
        self.log.append(f"Footnotes parsed: {len(footnotes)} ({', '.join(map(str, sorted(footnotes)))})")
        return elements, footnotes

    def vision_elements(self) -> list[Element]:
        """Optional extra descriptions from the vision model (only validated statements are used)."""
        p = path("extracted_dir") / "imac_vision.json"
        if not (self.cfg.get("use_vision_cache") and p.exists()):
            return []
        data = json.loads(p.read_text(encoding="utf-8"))
        out = []
        for item in data.get("items", []):
            if item.get("validated"):
                out.append(Element(f"VISUAL DESCRIPTION (vision model, validated): {item['text']}", item["page"],
                                   "visual", {"page_title": item.get("page_title", ""), "lifecycle_stage": None}))
        self.log.append(f"Vision cache: {len(out)} validated descriptions added")
        return out

    # ================================================================== chunking
    def link_definitions(self, elements: list[Element]) -> None:
        """Table rows such as "Production | 50%" are defined on the Definitions page (p. 9), e.g.
        "Production: Includes ... the manufacture, transport, and assembly of all parts ...".
        Attach those definitions to the table, so a question about "manufacturing" finds the
        "Production" share and the model knows the two are the same life-cycle phase."""
        defs = {}
        for el in elements:
            m = re.match(r"^•\s*([A-Z][\w -]{2,30}):\s*(.+)", el.text)
            if el.meta.get("page_title") == "Definitions" and m:
                first_sentence = re.split(r"(?<=\.)\s", m.group(2).strip())[0]   # keep the table chunk short
                defs[m.group(1).strip()] = (first_sentence, el.page)
        for el in elements:
            if el.element_type != "table":
                continue
            rows = re.findall(r"^•\s*([^|]+?)\s*\|", el.text, flags=re.M)
            linked = []
            for label, (text, page) in defs.items():
                if any(label.lower()[:5] in row.lower() for row in rows):
                    linked.append(f"- {label} (definition, p. {page}): {text}")
            if linked:
                el.text += "\nDefinitions of these life-cycle phases from the report:\n" + "\n".join(linked)
                self.log.append(f"Table p.{el.page}: linked {len(linked)} life-cycle definitions from the Definitions page")

    def build_chunks(self) -> list[Chunk]:
        elements, footnotes = self.parse()
        elements += self.vision_elements()
        self.link_definitions(elements)
        chunks: list[Chunk] = []

        def with_footnotes(text: str) -> tuple[str, list[int]]:
            refs = sorted({int(n) for n in FOOTNOTE_RE.findall(text)})
            refs = [r for r in refs if r in footnotes]
            if refs:
                text += "\nFootnotes referenced:\n" + "\n".join(f"[^{r}] {footnotes[r]['text']}" for r in refs)
            return text, refs

        def add(pack: list[Element], etype: str, extra_meta: dict | None = None, label: str = ""):
            first = pack[0]
            body, refs = with_footnotes("\n".join(e.text for e in pack))
            spath = first.meta.get("page_title", "") + (f" > {label}" if label else "")
            pages = sorted({e.page for e in pack})
            header = f"[Apple iMac Product Environmental Report | {spath} | p. {pages[0]} | {etype}]"
            meta = {"page_title": first.meta.get("page_title"), "lifecycle_stage": first.meta.get("lifecycle_stage"),
                    "footnote_refs": refs or None, **(extra_meta or {})}
            chunks.append(make_chunk("imac", len(chunks), pack, header, spath, etype, meta, body))

        # call-outs, charts, tables, visuals: one chunk each. Paragraphs: packed per page.
        paragraphs: list[Element] = []
        for el in elements:
            if el.element_type == "paragraph":
                if paragraphs and paragraphs[-1].page != el.page:
                    for pack in pack_elements(paragraphs, self.cfg["chunk_words"], overlap=0):
                        add(pack, "paragraph")
                    paragraphs = []
                paragraphs.append(el)
            else:
                label = el.meta.get("title") or el.text.split(":")[0][:60] if el.element_type == "callout" else \
                    el.meta.get("title", "")
                add([el], el.element_type, label=label if el.element_type != "table" else "")
        for pack in pack_elements(paragraphs, self.cfg["chunk_words"], overlap=0):
            add(pack, "paragraph")

        # every endnote is also searchable on its own
        for n, fn in sorted(footnotes.items()):
            el = Element(f"Endnote [^{n}]: {fn['text']}", fn["page"], "footnote", {"page_title": "Endnotes"})
            add([el], "footnote", {"footnote_id": n}, label=f"Endnote {n}")
        return chunks

    # ================================================================== source-grounded answer checks
    # Three mistakes seen with the small local model, each checked against the retrieved excerpts:
    #  1. part vs. whole: "more than 15% of the stand is recycled" - 15% is the WHOLE device's recycled
    #     content (p. 1); the stand-specific fact is "100 percent recycled aluminum in the stand" (p. 4).
    #  2. unlabelled chart values: "5% of the 445 kg baseline comes from transportation" - the chart labels
    #     only bar totals; 5% comes from the p. 10 table (share of the 346 kg life-cycle footprint).
    #  3. units: "the iMac weighs 391 kg CO2e" - kg CO2e is greenhouse-gas emissions, not weight.

    PARTS = ("stand", "packaging", "battery", "magnets", "solder", "printed circuit boards",
             "thermal module", "glass", "enclosure")
    CHART_SEGMENTS = ("transportation", "transport", "materials", "electricity", "manufacturing",
                      "product use", "shipping")
    _PCT = re.compile(r"(\d+(?:\.\d+)?)\s*(?:%|percent)")

    def _part_percents(self, sentence: str, part: str) -> set[str]:
        """Percentages in the same clause as the part: in "contains more than 15 percent recycled content,
        including 100 percent recycled aluminum in the stand" only 100 belongs to the stand."""
        found = set()
        for clause in re.split(r"[,;:]|\band\b|\bwhile\b", sentence):
            if re.search(rf"\b{part}\b", clause, re.I):
                found |= set(self._PCT.findall(clause))
        return found

    @staticmethod
    def _sentences(text: str) -> list[str]:
        return [s for s in re.split(r"(?<!\(p\.)(?<!pp\.)(?<! p\.)(?<=[.!?])\s+(?=[A-Z0-9])",
                                    text.replace("\n", " ")) if s.strip()]

    def _check_part_claims(self, question: str, answer: str, hits: list) -> tuple[str, list[str]]:
        q = question.lower()
        part = next((p for p in self.PARTS if re.search(rf"\b{p}\b", q)), None)
        if not part:
            return answer, []
        evidence = []                                         # (sentence, pages) that name the part
        for h in select_context(hits):
            own = h.chunk.text.split("\nFootnotes referenced:")[0].split("\n", 1)[-1]
            for sent in self._sentences(own):
                if self._part_percents(sent, part):
                    evidence.append((re.sub(r"\[\^\d+\]", "", sent).strip(), h.chunk.pages))
        if not evidence:
            return answer, []
        allowed = {m for s, _ in evidence for m in self._part_percents(s, part)}
        kept, removed = [], []
        for line in answer.split("\n"):
            parts = []
            for sent in self._sentences(line):
                pcts = self._part_percents(sent, part)
                if pcts and not pcts <= allowed:
                    removed.append(sent)                      # percentage that is not about this part
                else:
                    parts.append(sent)
            if parts and " ".join(parts).strip() not in ("-", "*", "•"):
                kept.append(" ".join(parts))
        answer_pcts = set().union(*(self._part_percents(x, part) for x in self._sentences("\n".join(kept))))
        notes = []
        if removed:
            notes.append(f"Part check: removed a percentage that the report does not give for the {part}: "
                         f"{' '.join(removed)!r}")
        if removed or not (answer_pcts & allowed):
            sent, pages = evidence[0]
            kept.append(f"- {sent.rstrip('.')} ({page_label(pages)}).")
            notes.append(f"Part check: added the {part}-specific fact quoted from the report.")
            return "\n".join(kept), notes
        return answer, []

    def _check_chart_segment(self, question: str, answer: str, hits: list) -> tuple[str, list[str]]:
        q = question.lower()
        segment = next((s for s in self.CHART_SEGMENTS if s in q), None)
        if not (segment and "chart" in q and re.search(r"how (much|many)|\bkg\b|value|amount", q)):
            return answer, []
        chart = next((h.chunk for h in hits if h.chunk.element_type == "chart"), None)
        if not chart or "NO numeric labels" not in chart.text:
            return answer, []
        if re.search(r"(not|n't) (label|labell?ed|stated|given|provided|specified|shown)|no numeric label"
                     r"|unlabell?ed", answer, re.I):
            return answer, []                                 # the model already refused correctly
        totals = re.search(r"only the bar totals are labelled with numbers \(([^)]*)\)", chart.text)
        fixed = (f"- The chart does not label the size of the {segment} segment - only the bar totals are "
                 f"labelled ({totals.group(1) if totals else 'see chart'}) ({page_label(chart.pages)}). "
                 "So the exact kg value for that segment is not given in the report.")
        table = next((h.chunk for h in hits if h.chunk.element_type == "table"
                      and re.search(rf"•\s*{segment}\w*\s*\|\s*[<\d]", h.chunk.text, re.I)), None)
        if table:
            row = re.search(rf"•\s*({segment}\w*)\s*\|\s*([^\n]+)", table.text, re.I)
            fixed += (f"\n- The Carbon Footprint table gives {row.group(1)} as {row.group(2).strip()} of the "
                      f"life-cycle footprint of the configuration in that table - a share, not a chart "
                      f"value ({page_label(table.pages)}).")
        return fixed, [f"Chart check: the answer gave a value for the unlabelled '{segment}' chart segment; "
                       "replaced it with what the report actually states."]

    def _check_weight_units(self, question: str, answer: str, hits: list) -> tuple[str, list[str]]:
        if not re.search(r"\bweigh|\bweight\b|how heavy|\bmass\b", question, re.I):
            return answer, []
        context = " ".join(h.chunk.text for h in select_context(hits))
        real_weight = re.search(r"\bweigh\w*\b[^.]{0,60}\d+(\.\d+)?\s*(kg|g|lb|pounds|kilograms)\b(?!\s*CO)",
                                context, re.I)
        if real_weight or not re.search(r"kg\s*CO2e", answer):
            return answer, []
        table = next((h.chunk for h in hits if "Total product footprint" in h.chunk.text), None)
        extra = ""
        if table:
            m = re.search(r"Total product footprint \| ([^\n]+)", table.text)
            extra = (f" The report gives the carbon footprint ({m.group(1).strip()}, {page_label(table.pages)}); "
                     "kg CO2e measures greenhouse-gas emissions, not the iMac's physical weight.")
        return NOT_FOUND + extra, ["Unit check: the answer reported kg CO2e (emissions) as the iMac's weight; "
                                   "the report does not state the weight."]

    def postprocess_answer(self, question: str, answer: str, hits: list) -> tuple[str, list[str]]:
        notes: list[str] = []
        for check in (self._check_weight_units, self._check_chart_segment, self._check_part_claims):
            answer, n = check(question, answer, hits)
            notes += n
        return answer, notes
