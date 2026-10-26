"""The two data structures used everywhere: Element (parsed piece of a PDF) and Chunk (indexed unit)."""
from __future__ import annotations

from dataclasses import asdict, dataclass, field


@dataclass
class Element:
    """One meaningful piece of a page after parsing (a paragraph, a table, a chart, a footnote...)."""
    text: str
    page: int                      # 1-based PDF page number
    element_type: str = "paragraph"  # paragraph | heading | table | chart | callout | caption | footnote | stat
    meta: dict = field(default_factory=dict)  # section info, footnote refs, etc.


@dataclass
class Chunk:
    """One unit stored in the vector database."""
    chunk_id: str
    doc_id: str
    text: str                      # what the LLM reads (includes a short context header)
    page_start: int
    page_end: int
    element_type: str
    section_path: str              # e.g. "Ch. 2 Regional Outlooks > South Asia > Outlook"
    metadata: dict = field(default_factory=dict)  # document-specific extras (region, chapter_num...)

    @property
    def pages(self) -> str:
        return str(self.page_start) if self.page_start == self.page_end else f"{self.page_start}-{self.page_end}"

    def chroma_metadata(self) -> dict:
        """ChromaDB only accepts str/int/float/bool values (no None, no lists)."""
        base = {
            "doc_id": self.doc_id,
            "page_start": self.page_start,
            "page_end": self.page_end,
            "pages": self.pages,
            "element_type": self.element_type,
            "section_path": self.section_path,
        }
        for k, v in self.metadata.items():
            if v is None:
                continue
            base[k] = ",".join(map(str, v)) if isinstance(v, (list, tuple)) else v
        return base

    def to_dict(self) -> dict:
        return asdict(self)

    @staticmethod
    def from_dict(d: dict) -> "Chunk":
        return Chunk(**d)
