"""Context building + grounded answer generation + a simple hallucination check."""
from __future__ import annotations

import re

from .config import load_config
from .retriever import Hit

NOT_FOUND = "I could not find this information in the document."

SYSTEM_PROMPT = """You answer questions about ONE document: {title}.

RULES
1. Use ONLY the context excerpts. Never use outside knowledge.
2. Cite pages INSIDE the answer: end every sentence or bullet with the PDF page of the excerpt it came
   from, written like (p. 96) or (pp. 96-98). Each excerpt label shows its page as "PDF p. ...".
   A list of sources at the end is NOT enough.
3. Answer EVERY part of the question. If it asks for several values or items (for example "X versus Y",
   or "what ... and when ..."), give each one as its own bullet with its own page.
4. Quote numbers exactly as written in the excerpts (keep units and words like "more than", "about").
5. "When" questions: give the date of exactly the event that was asked about. If the excerpts give
   different dates for related events (for example a new system's deployment and an old system's
   retirement), state each event with its own date - do not merge them.
6. If excerpts disagree, report both versions with their pages.
7. Charts: only state values that are explicitly labelled. Never estimate a value that is not labelled,
   and never take a number from a table or text and present it as a chart value.
8. Footnotes ([^N]) qualify claims. Mention a footnote's condition when it changes the meaning.
9. If the excerpts do not contain the answer, reply exactly: "{not_found}"
10. Keep quantities apart: a number belongs only to the item and unit it describes in the excerpt
   (a percentage for the whole product is not a percentage for one part; kg CO2e is emissions, not weight).
11. Be concise: 2-6 sentences or a short bullet list.

Required answer format (example):
- First fact that answers the question (p. 12).
- Second fact for the other part of the question (pp. 30-31)."""


def select_context(hits: list[Hit]) -> list[Hit]:
    """The hits that fit the word budget (best-ranked first), shown to the LLM in document order."""
    max_words = load_config()["retrieval"]["max_context_words"]
    kept, words = [], 0
    for h in hits:
        w = len(h.chunk.text.split())
        if kept and words + w > max_words:
            break
        kept.append(h)
        words += w
    return sorted(kept, key=lambda h: (h.chunk.page_start, h.chunk.chunk_id))


def page_label(pages: str) -> str:
    return f"pp. {pages}" if "-" in pages else f"p. {pages}"


def build_context(hits: list[Hit]) -> str:
    """Label each excerpt with its PDF page + section so the LLM can cite it.

    Neighbouring chunks share one overlap paragraph. A paragraph that was already shown in an earlier
    excerpt is NOT repeated - otherwise a small model tends to repeat that fact several times.
    """
    parts, seen = [], set()
    for h in select_context(hits):
        c = h.chunk
        header, *lines = c.text.split("\n")
        new_lines = []
        for line in lines:
            key = " ".join(line.split()).lower()
            if len(line.split()) >= 6 and key in seen:        # long duplicate paragraph -> skip
                continue
            seen.add(key)
            new_lines.append(line)
        if not any(len(line.split()) >= 6 for line in new_lines):   # nothing new in this excerpt
            continue
        n = len(parts) + 1
        parts.append(f"[Excerpt {n} | PDF {page_label(c.pages)} | {c.section_path} | {c.element_type}]\n"
                     + "\n".join([header] + new_lines))
    return "\n\n".join(parts)


def _statement_key(text: str) -> set[str]:
    text = re.sub(r"\(pp?\.[^)]*\)", " ", text.lower())               # ignore citations
    return set(re.findall(r"[a-z]+|\d+(?:\.\d+)?", text)) - _STOP


def remove_duplicate_statements(answer: str) -> tuple[str, int]:
    """Drop a sentence/bullet that repeats another one. If one statement is fully contained in a longer
    one (same words and numbers + extra detail), the longer, more informative one is kept.
    Returns (answer, number of statements removed)."""
    statements = []                                   # (line number, text)
    lines = answer.split("\n")
    for n, line in enumerate(lines):
        for piece in re.split(r"(?<!\(p\.)(?<!pp\.)(?<! p\.)(?<=[.!?])\s+(?=[A-Z0-9(\"“])", line):
            statements.append((n, piece))
    keys = [_statement_key(t) for _, t in statements]
    nums = [{w for w in k if w[0].isdigit()} for k in keys]
    dropped = set()
    for i, ki in enumerate(keys):
        if len(ki) < 4:
            continue
        for j in range(i):
            kj = keys[j]
            if j in dropped or len(kj) < 4:
                continue
            shared = len(ki & kj)
            if shared / len(ki) >= 0.85 and nums[i] <= nums[j]:      # i repeats j -> drop i
                dropped.add(i)
                break
            if shared / len(kj) >= 0.85 and nums[j] <= nums[i]:      # j is a shorter copy of i -> drop j
                dropped.add(j)
    out_lines = []
    for n, line in enumerate(lines):
        kept = [t for idx, (ln, t) in enumerate(statements) if ln == n and idx not in dropped]
        new_line = " ".join(kept).strip()
        if new_line and new_line not in ("-", "*", "•"):
            out_lines.append(new_line)
    return "\n".join(out_lines).strip(), len(dropped)


def generate_answer(question: str, hits: list[Hit], doc_title: str) -> str:
    """Ask the LOCAL model (Ollama, free) to answer from the retrieved excerpts only."""
    from . import ollama_client
    return ollama_client.chat([
        {"role": "system", "content": SYSTEM_PROMPT.format(title=doc_title, not_found=NOT_FOUND)},
        # the question is shown before AND after the excerpts - small models follow it better
        {"role": "user", "content": f"Question: {question}\n\nContext excerpts:\n\n{build_context(hits)}\n\n"
                                    f"Question: {question}\n"
                                    "Answer every part of the question and put a (p. N) citation after each fact."},
    ])


# ---------------------------------------------------------------- citation safety net

_CITE_RE = re.compile(r"\(pp?\.\s*\d")
_STOP = set("""a an and are as at be by for from has have in is it its of on or that the this to was were
which with will would than more about also into their there these those they not""".split())


def _content_words(text: str) -> set[str]:
    return {w for w in re.findall(r"[a-z]{3,}", text.lower()) if w not in _STOP}


def _support_text(hit: Hit) -> str:
    """The chunk's own text. Endnote texts appended to a claim ("Footnotes referenced:") are left out,
    because they are printed on the endnote page - a fact taken from them must cite that page."""
    return hit.chunk.text.split("\nFootnotes referenced:")[0]


def _fact_numbers(text: str) -> set[str]:
    """Numbers that carry the fact (years such as 2018 or 2026 are ignored - they appear everywhere)."""
    return {n for n in _numbers(text) if not re.fullmatch(r"(19|20)\d\d", n)}


def _best_source(statement: str, kept: list[Hit]) -> Hit | None:
    """The excerpt that supports a statement: all its fact numbers, else the largest word overlap (>= 50%)."""
    nums, words = _fact_numbers(statement), _content_words(statement)
    best, best_score = None, 0.0
    for h in kept:
        text = _support_text(h)
        if nums and not nums <= _numbers(text):
            continue
        overlap = len(words & _content_words(text)) / len(words) if words else 0.0
        if (nums or overlap >= 0.5) and overlap + len(nums) > best_score:
            best, best_score = h, overlap + len(nums)
    return best


_CITE_FULL_RE = re.compile(r"\((pp?)\.\s*([\d][\d\s,\-–and]*)\)")


def _cited(citation_body: str) -> set[int]:
    pages = set()
    for a, b in re.findall(r"(\d+)(?:\s*[-–]\s*(\d+))?", citation_body):
        pages.update(range(int(a), int(b or a) + 1))
    return pages


def repair_citations(answer: str, hits: list[Hit]) -> tuple[str, list[str]]:
    """Check every (p. N) citation against the excerpts the model actually saw.

    A citation is wrong if no excerpt on that page supports the statement (its numbers / words), e.g.
    "0.57W at 115V (p. 7)" when 0.57W is only in the p. 12 table. It is then replaced by the page of
    the excerpt that does support the statement. Unclear cases are left unchanged.
    Returns (answer, list of changes)."""
    if NOT_FOUND[:30] in answer:
        return answer, []
    kept = select_context(hits)
    changes = []
    out_lines = []
    for line in answer.split("\n"):
        pieces = re.split(r"(?<!\(p\.)(?<!pp\.)(?<! p\.)(?<=[.!?])\s+(?=[A-Z0-9(\"“])", line)
        new_pieces = []
        for piece in pieces:
            statement = _CITE_FULL_RE.sub(" ", piece)
            for m in list(_CITE_FULL_RE.finditer(piece)):
                cited = _cited(m.group(2))
                on_page = [h for h in kept if cited & set(range(h.chunk.page_start, h.chunk.page_end + 1))]
                best = _best_source(statement, kept)
                if best is None or best in on_page or (on_page and _best_source(statement, on_page)):
                    continue                     # citation is fine (or we cannot tell)
                new = f"({page_label(best.chunk.pages)})"
                piece = piece.replace(m.group(0), new, 1)
                changes.append(f"{m.group(0)} -> {new}")
            new_pieces.append(piece)
        out_lines.append(" ".join(new_pieces))
    return "\n".join(out_lines), changes


def add_missing_citations(answer: str, hits: list[Hit]) -> tuple[str, int]:
    """Small local models sometimes forget inline citations. For every sentence/bullet WITHOUT a
    (p. N) citation, find the excerpt it was taken from (shared numbers first, then shared words) and
    append that excerpt's page. Sentences that match no excerpt well are left alone.
    Returns (answer, number of citations added)."""
    if NOT_FOUND[:30] in answer:
        return answer, 0
    kept = select_context(hits)
    added = 0
    out_lines = []
    for line in answer.split("\n"):
        # split into sentences, but never inside a citation like "(p. 5)" or "(pp. 96-98)"
        pieces = re.split(r"(?<!\(p\.)(?<!pp\.)(?<! p\.)(?<=[.!?])\s+(?=[A-Z0-9(\"“])", line)
        new_pieces = []
        for piece in pieces:
            words = _content_words(piece)
            if _CITE_RE.search(piece) or len(piece.split()) < 4 or piece.rstrip().endswith(":") or not words:
                new_pieces.append(piece)
                continue
            nums = _numbers(piece)
            best, best_score = None, 0.0
            for h in kept:
                shared_nums = len(nums & _numbers(_support_text(h)))
                overlap = len(words & _content_words(_support_text(h))) / len(words)
                score = 2 * shared_nums + overlap
                if (shared_nums or overlap >= 0.5) and score > best_score:
                    best, best_score = h, score
            if best is None:
                new_pieces.append(piece)
                continue
            cite = f" ({page_label(best.chunk.pages)})"
            m = re.search(r"([.!?])\s*$", piece)
            new_pieces.append(piece[:m.start()] + cite + m.group(1) if m else piece.rstrip() + cite)
            added += 1
        out_lines.append(" ".join(new_pieces))
    return "\n".join(out_lines), added


# ---------------------------------------------------------------- hallucination check

_NUM_RE = re.compile(r"(?<![\w.])[-–]?\$?\d[\d,]*(?:\.\d+)?")


def _numbers(text: str) -> set[str]:
    # remove citations like "(p. 5)" / "(pp. 96-98)" and footnote markers before collecting numbers
    text = re.sub(r"\(pp?\.\s*[\d\s,\-–and]+\)", " ", text)
    text = re.sub(r"\[\^\d+\]|\[\d+\]", " ", text)
    out = set()
    for m in _NUM_RE.findall(text):
        n = m.replace("$", "").replace(",", "").replace("–", "-").lstrip("-")
        if n.endswith(".0"):
            n = n[:-2]
        out.add(n)
    return out


def unsupported_numbers(answer: str, context: str) -> list[str]:
    """Numbers that appear in the answer but nowhere in the retrieved context.

    A cheap, transparent guard: an LLM that invents a figure is caught here.
    """
    return sorted(_numbers(answer) - _numbers(context), key=lambda x: (len(x), x))
