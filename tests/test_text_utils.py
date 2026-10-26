from advanced_rag.chunking import merge_continuations, pack_elements
from advanced_rag.generator import unsupported_numbers
from advanced_rag.models import Element
from advanced_rag.pdf_utils import _join_lines, clean_text, order_blocks


def test_clean_text_fixes_broken_ligatures_and_co2e():
    assert clean_text("Te global economy") == "The global economy"
    assert clean_text("Tis is ﬁne") == "This is fine"
    assert clean_text("346 kg CO 2 e") == "346 kg CO2e"


def test_hyphenated_line_breaks():
    assert _join_lines(["fiscal sustainabil-", "ity matters"]) == "fiscal sustainability matters"
    assert _join_lines(["low-", "carbon electricity"]) == "low-carbon electricity"


def test_two_column_reading_order():
    def blk(text, x0, y0, x1):
        return {"text": text, "bbox": (x0, y0, x1, y0 + 10)}
    blocks = [blk("R1", 330, 100, 570), blk("L1", 40, 100, 290), blk("L2", 40, 200, 290),
              blk("TITLE", 40, 50, 570), blk("R2", 330, 200, 570)]
    assert [b["text"] for b in order_blocks(blocks)] == ["TITLE", "L1", "L2", "R1", "R2"]


def test_paragraph_continuation_is_merged():
    els = [Element("... experienced over 35", 4, "paragraph", {"section": "A"}),
           Element("A photo caption.", 4, "caption", {"section": "A"}),
           Element("million title retrievals each month.", 5, "paragraph", {"section": "A"})]
    merged = merge_continuations(els)
    assert merged[0].text.endswith("over 35 million title retrievals each month.")
    assert merged[0].meta["page_end"] == 5


def test_packing_never_emits_overlap_only_chunks():
    els = [Element("short intro.", 1), Element("word " * 300, 1), Element("word " * 300, 2)]
    packs = pack_elements(els, max_words=220, overlap=1)
    texts = ["|".join(e.text for e in p) for p in packs]
    assert len(texts) == len(set(texts))


def test_unsupported_numbers_guard():
    ctx = "GPO staff visited 166 depository libraries in 25 states."
    assert unsupported_numbers("They visited 166 libraries (p. 5).", ctx) == []
    assert unsupported_numbers("They visited 170 libraries.", ctx) == ["170"]
