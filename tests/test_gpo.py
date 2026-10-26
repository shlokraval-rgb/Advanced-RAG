def test_by_the_numbers_is_clean_label_value_table(chunks):
    btn = [c for c in chunks["gpo"] if "By the Numbers" in c.section_path]
    assert len(btn) == 1 and btn[0].element_type == "table"
    text = btn[0].text
    assert "Tangible copies distributed to depository libraries: 802,834" in text
    assert "Revenues from procured printing: $375.7 million" in text


def test_sample_answer_paragraphs_are_intact(chunks):
    texts = [c.text for c in chunks["gpo"]]
    assert any("GPO staff visited 166 depository libraries in 25 states" in t for t in texts)
    assert any("procured more than $375 million worth of printing" in t for t in texts)
    # paragraph split by a page break is re-joined
    assert any("over 35 million title retrievals each month" in t for t in texts)


def test_footers_removed_and_page_metadata_kept(chunks):
    for c in chunks["gpo"]:
        assert "A N N U A L" not in c.text
        assert "Financial Year In Review" not in c.text
    first = next(c for c in chunks["gpo"] if "166 depository libraries in 25 states" in c.text
                 and c.page_start == 5)
    assert first.metadata["part"] == "Narrative report"


def test_financial_report_sections_are_searchable(chunks):
    sections = {c.metadata["section"] for c in chunks["gpo"] if c.metadata["part"] == "Financial report"}
    assert any(s.startswith("Section I:") for s in sections)
    assert any(s.startswith("Section IV") for s in sections)
    assert any("Consolidated" in c.text and "1,018,063" in c.text for c in chunks["gpo"])
