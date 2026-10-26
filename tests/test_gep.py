from advanced_rag.profiles import get_profile


def test_bookmark_for_south_asia_is_corrected():
    p = get_profile("gep")
    starts = p.verify_region_starts(p.open())
    assert starts["South Asia"] == 96          # bookmark says 95, heading is on 96
    assert starts["Sub-Saharan Africa"] == 99


def test_region_chunks_stay_inside_their_region(chunks):
    sar = [c for c in chunks["gep"] if c.metadata.get("region") == "South Asia"]
    assert sar and all(96 <= c.page_start <= 98 for c in sar)
    assert any(c.element_type == "table" and "TABLE 2.9" in c.text for c in sar)
    assert any("6.3 percent in 2026" in c.text for c in sar)
    # the Middle East table on p. 95 must NOT be labelled South Asia
    assert not any("TABLE 2.8" in c.text for c in sar)


def test_chapter_and_section_metadata(chunks):
    gep = chunks["gep"]
    exec_sum = [c for c in gep if c.metadata["chapter"] == "Executive Summary"]
    assert exec_sum and all(c.page_start in (19, 20) for c in exec_sum)
    ch3 = [c for c in gep if c.metadata["chapter_num"] == 3]
    assert ch3 and all(107 <= c.page_start <= 142 for c in ch3)
    assert any(c.metadata.get("box_id") == "3.1" for c in ch3)


def test_noise_is_not_indexed(chunks):
    for c in chunks["gep"]:
        assert c.metadata["section"] != "References"
        assert "Te global economy" not in c.text


def test_query_analysis_filters():
    p = get_profile("gep")
    assert p.analyze_query("What is the growth outlook for South Asia?").filters == {"region": "South Asia"}
    assert p.analyze_query("Growth in SAR?").filters == {"region": "South Asia"}
    assert p.analyze_query("What does Chapter 3 say about debt?").filters == {"chapter_num": 3}
    assert p.analyze_query("According to the executive summary, what are the risks?").filters == \
        {"chapter": "Executive Summary"}
    # comparing two regions -> no single-region filter
    assert "region" not in p.analyze_query("Compare South Asia and Sub-Saharan Africa").filters
