from advanced_rag.profiles.imac import validate_numbers


def _by_type(chunks, t):
    return [c for c in chunks["imac"] if c.element_type == t]


def test_chart_only_states_labelled_values(chunks):
    chart = _by_type(chunks, "chart")
    assert len(chart) == 1
    text = chart[0].text
    for v in ("445 kg", "–99 kg", "346 kg"):
        assert v in text
    assert "NO numeric labels" in text            # tells the LLM not to estimate segments
    assert "Baseline emissions[^7]" in text and "[^7] Carbon reductions" in text


def test_tables_are_rebuilt_row_by_row(chunks):
    tables = " ".join(c.text for c in _by_type(chunks, "table"))
    assert "• Production | 50%" in tables
    assert "• Product use | 45%" in tables
    assert "Total product footprint | 346 kg CO2e" in tables
    assert "Sleep | 0.56W | 0.57W | 0.59W" in tables
    assert "part of endnote [^18]" in tables


def test_claims_carry_their_footnotes(chunks):
    energy = [c for c in chunks["imac"] if "58 percent less energy" in c.text]
    assert energy and all("[^18] Energy consumption and energy efficiency values" in c.text for c in energy)
    assert len(_by_type(chunks, "footnote")) == 19


def test_callouts_are_captured(chunks):
    callouts = [c.text for c in _by_type(chunks, "callout")]
    assert any("iMac uses 100% recycled aluminum in the stand" in t for t in callouts)
    assert any("76% recycled content in fiber packaging" in t for t in callouts)
    assert any("\nAluminum. We use 100 percent recycled aluminum in the stand" in t for t in callouts)


def test_number_validation():
    page_text = "445 kg Baseline emissions 346 kg"
    assert validate_numbers("Baseline 445 kg, final 346 kg", page_text) == []
    assert validate_numbers("Transportation is 30 kg", page_text) == ["30"]
