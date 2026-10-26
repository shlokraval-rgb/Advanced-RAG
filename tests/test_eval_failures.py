"""Regression tests for the 7 failures of the full local-model evaluation (eval_online_20261026-145516).
The answers below are the REAL qwen2.5:3b outputs from that run."""
import importlib.util
import re
import sys
from pathlib import Path

import yaml

from advanced_rag import generator, pipeline
from advanced_rag.profiles import get_profile

ROOT = Path(__file__).resolve().parents[1]
QUESTIONS = {q["id"]: q for q in yaml.safe_load(open(ROOT / "eval" / "questions.yaml", encoding="utf-8"))}


def _ev():
    sys.path.insert(0, str(ROOT / "scripts"))
    spec = importlib.util.spec_from_file_location("evaluate_script", ROOT / "scripts" / "evaluate.py")
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def _hits(qid):
    q = QUESTIONS[qid]
    return pipeline.ask(q["doc"], q["question"]).hits


def _refused(answer, q):
    ev = _ev()
    return any(p.lower() in answer.lower() for p in ev.REFUSAL_PHRASES + q.get("accept_refusals", []))


# ---------------------------------------------------------------- wrong page citations (gpo-3, imac-x1, imac-x3)

def test_gpo3_wrong_page_is_corrected(ingested):
    q = QUESTIONS["gpo-3"]
    real = "- GPO procured more than $375 million worth of printing and other information products in FY 2018 (p. 16)."
    fixed, changes = generator.repair_citations(real, _hits("gpo-3"))
    assert changes and "(p. 16)" not in fixed
    assert _ev().cited_pages(fixed) & set(q["gold_pages"])          # now cites p. 5 where the fact is


def test_imac_x1_table_value_cites_the_table_page(ingested):
    real = "- The iMac's power consumption in sleep mode at 115V is 0.57W (p. 7)."
    fixed, changes = generator.repair_citations(real, _hits("imac-x1"))
    assert "0.57W (p. 12)." in fixed and changes                      # 0.57W is in the p. 12 power table


def test_imac_x3_footnote_fact_cites_the_endnote_page(ingested):
    real = ("- According to the footnote (p. 12), the recycled content percentage excludes packaging "
            "or in-box accessories.")
    fixed, changes = generator.repair_citations(real, _hits("imac-x3"))
    assert "(p. 11)" in fixed and changes                              # endnote 1 is printed on p. 11
    assert _ev().cited_pages(fixed) & set(QUESTIONS["imac-x3"]["gold_pages"])


def test_correct_citations_are_not_touched(ingested):
    q = "How many depository libraries did GPO staff visit in FY 2018?"
    good = "- GPO staff visited 166 depository libraries in 25 states (p. 5)."
    assert generator.repair_citations(good, pipeline.ask("gpo", q).hits) == (good, [])


# ---------------------------------------------------------------- gep-3 answer criteria

def test_gep3_real_answer_meets_the_risk_criteria():
    real = ("- The main risks to the global outlook are a renewed escalation of hostilities or more prolonged "
            "disruptions to commodity flows, which could further raise commodity prices, intensify inflationary "
            "pressures, and food insecurity, triggering financial stress and lowering growth (p. 19).\n"
            "- Persistent trade policy uncertainty, geopolitical strains, and weather-related shocks also pose "
            "material risks to the global outlook (p. 19).")
    assert _ev().check_answer(real, QUESTIONS["gep-3"]) == (True, [])


def test_gep3_criteria_are_not_weaker():
    q = QUESTIONS["gep-3"]
    vague = "Risks to the outlook remain skewed to the downside (p. 19)."
    assert not _ev().check_answer(vague, q)[0]                         # no actual risks -> FAILS
    partial = "A renewed escalation of hostilities is the main risk (p. 19)."
    assert not _ev().check_answer(partial, q)[0]                       # only one risk -> FAILS
    assert len(q["must_contain"]) == 3


# ---------------------------------------------------------------- imac-3 stand: part-specific evidence wins

def test_imac3_whole_device_percentage_is_not_given_to_the_stand(ingested):
    q = QUESTIONS["imac-3"]
    real = "- The iMac stand is made from recycled aluminum (p. 4).\n- More than 15% of the iMac stand is recycled (p. 1)."
    fixed, notes = get_profile("imac").postprocess_answer(q["question"], real, _hits("imac-3"))
    assert "15%" not in fixed
    assert re.search(r"100(%| percent) recycled aluminum in the stand", fixed)
    assert notes and _ev().check_answer(fixed, q)[0]


def test_imac3_correct_stand_answer_is_unchanged(ingested):
    q = QUESTIONS["imac-3"]
    good = "- The iMac stand is made from 100% recycled aluminum (p. 4)."
    assert get_profile("imac").postprocess_answer(q["question"], good, _hits("imac-3")) == (good, [])


# ---------------------------------------------------------------- imac-neg-1 unlabelled chart segment

def test_imac_neg1_no_value_for_an_unlabelled_chart_segment(ingested):
    q = QUESTIONS["imac-neg-1"]
    real = "- In the carbon emissions chart, 5% of the 445 kg baseline comes from transportation (p. 2)."
    fixed, notes = get_profile("imac").postprocess_answer(q["question"], real, _hits("imac-neg-1"))
    assert notes and _refused(fixed, q)
    assert "of the 445 kg baseline comes from transportation" not in fixed
    # no kg value is attributed to transportation
    assert not re.search(r"\d+(\.\d+)?\s*kg[^.\n]{0,25}\b(from|of|for|is) transportation", fixed, re.I)
    assert not re.search(r"transportation[^.\n]{0,20}\b(is|=|:|accounts for)\s*\d+(\.\d+)?\s*kg", fixed, re.I)


def test_imac_neg1_correct_refusal_is_unchanged(ingested):
    q = QUESTIONS["imac-neg-1"]
    good = "The chart does not label the transportation segment; only the totals are labelled (p. 2)."
    assert get_profile("imac").postprocess_answer(q["question"], good, _hits("imac-neg-1")) == (good, [])


# ---------------------------------------------------------------- imac-neg-3 weight vs kg CO2e

def test_imac_neg3_emissions_are_not_reported_as_weight(ingested):
    q = QUESTIONS["imac-neg-3"]
    real = ("- The iMac weighs 391 kg CO2e (p. 10).\n"
            "- However, the weight of the iMac product itself is not directly provided in the given document. (p. 1)")
    fixed, notes = get_profile("imac").postprocess_answer(q["question"], real, _hits("imac-neg-3"))
    assert fixed.startswith(generator.NOT_FOUND) and _refused(fixed, q)
    assert "weighs 391" not in fixed and notes


def test_unit_check_only_applies_to_weight_questions(ingested):
    q = QUESTIONS["imac-2"]
    answer = "- The total product footprint is 346 kg CO2e (p. 10)."
    assert get_profile("imac").postprocess_answer(q["question"], answer, _hits("imac-2"))[0] == answer


def test_prompt_keeps_quantities_and_chart_values_apart():
    p = generator.SYSTEM_PROMPT
    assert "kg CO2e is emissions, not weight" in p and "never take a number from a table" in p
