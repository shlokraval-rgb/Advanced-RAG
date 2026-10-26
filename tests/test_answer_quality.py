"""Regression tests for the issues found in the manual local-model run:
1. GPO FDsys question: govinfo's full-scale deployment (FY 2018) vs FDsys retirement (December 2018)
2. World Bank answers: page citations must appear INSIDE the answer, not only in the source list
3. iMac carbon footprint: multi-value questions must get every value (product use AND production)
"""
from advanced_rag import generator, pipeline
from advanced_rag.profiles import get_profile


# ---------------------------------------------------------------- 1. GPO FDsys / govinfo

FDSYS_Q = "What new digital system replaced FDsys, and when was it fully deployed?"


def test_fdsys_question_uses_the_reports_vocabulary():
    plan = get_profile("gpo").analyze_query(FDSYS_Q)
    assert "full-scale deployment" in plan.search_text      # the report's wording on p. 4


def test_fdsys_context_contains_both_distinct_facts(ingested):
    res = pipeline.ask("gpo", FDSYS_Q)
    ctx = generator.build_context(res.hits)
    # the deployment fact (p. 4) AND the retirement date must both reach the LLM
    assert "full-scale deployment of govinfo" in ctx
    assert "over the course of FY 2018" in ctx
    assert "retiring FDsys in December 2018" in ctx
    assert "PDF p. 4" in ctx


def test_prompt_separates_dates_of_different_events():
    p = generator.SYSTEM_PROMPT
    assert "date of exactly the event" in p and "retirement" in p and "do not merge" in p


# ---------------------------------------------------------------- 2. inline page citations

def test_prompt_and_context_ask_for_inline_citations(ingested):
    assert "INSIDE the answer" in generator.SYSTEM_PROMPT
    assert "(p. 12)" in generator.SYSTEM_PROMPT                  # format example
    res = pipeline.ask("gep", "What is the growth outlook for South Asia in this report?")
    ctx = generator.build_context(res.hits)
    assert "[Excerpt 1 | PDF p. 9" in ctx                        # page shown in every excerpt label


def test_missing_citations_are_added_from_the_matching_excerpt(ingested):
    res = pipeline.ask("gep", "What is the growth outlook for South Asia in this report?")
    answer = ("Growth in South Asia is expected to soften to 6.3 percent in 2026.\n"
              "- Growth is set to pick up to an average rate of 7 percent per year in 2027-28.")
    fixed, n = generator.add_missing_citations(answer, res.hits)
    assert n == 2
    assert "6.3 percent in 2026 (p. 96)." in fixed
    assert "2027-28 (p. 96)." in fixed or "2027-28 (pp. 96-97)." in fixed


def test_existing_citations_and_unrelated_sentences_are_left_alone(ingested):
    res = pipeline.ask("gep", "What is the growth outlook for South Asia in this report?")
    answer = "Growth softens to 6.3 percent (p. 96). Bananas are yellow and tasty fruit."
    fixed, n = generator.add_missing_citations(answer, res.hits)
    assert n == 0 and fixed == answer
    assert generator.add_missing_citations(generator.NOT_FOUND, res.hits) == (generator.NOT_FOUND, 0)


def test_official_gep_questions_bring_gold_pages_into_context(ingested):
    cases = {
        "What is the growth outlook for South Asia in this report?": "PDF p. 96",
        "What does Chapter 3 say about sovereign debt in developing economies?": "PDF p. 109",
        "According to the executive summary, what are the main risks to the global outlook?": "PDF p. 19",
    }
    for q, label in cases.items():
        assert label in generator.build_context(pipeline.ask("gep", q).hits), q


# ---------------------------------------------------------------- 3. iMac multi-value completeness

CARBON_Q = ("What is the total carbon footprint of the iMac, and how much of that comes from "
            "product use versus manufacturing?")


def test_carbon_table_links_production_to_manufacturing(chunks):
    table = next(c for c in chunks["imac"] if "• Production | 50%" in c.text)
    assert "• Product use | 45%" in table.text and "346 kg CO2e" in table.text
    assert "Production (definition, p. 9)" in table.text and "the manufacture" in table.text


def test_carbon_question_context_has_every_requested_value(ingested):
    plan = get_profile("imac").analyze_query(CARBON_Q)
    assert "production" in plan.search_text
    ctx = generator.build_context(pipeline.ask("imac", CARBON_Q).hits)
    for value in ("346 kg CO2e", "• Product use | 45%", "• Production | 50%", "the manufacture"):
        assert value in ctx


def test_prompt_requires_every_part_of_multi_value_questions():
    assert "Answer EVERY part of the question" in generator.SYSTEM_PROMPT


# ---------------------------------------------------------------- 1b. govinfo deployment vs FDsys retirement
# Real qwen2.5:3b output that was WRONG:  "govinfo was fully deployed in December 2018 (p. 4)."
# The report (p. 4): full-scale deployment of govinfo during FY 2018; FDsys to be retired in December 2018.

import importlib.util  # noqa: E402
import re  # noqa: E402
import sys  # noqa: E402
from pathlib import Path  # noqa: E402

import yaml  # noqa: E402

WRONG_ANSWER = "- govinfo replaced FDsys (p. 4).\n- govinfo was fully deployed in December 2018 (p. 4)."
DEPLOYED_IN_DEC = re.compile(r"^(?!.*retir).*deploy.*December 2018", re.I)


def _evaluate_module():
    scripts = Path(__file__).resolve().parents[1] / "scripts"
    sys.path.insert(0, str(scripts))
    spec = importlib.util.spec_from_file_location("evaluate_script", scripts / "evaluate.py")
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def _fdsys_question():
    qs = yaml.safe_load(open(Path(__file__).resolve().parents[1] / "eval" / "questions.yaml", encoding="utf-8"))
    return next(q for q in qs if q["id"] == "gpo-2")


def test_deployment_passage_is_pinned_first(ingested):
    res = pipeline.ask("gpo", FDSYS_Q)
    assert "full-scale deployment of govinfo" in res.hits[0].chunk.text
    assert res.hits[0].chunk.page_start == 4
    assert any("pinned" in n for n in res.plan_notes)


def test_wrong_december_deployment_claim_is_corrected_from_sources(ingested):
    res = pipeline.ask("gpo", FDSYS_Q)
    fixed, notes = get_profile("gpo").postprocess_answer(FDSYS_Q, WRONG_ANSWER, res.hits)
    lines = fixed.split("\n")
    assert not any(DEPLOYED_IN_DEC.search(line) for line in lines)          # wrong claim gone
    assert "govinfo replaced FDsys (p. 4)." in fixed                           # correct part kept
    assert "full-scale deployment took place during FY 2018 (p. 4)" in fixed  # deployment period
    assert "December 2018 is the date for retiring FDsys" in fixed            # separate retirement date
    assert "retiring FDsys in December 2018" in fixed                         # quoted from the report
    assert notes and "removed" in notes[0]


def test_correct_answer_is_left_unchanged(ingested):
    res = pipeline.ask("gpo", FDSYS_Q)
    good = ("- govinfo replaced FDsys and its full-scale deployment took place during FY 2018 (p. 4).\n"
            "- FDsys was to be retired in December 2018 (p. 4).")
    assert get_profile("gpo").postprocess_answer(FDSYS_Q, good, res.hits) == (good, [])


def test_other_questions_are_not_touched(ingested):
    q = "How many depository libraries did GPO staff visit in FY 2018?"
    res = pipeline.ask("gpo", q)
    answer = "GPO staff visited 166 depository libraries (p. 5)."
    assert get_profile("gpo").postprocess_answer(q, answer, res.hits) == (answer, [])


def test_evaluation_rejects_december_deployment_and_accepts_correct_answer(ingested):
    ev, q = _evaluate_module(), _fdsys_question()
    ok, wrong = ev.check_answer(WRONG_ANSWER, q)
    assert not ok and wrong                                                   # the reported bug FAILS
    variant = "govinfo replaced FDsys; it was deployed fully in December 2018 (p. 4). FY 2018 (p. 4)."
    assert not ev.check_answer(variant, q)[0]                                 # reworded bug also FAILS
    res = pipeline.ask("gpo", FDSYS_Q)
    fixed, _ = get_profile("gpo").postprocess_answer(FDSYS_Q, WRONG_ANSWER, res.hits)
    assert ev.check_answer(fixed, q) == (True, [])                            # corrected answer PASSES


# ---------------------------------------------------------------- 2b. South Asia: duplicates + 2027-28 projection
# Real qwen2.5:3b output repeated the "excluding India" forecast and left out "7 percent ... in 2027-28".

SAR_Q = "What is the growth outlook for South Asia in this report?"
REPETITIVE_SAR_ANSWER = (
    "- Growth in South Asia is expected to soften to 6.3 percent in 2026 (p. 96).\n"
    "- Excluding India, growth in the region is forecast to increase to 4 percent in 2026 (p. 96).\n"
    "- Excluding India, growth in the region is forecast to increase to 4 percent in 2026 from 3.9 percent "
    "in 2025 (pp. 96-97).\n"
    "- Excluding India, growth in the region is forecast to increase to 4 percent in 2026 (p. 96).\n"
    "- Growth in India is projected to moderate to 6.6 percent in fiscal year 2026/27 (p. 96).")


def test_duplicate_statements_are_removed_keeping_the_most_detailed():
    fixed, n = generator.remove_duplicate_statements(REPETITIVE_SAR_ANSWER)
    assert n == 2
    assert fixed.count("Excluding India") == 1
    assert "from 3.9 percent in 2025" in fixed                      # the most informative copy is kept
    assert "6.3 percent in 2026" in fixed and "6.6 percent" in fixed  # different facts are kept


def test_distinct_statements_are_not_treated_as_duplicates():
    answer = ("- Growth in SAR is expected to soften to 6.3 percent in 2026 (p. 96).\n"
              "- Growth in SSA is forecast to edge down to 4 percent in 2026 (p. 99).\n"
              "- • Production | 50% and • Product use | 45% (p. 10).")
    assert generator.remove_duplicate_statements(answer) == (answer, 0)


def test_overlapping_chunks_do_not_repeat_paragraphs_in_context(ingested):
    ctx = generator.build_context(pipeline.ask("gep", SAR_Q).hits)
    assert ctx.count("Excluding India, growth in the region is forecast to increase to 4 percent") == 1
    assert "6.3 percent in 2026" in ctx and "7 percent per year in 2027" in ctx


def test_missing_2027_28_projection_is_added_from_the_outlook_section(ingested):
    hits = pipeline.ask("gep", SAR_Q).hits
    deduped, _ = generator.remove_duplicate_statements(REPETITIVE_SAR_ANSWER)
    fixed, notes = get_profile("gep").postprocess_answer(SAR_Q, deduped, hits)
    assert "6.3 percent in 2026" in fixed
    assert "average rate of 7 percent per year in 2027–28 (p. 96)." in fixed      # quoted, with page
    assert fixed.count("Excluding India") == 1
    assert notes and "Completeness check" in notes[0]
    # the strengthened evaluation check now passes for the repaired answer
    q = next(q for q in yaml.safe_load(open(Path(__file__).resolve().parents[1] / "eval" / "questions.yaml",
                                            encoding="utf-8")) if q["id"] == "gep-1")
    ev = _evaluate_module()
    assert not ev.check_answer(deduped, q)[0]          # without the 2027-28 projection -> FAILS
    assert ev.check_answer(fixed, q) == (True, [])     # repaired answer -> PASSES


def test_complete_sar_answer_and_other_questions_are_unchanged(ingested):
    hits = pipeline.ask("gep", SAR_Q).hits
    complete = ("- Growth in SAR is expected to soften to 6.3 percent in 2026 (p. 96).\n"
                "- Growth is set to pick up to an average rate of 7 percent per year in 2027-28 (p. 96).")
    assert get_profile("gep").postprocess_answer(SAR_Q, complete, hits) == (complete, [])
    ch3 = "What does Chapter 3 say about sovereign debt in developing economies?"
    other = "Rising debt raises interest rates (p. 109)."
    assert get_profile("gep").postprocess_answer(ch3, other, pipeline.ask("gep", ch3).hits) == (other, [])
