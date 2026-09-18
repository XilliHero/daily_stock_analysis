import json
from src.advisor.candidates import Candidate
from src.advisor.select import select_candidates_with_llm


def _cands():
    return [Candidate("INGR", strategy_count=4, sector="Consumer", verdict="Buy"),
            Candidate("TRV", strategy_count=4, sector="Financials"),
            Candidate("OKLO", strategy_count=1, sector="Energy", verdict="Hold")]


def test_llm_reorders_and_filters_to_real_tickers():
    # LLM prefers TRV then INGR, and hallucinates FAKE (must be dropped).
    fake = lambda system, user: json.dumps({"order": ["TRV", "FAKE", "INGR"], "rationale": "Quality first."})
    ordered, rationale = select_candidates_with_llm(_cands(), profile=None, equity_budget=1000.0,
                                                    max_position_pct=0.15, complete_fn=fake)
    assert [c.ticker for c in ordered] == ["TRV", "INGR"]  # FAKE dropped, OKLO omitted by LLM
    assert rationale == "Quality first."


def test_empty_or_invalid_selection_returns_none():
    # No valid tickers -> signal failure so the caller falls back to deterministic.
    fake = lambda system, user: json.dumps({"order": ["FAKE"], "rationale": "x"})
    assert select_candidates_with_llm(_cands(), None, 1000.0, 0.15, complete_fn=fake) is None


def test_bad_json_returns_none():
    fake = lambda system, user: "not json at all"
    assert select_candidates_with_llm(_cands(), None, 1000.0, 0.15, complete_fn=fake) is None


def test_llm_exception_returns_none():
    def boom(system, user):
        raise RuntimeError("quota")
    assert select_candidates_with_llm(_cands(), None, 1000.0, 0.15, complete_fn=boom) is None
