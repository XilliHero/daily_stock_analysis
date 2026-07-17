# -*- coding: utf-8 -*-
"""Tests for the scanner→agents deep-dive handoff (offline, mocked analyzer)."""

from types import SimpleNamespace

from src.scanner.deepdive import (
    render_deepdive_markdown,
    run_deep_dives,
    select_deepdive_tickers,
)


def _board(stocks):
    return SimpleNamespace(
        stocks=[
            SimpleNamespace(ticker=t, name=n, strategy_count=c, strategies=s)
            for t, n, c, s in stocks
        ]
    )


def _analysis(price=100.0, advice="Buy on pullback", buy="98-100", stop="92", target="115"):
    return {
        "stock_code": "X",
        "stock_name": "Example Corp",
        "report": {
            "meta": {"current_price": price, "change_pct": 1.5},
            "summary": {
                "operation_advice": advice,
                "sentiment_label": "Bullish",
                "trend_prediction": "Uptrend intact",
                "analysis_summary": "Solid setup with volume confirmation.",
            },
            "strategy": {"ideal_buy": buy, "stop_loss": stop, "take_profit": target},
        },
    }


BOARD = _board([
    ("LNC", "Lincoln", 4, ["dividend", "growth", "recovery", "value"]),
    ("ABT", "Abbott", 3, ["dividend", "growth", "recovery"]),
    ("BNS", "Bank NS", 2, ["dividend", "growth"]),
])


def test_selection_respects_limit_and_order():
    picks = select_deepdive_tickers(BOARD, limit=2)
    assert [p.ticker for p in picks] == ["LNC", "ABT"]  # top-2 by conviction


def test_limit_zero_disables():
    assert select_deepdive_tickers(BOARD, limit=0) == []
    assert select_deepdive_tickers(None, limit=3) == []


def test_run_deep_dives_parses_response():
    picks = select_deepdive_tickers(BOARD, limit=1)
    results = run_deep_dives(picks, analyze_fn=lambda t: _analysis())
    assert len(results) == 1
    r = results[0]
    assert r.ticker == "LNC"
    assert r.current_price == 100.0
    assert r.verdict == "Buy on pullback"
    assert r.ideal_buy == "98-100" and r.stop_loss == "92" and r.take_profit == "115"
    assert r.strategy_count == 4


def test_none_analysis_is_skipped():
    picks = select_deepdive_tickers(BOARD, limit=2)
    # LNC returns None (e.g. quota/failure), ABT succeeds
    def fake(t):
        return None if t == "LNC" else _analysis()
    results = run_deep_dives(picks, analyze_fn=fake)
    assert [r.ticker for r in results] == ["ABT"]


def test_analysis_exception_is_skipped():
    picks = select_deepdive_tickers(BOARD, limit=1)
    def boom(t):
        raise RuntimeError("gemini down")
    assert run_deep_dives(picks, analyze_fn=boom) == []


def test_render_contains_verdict_and_levels():
    picks = select_deepdive_tickers(BOARD, limit=1)
    results = run_deep_dives(picks, analyze_fn=lambda t: _analysis())
    md = render_deepdive_markdown(results)
    assert "## 🔬 Deep Dive — Top Conviction Picks" in md
    assert "LNC" in md and "4/4 strategies" in md
    assert "Buy on pullback" in md
    assert "Buy 98-100" in md and "Stop 92" in md and "Target 115" in md


def test_render_none_when_empty():
    assert render_deepdive_markdown([]) is None
