# -*- coding: utf-8 -*-
"""Tests for the High-Conviction Overlap Board (offline, synthetic records)."""

from src.scanner.overlap import build_overlap_board, render_overlap_markdown
from src.scanner.scorecard.pick_store import PickRecord


def _rec(date, strat, ticker, score):
    return PickRecord(scan_date=date, strategy=strat, ticker=ticker, name=ticker, score=score)


def test_only_latest_date_considered():
    recs = [
        _rec("2026-06-01", "value", "OLD", 99),   # older date — ignored
        _rec("2026-06-02", "value", "AAA", 90),
        _rec("2026-06-02", "growth", "AAA", 85),
    ]
    board = build_overlap_board(recs)
    assert board.scan_date == "2026-06-02"
    assert [s.ticker for s in board.stocks] == ["AAA"]


def test_overlap_counts_and_ordering():
    recs = [
        # LNC in 3 strategies, AAA in 2, BBB in 1 (excluded)
        _rec("2026-06-02", "value", "LNC", 90),
        _rec("2026-06-02", "growth", "LNC", 80),
        _rec("2026-06-02", "dividend", "LNC", 88),
        _rec("2026-06-02", "value", "AAA", 95),
        _rec("2026-06-02", "recovery", "AAA", 70),
        _rec("2026-06-02", "value", "BBB", 60),
    ]
    board = build_overlap_board(recs)
    tickers = [s.ticker for s in board.stocks]
    assert tickers == ["LNC", "AAA"]         # 3-strategy first, then 2
    assert "BBB" not in tickers               # single strategy excluded
    lnc = board.stocks[0]
    assert lnc.strategy_count == 3
    assert lnc.strategies == ["dividend", "growth", "value"]


def test_empty_when_no_overlap():
    recs = [
        _rec("2026-06-02", "value", "AAA", 90),
        _rec("2026-06-02", "growth", "BBB", 80),
    ]
    board = build_overlap_board(recs)
    assert board.has_any is False
    assert "No multi-strategy overlaps" in render_overlap_markdown(board)


def test_renderer_marks_high_conviction():
    recs = [
        _rec("2026-06-02", s, "LNC", 90)
        for s in ["value", "growth", "dividend", "recovery"]
    ]
    board = build_overlap_board(recs)
    md = render_overlap_markdown(board)
    assert "## 🎯 High-Conviction Board" in md
    assert "LNC" in md
    assert "4/4" in md
    assert "🔥" in md  # 3+ strategies flagged
