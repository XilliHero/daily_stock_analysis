# -*- coding: utf-8 -*-
"""Tests for the What-Changed detector (offline, synthetic records)."""

from src.scanner.changes import detect_changes, render_changes_markdown
from src.scanner.scorecard.pick_store import PickRecord


def _rec(date, strat, ticker, score):
    return PickRecord(scan_date=date, strategy=strat, ticker=ticker, name=ticker, score=score)


def test_none_when_single_day():
    recs = [_rec("2026-06-01", "value", "AAA", 90)]
    assert detect_changes(recs) is None


def test_new_and_dropped_per_strategy():
    recs = [
        # value: prev day AAA, BBB  →  latest day AAA, CCC  (BBB dropped, CCC new)
        _rec("2026-06-01", "value", "AAA", 90),
        _rec("2026-06-01", "value", "BBB", 80),
        _rec("2026-06-02", "value", "AAA", 90),
        _rec("2026-06-02", "value", "CCC", 85),
    ]
    report = detect_changes(recs)
    assert report is not None
    val = next(s for s in report.per_strategy if s.strategy == "value")
    assert [e.ticker for e in val.new_entries] == ["CCC"]
    assert [e.ticker for e in val.dropped] == ["BBB"]


def test_each_strategy_uses_its_own_previous_day():
    # growth skips 06-02; on 06-03 it must diff against 06-01, not the empty 06-02.
    recs = [
        _rec("2026-06-01", "value", "AAA", 90),
        _rec("2026-06-02", "value", "AAA", 90),
        _rec("2026-06-03", "value", "AAA", 90),
        _rec("2026-06-01", "growth", "GGG", 70),
        _rec("2026-06-03", "growth", "HHH", 75),  # no growth on 06-02
    ]
    report = detect_changes(recs)
    growth = next(s for s in report.per_strategy if s.strategy == "growth")
    assert [e.ticker for e in growth.new_entries] == ["HHH"]
    assert [e.ticker for e in growth.dropped] == ["GGG"]


def test_degenerate_baseline_is_skipped():
    # 06-02 value is all-tied (degraded) → 06-03 should diff against 06-01.
    recs = [
        _rec("2026-06-01", "value", "AAA", 90),
        _rec("2026-06-01", "value", "BBB", 80),
        _rec("2026-06-02", "value", "XXX", 40),
        _rec("2026-06-02", "value", "YYY", 40),  # tied scores = degenerate
        _rec("2026-06-03", "value", "AAA", 90),
        _rec("2026-06-03", "value", "BBB", 80),
    ]
    report = detect_changes(recs)
    val = next(s for s in report.per_strategy if s.strategy == "value")
    # Diffs vs healthy 06-01 (AAA, BBB present both days) → no spurious changes.
    assert val.new_entries == []
    assert val.dropped == []


def test_rising_conviction_across_strategies():
    # LNC in 1 strategy on 06-01, in 2 on 06-02 → conviction rise.
    recs = [
        _rec("2026-06-01", "value", "LNC", 90),
        _rec("2026-06-01", "growth", "ZZZ", 70),
        _rec("2026-06-02", "value", "LNC", 90),
        _rec("2026-06-02", "growth", "LNC", 88),
    ]
    report = detect_changes(recs)
    assert len(report.conviction_up) == 1
    rise = report.conviction_up[0]
    assert rise.ticker == "LNC"
    assert rise.prev_count == 1
    assert rise.latest_count == 2
    assert rise.strategies == ["growth", "value"]


def test_renderer_outputs_sections():
    recs = [
        _rec("2026-06-01", "value", "AAA", 90),
        _rec("2026-06-01", "value", "BBB", 80),
        _rec("2026-06-02", "value", "AAA", 90),
        _rec("2026-06-02", "value", "CCC", 85),
    ]
    md = render_changes_markdown(detect_changes(recs))
    assert "## 🔄 What Changed Since 2026-06-01" in md
    assert "🆕 **New:**" in md
    assert "CCC" in md
    assert "❌ **Dropped:**" in md
    assert "BBB" in md
