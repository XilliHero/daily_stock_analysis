# -*- coding: utf-8 -*-
"""Tests for the scorecard evaluator (offline, injected price fetcher)."""

from datetime import date

from src.scanner.scorecard.evaluator import build_scorecard
from src.scanner.scorecard.pick_store import PickRecord
from src.scanner.scorecard.renderer import render_scorecard_markdown

AS_OF = date(2026, 7, 9)


def _records():
    return [
        # AAA first appears 06-01 @100, reappears 06-10 (strategy's latest date)
        PickRecord(scan_date="2026-06-01", strategy="value", ticker="AAA", name="Alpha", price=100.0),
        PickRecord(scan_date="2026-06-10", strategy="value", ticker="AAA", name="Alpha", price=110.0),
        # BBB only on 06-01 (not on the latest value date)
        PickRecord(scan_date="2026-06-01", strategy="value", ticker="BBB", name="Beta", price=50.0),
        # CCC in growth
        PickRecord(scan_date="2026-06-05", strategy="growth", ticker="CCC", name="Gamma", price=200.0),
    ]


def _stub_history(tickers, start):
    # Adjusted daily closes {ticker: {date: adj_close}}.
    return {
        "AAA": {"2026-06-01": 100.0, "2026-06-10": 110.0, "2026-07-09": 120.0},  # +20%
        "BBB": {"2026-06-01": 50.0, "2026-07-09": 40.0},                          # -20%
        "CCC": {"2026-06-05": 200.0, "2026-07-09": 260.0},                        # +30%
    }


def test_returns_computed_from_first_appearance():
    sc = build_scorecard(records=_records(), price_fetcher=_stub_history, as_of=AS_OF)
    val = next(s for s in sc.per_strategy if s.strategy == "value")
    aaa = next(p for p in val.picks if p.ticker == "AAA")
    # Baseline is the 06-01 price (first appearance), not 06-10.
    assert abs(aaa.return_pct - 0.20) < 1e-9
    assert aaa.days_tracked == (AS_OF - date(2026, 6, 1)).days


def test_per_strategy_aggregation():
    sc = build_scorecard(records=_records(), price_fetcher=_stub_history, as_of=AS_OF)
    val = next(s for s in sc.per_strategy if s.strategy == "value")
    assert val.count == 2
    assert abs(val.avg_return) < 1e-9      # +20% and -20% average to 0
    assert abs(val.hit_rate - 0.5) < 1e-9  # 1 of 2 green
    assert val.best.ticker == "AAA"
    assert val.worst.ticker == "BBB"


def test_overall_spans_all_strategies():
    sc = build_scorecard(records=_records(), price_fetcher=_stub_history, as_of=AS_OF)
    assert sc.overall.count == 3
    assert sc.n_tickers == 3
    # Overall avg = (0.20 - 0.20 + 0.30) / 3
    assert abs(sc.overall.avg_return - (0.30 / 3)) < 1e-9


def test_active_flag_tracks_latest_scan_date():
    sc = build_scorecard(records=_records(), price_fetcher=_stub_history, as_of=AS_OF)
    val = next(s for s in sc.per_strategy if s.strategy == "value")
    aaa = next(p for p in val.picks if p.ticker == "AAA")
    bbb = next(p for p in val.picks if p.ticker == "BBB")
    assert aaa.active is True    # present on latest value date (06-10)
    assert bbb.active is False   # last seen 06-01


def test_split_immunity_uses_adjusted_history():
    # Recorded price is a pre-split high; adjusted history is the source of truth.
    recs = [PickRecord(scan_date="2026-06-01", strategy="value", ticker="KLAC",
                       name="KLA", price=2000.0)]

    def fetch(tickers, start):
        return {"KLAC": {"2026-06-01": 200.0, "2026-07-09": 230.0}}  # 10:1 adjusted

    sc = build_scorecard(records=recs, price_fetcher=fetch, as_of=AS_OF)
    p = sc.per_strategy[0].picks[0]
    # +15% from the adjusted baseline, NOT -88% from the raw recorded price.
    assert abs(p.return_pct - 0.15) < 1e-9


def test_missing_price_is_excluded_and_counted():
    recs = [PickRecord(scan_date="2026-06-01", strategy="value", ticker="ZZZ",
                       name="Z", price=10.0)]
    sc = build_scorecard(records=recs, price_fetcher=lambda t, s: {}, as_of=AS_OF)
    assert sc.overall.count == 0
    assert sc.n_no_data == 1


def test_renderer_produces_scorecard_section():
    sc = build_scorecard(records=_records(), price_fetcher=_stub_history, as_of=AS_OF)
    md = render_scorecard_markdown(sc)
    assert "## 📊 Performance Scorecard" in md
    assert "Hit Rate" in md
    assert "Value" in md
    assert "AAA" in md  # best pick surfaced
