# -*- coding: utf-8 -*-
"""Tests for the Watchlist Spotlight (offline, injected fetchers)."""

import numpy as np
import pandas as pd

from src.scanner.watchlist import (
    build_watchlist,
    get_watchlist_tickers,
    render_watchlist_markdown,
)


def _df(prices):
    n = len(prices)
    return pd.DataFrame(
        {"Open": prices, "High": [p * 1.01 for p in prices],
         "Low": [p * 0.99 for p in prices], "Close": prices, "Volume": [1000] * n}
    )


def _rising(n=60, start=100.0, step=1.0):
    return [start + i * step for i in range(n)]


def test_get_watchlist_tickers_parses_env(monkeypatch):
    monkeypatch.setenv("STOCK_LIST", "xle, btc-usd , CORN")
    assert get_watchlist_tickers() == ["XLE", "BTC-USD", "CORN"]


def test_none_when_empty(monkeypatch):
    monkeypatch.setenv("STOCK_LIST", "")
    assert build_watchlist() is None


def test_basic_metrics_and_missing():
    prices = _rising()
    report = build_watchlist(
        tickers=["AAA", "GHOST"],
        price_fetcher=lambda t: {"AAA": _df(prices)},
        scan_membership=lambda t: {},
    )
    assert report.missing == ["GHOST"]
    assert len(report.items) == 1
    aaa = report.items[0]
    assert aaa.price == prices[-1]
    assert aaa.ma20 is not None and aaa.ma50 is not None
    # steadily rising series → price above MA20, near 52w high
    assert aaa.price >= aaa.ma20
    assert any("52-week high" in a for a in aaa.alerts)


def test_rsi_oversold_alert_on_falling_series():
    falling = _rising(n=60, start=160.0, step=-1.0)  # 160 down to 101
    report = build_watchlist(
        tickers=["DOWN"],
        price_fetcher=lambda t: {"DOWN": _df(falling)},
        scan_membership=lambda t: {},
    )
    it = report.items[0]
    assert it.rsi is not None and it.rsi < 30
    assert any("oversold" in a for a in it.alerts)
    assert any("52-week low" in a for a in it.alerts)


def test_scan_membership_alert():
    report = build_watchlist(
        tickers=["AAA"],
        price_fetcher=lambda t: {"AAA": _df(_rising())},
        scan_membership=lambda t: {"AAA": ["value", "growth"]},
    )
    it = report.items[0]
    assert it.in_scan == ["value", "growth"]
    assert any("in today's value, growth scan" in a for a in it.alerts)


def test_big_move_alert():
    prices = _rising(n=60)
    prices[-1] = prices[-2] * 1.08  # +8% jump on the last day
    report = build_watchlist(
        tickers=["JUMP"],
        price_fetcher=lambda t: {"JUMP": _df(prices)},
        scan_membership=lambda t: {},
    )
    assert any("big move" in a for a in report.items[0].alerts)


def test_renderer_outputs_table_and_alerts():
    report = build_watchlist(
        tickers=["AAA"],
        price_fetcher=lambda t: {"AAA": _df(_rising())},
        scan_membership=lambda t: {"AAA": ["value"]},
    )
    md = render_watchlist_markdown(report)
    assert "## ⭐ Watchlist Spotlight" in md
    assert "AAA" in md
    assert "**Alerts:**" in md
