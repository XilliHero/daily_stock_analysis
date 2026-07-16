# -*- coding: utf-8 -*-
"""The multi-strategy scan must fetch price history ONCE and share it, not
re-download the universe for every strategy (offline, monkeypatched)."""

import pandas as pd

from src.scanner import pipeline as pl
from src.scanner import screener_agent as sa
from src.scanner.screener_agent import ScreenerAgent
from src.scanner.strategy_profiles import get_strategy
from src.scanner.universe_agent import StockEntry


def _fake_df():
    # 15 rows of flat OHLCV — enough to pass the len>=10 gate.
    return pd.DataFrame(
        {
            "Open": [10.0] * 15,
            "High": [11.0] * 15,
            "Low": [9.0] * 15,
            "Close": [10.0] * 15,
            "Volume": [1000] * 15,
        }
    )


def test_screener_uses_supplied_cache_without_downloading(monkeypatch):
    calls = {"n": 0}
    monkeypatch.setattr(
        sa, "fetch_price_history",
        lambda tickers, *a, **k: (calls.__setitem__("n", calls["n"] + 1) or {}),
    )
    cache = {"AAA": _fake_df(), "BBB": _fake_df()}
    stocks = [StockEntry(ticker="AAA"), StockEntry(ticker="BBB")]

    ScreenerAgent(get_strategy("growth")).run(stocks, top_n=10, price_cache=cache)
    assert calls["n"] == 0  # cache supplied → no download


def test_screener_downloads_when_no_cache(monkeypatch):
    calls = {"n": 0}
    monkeypatch.setattr(
        sa, "fetch_price_history",
        lambda tickers, *a, **k: (calls.__setitem__("n", calls["n"] + 1) or {"AAA": _fake_df()}),
    )
    ScreenerAgent(get_strategy("growth")).run([StockEntry(ticker="AAA")], top_n=10)
    assert calls["n"] == 1  # no cache → fetch once


def test_multi_scan_prefetches_prices_once(monkeypatch):
    """4 strategies → exactly ONE price-history fetch, shared across all."""
    fetch_calls = {"n": 0}

    def _fake_fetch(tickers, *a, **k):
        fetch_calls["n"] += 1
        return {t: _fake_df() for t in tickers}

    # Stub the universe so we don't touch CSVs/network.
    class _FakeUniverseResult:
        stocks = [StockEntry(ticker="AAA"), StockEntry(ticker="BBB")]

    class _FakeUniverseAgent:
        def __init__(self, *a, **k): ...
        def run(self):
            return _FakeUniverseResult()

    monkeypatch.setattr(pl, "fetch_price_history", _fake_fetch)
    monkeypatch.setattr(pl, "UniverseAgent", _FakeUniverseAgent)

    pl.run_multi_strategy_scan(strategies=["value", "growth", "dividend", "recovery"])

    # Universe prefetch = 1 call total, NOT one per strategy.
    assert fetch_calls["n"] == 1
