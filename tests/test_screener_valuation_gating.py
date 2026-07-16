# -*- coding: utf-8 -*-
"""Tests for the screener's valuation-gating optimization (no network).

Growth and recovery strategies define no valuation criteria, so the per-stock
`.info` fetch (the biggest yfinance rate-limit trigger) must be skipped for them.
"""

import pytest

from src.scanner import screener_agent as sa_mod
from src.scanner.screener_agent import ScreenerAgent, StockSignals
from src.scanner.strategy_profiles import get_strategy


@pytest.mark.parametrize(
    "strategy,expected",
    [("value", True), ("dividend", True), ("growth", False), ("recovery", False)],
)
def test_uses_valuation_by_strategy(strategy, expected):
    agent = ScreenerAgent(get_strategy(strategy))
    assert agent._uses_valuation() is expected


def test_valuation_skipped_strategy_does_not_touch_yfinance(monkeypatch):
    """For growth, _check_valuation_signals must not call yf.Ticker at all."""
    calls = {"n": 0}

    def _boom(*args, **kwargs):
        calls["n"] += 1
        raise AssertionError("yf.Ticker should not be called for growth")

    monkeypatch.setattr(sa_mod.yf, "Ticker", _boom)

    agent = ScreenerAgent(get_strategy("growth"))
    sig = StockSignals(ticker="AAA")
    agent._check_valuation_signals(sig)  # must be a no-op, no exception

    assert calls["n"] == 0
    assert sig.pe_ratio is None
    assert "low_pe" not in sig.signal_names


def test_valuation_strategy_still_fetches_info(monkeypatch):
    """For value, .info IS fetched and low_pe fires when P/E is under the cap."""
    class _FakeTicker:
        def __init__(self, ticker):
            self.info = {"trailingPE": 10.0, "priceToBook": 1.0, "marketCap": 1e9}

    monkeypatch.setattr(sa_mod.yf, "Ticker", _FakeTicker)

    agent = ScreenerAgent(get_strategy("value"))  # pe_max=20, pb_max=1.5
    sig = StockSignals(ticker="AAA")
    agent._check_valuation_signals(sig)

    assert sig.pe_ratio == 10.0
    assert "low_pe" in sig.signal_names
    assert "low_pb" in sig.signal_names
